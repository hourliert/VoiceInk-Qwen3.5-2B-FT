# Qwen3.5-2B-VoiceInk

A fine-tuned [Qwen 3.5 2B](https://huggingface.co/Qwen) model for improving raw speech-to-text transcriptions from [VoiceInk](https://voiceink.app), a macOS dictation app. This repo contains the full pipeline: inference server, data collection, LLM-as-judge labeling, synthetic data generation, fine-tuning, and evaluation.

The fine-tuned model runs at ~250 tokens/second on a single RTX 4080 Super, outperforms the 4B base model on quality, and handles everything from short dictation snippets to 30-minute QA session transcripts.

**[Read the full blog post](docs/BLOG_POST.md)** for the complete story — from initial setup through five training iterations, a production bug caused by repetition amplification, and the synthetic data fix.

## How it works

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   VoiceInk   │ ──────> │   Reverse Proxy  │ ──────> │  llama-server    │
│   (macOS)    │  HTTP   │   (port 8001)    │  HTTP   │  (port 8002)     │
│              │ <────── │   JSONL logging   │ <────── │  Qwen 3.5 2B FT  │
└──────────────┘         └──────────────────┘         └──────────────────┘
                                  │
                                  ▼
                         logs/requests.jsonl
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
              Label with    Generate         Evaluate
              Claude judge  synthetic data   A/B with judge
                    │             │              │
                    ▼             ▼              ▼
              datasets/     datasets/        results/
              labeled.jsonl synthetic/       eval_*.jsonl
                    │             │
                    └──────┬──────┘
                           ▼
                    Prepare dataset
                           │
                           ▼
                    Fine-tune with
                    Unsloth LoRA
                           │
                           ▼
                    models/*.gguf
```

## Architecture

- **Host machine**: Remote gaming PC (RTX 4080 Super, 16GB VRAM) running Linux, accessible from the Mac over the network.
- **LLM backend**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) on port 8002, serving multiple Qwen 3.5 model variants via an OpenAI-compatible API.
- **Reverse proxy**: A lightweight Python proxy (`src/voiceink_proxy/server.py`) on port 8001 that forwards VoiceInk requests to llama-server and logs every request/response pair as JSONL for training data collection.
- **Startup**: `bin/start.sh` launches both processes. A systemd unit (`systemd/llama-router.service`) runs it on boot.

## The fine-tuning pipeline

The entire fine-tuning pipeline — from raw proxy logs to a deployed GGUF model — is built with Python (standard library only, no pip dependencies for the pipeline scripts) and [Claude](https://claude.ai) as an LLM judge via the [Claude CLI](https://github.com/anthropics/claude-code).

### 1. Data collection

VoiceInk sends OpenAI-compatible chat completion requests with structured XML context:

```
System message:
  <SYSTEM_INSTRUCTIONS>VoiceInk cleanup prompt</SYSTEM_INSTRUCTIONS>
  <CUSTOM_VOCABULARY>Alien, brake, chicane, Claude Code, GT Coach, ...</CUSTOM_VOCABULARY>
  <CURRENT_WINDOW_CONTEXT>Terminal - claude</CURRENT_WINDOW_CONTEXT>
  <CLIPBOARD_CONTEXT>...</CLIPBOARD_CONTEXT>

User message:
  <TRANSCRIPT>yeah so like i think voice ink is working pretty well actually</TRANSCRIPT>
```

The proxy logs these verbatim to `logs/voiceink_proxy_requests.jsonl`. A shared extraction module (`src/common/extract.py`) parses the XML tags into structured fields for downstream use.

### 2. Labeling (LLM-as-judge)

```bash
python3 src/labeling/label.py --parallel 5
python3 src/labeling/label.py --limit 50 --force  # relabel a subset
```

The labeling script sends each logged transcript to **Claude Sonnet 4.6** with a detailed judge prompt (`src/labeling/judge_prompt.txt`) that specifies exactly how to clean the transcript. Claude produces the gold-standard label — the ideal cleaned version of each transcript.

The judge prompt encodes rules for:
- Filler word removal ("so", "like", "basically", "um")
- STT error correction using phonetic context ("break" → "brake", "cloud code" → "Claude Code", "slab" → "lap")
- French-English transfer pattern fixes ("we are Monday" → "today is Monday")
- Word splitting recombination ("voice ink" → "VoiceInk")
- Preserving meaning, tone, and speaker opinions faithfully

Output: `datasets/labeled.jsonl` — each record contains the original request, the model's original response, and Claude's gold-standard label. Dedup-safe by `request_id`.

### 3. Synthetic data generation

```bash
python3 src/synthetic/generate.py --count 120 --parallel 5
```

The fine-tuned model initially failed on long QA debrief transcripts (500-3500 words) — it amplified repetitive coaching phrases and filled the entire 16K context window. The root cause: only 4 long samples existed in 1,175 training records.

The synthetic generator uses Claude Sonnet 4.6 to produce realistic QA debrief transcripts for [GT Coach](https://gtcoach.app) (a sim-racing coaching app). Each sample includes:
- Naturally repetitive corner-by-corner coaching phrases ("Corner 2, brake one beat earlier. It carried into corner 3. Your mid-corner speed is down.")
- Realistic STT errors at proper density
- Speaker narration interleaved with coaching feedback
- Both the messy raw transcript and its gold-standard cleaned version

A scenario matrix cycles through 12 tracks, 12 scenarios, and 8 word targets (500-3500) to ensure diversity. Generation is resumable — already-generated IDs are skipped on rerun.

Output: `datasets/synthetic/labeled.jsonl` — same schema as the real labels, with `syn-XXX` IDs and `original_model: "synthetic"`.

### 4. Dataset preparation

```bash
python3 src/training/prepare_dataset.py --extra-input datasets/synthetic/labeled.jsonl
```

Converts labeled records into Unsloth training format (OpenAI-style chat messages with typed content blocks for Qwen 3.5's VLM architecture). The system prompt is decoupled from what VoiceInk sent at recording time — training uses `docs/VOICEINK_PROMPT` so the prompt can evolve independently.

The `--extra-input` flag merges synthetic data into the training pipeline. Records are shuffled with a fixed seed and split 90/10 into train/eval.

### 5. Fine-tuning

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python3 src/training/finetune.py \
  --epochs 2 --export-gguf q4_k_m
```

Uses [Unsloth](https://github.com/unslothai/unsloth) for LoRA fine-tuning with automatic GGUF export. Key settings:

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank | 32 | r=32, alpha=64 |
| Learning rate | 2e-4 | Cosine scheduler |
| Batch size | 1 | Gradient accumulation: 8 (effective batch: 8) |
| Epochs | 2 | No improvement beyond 2 |
| Training mode | Completions-only | Loss computed only on assistant responses |
| Context | 16384 | Matches inference context |

The script auto-snapshots `datasets/labeled.jsonl` before training and auto-backs up existing GGUF files with version numbers (`.v1`, `.v2`, etc.).

**VRAM management**: Long synthetic samples (11K tokens) can exceed 16GB VRAM. Options:
- `--load-in-4bit` — loads base model in 4-bit quantization (most savings)
- `--load-in-8bit` — loads in 8-bit (moderate savings)
- `--offload-optimizer` — moves optimizer states to CPU RAM (no quality impact, slower)

### 6. Evaluation

```bash
python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk
```

Runs blind A/B evaluation using Claude Sonnet 4.6 as judge. For each eval sample:
1. Both models generate a cleaned transcript
2. Outputs are randomly assigned as "Response A" / "Response B"
3. Claude scores each on 6 weighted dimensions

**Scoring rubric** (from `src/eval/judge_prompt.txt`):

| Dimension | Weight | What it measures |
|---|---|---|
| Meaning preservation | 3x | Did the model keep the speaker's intent? |
| Instruction following | 3x | Did it clean (not answer/summarize) the transcript? |
| Filler removal | 2x | Were verbal tics removed? |
| Grammar & fluency | 2x | Is the output grammatically correct? |
| Technical accuracy | 2x | Are technical terms, names, numbers correct? |
| Conciseness | 1x | Is unnecessary verbosity removed? |

Supports `--resume` for interrupted evaluations and `--parallel` for concurrent judge calls.

## Results

### Quality (v5, latest)

| Eval | Fine-tuned 2B | Baseline | Gap | Win rate |
|---|---|---|---|---|
| vs Qwen 3.5 4B | **91.6** | 81.9 | +9.7 | 76% (98/129) |
| vs Qwen 3.5 2B base | **91.8** | 81.1 | +10.7 | 74% (87/117) |

The fine-tuned 2B model outperforms the 4B base model across all dimensions while running **2.3x faster** (~250 vs ~140 tokens/second).

### Long transcript handling

Before synthetic data training, the model amplified repetitive coaching phrases on long QA debriefs (30+ minute sessions), filling the entire 16K context window:

| Metric | Before (v4) | After (v5) |
|---|---|---|
| Finish reason | `length` (hit ceiling) | `stop` (natural) |
| Completion tokens | 10,294 | 2,484 |
| Time | 40.3s | 9.8s |

### Evolution

| Version | What changed | Score vs 4B |
|---|---|---|
| v3 | Better labels, LoRA rank 16→32, relabeled all 1,175 samples | 89.8 |
| v4 | Completions-only training, cosine scheduler, 2e-4 LR | 92.1 |
| v5 | + 120 synthetic QA debrief samples | 91.6 |

v5 trades a marginal 0.5 points on short dictation (within noise) for reliable long transcript handling.

## Key learnings

- **Model size doesn't help without fine-tuning**: Base 9B scores the same as base 4B (82.9 vs 82.3). Fine-tuned 2B beats both.
- **Completions-only training is a big win**: Training loss drops from ~0.85 to ~0.15 when masking system/user tokens.
- **LLM-as-judge labeling works**: Claude Sonnet 4.6 produces consistent, high-quality labels that translate directly into model improvement.
- **Synthetic data solves tail cases**: 120 synthetic long transcripts completely eliminated the repetition amplification bug that affected all model sizes.
- **Cosine scheduler + higher LR outperforms linear**: 2e-4 with cosine beats 1e-4 with linear.
- **2 epochs is sufficient**: No improvement beyond 2 epochs for this dataset size (~1,300 samples).

## Project structure

```
bin/start.sh                     # Launches llama-server + proxy
config/models.ini                # Model definitions and inference params
systemd/llama-router.service     # Systemd unit for boot startup
docs/
  PRODUCT_SPEC.md                # Product requirements (source of truth)
  VOICEINK_PROMPT                # System prompt VoiceInk sends
src/
  voiceink_proxy/server.py       # Reverse proxy with JSONL logging
  common/extract.py              # Structured XML extraction from requests
  labeling/
    label.py                     # Gold-standard label generation (Claude judge)
    judge_prompt.txt             # Labeling judge prompt
  synthetic/
    generate.py                  # Synthetic QA debrief generator
    generator_prompt.txt         # Generator prompt template
  training/
    prepare_dataset.py           # Convert labels to training format
    finetune.py                  # Unsloth LoRA fine-tuning + GGUF export
  eval/
    evaluate.py                  # A/B evaluation pipeline
    judge_prompt.txt             # Evaluation scoring rubric
datasets/                        # Training data (gitignored, *.jsonl)
models/                          # GGUF model files (gitignored)
logs/                            # Proxy request logs (gitignored)
results/                         # Evaluation results (gitignored)
```

## Reproducing the pipeline

### Prerequisites

- Linux machine with NVIDIA GPU (16GB+ VRAM recommended)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with CUDA support
- Python 3.12+ with a venv containing [Unsloth](https://github.com/unslothai/unsloth) and PyTorch
- [Claude CLI](https://github.com/anthropics/claude-code) installed and authenticated (for labeling, synthetic data, and evaluation)
- A Qwen 3.5 2B base model in GGUF format

### Steps

```bash
# 1. Collect data — run the proxy, use VoiceInk normally
bin/start.sh

# 2. Label your data with Claude as judge
python3 src/labeling/label.py --parallel 5

# 3. (Optional) Generate synthetic data for edge cases
python3 src/synthetic/generate.py --count 120 --parallel 5

# 4. Prepare the training dataset
python3 src/training/prepare_dataset.py \
  --extra-input datasets/synthetic/labeled.jsonl

# 5. Fine-tune
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python3 src/training/finetune.py \
  --epochs 2 --export-gguf q4_k_m

# 6. Evaluate against baseline
python3 src/eval/evaluate.py \
  --baseline Qwen3.5-4B \
  --candidate Qwen3.5-2B-VoiceInk

# 7. Deploy
sudo systemctl restart llama-router
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 4080 Super (16GB VRAM)
- **Inference**: ~250 tokens/second (Qwen 3.5 2B, Q4_K_M quantization)
- **Training**: ~5 min/epoch for 1,300 samples with LoRA
- **Context window**: 16,384 tokens (training and inference)

## Built with

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — LLM inference
- [Unsloth](https://github.com/unslothai/unsloth) — LoRA fine-tuning
- [Qwen 3.5](https://huggingface.co/Qwen) — Base model family
- [Claude](https://claude.ai) via [Claude CLI](https://github.com/anthropics/claude-code) — Labeling judge, synthetic data generation, evaluation judge
- [VoiceInk](https://voiceink.app) — macOS dictation app (the client)
- [GT Coach](https://gtcoach.app) — Sim-racing coaching app (source of QA debrief transcripts)
