# Qwen3.8 2B VoiceInk experiment

> The direct commands below document the one-time pre-canonical V1 run. Once
> that run is archived, all new Qwen3.8 jobs must use the sealed-release entry
> point described in [CANONICAL_PIPELINE.md](CANONICAL_PIPELINE.md):
>
> ```bash
> .venv/bin/python3 src/training/train.py sft \
>   --profile qwen38-2b-sft \
>   --release-manifest datasets/releases/voiceink-data-v1/manifest.json \
>   --version v2 --export-gguf q4_k_m q8_0
> ```

This experiment fine-tunes `empero-ai/Qwen3.8-2B` against the production
Qwen3.5 2B VoiceInk V3 baseline. The upstream model is a full-parameter
distillation into the Qwen3.5-2B architecture, so it uses the shared Qwen
Unsloth trainer and production message layout.

The base snapshot is local at `models/Qwen3.8-2B`, pinned by
`.voiceink-source.json`. Training uses the exact V3 train corpus (3,962 rows)
and the complete immutable evaluation holdout (440 rows). Nothing overwrites
the production V3 model.

## Preflight

```bash
.venv/bin/python3 src/training/finetune_qwen38_2b.py --check-only
```

## One-step smoke run

The smoke run includes one full 440-row loss evaluation and writes to isolated
directories:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python3 src/training/finetune_qwen38_2b.py \
    --max-steps 1 \
    --eval-steps 1 \
    --save-steps 1 \
    --lora-dir training/qwen38-2b-voiceink-v1/smoke-lora \
    --output-dir training/qwen38-2b-voiceink-v1/smoke-outputs \
    --mlflow-run-name qwen38-2b-voiceink-v1-smoke
```

## Full run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python3 src/training/finetune_qwen38_2b.py \
    --export-gguf q4_k_m q8_0
```

The full run performs 496 optimizer steps, evaluates and checkpoints every 62
steps including the final step, reloads the lowest-eval-loss checkpoint, and
exports the candidate to:

- `models/Qwen3.8-2B-VoiceInk-v1_gguf/Qwen3.8-2B.Q4_K_M.gguf`
- `models/Qwen3.8-2B-VoiceInk-v1_gguf/Qwen3.8-2B.Q8_0.gguf`

The matching llama-server presets are `Qwen3.8-2B-VoiceInk-v1` and
`Qwen3.8-2B-VoiceInk-v1-Q8`. Both disable thinking at inference, matching the
production VoiceInk contract.

## Locked-440 comparison against production V3

After export, restart `llama-router` so its model registry sees the new GGUF.
Generate both sides using the production message layout and Q4 presets:

```bash
.venv/bin/python3 src/eval/evaluate.py \
  --baseline Qwen3.5-2B-VoiceInk-v3 \
  --candidate Qwen3.8-2B-VoiceInk-v1 \
  --generate-only \
  --output-dir results/qwen38-2b-v1-vs-qwen35-2b-v3-strict-v2 \
  --generation-output results/qwen38-2b-v1-vs-qwen35-2b-v3-strict-v2/generations.jsonl \
  --mlflow-run-name qwen38-2b-v1-vs-qwen35-2b-v3-generation
```

Then unload the local models if desired and run the blinded Luna judge:

```bash
.venv/bin/python3 src/eval/evaluate.py \
  --baseline Qwen3.5-2B-VoiceInk-v3 \
  --candidate Qwen3.8-2B-VoiceInk-v1 \
  --outputs results/qwen38-2b-v1-vs-qwen35-2b-v3-strict-v2/generations.jsonl \
  --judge-provider codex \
  --judge-model gpt-5.6-luna \
  --judge-reasoning-effort low \
  --judge-rubric strict-v2 \
  --parallel 3 \
  --output-dir results/qwen38-2b-v1-vs-qwen35-2b-v3-strict-v2 \
  --mlflow-run-name qwen38-2b-v1-vs-qwen35-2b-v3-strict-v2
```

No `--eval-data` is supplied intentionally: the evaluator defaults to and
validates `datasets/regression/voiceink-locked-440.jsonl` by path, 440-row
count, and fingerprint before doing any work.
