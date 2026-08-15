# Qwen3.5 2B VoiceInk V3 DPO experiment

## Why DPO

Additional SFT lowered training loss while validation loss worsened, so another
pass over the same targets is not justified. DPO instead teaches a relative
preference: for a particular VoiceInk prompt, increase the probability of a
verified correction (`chosen`) relative to V3's current output (`rejected`).
The merged production V3 model remains the reference policy and a fresh LoRA
adapter is the only trainable state. There is no reward model, PPO loop, or
online model rollout during training.

## Do we need more labels?

Not for the first experiment. The strategic V3 training split already contains
897 approved corrections and has zero normalized-transcript overlap with the
locked 440 evaluation corpus. What is new is preference labeling:

1. Generate current production V3 outputs on those 897 prompts.
2. Compare each output with its existing approved correction using the blind
   strict-v2 Luna judge.
3. Keep only medium/high-confidence, material wins where the correction
   improves meaning or technical accuracy and does not regress meaning,
   technical accuracy, or instruction following.

This is deliberately stricter than ordinary SFT. Ties, style-only wins, failed
judgments, stale-model generations, prompt mismatches, and large rewrites are
excluded. If fewer than roughly 100 strong pairs survive, do not lower the
filter merely to make a run possible. Instead, mine new V3 production failures
and obtain new human or Luna-reviewed corrections for those cases.

The canonical 440 is never used to generate training pairs. It remains the
final promotion test.

## Pipeline

The commands below resume their private JSONL outputs where applicable and log
counts, fingerprints, prompt versions, metrics, and external artifact paths to
MLflow without uploading transcript text.

### 1. Generate current V3 rejected responses

Ensure `Qwen3.5-2B-VoiceInk-v3` is available from the llama-server on port 8002,
then run:

```bash
.venv/bin/python3 src/training/generate_qwen_v3_rejections.py
```

Preflight without inference:

```bash
.venv/bin/python3 src/training/generate_qwen_v3_rejections.py --check-only
```

Output:
`datasets/qwen35-2b-voiceink-dpo-v3/v3-rejections.jsonl`.

### 2. Construct unjudged candidate pairs

```bash
.venv/bin/python3 src/training/prepare_qwen_v3_preferences.py
```

This verifies the V3 model name, exact prompt fingerprint, approved-label
provenance, strategic split membership, engineering holdout exclusion, and zero
normalized-transcript overlap with all canonical 440 samples.

Output: `datasets/qwen35-2b-voiceink-dpo-v3/candidates.jsonl`.

### 3. Label preferences with Luna strict-v2

```bash
.venv/bin/python3 src/training/judge_qwen_v3_preferences.py \
  --judge-model gpt-5.6-luna \
  --judge-reasoning-effort low \
  --parallel 3
```

The output order is blinded. Existing successful judgments are reused on a
rerun; failed records are retried.

Output: `datasets/qwen35-2b-voiceink-dpo-v3/v3-judgments.jsonl`.

### 4. Finalize the strongly filtered preference split

```bash
.venv/bin/python3 src/training/prepare_qwen_v3_preferences.py \
  --judgments datasets/qwen35-2b-voiceink-dpo-v3/v3-judgments.jsonl
```

Inspect `datasets/qwen35-2b-voiceink-dpo-v3/dataset-report.json` before
training. The initial go/no-go threshold is at least 100 selected pairs with a
reasonable spread across representative, long, context-rich, and Qwen-risk
strata.

Outputs:

- `datasets/qwen35-2b-voiceink-dpo-v3/train.jsonl`
- `datasets/qwen35-2b-voiceink-dpo-v3/eval.jsonl`

### 5. DPO preflight and smoke test

The V3 wrapper enables 4-bit base loading, reference-log-probability
precomputation, and chunked DPO logits by default. Chunked mode preserves
autocast precision at the model boundary and uses Unsloth's selective
log-softmax in bounded chunks, avoiding a full FP32 copy of Qwen's large
vocabulary logits. This is required for the 16 GB training target.
The DPO completion cap is 2,048 tokens: only 4 of 343 training pairs exceed
that limit, while prompts and the model sequence window remain at 12,288 and
16,384 tokens respectively. This cap applies only to DPO examples and does not
change the exported model's runtime context or output limits.

```bash
.venv/bin/python3 src/training/finetune_qwen35_2b_v3_dpo.py --check-only
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python3 src/training/finetune_qwen35_2b_v3_dpo.py \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --lora-dir training/qwen35-2b-voiceink-v3-dpo-v1/smoke-lora \
  --output-dir training/qwen35-2b-voiceink-v3-dpo-v1/smoke-outputs
```

### 6. Full DPO run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python3 src/training/finetune_qwen35_2b_v3_dpo.py \
  --export-gguf q4_k_m q8_0
```

The conservative initial recipe is one epoch, effective batch size 8, LoRA
rank 16/alpha 32, learning rate `5e-6`, and DPO beta `0.1`. The wrapper aligns
evaluation/checkpoint cadence with the terminal optimizer step so the final
policy is always considered. It retains the 16K sequence window while using
4-bit frozen base weights and chunked vocabulary-logit processing during
training.

Artifacts are isolated under:

- `training/qwen35-2b-voiceink-v3-dpo-v1/`
- `models/Qwen3.5-2B-VoiceInk-v3-DPO-v1`
- `models/Qwen3.5-2B-VoiceInk-v3-DPO-v1_gguf`

## Promotion decision

After export, compare DPO V1 against production V3 on all 440 samples with the
strict-v2 Luna judge. Promotion requires no regression in meaning preservation
or technical accuracy, a positive material-win balance, and no meaningful
latency regression. Preference loss alone is not a production-quality metric.
