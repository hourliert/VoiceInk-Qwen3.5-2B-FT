#!/usr/bin/env python3
"""DPO-tune merged production Qwen3.5 2B VoiceInk V3."""
import json
import math
import sys
from pathlib import Path

try:
    from .finetune import align_step_interval
    from .finetune_qwen_dpo import main as shared_main
except ImportError:
    from finetune import align_step_interval
    from finetune_qwen_dpo import main as shared_main

TRAIN_PATH = Path("datasets/qwen35-2b-voiceink-dpo-v3/train.jsonl")


def record_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip() and json.loads(line))


def profile_argv(argv=None) -> list[str]:
    overrides = sys.argv[1:] if argv is None else list(argv)
    count = record_count(TRAIN_PATH)
    # One optimizer update consumes eight preference pairs. Aligning the
    # cadence to a divisor guarantees that the terminal policy is evaluated.
    total_steps = max(1, math.ceil(count / 8))
    cadence = align_step_interval(total_steps, 25)
    defaults = [
        "--base-model", "models/Qwen3.5-2B-VoiceInk-v3",
        "--train", str(TRAIN_PATH),
        "--eval", "datasets/qwen35-2b-voiceink-dpo-v3/eval.jsonl",
        "--max-completion-length", "2048",
        "--r", "16",
        "--lora-alpha", "32",
        "--epochs", "1",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--lr", "5e-6",
        "--warmup-ratio", "0.10",
        "--beta", "0.10",
        "--load-in-4bit",
        "--chunked-logits",
        "--precompute-ref-log-probs",
        "--torch-empty-cache-steps", "1",
        "--eval-steps", str(cadence),
        "--save-steps", str(cadence),
        "--lora-dir", "training/qwen35-2b-voiceink-v3-dpo-v1/lora",
        "--output-dir", "training/qwen35-2b-voiceink-v3-dpo-v1/outputs",
        "--gguf-base", "models/Qwen3.5-2B-VoiceInk-v3-DPO-v1",
        "--mlflow-experiment", "voiceink-training",
        "--mlflow-run-name", "qwen35-2b-voiceink-v3-dpo-v1",
    ]
    return [*defaults, *overrides]


def main(argv=None) -> None:
    original = sys.argv
    try:
        sys.argv = [original[0], *profile_argv(argv)]
        shared_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
