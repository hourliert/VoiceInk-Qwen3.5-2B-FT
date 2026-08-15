#!/usr/bin/env python3
"""Continue production Qwen3.5 2B VoiceInk V3 for half an epoch.

V4 deliberately starts a fresh LoRA optimization run from the merged V3
weights.  It does not resume V3's exhausted optimizer/scheduler state.
"""
import sys

try:
    from .finetune import main as shared_main
    from .finetune import parse_args as shared_parse_args
except ImportError:  # Direct execution from the repository root.
    from finetune import main as shared_main
    from finetune import parse_args as shared_parse_args


PROFILE_ARGS = [
    "--base-model", "models/Qwen3.5-2B-VoiceInk-v3",
    "--train", "datasets/qwen35-2b-voiceink-v3/train.jsonl",
    "--eval", "datasets/qwen35-2b-voiceink-v3/eval-regression-340.jsonl",
    # 496 optimizer steps was one V3 epoch; 248 is an exact half epoch.
    "--max-steps", "248",
    "--r", "32",
    "--lora-alpha", "64",
    "--batch-size", "4",
    "--grad-accum", "2",
    "--eval-batch-size", "4",
    "--lr", "5e-5",
    "--warmup-steps", "5",
    # 31 divides 248, guaranteeing evaluation/checkpointing at the last step.
    "--eval-steps", "31",
    "--save-steps", "31",
    "--load-best-model-at-end",
    "--lora-dir", "training/qwen35-2b-voiceink-v4/lora",
    "--output-dir", "training/qwen35-2b-voiceink-v4/outputs",
    "--gguf-base", "models/Qwen3.5-2B-VoiceInk-v4",
    "--mlflow-experiment", "voiceink-training",
    "--mlflow-run-name", "qwen35-2b-voiceink-v4-sft-half-epoch",
]


def profile_argv(argv=None) -> list[str]:
    """Apply V4 defaults while retaining explicit CLI overrides."""
    overrides = sys.argv[1:] if argv is None else list(argv)
    return [*PROFILE_ARGS, *overrides]


def parse_args(argv=None):
    """Expose the resolved profile for checks and tests."""
    return shared_parse_args(profile_argv(argv))


def main(argv=None) -> None:
    shared_main(profile_argv(argv))


if __name__ == "__main__":
    main()
