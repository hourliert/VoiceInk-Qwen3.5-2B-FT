#!/usr/bin/env python3
"""Fine-tune Empero's Qwen3.8 2B distill on the locked VoiceInk V3 corpus.

The upstream model retains the Qwen3.5-2B architecture, tokenizer, and chat
template, so this profile intentionally reuses the hardened Qwen SFT trainer.
All checkpoints, adapters, MLflow metadata, and GGUF exports are isolated from
the production Qwen3.5-2B-VoiceInk-v3 model.
"""
import sys

try:
    from .finetune import main as shared_main
    from .finetune import parse_args as shared_parse_args
except ImportError:  # Direct execution from the repository root.
    from finetune import main as shared_main
    from finetune import parse_args as shared_parse_args


PROFILE_ARGS = [
    "--base-model", "models/Qwen3.8-2B",
    "--train", "datasets/qwen35-2b-voiceink-v3/train.jsonl",
    # Use the complete immutable holdout by construction, not the legacy 340.
    "--eval", "datasets/qwen35-2b-voiceink-v3/eval-all-440.jsonl",
    "--epochs", "1",
    "--r", "32",
    "--lora-alpha", "64",
    "--batch-size", "4",
    "--grad-accum", "2",
    "--eval-batch-size", "4",
    "--lr", "2e-4",
    "--warmup-steps", "10",
    # 62 divides the expected 496 steps, guaranteeing terminal evaluation.
    "--eval-steps", "62",
    "--save-steps", "62",
    "--load-best-model-at-end",
    "--lora-dir", "training/qwen38-2b-voiceink-v1/lora",
    "--output-dir", "training/qwen38-2b-voiceink-v1/outputs",
    "--gguf-base", "models/Qwen3.8-2B-VoiceInk-v1",
    "--mlflow-experiment", "voiceink-training",
    "--mlflow-run-name", "qwen38-2b-voiceink-v1-sft",
]


def profile_argv(argv=None) -> list[str]:
    """Apply profile defaults while retaining explicit CLI overrides."""
    overrides = sys.argv[1:] if argv is None else list(argv)
    return [*PROFILE_ARGS, *overrides]


def parse_args(argv=None):
    """Expose the resolved profile for checks and tests."""
    return shared_parse_args(profile_argv(argv))


def main(argv=None) -> None:
    shared_main(profile_argv(argv))


if __name__ == "__main__":
    main()
