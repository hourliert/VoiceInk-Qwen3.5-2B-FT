#!/usr/bin/env python3
"""Fine-tune Qwen3.5 0.8B on the locked VoiceInk V3 dataset.

This is an isolated profile over the shared Qwen3.5 trainer. It intentionally
reuses the exact V3 train and regression-eval JSONL files used by the latest
Qwen3.5 2B run while writing every adapter, checkpoint, and GGUF export to a
new 0.8B-specific directory.
"""
import sys

try:
    from .finetune import main as shared_main
    from .finetune import parse_args as shared_parse_args
except ImportError:  # Direct execution: python src/training/finetune_qwen35_08b.py
    from finetune import main as shared_main
    from finetune import parse_args as shared_parse_args


PROFILE_ARGS = [
    "--base-model", "unsloth/Qwen3.5-0.8B",
    "--train", "datasets/qwen35-2b-voiceink-v3/train.jsonl",
    "--eval", "datasets/qwen35-2b-voiceink-v3/eval-regression-340.jsonl",
    "--epochs", "1",
    "--r", "32",
    "--lora-alpha", "64",
    "--batch-size", "4",
    "--grad-accum", "2",
    "--eval-batch-size", "4",
    "--lr", "2e-4",
    "--warmup-steps", "10",
    "--eval-steps", "50",
    "--save-steps", "50",
    "--lora-dir", "training/qwen35-08b-voiceink-v1/lora",
    "--output-dir", "training/qwen35-08b-voiceink-v1/outputs",
    "--gguf-base", "models/Qwen3.5-0.8B-VoiceInk-v1",
]


def profile_argv(argv=None) -> list[str]:
    """Apply profile defaults while allowing explicit CLI overrides."""
    overrides = sys.argv[1:] if argv is None else list(argv)
    return [*PROFILE_ARGS, *overrides]


def parse_args(argv=None):
    """Expose the resolved profile for tests and preflight tooling."""
    return shared_parse_args(profile_argv(argv))


def main(argv=None) -> None:
    shared_main(profile_argv(argv))


if __name__ == "__main__":
    main()
