#!/usr/bin/env python3
"""Fine-tune LFM2.5 1.2B on the exact locked VoiceInk V3 samples."""
import sys
from pathlib import Path

try:
    from .finetune_lfm25 import TrainingProfile, main as shared_main
    from .finetune_lfm25 import parse_args as shared_parse_args
except ImportError:
    from finetune_lfm25 import TrainingProfile, main as shared_main
    from finetune_lfm25 import parse_args as shared_parse_args


PROFILE = TrainingProfile(
    name="LFM2.5 1.2B VoiceInk V3",
    description="Fine-tune LFM2.5 1.2B on the locked VoiceInk V3 dataset.",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    lora_dir=Path("training/lfm25-1.2b-voiceink-v3/lora"),
    output_dir=Path("training/lfm25-1.2b-voiceink-v3/outputs"),
    gguf_base=Path("models/LFM2.5-1.2B-VoiceInk-v3"),
    r=16,
    lora_alpha=16,
    epochs=1,
    learning_rate=2e-4,
)

PROFILE_ARGS = [
    "--train", "datasets/lfm25-v3/train.jsonl",
    "--eval", "datasets/lfm25-v3/eval-regression-340.jsonl",
    "--batch-size", "4",
    "--grad-accum", "2",
    "--eval-batch-size", "4",
    "--eval-steps", "50",
    "--save-steps", "50",
    "--load-best-model-at-end",
]


def profile_argv(argv=None) -> list[str]:
    overrides = sys.argv[1:] if argv is None else list(argv)
    return [*PROFILE_ARGS, *overrides]


def parse_args(argv=None):
    return shared_parse_args(PROFILE, profile_argv(argv))


def main(argv=None) -> None:
    shared_main(PROFILE, profile_argv(argv))


if __name__ == "__main__":
    main()
