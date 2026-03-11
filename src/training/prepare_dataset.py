#!/usr/bin/env python3
"""Convert labeled.jsonl into Unsloth-compatible training format.

Reads the labeled dataset and produces a JSONL file where each line has a
"messages" key with the original VoiceInk messages (system + user) plus the
gold-standard label as the assistant response. The input messages are passed
through exactly as VoiceInk sends them — same system prompt, same user message.

Usage:
    python3 src/training/prepare_dataset.py
    python3 src/training/prepare_dataset.py --eval-split 0.1
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_TRAIN_OUTPUT = ROOT / "datasets" / "train.jsonl"
DEFAULT_EVAL_OUTPUT = ROOT / "datasets" / "eval.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare training data for Unsloth fine-tuning.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Input labeled JSONL (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_TRAIN_OUTPUT,
                   help=f"Output training JSONL (default: {DEFAULT_TRAIN_OUTPUT})")
    p.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT,
                   help=f"Output eval JSONL (default: {DEFAULT_EVAL_OUTPUT})")
    p.add_argument("--eval-split", type=float, default=0.1,
                   help="Fraction of data for eval set (default: 0.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/eval split")
    return p.parse_args()


def convert_record(record: dict) -> dict | None:
    """Convert a labeled record to chat messages format.

    Takes the original VoiceInk messages as-is and appends the label as
    the assistant response.
    """
    try:
        req = json.loads(record["raw_request_json"])
    except (json.JSONDecodeError, KeyError):
        return None

    label = record.get("label", "").strip()
    if not label:
        return None

    # Pass through the original messages exactly as VoiceInk sent them.
    # Qwen 3.5 is a unified VLM, so content must be a list of typed blocks
    # (even for text-only input) to avoid the vision processor treating
    # plain strings as image paths.
    messages = []
    for msg in req.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })

    if not messages:
        return None

    # Append the gold-standard label as the assistant response
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": label}],
    })

    return {"messages": messages}


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load labeled records
    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records)} labeled records")

    # Convert to training format
    converted = []
    skipped = 0
    for record in records:
        result = convert_record(record)
        if result:
            converted.append(result)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} records (missing data or label)")
    print(f"Converted {len(converted)} records")

    if not converted:
        print("Nothing to write.", file=sys.stderr)
        sys.exit(1)

    # Split train/eval
    random.seed(args.seed)
    random.shuffle(converted)

    eval_size = int(len(converted) * args.eval_split)
    eval_set = converted[:eval_size]
    train_set = converted[eval_size:]

    # Write output files
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for record in train_set:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_set)} training samples to {args.output}")

    if eval_set:
        with args.eval_output.open("w", encoding="utf-8") as f:
            for record in eval_set:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(eval_set)} eval samples to {args.eval_output}")
    else:
        print("No eval split (dataset too small or --eval-split 0)")


if __name__ == "__main__":
    main()
