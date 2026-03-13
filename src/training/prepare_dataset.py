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
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_from_record

DEFAULT_INPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_TRAIN_OUTPUT = ROOT / "datasets" / "train.jsonl"
DEFAULT_EVAL_OUTPUT = ROOT / "datasets" / "eval.jsonl"
DEFAULT_SYSTEM_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"


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
    p.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT,
                   help=f"System prompt file to use for training (default: {DEFAULT_SYSTEM_PROMPT})")
    p.add_argument("--extra-input", type=Path, nargs="*", default=[],
                   help="Additional labeled JSONL files to merge (e.g., synthetic data)")
    return p.parse_args()


def build_user_message(components: dict) -> str:
    """Reconstruct the user message from extracted components."""
    parts = []

    if components["window_context"]:
        parts.append(f"<CURRENT_WINDOW_CONTEXT>\n{components['window_context']}\n</CURRENT_WINDOW_CONTEXT>")

    if components["clipboard_context"]:
        parts.append(f"<CLIPBOARD_CONTEXT>\n{components['clipboard_context']}\n</CLIPBOARD_CONTEXT>")

    if components["custom_vocabulary"]:
        parts.append(f"<CUSTOM_VOCABULARY>\n{components['custom_vocabulary']}\n</CUSTOM_VOCABULARY>")

    parts.append(f"<TRANSCRIPT>\n{components['transcript']}\n</TRANSCRIPT>")

    return "\n\n".join(parts)


def convert_record(record: dict, system_prompt: str) -> dict | None:
    """Convert a labeled record to chat messages format.

    Extracts structured components from the original request, then
    reconstructs messages using the provided system prompt. This
    decouples training data from whatever prompt VoiceInk sent at
    recording time.

    Qwen 3.5 is a unified VLM, so content must be a list of typed blocks
    (even for text-only input) to avoid the vision processor treating
    plain strings as image paths.
    """
    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return None

    label = record.get("label", "").strip()
    if not label or not components["transcript"]:
        return None

    user_content = build_user_message(components)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label}],
        },
    ]

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

    # Load extra inputs (e.g., synthetic data)
    for extra_path in args.extra_input:
        if not extra_path.exists():
            print(f"Extra input not found: {extra_path}", file=sys.stderr)
            sys.exit(1)
        extra_count = 0
        with extra_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                    extra_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {extra_count} extra records from {extra_path}")
    if args.extra_input:
        print(f"Total records after merge: {len(records)}")

    # Only include manually reviewed records (synthetic data is always included)
    before = len(records)
    records = [r for r in records
               if r.get("manually_reviewed") or r.get("request_id", "").startswith("syn-")]
    excluded = before - len(records)
    if excluded:
        print(f"Excluded {excluded} records not manually reviewed")

    # Load system prompt
    if not args.system_prompt.exists():
        print(f"System prompt not found: {args.system_prompt}", file=sys.stderr)
        sys.exit(1)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    print(f"Using system prompt from {args.system_prompt} ({len(system_prompt)} chars)")

    # Convert to training format
    converted = []
    skipped = 0
    for record in records:
        result = convert_record(record, system_prompt)
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
