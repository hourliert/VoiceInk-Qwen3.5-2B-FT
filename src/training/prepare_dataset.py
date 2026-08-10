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
    p.add_argument("--content-format", choices=("text-blocks", "string"),
                   default="text-blocks",
                   help="Message content representation: text-blocks for Qwen VLMs "
                        "(default), string for text-only models such as LFM2.5")
    p.add_argument(
        "--message-layout",
        choices=("voiceink", "prepared"),
        default="voiceink",
        help=(
            "Message placement: voiceink reproduces production with instructions "
            "and context in system, transcript in user (default); prepared keeps "
            "the legacy training-only layout"
        ),
    )
    return p.parse_args()


def build_user_message(components: dict) -> str:
    """Reconstruct the legacy prepared-layout user message."""
    parts = []

    if components["window_context"]:
        parts.append(f"<CURRENT_WINDOW_CONTEXT>\n{components['window_context']}\n</CURRENT_WINDOW_CONTEXT>")

    if components["clipboard_context"]:
        parts.append(f"<CLIPBOARD_CONTEXT>\n{components['clipboard_context']}\n</CLIPBOARD_CONTEXT>")

    if components["custom_vocabulary"]:
        parts.append(f"<CUSTOM_VOCABULARY>\n{components['custom_vocabulary']}\n</CUSTOM_VOCABULARY>")

    parts.append(f"<TRANSCRIPT>\n{components['transcript']}\n</TRANSCRIPT>")

    return "\n\n".join(parts)


def build_voiceink_system_message(components: dict, system_prompt: str) -> str:
    """Reconstruct the system message structure sent by VoiceInk in production."""
    stripped_prompt = system_prompt.strip()
    if (
        stripped_prompt.startswith("<SYSTEM_INSTRUCTIONS>")
        and stripped_prompt.endswith("</SYSTEM_INSTRUCTIONS>")
    ):
        wrapped_prompt = stripped_prompt
    else:
        wrapped_prompt = (
            f"<SYSTEM_INSTRUCTIONS>\n{stripped_prompt}\n</SYSTEM_INSTRUCTIONS>"
        )
    parts = [wrapped_prompt]
    for tag, field in (
        ("CURRENT_WINDOW_CONTEXT", "window_context"),
        ("CLIPBOARD_CONTEXT", "clipboard_context"),
        ("CUSTOM_VOCABULARY", "custom_vocabulary"),
    ):
        if components[field]:
            parts.append(f"<{tag}>\n{components[field]}\n</{tag}>")
    return "\n\n".join(parts)


def build_voiceink_user_message(components: dict) -> str:
    """Reconstruct the transcript-only user message sent by VoiceInk."""
    return f"<TRANSCRIPT>\n{components['transcript']}\n</TRANSCRIPT>"


def message_content(text: str, content_format: str) -> str | list[dict]:
    """Represent message text in the format expected by the target model."""
    if content_format == "string":
        return text
    return [{"type": "text", "text": text}]


def content_text(content) -> str:
    """Extract text from a string or typed text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    raise ValueError(f"Unsupported message content: {type(content).__name__}")


def extract_last_tag_content(text: str, tag: str) -> str:
    """Extract the last complete tag pair, avoiding prompt examples."""
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    close_index = text.rfind(closing)
    if close_index < 0:
        return ""
    open_index = text.rfind(opening, 0, close_index)
    if open_index < 0:
        return ""
    return text[open_index + len(opening):close_index].strip()


def split_voiceink_system(text: str) -> tuple[str, str]:
    """Split the outer instructions wrapper from the exact dynamic-context tail."""
    stripped = text.strip()
    closing = "</SYSTEM_INSTRUCTIONS>"
    close_index = stripped.find(closing)
    if not stripped.startswith("<SYSTEM_INSTRUCTIONS>") or close_index < 0:
        return stripped, ""
    boundary = close_index + len(closing)
    return stripped[:boundary], stripped[boundary:].strip()


def split_prepared_user(text: str) -> tuple[str, str]:
    """Split legacy context blocks from its final outer transcript block."""
    stripped = text.strip()
    if stripped.startswith("<TRANSCRIPT>"):
        return "", stripped

    marker = "\n\n<TRANSCRIPT>"
    search_from = 0
    candidates = []
    while True:
        index = stripped.find(marker, search_from)
        if index < 0:
            break
        prefix = stripped[:index].rstrip()
        if any(
            prefix.endswith(f"</{tag}>")
            for tag in (
                "CURRENT_WINDOW_CONTEXT",
                "CLIPBOARD_CONTEXT",
                "CUSTOM_VOCABULARY",
            )
        ):
            candidates.append(index)
        search_from = index + len(marker)
    if not candidates:
        raise ValueError("Could not find the outer transcript block")
    boundary = candidates[-1]
    return stripped[:boundary].strip(), stripped[boundary + 2:].strip()


def convert_conversation_layout(conversation: dict, message_layout: str) -> dict:
    """Re-layout an existing prepared conversation without changing its label."""
    messages = conversation.get("messages", [])
    if [message.get("role") for message in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError("Conversation must contain system, user, assistant roles")

    system_text = content_text(messages[0]["content"])
    user_text = content_text(messages[1]["content"])
    assistant_text = content_text(messages[2]["content"])
    content_format = "string" if isinstance(messages[0]["content"], str) else "text-blocks"

    if message_layout == "voiceink":
        context_tail, transcript_message = split_prepared_user(user_text)
        new_system = system_text.strip()
        if context_tail:
            new_system = f"{new_system}\n\n{context_tail}"
        new_user = transcript_message
    elif message_layout == "prepared":
        system_prompt, context_tail = split_voiceink_system(system_text)
        new_system = system_prompt
        new_user = "\n\n".join(
            part for part in (context_tail, user_text.strip()) if part
        )
    else:
        raise ValueError(f"Unsupported message layout: {message_layout}")
    return {
        "messages": [
            {"role": "system", "content": message_content(new_system, content_format)},
            {"role": "user", "content": message_content(new_user, content_format)},
            {
                "role": "assistant",
                "content": message_content(assistant_text, content_format),
            },
        ]
    }


def convert_record(record: dict, system_prompt: str,
                   content_format: str = "text-blocks",
                   message_layout: str = "voiceink") -> dict | None:
    """Convert a labeled record to chat messages format.

    Extracts structured components from the original request, then
    reconstructs messages using the provided system prompt. This
    decouples training data from whatever prompt VoiceInk sent at
    recording time.

    The default typed blocks are required by the Qwen 3.5 VLM processor.
    Text-only models such as LFM2.5 use plain string content instead.
    """
    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return None

    label = record.get("label", "").strip()
    if not label or not components["transcript"]:
        return None

    if message_layout == "voiceink":
        try:
            request = json.loads(record["raw_request_json"])
            source_messages = request.get("messages", [])
            source_system = next(
                content_text(message["content"])
                for message in source_messages if message.get("role") == "system"
            )
            source_user = next(
                content_text(message["content"])
                for message in source_messages if message.get("role") == "user"
            )
            _, context_tail = split_voiceink_system(source_system)
            empty_components = {
                "window_context": "",
                "clipboard_context": "",
                "custom_vocabulary": "",
                "transcript": "",
            }
            system_content = build_voiceink_system_message(
                empty_components, system_prompt
            )
            if context_tail:
                system_content = f"{system_content}\n\n{context_tail}"
            user_content = source_user.strip()
        except (KeyError, StopIteration, TypeError, ValueError):
            system_content = build_voiceink_system_message(components, system_prompt)
            user_content = build_voiceink_user_message(components)
    elif message_layout == "prepared":
        system_content = system_prompt
        user_content = build_user_message(components)
    else:
        raise ValueError(f"Unsupported message layout: {message_layout}")

    messages = [
        {
            "role": "system",
            "content": message_content(system_content, content_format),
        },
        {
            "role": "user",
            "content": message_content(user_content, content_format),
        },
        {
            "role": "assistant",
            "content": message_content(label, content_format),
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

    # Include human-reviewed or conservatively auto-approved real records.
    before = len(records)
    records = [r for r in records
               if (r.get("manually_reviewed")
                   or (r.get("auto_review") or {}).get("status") == "approved"
                   or r.get("request_id", "").startswith("syn-"))]
    excluded = before - len(records)
    if excluded:
        print(f"Excluded {excluded} records pending or rejected by review")

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
        result = convert_record(
            record, system_prompt, args.content_format, args.message_layout
        )
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
