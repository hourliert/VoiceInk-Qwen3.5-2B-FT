#!/usr/bin/env python3
"""Convert the locked Qwen V3 conversations to LFM string-message JSONL."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_SOURCE = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "train.jsonl"
)
DEFAULT_EVAL_SOURCE = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "eval-regression-340.jsonl"
)
DEFAULT_TRAIN_OUTPUT = ROOT / "datasets" / "lfm25-v3" / "train.jsonl"
DEFAULT_EVAL_OUTPUT = (
    ROOT / "datasets" / "lfm25-v3" / "eval-regression-340.jsonl"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert locked Qwen V3 data to LFM string messages."
    )
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--eval-source", type=Path, default=DEFAULT_EVAL_SOURCE)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    return parser.parse_args(argv)


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    raise ValueError(f"Unsupported message content: {type(content).__name__}")


def convert_conversation(record: dict, source: Path, line_number: int) -> dict:
    messages = record.get("messages")
    expected_roles = ["system", "user", "assistant"]
    if not isinstance(messages, list):
        raise ValueError(f"{source}:{line_number}: missing messages list")
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError(f"{source}:{line_number}: unexpected message roles")

    converted = []
    for message in messages:
        text = content_text(message.get("content"))
        if not text.strip():
            raise ValueError(f"{source}:{line_number}: empty message content")
        converted.append({"role": message["role"], "content": text})
    return {"messages": converted}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_file(source: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as input_stream:
        with output.open("w", encoding="utf-8") as output_stream:
            for line_number, line in enumerate(input_stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                converted = convert_conversation(record, source, line_number)
                output_stream.write(
                    json.dumps(converted, ensure_ascii=False) + "\n"
                )
                count += 1
    return count


def main(argv=None) -> None:
    args = parse_args(argv)
    for source in (args.train_source, args.eval_source):
        if not source.is_file():
            raise SystemExit(f"Source dataset not found: {source}")

    for label, source, output in (
        ("Train", args.train_source, args.train_output),
        ("Eval", args.eval_source, args.eval_output),
    ):
        count = convert_file(source, output)
        print(
            f"{label}: {source} -> {output} ({count} samples, "
            f"sha256={sha256_file(output)})"
        )


if __name__ == "__main__":
    main()
