#!/usr/bin/env python3
"""Refresh an audit dataset with outputs from the current local model alias."""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluate import query_llama
from labeling.validate import LabeledDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add fresh local-model outputs to a label audit JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.5-2B-VoiceInk")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_messages(record: dict) -> list[dict]:
    request = json.loads(record["raw_request_json"])
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("raw request has no messages")
    return messages


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.parallel < 1:
        print("--parallel must be at least 1", file=sys.stderr)
        sys.exit(2)

    dataset = LabeledDataset(args.input)
    records = dataset.records()
    pending = [
        record for record in records
        if args.force or record.get("current_model") != args.model
        or not record.get("current_model_response")
    ]
    print(f"Loaded {len(records)} audit records; {len(pending)} need {args.model}")
    if not pending:
        return

    print(f"Generating locally with parallel={args.parallel}...")
    write_lock = threading.Lock()

    def generate(record: dict) -> tuple[str, str, float]:
        text, duration_ms = query_llama(
            request_messages(record), args.model, args.host, args.port
        )
        if not text.strip():
            raise ValueError("local model returned an empty response")
        return record["request_id"], text.strip(), duration_ms

    errors = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(generate, record): record for record in pending}
        for index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                request_id, text, duration_ms = future.result()
                with write_lock:
                    with dataset._lock:
                        stored = dataset._records[request_id]
                        stored["current_model"] = args.model
                        stored["current_model_response"] = text
                        stored["current_model_duration_ms"] = round(duration_ms, 3)
                        dataset._flush()
                print(f"[{index}/{len(pending)}] {request_id} {duration_ms:.0f}ms")
            except Exception as exc:
                errors += 1
                print(
                    f"[{index}/{len(pending)}] ERROR {record['request_id']}: {exc}",
                    file=sys.stderr,
                )

    print(f"Done. Refreshed: {len(pending) - errors}; errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
