#!/usr/bin/env python3
"""Generate gold-standard labels for VoiceInk transcription samples using Claude as judge.

Reads raw request logs, sends the original VoiceInk input (system prompt + user
message) to Claude with a judge wrapper prompt, and writes the labeled dataset
to a JSONL file. Already-labeled request IDs are skipped.

Usage:
    python3 src/labeling/label.py --limit 10
    python3 src/labeling/label.py --limit 50 --parallel 4
    python3 src/labeling/label.py  # label everything
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_from_record
DEFAULT_INPUT = ROOT / "logs" / "voiceink_proxy_requests.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_MODEL = "claude-sonnet-4-6"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label VoiceInk transcription samples using Claude as judge.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Input JSONL log file (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output labeled JSONL file (default: {DEFAULT_OUTPUT})")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to label (0 = all)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel Claude CLI calls")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Claude model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling Claude")
    p.add_argument("--longest", type=int, default=0,
                   help="Pick the N longest transcripts (by response length)")
    p.add_argument("--ids", nargs="*", default=None,
                   help="Label only these specific request IDs (prefix match)")
    p.add_argument("--force", action="store_true",
                   help="Re-label even if already labeled (use with --ids)")
    p.add_argument("--shuffle", action="store_true",
                   help="Randomize the order of entries before labeling")
    return p.parse_args()


def load_logs(path: Path) -> list[dict]:
    """Load and parse the proxy request log."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("request_json_valid") and record.get("response_json_valid"):
                records.append(record)
    return records


class LabeledDataset:
    """Thread-safe labeled dataset backed by a JSONL file.

    Enforces uniqueness by request_id: saving a record with an existing ID
    replaces the old entry. All reads and writes go through this class.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}  # request_id -> record
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._records[record["request_id"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue

    def _flush(self) -> None:
        """Rewrite the entire file from the in-memory dict."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def labeled_ids(self) -> set[str]:
        with self._lock:
            return set(self._records.keys())

    def save(self, record: dict) -> None:
        """Save a labeled record, replacing any existing entry with the same request_id."""
        with self._lock:
            self._records[record["request_id"]] = record
            self._flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def build_prompt(record: dict, judge_template: str) -> str | None:
    """Build the judge prompt with structured context from a record.

    Extracts transcript, vocabulary, clipboard, and window context
    from the record and passes them as separate placeholders to the
    judge prompt template.
    """
    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return None

    if not components["transcript"]:
        return None

    return judge_template.format(
        transcript=components["transcript"],
        custom_vocabulary=components["custom_vocabulary"] or "(none provided)",
        clipboard_context=components["clipboard_context"] or "(empty)",
        window_context=components["window_context"] or "(empty)",
    )


def call_claude(prompt: str, model: str) -> str:
    """Call the Claude CLI and return the response text."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--model", model,
            "--disable-slash-commands",
            "--allowed-tools", "",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def label_one(record: dict, model: str, dry_run: bool, judge_template: str) -> dict | None:
    """Label a single record. Returns the labeled record or None on failure."""
    prompt = build_prompt(record, judge_template)
    if prompt is None:
        return None

    if dry_run:
        print(f"--- DRY RUN [{record['request_id']}] ---")
        print(prompt[:500])
        print("...")
        return None

    try:
        label = call_claude(prompt, model)
    except Exception as exc:
        print(f"  ERROR [{record['request_id']}]: {exc}", file=sys.stderr)
        return None

    return {
        "request_id": record["request_id"],
        "timestamp": record["timestamp"],
        "model_used_for_label": model,
        "original_model": record.get("model", ""),
        "raw_request_json": record.get("raw_request_json", ""),
        "original_response": record.get("response_text", ""),
        "label": label,
    }


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(args.input)
    print(f"Loaded {len(logs)} valid log entries from {args.input}")

    dataset = LabeledDataset(args.output)
    labeled_ids = dataset.labeled_ids()
    if labeled_ids:
        print(f"Found {len(labeled_ids)} already-labeled entries in {args.output}")

    # Filter by specific IDs (prefix match) -- skip dedup when --force is set
    if args.ids:
        pool = logs if args.force else [r for r in logs if r["request_id"] not in labeled_ids]
        to_label = [r for r in pool
                    if any(r["request_id"].startswith(prefix) for prefix in args.ids)]
        print(f"Filtered to {len(to_label)} entries matching --ids{' (force)' if args.force else ''}")
    else:
        to_label = [r for r in logs if r["request_id"] not in labeled_ids]
        print(f"{len(to_label)} entries need labeling")

    # Shuffle entries
    if args.shuffle:
        random.shuffle(to_label)
        print("Shuffled entries randomly")

    # Pick the N longest by response text length
    if args.longest > 0:
        to_label.sort(key=lambda r: len(r.get("response_text", "")), reverse=True)
        to_label = to_label[:args.longest]
        print(f"Selected {len(to_label)} longest entries")

    if args.limit > 0:
        to_label = to_label[:args.limit]
        print(f"Limiting to {len(to_label)} entries")

    if not to_label:
        print("Nothing to label.")
        return

    # Read template once, pass to all workers
    judge_template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    labeled_count = 0
    error_count = 0

    if args.parallel <= 1:
        for i, record in enumerate(to_label, 1):
            print(f"[{i}/{len(to_label)}] Labeling {record['request_id']}...")
            result = label_one(record, args.model, args.dry_run, judge_template)
            if result:
                dataset.save(result)
                labeled_count += 1
            else:
                error_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(label_one, record, args.model, args.dry_run, judge_template): record
                for record in to_label
            }
            for i, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                print(f"[{i}/{len(to_label)}] Done {record['request_id']}")
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    error_count += 1
                    continue
                if result:
                    dataset.save(result)
                    labeled_count += 1
                else:
                    error_count += 1

    print(f"\nDone. Labeled: {labeled_count}, Errors: {error_count}, Total: {len(dataset)}")
    if labeled_count:
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
