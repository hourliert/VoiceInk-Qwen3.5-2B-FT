#!/usr/bin/env python3
"""Validate labeled dataset quality using a cheap LLM as reviewer.

Runs each labeled sample through a fast model (Haiku by default) to flag
labels that may contain hallucinations, meaning changes, over-deletions,
repetitions, or broken output. Results are written back into the labeled
JSONL file as a "validation" field on each record.

Designed as a quality gate between labeling and training — prepare_dataset.py
skips records where validation.status == "fail".

Usage:
    python3 src/labeling/validate.py --parallel 10
    python3 src/labeling/validate.py --limit 50 --dry-run
    python3 src/labeling/validate.py --force --parallel 10  # re-validate all
    python3 src/labeling/validate.py --show-failures
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

DEFAULT_INPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_MODEL = "claude-sonnet-4-6"
VALIDATE_PROMPT_PATH = Path(__file__).resolve().parent / "validate_prompt.txt"
VOCABULARY_PATH = ROOT / "config" / "vocabulary.txt"


class LabeledDataset:
    """Thread-safe labeled dataset backed by a JSONL file.

    Preserves all existing fields while allowing updates to individual records.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        with self._lock:
            return list(self._records.values())

    def update(self, request_id: str, validation: dict) -> None:
        """Set the validation field on a record and flush to disk."""
        with self._lock:
            if request_id in self._records:
                self._records[request_id]["validation"] = validation
                self._flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate labeled dataset quality using LLM reviewer.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Labeled JSONL file to validate (default: {DEFAULT_INPUT})")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to validate (0 = all)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel Claude CLI calls")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Claude model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling Claude")
    p.add_argument("--ids", nargs="*", default=None,
                   help="Validate only these specific request IDs (prefix match)")
    p.add_argument("--force", action="store_true",
                   help="Re-validate records that already have a validation field")
    p.add_argument("--shuffle", action="store_true",
                   help="Randomize the order of records before validating")
    p.add_argument("--show-failures", action="store_true",
                   help="Show all failed records and exit (no validation run)")
    p.add_argument("--inspect", nargs="*", default=None,
                   help="Show transcript, label, and validation for given IDs (prefix match)")
    return p.parse_args()


def build_prompt(record: dict, template: str, custom_vocabulary: str) -> str | None:
    """Build the validation prompt for a labeled record."""
    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return None

    transcript = components["transcript"]
    label = record.get("label", "")
    if not transcript or not label:
        return None

    return template.format(
        transcript=transcript,
        label=label,
        custom_vocabulary=custom_vocabulary,
        window_context=components["window_context"] or "(empty)",
        clipboard_context=components["clipboard_context"] or "(empty)",
    )


def call_claude(prompt: str, model: str) -> str:
    """Call the Claude CLI and return the response text."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    env["CLAUDE_CODE_SKIP_UPDATE_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
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
        timeout=120,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def parse_result(response: str) -> tuple[bool, str, str]:
    """Parse reviewer response into (passed, failure_type, reason)."""
    response = response.strip()
    if response.upper() == "PASS":
        return True, "", ""

    if response.upper().startswith("FAIL"):
        # Expected: "FAIL TYPE: reason"
        rest = response[4:].strip()
        if ":" in rest:
            failure_type, reason = rest.split(":", 1)
            return False, failure_type.strip(), reason.strip()
        return False, "UNKNOWN", rest

    # Unexpected format — treat as pass but note it
    return True, "", ""


def validate_one(record: dict, model: str, dry_run: bool,
                 template: str, custom_vocabulary: str) -> tuple[str, dict] | None:
    """Validate a single record. Returns (request_id, validation_dict) or None."""
    request_id = record["request_id"]
    prompt = build_prompt(record, template, custom_vocabulary)
    if prompt is None:
        return None

    if dry_run:
        print(f"--- DRY RUN [{request_id}] ---")
        print(prompt[:500])
        print("...\n")
        return None

    try:
        response = call_claude(prompt, model)
        passed, failure_type, reason = parse_result(response)
        if passed:
            return request_id, {"status": "pass"}
        else:
            return request_id, {"status": "fail", "type": failure_type, "reason": reason}
    except Exception as exc:
        print(f"  ERROR [{request_id}]: {exc}", file=sys.stderr)
        return None


def show_failures(dataset: LabeledDataset) -> None:
    """Print all records with validation.status == 'fail'."""
    failures = []
    for record in dataset.records():
        v = record.get("validation", {})
        if v.get("status") == "fail":
            failures.append(record)

    if not failures:
        print("No failed records.")
        return

    # Group by failure type
    by_type: dict[str, list[dict]] = {}
    for record in failures:
        v = record["validation"]
        by_type.setdefault(v.get("type", "UNKNOWN"), []).append(record)

    print(f"{len(failures)} failed records:\n")
    for ftype, records in sorted(by_type.items()):
        print(f"  {ftype} ({len(records)}):")
        for record in records:
            v = record["validation"]
            print(f"    {record['request_id']}: {v.get('reason', '')}")
        print()


def inspect_records(dataset: LabeledDataset, prefixes: list[str]) -> None:
    """Print transcript, label, and validation details for matching records.

    If prefixes is empty, shows all failed records.
    """
    if prefixes:
        matches = [r for r in dataset.records()
                   if any(r["request_id"].startswith(p) for p in prefixes)]
    else:
        matches = [r for r in dataset.records()
                   if r.get("validation", {}).get("status") == "fail"]

    if not matches:
        print("No records found." if prefixes else "No failed records.")
        return

    for i, record in enumerate(matches):
        if i > 0:
            print()

        request_id = record["request_id"]
        validation = record.get("validation", {})

        try:
            components = extract_from_record(record)
            transcript = components["transcript"]
        except (json.JSONDecodeError, KeyError):
            transcript = "(failed to extract)"

        label = record.get("label", "(no label)")

        print(f"{'='*60}")
        print(f"ID: {request_id}")
        status = validation.get("status", "not validated")
        if status == "fail":
            print(f"Validation: FAIL {validation.get('type', '')} — {validation.get('reason', '')}")
        else:
            print(f"Validation: {status}")
        print(f"\n--- RAW TRANSCRIPT ---\n{transcript}")
        print(f"\n--- LABEL ---\n{label}")
        print(f"{'='*60}")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    dataset = LabeledDataset(args.input)
    print(f"Loaded {len(dataset)} labeled records from {args.input}")

    if args.inspect is not None:
        inspect_records(dataset, args.inspect)
        return

    if args.show_failures:
        show_failures(dataset)
        return

    # Select records to validate
    all_records = dataset.records()

    if args.ids:
        to_validate = [r for r in all_records
                       if any(r["request_id"].startswith(prefix) for prefix in args.ids)]
        print(f"Filtered to {len(to_validate)} records matching --ids")
    elif args.force:
        to_validate = all_records
        print(f"Force re-validating all {len(to_validate)} records")
    else:
        to_validate = [r for r in all_records if "validation" not in r]
        already = len(all_records) - len(to_validate)
        if already:
            print(f"Skipping {already} already-validated records")
        print(f"{len(to_validate)} records need validation")

    if args.shuffle:
        random.shuffle(to_validate)
        print("Shuffled records randomly")

    if args.limit > 0:
        to_validate = to_validate[:args.limit]
        print(f"Limiting to {len(to_validate)} records")

    if not to_validate:
        print("Nothing to validate.")
        return

    template = VALIDATE_PROMPT_PATH.read_text(encoding="utf-8")
    custom_vocabulary = VOCABULARY_PATH.read_text(encoding="utf-8").strip()
    print(f"Validating with {args.model}, parallel={args.parallel}\n")

    fail_count = 0
    pass_count = 0
    errors = 0

    if args.parallel <= 1:
        for i, record in enumerate(to_validate, 1):
            print(f"[{i}/{len(to_validate)}] Validating {record['request_id']}...")
            result = validate_one(record, args.model, args.dry_run, template, custom_vocabulary)
            if result is None:
                errors += 1
                continue
            request_id, validation = result
            dataset.update(request_id, validation)
            if validation["status"] == "pass":
                pass_count += 1
            else:
                fail_count += 1
                print(f"  FAIL {validation['type']}: {validation['reason']}")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(validate_one, record, args.model, args.dry_run, template, custom_vocabulary): record
                for record in to_validate
            }
            for i, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                print(f"[{i}/{len(to_validate)}] Done {record['request_id']}")
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    errors += 1
                    continue
                if result is None:
                    errors += 1
                    continue
                request_id, validation = result
                dataset.update(request_id, validation)
                if validation["status"] == "pass":
                    pass_count += 1
                else:
                    fail_count += 1
                    print(f"  FAIL {validation['type']}: {validation['reason']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Passed: {pass_count}, Failed: {fail_count}, Errors: {errors}")

    if fail_count:
        print(f"\nRun --show-failures to review failed records.")
        print(f"Failed records will be excluded from training by prepare_dataset.py.")


if __name__ == "__main__":
    main()
