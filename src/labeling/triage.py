#!/usr/bin/env python3
"""Classify validated labels into automatic approval or human review."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.extract import extract_from_record
from labeling.review_server import suspicion_reasons


AUTO_REVIEW_POLICY = "validator-pass-and-local-low-ambiguity-v1"


def classify_record(record: dict) -> tuple[str, list[str]]:
    """Return auto-approved, human-review, or human-complete with reasons."""
    manual_status = (record.get("manual_review") or {}).get("status")
    if manual_status in {"approved", "edited", "rejected"}:
        return "human-complete", [f"manual decision: {manual_status}"]

    validation = record.get("validation") or {}
    if validation.get("status") != "pass":
        reason = (
            f"validator {validation.get('status')}"
            if validation.get("status")
            else "missing validation result"
        )
        return "human-review", [reason]

    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return "human-review", ["request extraction failed"]
    reasons = suspicion_reasons(record, components)
    if reasons:
        return "human-review", reasons
    return "auto-approved", []


def triage_records(records: list[dict], migrate_legacy_reviewed: bool,
                   timestamp: str) -> dict[str, int]:
    counts = {
        "auto-approved": 0,
        "human-review": 0,
        "human-complete": 0,
        "legacy-preserved": 0,
    }
    for record in records:
        manual_status = (record.get("manual_review") or {}).get("status")
        legacy_reviewed = bool(record.get("manually_reviewed")) and not manual_status
        if legacy_reviewed and not migrate_legacy_reviewed:
            record.pop("auto_review", None)
            counts["legacy-preserved"] += 1
            continue
        if legacy_reviewed:
            record.pop("manually_reviewed", None)

        classification, reasons = classify_record(record)
        if classification == "auto-approved":
            existing = record.get("auto_review") or {}
            record["auto_review"] = {
                "status": "approved",
                "policy": AUTO_REVIEW_POLICY,
                "timestamp": existing.get("timestamp", timestamp),
            }
        else:
            record.pop("auto_review", None)
        if classification == "human-review":
            record["review_queue_reason"] = "; ".join(reasons)
        else:
            record.pop("review_queue_reason", None)
        counts[classification] += 1
    return counts


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-approve validated low-ambiguity labels"
    )
    parser.add_argument(
        "--input", dest="inputs", type=Path, action="append", required=True,
        help="Labeled JSONL file; repeat for multiple batches",
    )
    parser.add_argument(
        "--migrate-legacy-reviewed", action="store_true",
        help="Reclassify manually_reviewed records that lack manual_review provenance",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write classifications back to each input (default is preview only)",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    for path in args.inputs:
        if not path.is_file():
            print(f"Input file not found: {path}", file=sys.stderr)
            sys.exit(1)
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts = triage_records(
            records, args.migrate_legacy_reviewed, timestamp
        )
        print(f"Triage: {path}")
        print(f"  Records: {len(records)}")
        print(f"  Auto-approved: {counts['auto-approved']}")
        print(f"  Human-review queue: {counts['human-review']}")
        print(f"  Existing human decisions: {counts['human-complete']}")
        if counts["legacy-preserved"]:
            print(f"  Legacy reviewed preserved: {counts['legacy-preserved']}")
        if args.apply:
            write_records(path, records)
            print("  Applied")
        else:
            print("  Preview only; pass --apply to write")


if __name__ == "__main__":
    main()
