#!/usr/bin/env python3
"""Build and score a deterministic calibration set for the label validator.

The build step uses manually reviewed reference labels as known-good examples
and creates clearly defective variants without calling an external model. The
normal validate.py command then reviews the fixture, and the score step checks
whether material defects were caught without rejecting too many good labels.
"""
import argparse
import copy
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_from_record

FAILURE_TYPES = (
    "HALLUCINATION",
    "MEANING_ALTERED",
    "OVER_DELETION",
    "REPETITION",
    "BROKEN_OUTPUT",
    "UNDER_CLEANING",
)
CRITICAL_TYPES = {"HALLUCINATION", "MEANING_ALTERED", "OVER_DELETION"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or score a deterministic Luna validator calibration set."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Create a calibration JSONL fixture")
    build.add_argument("--input", type=Path, required=True,
                       help="Labeller calibration JSONL containing reviewed references")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--good-count", type=int, default=20)
    build.add_argument("--bad-count", type=int, default=30)

    score = subparsers.add_parser("score", help="Score completed validation results")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--min-defect-recall", type=float, default=0.90)
    score.add_argument("--max-good-fail-rate", type=float, default=0.15)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def calibration_record(source: dict, label: str, suffix: str,
                       expected_status: str,
                       expected_type: str = "") -> dict:
    record = copy.deepcopy(source)
    source_id = str(source["request_id"])
    record["request_id"] = f"{source_id}--validator-{suffix}"
    record["label"] = label.strip()
    record.pop("validation", None)
    record.pop("manually_reviewed", None)
    record["calibration_source_id"] = source_id
    record["expected_validation"] = expected_status
    record["expected_failure_type"] = expected_type
    return record


def flip_meaning(label: str) -> str:
    replacements = (
        (r"\bshould not\b", "should"),
        (r"\bshould\b", "should not"),
        (r"\bdo not\b", "do"),
        (r"\bdon't\b", "do"),
        (r"\bwill not\b", "will"),
        (r"\bwill\b", "will not"),
        (r"\bcan\b", "cannot"),
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
    )
    for pattern, replacement in replacements:
        changed, count = re.subn(pattern, replacement, label, count=1,
                                 flags=re.IGNORECASE)
        if count:
            return changed
    return "Do not proceed. " + label


def over_delete(label: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", label.strip())
    if len(sentences) > 1:
        return " ".join(sentences[:max(1, len(sentences) // 2)])
    words = label.split()
    return " ".join(words[:max(1, len(words) // 2)]).rstrip(".,;:") + "."


def repeat_content(label: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", label.strip())
    repeated = sentences[-1] if sentences else label
    return f"{label.rstrip()} {repeated}"


def break_output(label: str) -> str:
    words = label.split()
    cutoff = max(2, len(words) // 2)
    return " ".join(words[:cutoff]).rstrip(".,;:!?")


def under_clean(source: dict, reference: str) -> str:
    try:
        raw = extract_from_record(source)["transcript"].strip()
    except (json.JSONDecodeError, KeyError):
        raw = ""
    if raw and SequenceMatcher(None, raw, reference).ratio() < 0.995:
        return raw
    return "Okay, so " + reference[0].lower() + reference[1:]


def corrupt_label(source: dict, failure_type: str) -> str:
    reference = str(source["reference_label"]).strip()
    if failure_type == "HALLUCINATION":
        return f"{reference.rstrip()} The deployment key is VX-9472."
    if failure_type == "MEANING_ALTERED":
        return flip_meaning(reference)
    if failure_type == "OVER_DELETION":
        return over_delete(reference)
    if failure_type == "REPETITION":
        return repeat_content(reference)
    if failure_type == "BROKEN_OUTPUT":
        return break_output(reference)
    if failure_type == "UNDER_CLEANING":
        return under_clean(source, reference)
    raise ValueError(f"Unsupported failure type: {failure_type}")


def build_fixture(records: list[dict], good_count: int,
                  bad_count: int) -> list[dict]:
    references = [
        record for record in records
        if record.get("reference_label")
        and record.get("reference_manually_reviewed")
    ]
    if not references:
        raise ValueError("No manually reviewed reference labels found")
    if good_count < 0 or bad_count < 0:
        raise ValueError("Calibration counts cannot be negative")

    fixture = []
    for index in range(good_count):
        source = references[index % len(references)]
        fixture.append(calibration_record(
            source,
            str(source["reference_label"]),
            f"good-{index:03d}",
            "pass",
        ))

    for index in range(bad_count):
        source = references[index % len(references)]
        failure_type = FAILURE_TYPES[index % len(FAILURE_TYPES)]
        fixture.append(calibration_record(
            source,
            corrupt_label(source, failure_type),
            f"bad-{index:03d}-{failure_type.lower()}",
            "fail",
            failure_type,
        ))
    return fixture


def score_fixture(records: list[dict], min_defect_recall: float,
                  max_good_fail_rate: float) -> bool:
    good = [record for record in records
            if record.get("expected_validation") == "pass"]
    bad = [record for record in records
           if record.get("expected_validation") == "fail"]
    missing = [record for record in records
               if (record.get("validation") or {}).get("status") not in {"pass", "fail"}]
    false_passes = [record for record in bad
                    if record.get("validation", {}).get("status") == "pass"]
    false_fails = [record for record in good
                   if record.get("validation", {}).get("status") == "fail"]
    caught = len(bad) - len(false_passes)
    defect_recall = caught / len(bad) if bad else 1.0
    good_fail_rate = len(false_fails) / len(good) if good else 0.0
    critical_false_passes = [
        record for record in false_passes
        if record.get("expected_failure_type") in CRITICAL_TYPES
    ]
    exact_types = sum(
        record.get("validation", {}).get("type")
        == record.get("expected_failure_type")
        for record in bad
        if record.get("validation", {}).get("status") == "fail"
    )

    print("Luna label-validator calibration")
    print(f"  Good labels: {len(good)}; false failures: {len(false_fails)} "
          f"({good_fail_rate:.1%})")
    print(f"  Defective labels: {len(bad)}; caught: {caught} "
          f"({defect_recall:.1%})")
    print(f"  Critical false passes: {len(critical_false_passes)}")
    print(f"  Exact failure-type matches: {exact_types}/{caught}")
    print(f"  Missing validation results: {len(missing)}")

    if false_passes:
        print("\nFalse passes:")
        for record in false_passes:
            print(f"  {record['request_id']}: {record.get('expected_failure_type')}")
    if false_fails:
        print("\nFalse failures:")
        for record in false_fails:
            reason = record.get("validation", {}).get("reason", "")
            print(f"  {record['request_id']}: {reason}")

    passed = (
        not missing
        and defect_recall >= min_defect_recall
        and good_fail_rate <= max_good_fail_rate
        and not critical_false_passes
    )
    print(f"\nCalibration gate: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    records = load_jsonl(args.input)
    if args.command == "build":
        try:
            fixture = build_fixture(records, args.good_count, args.bad_count)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        write_jsonl(args.output, fixture)
        print(f"Wrote {len(fixture)} calibration records to {args.output}")
        print(f"  Known good: {args.good_count}; deterministic defects: {args.bad_count}")
        return

    passed = score_fixture(
        records,
        min_defect_recall=args.min_defect_recall,
        max_good_fail_rate=args.max_good_fail_rate,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
