#!/usr/bin/env python3
"""Build an explicit, fail-closed approval manifest for reviewed labels."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from labeling.label import load_id_prefixes


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def build_approval_manifest(records: list[dict], accepted_failures: set[str],
                            excluded_ids: set[str]) -> list[dict]:
    """Combine validator passes and explicit adjudication without implicit gaps."""
    if accepted_failures & excluded_ids:
        overlap = sorted(accepted_failures & excluded_ids)
        raise ValueError(f"IDs cannot be both accepted and excluded: {overlap}")

    by_id = {str(record.get("request_id", "")): record for record in records}
    requested_ids = accepted_failures | excluded_ids
    missing_ids = sorted(requested_ids - by_id.keys())
    if missing_ids:
        raise ValueError(f"Adjudication IDs not found: {missing_ids}")

    approved = []
    unresolved = []
    for request_id, record in by_id.items():
        status = (record.get("validation") or {}).get("status")
        if request_id in excluded_ids:
            continue
        if request_id in accepted_failures:
            decision = "manual-accept"
        elif status == "pass":
            decision = "validator-pass"
        elif status == "fail":
            unresolved.append(request_id)
            continue
        else:
            unresolved.append(request_id)
            continue
        approved.append({
            "request_id": request_id,
            "decision": decision,
            "validation_status": status or "missing",
        })

    if unresolved:
        raise ValueError(
            "Every failed or unvalidated record must be explicitly accepted or "
            f"excluded; unresolved IDs: {sorted(unresolved)}"
        )
    return approved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit approval manifest from validation and adjudication."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accept-ids-file", type=Path, required=True)
    parser.add_argument("--exclude-ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records = load_records(args.input)
        accepted = set(load_id_prefixes(args.accept_ids_file))
        excluded = set(load_id_prefixes(args.exclude_ids_file))
        approved = build_approval_manifest(records, accepted, excluded)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in approved:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Label adjudication manifest")
    print(f"  Input records: {len(records)}")
    print(f"  Explicitly accepted validator failures: {len(accepted)}")
    print(f"  Explicitly excluded records: {len(excluded)}")
    print(f"  Approved records: {len(approved)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
