#!/usr/bin/env python3
"""Select a reproducible, non-overlapping pilot from unlabeled VoiceInk logs."""
import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_from_record
from labeling.label import load_logs

DEFAULT_LOGS = ROOT / "logs" / "voiceink_proxy_requests.jsonl"
DEFAULT_LABELED = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "strategic" / "pilot-100-manifest.jsonl"
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a reproducible strategic labeling manifest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    parser.add_argument(
        "--calibration-references", type=Path, default=None,
        help="Select only reviewed reference IDs instead of the unlabeled pool",
    )
    parser.add_argument(
        "--exclude-ids-file", type=Path, default=None,
        help="Exclude request IDs listed in a text file or JSONL manifest",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--representative", type=int, default=40)
    parser.add_argument("--context-rich", type=int, default=20)
    parser.add_argument("--long", type=int, default=20)
    parser.add_argument("--qwen-risk", type=int, default=20)
    parser.add_argument("--long-threshold", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_labeled_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                request_id = json.loads(line).get("request_id")
            except json.JSONDecodeError:
                continue
            if request_id:
                ids.add(str(request_id))
    return ids


def normalize_transcript(text: str) -> str:
    return " ".join(text.casefold().split())


def qwen_risk(raw: str, output: str) -> tuple[float, list[str]]:
    reasons = []
    if not output.strip():
        return 100.0, ["empty production output"]
    raw_numbers = set(NUMBER_RE.findall(raw))
    output_numbers = set(NUMBER_RE.findall(output))
    missing_numbers = raw_numbers - output_numbers
    added_numbers = output_numbers - raw_numbers
    if missing_numbers:
        reasons.append(f"drops {len(missing_numbers)} numeric token(s)")
    if added_numbers:
        reasons.append(f"adds {len(added_numbers)} numeric token(s)")

    length_ratio = len(output) / max(1, len(raw))
    if length_ratio < 0.60:
        reasons.append(f"short output ratio {length_ratio:.2f}")
    elif length_ratio > 1.25:
        reasons.append(f"long output ratio {length_ratio:.2f}")

    similarity = difflib.SequenceMatcher(None, raw.casefold(), output.casefold()).ratio()
    if similarity < 0.55:
        reasons.append(f"low raw/output similarity {similarity:.2f}")

    score = (
        4 * len(missing_numbers)
        + 3 * len(added_numbers)
        + max(0.0, 0.65 - similarity) * 10
        + max(0.0, 0.70 - length_ratio) * 8
        + max(0.0, length_ratio - 1.15) * 5
    )
    return score, reasons or ["largest remaining production edit"]


def build_pool(logs: list[dict], labeled_ids: set[str],
               eligible_ids: set[str] | None = None,
               excluded_ids: set[str] | None = None) -> list[dict]:
    excluded_ids = excluded_ids or set()
    by_id = {}
    for record in logs:
        request_id = record.get("request_id")
        if (
            request_id
            and request_id not in labeled_ids
            and request_id not in excluded_ids
            and (eligible_ids is None or request_id in eligible_ids)
        ):
            by_id[str(request_id)] = record

    pool = []
    seen_transcripts = set()
    for request_id, record in by_id.items():
        try:
            components = extract_from_record(record)
        except (json.JSONDecodeError, KeyError):
            continue
        transcript = components["transcript"].strip()
        normalized = normalize_transcript(transcript)
        if not normalized or normalized in seen_transcripts:
            continue
        seen_transcripts.add(normalized)
        output = str(record.get("response_text", ""))
        risk_score, risk_reasons = qwen_risk(transcript, output)
        context_score = (
            (4 if components["custom_vocabulary"] else 0)
            + min(4, len(components["window_context"]) / 500)
            + min(3, len(components["clipboard_context"]) / 500)
        )
        pool.append({
            "request_id": request_id,
            "timestamp": record.get("timestamp", ""),
            "transcript_length": len(transcript),
            "has_window_context": bool(components["window_context"]),
            "has_clipboard_context": bool(components["clipboard_context"]),
            "has_custom_vocabulary": bool(components["custom_vocabulary"]),
            "context_score": round(context_score, 3),
            "qwen_risk_score": round(risk_score, 3),
            "qwen_risk_reasons": risk_reasons,
        })
    return pool


def take_ranked(candidates: list[dict], count: int, selected: set[str],
                key) -> list[dict]:
    available = [record for record in candidates
                 if record["request_id"] not in selected]
    chosen = sorted(available, key=key, reverse=True)[:count]
    selected.update(record["request_id"] for record in chosen)
    return chosen


def select_manifest(pool: list[dict], representative: int, context_rich: int,
                    long_count: int, qwen_risk_count: int,
                    long_threshold: int, seed: int) -> list[dict]:
    requested = representative + context_rich + long_count + qwen_risk_count
    if min(representative, context_rich, long_count, qwen_risk_count) < 0:
        raise ValueError("Selection counts cannot be negative")
    if len(pool) < requested:
        raise ValueError(f"Need {requested} unique candidates, found {len(pool)}")

    selected = set()
    result = []

    risk = take_ranked(
        pool, qwen_risk_count, selected,
        key=lambda record: (record["qwen_risk_score"], record["transcript_length"]),
    )
    result.extend({**record, "stratum": "qwen-risk"} for record in risk)

    long_candidates = [record for record in pool
                       if record["transcript_length"] >= long_threshold]
    if len([record for record in long_candidates
            if record["request_id"] not in selected]) < long_count:
        long_candidates = pool
    long_records = take_ranked(
        long_candidates, long_count, selected,
        key=lambda record: record["transcript_length"],
    )
    result.extend({**record, "stratum": "long"} for record in long_records)

    contextual_candidates = [
        record for record in pool
        if record["context_score"] > 0
        and record["transcript_length"] < long_threshold
    ]
    if len([record for record in contextual_candidates
            if record["request_id"] not in selected]) < context_rich:
        contextual_candidates = [
            record for record in pool if record["context_score"] > 0
        ]
    contextual = []
    length_buckets = (
        (0, 150),
        (150, 300),
        (300, 600),
        (600, long_threshold),
    )
    base_per_bucket, remainder = divmod(context_rich, len(length_buckets))
    for bucket_index, (minimum, maximum) in enumerate(length_buckets):
        bucket_count = base_per_bucket + (bucket_index < remainder)
        bucket = [
            record for record in contextual_candidates
            if minimum <= record["transcript_length"] < maximum
        ]
        contextual.extend(take_ranked(
            bucket, bucket_count, selected,
            key=lambda record: record["context_score"],
        ))
    if len(contextual) < context_rich:
        contextual.extend(take_ranked(
            contextual_candidates, context_rich - len(contextual), selected,
            key=lambda record: record["context_score"],
        ))
    result.extend({**record, "stratum": "context-rich"} for record in contextual)

    remaining = [record for record in pool if record["request_id"] not in selected]
    rng = random.Random(seed)
    representative_records = rng.sample(remaining, representative)
    result.extend({**record, "stratum": "representative"}
                  for record in representative_records)
    return result


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    calibration_ids = None
    labeled_ids = load_labeled_ids(args.labeled)
    if args.calibration_references is not None:
        if not args.calibration_references.is_file():
            print(
                f"Calibration references not found: {args.calibration_references}",
                file=sys.stderr,
            )
            sys.exit(1)
        calibration_ids = load_labeled_ids(args.calibration_references)
        labeled_ids = set()
    if args.exclude_ids_file is not None and not args.exclude_ids_file.is_file():
        print(f"Exclusion ID file not found: {args.exclude_ids_file}", file=sys.stderr)
        sys.exit(1)
    excluded_ids = (
        load_labeled_ids(args.exclude_ids_file)
        if args.exclude_ids_file is not None else set()
    )
    logs = load_logs(args.input)
    pool = build_pool(
        logs,
        labeled_ids,
        eligible_ids=calibration_ids,
        excluded_ids=excluded_ids,
    )
    try:
        manifest = select_manifest(
            pool,
            representative=args.representative,
            context_rich=args.context_rich,
            long_count=args.long,
            qwen_risk_count=args.qwen_risk,
            long_threshold=args.long_threshold,
            seed=args.seed,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    write_manifest(args.output, manifest)

    print("Strategic labeling manifest")
    print(f"  Valid logs: {len(logs)}")
    print(f"  Existing labeled IDs excluded: {len(labeled_ids)}")
    if calibration_ids is not None:
        print(f"  Reviewed calibration IDs eligible: {len(calibration_ids)}")
    if excluded_ids:
        print(f"  Explicit IDs excluded: {len(excluded_ids)}")
    print(f"  Unique eligible transcript pool: {len(pool)}")
    for stratum in ("representative", "context-rich", "long", "qwen-risk"):
        print(f"  {stratum}: {sum(r['stratum'] == stratum for r in manifest)}")
    print(f"  Output: {args.output}")
    print("No external model was called.")


if __name__ == "__main__":
    main()
