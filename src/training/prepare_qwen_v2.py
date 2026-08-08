#!/usr/bin/env python3
"""Build the locked Qwen3.5-2B-VoiceInk-v2 train and evaluation datasets.

The legacy regression eval is copied byte-for-byte. A proportional,
length-stratified holdout is selected from the reviewed strategic labels, and
all remaining unique strategic examples are added to the legacy train set.
"""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.extract import extract_from_record
from training.prepare_dataset import convert_record

DEFAULT_BASE_TRAIN = ROOT / "datasets" / "train.jsonl"
DEFAULT_BASE_EVAL = ROOT / "datasets" / "eval.jsonl"
DEFAULT_STRATEGIC_INPUTS = [
    ROOT / "datasets" / "strategic" / "luna56-pilot-100.jsonl",
    ROOT / "datasets" / "strategic" / "luna56-batch-900.jsonl",
]
DEFAULT_STRATEGIC_MANIFESTS = [
    ROOT / "datasets" / "strategic" / "pilot-100-manifest.jsonl",
    ROOT / "datasets" / "strategic" / "batch-900-manifest.jsonl",
]
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "qwen35-2b-voiceink-v2"
DEFAULT_SYSTEM_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"
LENGTH_BUCKETS = (
    (0, 50, "0-50"),
    (50, 100, "50-100"),
    (100, 200, "100-200"),
    (200, 500, "200-500"),
    (500, math.inf, "500+"),
)
TRANSCRIPT_RE = re.compile(r"<TRANSCRIPT>\s*(.*?)\s*</TRANSCRIPT>", re.DOTALL)
EMPTY_OPENER_RE = re.compile(r"(?:^|\n\s*)(?:yeah|okay|ok|so)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the locked Qwen3.5-2B-VoiceInk-v2 dataset."
    )
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument(
        "--strategic-input", dest="strategic_inputs", type=Path,
        action="append", default=None,
    )
    parser.add_argument(
        "--strategic-manifest", dest="strategic_manifests", type=Path,
        action="append", default=None,
    )
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--holdout-size", type=int, default=100)
    parser.add_argument(
        "--max-input-chars", type=int, default=40000,
        help=(
            "Quarantine new training examples above this system+user character "
            "count; 40k conservatively guards the 16,384-token training limit"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict)
        )
    return ""


def message_for_role(conversation: dict, role: str) -> str:
    for message in conversation.get("messages", []):
        if message.get("role") == role:
            return content_text(message.get("content"))
    return ""


def prepared_transcript(conversation: dict) -> str:
    user = message_for_role(conversation, "user")
    matches = TRANSCRIPT_RE.findall(user)
    return (matches[-1] if matches else user).strip()


def normalize_transcript(text: str) -> str:
    return " ".join(text.casefold().split())


def review_rank(record: dict) -> tuple[int, str]:
    manual = record.get("manual_review") or {}
    if manual.get("status") == "edited":
        return 3, record.get("request_id", "")
    if record.get("manually_reviewed"):
        return 2, record.get("request_id", "")
    if (record.get("auto_review") or {}).get("status") == "approved":
        return 1, record.get("request_id", "")
    return 0, record.get("request_id", "")


def is_training_approved(record: dict) -> bool:
    manual = record.get("manual_review") or {}
    if manual.get("status") == "rejected":
        return False
    return bool(record.get("manually_reviewed")) or (
        (record.get("auto_review") or {}).get("status") == "approved"
    )


def deduplicate_strategic(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep one label per normalized transcript, preferring human review."""
    groups = defaultdict(list)
    for record in records:
        transcript = extract_from_record(record)["transcript"]
        groups[normalize_transcript(transcript)].append(record)

    kept = []
    excluded = []
    for matches in groups.values():
        winner = max(matches, key=review_rank)
        kept.append(winner)
        for record in matches:
            if record is winner:
                continue
            excluded.append({
                "request_id": record["request_id"],
                "kept_request_id": winner["request_id"],
                "reason": "duplicate normalized transcript; kept strongest review provenance",
            })
    kept.sort(key=lambda record: record["request_id"])
    excluded.sort(key=lambda record: record["request_id"])
    return kept, excluded


def proportional_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or total > sum(counts.values()):
        raise ValueError("Invalid proportional allocation total")
    population = sum(counts.values())
    exact = {name: total * count / population for name, count in counts.items()}
    allocated = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(allocated.values())
    order = sorted(
        counts,
        key=lambda name: (exact[name] - allocated[name], counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        allocated[name] += 1
    return allocated


def select_length_stratified_holdout(records: list[dict], strata: dict[str, str],
                                      size: int, seed: int) -> set[str]:
    by_stratum = defaultdict(list)
    for record in records:
        request_id = record["request_id"]
        if request_id not in strata:
            raise ValueError(f"Missing strategic stratum for {request_id}")
        words = len(extract_from_record(record)["transcript"].split())
        by_stratum[strata[request_id]].append((words, request_id))

    allocation = proportional_allocation(
        {name: len(items) for name, items in by_stratum.items()}, size
    )
    selected = set()
    for name, items in sorted(by_stratum.items()):
        count = allocation[name]
        items.sort()
        rng = random.Random(f"{seed}:{name}")
        for index in range(count):
            start = math.floor(index * len(items) / count)
            end = math.floor((index + 1) * len(items) / count)
            bucket = items[start:max(start + 1, end)]
            selected.add(rng.choice(bucket)[1])
    if len(selected) != size:
        raise ValueError(f"Expected {size} holdout IDs, selected {len(selected)}")
    return selected


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0


def length_summary(conversations: list[dict]) -> dict:
    transcript_words = [len(prepared_transcript(record).split()) for record in conversations]
    input_chars = [
        len(message_for_role(record, "system"))
        + len(message_for_role(record, "user"))
        for record in conversations
    ]
    assistant_words = [
        len(message_for_role(record, "assistant").split()) for record in conversations
    ]
    buckets = Counter()
    for words in transcript_words:
        for lower, upper, name in LENGTH_BUCKETS:
            if lower <= words < upper:
                buckets[name] += 1
                break

    def stats(values: list[int]) -> dict:
        return {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0),
        }

    return {
        "count": len(conversations),
        "transcript_words": stats(transcript_words),
        "input_chars": stats(input_chars),
        "assistant_words": stats(assistant_words),
        "transcript_word_buckets": {
            name: {
                "count": buckets[name],
                "percent": round(100 * buckets[name] / len(conversations), 3)
                if conversations else 0,
            }
            for _, _, name in LENGTH_BUCKETS
        },
    }


def filler_summary(conversations: list[dict]) -> dict:
    labels = [message_for_role(record, "assistant") for record in conversations]
    return {
        "count": len(labels),
        "empty_opener_labels": sum(bool(EMPTY_OPENER_RE.search(label)) for label in labels),
        "like_labels": sum(bool(re.search(r"\blike\b", label, re.IGNORECASE)) for label in labels),
        "you_know_labels": sum(bool(re.search(r"\byou know\b", label, re.IGNORECASE)) for label in labels),
    }


def transcript_fingerprints(records: list[dict]) -> set[str]:
    return {normalize_transcript(prepared_transcript(record)) for record in records}


def source_manifest_record(record: dict, stratum: str, destination: str) -> dict:
    components = extract_from_record(record)
    manual = record.get("manual_review") or {}
    review = manual.get("status") or (
        "auto-approved"
        if (record.get("auto_review") or {}).get("status") == "approved"
        else "unknown"
    )
    return {
        "request_id": record["request_id"],
        "timestamp": record.get("timestamp", ""),
        "stratum": stratum,
        "destination": destination,
        "review": review,
        "transcript_words": len(components["transcript"].split()),
    }


def main() -> None:
    args = parse_args()
    strategic_inputs = args.strategic_inputs or DEFAULT_STRATEGIC_INPUTS
    strategic_manifests = args.strategic_manifests or DEFAULT_STRATEGIC_MANIFESTS
    required = [args.base_train, args.base_eval, args.system_prompt,
                *strategic_inputs, *strategic_manifests]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(f"Missing input files: {missing}", file=sys.stderr)
        sys.exit(1)

    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    strategic_raw = [
        record for path in strategic_inputs for record in load_jsonl(path)
    ]
    if len({record["request_id"] for record in strategic_raw}) != len(strategic_raw):
        raise ValueError("Duplicate strategic request IDs")
    if not all(is_training_approved(record) for record in strategic_raw):
        raise ValueError("Every strategic record must be training-approved")

    strata = {}
    for path in strategic_manifests:
        for record in load_jsonl(path):
            request_id = record["request_id"]
            if request_id in strata:
                raise ValueError(f"Duplicate strategic manifest ID: {request_id}")
            strata[request_id] = record["stratum"]
    if set(strata) != {record["request_id"] for record in strategic_raw}:
        raise ValueError("Strategic labels and manifests do not contain identical IDs")

    strategic, duplicates = deduplicate_strategic(strategic_raw)
    holdout_ids = select_length_stratified_holdout(
        strategic, strata, args.holdout_size, args.seed
    )
    strategic_train_source = [
        record for record in strategic if record["request_id"] not in holdout_ids
    ]
    holdout_source = [
        record for record in strategic if record["request_id"] in holdout_ids
    ]

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    converted = {
        record["request_id"]: convert_record(record, system_prompt, "text-blocks")
        for record in strategic
    }
    if any(value is None for value in converted.values()):
        raise ValueError("A strategic record could not be converted")
    strategic_train = [converted[record["request_id"]] for record in strategic_train_source]
    engineering_eval = [converted[record["request_id"]] for record in holdout_source]

    overlength_source = []
    bounded_train_source = []
    bounded_strategic_train = []
    for source, conversation in zip(strategic_train_source, strategic_train):
        input_chars = (
            len(message_for_role(conversation, "system"))
            + len(message_for_role(conversation, "user"))
        )
        if input_chars > args.max_input_chars:
            overlength_source.append((source, input_chars))
        else:
            bounded_train_source.append(source)
            bounded_strategic_train.append(conversation)
    strategic_train_source = bounded_train_source
    strategic_train = bounded_strategic_train

    base_train_fingerprints = transcript_fingerprints(base_train)
    base_eval_fingerprints = transcript_fingerprints(base_eval)
    strategic_train_fingerprints = transcript_fingerprints(strategic_train)
    engineering_fingerprints = transcript_fingerprints(engineering_eval)
    if strategic_train_fingerprints & base_train_fingerprints:
        raise ValueError("New strategic training data overlaps legacy training data")
    if strategic_train_fingerprints & base_eval_fingerprints:
        raise ValueError("New strategic training data overlaps locked regression eval")
    if engineering_fingerprints & (base_train_fingerprints | base_eval_fingerprints):
        raise ValueError("Engineering holdout overlaps a legacy split")
    if engineering_fingerprints & strategic_train_fingerprints:
        raise ValueError("Engineering holdout leaks into strategic training data")

    final_train = base_train + strategic_train
    random.Random(args.seed).shuffle(final_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    regression_eval_path = args.output_dir / "eval-regression-340.jsonl"
    engineering_eval_path = args.output_dir / "eval-engineering-100.jsonl"
    train_manifest_path = args.output_dir / "strategic-train-manifest.jsonl"
    holdout_manifest_path = args.output_dir / "engineering-holdout-manifest.jsonl"
    duplicates_path = args.output_dir / "deduplicated-strategic.jsonl"
    overlength_path = args.output_dir / "overlength-strategic.jsonl"
    report_path = args.output_dir / "dataset-report.json"

    write_jsonl(train_path, final_train)
    shutil.copy2(args.base_eval, regression_eval_path)
    write_jsonl(engineering_eval_path, engineering_eval)
    write_jsonl(train_manifest_path, [
        source_manifest_record(record, strata[record["request_id"]], "train")
        for record in strategic_train_source
    ])
    write_jsonl(holdout_manifest_path, [
        source_manifest_record(record, strata[record["request_id"]], "engineering-eval")
        for record in holdout_source
    ])
    write_jsonl(duplicates_path, duplicates)
    write_jsonl(overlength_path, [
        source_manifest_record(record, strata[record["request_id"]], "overlength")
        | {
            "input_chars": input_chars,
            "reason": (
                f"system+user input exceeds conservative "
                f"{args.max_input_chars}-character training cap"
            ),
        }
        for record, input_chars in overlength_source
    ])

    distributions = {
        "legacy_train": length_summary(base_train),
        "locked_regression_eval": length_summary(base_eval),
        "strategic_unique_all": length_summary(list(converted.values())),
        "strategic_train": length_summary(strategic_train),
        "engineering_holdout": length_summary(engineering_eval),
        "final_train": length_summary(final_train),
    }
    all_strategic_buckets = distributions["strategic_unique_all"]["transcript_word_buckets"]
    holdout_buckets = distributions["engineering_holdout"]["transcript_word_buckets"]
    max_holdout_drift = max(
        abs(all_strategic_buckets[name]["percent"] - holdout_buckets[name]["percent"])
        for _, _, name in LENGTH_BUCKETS
    )
    legacy_overlap = len(base_train_fingerprints & base_eval_fingerprints)
    report = {
        "model_name": "Qwen3.5-2B-VoiceInk-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "counts": {
            "legacy_train": len(base_train),
            "locked_regression_eval": len(base_eval),
            "strategic_labeled": len(strategic_raw),
            "strategic_unique": len(strategic),
            "deduplicated_strategic": len(duplicates),
            "overlength_strategic": len(overlength_source),
            "strategic_train": len(strategic_train),
            "engineering_holdout": len(engineering_eval),
            "final_train": len(final_train),
        },
        "holdout_strata": dict(Counter(
            strata[record["request_id"]] for record in holdout_source
        )),
        "integrity": {
            "locked_eval_byte_identical": (
                sha256_file(args.base_eval) == sha256_file(regression_eval_path)
            ),
            "legacy_train_eval_normalized_overlap": legacy_overlap,
            "new_overlap_with_legacy_train": 0,
            "new_overlap_with_locked_eval": 0,
            "engineering_holdout_train_overlap": 0,
            "max_holdout_bucket_drift_percentage_points": round(max_holdout_drift, 3),
        },
        "filler_labels": {
            "legacy_train": filler_summary(base_train),
            "strategic_train": filler_summary(strategic_train),
            "engineering_holdout": filler_summary(engineering_eval),
        },
        "distributions": distributions,
        "sources": {
            str(path.relative_to(ROOT)): {
                "count": len(load_jsonl(path)) if path.suffix == ".jsonl" else None,
                "sha256": sha256_file(path),
            }
            for path in required
        },
        "outputs": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256_file(path),
            }
            for path in (
                train_path, regression_eval_path, engineering_eval_path,
                train_manifest_path, holdout_manifest_path, duplicates_path,
                overlength_path,
            )
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Qwen3.5-2B-VoiceInk-v2 dataset")
    for name, count in report["counts"].items():
        print(f"  {name}: {count}")
    print(f"  holdout strata: {report['holdout_strata']}")
    print(f"  locked eval byte-identical: {report['integrity']['locked_eval_byte_identical']}")
    print(f"  legacy train/eval overlap retained: {legacy_overlap}")
    print(f"  new leakage: 0")
    print(f"  holdout max length-bucket drift: {max_holdout_drift:.1f}pp")
    print("\nTranscript word distribution")
    headers = ("legacy_train", "strategic_train", "engineering_holdout", "final_train")
    for _, _, bucket in LENGTH_BUCKETS:
        values = "  ".join(
            f"{name}={distributions[name]['transcript_word_buckets'][bucket]['count']} "
            f"({distributions[name]['transcript_word_buckets'][bucket]['percent']:.1f}%)"
            for name in headers
        )
        print(f"  {bucket}: {values}")
    print("\nInput-character percentiles")
    for name in headers:
        print(f"  {name}: {distributions[name]['input_chars']}")
    print("\nFiller-label indicators")
    for name, summary in report["filler_labels"].items():
        print(f"  {name}: {summary}")
    print(f"\nReport: {report_path}")
    print("Preparation complete; no model was loaded and no training was run.")


if __name__ == "__main__":
    main()
