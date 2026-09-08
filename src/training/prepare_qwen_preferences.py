#!/usr/bin/env python3
"""Build leakage-safe Qwen DPO pairs from approved strategic labels.

The approved Luna or human-edited cleanup is the chosen completion. The old
production Qwen response is the rejected completion. Only IDs already assigned
to the strategic training split are eligible; the engineering holdout remains
untouched.
"""
import argparse
import datetime
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.prepare_dataset import convert_record, message_content

DEFAULT_LABELS = [
    ROOT / "datasets" / "strategic" / "luna56-pilot-100.jsonl",
    ROOT / "datasets" / "strategic" / "luna56-batch-900.jsonl",
]
DEFAULT_TRAIN_MANIFEST = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v2" / "strategic-train-manifest.jsonl"
)
DEFAULT_HOLDOUT_MANIFEST = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v2" / "engineering-holdout-manifest.jsonl"
)
DEFAULT_SYSTEM_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "qwen35-2b-voiceink-dpo"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qwen VoiceInk DPO pairs")
    parser.add_argument("--labels", type=Path, nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-edit-fraction", type=float, default=0.08)
    parser.add_argument("--max-edit-fraction", type=float, default=0.50)
    parser.add_argument("--min-changed-words", type=int, default=3)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_source(record: dict) -> str | None:
    manual = record.get("manual_review") or {}
    if manual.get("status") in {"approved", "edited"}:
        return f"manual-{manual['status']}"
    automatic = record.get("auto_review") or {}
    if automatic.get("status") == "approved":
        return f"auto:{automatic.get('policy', 'unknown')}"
    return None


def word_edit_stats(rejected: str, chosen: str) -> tuple[int, float, float]:
    old_words = WORD_RE.findall(rejected.lower())
    new_words = WORD_RE.findall(chosen.lower())
    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    changed = sum(
        max(old_end - old_start, new_end - new_start)
        for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes()
        if operation != "equal"
    )
    edit_fraction = changed / max(len(old_words), len(new_words), 1)
    return changed, edit_fraction, matcher.ratio()


def preference_record(record: dict, stratum: str, system_prompt: str,
                      minimum_fraction: float, maximum_fraction: float,
                      minimum_changed: int) -> tuple[dict | None, str]:
    source = review_source(record)
    if source is None:
        return None, "not-approved"
    if record.get("original_model") != "Qwen3.5-2B-VoiceInk":
        return None, "non-qwen-rejection"

    chosen_text = record.get("label", "").strip()
    rejected_text = record.get("original_response", "").strip()
    if not chosen_text or not rejected_text:
        return None, "empty-completion"

    changed, fraction, similarity = word_edit_stats(rejected_text, chosen_text)
    if changed < minimum_changed:
        return None, "too-few-changed-words"
    if fraction < minimum_fraction:
        return None, "edit-fraction-too-small"
    if fraction > maximum_fraction:
        return None, "edit-fraction-too-large"

    conversation = convert_record(record, system_prompt, "text-blocks")
    if conversation is None:
        return None, "conversion-failed"
    prompt = conversation["messages"][:2]
    chosen = [{"role": "assistant", "content": message_content(chosen_text, "text-blocks")}]
    rejected = [{
        "role": "assistant",
        "content": message_content(rejected_text, "text-blocks"),
    }]
    return {
        "request_id": record["request_id"],
        "stratum": stratum,
        "review_source": source,
        "changed_words": changed,
        "edit_fraction": round(fraction, 6),
        "word_similarity": round(similarity, 6),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }, "selected"


def stratified_split(records: list[dict], eval_fraction: float,
                     seed: int) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record)

    rng = random.Random(seed)
    train = []
    evaluation = []
    for stratum in sorted(groups):
        group = groups[stratum]
        rng.shuffle(group)
        eval_count = max(1, round(len(group) * eval_fraction))
        evaluation.extend(group[:eval_count])
        train.extend(group[eval_count:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def main() -> None:
    args = parse_args()
    required = [*args.labels, args.train_manifest, args.holdout_manifest,
                args.system_prompt]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(f"Missing input files: {missing}", file=sys.stderr)
        sys.exit(1)
    if not 0 < args.eval_fraction < 1:
        print("--eval-fraction must be between 0 and 1", file=sys.stderr)
        sys.exit(2)
    if not 0 <= args.min_edit_fraction <= args.max_edit_fraction <= 1:
        print("Edit-fraction bounds must satisfy 0 <= min <= max <= 1", file=sys.stderr)
        sys.exit(2)

    labels = {}
    for path in args.labels:
        for record in load_jsonl(path):
            request_id = record["request_id"]
            if request_id in labels:
                raise ValueError(f"Duplicate label request ID: {request_id}")
            labels[request_id] = record

    train_manifest = load_jsonl(args.train_manifest)
    train_strata = {record["request_id"]: record["stratum"] for record in train_manifest}
    holdout_ids = {record["request_id"] for record in load_jsonl(args.holdout_manifest)}
    if set(train_strata) & holdout_ids:
        raise ValueError("Strategic train manifest overlaps engineering holdout")
    if not set(train_strata) <= set(labels):
        raise ValueError("Strategic train manifest contains IDs without labels")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    selected = []
    exclusions = Counter()
    for request_id, stratum in train_strata.items():
        preference, reason = preference_record(
            labels[request_id], stratum, system_prompt,
            args.min_edit_fraction, args.max_edit_fraction,
            args.min_changed_words,
        )
        exclusions[reason] += 1
        if preference is not None:
            selected.append(preference)

    train, evaluation = stratified_split(selected, args.eval_fraction, args.seed)
    train_ids = {record["request_id"] for record in train}
    eval_ids = {record["request_id"] for record in evaluation}
    if train_ids & eval_ids or (train_ids | eval_ids) & holdout_ids:
        raise ValueError("Preference split leakage detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    report_path = args.output_dir / "dataset-report.json"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, evaluation)

    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "selection": {
            "min_changed_words": args.min_changed_words,
            "min_edit_fraction": args.min_edit_fraction,
            "max_edit_fraction": args.max_edit_fraction,
            "eval_fraction": args.eval_fraction,
        },
        "counts": {
            "approved_strategic_train_candidates": len(train_strata),
            "selected_preferences": len(selected),
            "preference_train": len(train),
            "preference_eval": len(evaluation),
            "protected_engineering_holdout": len(holdout_ids),
        },
        "exclusions": dict(exclusions),
        "selected_strata": dict(Counter(record["stratum"] for record in selected)),
        "train_strata": dict(Counter(record["stratum"] for record in train)),
        "eval_strata": dict(Counter(record["stratum"] for record in evaluation)),
        "integrity": {
            "train_eval_overlap": 0,
            "preference_engineering_holdout_overlap": 0,
        },
        "files": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "eval": {"path": str(eval_path), "sha256": sha256_file(eval_path)},
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Qwen VoiceInk preference dataset")
    print(f"  Eligible strategic train IDs: {len(train_strata)}")
    print(f"  Selected strong preferences: {len(selected)}")
    print(f"  Train: {train_path} ({len(train)})")
    print(f"  Eval: {eval_path} ({len(evaluation)})")
    print(f"  Engineering holdout protected: {len(holdout_ids)}")
    print(f"  Exclusions: {dict(exclusions)}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
