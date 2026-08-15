#!/usr/bin/env python3
"""Build and finalize leakage-safe Qwen V3 DPO preferences."""
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

from common.extract import extract_from_record  # noqa: E402
from common.llm_cli import SCORE_DIMENSIONS  # noqa: E402
from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    output_reference,
    start_mlflow_run,
)
from eval.evaluate import WEIGHTS, load_eval_data, message_text  # noqa: E402
from training.prepare_dataset import convert_record, message_content  # noqa: E402
from training.prepare_qwen_v2 import normalize_transcript  # noqa: E402

DEFAULT_LABELS = [
    ROOT / "datasets" / "strategic" / "luna56-pilot-100.jsonl",
    ROOT / "datasets" / "strategic" / "luna56-batch-900.jsonl",
]
DEFAULT_TRAIN_MANIFEST = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "strategic-train-manifest.jsonl"
)
DEFAULT_HOLDOUT_MANIFEST = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "engineering-holdout-manifest.jsonl"
)
DEFAULT_CANONICAL_EVAL = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "eval-all-440.jsonl"
)
DEFAULT_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "qwen35-2b-voiceink-dpo-v3"
DEFAULT_REJECTIONS = DEFAULT_OUTPUT_DIR / "v3-rejections.jsonl"
DEFAULT_REJECTION_MODEL = "Qwen3.5-2B-VoiceInk-v3"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qwen V3 DPO pairs")
    parser.add_argument("--labels", type=Path, nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--canonical-eval", type=Path, default=DEFAULT_CANONICAL_EVAL)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--expected-rejection-model", default=DEFAULT_REJECTION_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-changed-words", type=int, default=1)
    parser.add_argument("--max-edit-fraction", type=float, default=0.50)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3407)
    add_mlflow_args(parser, default_experiment="voiceink-training")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def review_source(record: dict) -> str | None:
    manual = record.get("manual_review") or {}
    if manual.get("status") in {"approved", "edited"}:
        return f"manual-{manual['status']}"
    automatic = record.get("auto_review") or {}
    if automatic.get("status") == "approved":
        return f"auto:{automatic.get('policy', 'unknown')}"
    return None


def word_edit_stats(rejected: str, chosen: str) -> tuple[int, float, float]:
    old_words = WORD_RE.findall(rejected.casefold())
    new_words = WORD_RE.findall(chosen.casefold())
    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    changed = sum(
        max(old_end - old_start, new_end - new_start)
        for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes()
        if operation != "equal"
    )
    return (
        changed,
        changed / max(len(old_words), len(new_words), 1),
        matcher.ratio(),
    )


def prompt_digest(prompt: list[dict]) -> str:
    normalized = [
        {"role": message["role"], "content": message_text(message["content"])}
        for message in prompt
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_record(
    label: dict,
    rejection: dict | None,
    stratum: str,
    system_prompt: str,
    expected_model: str,
    min_changed_words: int,
    max_edit_fraction: float,
) -> tuple[dict | None, str]:
    source = review_source(label)
    if source is None:
        return None, "not-approved"
    if rejection is None:
        return None, "missing-v3-rejection"
    if rejection.get("model") != expected_model:
        return None, "wrong-rejection-model"
    if rejection.get("error") or not str(rejection.get("response", "")).strip():
        return None, "failed-v3-rejection"

    chosen_text = str(label.get("label", "")).strip()
    rejected_text = str(rejection["response"]).strip()
    if not chosen_text:
        return None, "empty-chosen"
    changed, fraction, similarity = word_edit_stats(rejected_text, chosen_text)
    if changed < min_changed_words:
        return None, "equivalent-or-too-small"
    if fraction > max_edit_fraction:
        return None, "edit-fraction-too-large"

    conversation = convert_record(label, system_prompt, "text-blocks")
    if conversation is None:
        return None, "conversion-failed"
    prompt = conversation["messages"][:2]
    if rejection.get("prompt_sha256") != prompt_digest(prompt):
        return None, "prompt-fingerprint-mismatch"

    context = extract_from_record(label)
    return {
        "request_id": label["request_id"],
        "stratum": stratum,
        "review_source": source,
        "rejection_model": expected_model,
        "changed_words": changed,
        "edit_fraction": round(fraction, 6),
        "word_similarity": round(similarity, 6),
        "prompt": prompt,
        "chosen": [{
            "role": "assistant",
            "content": message_content(chosen_text, "text-blocks"),
        }],
        "rejected": [{
            "role": "assistant",
            "content": message_content(rejected_text, "text-blocks"),
        }],
        "judge_sample": {
            "raw_transcript": context["transcript"],
            "gold_label": chosen_text,
            "custom_vocabulary": context["custom_vocabulary"],
            "window_context": context["window_context"],
            "clipboard_context": context["clipboard_context"],
        },
    }, "candidate"


def weighted_score(scores: dict) -> int:
    return sum(int(scores[dimension]) * WEIGHTS[dimension] for dimension in WEIGHTS)


def accept_judgment(judgment: dict | None) -> tuple[bool, str]:
    if judgment is None:
        return False, "missing-judgment"
    if judgment.get("error"):
        return False, "judge-error"
    if judgment.get("pairwise_preference") != "chosen":
        return False, f"judge-{judgment.get('pairwise_preference', 'invalid')}"
    if judgment.get("pairwise_confidence") not in {"medium", "high"}:
        return False, "low-confidence"
    if judgment.get("pairwise_material_difference") is not True:
        return False, "non-material"
    chosen = judgment.get("chosen_scores") or {}
    rejected = judgment.get("rejected_scores") or {}
    if set(chosen) != set(SCORE_DIMENSIONS) or set(rejected) != set(SCORE_DIMENSIONS):
        return False, "invalid-scores"
    protected = ("meaning_preservation", "technical_accuracy", "instruction_following")
    if any(chosen[name] < rejected[name] for name in protected):
        return False, "protected-dimension-regression"
    if not any(
        chosen[name] > rejected[name]
        for name in ("meaning_preservation", "technical_accuracy")
    ):
        return False, "no-semantic-or-technical-gain"
    if weighted_score(chosen) <= weighted_score(rejected):
        return False, "no-weighted-gain"
    return True, "selected"


def stratified_split(records: list[dict], eval_fraction: float, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record)
    rng = random.Random(seed)
    train, evaluation = [], []
    for stratum in sorted(groups):
        group = groups[stratum]
        rng.shuffle(group)
        eval_count = max(1, round(len(group) * eval_fraction))
        evaluation.extend(group[:eval_count])
        train.extend(group[eval_count:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def finalized_record(candidate: dict, judgment: dict) -> dict:
    return {
        key: value for key, value in candidate.items() if key != "judge_sample"
    } | {
        "preference_confidence": judgment["pairwise_confidence"],
        "preference_reason": judgment["pairwise_reason"],
        "chosen_scores": judgment["chosen_scores"],
        "rejected_scores": judgment["rejected_scores"],
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    required = [
        *args.labels,
        args.train_manifest,
        args.holdout_manifest,
        args.canonical_eval,
        args.system_prompt,
        args.rejections,
    ]
    if args.judgments:
        required.append(args.judgments)
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(f"Missing inputs: {missing}", file=sys.stderr)
        sys.exit(1)
    if not 0 < args.eval_fraction < 1 or args.min_changed_words < 1:
        print("Invalid split fraction or changed-word threshold", file=sys.stderr)
        sys.exit(2)

    labels = {}
    for path in args.labels:
        for record in load_jsonl(path):
            request_id = record["request_id"]
            if request_id in labels:
                raise ValueError(f"Duplicate label request ID: {request_id}")
            labels[request_id] = record
    manifest = load_jsonl(args.train_manifest)
    strata = {record["request_id"]: record["stratum"] for record in manifest}
    holdout_ids = {record["request_id"] for record in load_jsonl(args.holdout_manifest)}
    if set(strata) & holdout_ids:
        raise ValueError("Strategic train manifest overlaps engineering holdout")
    canonical_transcripts = {
        normalize_transcript(sample["raw_transcript"])
        for sample in load_eval_data(args.canonical_eval)
    }
    training_transcripts = {
        normalize_transcript(extract_from_record(labels[request_id])["transcript"])
        for request_id in strata
    }
    overlap = training_transcripts & canonical_transcripts
    if overlap:
        raise ValueError(
            f"Strategic DPO source overlaps {len(overlap)} canonical-440 transcripts"
        )

    rejections = {record["request_id"]: record for record in load_jsonl(args.rejections)}
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    candidates, exclusions = [], Counter()
    for request_id, stratum in strata.items():
        candidate, reason = candidate_record(
            labels[request_id],
            rejections.get(request_id),
            stratum,
            system_prompt,
            args.expected_rejection_model,
            args.min_changed_words,
            args.max_edit_fraction,
        )
        exclusions[reason] += 1
        if candidate:
            candidates.append(candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    train, evaluation, judgment_exclusions = [], [], Counter()
    if args.judgments:
        judgments = {
            record["request_id"]: record for record in load_jsonl(args.judgments)
        }
        selected = []
        for candidate in candidates:
            judgment = judgments.get(candidate["request_id"])
            accepted, reason = accept_judgment(judgment)
            judgment_exclusions[reason] += 1
            if accepted:
                selected.append(finalized_record(candidate, judgment))
        if not selected:
            raise ValueError("No strong V3-specific DPO preferences passed filtering")
        train, evaluation = stratified_split(selected, args.eval_fraction, args.seed)
        write_jsonl(args.output_dir / "train.jsonl", train)
        write_jsonl(args.output_dir / "eval.jsonl", evaluation)

    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "final" if args.judgments else "candidates",
        "rejection_model": args.expected_rejection_model,
        "selection_policy": {
            "pairwise_preference": "chosen",
            "confidence": ["medium", "high"],
            "material_difference": True,
            "protected_dimensions_nonregressing": [
                "meaning_preservation", "technical_accuracy", "instruction_following"
            ],
            "required_gain": ["meaning_preservation", "technical_accuracy"],
        },
        "counts": {
            "strategic_train_prompts": len(strata),
            "canonical_eval_protected": len(canonical_transcripts),
            "candidate_preferences": len(candidates),
            "preference_train": len(train),
            "preference_eval": len(evaluation),
        },
        "candidate_exclusions": dict(exclusions),
        "judgment_outcomes": dict(judgment_exclusions),
        "integrity": {
            "strategic_holdout_id_overlap": 0,
            "canonical_440_normalized_transcript_overlap": 0,
            "train_eval_id_overlap": len(
                {record["request_id"] for record in train}
                & {record["request_id"] for record in evaluation}
            ),
        },
        "files": {
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            **({
                "train": {
                    "path": str(args.output_dir / "train.jsonl"),
                    "sha256": sha256_file(args.output_dir / "train.jsonl"),
                },
                "eval": {
                    "path": str(args.output_dir / "eval.jsonl"),
                    "sha256": sha256_file(args.output_dir / "eval.jsonl"),
                },
            } if args.judgments else {}),
        },
    }
    report_path = args.output_dir / "dataset-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tracking = start_mlflow_run(
        args,
        run_name=f"qwen35-2b-v3-dpo-{report['stage']}-preparation",
        run_kind="data.preference_preparation.qwen",
        params={
            "stage": report["stage"],
            "rejection_model": args.expected_rejection_model,
            "min_changed_words": args.min_changed_words,
            "max_edit_fraction": args.max_edit_fraction,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
        },
        datasets=[
            dataset_metadata("v3_rejections", args.rejections, len(rejections)),
            dataset_metadata("canonical_440_guard", args.canonical_eval, 440),
            *(
                [dataset_metadata("preference_judgments", args.judgments, len(load_jsonl(args.judgments)))]
                if args.judgments else []
            ),
        ],
    )
    tracking.log_metrics(report["counts"])
    tracking.log_dict(
        {"outputs": [
            output_reference("private_candidates", candidates_path, private=True),
            output_reference("dataset_report", report_path, private=False),
            *(
                [
                    output_reference("private_preference_train", args.output_dir / "train.jsonl", private=True),
                    output_reference("private_preference_eval", args.output_dir / "eval.jsonl", private=True),
                ] if args.judgments else []
            ),
        ]},
        "metadata/outputs.json",
    )
    tracking.finish()

    print("Qwen V3 DPO preference preparation")
    print(f"  Stage: {report['stage']}")
    print(f"  Strategic prompts: {len(strata)}; canonical 440 overlap: 0")
    print(f"  Candidate pairs: {len(candidates)}")
    if args.judgments:
        print(f"  Strong selected pairs: {len(train) + len(evaluation)}")
        print(f"  Train/eval: {len(train)}/{len(evaluation)}")
        print(f"  Judgment outcomes: {dict(judgment_exclusions)}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
