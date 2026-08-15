#!/usr/bin/env python3
"""Use the strict-v2 Luna judge to validate V3-specific DPO candidates."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm_cli import (  # noqa: E402
    add_provider_args,
    provider_metadata,
    provider_prompt_path,
    resolve_model,
)
from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    output_reference,
    start_mlflow_run,
)
from eval.evaluate import (  # noqa: E402
    STRICT_V2_JUDGE_PROMPT_PATH,
    judge_one,
    message_text,
)

DEFAULT_DIR = ROOT / "datasets" / "qwen35-2b-voiceink-dpo-v3"
DEFAULT_CANDIDATES = DEFAULT_DIR / "candidates.jsonl"
DEFAULT_OUTPUT = DEFAULT_DIR / "v3-judgments.jsonl"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge whether approved labels materially beat current Qwen V3"
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    add_provider_args(parser, prefix="judge", default_provider="codex")
    add_mlflow_args(parser, default_experiment="voiceink-evaluation")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def completion_text(messages: list[dict]) -> str:
    return message_text(messages[0]["content"]).strip()


def judge_candidate(
    index: int,
    candidate: dict,
    provider: str,
    model: str,
    reasoning_effort: str,
    assignment_seed: int,
    template: str,
    prompt_sha256: str,
) -> dict:
    result = judge_one(
        candidate["judge_sample"],
        completion_text(candidate["rejected"]),
        completion_text(candidate["chosen"]),
        provider,
        model,
        reasoning_effort,
        False,
        assignment_seed,
        index,
        template,
        "strict-v2",
    )
    if result is None:
        return {"request_id": candidate["request_id"], "error": "judge-failed"}
    preference = result.get("pairwise_preference")
    mapped = {
        "baseline": "rejected",
        "candidate": "chosen",
        "tie": "tie",
    }.get(preference, "invalid")
    return {
        "request_id": candidate["request_id"],
        "pairwise_preference": mapped,
        "pairwise_confidence": result.get("pairwise_confidence"),
        "pairwise_material_difference": result.get("pairwise_material_difference"),
        "pairwise_reason": result.get("pairwise_reason"),
        "chosen_scores": result["candidate_scores"],
        "rejected_scores": result["baseline_scores"],
        "chosen_context_analysis": result.get("candidate_context_analysis"),
        "rejected_context_analysis": result.get("baseline_context_analysis"),
        "chosen_score_analysis": result.get("candidate_score_analysis"),
        "rejected_score_analysis": result.get("baseline_score_analysis"),
        "judge": {
            **provider_metadata(provider, model, reasoning_effort),
            "rubric": "strict-v2",
            "prompt_sha256": prompt_sha256,
        },
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.candidates.is_file():
        print(f"Candidates not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)
    if args.judge_provider != "codex":
        print("V3 preference validation requires the Codex strict-v2 judge", file=sys.stderr)
        sys.exit(2)
    if args.parallel <= 0 or args.limit < 0:
        print("--parallel must be positive and --limit non-negative", file=sys.stderr)
        sys.exit(2)

    model = resolve_model(args.judge_provider, args.judge_model)
    prompt_path = provider_prompt_path(STRICT_V2_JUDGE_PROMPT_PATH, args.judge_provider)
    template = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    candidates = load_jsonl(args.candidates)
    if args.limit:
        candidates = candidates[:args.limit]

    latest = {}
    if args.output.is_file():
        for record in load_jsonl(args.output):
            latest[record["request_id"]] = record
    pending = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate["request_id"] not in latest or latest[candidate["request_id"]].get("error")
    ]
    print("Qwen V3 DPO preference judgment")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Cached successful: {len(candidates) - len(pending)}; pending: {len(pending)}")
    print(f"  Judge: {args.judge_provider}/{model} ({args.judge_reasoning_effort})")
    print("  Rubric: strict-v2; chosen/rejected ordering blinded")

    tracking = start_mlflow_run(
        args,
        run_name="qwen35-2b-v3-dpo-preference-judgment",
        run_kind="evaluation.preference_judgment.qwen",
        params={
            "judge_provider": args.judge_provider,
            "judge_model": model,
            "judge_reasoning_effort": args.judge_reasoning_effort,
            "judge_rubric": "strict-v2",
            "judge_prompt_sha256": prompt_sha256,
            "parallel": args.parallel,
            "assignment_seed": args.seed,
            "candidates": len(candidates),
            "cached": len(candidates) - len(pending),
            "pending": len(pending),
        },
        datasets=[dataset_metadata("dpo_candidates", args.candidates, len(candidates))],
        tags={
            "voiceink.baseline_model": "Qwen3.5-2B-VoiceInk-v3",
            "voiceink.candidate_model": "approved-strategic-label",
        },
    )
    tracking.register_prompt(
        name="voiceink-strict-v2-codex-judge",
        template=template,
        sha256=prompt_sha256,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    with args.output.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    judge_candidate,
                    index,
                    candidate,
                    args.judge_provider,
                    model,
                    args.judge_reasoning_effort,
                    args.seed,
                    template,
                    prompt_sha256,
                ): candidate["request_id"]
                for index, candidate in pending
            }
            for position, future in enumerate(as_completed(futures), 1):
                request_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"request_id": request_id, "error": str(exc)}
                completed.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                outcome = record.get("pairwise_preference", "error")
                print(f"  [{position}/{len(pending)}] {request_id[:8]}... {outcome}")

    all_latest = latest | {record["request_id"]: record for record in completed}
    relevant = [all_latest[candidate["request_id"]] for candidate in candidates]
    outcomes = Counter(
        record.get("pairwise_preference", "error") for record in relevant
    )
    confidence = Counter(
        record.get("pairwise_confidence", "error") for record in relevant
    )
    material = sum(record.get("pairwise_material_difference") is True for record in relevant)
    tracking.log_metrics({
        "candidates": len(candidates),
        "judged": len(relevant) - outcomes["error"],
        "errors": outcomes["error"],
        "preference_chosen": outcomes["chosen"],
        "preference_rejected": outcomes["rejected"],
        "preference_tie": outcomes["tie"],
        "confidence_high": confidence["high"],
        "confidence_medium": confidence["medium"],
        "confidence_low": confidence["low"],
        "material_differences": material,
    })
    tracking.log_dict(
        {"outputs": [output_reference("private_preference_judgments", args.output, private=True)]},
        "metadata/outputs.json",
    )
    tracking.finish()
    print(f"Outcomes: {dict(outcomes)}; material={material}")
    print(f"Judgments: {args.output}")


if __name__ == "__main__":
    main()
