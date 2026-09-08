#!/usr/bin/env python3
"""Run and verify complete 440-regression plus 150-acceptance evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    start_mlflow_run,
)
from data.manifest import load_manifest, sha256_file, split_path  # noqa: E402


def _latest(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"eval_*{suffix}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise RuntimeError(f"No {suffix} result found in {directory}")
    return matches[-1]


def _critical_errors(details: Path) -> int:
    count = 0
    with details.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            scores = record["candidate_scores"]
            if scores["meaning_preservation"] <= 2 or scores["technical_accuracy"] <= 2:
                count += 1
    return count


def verify_pair(regression_summary: Path, acceptance_summary: Path,
                regression_details: Path, acceptance_details: Path,
                manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    regression = json.loads(regression_summary.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_summary.read_text(encoding="utf-8"))
    failures = []
    if regression.get("n_samples") != 440:
        failures.append(f"historical regression incomplete: {regression.get('n_samples')}/440")
    if acceptance.get("n_samples") != 150:
        failures.append(f"sealed acceptance incomplete: {acceptance.get('n_samples')}/150")
    candidate = acceptance.get("candidate_model")
    if candidate != regression.get("candidate_model"):
        failures.append("candidate model differs between regression and acceptance")
    critical = _critical_errors(acceptance_details)
    gates = {
        "overall_diff_min": -0.5,
        "meaning_diff_min": -0.10,
        "technical_diff_min": -0.10,
        "p50_ratio_max": 1.10,
        "p90_ratio_max": 1.15,
        "critical_errors_max": 0,
    }
    summaries = {"regression": regression, "acceptance": acceptance}
    corpus_observed = {}
    for corpus, summary in summaries.items():
        dimensions = summary.get("per_dimension", {})
        baseline_latency = summary.get("baseline_latency", {})
        candidate_latency = summary.get("candidate_latency", {})
        corpus_observed[corpus] = {
            "overall_diff": summary.get("candidate_avg_score", 0)
            - summary.get("baseline_avg_score", 0),
            "meaning_diff": dimensions.get("meaning_preservation", {}).get(
                "candidate_avg", 0
            )
            - dimensions.get("meaning_preservation", {}).get("baseline_avg", 0),
            "technical_diff": dimensions.get("technical_accuracy", {}).get(
                "candidate_avg", 0
            )
            - dimensions.get("technical_accuracy", {}).get("baseline_avg", 0),
            "p50_ratio": candidate_latency.get("p50", float("inf"))
            / max(baseline_latency.get("p50", 0), 1e-9),
            "p90_ratio": candidate_latency.get("p90", float("inf"))
            / max(baseline_latency.get("p90", 0), 1e-9),
        }
    overall_diff = min(values["overall_diff"] for values in corpus_observed.values())
    meaning_diff = min(values["meaning_diff"] for values in corpus_observed.values())
    technical_diff = min(values["technical_diff"] for values in corpus_observed.values())
    p50_ratio = max(values["p50_ratio"] for values in corpus_observed.values())
    p90_ratio = max(values["p90_ratio"] for values in corpus_observed.values())
    checks = {
        "overall": overall_diff >= gates["overall_diff_min"],
        "meaning": meaning_diff >= gates["meaning_diff_min"],
        "technical": technical_diff >= gates["technical_diff_min"],
        "p50_latency": p50_ratio <= gates["p50_ratio_max"],
        "p90_latency": p90_ratio <= gates["p90_ratio_max"],
        "critical_errors": critical == 0,
        "reproducible": bool(manifest.get("content_sha256")),
    }
    # A statistically significant protected regression in either corpus is fatal.
    for corpus, summary in summaries.items():
        dimensions = summary.get("per_dimension", {})
        for dimension in ("meaning_preservation", "technical_accuracy"):
            values = dimensions.get(dimension, {})
            if (
                values.get("candidate_avg", 0) < values.get("baseline_avg", 0)
                and values.get("p_value") is not None
                and values["p_value"] < 0.05
            ):
                failures.append(
                    f"protected significant regression: {corpus}/{dimension}"
                )
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema": "voiceink.promotion-evidence/v1", "passed": not failures,
        "candidate_model": candidate, "baseline_model": acceptance.get("baseline_model"),
        "dataset_release": manifest["name"], "dataset_release_sha256": manifest["content_sha256"],
        "gates": gates, "checks": checks, "failures": failures,
        "observed": {"overall_diff": overall_diff, "meaning_diff": meaning_diff,
                     "technical_diff": technical_diff, "p50_ratio": p50_ratio,
                     "p90_ratio": p90_ratio, "critical_errors": critical,
                     "corpora": corpus_observed},
        "evidence": {
            "regression_summary": str(regression_summary.resolve()),
            "regression_summary_sha256": sha256_file(regression_summary),
            "regression_details": str(regression_details.resolve()),
            "regression_details_sha256": sha256_file(regression_details),
            "acceptance_summary": str(acceptance_summary.resolve()),
            "acceptance_summary_sha256": sha256_file(acceptance_summary),
            "acceptance_details": str(acceptance_details.resolve()),
            "acceptance_details_sha256": sha256_file(acceptance_details),
        },
    }


def run(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.release_manifest)
    regression_data = Path(manifest["historical_regression"]["resolved_path"])
    acceptance_data = split_path(args.release_manifest, manifest, "acceptance")
    root = args.output_dir / f"{args.candidate}-vs-{args.baseline}"
    regression_dir = root / "regression-440-strict-v2"
    acceptance_dir = root / "acceptance-150-strict-v3"
    tracking = start_mlflow_run(
        args,
        run_name=f"{args.candidate}-vs-{args.baseline}-promotion-gate",
        run_kind="evaluation.promotion-gate",
        params={
            "baseline_model": args.baseline,
            "candidate_model": args.candidate,
            "dataset_release": manifest["name"],
            "dataset_release_sha256": manifest["content_sha256"],
            "judge_model": args.judge_model,
            "judge_reasoning_effort": args.judge_reasoning_effort,
            "parallel": args.parallel,
            "regression_expected": 440,
            "acceptance_expected": 150,
        },
        datasets=[
            dataset_metadata("historical_regression", regression_data, 440),
            dataset_metadata("sealed_acceptance", acceptance_data, 150),
        ],
        tags={
            "voiceink.dataset_release": manifest["name"],
            "voiceink.dataset_release_sha256": manifest["content_sha256"],
            "voiceink.baseline_model": args.baseline,
            "voiceink.candidate_model": args.candidate,
        },
    )
    common = ["--baseline", args.baseline, "--candidate", args.candidate,
              "--judge-provider", "codex", "--judge-model", args.judge_model,
              "--judge-reasoning-effort", args.judge_reasoning_effort,
              "--parallel", str(args.parallel), "--llama-host", args.llama_host,
              "--llama-port", str(args.llama_port)]
    if tracking.run_id:
        common.extend(["--mlflow-parent-run-id", tracking.run_id])
    else:
        common.append("--no-mlflow")
    try:
        subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate.py"), *common,
                        "--eval-data", str(regression_data), "--judge-rubric", "strict-v2",
                        "--output-dir", str(regression_dir)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate.py"), *common,
                        "--eval-data", str(acceptance_data), "--allow-noncanonical-eval",
                        "--judge-rubric", "strict-v3", "--output-dir", str(acceptance_dir)],
                       cwd=ROOT, check=True)
        report = verify_pair(
            _latest(regression_dir, "_summary.json"),
            _latest(acceptance_dir, "_summary.json"),
            _latest(regression_dir, ".jsonl"),
            _latest(acceptance_dir, ".jsonl"),
            args.release_manifest,
        )
        tracking.log_metrics(report["observed"])
        tracking.log_metrics({
            "promotion_passed": int(report["passed"]),
            "promotion_failures": len(report["failures"]),
        })
        tracking.log_dict(report, "promotion/evidence.json")
        tracking.set_tags({
            "voiceink.promotion_gate": "passed" if report["passed"] else "failed"
        })
        tracking.finish()
        return report
    except BaseException:
        tracking.finish("FAILED")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--baseline", required=True)
    run_parser.add_argument("--candidate", required=True)
    run_parser.add_argument("--release-manifest", type=Path, required=True)
    run_parser.add_argument("--judge-model", default="gpt-5.6-luna")
    run_parser.add_argument("--judge-reasoning-effort", default="low")
    run_parser.add_argument("--parallel", type=int, default=3)
    run_parser.add_argument("--llama-host", default="127.0.0.1")
    run_parser.add_argument("--llama-port", type=int, default=8002)
    run_parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "canonical")
    run_parser.add_argument("--report", type=Path, default=ROOT / "results" / "canonical" / "promotion-evidence.json")
    add_mlflow_args(run_parser, default_experiment="voiceink-evaluation")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--regression-summary", type=Path, required=True)
    verify_parser.add_argument("--acceptance-summary", type=Path, required=True)
    verify_parser.add_argument("--regression-details", type=Path, required=True)
    verify_parser.add_argument("--acceptance-details", type=Path, required=True)
    verify_parser.add_argument("--release-manifest", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path, default=ROOT / "results" / "canonical" / "promotion-evidence.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        report = run(args)
    else:
        report = verify_pair(args.regression_summary, args.acceptance_summary,
                             args.regression_details, args.acceptance_details,
                             args.release_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
