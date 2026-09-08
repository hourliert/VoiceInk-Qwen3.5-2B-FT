#!/usr/bin/env python3
"""Backfill completed VoiceInk trainer/evaluation metadata into MLflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    output_reference,
    start_mlflow_run,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill an existing trainer state or eval summary into MLflow"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-output", type=Path)
    source.add_argument("--eval-summary", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--gguf-dir", type=Path)
    add_mlflow_args(parser, default_experiment="voiceink-training")
    return parser.parse_args(argv)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def latest_trainer_state(output_dir: Path) -> tuple[Path, dict]:
    candidates = []
    direct = output_dir / "trainer_state.json"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json under {output_dir}")
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in candidates]
    return max(loaded, key=lambda item: int(item[1].get("global_step", 0)))


def datasets_from_args(args: argparse.Namespace) -> list[dict]:
    datasets = []
    if args.train_data:
        datasets.append(dataset_metadata(
            "train", args.train_data, count_jsonl(args.train_data)
        ))
    if args.eval_data:
        datasets.append(dataset_metadata(
            "eval", args.eval_data, count_jsonl(args.eval_data)
        ))
    return datasets


def backfill_training(args: argparse.Namespace) -> None:
    state_path, state = latest_trainer_state(args.training_output)
    run = start_mlflow_run(
        args,
        run_name=args.run_name,
        run_kind="training.backfill",
        params={
            "base_model": args.base_model,
            "training_output": args.training_output,
            "global_step": state.get("global_step", 0),
            "max_steps": state.get("max_steps", 0),
            "num_train_epochs": state.get("num_train_epochs", 0),
            "best_metric": state.get("best_metric", ""),
            "best_model_checkpoint": state.get("best_model_checkpoint", ""),
            "source_trainer_state": state_path,
        },
        datasets=datasets_from_args(args),
        tags={"voiceink.backfilled": "true"},
    )
    for entry in state.get("log_history", []):
        step = int(entry.get("step", 0))
        run.log_metrics(entry, step=step)
    if state.get("best_metric") is not None:
        run.log_metrics({"best_eval_loss": state["best_metric"]})
    run.log_dict(state, "training/trainer_state.json")
    references = [output_reference(
        "checkpoints", args.training_output, private=False
    )]
    if args.lora_dir:
        references.append(output_reference(
            "lora_adapter", args.lora_dir, private=False
        ))
    if args.gguf_dir:
        references.append(output_reference(
            "gguf_export", args.gguf_dir, private=False
        ))
    run.log_dict({"outputs": references}, "metadata/outputs.json")
    run.log_external_model(
        name=args.run_name,
        model_type="voiceink-transcript-cleanup",
        params={
            "base_model": args.base_model,
            "lora_dir": args.lora_dir or "not-provided",
            "gguf_dir": args.gguf_dir or "not-provided",
            "source_trainer_state": state_path,
        },
        tags={"voiceink.backfilled": "true"},
        metrics={
            "best_eval_loss": state.get("best_metric"),
            "global_step": state.get("global_step", 0),
        },
        dataset_role="eval",
    )
    run.finish()
    print(f"Backfilled {len(state.get('log_history', []))} trainer log entries")


def backfill_evaluation(args: argparse.Namespace) -> None:
    summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))
    if args.mlflow_experiment == "voiceink-training":
        args.mlflow_experiment = "voiceink-evaluation"
    run = start_mlflow_run(
        args,
        run_name=args.run_name,
        run_kind="evaluation.backfill",
        params={
            "baseline_model": summary.get("baseline_model", ""),
            "candidate_model": summary.get("candidate_model", ""),
            "samples": summary.get("n_samples", 0),
            "summary_path": args.eval_summary,
        },
        datasets=datasets_from_args(args),
        tags={
            "voiceink.backfilled": "true",
            "voiceink.winner": summary.get("winner", "unknown"),
        },
    )
    run.log_metrics(summary)
    run.log_artifact(args.eval_summary, artifact_path="summaries")
    run.log_dict(
        {"outputs": [output_reference(
            "evaluation_summary", args.eval_summary, private=False
        )]},
        "metadata/outputs.json",
    )
    run.finish()
    print(f"Backfilled evaluation summary: {args.eval_summary}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.training_output:
        backfill_training(args)
    else:
        backfill_evaluation(args)


if __name__ == "__main__":
    main()
