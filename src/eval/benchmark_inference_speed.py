#!/usr/bin/env python3
"""Compare two llama-server endpoints on production-layout VoiceInk requests."""

import argparse
import datetime
import http.client
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluate import (  # noqa: E402
    DEFAULT_EVAL,
    load_eval_data,
    messages_for_layout,
    validate_eval_corpus,
)
from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    output_reference,
    start_mlflow_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark paired production-layout requests on two llama servers."
    )
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-host", default="127.0.0.1")
    parser.add_argument("--candidate-host", default="127.0.0.1")
    parser.add_argument("--baseline-port", type=int, required=True)
    parser.add_argument("--candidate-port", type=int, required=True)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--allow-noncanonical-eval", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    add_mlflow_args(parser, default_experiment="voiceink-evaluation")
    return parser.parse_args()


def query(
    *,
    host: str,
    port: int,
    model: str,
    messages: list[dict],
    temperature: float,
    seed: int,
    max_tokens: int,
) -> dict:
    request = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    payload = json.dumps(request).encode("utf-8")
    started = time.perf_counter()
    connection = http.client.HTTPConnection(host, port, timeout=300)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8")
    finally:
        connection.close()
    wall_ms = (time.perf_counter() - started) * 1000
    parsed = json.loads(body)
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {parsed}")
    choice = parsed.get("choices", [{}])[0]
    message = choice.get("message", {})
    timings = parsed.get("timings", {})
    usage = parsed.get("usage", {})
    return {
        "wall_ms": wall_ms,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_ms": timings.get("prompt_ms", 0),
        "predicted_ms": timings.get("predicted_ms", 0),
        "prompt_tps": timings.get("prompt_per_second", 0),
        "generation_tps": timings.get("predicted_per_second", 0),
        "reasoning_chars": len(message.get("reasoning_content", "") or ""),
        "output_chars": len(message.get("content", "") or ""),
        "finish_reason": choice.get("finish_reason"),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def summarize(records: list[dict]) -> dict:
    walls = [record["wall_ms"] for record in records]
    prompt_ms = sum(record["prompt_ms"] for record in records)
    predicted_ms = sum(record["predicted_ms"] for record in records)
    return {
        "requests": len(records),
        "avg_wall_ms": statistics.fmean(walls),
        "p50_wall_ms": statistics.median(walls),
        "p90_wall_ms": percentile(walls, 0.9),
        "prompt_tps": (
            1000 * sum(record["prompt_tokens"] for record in records) / prompt_ms
            if prompt_ms
            else 0
        ),
        "generation_tps": (
            1000 * sum(record["completion_tokens"] for record in records)
            / predicted_ms
            if predicted_ms
            else 0
        ),
        "avg_completion_tokens": statistics.fmean(
            record["completion_tokens"] for record in records
        ),
        "reasoning_responses": sum(
            record["reasoning_chars"] > 0 for record in records
        ),
    }


def print_summary(name: str, summary: dict) -> None:
    print(
        f"{name:28} "
        f"{summary['avg_wall_ms']:9.1f} "
        f"{summary['p50_wall_ms']:9.1f} "
        f"{summary['p90_wall_ms']:9.1f} "
        f"{summary['prompt_tps']:10.1f} "
        f"{summary['generation_tps']:8.1f} "
        f"{summary['avg_completion_tokens']:8.1f} "
        f"{summary['reasoning_responses']:9d}"
    )


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.warmups < 0 or args.max_tokens < 1:
        raise SystemExit("--samples and --max-tokens must be positive; --warmups >= 0")
    samples = load_eval_data(args.eval_data)
    validate_eval_corpus(
        args.eval_data,
        len(samples),
        allow_noncanonical=args.allow_noncanonical_eval,
    )
    if args.samples > len(samples):
        raise SystemExit(
            f"--samples {args.samples} exceeds corpus size {len(samples)}"
        )
    indices = sorted(random.Random(args.seed).sample(range(len(samples)), args.samples))
    tracking = start_mlflow_run(
        args,
        run_name=f"{args.candidate_model}-vs-{args.baseline_model}-speed",
        run_kind="evaluation.inference_speed",
        params={
            "baseline_model": args.baseline_model,
            "candidate_model": args.candidate_model,
            "baseline_endpoint": f"{args.baseline_host}:{args.baseline_port}",
            "candidate_endpoint": f"{args.candidate_host}:{args.candidate_port}",
            "samples": args.samples,
            "seed": args.seed,
            "warmups": args.warmups,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "sample_indices": indices,
        },
        datasets=[dataset_metadata("evaluation", args.eval_data, len(samples))],
        tags={
            "voiceink.baseline_model": args.baseline_model,
            "voiceink.candidate_model": args.candidate_model,
        },
    )
    endpoints = {
        "baseline": {
            "model": args.baseline_model,
            "host": args.baseline_host,
            "port": args.baseline_port,
        },
        "candidate": {
            "model": args.candidate_model,
            "host": args.candidate_host,
            "port": args.candidate_port,
        },
    }

    warmup_messages = messages_for_layout(samples[indices[0]], "voiceink")
    for label, endpoint in endpoints.items():
        for _ in range(args.warmups):
            query(
                **endpoint,
                messages=warmup_messages,
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=args.max_tokens,
            )
        print(f"Warmed {label}: {endpoint['model']}")

    results = {"baseline": [], "candidate": []}
    for position, index in enumerate(indices):
        order = ("baseline", "candidate") if position % 2 == 0 else (
            "candidate",
            "baseline",
        )
        messages = messages_for_layout(samples[index], "voiceink")
        print(f"[{position + 1:02d}/{len(indices)}] sample {index}: ", end="", flush=True)
        for label in order:
            record = query(
                **endpoints[label],
                messages=messages,
                temperature=args.temperature,
                seed=args.seed + index,
                max_tokens=args.max_tokens,
            )
            record["sample_index"] = index
            results[label].append(record)
            print(f"{label}={record['wall_ms']:.0f}ms ", end="", flush=True)
        print()

    summaries = {label: summarize(records) for label, records in results.items()}
    print("\nModel                          avg ms    p50 ms    p90 ms prompt tok/s gen tok/s avg toks reasoning")
    print_summary(args.baseline_model, summaries["baseline"])
    print_summary(args.candidate_model, summaries["candidate"])
    ratio = (
        summaries["baseline"]["avg_wall_ms"]
        / summaries["candidate"]["avg_wall_ms"]
    )
    print(f"\nCandidate end-to-end speed ratio: {ratio:.2f}x")
    tracking.log_metrics({
        "baseline": summaries["baseline"],
        "candidate": summaries["candidate"],
        "candidate_speed_ratio": ratio,
    })
    model_ids = {}
    for role in ("baseline", "candidate"):
        endpoint = endpoints[role]
        model_ids[role] = tracking.log_external_model(
            name=endpoint["model"],
            model_type="voiceink-transcript-cleanup",
            params={
                "comparison_role": role,
                "endpoint": f"{endpoint['host']}:{endpoint['port']}",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
            tags={"voiceink.comparison_role": role},
            metrics={"samples": args.samples, "latency": summaries[role]},
            dataset_role="evaluation",
        )
    tracking.set_tags({
        f"voiceink.{role}_logged_model_id": model_id
        for role, model_id in model_ids.items() if model_id
    })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "arguments": vars(args)
            | {
                "eval_data": str(args.eval_data),
                "output": str(args.output),
            },
            "sample_indices": indices,
            "summaries": summaries,
            "records": results,
        }
        args.output.write_text(
            json.dumps(artifact, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved metrics: {args.output}")
        tracking.log_artifact(args.output, artifact_path="benchmarks")
        tracking.log_dict(
            {"outputs": [output_reference(
                "speed_benchmark", args.output, private=False
            )]},
            "metadata/outputs.json",
        )
    tracking.finish()


if __name__ == "__main__":
    main()
