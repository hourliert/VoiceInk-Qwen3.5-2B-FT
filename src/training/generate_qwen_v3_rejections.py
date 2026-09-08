#!/usr/bin/env python3
"""Generate production Qwen V3 outputs for leakage-safe DPO candidates."""
import argparse
import hashlib
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
from eval.evaluate import message_text, query_llama  # noqa: E402
from training.prepare_dataset import convert_record  # noqa: E402

DEFAULT_LABELS = [
    ROOT / "datasets" / "strategic" / "luna56-pilot-100.jsonl",
    ROOT / "datasets" / "strategic" / "luna56-batch-900.jsonl",
]
DEFAULT_MANIFEST = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "strategic-train-manifest.jsonl"
)
DEFAULT_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "qwen35-2b-voiceink-dpo-v3" / "v3-rejections.jsonl"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate current Qwen V3 responses for DPO preference mining"
    )
    parser.add_argument("--labels", type=Path, nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen3.5-2B-VoiceInk-v3")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=8002)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    add_mlflow_args(parser, default_experiment="voiceink-data")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_messages(record: dict, system_prompt: str) -> list[dict]:
    converted = convert_record(record, system_prompt, "text-blocks")
    if converted is None:
        raise ValueError(f"Could not convert request {record.get('request_id')}")
    return [
        {"role": message["role"], "content": message_text(message["content"])}
        for message in converted["messages"][:2]
    ]


def prompt_digest(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv=None) -> None:
    args = parse_args(argv)
    required = [*args.labels, args.train_manifest, args.system_prompt]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(f"Missing inputs: {missing}", file=sys.stderr)
        sys.exit(1)
    if args.temperature < 0 or args.limit < 0:
        print("--temperature and --limit must be non-negative", file=sys.stderr)
        sys.exit(2)

    labels = {}
    for path in args.labels:
        for record in load_jsonl(path):
            request_id = record["request_id"]
            if request_id in labels:
                raise ValueError(f"Duplicate label request ID: {request_id}")
            labels[request_id] = record
    manifest = load_jsonl(args.train_manifest)
    request_ids = [record["request_id"] for record in manifest]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Strategic train manifest contains duplicate IDs")
    missing_labels = set(request_ids) - set(labels)
    if missing_labels:
        raise ValueError(f"Manifest has {len(missing_labels)} IDs without labels")
    if args.limit:
        request_ids = request_ids[:args.limit]

    cached = {}
    if args.output.is_file():
        for record in load_jsonl(args.output):
            request_id = record["request_id"]
            if request_id in cached:
                raise ValueError(f"Duplicate cached rejection ID: {request_id}")
            if record.get("model") != args.model:
                raise ValueError(
                    f"Cached rejection {request_id} came from {record.get('model')}, "
                    f"expected {args.model}"
                )
            cached[request_id] = record

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    pending = [request_id for request_id in request_ids if request_id not in cached]
    print("Qwen V3 DPO rejection generation")
    print(f"  Protected strategic train prompts: {len(request_ids)}")
    print(f"  Cached: {len(request_ids) - len(pending)}; pending: {len(pending)}")
    print(f"  Model: {args.model}; temperature={args.temperature}; seed={args.seed}")
    print(f"  Output: {args.output}")
    if args.check_only:
        print("Preflight complete; no inference was run.")
        return

    tracking = start_mlflow_run(
        args,
        run_name="qwen35-2b-v3-dpo-rejection-generation",
        run_kind="data.preference_generation.qwen",
        params={
            "model": args.model,
            "temperature": args.temperature,
            "inference_seed": args.seed,
            "requested": len(request_ids),
            "cached": len(request_ids) - len(pending),
            "pending": len(pending),
            "output": args.output,
        },
        datasets=[
            *[
                dataset_metadata(f"approved_labels_{index}", path, len(load_jsonl(path)))
                for index, path in enumerate(args.labels, 1)
            ],
            dataset_metadata("strategic_train_manifest", args.train_manifest, len(manifest)),
        ],
        tags={"voiceink.candidate_model": args.model},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    latencies = []
    with args.output.open("a", encoding="utf-8") as stream:
        for position, request_id in enumerate(pending, 1):
            messages = prompt_messages(labels[request_id], system_prompt)
            print(f"  [{position}/{len(pending)}] {request_id[:8]}...", end=" ", flush=True)
            try:
                response, duration_ms = query_llama(
                    messages,
                    args.model,
                    args.llama_host,
                    args.llama_port,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                record = {
                    "request_id": request_id,
                    "model": args.model,
                    "response": response,
                    "duration_ms": round(duration_ms, 3),
                    "prompt_sha256": prompt_digest(messages),
                }
                latencies.append(duration_ms)
                print(f"{duration_ms:.0f}ms")
            except Exception as exc:
                errors += 1
                record = {
                    "request_id": request_id,
                    "model": args.model,
                    "response": "",
                    "duration_ms": 0,
                    "prompt_sha256": prompt_digest(messages),
                    "error": str(exc),
                }
                print(f"ERROR: {exc}")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    tracking.log_metrics({
        "requested": len(request_ids),
        "generated": len(pending) - errors,
        "errors": errors,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    })
    tracking.log_dict(
        {"outputs": [output_reference("private_v3_rejections", args.output, private=True)]},
        "metadata/outputs.json",
    )
    tracking.finish()
    print(f"Complete: {len(pending) - errors} generated, {errors} errors")


if __name__ == "__main__":
    main()
