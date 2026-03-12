#!/usr/bin/env python3
"""Evaluate baseline vs candidate model on transcription cleanup quality.

Runs both models on the eval dataset, has Claude Sonnet 4.6 judge each pair
blindly, and aggregates scores to determine a winner.

Usage:
    python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk
    python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk --limit 3 --dry-run
"""
import argparse
import datetime
import http.client
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_components
DEFAULT_EVAL = ROOT / "datasets" / "eval.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"

WEIGHTS = {
    "meaning_preservation": 3,
    "instruction_following": 3,
    "filler_removal": 2,
    "grammar_fluency": 2,
    "technical_accuracy": 2,
    "conciseness": 1,
}
TOTAL_WEIGHT = sum(WEIGHTS.values())  # 13
MAX_SCORE = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs candidate model.")
    p.add_argument("--baseline", required=True,
                   help="Baseline model alias on llama-server (e.g., Qwen3.5-4B)")
    p.add_argument("--candidate", required=True,
                   help="Candidate model alias on llama-server (e.g., Qwen3.5-2B-VoiceInk)")
    p.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL,
                   help=f"Eval dataset JSONL (default: {DEFAULT_EVAL})")
    p.add_argument("--llama-host", default="127.0.0.1")
    p.add_argument("--llama-port", type=int, default=8002)
    p.add_argument("--judge-model", default="claude-sonnet-4-6")
    p.add_argument("--parallel", type=int, default=3,
                   help="Parallel judge calls")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to evaluate (0 = all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for A/B assignment")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a previous eval JSONL (skips already-judged samples)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print judge prompts without calling Claude")
    return p.parse_args()


# ---- Data loading ----

def load_eval_data(path: Path) -> list[dict]:
    """Load eval.jsonl and extract messages + gold label."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            msgs = record["messages"]

            # Extract plain text from VLM typed content blocks
            system_text = msgs[0]["content"][0]["text"]
            user_text = msgs[1]["content"][0]["text"]
            gold_label = msgs[2]["content"][0]["text"]

            # Extract raw transcript and vocabulary from user message
            m = re.search(r"<TRANSCRIPT>\s*(.*?)\s*</TRANSCRIPT>", user_text, re.DOTALL)
            raw_transcript = m.group(1).strip() if m else user_text.strip()

            v = re.search(r"<CUSTOM_VOCABULARY>\s*(.*?)\s*</CUSTOM_VOCABULARY>", user_text, re.DOTALL)
            custom_vocabulary = v.group(1).strip() if v else ""

            samples.append({
                "system_text": system_text,
                "user_text": user_text,
                "gold_label": gold_label,
                "raw_transcript": raw_transcript,
                "custom_vocabulary": custom_vocabulary,
                "messages_for_llama": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
            })
    return samples


def load_cached_results(path: Path) -> dict[str, dict]:
    """Load previous eval JSONL, keyed by raw_transcript for matching."""
    cache = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("raw_transcript", "")
            if key and "baseline_scores" in record and "candidate_scores" in record:
                cache[key] = record
    return cache


# ---- Model inference ----

def query_llama(messages: list[dict], model: str, host: str, port: int) -> tuple[str, float]:
    """Send a chat completion request. Returns (response_text, duration_ms)."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }).encode("utf-8")

    start = time.monotonic()
    conn = http.client.HTTPConnection(host, port, timeout=300)
    try:
        conn.request("POST", "/v1/chat/completions", payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()
    duration_ms = (time.monotonic() - start) * 1000

    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text, duration_ms


def generate_outputs(samples: list[dict], model: str, host: str, port: int) -> list[dict]:
    """Run a model on all eval samples sequentially."""
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {model}...", end=" ", flush=True)
        try:
            text, duration_ms = query_llama(sample["messages_for_llama"], model, host, port)
            print(f"{duration_ms:.0f}ms")
            results.append({"text": text, "duration_ms": duration_ms})
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"text": "", "duration_ms": 0, "error": str(exc)})
    return results


# ---- Judge ----

def call_claude(prompt: str, model: str) -> str:
    """Call the Claude CLI and return the response text."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", model,
         "--disable-slash-commands", "--allowed-tools", ""],
        capture_output=True, text=True, timeout=600, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def parse_judge_response(raw: str) -> dict | None:
    """Extract JSON from judge response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def judge_one(
    sample: dict,
    baseline_text: str,
    candidate_text: str,
    judge_model: str,
    dry_run: bool,
    seed_offset: int,
) -> dict | None:
    """Judge a single sample. Returns structured result or None on failure."""
    rng = random.Random(42 + seed_offset)
    coin = rng.random() < 0.5

    if coin:
        output_a, output_b = baseline_text, candidate_text
        a_is = "baseline"
    else:
        output_a, output_b = candidate_text, baseline_text
        a_is = "candidate"

    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        raw_transcript=sample["raw_transcript"],
        gold_label=sample["gold_label"],
        output_a=output_a,
        output_b=output_b,
        custom_vocabulary=sample.get("custom_vocabulary", "(none)"),
    )

    if dry_run:
        print(f"  --- DRY RUN ---")
        print(f"  Transcript: {sample['raw_transcript'][:100]}...")
        print(f"  A={a_is}, B={'candidate' if a_is == 'baseline' else 'baseline'}")
        return None

    for attempt in range(2):
        try:
            raw = call_claude(prompt, judge_model)
        except Exception as exc:
            print(f"  JUDGE ERROR: {exc}", file=sys.stderr)
            return None

        parsed = parse_judge_response(raw)
        if parsed and "output_a" in parsed and "output_b" in parsed:
            break
        label = "PARSE ERROR" if attempt == 0 else "PARSE ERROR (retry failed)"
        print(f"  {label}: {raw[:200]}", file=sys.stderr)
        # Log full failed response for debugging
        err_path = ROOT / "results" / "judge_errors.jsonl"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with err_path.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps({"sample_index": seed_offset, "attempt": attempt,
                                 "raw_response": raw}, ensure_ascii=False) + "\n")
    else:
        return None

    # Map A/B back to baseline/candidate
    if a_is == "baseline":
        baseline_scores = parsed["output_a"]
        candidate_scores = parsed["output_b"]
    else:
        baseline_scores = parsed["output_b"]
        candidate_scores = parsed["output_a"]

    return {
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
    }


# ---- Scoring ----

def weighted_score(scores: dict) -> float:
    """Compute weighted score normalized to 0-100."""
    raw = sum(scores.get(dim, 3) * w for dim, w in WEIGHTS.items())
    return round(raw / (MAX_SCORE * TOTAL_WEIGHT) * 100, 1)


def aggregate(judgments: list[dict], baseline_outputs: list[dict],
              candidate_outputs: list[dict], baseline_model: str,
              candidate_model: str) -> dict:
    """Compute aggregate scores and determine winner."""
    wins = {"baseline": 0, "candidate": 0, "tie": 0}
    baseline_weighted = []
    candidate_weighted = []
    per_dim_baseline = {d: [] for d in WEIGHTS}
    per_dim_candidate = {d: [] for d in WEIGHTS}

    for j in judgments:
        b_ws = weighted_score(j["baseline_scores"])
        c_ws = weighted_score(j["candidate_scores"])
        baseline_weighted.append(b_ws)
        candidate_weighted.append(c_ws)

        if b_ws > c_ws:
            wins["baseline"] += 1
        elif c_ws > b_ws:
            wins["candidate"] += 1
        else:
            wins["tie"] += 1

        for dim in WEIGHTS:
            per_dim_baseline[dim].append(j["baseline_scores"].get(dim, 3))
            per_dim_candidate[dim].append(j["candidate_scores"].get(dim, 3))

    n = len(judgments)
    b_avg = round(sum(baseline_weighted) / n, 1)
    c_avg = round(sum(candidate_weighted) / n, 1)

    # Latency
    b_durations = [o["duration_ms"] for o in baseline_outputs if o.get("duration_ms")]
    c_durations = [o["duration_ms"] for o in candidate_outputs if o.get("duration_ms")]
    b_latency = round(sum(b_durations) / len(b_durations), 1) if b_durations else 0
    c_latency = round(sum(c_durations) / len(c_durations), 1) if c_durations else 0

    # Winner determination
    winner, reason = determine_winner(
        b_avg, c_avg,
        {d: sum(v) / len(v) for d, v in per_dim_baseline.items()},
        {d: sum(v) / len(v) for d, v in per_dim_candidate.items()},
    )

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "n_samples": n,
        "baseline_avg_score": b_avg,
        "candidate_avg_score": c_avg,
        "wins": wins,
        "per_dimension": {
            dim: {
                "baseline_avg": round(sum(per_dim_baseline[dim]) / n, 2),
                "candidate_avg": round(sum(per_dim_candidate[dim]) / n, 2),
                "weight": WEIGHTS[dim],
            }
            for dim in WEIGHTS
        },
        "baseline_avg_latency_ms": b_latency,
        "candidate_avg_latency_ms": c_latency,
        "speed_ratio": round(b_latency / c_latency, 2) if c_latency > 0 else 0,
        "winner": winner,
        "winner_reason": reason,
    }


def determine_winner(b_avg, c_avg, b_dims, c_dims):
    """Determine winner with critical dimension checks."""
    for dim in ("meaning_preservation", "instruction_following"):
        if c_dims[dim] < 3.0 <= b_dims[dim]:
            return "baseline", f"Candidate fails critical dimension: {dim}"
        if b_dims[dim] < 3.0 <= c_dims[dim]:
            return "candidate", f"Baseline fails critical dimension: {dim}"

    diff = abs(b_avg - c_avg)
    if diff < 2.0:
        return "tie", f"Scores within margin (baseline={b_avg}, candidate={c_avg})"
    elif c_avg > b_avg:
        return "candidate", f"Candidate wins {c_avg} vs {b_avg} (+{round(c_avg - b_avg, 1)})"
    else:
        return "baseline", f"Baseline wins {b_avg} vs {c_avg} (+{round(b_avg - c_avg, 1)})"


# ---- Output ----

def print_summary(summary: dict) -> None:
    """Print formatted summary table."""
    b_model = summary["baseline_model"]
    c_model = summary["candidate_model"]
    n = summary["n_samples"]

    print()
    print("=" * 64)
    print(f"  EVALUATION: {b_model} vs {c_model} ({n} samples)")
    print("=" * 64)
    print()
    print("  Per-dimension scores (1-5 scale):")
    print(f"  {'':30s} {'Baseline':>8s}  {'Candidate':>9s}  {'Weight':>6s}")
    for dim in WEIGHTS:
        info = summary["per_dimension"][dim]
        b_val = info["baseline_avg"]
        c_val = info["candidate_avg"]
        flag = " ***" if abs(b_val - c_val) >= 0.5 else ""
        print(f"  {dim:30s} {b_val:8.2f}  {c_val:9.2f}  {info['weight']:5d}x{flag}")

    print()
    print("  Overall weighted score (0-100):")
    print(f"    Baseline:    {summary['baseline_avg_score']}")
    print(f"    Candidate:   {summary['candidate_avg_score']}")

    w = summary["wins"]
    print()
    print(f"  Win/Loss/Tie:  {w['baseline']} / {w['candidate']} / {w['tie']}"
          f"  (baseline / candidate / tie)")

    b_lat = summary["baseline_avg_latency_ms"]
    c_lat = summary["candidate_avg_latency_ms"]
    if b_lat and c_lat:
        print()
        print(f"  Latency:")
        print(f"    Baseline avg:    {b_lat:.0f} ms")
        ratio = summary["speed_ratio"]
        faster = f"({ratio:.1f}x faster)" if ratio > 1 else ""
        print(f"    Candidate avg:   {c_lat:.0f} ms  {faster}")

    print()
    winner = summary["winner"].upper()
    reason = summary["winner_reason"]
    print(f"  WINNER: {winner}")
    print(f"  Reason: {reason}")
    print("=" * 64)
    print()


def write_results(output_dir: Path, samples: list[dict], baseline_outputs: list[dict],
                  candidate_outputs: list[dict], judgments: list[dict],
                  summary: dict) -> None:
    """Write per-sample results and summary to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-sample JSONL
    detail_path = output_dir / f"eval_{ts}.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for i, (sample, b_out, c_out, judgment) in enumerate(
            zip(samples, baseline_outputs, candidate_outputs, judgments)
        ):
            record = {
                "sample_index": i,
                "raw_transcript": sample["raw_transcript"],
                "gold_label": sample["gold_label"],
                "baseline_output": b_out["text"],
                "candidate_output": c_out["text"],
                "baseline_duration_ms": b_out["duration_ms"],
                "candidate_duration_ms": c_out["duration_ms"],
                "baseline_scores": judgment["baseline_scores"],
                "candidate_scores": judgment["candidate_scores"],
                "baseline_weighted": weighted_score(judgment["baseline_scores"]),
                "candidate_weighted": weighted_score(judgment["candidate_scores"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Details: {detail_path}")

    # Summary JSON
    summary_path = output_dir / f"eval_{ts}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


# ---- Main ----

def main() -> None:
    args = parse_args()

    if not args.eval_data.exists():
        print(f"Eval data not found: {args.eval_data}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)

    # Load eval samples
    samples = load_eval_data(args.eval_data)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} eval samples")

    # Load cache from previous run if resuming
    cache = {}
    if args.resume:
        if args.resume.exists():
            cache = load_cached_results(args.resume)
            print(f"Loaded {len(cache)} cached results from {args.resume}")
        else:
            print(f"WARNING: resume file not found: {args.resume}")

    # Split samples into cached vs uncached
    cached_indices = []
    uncached_indices = []
    for i, sample in enumerate(samples):
        if sample["raw_transcript"] in cache:
            cached_indices.append(i)
        else:
            uncached_indices.append(i)

    if cache:
        print(f"  {len(cached_indices)} cached, {len(uncached_indices)} to evaluate")

    # Build result arrays — pre-fill from cache
    baseline_outputs = [None] * len(samples)
    candidate_outputs = [None] * len(samples)
    judge_results = {}

    for i in cached_indices:
        c = cache[samples[i]["raw_transcript"]]
        baseline_outputs[i] = {
            "text": c["baseline_output"],
            "duration_ms": c.get("baseline_duration_ms", 0),
        }
        candidate_outputs[i] = {
            "text": c["candidate_output"],
            "duration_ms": c.get("candidate_duration_ms", 0),
        }
        judge_results[i] = {
            "baseline_scores": c["baseline_scores"],
            "candidate_scores": c["candidate_scores"],
        }

    # Run inference + judging only for uncached samples
    if uncached_indices:
        uncached_samples = [samples[i] for i in uncached_indices]

        # Warm up + generate for each model
        warmup_msgs = [{"role": "user", "content": "Hello"}]

        print(f"Warming up {args.baseline}...", end=" ", flush=True)
        try:
            _, ms = query_llama(warmup_msgs, args.baseline, args.llama_host, args.llama_port)
            print(f"{ms:.0f}ms (discarded)")
        except Exception as exc:
            print(f"WARNING: warmup failed: {exc}")

        print(f"\nGenerating baseline outputs ({args.baseline})...")
        b_outs = generate_outputs(uncached_samples, args.baseline, args.llama_host, args.llama_port)

        print(f"\nWarming up {args.candidate}...", end=" ", flush=True)
        try:
            _, ms = query_llama(warmup_msgs, args.candidate, args.llama_host, args.llama_port)
            print(f"{ms:.0f}ms (discarded)")
        except Exception as exc:
            print(f"WARNING: warmup failed: {exc}")

        print(f"\nGenerating candidate outputs ({args.candidate})...")
        c_outs = generate_outputs(uncached_samples, args.candidate, args.llama_host, args.llama_port)

        for j, i in enumerate(uncached_indices):
            baseline_outputs[i] = b_outs[j]
            candidate_outputs[i] = c_outs[j]

        # Judge uncached pairs
        print(f"\nJudging {len(uncached_indices)} new outputs ({args.judge_model})...")
        errors = 0

        if args.parallel <= 1 or args.dry_run:
            for j, i in enumerate(uncached_indices):
                print(f"  [{j+1}/{len(uncached_indices)}]", end=" ")
                result = judge_one(samples[i], baseline_outputs[i]["text"],
                                  candidate_outputs[i]["text"],
                                  args.judge_model, args.dry_run, i)
                judge_results[i] = result
                if not result:
                    errors += 1
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {}
                for j, i in enumerate(uncached_indices):
                    fut = pool.submit(judge_one, samples[i],
                                      baseline_outputs[i]["text"],
                                      candidate_outputs[i]["text"],
                                      args.judge_model, False, i)
                    futures[fut] = i

                done_count = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    done_count += 1
                    print(f"  [{done_count}/{len(uncached_indices)}] Judged sample {i}")
                    try:
                        result = fut.result()
                    except Exception as exc:
                        print(f"  ERROR: {exc}", file=sys.stderr)
                        result = None
                    judge_results[i] = result
                    if not result:
                        errors += 1
    else:
        errors = 0
        print("\nAll samples cached, skipping inference and judging.")

    if args.dry_run:
        print("\nDry run complete.")
        return

    # Filter to only samples with successful judgments (keeps indices aligned)
    good_indices = sorted(i for i, j in judge_results.items() if j is not None)

    if not good_indices:
        print("No successful judgments.", file=sys.stderr)
        sys.exit(1)

    if errors:
        print(f"\n{errors} samples failed (excluded from results)")

    filtered_samples = [samples[i] for i in good_indices]
    filtered_baseline = [baseline_outputs[i] for i in good_indices]
    filtered_candidate = [candidate_outputs[i] for i in good_indices]
    filtered_judgments = [judge_results[i] for i in good_indices]

    # Aggregate and print
    summary = aggregate(filtered_judgments, filtered_baseline, filtered_candidate,
                        args.baseline, args.candidate)
    print_summary(summary)

    # Write results
    write_results(args.output_dir, filtered_samples, filtered_baseline,
                  filtered_candidate, filtered_judgments, summary)


if __name__ == "__main__":
    main()
