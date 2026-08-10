#!/usr/bin/env python3
"""Evaluate baseline vs candidate model on transcription cleanup quality.

Runs both models on the eval dataset, has a configured LLM judge each pair
blindly, and aggregates scores to determine a winner. Generation artifacts can
be saved and judged separately so inference never needs to be repeated.

Usage:
    python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk
    python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk --limit 3 --dry-run
"""
import argparse
import datetime
import hashlib
import http.client
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.llm_cli import (
    CODEX_EVAL_JUDGE_SCHEMA,
    CODEX_EVAL_JUDGE_STRICT_V2_SCHEMA,
    SCORE_DIMENSIONS,
    add_provider_args,
    call_llm,
    parse_json_response,
    provider_metadata,
    provider_prompt_path,
    resolve_model,
)
from training.prepare_dataset import (
    build_user_message,
    build_voiceink_system_message,
    build_voiceink_user_message,
)
CANONICAL_EVAL = (
    ROOT / "datasets" / "qwen35-2b-voiceink-v3" / "eval-all-440.jsonl"
)
CANONICAL_EVAL_COUNT = 440
CANONICAL_EVAL_SHA256 = (
    "0f08d1eb8788c716f265f2ec90d6495f7901d2dfe91d605dba130ef04db980f5"
)
DEFAULT_EVAL = CANONICAL_EVAL
DEFAULT_OUTPUT_DIR = ROOT / "results"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"
STRICT_V2_JUDGE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "judge_prompt_strict_v2.txt"
)

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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs candidate model.")
    p.add_argument("--baseline", required=True,
                   help="Baseline model alias on llama-server (e.g., Qwen3.5-4B)")
    p.add_argument("--candidate", required=True,
                   help="Candidate model alias on llama-server (e.g., Qwen3.5-2B-VoiceInk)")
    p.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL,
                   help=f"Eval dataset JSONL (default: canonical 440 at {DEFAULT_EVAL})")
    p.add_argument(
        "--allow-noncanonical-eval",
        action="store_true",
        help="Explicitly allow a full evaluation corpus other than canonical V3 440",
    )
    p.add_argument("--llama-host", default="127.0.0.1")
    p.add_argument("--llama-port", type=int, default=8002)
    p.add_argument(
        "--baseline-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for baseline inference (default: 0.3)",
    )
    p.add_argument(
        "--candidate-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for candidate inference (default: 0.3)",
    )
    p.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Optional shared llama.cpp sampling seed for reproducible paired inference",
    )
    p.add_argument(
        "--baseline-message-layout",
        choices=("prepared", "voiceink"),
        default="voiceink",
        help="Message layout for baseline inference (default: voiceink production layout)",
    )
    p.add_argument(
        "--candidate-message-layout",
        choices=("prepared", "voiceink"),
        default="voiceink",
        help="Message layout for candidate inference (default: voiceink production layout)",
    )
    add_provider_args(p, prefix="judge")
    p.add_argument(
        "--judge-rubric",
        choices=("legacy", "strict-v2"),
        default="legacy",
        help="Judge prompt/schema version (default: legacy)",
    )
    p.add_argument("--parallel", type=int, default=3,
                   help="Parallel judge calls")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to evaluate (0 = all)")
    p.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated zero-based eval indices; cannot be combined with --limit",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for A/B assignment")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a previous eval JSONL (skips already-judged samples)")
    p.add_argument("--outputs", type=Path, default=None,
                   help="Judge saved generation/eval JSONL without running local inference")
    p.add_argument("--generation-output", type=Path, default=None,
                   help="Path for generated model outputs (default: timestamped in output dir)")
    p.add_argument("--generate-only", action="store_true",
                   help="Generate and save model outputs, then exit before judging")
    p.add_argument("--dry-run", action="store_true",
                   help="Print judge prompts without calling the configured provider")
    return p.parse_args(argv)


# ---- Data loading ----

def message_text(content) -> str:
    """Extract text from either text-only or VLM message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    raise ValueError(f"Unsupported message content type: {type(content).__name__}")


def extract_last_tag_content(text: str, tag: str) -> str | None:
    """Extract the last complete tag pair, ignoring examples in earlier context."""
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    close_index = text.rfind(closing)
    if close_index < 0:
        return None
    open_index = text.rfind(opening, 0, close_index)
    if open_index < 0:
        return None
    return text[open_index + len(opening):close_index].strip()


def unwrap_voiceink_system(text: str) -> tuple[str, str]:
    """Return canonical instructions and dynamic context from a system message."""
    stripped = text.strip()
    opening = "<SYSTEM_INSTRUCTIONS>"
    closing = "</SYSTEM_INSTRUCTIONS>"
    if not stripped.startswith(opening):
        return stripped, ""
    close_index = stripped.find(closing)
    if close_index < len(opening):
        raise ValueError("Unclosed SYSTEM_INSTRUCTIONS wrapper in eval data")
    instructions = stripped[len(opening):close_index].strip()
    return instructions, stripped[close_index + len(closing):]


def load_eval_data(path: Path) -> list[dict]:
    """Load eval.jsonl and extract messages + gold label."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for sample_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            msgs = record["messages"]

            stored_system_text = message_text(msgs[0]["content"])
            user_text = message_text(msgs[1]["content"])
            gold_label = message_text(msgs[2]["content"])

            system_text, context_text = unwrap_voiceink_system(stored_system_text)
            stripped_user = user_text.strip()
            stored_message_layout = (
                "voiceink"
                if (
                    stored_system_text.strip().startswith("<SYSTEM_INSTRUCTIONS>")
                    and stripped_user.startswith("<TRANSCRIPT>")
                    and stripped_user.endswith("</TRANSCRIPT>")
                )
                else "prepared"
            )

            # Extract raw transcript and vocabulary from user message
            if (
                stripped_user.startswith("<TRANSCRIPT>")
                and stripped_user.endswith("</TRANSCRIPT>")
            ):
                transcript = stripped_user[
                    len("<TRANSCRIPT>"):-len("</TRANSCRIPT>")
                ].strip()
            else:
                transcript = extract_last_tag_content(user_text, "TRANSCRIPT")
            raw_transcript = transcript if transcript is not None else user_text.strip()

            vocabulary = (
                extract_last_tag_content(user_text, "CUSTOM_VOCABULARY")
                or extract_last_tag_content(context_text, "CUSTOM_VOCABULARY")
            )
            custom_vocabulary = vocabulary if vocabulary is not None else ""

            window = (
                extract_last_tag_content(user_text, "CURRENT_WINDOW_CONTEXT")
                or extract_last_tag_content(context_text, "CURRENT_WINDOW_CONTEXT")
            )
            window_context = window if window is not None else ""

            clipboard = (
                extract_last_tag_content(user_text, "CLIPBOARD_CONTEXT")
                or extract_last_tag_content(context_text, "CLIPBOARD_CONTEXT")
            )
            clipboard_context = clipboard if clipboard is not None else ""

            samples.append({
                "sample_index": sample_index,
                "system_text": system_text,
                "user_text": user_text,
                "gold_label": gold_label,
                "raw_transcript": raw_transcript,
                "custom_vocabulary": custom_vocabulary,
                "window_context": window_context,
                "clipboard_context": clipboard_context,
                "stored_system_text": stored_system_text,
                "stored_user_text": user_text,
                "stored_message_layout": stored_message_layout,
            })
    return samples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_eval_corpus(
    path: Path, sample_count: int, allow_noncanonical: bool = False
) -> None:
    """Fail closed unless a full benchmark uses the locked V3 440 corpus."""
    if allow_noncanonical:
        return
    if path.resolve() != CANONICAL_EVAL.resolve():
        raise ValueError(
            f"Full evaluations require canonical {CANONICAL_EVAL_COUNT}-sample "
            f"corpus {CANONICAL_EVAL}; pass --allow-noncanonical-eval to override"
        )
    if sample_count != CANONICAL_EVAL_COUNT:
        raise ValueError(
            f"Canonical evaluation must contain {CANONICAL_EVAL_COUNT} samples; "
            f"found {sample_count}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != CANONICAL_EVAL_SHA256:
        raise ValueError(
            "Canonical evaluation fingerprint mismatch: "
            f"expected {CANONICAL_EVAL_SHA256}, found {actual_hash}"
        )


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


def select_sample_indices(samples: list[dict], specification: str) -> list[dict]:
    """Select unique zero-based samples in the caller-provided order."""
    try:
        indices = [int(value.strip()) for value in specification.split(",")]
    except ValueError as exc:
        raise ValueError("--sample-indices must contain only integers") from exc
    if not indices or any(not value.strip() for value in specification.split(",")):
        raise ValueError("--sample-indices cannot contain empty values")
    if len(indices) != len(set(indices)):
        raise ValueError("--sample-indices cannot contain duplicates")
    invalid = [index for index in indices if not 0 <= index < len(samples)]
    if invalid:
        raise ValueError(
            f"--sample-indices out of range for {len(samples)} samples: {invalid}"
        )
    return [samples[index] for index in indices]


def load_saved_outputs(path: Path) -> dict[str | int, dict]:
    """Load generated outputs from either a generation or completed eval JSONL."""
    outputs = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("raw_transcript", "")
            if not key or "baseline_output" not in record or "candidate_output" not in record:
                continue
            output = {
                "baseline": {
                    "text": record["baseline_output"],
                    "duration_ms": record.get("baseline_duration_ms", 0),
                },
                "candidate": {
                    "text": record["candidate_output"],
                    "duration_ms": record.get("candidate_duration_ms", 0),
                },
            }
            outputs[key] = output
            sample_index = record.get("sample_index")
            if isinstance(sample_index, int):
                outputs[sample_index] = output
    return outputs


# ---- Model inference ----

def messages_for_layout(sample: dict, layout: str) -> list[dict]:
    """Build model messages using prepared-dataset or live VoiceInk placement."""
    if layout == sample.get("stored_message_layout"):
        return [
            {"role": "system", "content": sample["stored_system_text"]},
            {"role": "user", "content": sample["stored_user_text"]},
        ]
    components = {
        "window_context": sample["window_context"],
        "clipboard_context": sample["clipboard_context"],
        "custom_vocabulary": sample["custom_vocabulary"],
        "transcript": sample["raw_transcript"],
    }
    if layout == "prepared":
        system_text = sample["system_text"]
        user_text = build_user_message(components)
    elif layout == "voiceink":
        system_text = build_voiceink_system_message(
            components, sample["system_text"]
        )
        user_text = build_voiceink_user_message(components)
    else:
        raise ValueError(f"Unsupported message layout: {layout}")
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def query_llama(messages: list[dict], model: str, host: str, port: int,
                temperature: float = 0.3,
                seed: int | None = None) -> tuple[str, float]:
    """Send a chat completion request. Returns (response_text, duration_ms)."""
    request = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed
    payload = json.dumps(request).encode("utf-8")

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


def generate_outputs(samples: list[dict], model: str, host: str, port: int,
                     temperature: float = 0.3,
                     seed: int | None = None,
                     message_layout: str = "voiceink") -> list[dict]:
    """Run a model on all eval samples sequentially."""
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {model}...", end=" ", flush=True)
        try:
            text, duration_ms = query_llama(
                messages_for_layout(sample, message_layout), model, host, port,
                temperature=temperature, seed=seed,
            )
            print(f"{duration_ms:.0f}ms")
            results.append({"text": text, "duration_ms": duration_ms})
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"text": "", "duration_ms": 0, "error": str(exc)})
    return results


def parse_judge_response(raw: str, *, require_context_analysis: bool = False,
                         require_pairwise: bool = False) -> dict | None:
    """Extract and strictly validate a judge response."""
    try:
        parsed = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    required_keys = {"output_a", "output_b"}
    if require_context_analysis:
        required_keys.update(("context_analysis", "score_analysis"))
    if require_pairwise:
        required_keys.add("pairwise")
    if set(parsed) != required_keys:
        return None
    for output in (parsed["output_a"], parsed["output_b"]):
        if not isinstance(output, dict) or set(output) != set(SCORE_DIMENSIONS):
            return None
        if any(type(output[dimension]) is not int or not 1 <= output[dimension] <= 5
               for dimension in SCORE_DIMENSIONS):
            return None
    if require_context_analysis:
        for analysis_name in ("context_analysis", "score_analysis"):
            analysis = parsed[analysis_name]
            if (not isinstance(analysis, dict)
                    or set(analysis) != {"output_a", "output_b"}):
                return None
            if any(not isinstance(analysis[key], str) or not analysis[key].strip()
                   for key in ("output_a", "output_b")):
                return None
    if require_pairwise:
        pairwise = parsed["pairwise"]
        if not isinstance(pairwise, dict) or set(pairwise) != {
            "preference", "confidence", "material_difference", "reason"
        }:
            return None
        if pairwise["preference"] not in ("output_a", "output_b", "tie"):
            return None
        if pairwise["confidence"] not in ("low", "medium", "high"):
            return None
        if type(pairwise["material_difference"]) is not bool:
            return None
        if not isinstance(pairwise["reason"], str) or not pairwise["reason"].strip():
            return None
    return parsed


def judge_one(
    sample: dict,
    baseline_text: str,
    candidate_text: str,
    judge_provider: str,
    judge_model: str,
    reasoning_effort: str,
    dry_run: bool,
    assignment_seed: int,
    seed_offset: int,
    judge_template: str,
    judge_rubric: str = "legacy",
) -> dict | None:
    """Judge a single sample. Returns structured result or None on failure."""
    rng = random.Random(assignment_seed + seed_offset)
    coin = rng.random() < 0.5

    if coin:
        output_a, output_b = baseline_text, candidate_text
        a_is = "baseline"
    else:
        output_a, output_b = candidate_text, baseline_text
        a_is = "candidate"

    prompt = judge_template.format(
        raw_transcript=sample["raw_transcript"],
        gold_label=sample["gold_label"],
        output_a=output_a,
        output_b=output_b,
        custom_vocabulary=sample.get("custom_vocabulary", "(none)"),
        window_context=sample.get("window_context", "(none)"),
        clipboard_context=sample.get("clipboard_context", "(none)"),
    )

    if dry_run:
        print(f"  --- DRY RUN ---")
        print(f"  Transcript: {sample['raw_transcript'][:100]}...")
        print(f"  A={a_is}, B={'candidate' if a_is == 'baseline' else 'baseline'}")
        return None

    for attempt in range(2):
        try:
            output_schema = CODEX_EVAL_JUDGE_SCHEMA
            if judge_rubric == "strict-v2":
                output_schema = CODEX_EVAL_JUDGE_STRICT_V2_SCHEMA
            raw = call_llm(
                prompt,
                provider=judge_provider,
                model=judge_model,
                reasoning_effort=reasoning_effort,
                output_schema=output_schema if judge_provider == "codex" else None,
                timeout=600,
            )
        except Exception as exc:
            print(f"  JUDGE ERROR: {exc}", file=sys.stderr)
            return None

        parsed = parse_judge_response(
            raw,
            require_context_analysis=judge_provider == "codex",
            require_pairwise=(
                judge_provider == "codex" and judge_rubric == "strict-v2"
            ),
        )
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
        baseline_context_analysis = parsed.get("context_analysis", {}).get("output_a")
        candidate_context_analysis = parsed.get("context_analysis", {}).get("output_b")
        baseline_score_analysis = parsed.get("score_analysis", {}).get("output_a")
        candidate_score_analysis = parsed.get("score_analysis", {}).get("output_b")
    else:
        baseline_scores = parsed["output_b"]
        candidate_scores = parsed["output_a"]
        baseline_context_analysis = parsed.get("context_analysis", {}).get("output_b")
        candidate_context_analysis = parsed.get("context_analysis", {}).get("output_a")
        baseline_score_analysis = parsed.get("score_analysis", {}).get("output_b")
        candidate_score_analysis = parsed.get("score_analysis", {}).get("output_a")

    result = {
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "baseline_context_analysis": baseline_context_analysis,
        "candidate_context_analysis": candidate_context_analysis,
        "baseline_score_analysis": baseline_score_analysis,
        "candidate_score_analysis": candidate_score_analysis,
    }
    pairwise = parsed.get("pairwise")
    if pairwise:
        preference = pairwise["preference"]
        if preference == "tie":
            mapped_preference = "tie"
        elif (preference == "output_a") == (a_is == "baseline"):
            mapped_preference = "baseline"
        else:
            mapped_preference = "candidate"
        result.update({
            "pairwise_preference": mapped_preference,
            "pairwise_confidence": pairwise["confidence"],
            "pairwise_material_difference": pairwise["material_difference"],
            "pairwise_reason": pairwise["reason"],
        })
    return result


# ---- Scoring ----

def weighted_score(scores: dict) -> float:
    """Compute weighted score normalized to 0-100."""
    raw = sum(scores.get(dim, 3) * w for dim, w in WEIGHTS.items())
    return round(raw / (MAX_SCORE * TOTAL_WEIGHT) * 100, 1)


def _p_value(baseline: list[float], candidate: list[float]) -> float | None:
    """Paired two-sided p-value. Wilcoxon if scipy available, else permutation."""
    diffs = [c - b for b, c in zip(baseline, candidate)]
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 10:
        return None
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(nonzero)
        return float(p)
    except ImportError:
        n = len(diffs)
        observed = abs(sum(diffs) / n)
        rng = random.Random(42)
        count = sum(
            1 for _ in range(10000)
            if abs(sum(d * rng.choice((-1, 1)) for d in diffs) / n) >= observed
        )
        return count / 10000


def _bootstrap_ci(baseline: list[float], candidate: list[float]) -> tuple[float, float]:
    """Bootstrap 95% CI for mean difference (candidate - baseline)."""
    diffs = [c - b for b, c in zip(baseline, candidate)]
    n = len(diffs)
    rng = random.Random(42)
    boot_means = sorted(
        sum(diffs[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(10000)
    )
    return round(boot_means[250], 2), round(boot_means[9749], 2)


def _wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * (p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5
    return round((centre - spread) / denom, 3), round((centre + spread) / denom, 3)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_values:
        return 0
    k = (len(sorted_values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _latency_stats(sorted_durations: list[float]) -> dict:
    """Compute latency stats from a pre-sorted list of durations in ms."""
    if not sorted_durations:
        return {"avg": 0, "p50": 0, "p90": 0, "p99": 0}
    avg = sum(sorted_durations) / len(sorted_durations)
    return {
        "avg": round(avg, 1),
        "p50": round(_percentile(sorted_durations, 50), 1),
        "p90": round(_percentile(sorted_durations, 90), 1),
        "p99": round(_percentile(sorted_durations, 99), 1),
    }


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

    # Per-dimension significance
    per_dimension = {}
    for dim in WEIGHTS:
        b_vals = per_dim_baseline[dim]
        c_vals = per_dim_candidate[dim]
        per_dimension[dim] = {
            "baseline_avg": round(sum(b_vals) / n, 2),
            "candidate_avg": round(sum(c_vals) / n, 2),
            "weight": WEIGHTS[dim],
            "p_value": _p_value(b_vals, c_vals),
        }

    # Overall significance
    overall_p = _p_value(baseline_weighted, candidate_weighted)
    overall_ci = _bootstrap_ci(baseline_weighted, candidate_weighted)

    # Win rate CI (candidate wins / non-tied samples)
    n_decided = wins["baseline"] + wins["candidate"]
    win_rate_ci = _wilson_ci(wins["candidate"], n_decided) if n_decided > 0 else (0, 0)

    # Latency
    b_durations = sorted(o["duration_ms"] for o in baseline_outputs if o.get("duration_ms"))
    c_durations = sorted(o["duration_ms"] for o in candidate_outputs if o.get("duration_ms"))
    b_latency = _latency_stats(b_durations)
    c_latency = _latency_stats(c_durations)
    latency_p = _p_value(b_durations, c_durations) if b_durations and c_durations else None

    # Winner determination
    winner, reason = determine_winner(
        b_avg, c_avg,
        {d: sum(v) / len(v) for d, v in per_dim_baseline.items()},
        {d: sum(v) / len(v) for d, v in per_dim_candidate.items()},
    )

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "n_samples": n,
        "baseline_avg_score": b_avg,
        "candidate_avg_score": c_avg,
        "wins": wins,
        "win_rate_ci_95": win_rate_ci,
        "per_dimension": per_dimension,
        "overall_p_value": overall_p,
        "overall_ci_95": overall_ci,
        "baseline_latency": b_latency,
        "candidate_latency": c_latency,
        "speed_ratio": round(b_latency["avg"] / c_latency["avg"], 2) if c_latency["avg"] > 0 else 0,
        "latency_p_value": latency_p,
        "winner": winner,
        "winner_reason": reason,
    }
    pairwise_judgments = [
        judgment for judgment in judgments
        if judgment.get("pairwise_preference") in ("baseline", "candidate", "tie")
    ]
    if pairwise_judgments:
        preferences = {
            name: sum(
                judgment["pairwise_preference"] == name
                for judgment in pairwise_judgments
            )
            for name in ("baseline", "candidate", "tie")
        }
        material = {
            name: sum(
                judgment["pairwise_preference"] == name
                and judgment.get("pairwise_material_difference") is True
                for judgment in pairwise_judgments
            )
            for name in ("baseline", "candidate")
        }
        summary["judge_pairwise"] = {
            "n": len(pairwise_judgments),
            "preferences": preferences,
            "material_wins": material,
            "confidence": dict(Counter(
                judgment.get("pairwise_confidence", "unknown")
                for judgment in pairwise_judgments
            )),
        }
    return summary


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

def _fmt_p(p: float | None) -> str:
    """Format a p-value."""
    if p is None:
        return "n/a   "
    if p < 0.0001:
        return "<.0001"
    return f"{p:.4f}"


def print_summary(summary: dict) -> None:
    """Print formatted summary table."""
    b_model = summary["baseline_model"]
    c_model = summary["candidate_model"]
    n = summary["n_samples"]
    W = 72

    print()
    print("=" * W)
    print(f"  EVALUATION: {b_model} vs {c_model}  (n={n})")
    print("=" * W)

    # ---- Quality per dimension ----
    print()
    print("  Quality (1-5 scale):")
    print(f"  {'':26s} {'Wt':>2s}  {'Baseline':>8s}  {'Candidate':>9s}  {'Diff':>6s}  {'p-value':>9s}")
    print(f"  {'':26s} {'--':>2s}  {'--------':>8s}  {'---------':>9s}  {'----':>6s}  {'---------':>9s}")
    for dim in WEIGHTS:
        info = summary["per_dimension"][dim]
        b_val = info["baseline_avg"]
        c_val = info["candidate_avg"]
        diff = c_val - b_val
        p_str = _fmt_p(info.get("p_value"))
        label = dim.replace("_", " ")
        print(f"  {label:26s} {info['weight']:>2d}  {b_val:8.2f}  {c_val:9.2f}  {diff:+5.2f}  {p_str}")

    # ---- Overall score ----
    print(f"  {'-' * (W - 2)}")
    b_avg = summary["baseline_avg_score"]
    c_avg = summary["candidate_avg_score"]
    diff = round(c_avg - b_avg, 1)
    overall_p = _fmt_p(summary.get("overall_p_value"))
    ci = summary.get("overall_ci_95", (0, 0))
    print(f"  {'Overall (weighted, 0-100)':26s}     {b_avg:8.1f}  {c_avg:9.1f}  {diff:+5.1f}  {overall_p}")
    print(f"  {'':26s}     {'':8s}  {'':9s}  {'':6s}  95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    # ---- Win rate ----
    w = summary["wins"]
    n_decided = w["baseline"] + w["candidate"]
    win_pct = w["candidate"] / n_decided * 100 if n_decided else 0
    wr_ci = summary.get("win_rate_ci_95", (0, 0))
    print()
    print(f"  Win rate: {win_pct:.0f}% ({w['candidate']}/{n_decided})"
          f"  95% CI [{wr_ci[0]*100:.0f}%, {wr_ci[1]*100:.0f}%]"
          f"  (ties: {w['tie']})")

    judge_pairwise = summary.get("judge_pairwise")
    if judge_pairwise:
        preferences = judge_pairwise["preferences"]
        material = judge_pairwise["material_wins"]
        print(
            "  Strict pairwise: "
            f"candidate {preferences['candidate']}, "
            f"baseline {preferences['baseline']}, ties {preferences['tie']}"
        )
        print(
            "  Material wins:  "
            f"candidate {material['candidate']}, baseline {material['baseline']}"
        )

    # ---- Latency ----
    b_lat = summary.get("baseline_latency", {})
    c_lat = summary.get("candidate_latency", {})
    if b_lat.get("avg") and c_lat.get("avg"):
        ratio = summary["speed_ratio"]
        lat_p = _fmt_p(summary.get("latency_p_value"))
        print()
        print(f"  Latency (ms):          {'Baseline':>10s}  {'Candidate':>10s}  {'Speedup':>7s}")
        print(f"  {'':22s}  {'----------':>10s}  {'----------':>10s}  {'-------':>7s}")
        for stat in ("avg", "p50", "p90", "p99"):
            b_v = b_lat[stat]
            c_v = c_lat[stat]
            spd = f"{b_v / c_v:.1f}x" if c_v > 0 else ""
            print(f"    {stat:20s}  {b_v:10.0f}  {c_v:10.0f}  {spd:>7s}")
        print(f"  {'':22s}  {'':10s}  {'':10s}  {lat_p}")

    # ---- Verdict ----
    print()
    print(f"  {'-' * (W - 2)}")
    winner = summary["winner"].upper()
    reason = summary["winner_reason"]
    print(f"  WINNER: {winner}  |  {reason}")
    print("=" * W)
    print()


def write_generation_outputs(path: Path, samples: list[dict],
                             baseline_outputs: list[dict],
                             candidate_outputs: list[dict],
                             baseline_model: str, candidate_model: str,
                             baseline_temperature: float = 0.3,
                             candidate_temperature: float = 0.3,
                             inference_seed: int | None = None,
                             baseline_message_layout: str = "voiceink",
                             candidate_message_layout: str = "voiceink") -> None:
    """Persist model outputs before any external judging begins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, (sample, baseline, candidate) in enumerate(
            zip(samples, baseline_outputs, candidate_outputs)
        ):
            record = {
                "sample_index": sample.get("sample_index", index),
                "raw_transcript": sample["raw_transcript"],
                "gold_label": sample["gold_label"],
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "baseline_sampling": {
                    "temperature": baseline_temperature,
                    "seed": inference_seed,
                    "message_layout": baseline_message_layout,
                },
                "candidate_sampling": {
                    "temperature": candidate_temperature,
                    "seed": inference_seed,
                    "message_layout": candidate_message_layout,
                },
                "baseline_output": baseline["text"],
                "candidate_output": candidate["text"],
                "baseline_duration_ms": baseline.get("duration_ms", 0),
                "candidate_duration_ms": candidate.get("duration_ms", 0),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated outputs: {path}")


def write_results(output_dir: Path, samples: list[dict], baseline_outputs: list[dict],
                  candidate_outputs: list[dict], judgments: list[dict],
                  summary: dict, judge_metadata: dict) -> None:
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
                "sample_index": sample.get("sample_index", i),
                "raw_transcript": sample["raw_transcript"],
                "gold_label": sample["gold_label"],
                "baseline_output": b_out["text"],
                "candidate_output": c_out["text"],
                "baseline_duration_ms": b_out["duration_ms"],
                "candidate_duration_ms": c_out["duration_ms"],
                "baseline_scores": judgment["baseline_scores"],
                "candidate_scores": judgment["candidate_scores"],
                "baseline_context_analysis": judgment.get("baseline_context_analysis"),
                "candidate_context_analysis": judgment.get("candidate_context_analysis"),
                "baseline_score_analysis": judgment.get("baseline_score_analysis"),
                "candidate_score_analysis": judgment.get("candidate_score_analysis"),
                "pairwise_preference": judgment.get("pairwise_preference"),
                "pairwise_confidence": judgment.get("pairwise_confidence"),
                "pairwise_material_difference": judgment.get(
                    "pairwise_material_difference"
                ),
                "pairwise_reason": judgment.get("pairwise_reason"),
                "judge": judge_metadata,
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
    if args.judge_rubric == "strict-v2" and args.judge_provider != "codex":
        print(
            "--judge-rubric strict-v2 currently requires --judge-provider codex; "
            "legacy Claude judging remains available",
            file=sys.stderr,
        )
        sys.exit(2)
    judge_model = resolve_model(args.judge_provider, args.judge_model)

    judge_template = ""
    judge_metadata = provider_metadata(
        args.judge_provider, judge_model, args.judge_reasoning_effort
    )
    judge_metadata["rubric"] = args.judge_rubric
    if not args.generate_only:
        prompt_base = (
            STRICT_V2_JUDGE_PROMPT_PATH
            if args.judge_rubric == "strict-v2"
            else JUDGE_PROMPT_PATH
        )
        prompt_path = provider_prompt_path(prompt_base, args.judge_provider)
        if not prompt_path.is_file():
            print(f"Judge prompt not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        judge_template = prompt_path.read_text(encoding="utf-8")
        judge_metadata["prompt_sha256"] = hashlib.sha256(
            judge_template.encode("utf-8")
        ).hexdigest()

    if not args.eval_data.exists():
        print(f"Eval data not found: {args.eval_data}", file=sys.stderr)
        sys.exit(1)

    if args.generate_only and args.outputs:
        print("--generate-only cannot be combined with --outputs", file=sys.stderr)
        sys.exit(2)
    if args.baseline_temperature < 0 or args.candidate_temperature < 0:
        print("Inference temperatures must be non-negative", file=sys.stderr)
        sys.exit(2)

    # Load and validate the full corpus before applying an explicit debug slice.
    samples = load_eval_data(args.eval_data)
    try:
        validate_eval_corpus(
            args.eval_data,
            len(samples),
            allow_noncanonical=args.allow_noncanonical_eval,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    if args.sample_indices and args.limit > 0:
        print("--sample-indices cannot be combined with --limit", file=sys.stderr)
        sys.exit(2)
    if args.sample_indices:
        try:
            samples = select_sample_indices(samples, args.sample_indices)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
    elif args.limit > 0:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} eval samples")

    # Load cache from previous run if resuming
    cache = {}
    if args.resume:
        if args.resume.exists():
            cache = load_cached_results(args.resume)
            print(f"Loaded {len(cache)} cached results from {args.resume}")
            expected_prompt_hash = judge_metadata.get("prompt_sha256")
            compatible_cache = {
                key: record for key, record in cache.items()
                if record.get("judge", {}).get("prompt_sha256") == expected_prompt_hash
            }
            ignored = len(cache) - len(compatible_cache)
            cache = compatible_cache
            if ignored:
                print(
                    f"Ignored {ignored} cached results from a different or "
                    "unversioned judge prompt"
                )
        else:
            print(f"WARNING: resume file not found: {args.resume}")

    cached_indices = [
        i for i, sample in enumerate(samples)
        if sample["raw_transcript"] in cache
    ]
    uncached_indices = [i for i in range(len(samples)) if i not in cached_indices]

    if cache:
        print(f"  {len(cached_indices)} cached, {len(uncached_indices)} to evaluate")

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
            "baseline_context_analysis": c.get("baseline_context_analysis"),
            "candidate_context_analysis": c.get("candidate_context_analysis"),
            "baseline_score_analysis": c.get("baseline_score_analysis"),
            "candidate_score_analysis": c.get("candidate_score_analysis"),
            "pairwise_preference": c.get("pairwise_preference"),
            "pairwise_confidence": c.get("pairwise_confidence"),
            "pairwise_material_difference": c.get(
                "pairwise_material_difference"
            ),
            "pairwise_reason": c.get("pairwise_reason"),
        }

    if args.outputs:
        if not args.outputs.is_file():
            print(f"Saved outputs not found: {args.outputs}", file=sys.stderr)
            sys.exit(1)
        saved_outputs = load_saved_outputs(args.outputs)
        missing = []
        for i in uncached_indices:
            saved = (
                saved_outputs.get(samples[i]["raw_transcript"])
                or saved_outputs.get(i)
            )
            if not saved:
                missing.append(i)
                continue
            baseline_outputs[i] = saved["baseline"]
            candidate_outputs[i] = saved["candidate"]
        if missing:
            print(
                f"Saved outputs are missing {len(missing)} selected eval samples",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loaded outputs for {len(uncached_indices)} samples from {args.outputs}")
    elif uncached_indices:
        uncached_samples = [samples[i] for i in uncached_indices]
        warmup_msgs = [{"role": "user", "content": "Hello"}]

        for model_name, temperature, message_layout, destination in (
            (args.baseline, args.baseline_temperature,
             args.baseline_message_layout, baseline_outputs),
            (args.candidate, args.candidate_temperature,
             args.candidate_message_layout, candidate_outputs),
        ):
            print(
                f"Warming up {model_name} (temperature={temperature:g}, "
                f"layout={message_layout})...",
                end=" ", flush=True,
            )
            try:
                _, milliseconds = query_llama(
                    warmup_msgs, model_name, args.llama_host, args.llama_port,
                    temperature=temperature, seed=args.inference_seed,
                )
                print(f"{milliseconds:.0f}ms (discarded)")
            except Exception as exc:
                print(f"WARNING: warmup failed: {exc}")

            print(
                f"\nGenerating outputs ({model_name}, temperature={temperature:g}, "
                f"layout={message_layout})..."
            )
            generated = generate_outputs(
                uncached_samples, model_name, args.llama_host, args.llama_port,
                temperature=temperature, seed=args.inference_seed,
                message_layout=message_layout,
            )
            for position, sample_index in enumerate(uncached_indices):
                destination[sample_index] = generated[position]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        generation_path = args.generation_output or (
            args.output_dir / f"generations_{timestamp}.jsonl"
        )
        write_generation_outputs(
            generation_path, samples, baseline_outputs, candidate_outputs,
            args.baseline, args.candidate,
            args.baseline_temperature, args.candidate_temperature,
            args.inference_seed, args.baseline_message_layout,
            args.candidate_message_layout,
        )

    if args.generate_only:
        print("Generation-only run complete; no judge was called.")
        return

    errors = 0
    if uncached_indices:
        print(
            f"\nJudging {len(uncached_indices)} new outputs "
            f"({args.judge_provider}/{judge_model}, "
            f"effort={args.judge_reasoning_effort})..."
        )

        if args.parallel <= 1 or args.dry_run:
            for j, i in enumerate(uncached_indices):
                print(f"  [{j+1}/{len(uncached_indices)}]", end=" ")
                result = judge_one(
                    samples[i], baseline_outputs[i]["text"],
                    candidate_outputs[i]["text"], args.judge_provider,
                    judge_model, args.judge_reasoning_effort, args.dry_run,
                    args.seed, samples[i]["sample_index"], judge_template,
                    args.judge_rubric,
                )
                judge_results[i] = result
                if not result:
                    errors += 1
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {}
                for j, i in enumerate(uncached_indices):
                    fut = pool.submit(
                        judge_one, samples[i], baseline_outputs[i]["text"],
                        candidate_outputs[i]["text"], args.judge_provider,
                        judge_model, args.judge_reasoning_effort, False,
                        args.seed, samples[i]["sample_index"], judge_template,
                        args.judge_rubric,
                    )
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
    summary["judge"] = judge_metadata
    print_summary(summary)

    # Write results
    write_results(args.output_dir, filtered_samples, filtered_baseline,
                  filtered_candidate, filtered_judgments, summary,
                  judge_metadata)


if __name__ == "__main__":
    main()
