#!/usr/bin/env python3
"""Generate synthetic QA debrief transcripts for fine-tuning training data.

Produces long, realistic QA debrief transcripts (500-3500 words) with
gold-standard labels using Claude Sonnet as generator. Each sample mimics
a real GT Coach QA session with repetitive coaching phrases, STT errors,
and speaker narration.

Output: datasets/synthetic/labeled.jsonl (same schema as datasets/labeled.jsonl)

Usage:
    python3 src/synthetic/generate.py --count 5 --dry-run
    python3 src/synthetic/generate.py --count 120 --parallel 5
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "datasets" / "synthetic" / "labeled.jsonl"
DEFAULT_MODEL = "claude-sonnet-4-6"
GENERATOR_PROMPT_PATH = Path(__file__).resolve().parent / "generator_prompt.txt"
VOICEINK_PROMPT_PATH = ROOT / "docs" / "VOICEINK_PROMPT"
VOCABULARY_PATH = ROOT / "config" / "vocabulary.txt"

TRACKS = [
    "Monza", "Dragon Trail", "Nürburgring", "Spa", "Suzuka",
    "Brands Hatch", "Mount Panorama", "Laguna Seca", "Interlagos",
    "Silverstone", "Tsukuba", "Fuji Speedway",
]

SCENARIOS = [
    "clean run, coaching feedback on braking and corner exits",
    "deliberate crash testing in a specific corner, outlier detection verification",
    "comparing laps, reviewing improvement arc across the session",
    "bug found in coaching overlay, discussing fix with Claude Code",
    "integration test recording for automated testing later",
    "new feature QA after zone pipeline redesign",
    "first session on a new track, coach giving many initial recommendations",
    "fast laps chasing personal best, detailed sector analysis",
    "testing volume and timing changes in coaching messages",
    "endurance session, fatigue affecting consistency, coach adjusting",
    "reviewing telemetry data while debriefing post-session",
    "multiple crash scenarios testing outlier detection robustness",
]

WORD_TARGETS = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500]

WINDOW_CONTEXTS = [
    "GT Coach - Race Analysis",
    "GT Coach - Live Session",
    "Terminal - claude",
    "Gran Turismo 7",
    "GT Coach - Review",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic QA debrief training data.")
    p.add_argument("--count", type=int, default=120,
                   help="Number of synthetic samples to generate (default: 120)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel Claude CLI calls")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Claude model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling Claude")
    return p.parse_args()


def build_scenario_matrix(count: int) -> list[dict]:
    """Build a diverse matrix of (track, scenario, word_target) assignments."""
    assignments = []
    for i in range(count):
        assignments.append({
            "track": TRACKS[i % len(TRACKS)],
            "scenario": SCENARIOS[i % len(SCENARIOS)],
            "target_words": WORD_TARGETS[i % len(WORD_TARGETS)],
            "seed": i + 1,
            "index": i + 1,
        })
    return assignments


def build_prompt(assignment: dict, template: str) -> str:
    """Build the generator prompt for a specific assignment."""
    return template.format(
        track=assignment["track"],
        scenario=assignment["scenario"],
        target_words=assignment["target_words"],
        seed=assignment["seed"],
    )


def call_claude(prompt: str, model: str) -> str:
    """Call the Claude CLI and return the response text."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    env["CLAUDE_CODE_SKIP_UPDATE_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--model", model,
            "--disable-slash-commands",
            "--allowed-tools", "",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def parse_response(response: str) -> tuple[str, str]:
    """Extract raw transcript and clean transcript from Claude's response."""
    raw_match = re.search(r"<RAW_TRANSCRIPT>\s*(.*?)\s*</RAW_TRANSCRIPT>", response, re.DOTALL)
    clean_match = re.search(r"<CLEAN_TRANSCRIPT>\s*(.*?)\s*</CLEAN_TRANSCRIPT>", response, re.DOTALL)

    if not raw_match or not clean_match:
        raise ValueError("Response missing <RAW_TRANSCRIPT> or <CLEAN_TRANSCRIPT> tags")

    return raw_match.group(1).strip(), clean_match.group(1).strip()


def build_raw_request_json(transcript: str, window_context: str,
                           voiceink_prompt: str, custom_vocabulary: str) -> str:
    """Construct a valid VoiceInk-format raw_request_json envelope."""

    system_content = (
        f"{voiceink_prompt}\n\n"
        f"<CURRENT_WINDOW_CONTEXT>\n{window_context}\n</CURRENT_WINDOW_CONTEXT>\n\n"
        f"<CUSTOM_VOCABULARY>\n{custom_vocabulary}\n</CUSTOM_VOCABULARY>"
    )

    user_content = f"<TRANSCRIPT>\n{transcript}\n</TRANSCRIPT>"

    request = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "model": "Qwen3.5-2B-VoiceInk",
        "stream": True,
        "stop": ["</think>"],
    }

    return json.dumps(request, ensure_ascii=False)


def generate_one(assignment: dict, model: str, dry_run: bool,
                 generator_template: str, voiceink_prompt: str,
                 custom_vocabulary: str) -> tuple[dict, int, int] | None:
    """Generate a single synthetic sample. Returns (record, raw_words, clean_words) or None."""
    prompt = build_prompt(assignment, generator_template)
    idx = assignment["index"]

    if dry_run:
        print(f"--- DRY RUN [syn-{idx:03d}] ---")
        print(f"Track: {assignment['track']}, Scenario: {assignment['scenario']}")
        print(f"Target: {assignment['target_words']} words")
        print(prompt[:300])
        print("...\n")
        return None

    try:
        response = call_claude(prompt, model)
        raw_transcript, clean_transcript = parse_response(response)
    except Exception as exc:
        print(f"  ERROR [syn-{idx:03d}]: {exc}", file=sys.stderr)
        return None

    raw_words = len(raw_transcript.split())
    clean_words = len(clean_transcript.split())

    # Pick a plausible window context
    window_ctx = WINDOW_CONTEXTS[idx % len(WINDOW_CONTEXTS)]

    record = {
        "request_id": f"syn-{idx:03d}",
        "timestamp": datetime.now().isoformat(),
        "model_used_for_label": model,
        "original_model": "synthetic",
        "raw_request_json": build_raw_request_json(raw_transcript, window_ctx, voiceink_prompt, custom_vocabulary),
        "original_response": "",
        "label": clean_transcript,
    }

    return record, raw_words, clean_words


class SyntheticDataset:
    """Thread-safe output file for synthetic records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._records[record["request_id"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save(self, record: dict) -> None:
        with self._lock:
            self._records[record["request_id"]] = record
            self._flush()

    def existing_ids(self) -> set[str]:
        with self._lock:
            return set(self._records.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def main() -> None:
    args = parse_args()

    if not GENERATOR_PROMPT_PATH.exists():
        print(f"Generator prompt not found: {GENERATOR_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    if not VOICEINK_PROMPT_PATH.exists():
        print(f"VoiceInk prompt not found: {VOICEINK_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read templates once
    generator_template = GENERATOR_PROMPT_PATH.read_text(encoding="utf-8")
    voiceink_prompt = VOICEINK_PROMPT_PATH.read_text(encoding="utf-8").strip()
    custom_vocabulary = VOCABULARY_PATH.read_text(encoding="utf-8").strip()

    dataset = SyntheticDataset(args.output)
    existing = dataset.existing_ids()

    assignments = build_scenario_matrix(args.count)

    # Skip already-generated samples
    todo = [a for a in assignments if f"syn-{a['index']:03d}" not in existing]
    if len(todo) < len(assignments):
        print(f"Skipping {len(assignments) - len(todo)} already-generated samples")

    if not todo:
        print("All samples already generated.")
        return

    print(f"Generating {len(todo)} synthetic QA debrief samples")
    print(f"  Model: {args.model}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output: {args.output}")
    print()

    completed = 0
    errors = 0

    def process(assignment):
        return generate_one(assignment, args.model, args.dry_run,
                           generator_template, voiceink_prompt, custom_vocabulary)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process, a): a for a in todo}

        for future in as_completed(futures):
            assignment = futures[future]
            idx = assignment["index"]
            try:
                result = future.result()
                if result is None:
                    if not args.dry_run:
                        errors += 1
                    continue

                record, raw_words, clean_words = result
                dataset.save(record)
                completed += 1
                print(f"  [{completed}/{len(todo)}] syn-{idx:03d}: "
                      f"{assignment['track']}/{assignment['target_words']}w target → "
                      f"{raw_words}w raw, {clean_words}w clean")

            except Exception as exc:
                errors += 1
                print(f"  ERROR [syn-{idx:03d}]: {exc}", file=sys.stderr)

    if not args.dry_run:
        print(f"\nDone! {completed} generated, {errors} errors")
        print(f"Total synthetic samples: {len(dataset)}")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
