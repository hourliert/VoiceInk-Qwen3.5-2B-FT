#!/usr/bin/env python3
"""Run a release-aware VoiceInk model comparison.

The sealed acceptance split is the default. The historical regression corpus
is intentionally opt-in and belongs at the final promotion stage only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import load_manifest, split_path  # noqa: E402


def resolved_command(args: argparse.Namespace) -> list[str]:
    manifest = load_manifest(args.release_manifest)
    if manifest.get("task", "sft") != "sft":
        raise ValueError("Model evaluation requires an SFT dataset release")
    common = [
        "--baseline", args.baseline, "--candidate", args.candidate,
        "--release-manifest", str(args.release_manifest),
        "--judge-model", args.judge_model,
        "--judge-reasoning-effort", args.judge_reasoning_effort,
        "--parallel", str(args.parallel), "--llama-host", args.llama_host,
        "--llama-port", str(args.llama_port),
    ]
    if args.include_regression:
        return [
            sys.executable, str(ROOT / "src/eval/strict.py"), "run", *common,
            "--output-dir", str(args.output_dir),
            "--report", str(args.output_dir / "promotion-evidence.json"),
        ]
    acceptance = split_path(args.release_manifest, manifest, "acceptance")
    comparison_dir = (
        args.output_dir / f"{args.candidate}-vs-{args.baseline}" / "acceptance-150"
    )
    return [
        sys.executable, str(ROOT / "src/eval/evaluate.py"),
        "--baseline", args.baseline, "--candidate", args.candidate,
        "--eval-data", str(acceptance), "--allow-noncanonical-eval",
        "--judge-provider", "codex", "--judge-rubric", "strict-v3",
        "--judge-model", args.judge_model,
        "--judge-reasoning-effort", args.judge_reasoning_effort,
        "--parallel", str(args.parallel), "--llama-host", args.llama_host,
        "--llama-port", str(args.llama_port),
        "--output-dir", str(comparison_dir),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--include-regression", action="store_true",
        help="Run the optional locked 440 regression suite and final promotion gate",
    )
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--judge-reasoning-effort", default="low")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=8002)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/canonical")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    command = resolved_command(args)
    mode = "acceptance + final regression" if args.include_regression else "acceptance only"
    print(f"VoiceInk release evaluation: {mode}")
    print(f"  Release: {args.release_manifest}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Candidate: {args.candidate}")
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
