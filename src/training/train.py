#!/usr/bin/env python3
"""Manifest-only profile runner for canonical VoiceInk SFT and DPO jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import load_manifest, sha256_file, split_path  # noqa: E402

DEFAULT_PROFILES = ROOT / "config" / "training_profiles.toml"
SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def should_load_best_model(trainer_args: list[str]) -> bool:
    """Best-checkpoint loading is invalid when a smoke run disables eval."""
    return "--skip-eval" not in trainer_args


def load_profile(path: Path, name: str) -> dict:
    profiles = tomllib.loads(path.read_text(encoding="utf-8")).get("profiles", {})
    if name not in profiles:
        raise ValueError(f"Unknown training profile {name!r}; available: {', '.join(sorted(profiles))}")
    return profiles[name]


def resolved_command(args: argparse.Namespace) -> tuple[list[str], dict]:
    manifest = load_manifest(args.release_manifest)
    profile = load_profile(args.profiles, args.profile)
    if profile["mode"] != args.mode:
        raise ValueError(
            f"Profile {args.profile!r} is for {profile['mode']}, not {args.mode}"
        )
    if args.mode == "dpo":
        representation = "preference"
    else:
        representation = "text" if profile["trainer"] in ("lfm", "minicpm") else "text-blocks"
    train = split_path(
        args.release_manifest, manifest, "train", representation=representation
    )
    validation = split_path(
        args.release_manifest, manifest, "validation", representation=representation
    )
    version = args.version or manifest["name"]
    if not SAFE_VERSION.fullmatch(version):
        raise ValueError(f"Invalid output version: {version!r}")
    output_root = ROOT / profile["output_root"] / version
    model_root = ROOT / f"{profile['model_root']}-{version}"
    common = [
        "--train", str(train), "--eval", str(validation),
        "--base-model", str(ROOT / profile["base_model"] if not Path(profile["base_model"]).is_absolute()
                              and "/" not in profile["base_model"] else profile["base_model"]),
        "--max-seq-length", str(profile["max_seq_length"]),
        "--r", str(profile["lora_r"]), "--lora-alpha", str(profile["lora_alpha"]),
        "--batch-size", str(profile["batch_size"]), "--grad-accum", str(profile["grad_accum"]),
        "--epochs", str(profile["epochs"]), "--lr", str(profile["learning_rate"]),
        "--lora-dir", str(output_root / "lora"), "--output-dir", str(output_root / "outputs"),
        "--gguf-base", str(model_root), "--mlflow-experiment", "voiceink-training",
        "--mlflow-run-name", f"{args.profile}-{version}",
    ]
    if args.mode == "sft":
        scripts = {"qwen": "finetune.py", "lfm": "finetune_lfm25.py", "minicpm": "finetune_minicpm5.py"}
        try:
            script = scripts[profile["trainer"]]
        except KeyError as exc:
            raise ValueError(f"Unsupported SFT trainer: {profile['trainer']!r}") from exc
        command = [sys.executable, str(ROOT / "src" / "training" / script), *common,
                   "--eval-batch-size", str(profile["eval_batch_size"]),
                   "--warmup-steps", str(profile["warmup_steps"]),
                   "--eval-steps", str(profile["eval_steps"]),
                   "--save-steps", str(profile["save_steps"])]
        if should_load_best_model(args.trainer_arg):
            command.append("--load-best-model-at-end")
    else:
        if manifest.get("task") != "preference":
            raise ValueError("DPO requires a sealed preference release manifest")
        command = [sys.executable, str(ROOT / "src" / "training" / "finetune_qwen_dpo.py"), *common,
                   "--max-prompt-length", str(profile["max_prompt_length"]),
                   "--max-completion-length", str(profile["max_completion_length"]),
                   "--load-in-4bit", "--precompute-ref-log-probs", "--chunked-logits",
                   "--logit-chunks", "16", "--torch-empty-cache-steps", "1"]
    if args.export_gguf:
        command.extend(["--export-gguf", *args.export_gguf])
    if args.check_only:
        command.append("--check-only")
    command.extend(args.trainer_arg)
    return command, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sft", "dpo"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--version", help="Output version; defaults to release name")
    parser.add_argument("--export-gguf", nargs="*", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--trainer-arg", action="append", default=[],
                        help="Pass one extra argument to the underlying trainer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command, manifest = resolved_command(args)
    lineage = {
        "release_name": manifest["name"],
        "release_content_sha256": manifest["content_sha256"],
        "release_manifest": str(args.release_manifest.resolve()),
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "training_profile": args.profile,
        "training_profile_path": str(args.profiles.resolve()),
        "training_profile_sha256": sha256_file(args.profiles),
    }
    print(json.dumps({"lineage": lineage, "command": command}, indent=2))
    if args.print_command:
        return
    environment = os.environ.copy()
    environment.update({
        "VOICEINK_DATASET_RELEASE": lineage["release_name"],
        "VOICEINK_DATASET_RELEASE_SHA256": lineage["release_content_sha256"],
        "VOICEINK_DATASET_MANIFEST": lineage["release_manifest"],
        "VOICEINK_TRAINING_PROFILE": args.profile,
        "VOICEINK_TRAINING_PROFILE_PATH": lineage["training_profile_path"],
        "VOICEINK_TRAINING_PROFILE_SHA256": lineage["training_profile_sha256"],
        "VOICEINK_DATASET_MANIFEST_SHA256": lineage["release_manifest_sha256"],
    })
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
