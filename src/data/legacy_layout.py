#!/usr/bin/env python3
"""Plan or apply a reversible migration of pre-canonical private artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_MODELS = {
    "Qwen3.5-2B-VoiceInk-v3",
    "Qwen3.5-2B-VoiceInk-v3_gguf",
    "Qwen3.8-2B-VoiceInk-v1",
    "Qwen3.8-2B-VoiceInk-v1_gguf",
}
DEFAULT_CANONICAL_TRAINING = {"qwen38-2b-voiceink-v1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _children(root: Path, name: str) -> Iterable[Path]:
    path = root / name
    return sorted(path.iterdir()) if path.is_dir() else []


def build_plan(
    root: Path,
    *,
    canonical_models: set[str] | None = None,
    canonical_training: set[str] | None = None,
) -> list[dict[str, str]]:
    """Return non-destructive moves into explicit canonical/legacy namespaces."""
    canonical_models = canonical_models or DEFAULT_CANONICAL_MODELS
    canonical_training = canonical_training or DEFAULT_CANONICAL_TRAINING
    moves: list[dict[str, str]] = []

    def add(source: Path, target: Path, classification: str) -> None:
        if source == target or not source.exists():
            return
        moves.append({
            "source": str(source.resolve()),
            "target": str(target.resolve()),
            "classification": classification,
        })

    dataset_keep = {"registry", "exports", "releases", "regression", "legacy"}
    for path in _children(root, "datasets"):
        if path.name == ".gitkeep":
            continue
        if path.is_dir() and path.name not in dataset_keep:
            add(path, root / "datasets" / "legacy" / path.name, "legacy-dataset")
        elif path.is_file():
            add(
                path,
                root / "datasets" / "legacy" / "root-files" / path.name,
                "legacy-dataset",
            )

    training_keep = {"canonical", "backups", "mlflow", "legacy"}
    for path in _children(root, "training"):
        if path.name in training_keep:
            continue
        if path.is_dir() and path.name in canonical_training:
            add(
                path,
                root / "training" / "canonical" / path.name,
                "canonical-training",
            )
        else:
            add(path, root / "training" / "legacy" / path.name, "legacy-training")

    for path in _children(root, "results"):
        if path.name not in {"canonical", "legacy"}:
            add(path, root / "results" / "legacy" / path.name, "legacy-evaluation")

    for path in _children(root, "models"):
        if not path.is_dir() or "voiceink" not in path.name.lower():
            continue
        if path.name in canonical_models:
            add(path, root / "models" / "canonical" / path.name, "canonical-model")
        else:
            add(path, root / "models" / "legacy" / path.name, "legacy-model")
    return moves


def running_training_processes() -> list[str]:
    commands = []
    proc = Path("/proc")
    if not proc.is_dir():
        return commands
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "src/training/" in command and "legacy_layout.py" not in command:
            commands.append(command.strip())
    return commands


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def apply_plan(
    root: Path,
    moves: list[dict[str, str]],
    *,
    allow_running_training: bool = False,
) -> dict:
    active = running_training_processes()
    if active and not allow_running_training:
        raise RuntimeError(
            "Refusing to move artifacts while training is active: " + "; ".join(active)
        )
    applied = []
    for move in moves:
        source = Path(move["source"])
        target = Path(move["target"])
        if not source.exists() and target.exists():
            applied.append(move | {"status": "already-moved", "bytes": _tree_bytes(target)})
            continue
        if not source.exists():
            applied.append(move | {"status": "source-missing", "bytes": 0})
            continue
        if target.exists():
            raise FileExistsError(f"Both legacy source and target exist: {source} / {target}")
        size = _tree_bytes(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        applied.append(move | {"status": "moved", "bytes": size})

    config = root / "config" / "models.ini"
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        updated = text
        model_moves = sorted(
            (
                move
                for move in applied
                if move["classification"] in {"legacy-model", "canonical-model"}
            ),
            key=lambda move: len(move["source"]),
            reverse=True,
        )
        for move in model_moves:
            updated = updated.replace(move["source"], move["target"])
        if updated != text:
            temporary = config.with_suffix(".ini.tmp")
            temporary.write_text(updated, encoding="utf-8")
            os.replace(temporary, config)

    payload = {
        "schema": "voiceink.legacy-migration/v1",
        "created_at": utc_now(),
        "reversible": True,
        "deleted": [],
        "moves": applied,
    }
    manifest_root = root / "datasets" / "registry" / "legacy-migrations"
    manifest_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = manifest_root / f"legacy-migration-{stamp}.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload | {"manifest": str(manifest)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--allow-running-training",
        action="store_true",
        help="Explicit override; normally migration fails closed while a trainer is active",
    )
    parser.add_argument("--canonical-model", action="append", default=[])
    parser.add_argument("--canonical-training", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = DEFAULT_CANONICAL_MODELS | set(args.canonical_model)
    training = DEFAULT_CANONICAL_TRAINING | set(args.canonical_training)
    moves = build_plan(
        args.root.resolve(),
        canonical_models=models,
        canonical_training=training,
    )
    if args.apply:
        result = apply_plan(
            args.root.resolve(),
            moves,
            allow_running_training=args.allow_running_training,
        )
    else:
        result = {
            "mode": "dry-run",
            "active_training": running_training_processes(),
            "moves": moves,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
