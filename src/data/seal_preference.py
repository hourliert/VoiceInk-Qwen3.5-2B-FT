#!/usr/bin/env python3
"""Seal reviewed DPO pairs as an immutable derivative of an SFT release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import canonical_json, load_manifest, sha256_file, split_path  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_pairs(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen = set()
    for line_number, row in enumerate(rows, 1):
        required = {"request_id", "prompt", "chosen", "rejected"}
        if not required <= row.keys():
            raise ValueError(f"{path}:{line_number} is missing preference fields")
        request_id = str(row["request_id"])
        if request_id in seen:
            raise ValueError(f"{path}:{line_number} duplicates request_id {request_id}")
        seen.add(request_id)
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{path}:{line_number} has identical chosen/rejected output")
    if not rows:
        raise ValueError(f"Preference split is empty: {path}")
    return rows


def acceptance_ids(parent_path: Path, parent: dict[str, Any]) -> set[str]:
    path = split_path(parent_path, parent, "acceptance")
    return {
        str(row["sample_id"])
        for row in load_jsonl(path)
        if row.get("sample_id") is not None
    }


def seal(
    name: str,
    train_path: Path,
    validation_path: Path,
    parent_path: Path,
    judgments_path: Path,
    judge_prompt_path: Path,
    release_root: Path,
) -> dict[str, Any]:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"Invalid preference release name: {name!r}")
    parent = load_manifest(parent_path)
    if parent.get("task", "sft") != "sft":
        raise ValueError("Preference releases must derive from a sealed SFT release")
    train = validate_pairs(train_path)
    validation = validate_pairs(validation_path)
    train_ids = {str(row["request_id"]) for row in train}
    validation_ids = {str(row["request_id"]) for row in validation}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Preference train/validation overlap: {len(overlap)} IDs")
    protected = acceptance_ids(parent_path, parent)
    leaked = protected & (train_ids | validation_ids)
    if leaked:
        raise ValueError(f"Sealed acceptance leakage into DPO: {len(leaked)} IDs")

    final_dir = (release_root / name).resolve()
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = final_dir.with_name(final_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    sources = {
        "train": (train_path, temporary / "train.jsonl", len(train)),
        "validation": (
            validation_path,
            temporary / "validation.jsonl",
            len(validation),
        ),
    }
    splits = {}
    for role, (source, target, records) in sources.items():
        shutil.copy2(source, target)
        splits[role] = {
            "path": target.name,
            "records": records,
            "sha256": sha256_file(target),
        }
    provenance = {}
    for name_, source in (
        ("judgments", judgments_path),
        ("judge_prompt", judge_prompt_path),
    ):
        target = temporary / source.name
        shutil.copy2(source, target)
        provenance[name_] = {
            "path": target.name,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    manifest: dict[str, Any] = {
        "schema": "voiceink.dataset-release/v2",
        "task": "preference",
        "name": name,
        "created_at": utc_now(),
        "sealed": True,
        "records": len(train) + len(validation),
        "splits": splits,
        "representations": {"preference": splits},
        "parent_release": {
            "name": parent["name"],
            "content_sha256": parent["content_sha256"],
            "manifest_path": str(parent_path.resolve()),
            "manifest_sha256": sha256_file(parent_path),
        },
        "provenance_files": provenance,
        "acceptance_excluded": True,
        "privacy": "private local files; MLflow logs fingerprints and counts only",
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode()
    ).hexdigest()
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final_dir)
    for path in final_dir.iterdir():
        path.chmod(0o444)
    final_dir.chmod(0o555)
    return manifest | {"manifest_path": str(final_dir / "manifest.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--judge-prompt", type=Path, required=True)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=ROOT / "datasets" / "releases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = seal(
        args.name,
        args.train,
        args.validation,
        args.parent_manifest,
        args.judgments,
        args.judge_prompt,
        args.release_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
