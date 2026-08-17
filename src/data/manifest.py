"""Read and verify immutable canonical dataset release manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = {"train": 900, "validation": 150, "acceptance": 150}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _verify_splits(
    manifest_path: Path,
    splits: dict[str, Any],
    *,
    require_complete: bool,
    representation: str,
    expected_counts: dict[str, int | None],
) -> None:
    for role, expected_count in expected_counts.items():
        metadata = splits.get(role)
        if not metadata:
            raise ValueError(
                f"Dataset manifest is missing {role!r} for {representation!r}"
            )
        file_path = (manifest_path.parent / metadata["path"]).resolve()
        if not file_path.is_file():
            raise ValueError(f"Dataset split does not exist: {file_path}")
        if sha256_file(file_path) != metadata["sha256"]:
            raise ValueError(
                f"Dataset split fingerprint mismatch: {representation}/{role}"
            )
        with file_path.open(encoding="utf-8") as stream:
            rows = sum(1 for line in stream if line.strip())
        if rows != metadata["records"]:
            raise ValueError(f"Dataset split row count mismatch: {representation}/{role}")
        if require_complete and expected_count is not None and rows != expected_count:
            raise ValueError(
                f"Canonical {role} must contain exactly {expected_count} rows; found {rows}"
            )
        metadata["resolved_path"] = str(file_path)


def load_manifest(path: Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Load a release and fail closed on missing, changed, or partial splits."""
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "voiceink.dataset-release/v2":
        raise ValueError(f"Unsupported dataset manifest schema: {manifest.get('schema')!r}")
    if not manifest.get("sealed"):
        raise ValueError("Training requires a sealed immutable dataset release")
    expected_digest = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    actual_digest = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("Dataset manifest content fingerprint mismatch")
    task = manifest.get("task", "sft")
    if task == "sft":
        default_representation = "text-blocks"
        expected_counts: dict[str, int | None] = REQUIRED_SPLITS
    elif task == "preference":
        default_representation = "preference"
        expected_counts = {"train": None, "validation": None}
    else:
        raise ValueError(f"Unsupported dataset release task: {task!r}")
    representations = manifest.get("representations") or {
        default_representation: manifest.get("splits", {})
    }
    if manifest.get("splits") != representations.get(default_representation):
        raise ValueError(
            f"Manifest splits must alias the canonical {default_representation} representation"
        )
    for name, splits in representations.items():
        _verify_splits(
            path,
            splits,
            require_complete=require_complete,
            representation=name,
            expected_counts=expected_counts,
        )
    prompt = manifest.get("system_prompt")
    if task == "sft" and not prompt:
        raise ValueError("SFT release is missing its sealed system prompt")
    if task == "sft":
        prompt_path = Path(prompt["path"])
        if not prompt_path.is_absolute():
            prompt_path = path.parent / prompt_path
        if not prompt_path.is_file() or sha256_file(prompt_path) != prompt["sha256"]:
            raise ValueError("Dataset release system prompt fingerprint mismatch")
        prompt["resolved_path"] = str(prompt_path.resolve())
    regression = manifest.get("historical_regression")
    if task == "sft" and not regression:
        raise ValueError("SFT release is missing its locked historical regression corpus")
    if task == "sft":
        regression_path = Path(regression["path"])
        if not regression_path.is_absolute():
            regression_path = path.parent / regression_path
        if not regression_path.is_file():
            raise ValueError(f"Historical regression corpus does not exist: {regression_path}")
        if sha256_file(regression_path) != regression["sha256"]:
            raise ValueError("Historical regression corpus fingerprint mismatch")
        with regression_path.open(encoding="utf-8") as stream:
            regression_rows = sum(1 for line in stream if line.strip())
        if regression_rows != regression["records"] or regression_rows != 440:
            raise ValueError(
                f"Historical regression corpus must contain exactly 440 rows; found {regression_rows}"
            )
        regression["resolved_path"] = str(regression_path.resolve())
    for name, metadata in manifest.get("provenance_files", {}).items():
        resource_path = Path(metadata["path"])
        if not resource_path.is_absolute():
            resource_path = path.parent / resource_path
        if not resource_path.is_file() or sha256_file(resource_path) != metadata["sha256"]:
            raise ValueError(f"Provenance file fingerprint mismatch: {name}")
        metadata["resolved_path"] = str(resource_path.resolve())
    if task == "preference":
        parent = manifest.get("parent_release")
        if not parent:
            raise ValueError("Preference release is missing its parent SFT release")
        parent_path = Path(parent["manifest_path"])
        if not parent_path.is_file() or sha256_file(parent_path) != parent["manifest_sha256"]:
            raise ValueError("Preference parent manifest fingerprint mismatch")
        parent_manifest = load_manifest(parent_path)
        if (
            parent_manifest["name"] != parent["name"]
            or parent_manifest["content_sha256"] != parent["content_sha256"]
        ):
            raise ValueError("Preference parent release lineage mismatch")
        parent["resolved_manifest_path"] = str(parent_path.resolve())
    return manifest


def split_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    role: str,
    *,
    representation: str = "text-blocks",
) -> Path:
    default_representation = (
        "preference" if manifest.get("task") == "preference" else "text-blocks"
    )
    representations = manifest.get("representations") or {
        default_representation: manifest["splits"]
    }
    if representation not in representations:
        raise ValueError(f"Release has no {representation!r} representation")
    return (
        manifest_path.resolve().parent
        / representations[representation][role]["path"]
    ).resolve()
