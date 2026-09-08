"""Load and verify immutable named evaluation suites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.consolidate_release import fingerprint, load_jsonl
from data.manifest import canonical_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = ROOT / "config" / "evaluation_suites"


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_suite(name: str) -> dict:
    """Verify a named suite's data fingerprint and declared train exclusions."""
    manifest_path = SUITE_ROOT / f"{name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "voiceink.evaluation-suite/v1":
        raise ValueError(f"Unsupported evaluation suite schema: {manifest.get('schema')!r}")
    if manifest.get("name") != name or not manifest.get("sealed"):
        raise ValueError("Evaluation suite must be sealed and match its filename")
    expected_digest = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    actual_digest = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("Evaluation suite manifest fingerprint mismatch")

    data_path = _resolve(manifest_path, manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("Evaluation suite data fingerprint mismatch")
    eval_rows = load_jsonl(data_path)
    eval_fps = {fingerprint(row) for row in eval_rows}
    if len(eval_rows) != manifest["data"]["records"] or len(eval_fps) != len(eval_rows):
        raise ValueError("Evaluation suite count or uniqueness check failed")

    audits = []
    for exclusion in manifest.get("training_exclusions", []):
        train_path = _resolve(manifest_path, exclusion["path"])
        if sha256_file(train_path) != exclusion["sha256"]:
            raise ValueError(f"Training corpus fingerprint mismatch: {exclusion['name']}")
        train_fps = {fingerprint(row) for row in load_jsonl(train_path)}
        overlap = len(eval_fps & train_fps)
        if overlap != exclusion.get("expected_overlap", 0):
            raise ValueError(
                f"Evaluation leakage for {exclusion['name']}: expected "
                f"{exclusion.get('expected_overlap', 0)}, found {overlap}"
            )
        audits.append({"name": exclusion["name"], "overlap": overlap})

    manifest["resolved_path"] = str(manifest_path.resolve())
    manifest["data"]["resolved_path"] = str(data_path)
    manifest["overlap_audits"] = audits
    return manifest
