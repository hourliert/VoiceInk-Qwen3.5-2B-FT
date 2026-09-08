#!/usr/bin/env python3
"""Build a deduplicated SFT release from V3 replay plus reviewed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args, dataset_metadata, start_mlflow_run,
)
from data.manifest import canonical_json, load_manifest, sha256_file, split_path  # noqa: E402

DEFAULT_LEGACY_TRAIN = ROOT / "datasets/legacy/qwen35-2b-voiceink-v3/train.jsonl"
DEFAULT_PARENT = ROOT / "datasets/releases/voiceink-data-v1/manifest.json"
DEFAULT_RELEASE_ROOT = ROOT / "datasets/releases"
TAG_PATTERN = re.compile(r"<transcript>\s*(.*?)\s*</transcript>", re.I | re.S)


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    raise ValueError(f"Unsupported message content: {type(content).__name__}")


def transcript(record: dict) -> str:
    user = next(
        content_text(message["content"])
        for message in reversed(record["messages"])
        if message.get("role") == "user"
    )
    matches = TAG_PATTERN.findall(user)
    if not matches:
        raise ValueError("Training row has no <TRANSCRIPT> block")
    return matches[-1].strip()


def fingerprint(record: dict) -> str:
    value = unicodedata.normalize("NFKC", transcript(record)).casefold()
    value = " ".join(value.split())
    return hashlib.sha256(value.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def string_record(record: dict) -> dict:
    converted = dict(record)
    converted["messages"] = [
        {**message, "content": content_text(message["content"])}
        for message in record["messages"]
    ]
    return converted


def with_lineage(record: dict, *, source: str, fp: str) -> dict:
    value = dict(record)
    value.setdefault("sample_id", f"legacy-v3-{fp[:24]}")
    value["lineage"] = {
        **(value.get("lineage") or {}),
        "consolidation_source": source,
        "transcript_sha256": fp,
    }
    return value


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def consolidate_rows(
    legacy_rows: list[dict],
    reviewed_rows: list[dict],
    forbidden_fps: set[str],
) -> tuple[list[dict], dict[str, int]]:
    """Deduplicate replay data and let reviewed labels win by transcript."""
    selected: dict[str, dict] = {}
    legacy_duplicates = 0
    excluded_holdouts = 0
    for row in legacy_rows:
        fp = fingerprint(row)
        if fp in forbidden_fps:
            excluded_holdouts += 1
            continue
        if fp in selected:
            legacy_duplicates += 1
            continue
        selected[fp] = with_lineage(row, source="legacy-v3", fp=fp)

    replacements = additions = 0
    for row in reviewed_rows:
        fp = fingerprint(row)
        if fp in forbidden_fps:
            raise RuntimeError("Reviewed train row overlaps a holdout")
        if fp in selected:
            replacements += 1
        else:
            additions += 1
        selected[fp] = with_lineage(row, source="reviewed-v1", fp=fp)

    rows = [selected[key] for key in sorted(selected)]
    report = {
        "legacy_duplicates_removed": legacy_duplicates,
        "legacy_holdout_rows_removed": excluded_holdouts,
        "reviewed_replacements": replacements,
        "reviewed_additions": additions,
    }
    return rows, report


def build(args: argparse.Namespace) -> dict:
    parent = load_manifest(args.parent_manifest)
    if parent.get("task", "sft") != "sft":
        raise ValueError("Parent must be an SFT release")
    output = (args.release_root / args.name).resolve()
    if output.exists():
        raise FileExistsError(f"Release already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    legacy_rows = load_jsonl(args.legacy_train)
    new_train_path = split_path(args.parent_manifest, parent, "train")
    new_rows = load_jsonl(new_train_path)
    validation_path = split_path(args.parent_manifest, parent, "validation")
    acceptance_path = split_path(args.parent_manifest, parent, "acceptance")
    validation_rows = load_jsonl(validation_path)
    acceptance_rows = load_jsonl(acceptance_path)
    regression_path = Path(parent["historical_regression"]["resolved_path"])
    regression_rows = load_jsonl(regression_path)

    validation_fps = {fingerprint(row) for row in validation_rows}
    acceptance_fps = {fingerprint(row) for row in acceptance_rows}
    regression_fps = {fingerprint(row) for row in regression_rows}
    if validation_fps & acceptance_fps:
        raise RuntimeError("Validation and acceptance overlap")
    forbidden = validation_fps | acceptance_fps | regression_fps

    train_rows, consolidation = consolidate_rows(legacy_rows, new_rows, forbidden)
    all_train_fps = {fingerprint(row) for row in train_rows}
    if all_train_fps & forbidden:
        raise RuntimeError("Consolidated training data leaks into a holdout")

    files: dict[str, dict[str, dict]] = {"text-blocks": {}, "text": {}}
    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "acceptance": acceptance_rows,
    }
    for representation in files:
        for split, rows in split_rows.items():
            suffix = "" if representation == "text-blocks" else "-text"
            path = temporary / f"{split}{suffix}.jsonl"
            payload = rows if representation == "text-blocks" else [string_record(r) for r in rows]
            write_jsonl(path, payload)
            files[representation][split] = {
                "path": path.name, "records": len(rows), "sha256": sha256_file(path),
            }

    parent_prompt = Path(parent["system_prompt"]["resolved_path"])
    prompt_path = temporary / "system-prompt.txt"
    shutil.copy2(parent_prompt, prompt_path)
    report = {
        "legacy_input_rows": len(legacy_rows),
        "reviewed_input_rows": len(new_rows),
        **consolidation,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "acceptance_records": len(acceptance_rows),
        "train_holdout_overlap": 0,
    }
    report_path = temporary / "consolidation-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "voiceink.dataset-release/v2",
        "name": args.name,
        "created_at": datetime.now(UTC).isoformat(),
        "sealed": True,
        "task": "sft",
        "cohort": parent["cohort"],
        "selection_seed": parent["selection_seed"],
        "records": sum(len(rows) for rows in split_rows.values()),
        "expected_splits": {split: len(rows) for split, rows in split_rows.items()},
        "splits": files["text-blocks"],
        "representations": files,
        "system_prompt": {"path": prompt_path.name, "sha256": sha256_file(prompt_path)},
        "historical_regression": {
            "path": str(regression_path.resolve()), "records": len(regression_rows),
            "sha256": sha256_file(regression_path), "checkpoint_selection": False,
        },
        "parent_release": {
            "name": parent["name"],
            "manifest_path": str(args.parent_manifest.resolve()),
            "manifest_sha256": sha256_file(args.parent_manifest),
            "content_sha256": parent["content_sha256"],
        },
        "sources": {
            "legacy_v3_train": {
                "path": str(args.legacy_train.resolve()),
                "records": len(legacy_rows), "sha256": sha256_file(args.legacy_train),
            },
            "reviewed_release": {
                "path": str(args.parent_manifest.resolve()),
                "sha256": sha256_file(args.parent_manifest),
            },
        },
        "provenance_files": {
            "consolidation_report": {
                "path": report_path.name, "sha256": sha256_file(report_path),
            }
        },
        "privacy": "private local files; MLflow logs fingerprints and counts only",
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return manifest | {"manifest_path": str(output / "manifest.json"), "report": report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--legacy-train", type=Path, default=DEFAULT_LEGACY_TRAIN)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    add_mlflow_args(parser, default_experiment="voiceink-data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    manifest_path = Path(result["manifest_path"])
    tracking = start_mlflow_run(
        args, run_name=f"dataset-consolidation-{args.name}",
        run_kind="data.consolidation",
        params={"release": args.name, **result["report"]},
        datasets=[
            dataset_metadata(split, manifest_path.parent / item["path"], item["records"])
            for split, item in result["splits"].items()
        ],
        tags={
            "voiceink.dataset_release": args.name,
            "voiceink.dataset_release_sha256": result["content_sha256"],
        },
    )
    tracking.log_metrics(result["report"])
    tracking.log_artifact(manifest_path, artifact_path="lineage/dataset-release")
    tracking.log_artifact(manifest_path.parent / "consolidation-report.json", artifact_path="lineage")
    tracking.finish()
    print(json.dumps({"manifest": str(manifest_path), **result["report"]}, indent=2))


if __name__ == "__main__":
    main()
