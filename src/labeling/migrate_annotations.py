#!/usr/bin/env python3
"""Non-destructively backfill historical VoiceInk datasets into the registry."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import add_mlflow_args, start_mlflow_run  # noqa: E402
from labeling.annotation_store import (  # noqa: E402
    AnnotationStore,
    DEFAULT_DATABASE,
    file_sha256,
    read_jsonl,
    utc_now,
)

DEFAULT_REPORT = ROOT / "datasets" / "registry" / "migration-report.json"


def classify(path: Path) -> str:
    try:
        value = str(path.relative_to(ROOT / "datasets"))
    except ValueError:
        value = path.name
    if "synthetic" in value:
        return "synthetic-labels"
    if "calibration" in value:
        return "calibration"
    if "audit" in value:
        return "audit"
    if "strategic" in value:
        return "strategic"
    if "dpo" in value:
        return "preference-data"
    if path.name.startswith("labeled"):
        return "historical-labels"
    if "eval" in path.name:
        return "evaluation-corpus"
    if path.name == "train.jsonl":
        return "prepared-training-corpus"
    return "derived-or-manifest"


def inventory(path: Path) -> dict:
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    rows = malformed = labels = request_ids = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                labels += bool(record.get("label"))
                request_ids += bool(record.get("request_id"))
    return {
        "path": display_path, "sha256": file_sha256(path),
        "bytes": path.stat().st_size, "role": classify(path), "rows": rows,
        "malformed": malformed, "labels": labels, "request_ids": request_ids,
    }


def provider_fields(record: dict) -> tuple[str, str, str, str, dict]:
    metadata = record.get("label_provider_metadata") or {}
    provider = str(record.get("label_provider") or metadata.get("provider") or "legacy")
    model = str(record.get("model_used_for_label") or metadata.get("model") or "")
    effort = str(metadata.get("reasoning_effort") or "")
    prompt_sha = str(metadata.get("prompt_sha256") or "")
    return provider, model, effort, prompt_sha, metadata


def approved_outcome(record: dict) -> str | None:
    request_id = str(record.get("request_id", ""))
    if request_id.startswith("syn-"):
        return "accept_luna"
    manual = record.get("manual_review") or {}
    if manual.get("status") == "rejected":
        return "exclude"
    if manual.get("status") == "edited":
        return "human_edit"
    if record.get("manually_reviewed") or manual.get("status") == "approved":
        return "accept_luna"
    if (record.get("auto_review") or {}).get("status") == "approved":
        return "accept_luna"
    return None


def import_file(store: AnnotationStore, path: Path, meta: dict) -> Counter:
    counts: Counter = Counter()
    source_id, created = store.register_source(
        path, meta["role"], meta["rows"], meta["malformed"]
    )
    if not created:
        counts["already_imported_files"] += 1
        return counts
    counts["imported_files"] += 1
    if meta["labels"] == 0:
        counts["registered_derived_rows"] += meta["rows"]
        return counts
    for line_number, record in read_jsonl(path):
        if not record.get("request_id") or not record.get("label"):
            continue
        proxy_record = {
            "request_id": record["request_id"],
            "timestamp": record.get("timestamp", "") or utc_now(),
            "raw_request_json": record.get("raw_request_json", ""),
            "raw_response_json": record.get("raw_response_json", ""),
            "original_response": record.get("current_model_response")
            or record.get("original_response", ""),
            "original_model": record.get("current_model")
            or record.get("original_model", ""),
        }
        if not store.import_proxy_record(
            proxy_record, source_file_id=source_id, source_line=line_number
        ):
            counts["labels_without_importable_request"] += 1
            continue
        request_id = str(record["request_id"])
        provider, model, effort, prompt_sha, provider_metadata = provider_fields(record)
        if request_id.startswith("syn-"):
            origin = "synthetic"
        elif provider == "codex" or "luna" in model.casefold():
            origin = "luna"
        else:
            origin = "legacy"
        annotation_id = store.create_annotation(
            request_id, str(record["label"]), origin=origin, provider=provider,
            model=model, reasoning_effort=effort, prompt_sha256=prompt_sha,
            metadata={
                "legacy_provider_metadata": provider_metadata,
                "validation": record.get("validation"),
                "reference_model": record.get("reference_model_used_for_label"),
            },
            source_file_id=source_id, source_line=line_number,
            created_at=(record.get("manual_review") or {}).get("timestamp")
            or record.get("timestamp") or utc_now(),
        )
        counts["annotations"] += 1
        outcome = approved_outcome(record)
        if outcome:
            current = store.sample(request_id).get("decision")
            # Later source files deliberately win. Identical current decisions
            # are left untouched to keep reruns and duplicate snapshots quiet.
            if not current or int(current.get("annotation_id") or 0) != annotation_id \
                    or current.get("outcome") != outcome:
                store.decide(
                    request_id, outcome,
                    annotation_id=None if outcome == "exclude" else annotation_id,
                    reviewer="historical-import",
                    context_dependency="unknown",
                )
                counts["decisions"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill all historical annotation data")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    add_mlflow_args(parser, default_experiment="voiceink-data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.datasets_root.rglob("*.jsonl"), key=lambda path: (
        path.name == "labeled.jsonl", "strategic" in str(path), str(path)
    ))
    # Never treat registry exports as historical input.
    paths = [path for path in paths if "registry" not in path.parts and "exports" not in path.parts]
    files = [inventory(path) for path in paths]
    report = {
        "created_at": utc_now(), "dry_run": args.dry_run,
        "datasets_root": str(args.datasets_root), "files": files,
        "totals": {
            "files": len(files), "rows": sum(item["rows"] for item in files),
            "malformed": sum(item["malformed"] for item in files),
            "labeled_rows": sum(item["labels"] for item in files),
        },
    }
    if not args.dry_run:
        store = AnnotationStore(args.database)
        counts: Counter = Counter()
        try:
            for path, meta in zip(paths, files):
                counts.update(import_file(store, path, meta))
            report["import"] = dict(counts)
            report["registry"] = store.stats()
            report["integrity"] = store.integrity_check()
            store.mirror_audit_events()
            report["backup"] = str(store.backup())
        finally:
            store.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tracking = start_mlflow_run(
        args, run_name="annotation-registry-backfill", run_kind="data.annotation-migration",
        params={
            "dry_run": args.dry_run, "files": report["totals"]["files"],
            "rows": report["totals"]["rows"],
            "labeled_rows": report["totals"]["labeled_rows"],
            "malformed": report["totals"]["malformed"],
        },
        tags={"voiceink.raw_data_logged": "false"},
    )
    tracking.log_metrics(report.get("registry", {}) | report["totals"])
    tracking.log_artifact(args.report, "data-lineage")
    tracking.finish()
    print("VoiceInk annotation migration")
    print(f"  Files: {report['totals']['files']}; rows: {report['totals']['rows']}")
    print(f"  Labeled rows: {report['totals']['labeled_rows']}; malformed: {report['totals']['malformed']}")
    if not args.dry_run:
        print(f"  Registry: {report['registry']}")
        print(f"  Integrity: {report['integrity']}")
    print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
