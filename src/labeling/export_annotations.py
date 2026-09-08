#!/usr/bin/env python3
"""Export approved registry annotations or freeze an immutable release."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import add_mlflow_args, dataset_metadata, start_mlflow_run  # noqa: E402
from labeling.annotation_store import (  # noqa: E402
    AnnotationStore,
    DEFAULT_DATABASE,
    DEFAULT_EXPORT,
    DEFAULT_LOCKED_EVAL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved VoiceInk annotations")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--locked-eval", type=Path, default=DEFAULT_LOCKED_EVAL)
    parser.add_argument("--release-name")
    parser.add_argument("--release-dir", type=Path)
    add_mlflow_args(parser, default_experiment="voiceink-data")
    args = parser.parse_args()
    if bool(args.release_name) != bool(args.release_dir):
        parser.error("--release-name and --release-dir must be used together")
    store = AnnotationStore(args.database)
    try:
        if args.release_name:
            result = store.create_release(args.release_name, args.release_dir, args.locked_eval)
        else:
            result = store.export_approved(args.output, args.locked_eval)
        store.mirror_audit_events()
    finally:
        store.close()
    exported = Path(result["output"])
    tracking = start_mlflow_run(
        args,
        run_name=f"annotation-release-{args.release_name or 'current'}",
        run_kind="data.annotation-release",
        params={
            "release_name": args.release_name or "current",
            "records": result["records"], "sha256": result["sha256"],
            "locked_eval_sha256": result.get("locked_eval_sha256") or "missing",
        },
        datasets=[dataset_metadata("approved_annotations", exported, result["records"])],
        tags={"voiceink.raw_data_logged": "false"},
    )
    tracking.log_metrics({"records": result["records"]})
    manifest_path = (args.release_dir / "manifest.json") if args.release_dir else exported.with_name(exported.stem + "-manifest.json")
    if manifest_path.is_file():
        tracking.log_artifact(manifest_path, "data-lineage")
    tracking.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
