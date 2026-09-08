#!/usr/bin/env python3
"""Promote a fully gated local model; never restarts production automatically."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import DEFAULT_TRACKING_URI  # noqa: E402
from data.manifest import load_manifest, sha256_file  # noqa: E402
from data.schema import migrate  # noqa: E402
from labeling.annotation_store import AnnotationStore, DEFAULT_DATABASE, utc_now  # noqa: E402


def update_production_alias(config: Path, model_path: Path) -> None:
    text = config.read_text(encoding="utf-8")
    section = (
        "[VoiceInk-Production]\n"
        f"model = {model_path.resolve()}\n"
        "load-on-startup = false\n"
        "chat-template-kwargs = {\"enable_thinking\":false}\n"
    )
    pattern = re.compile(r"(?ms)^\[VoiceInk-Production\]\n.*?(?=^\[|\Z)")
    updated = pattern.sub(section + "\n", text) if pattern.search(text) else text.rstrip() + "\n\n" + section
    temporary = config.with_suffix(config.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--mlflow-model-id")
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--models-config", type=Path, default=ROOT / "config/models.ini")
    parser.add_argument("--production-manifest", type=Path,
                        default=ROOT / "config/production-model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not evidence.get("passed"):
        raise SystemExit(f"Promotion refused: {evidence.get('failures')}")
    release = load_manifest(args.release_manifest)
    if evidence["dataset_release_sha256"] != release["content_sha256"]:
        raise SystemExit("Promotion evidence belongs to a different dataset release")
    if not args.model_path.is_file():
        raise SystemExit(f"Model does not exist: {args.model_path}")
    canonical_models = (ROOT / "models" / "canonical").resolve()
    try:
        args.model_path.resolve().relative_to(canonical_models)
    except ValueError as exc:
        raise SystemExit(
            f"Promotion requires a model under {canonical_models}; legacy paths are forbidden"
        ) from exc
    for kind in ("regression", "acceptance"):
        for artifact_type in ("summary", "details"):
            key = f"{kind}_{artifact_type}"
            artifact_path = Path(evidence["evidence"][key])
            expected = evidence["evidence"][f"{key}_sha256"]
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected:
                raise SystemExit(
                    f"Promotion evidence changed after verification: {key}"
                )
    store = AnnotationStore(args.database)
    migrate(store.connection)
    release_row = store.connection.execute(
        "SELECT id FROM canonical_releases WHERE name=?", (release["name"],)
    ).fetchone()
    if not release_row:
        store.close()
        raise SystemExit("Dataset release is not registered locally")
    duplicate = store.connection.execute(
        "SELECT 1 FROM model_promotions WHERE model_name=? AND model_version=?",
        (args.model_name, args.model_version),
    ).fetchone()
    if duplicate:
        store.close()
        raise SystemExit("This model name/version has already been promoted")
    import mlflow
    from mlflow.exceptions import MlflowException
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    source_run = client.get_run(args.mlflow_run_id)
    if source_run.info.status != "FINISHED":
        store.close()
        raise SystemExit(f"MLflow source run is not FINISHED: {source_run.info.status}")
    run_release = source_run.data.tags.get("voiceink.dataset_release_sha256")
    if run_release != release["content_sha256"]:
        store.close()
        raise SystemExit("MLflow run belongs to a different dataset release")
    try:
        client.create_registered_model(args.model_name, tags={
            "voiceink.dataset_release": release["name"],
            "voiceink.dataset_release_sha256": release["content_sha256"],
        })
    except MlflowException as exc:
        if "already exists" not in str(exc):
            store.close()
            raise
    model_sha256 = sha256_file(args.model_path)
    version = client.create_model_version(
        name=args.model_name, source=str(args.model_path.resolve()), run_id=args.mlflow_run_id,
        model_id=args.mlflow_model_id,
        tags={"voiceink.version": args.model_version,
              "voiceink.model_sha256": model_sha256,
              "voiceink.promotion_evidence_sha256": sha256_file(args.evidence)},
    )
    client.set_registered_model_alias(args.model_name, "candidate", version.version)
    store.connection.execute(
        "INSERT INTO model_promotions(model_name,model_version,release_id,mlflow_model_name,"
        "mlflow_model_version,regression_summary_sha256,acceptance_summary_sha256,decision_json,promoted_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (args.model_name, args.model_version, release_row["id"], args.model_name,
         str(version.version), evidence["evidence"]["regression_summary_sha256"],
         evidence["evidence"]["acceptance_summary_sha256"],
         json.dumps(evidence, sort_keys=True), utc_now()),
    )
    update_production_alias(args.models_config, args.model_path)
    production = {
        "schema": "voiceink.production-model/v1", "model_name": args.model_name,
        "model_version": args.model_version, "model_path": str(args.model_path.resolve()),
        "model_sha256": model_sha256, "dataset_release": release["name"],
        "dataset_release_sha256": release["content_sha256"],
        "mlflow_run_id": args.mlflow_run_id, "mlflow_model_version": str(version.version),
        "promotion_evidence": str(args.evidence.resolve()), "promoted_at": utc_now(),
        "restart_required": True,
    }
    temporary = args.production_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(production, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.production_manifest)
    store.close()
    print(json.dumps(production, indent=2))
    print("Promotion recorded. Restart llama-router explicitly when ready; no service was restarted.")


if __name__ == "__main__":
    main()
