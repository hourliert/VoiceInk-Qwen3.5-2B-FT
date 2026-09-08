#!/usr/bin/env python3
"""Canonical VoiceInk data CLI: bootstrap, verify, release, status, and backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import canonical_json, load_manifest, sha256_file  # noqa: E402
from data.cohort_review import cohort_status as canonical_cohort_status  # noqa: E402
from data.review_policy import (  # noqa: E402
    is_fresh_human_decision,
    resolved_annotation_id,
)
from data.schema import migrate  # noqa: E402
from labeling.annotation_store import (  # noqa: E402
    AnnotationStore,
    DEFAULT_DATABASE,
    DEFAULT_LOCKED_EVAL,
    locked_transcript_fingerprints,
    normalize_transcript,
    utc_now,
)
from training.prepare_dataset import convert_record  # noqa: E402

DEFAULT_SEED = 20260817
DEFAULT_RELEASE_ROOT = ROOT / "datasets" / "releases"
DEFAULT_SYSTEM_PROMPT = ROOT / "docs" / "VOICEINK_PROMPT"
DEFAULT_HISTORICAL_REGRESSION = DEFAULT_LOCKED_EVAL
AUDIT_PERCENT = 2

BOOTSTRAP_SPEC = {
    "representative": {"train": 315, "validation": 53, "acceptance": 52},
    "context-rich": {"train": 180, "validation": 30, "acceptance": 30},
    "technical-risk": {"train": 180, "validation": 30, "acceptance": 30},
    "long": {"train": 135, "validation": 22, "acceptance": 23},
    "no-op": {"train": 90, "validation": 15, "acceptance": 15},
}
EXPECTED_SPLITS = {"train": 900, "validation": 150, "acceptance": 150}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TECHNICAL_TERMS = re.compile(
    r"\b(?:API|CLI|JSON|SQL|Python|Swift|GitHub|Git|MLflow|Luna|Qwen|LFM|"
    r"Parakeet|CoreML|GGUF|LoRA|DPO|SFT|MCP|Betfair|Monza|Le Mans|Nürburgring|"
    r"[A-Z]{2,}[A-Z0-9._-]*|\d+(?:\.\d+)?(?:B|GB|MB|ms|%)?)\b"
)


def _safe_name(value: str, kind: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid {kind} name: {value!r}")
    return value


def _connection(path: Path) -> tuple[AnnotationStore, sqlite3.Connection]:
    store = AnnotationStore(path)
    migrate(store.connection)
    return store, store.connection


def _stable_rank(seed: int, request_id: str, salt: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{request_id}".encode()).hexdigest()


def _audit_selected(seed: int, request_id: str) -> int:
    return int(int(_stable_rank(seed, request_id, "audit")[:8], 16) % 100 < AUDIT_PERCENT)


def _eligible_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    excluded = locked_transcript_fingerprints()
    rows = connection.execute(
        "SELECT s.*,a.id annotation_id,a.text label,a.origin,d.reviewer,d.decided_at "
        "FROM decisions d JOIN annotations a ON a.id=d.annotation_id "
        "JOIN samples s ON s.request_id=d.request_id "
        "WHERE d.is_current=1 AND d.training_eligible=1"
    )
    seen: set[str] = set()
    result = []
    for row in rows:
        item = dict(row)
        if item["origin"] == "synthetic":
            continue
        fingerprint = item["transcript_fingerprint"]
        if fingerprint in excluded or fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(item)
    return result


def _matches(row: dict[str, Any], stratum: str) -> bool:
    transcript = row["transcript"].strip()
    if stratum == "no-op":
        return normalize_transcript(transcript) == normalize_transcript(row["label"])
    if stratum == "long":
        return len(transcript.split()) >= 200
    if stratum == "context-rich":
        return len((row["window_context"] or "").split()) >= 800
    if stratum == "technical-risk":
        return bool(TECHNICAL_TERMS.search(transcript))
    return True


def select_bootstrap(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Select the exact approved 1,200-row stratified cohort deterministically."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    synthetic = 0
    # Allocate scarce strata first; representative is the deterministic remainder.
    for stratum in ("no-op", "long", "technical-risk", "context-rich", "representative"):
        candidates = [
            row for row in rows
            if row["request_id"] not in used and _matches(row, stratum)
        ]
        candidates.sort(key=lambda row: _stable_rank(seed, row["request_id"], stratum))
        needed = sum(BOOTSTRAP_SPEC[stratum].values())
        picked = []
        for row in candidates:
            is_synthetic = row["origin"] == "synthetic"
            if is_synthetic and synthetic >= 90:
                continue
            picked.append(row)
            synthetic += int(is_synthetic)
            if len(picked) == needed:
                break
        if len(picked) != needed:
            raise RuntimeError(
                f"Insufficient eligible rows for {stratum}: need {needed}, found {len(picked)}"
            )
        cursor = 0
        for split in ("train", "validation", "acceptance"):
            count = BOOTSTRAP_SPEC[stratum][split]
            for row in picked[cursor:cursor + count]:
                selected.append(row | {"split": split, "stratum": stratum})
                used.add(row["request_id"])
            cursor += count
    counts = Counter(row["split"] for row in selected)
    if dict(counts) != EXPECTED_SPLITS:
        raise AssertionError(f"Bootstrap split invariant failed: {dict(counts)}")
    if len({row["transcript_fingerprint"] for row in selected}) != 1200:
        raise AssertionError("Bootstrap contains duplicate normalized transcripts")
    return selected


def bootstrap(connection: sqlite3.Connection, name: str, seed: int, output: Path) -> dict:
    _safe_name(name, "cohort")
    existing = connection.execute(
        "SELECT id FROM canonical_cohorts WHERE name=?", (name,)
    ).fetchone()
    if existing:
        raise ValueError(f"Cohort already exists: {name}")
    rows = select_bootstrap(_eligible_rows(connection), seed)
    now = utc_now()
    specification = {
        "schema": "voiceink.bootstrap-spec/v1",
        "seed": seed,
        "total": 1200,
        "splits": EXPECTED_SPLITS,
        "strata": BOOTSTRAP_SPEC,
        "fresh_luna_validation": "required-for-all",
        "human_review": (
            "2% deterministic audit; ambiguous or unsafe Luna proposal wins; "
            "safe production wins and validation-passing non-material cases are automatic"
        ),
        "synthetic": "train-only; maximum 90",
        "historical_440": "regression-only; excluded from selection",
    }
    connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = connection.execute(
            "INSERT INTO canonical_cohorts(name,state,seed,specification_json,created_at,updated_at) "
            "VALUES(?,'selected',?,?,?,?)",
            (name, seed, canonical_json(specification), now, now),
        )
        cohort_id = int(cursor.lastrowid)
        for row in rows:
            audit_selected = _audit_selected(seed, row["request_id"])
            connection.execute(
                "INSERT INTO canonical_cohort_members(cohort_id,request_id,annotation_id,split,stratum,audit_selected) "
                "VALUES(?,?,?,?,?,?)",
                (cohort_id, row["request_id"], row["annotation_id"], row["split"],
                 row["stratum"], audit_selected),
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({
                "request_id": row["request_id"],
                "split": row["split"],
                "stratum": row["stratum"],
                "action": "open in live review; fresh Luna analysis starts automatically",
            }) + "\n")
    return {"cohort": name, "cohort_id": cohort_id, "selected": 1200,
            "review_queue": str(output), "specification": specification}


def _cohort_row(connection: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM canonical_cohorts WHERE name=?", (name,)).fetchone()
    if not row:
        raise ValueError(f"Unknown cohort: {name}")
    return row


def apply_review_policy(connection: sqlite3.Connection, name: str) -> dict[str, Any]:
    """Apply the current deterministic audit policy to an unreleased cohort."""
    cohort = _cohort_row(connection, name)
    if cohort["state"] == "released":
        raise RuntimeError(f"Cannot change review policy for released cohort: {name}")
    members = list(connection.execute(
        "SELECT request_id,audit_selected FROM canonical_cohort_members WHERE cohort_id=?",
        (cohort["id"],),
    ))
    previous = sum(int(member["audit_selected"]) for member in members)
    specification = json.loads(cohort["specification_json"])
    specification["human_review"] = (
        "2% deterministic audit; ambiguous or unsafe Luna proposal wins; "
        "safe production wins and validation-passing non-material cases are automatic"
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        for member in members:
            connection.execute(
                "UPDATE canonical_cohort_members SET audit_selected=? "
                "WHERE cohort_id=? AND request_id=?",
                (_audit_selected(cohort["seed"], member["request_id"]),
                 cohort["id"], member["request_id"]),
            )
        connection.execute(
            "UPDATE canonical_cohorts SET specification_json=?,updated_at=? WHERE id=?",
            (canonical_json(specification), utc_now(), cohort["id"]),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    selected = connection.execute(
        "SELECT COUNT(*) FROM canonical_cohort_members "
        "WHERE cohort_id=? AND audit_selected=1", (cohort["id"],)
    ).fetchone()[0]
    preserved_human = connection.execute(
        "SELECT COUNT(*) FROM canonical_cohort_members m JOIN decisions d "
        "ON d.request_id=m.request_id AND d.is_current=1 "
        "WHERE m.cohort_id=? AND d.reviewer='human' AND d.decided_at>=?",
        (cohort["id"], cohort["created_at"]),
    ).fetchone()[0]
    return {
        "cohort": name, "previous_audit": previous, "selected_audit": selected,
        "preserved_human_decisions": preserved_human,
    }


def cohort_status(connection: sqlite3.Connection, name: str) -> dict[str, Any]:
    """Compatibility entry point backed by the canonical set-based projection."""
    return canonical_cohort_status(connection, name)


def _annotation_record(
    connection: sqlite3.Connection, cohort: sqlite3.Row, member: sqlite3.Row
) -> dict[str, Any]:
    analysis = connection.execute(
        "SELECT * FROM analyses WHERE request_id=? AND created_at>=? "
        "AND lower(model) LIKE '%luna%' ORDER BY id DESC LIMIT 1",
        (member["request_id"], cohort["created_at"]),
    ).fetchone()
    decision = connection.execute(
        "SELECT * FROM decisions WHERE request_id=? AND is_current=1",
        (member["request_id"],),
    ).fetchone()
    if is_fresh_human_decision(decision, cohort["created_at"]) and not decision["training_eligible"]:
        raise RuntimeError(f"Human-excluded cohort member: {member['request_id']}")
    annotation_id, resolution = resolved_annotation_id(
        member, analysis, decision, cohort["created_at"]
    )
    row = connection.execute(
        "SELECT s.*,a.text label,a.origin,a.id annotation_id "
        "FROM samples s JOIN annotations a ON a.id=? WHERE s.request_id=?",
        (annotation_id, member["request_id"]),
    ).fetchone()
    if not row:
        raise RuntimeError(f"Missing fixed annotation for {member['request_id']}")
    manual_review = None
    if is_fresh_human_decision(decision, cohort["created_at"]):
        manual_review = {
            "reviewer": decision["reviewer"], "outcome": decision["outcome"],
            "timestamp": decision["decided_at"],
        }
    return dict(row) | {
        "label_resolution": resolution,
        "manual_review": manual_review,
    }


def release_preview(connection: sqlite3.Connection, cohort_name: str,
                    correction_cutoff: str | None = None) -> dict[str, Any]:
    """Return a deterministic, hash-addressed release selection."""
    cohort = _cohort_row(connection, cohort_name)
    cutoff = correction_cutoff or utc_now()
    cohort_fingerprints = {row[0] for row in connection.execute(
        "SELECT s.transcript_fingerprint FROM canonical_cohort_members m "
        "JOIN samples s ON s.request_id=m.request_id WHERE m.cohort_id=?",
        (cohort["id"],),
    )}
    excluded = locked_transcript_fingerprints()
    seen = set(cohort_fingerprints)
    additions = []
    rows = connection.execute(
        "SELECT s.request_id,s.transcript_fingerprint,d.annotation_id,d.decided_at "
        "FROM decisions d JOIN samples s ON s.request_id=d.request_id "
        "WHERE d.is_current=1 AND d.training_eligible=1 AND d.reviewer='human' "
        "AND d.decided_at>=? AND d.decided_at<=? AND NOT EXISTS (SELECT 1 FROM "
        "canonical_cohort_members m WHERE m.cohort_id=? AND m.request_id=s.request_id) "
        "ORDER BY d.decided_at,s.request_id",
        (cohort["created_at"], cutoff, cohort["id"]),
    )
    for row in rows:
        fingerprint = row["transcript_fingerprint"]
        if fingerprint in excluded or fingerprint in seen:
            continue
        seen.add(fingerprint)
        additions.append({
            "request_id": row["request_id"], "annotation_id": row["annotation_id"],
            "decided_at": row["decided_at"],
        })
    preview = {
        "cohort": cohort_name, "correction_cutoff": cutoff,
        "additions": additions,
        "splits": {"train": 900 + len(additions), "validation": 150, "acceptance": 150},
        "records": 1200 + len(additions),
    }
    preview["selection_sha256"] = hashlib.sha256(
        canonical_json(preview).encode()
    ).hexdigest()
    return preview


def create_release(connection: sqlite3.Connection, cohort_name: str, release_name: str,
                   release_root: Path, system_prompt_path: Path,
                   parent: str | None = None, correction_cutoff: str | None = None,
                   selection_sha256: str | None = None) -> dict[str, Any]:
    _safe_name(cohort_name, "cohort")
    _safe_name(release_name, "release")
    status = cohort_status(connection, cohort_name)
    if not status["ready"]:
        raise RuntimeError(f"Cohort is not releasable: {json.dumps(status, sort_keys=True)}")
    cohort = _cohort_row(connection, cohort_name)
    preview = release_preview(connection, cohort_name, correction_cutoff)
    if selection_sha256 and selection_sha256 != preview["selection_sha256"]:
        raise RuntimeError("Release selection changed; refresh the preview before sealing")
    if connection.execute("SELECT 1 FROM canonical_releases WHERE name=?", (release_name,)).fetchone():
        raise ValueError(f"Release already exists: {release_name}")
    final_dir = (release_root / release_name).resolve()
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = final_dir.with_name(final_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    prompt_snapshot = temporary / "system-prompt.txt"
    prompt_snapshot.write_text(system_prompt, encoding="utf-8")
    members = list(connection.execute(
        "SELECT * FROM canonical_cohort_members WHERE cohort_id=? ORDER BY split,stratum,request_id",
        (cohort["id"],),
    ))
    members.extend({
        "request_id": item["request_id"], "annotation_id": item["annotation_id"],
        "split": "train", "stratum": "human-correction-delta",
    } for item in preview["additions"])
    resolved_sources: dict[str, dict[str, Any]] = {}
    representations: dict[str, dict[str, Any]] = {
        "text-blocks": {},
        "text": {},
    }
    synthetic_train = 0
    for split in ("train", "validation", "acceptance"):
        paths = {
            "text-blocks": temporary / f"{split}.jsonl",
            "text": temporary / f"{split}-text.jsonl",
        }
        count = 0
        with (
            paths["text-blocks"].open("w", encoding="utf-8") as blocks_stream,
            paths["text"].open("w", encoding="utf-8") as text_stream,
        ):
            for member in (item for item in members if item["split"] == split):
                if member["stratum"] == "human-correction-delta":
                    row = connection.execute(
                        "SELECT s.*,a.text label,a.origin,a.id annotation_id FROM samples s "
                        "JOIN annotations a ON a.id=? WHERE s.request_id=?",
                        (member["annotation_id"], member["request_id"]),
                    ).fetchone()
                    if not row:
                        raise RuntimeError(f"Missing correction annotation for {member['request_id']}")
                    source = dict(row) | {
                        "label_resolution": "human-correction-delta",
                        "manual_review": {"reviewer": "human"},
                    }
                else:
                    source = _annotation_record(connection, cohort, member)
                resolved_sources[member["request_id"]] = source
                if source["origin"] == "synthetic":
                    if split != "train":
                        raise RuntimeError("Synthetic rows are forbidden outside the train split")
                    synthetic_train += 1
                conversations = {
                    "text-blocks": convert_record(
                        source, system_prompt, "text-blocks", "voiceink"
                    ),
                    "text": convert_record(source, system_prompt, "string", "voiceink"),
                }
                if any(conversation is None for conversation in conversations.values()):
                    raise RuntimeError(f"Could not convert {member['request_id']}")
                lineage = {
                    "annotation_id": source["annotation_id"],
                    "stratum": member["stratum"],
                    "split": split,
                    "origin": source["origin"],
                    "label_resolution": source["label_resolution"],
                    "manual_review": source["manual_review"],
                }
                for conversation in conversations.values():
                    conversation["sample_id"] = member["request_id"]
                    conversation["lineage"] = lineage
                blocks_stream.write(
                    json.dumps(conversations["text-blocks"], ensure_ascii=False) + "\n"
                )
                text_stream.write(
                    json.dumps(conversations["text"], ensure_ascii=False) + "\n"
                )
                count += 1
        for representation, path in paths.items():
            representations[representation][split] = {
                "path": path.name,
                "records": count,
                "sha256": sha256_file(path),
            }
    if synthetic_train > 90:
        raise RuntimeError(f"Synthetic train cap exceeded: {synthetic_train} > 90")
    regression = DEFAULT_HISTORICAL_REGRESSION
    manifest: dict[str, Any] = {
        "schema": "voiceink.dataset-release/v2", "name": release_name,
        "created_at": utc_now(), "sealed": True, "cohort": cohort_name,
        "selection_seed": cohort["seed"], "records": preview["records"],
        "expected_splits": preview["splits"], "selection": preview,
        "splits": representations["text-blocks"],
        "representations": representations,
        "strata": BOOTSTRAP_SPEC,
        "system_prompt": {
            "path": prompt_snapshot.name,
            "sha256": sha256_file(prompt_snapshot),
            "source_path": str(system_prompt_path.resolve()),
        },
        "historical_regression": {
            "path": str(regression.resolve()), "records": 440,
            "sha256": sha256_file(regression), "checkpoint_selection": False,
        },
        "parent_release": parent, "synthetic_train_records": synthetic_train,
        "privacy": "private local files; MLflow logs fingerprints and counts only",
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final_dir)
    manifest_path = final_dir / "manifest.json"
    parent_row = connection.execute(
        "SELECT id FROM canonical_releases WHERE name=?", (parent,)
    ).fetchone() if parent else None
    if parent and not parent_row:
        shutil.rmtree(final_dir)
        raise ValueError(f"Unknown parent release: {parent}")
    connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = connection.execute(
            "INSERT INTO canonical_releases(name,cohort_id,parent_release_id,manifest_sha256,manifest_path,records,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (release_name, cohort["id"], parent_row["id"] if parent_row else None,
             sha256_file(manifest_path), str(manifest_path), preview["records"], utc_now()),
        )
        release_id = int(cursor.lastrowid)
        for member in members:
            connection.execute(
                "INSERT INTO canonical_release_members(release_id,request_id,annotation_id,split,stratum) "
                "VALUES(?,?,?,?,?)",
                (release_id, member["request_id"],
                 resolved_sources[member["request_id"]]["annotation_id"],
                 member["split"], member["stratum"]),
            )
        connection.execute(
            "UPDATE canonical_cohorts SET state=?,updated_at=? WHERE id=?",
            ("released", utc_now(), cohort["id"]),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        shutil.rmtree(final_dir)
        raise
    for path in final_dir.iterdir():
        path.chmod(0o444)
    final_dir.chmod(0o555)
    return manifest | {"release_id": release_id, "manifest_path": str(manifest_path)}


def archive_legacy(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "voiceink.legacy-training-run/v1", "archived_at": utc_now(),
        "mlflow_run_id": args.mlflow_run_id, "model": args.model,
        "training_data": {"path": str(args.train.resolve()), "sha256": sha256_file(args.train),
                          "records": sum(1 for line in args.train.open() if line.strip())},
        "validation_data": {"path": str(args.eval.resolve()), "sha256": sha256_file(args.eval),
                            "records": sum(1 for line in args.eval.open() if line.strip())},
        "canonical": False, "reason": "predates canonical dataset release v2",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload | {"output": str(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    sub = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = sub.add_parser("bootstrap")
    bootstrap_parser.add_argument("name")
    bootstrap_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    bootstrap_parser.add_argument("--review-queue", type=Path,
                                  default=ROOT / "datasets" / "registry" / "bootstrap-review-queue.jsonl")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("name", nargs="?")
    policy_parser = sub.add_parser("apply-review-policy")
    policy_parser.add_argument("name")
    release_parser = sub.add_parser("release")
    release_parser.add_argument("cohort")
    release_parser.add_argument("name")
    release_parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    release_parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    release_parser.add_argument("--parent")
    release_parser.add_argument("--correction-cutoff")
    release_parser.add_argument("--selection-sha256")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    sub.add_parser("backup")
    archive_parser = sub.add_parser("archive-legacy")
    archive_parser.add_argument("--mlflow-run-id", required=True)
    archive_parser.add_argument("--model", required=True)
    archive_parser.add_argument("--train", type=Path, required=True)
    archive_parser.add_argument("--eval", type=Path, required=True)
    archive_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        manifest = load_manifest(args.manifest)
        print(json.dumps({"valid": True, "name": manifest["name"], "splits": manifest["splits"]}, indent=2))
        return
    if args.command == "archive-legacy":
        print(json.dumps(archive_legacy(args), indent=2))
        return
    store, connection = _connection(args.database)
    try:
        if args.command == "bootstrap":
            result = bootstrap(connection, args.name, args.seed, args.review_queue)
        elif args.command == "apply-review-policy":
            result = apply_review_policy(connection, args.name)
            result["status"] = cohort_status(connection, args.name)
        elif args.command == "status":
            if args.name:
                result = cohort_status(connection, args.name)
            else:
                result = {"registry": store.stats(), "cohorts": [dict(row) for row in connection.execute(
                    "SELECT id,name,state,seed,created_at,updated_at FROM canonical_cohorts ORDER BY id DESC"
                )], "releases": [dict(row) for row in connection.execute(
                    "SELECT id,name,manifest_sha256,manifest_path,records,created_at FROM canonical_releases ORDER BY id DESC"
                )]}
        elif args.command == "release":
            result = create_release(connection, args.cohort, args.name, args.release_root,
                                    args.system_prompt, args.parent, args.correction_cutoff,
                                    args.selection_sha256)
        elif args.command == "backup":
            result = {"annotation_database": str(store.backup()),
                      "audit_events_mirrored": store.mirror_audit_events()}
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
