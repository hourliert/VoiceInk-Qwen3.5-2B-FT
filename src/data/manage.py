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
        "human_review": "all acceptance; material/uncertain/conflict; 10% deterministic audit",
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
            audit_selected = int(
                row["split"] != "acceptance"
                and int(_stable_rank(seed, row["request_id"], "audit")[:8], 16) % 10 == 0
            )
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


def cohort_status(connection: sqlite3.Connection, name: str) -> dict[str, Any]:
    cohort = _cohort_row(connection, name)
    members = list(connection.execute(
        "SELECT m.*,s.transcript_fingerprint FROM canonical_cohort_members m "
        "JOIN samples s ON s.request_id=m.request_id WHERE m.cohort_id=?",
        (cohort["id"],),
    ))
    fresh_luna = human = required_human = critical = 0
    split_counts = Counter()
    stratum_counts = Counter()
    for member in members:
        split_counts[member["split"]] += 1
        stratum_counts[(member["stratum"], member["split"])] += 1
        analysis = connection.execute(
            "SELECT * FROM analyses WHERE request_id=? AND created_at>=? "
            "AND lower(model) LIKE '%luna%' ORDER BY id DESC LIMIT 1",
            (member["request_id"], cohort["created_at"]),
        ).fetchone()
        decision = connection.execute(
            "SELECT * FROM decisions WHERE request_id=? AND is_current=1",
            (member["request_id"],),
        ).fetchone()
        is_human = bool(
            decision and decision["reviewer"] == "human"
            and decision["decided_at"] >= cohort["created_at"]
        )
        fresh_luna += int(analysis is not None)
        human += int(is_human)
        flagged = bool(
            member["split"] == "acceptance" or member["audit_selected"]
            or (analysis and (
                analysis["material_difference"]
                or analysis["preference"] == "production"
                or analysis["confidence"] != "high"
                or analysis["validation_status"] not in {"valid", "pass", "approved"}
            ))
        )
        required_human += int(flagged)
        if flagged and not is_human:
            critical += int(
                analysis is not None
                and analysis["validation_status"] not in {"valid", "pass", "approved"}
            )
        if is_human and decision["annotation_id"] != member["annotation_id"]:
            connection.execute(
                "UPDATE canonical_cohort_members SET annotation_id=? WHERE cohort_id=? AND request_id=?",
                (decision["annotation_id"], cohort["id"], member["request_id"]),
            )
    audit_stats: dict[str, list[int]] = {}
    for member in members:
        if not member["audit_selected"]:
            continue
        decision = connection.execute(
            "SELECT * FROM decisions WHERE request_id=? AND is_current=1 AND reviewer='human' AND decided_at>=?",
            (member["request_id"], cohort["created_at"]),
        ).fetchone()
        if not decision:
            continue
        luna = connection.execute(
            "SELECT text FROM annotations WHERE request_id=? AND origin='luna' AND created_at>=? ORDER BY id DESC LIMIT 1",
            (member["request_id"], cohort["created_at"]),
        ).fetchone()
        current = connection.execute(
            "SELECT text FROM annotations WHERE id=?", (decision["annotation_id"],)
        ).fetchone()
        if luna and current:
            values = audit_stats.setdefault(member["stratum"], [0, 0])
            values[0] += 1
            values[1] += int(normalize_transcript(luna["text"]) != normalize_transcript(current["text"]))
    expanded_strata = {
        stratum for stratum, (reviewed, errors) in audit_stats.items()
        if reviewed and errors / reviewed > 0.02
    }
    for member in members:
        if member["stratum"] not in expanded_strata:
            continue
        analysis = connection.execute(
            "SELECT * FROM analyses WHERE request_id=? AND created_at>=? AND lower(model) LIKE '%luna%' ORDER BY id DESC LIMIT 1",
            (member["request_id"], cohort["created_at"]),
        ).fetchone()
        already_required = bool(member["split"] == "acceptance" or member["audit_selected"] or (analysis and (analysis["material_difference"]
            or analysis["preference"] == "production"
            or analysis["confidence"] != "high" or analysis["validation_status"] not in {"valid", "pass", "approved"})))
        required_human += int(not already_required)
    required_human_done = sum(1 for member in members if connection.execute(
        "SELECT 1 FROM decisions WHERE request_id=? AND is_current=1 AND reviewer='human' AND decided_at>=?",
        (member["request_id"], cohort["created_at"]),
    ).fetchone() and (
        member["split"] == "acceptance" or member["audit_selected"] or member["stratum"] in expanded_strata or connection.execute(
            "SELECT 1 FROM analyses WHERE request_id=? AND created_at>=? AND lower(model) LIKE '%luna%' "
            "AND (material_difference=1 OR preference='production' OR confidence!='high' OR validation_status NOT IN ('valid','pass','approved')) "
            "ORDER BY id DESC LIMIT 1", (member["request_id"], cohort["created_at"])
        ).fetchone()
    ))
    ready = (
        len(members) == 1200 and fresh_luna == 1200
        and required_human_done == required_human and critical == 0
        and dict(split_counts) == EXPECTED_SPLITS
    )
    connection.execute(
        "UPDATE canonical_cohorts SET state=?,updated_at=? WHERE id=?",
        ("ready" if ready else "reviewing", utc_now(), cohort["id"]),
    )
    return {
        "name": name, "state": "ready" if ready else "reviewing", "ready": ready,
        "members": len(members), "fresh_luna": fresh_luna,
        "human_reviewed": human, "required_human": required_human,
        "required_human_done": required_human_done,
        "unresolved_critical": critical,
        "audit": {key: {"reviewed": value[0], "errors": value[1], "error_rate": value[1] / value[0]} for key, value in audit_stats.items()},
        "expanded_review_strata": sorted(expanded_strata),
        "splits": dict(split_counts),
        "strata": {f"{key[0]}/{key[1]}": value for key, value in stratum_counts.items()},
    }


def _annotation_record(connection: sqlite3.Connection, member: sqlite3.Row) -> dict[str, Any]:
    row = connection.execute(
        "SELECT s.*,a.text label,a.origin,a.id annotation_id,d.reviewer,d.outcome,d.decided_at "
        "FROM samples s JOIN annotations a ON a.id=? JOIN decisions d ON d.request_id=s.request_id "
        "AND d.is_current=1 WHERE s.request_id=?",
        (member["annotation_id"], member["request_id"]),
    ).fetchone()
    if not row:
        raise RuntimeError(f"Missing fixed annotation for {member['request_id']}")
    return dict(row) | {
        "manual_review": {"reviewer": row["reviewer"], "outcome": row["outcome"],
                          "timestamp": row["decided_at"]}
    }


def create_release(connection: sqlite3.Connection, cohort_name: str, release_name: str,
                   release_root: Path, system_prompt_path: Path,
                   parent: str | None = None) -> dict[str, Any]:
    _safe_name(cohort_name, "cohort")
    _safe_name(release_name, "release")
    status = cohort_status(connection, cohort_name)
    if not status["ready"]:
        raise RuntimeError(f"Cohort is not releasable: {json.dumps(status, sort_keys=True)}")
    cohort = _cohort_row(connection, cohort_name)
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
                source = _annotation_record(connection, member)
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
                    "annotation_id": member["annotation_id"],
                    "stratum": member["stratum"],
                    "split": split,
                    "origin": source["origin"],
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
        "selection_seed": cohort["seed"], "records": 1200,
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
             sha256_file(manifest_path), str(manifest_path), 1200, utc_now()),
        )
        release_id = int(cursor.lastrowid)
        for member in members:
            connection.execute(
                "INSERT INTO canonical_release_members(release_id,request_id,annotation_id,split,stratum) "
                "VALUES(?,?,?,?,?)",
                (release_id, member["request_id"], member["annotation_id"],
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
    release_parser = sub.add_parser("release")
    release_parser.add_argument("cohort")
    release_parser.add_argument("name")
    release_parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    release_parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    release_parser.add_argument("--parent")
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
                                    args.system_prompt, args.parent)
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
