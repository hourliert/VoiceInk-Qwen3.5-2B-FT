"""Durable annotation registry for live and historical VoiceInk samples."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / "datasets" / "registry" / "annotations.sqlite3"
DEFAULT_EXPORT = ROOT / "datasets" / "exports" / "live" / "current-approved.jsonl"
DEFAULT_AUDIT_MIRROR = ROOT / "datasets" / "registry" / "events.jsonl"
DEFAULT_BACKUPS = ROOT / "datasets" / "registry" / "backups"
DEFAULT_LOCKED_EVAL = ROOT / "datasets" / "regression" / "voiceink-locked-440.jsonl"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def normalize_transcript(value: str) -> str:
    return " ".join(value.casefold().split())


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return ""


def locked_transcript_fingerprints(path: Path = DEFAULT_LOCKED_EVAL) -> set[str]:
    fingerprints: set[str] = set()
    if not path.is_file():
        return fingerprints
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            for message in record.get("messages", []):
                if message.get("role") != "user":
                    continue
                text = _message_text(message.get("content", "")).strip()
                if text.startswith("<TRANSCRIPT>") and text.endswith("</TRANSCRIPT>"):
                    text = text[len("<TRANSCRIPT>"):-len("</TRANSCRIPT>")].strip()
                fingerprints.add(sha256_text(normalize_transcript(text)))
    return fingerprints


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    role TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    rows_seen INTEGER NOT NULL DEFAULT 0,
    malformed_rows INTEGER NOT NULL DEFAULT 0,
    UNIQUE(path, sha256)
);
CREATE TABLE IF NOT EXISTS samples (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    raw_request_json TEXT NOT NULL,
    raw_response_json TEXT NOT NULL DEFAULT '',
    transcript TEXT NOT NULL,
    custom_vocabulary TEXT NOT NULL DEFAULT '',
    clipboard_context TEXT NOT NULL DEFAULT '',
    window_context TEXT NOT NULL DEFAULT '',
    production_model TEXT NOT NULL DEFAULT '',
    production_output TEXT NOT NULL DEFAULT '',
    status_code INTEGER,
    duration_ms REAL,
    request_sha256 TEXT NOT NULL,
    response_sha256 TEXT NOT NULL,
    transcript_fingerprint TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_timestamp_idx ON samples(timestamp DESC);
CREATE INDEX IF NOT EXISTS samples_transcript_idx ON samples(transcript_fingerprint);
CREATE TABLE IF NOT EXISTS sample_sources (
    sample_id TEXT NOT NULL REFERENCES samples(request_id) ON DELETE CASCADE,
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    source_line INTEGER,
    PRIMARY KEY(sample_id, source_file_id, source_line)
);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES samples(request_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    origin TEXT NOT NULL,
    supersedes_id INTEGER REFERENCES annotations(id),
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    reasoning_effort TEXT NOT NULL DEFAULT '',
    prompt_sha256 TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(request_id, text_sha256, origin, provider, model, prompt_sha256)
);
CREATE INDEX IF NOT EXISTS annotations_request_idx ON annotations(request_id, id);
CREATE TABLE IF NOT EXISTS annotation_sources (
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    source_line INTEGER,
    PRIMARY KEY(annotation_id, source_file_id, source_line)
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES samples(request_id) ON DELETE CASCADE,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    cache_key TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    reasoning_effort TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_type TEXT NOT NULL DEFAULT '',
    validation_reason TEXT NOT NULL DEFAULT '',
    production_scores_json TEXT NOT NULL,
    proposal_scores_json TEXT NOT NULL,
    preference TEXT NOT NULL,
    confidence TEXT NOT NULL,
    material_difference INTEGER NOT NULL,
    context_analysis_json TEXT NOT NULL,
    score_analysis_json TEXT NOT NULL,
    pairwise_reason TEXT NOT NULL,
    blinded_assignment TEXT NOT NULL,
    raw_result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES samples(request_id) ON DELETE CASCADE,
    annotation_id INTEGER REFERENCES annotations(id),
    outcome TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    training_eligible INTEGER NOT NULL,
    context_dependency TEXT NOT NULL DEFAULT 'unknown',
    decided_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS decisions_current_idx
ON decisions(request_id) WHERE is_current = 1;
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES samples(request_id) ON DELETE CASCADE,
    cache_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    annotation_id INTEGER REFERENCES annotations(id),
    analysis_id INTEGER REFERENCES analyses(id),
    mlflow_run_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, id);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    request_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_releases (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    selection_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    records INTEGER NOT NULL,
    output_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_memberships (
    release_id INTEGER NOT NULL REFERENCES dataset_releases(id) ON DELETE CASCADE,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    role TEXT NOT NULL,
    PRIMARY KEY(release_id, annotation_id)
);
"""


class AnnotationStore:
    """Thread-safe SQLite registry with immutable annotations and analyses."""

    def __init__(self, path: Path = DEFAULT_DATABASE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            path, timeout=30, check_same_thread=False, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        with self._lock:
            self.connection.executescript(SCHEMA)
            self.connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def integrity_check(self) -> str:
        with self._lock:
            return str(self.connection.execute("PRAGMA quick_check").fetchone()[0])

    def audit(self, event_type: str, request_id: str | None, payload: dict) -> int:
        cursor = self.connection.execute(
            "INSERT INTO audit_events(event_type, request_id, payload_json, created_at) "
            "VALUES(?, ?, ?, ?)",
            (event_type, request_id, json_dump(payload), utc_now()),
        )
        return int(cursor.lastrowid)

    def register_source(self, path: Path, role: str, rows: int = 0,
                        malformed: int = 0) -> tuple[int, bool]:
        digest = file_sha256(path)
        with self._lock:
            row = self.connection.execute(
                "SELECT id FROM source_files WHERE path = ? AND sha256 = ?",
                (str(path.resolve()), digest),
            ).fetchone()
            if row:
                return int(row["id"]), False
            cursor = self.connection.execute(
                "INSERT INTO source_files(path, sha256, bytes, role, imported_at, "
                "rows_seen, malformed_rows) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (str(path.resolve()), digest, path.stat().st_size, role, utc_now(),
                 rows, malformed),
            )
            source_id = int(cursor.lastrowid)
            self.audit("source.imported", None, {
                "source_file_id": source_id, "path": str(path), "sha256": digest,
                "role": role, "rows": rows, "malformed": malformed,
            })
            return source_id, True

    def import_proxy_record(self, record: dict, *, source_file_id: int | None = None,
                            source_line: int | None = None) -> bool:
        from common.extract import extract_components

        request_id = str(record.get("request_id", "")).strip()
        raw_request = str(record.get("raw_request_json", ""))
        if not request_id or not raw_request:
            return False
        try:
            components = extract_components(raw_request)
        except (json.JSONDecodeError, TypeError):
            return False
        transcript = components["transcript"]
        if not transcript:
            return False
        response = str(record.get("response_text") or record.get("original_response") or "")
        raw_response = str(record.get("raw_response_json", ""))
        values = (
            request_id, str(record.get("timestamp", "") or utc_now()), raw_request,
            raw_response, transcript, components["custom_vocabulary"],
            components["clipboard_context"], components["window_context"],
            str(record.get("model") or record.get("original_model") or components["model"]),
            response, record.get("status_code"), record.get("duration_ms"),
            sha256_text(raw_request), sha256_text(response),
            sha256_text(normalize_transcript(transcript)), utc_now(),
        )
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "INSERT INTO samples(request_id,timestamp,raw_request_json,raw_response_json,"
                    "transcript,custom_vocabulary,clipboard_context,window_context,production_model,"
                    "production_output,status_code,duration_ms,request_sha256,response_sha256,"
                    "transcript_fingerprint,imported_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(request_id) DO UPDATE SET "
                    "timestamp=excluded.timestamp, raw_request_json=excluded.raw_request_json, "
                    "raw_response_json=CASE WHEN excluded.raw_response_json != '' THEN excluded.raw_response_json ELSE samples.raw_response_json END, "
                    "transcript=excluded.transcript, custom_vocabulary=excluded.custom_vocabulary, "
                    "clipboard_context=excluded.clipboard_context, window_context=excluded.window_context, "
                    "production_model=CASE WHEN excluded.production_model != '' THEN excluded.production_model ELSE samples.production_model END, "
                    "production_output=CASE WHEN excluded.production_output != '' THEN excluded.production_output ELSE samples.production_output END, "
                    "status_code=COALESCE(excluded.status_code,samples.status_code), "
                    "duration_ms=COALESCE(excluded.duration_ms,samples.duration_ms), "
                    "request_sha256=excluded.request_sha256, response_sha256=excluded.response_sha256, "
                    "transcript_fingerprint=excluded.transcript_fingerprint",
                    values,
                )
                if source_file_id is not None:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO sample_sources(sample_id,source_file_id,source_line) "
                        "VALUES(?,?,?)", (request_id, source_file_id, source_line),
                    )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        return True

    def create_annotation(self, request_id: str, text: str, *, origin: str,
                          provider: str = "", model: str = "",
                          reasoning_effort: str = "", prompt_sha256: str = "",
                          supersedes_id: int | None = None,
                          metadata: dict | None = None,
                          source_file_id: int | None = None,
                          source_line: int | None = None,
                          created_at: str | None = None) -> int:
        text = text.strip()
        if not text:
            raise ValueError("Annotation text cannot be empty")
        digest = sha256_text(text)
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "INSERT OR IGNORE INTO annotations(request_id,text,text_sha256,origin,"
                    "supersedes_id,provider,model,reasoning_effort,prompt_sha256,metadata_json,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (request_id, text, digest, origin, supersedes_id, provider, model,
                     reasoning_effort, prompt_sha256, json_dump(metadata or {}),
                     created_at or utc_now()),
                )
                row = self.connection.execute(
                    "SELECT id FROM annotations WHERE request_id=? AND text_sha256=? "
                    "AND origin=? AND provider=? AND model=? AND prompt_sha256=?",
                    (request_id, digest, origin, provider, model, prompt_sha256),
                ).fetchone()
                annotation_id = int(row["id"])
                if source_file_id is not None:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO annotation_sources(annotation_id,source_file_id,source_line) "
                        "VALUES(?,?,?)", (annotation_id, source_file_id, source_line),
                    )
                self.audit("annotation.created", request_id, {
                    "annotation_id": annotation_id, "origin": origin,
                    "text_sha256": digest, "model": model,
                })
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        return annotation_id

    def create_analysis(self, request_id: str, annotation_id: int, cache_key: str,
                        result: dict, *, provider: str, model: str,
                        reasoning_effort: str, prompt_sha256: str,
                        blinded_assignment: str) -> int:
        validation = result["label_validation"]
        pairwise = result["pairwise"]
        with self._lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO analyses(request_id,annotation_id,cache_key,provider,model,"
                "reasoning_effort,prompt_sha256,validation_status,validation_type,validation_reason,"
                "production_scores_json,proposal_scores_json,preference,confidence,material_difference,"
                "context_analysis_json,score_analysis_json,pairwise_reason,blinded_assignment,"
                "raw_result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, annotation_id, cache_key, provider, model, reasoning_effort,
                 prompt_sha256, validation["status"], validation.get("type", ""),
                 validation.get("reason", ""), json_dump(result["production_scores"]),
                 json_dump(result["proposal_scores"]), pairwise["preference"],
                 pairwise["confidence"], int(pairwise["material_difference"]),
                 json_dump(result["context_analysis"]), json_dump(result["score_analysis"]),
                 pairwise["reason"], blinded_assignment, json_dump(result), utc_now()),
            )
            row = self.connection.execute(
                "SELECT id FROM analyses WHERE cache_key=?", (cache_key,)
            ).fetchone()
            analysis_id = int(row["id"])
            self.audit("analysis.completed", request_id, {
                "analysis_id": analysis_id, "annotation_id": annotation_id,
                "cache_key": cache_key, "validation": validation["status"],
                "preference": pairwise["preference"],
            })
            return analysis_id

    def create_or_get_job(self, request_id: str, cache_key: str) -> dict:
        now = utc_now()
        with self._lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO jobs(request_id,cache_key,state,created_at,updated_at) "
                "VALUES(?,?,'pending',?,?)", (request_id, cache_key, now, now),
            )
            return dict(self.connection.execute(
                "SELECT * FROM jobs WHERE cache_key=?", (cache_key,)
            ).fetchone())

    def update_job(self, job_id: int, state: str, **fields: Any) -> None:
        allowed = {"attempts", "error", "annotation_id", "analysis_id", "mlflow_run_id"}
        assignments = ["state=?", "updated_at=?"]
        values: list[Any] = [state, utc_now()]
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            values.append(value)
        values.append(job_id)
        with self._lock:
            self.connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?", values
            )

    def recover_jobs(self) -> int:
        with self._lock:
            cursor = self.connection.execute(
                "UPDATE jobs SET state='pending', attempts=0, error='recovered after restart', updated_at=? "
                "WHERE state='running' OR (state='pending' AND attempts>=3)", (utc_now(),)
            )
            return cursor.rowcount

    def pending_jobs(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.connection.execute(
                "SELECT * FROM jobs WHERE state='pending' ORDER BY id"
            )]

    def retry_job(self, request_id: str) -> dict | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE request_id=? ORDER BY id DESC LIMIT 1", (request_id,)
            ).fetchone()
            if not row:
                return None
            self.connection.execute(
                "UPDATE jobs SET state='pending',attempts=0,error='',updated_at=? WHERE id=?",
                (utc_now(), row["id"]),
            )
            return dict(self.connection.execute(
                "SELECT * FROM jobs WHERE id=?", (row["id"],)
            ).fetchone())

    def decide(self, request_id: str, outcome: str, *, annotation_id: int | None,
               reviewer: str = "human", context_dependency: str = "unknown") -> int:
        if outcome not in {"accept_luna", "human_edit", "production_correct", "exclude"}:
            raise ValueError(f"Invalid outcome: {outcome}")
        eligible = outcome != "exclude"
        if eligible and annotation_id is None:
            raise ValueError("Approved outcomes require an annotation")
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "UPDATE decisions SET is_current=0 WHERE request_id=? AND is_current=1",
                    (request_id,),
                )
                cursor = self.connection.execute(
                    "INSERT INTO decisions(request_id,annotation_id,outcome,reviewer,"
                    "training_eligible,context_dependency,decided_at,is_current) "
                    "VALUES(?,?,?,?,?,?,?,1)",
                    (request_id, annotation_id, outcome, reviewer, int(eligible),
                     context_dependency, utc_now()),
                )
                decision_id = int(cursor.lastrowid)
                self.audit("decision.saved", request_id, {
                    "decision_id": decision_id, "annotation_id": annotation_id,
                    "outcome": outcome, "training_eligible": eligible,
                    "context_dependency": context_dependency,
                })
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        return decision_id

    def sample(self, request_id: str) -> dict | None:
        with self._lock:
            sample = self.connection.execute(
                "SELECT * FROM samples WHERE request_id=?", (request_id,)
            ).fetchone()
            if not sample:
                return None
            result = dict(sample)
            result["annotations"] = [dict(row) | {
                "metadata": json_load(row["metadata_json"], {})
            } for row in self.connection.execute(
                "SELECT * FROM annotations WHERE request_id=? ORDER BY id", (request_id,)
            )]
            for item in result["annotations"]:
                item.pop("metadata_json", None)
            analysis_rows = self.connection.execute(
                "SELECT * FROM analyses WHERE request_id=? ORDER BY id", (request_id,)
            )
            result["analyses"] = [self._decode_analysis(dict(row)) for row in analysis_rows]
            decision = self.connection.execute(
                "SELECT * FROM decisions WHERE request_id=? AND is_current=1", (request_id,)
            ).fetchone()
            result["decision"] = dict(decision) if decision else None
            job = self.connection.execute(
                "SELECT * FROM jobs WHERE request_id=? ORDER BY id DESC LIMIT 1", (request_id,)
            ).fetchone()
            result["job"] = dict(job) if job else None
            return result

    @staticmethod
    def _decode_analysis(row: dict) -> dict:
        for key in (
            "production_scores_json", "proposal_scores_json", "context_analysis_json",
            "score_analysis_json", "raw_result_json",
        ):
            row[key.removesuffix("_json")] = json_load(row.pop(key), {})
        row["material_difference"] = bool(row["material_difference"])
        return row

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.connection.execute(
                "SELECT s.request_id,s.timestamp,s.transcript,s.production_model,s.production_output,"
                "s.duration_ms,d.outcome,d.training_eligible,j.state AS job_state "
                "FROM samples s LEFT JOIN decisions d ON d.request_id=s.request_id AND d.is_current=1 "
                "LEFT JOIN jobs j ON j.id=(SELECT id FROM jobs WHERE request_id=s.request_id "
                "ORDER BY id DESC LIMIT 1) ORDER BY s.timestamp DESC LIMIT ?", (limit,)
            )]

    def history(self, *, query: str = "", status: str = "", limit: int = 50,
                before: str = "") -> list[dict]:
        clauses = ["1=1"]
        values: list[Any] = []
        if query:
            clauses.append("(s.request_id LIKE ? OR s.transcript LIKE ?)")
            values.extend((f"%{query}%", f"%{query}%"))
        if status == "reviewed":
            clauses.append("d.id IS NOT NULL")
        elif status == "pending":
            clauses.append("d.id IS NULL")
        elif status:
            clauses.append("d.outcome=?")
            values.append(status)
        if before:
            clauses.append("s.timestamp<?")
            values.append(before)
        values.append(min(max(limit, 1), 200))
        with self._lock:
            return [dict(row) for row in self.connection.execute(
                "SELECT s.request_id,s.timestamp,s.transcript,s.production_model,s.production_output,"
                "d.outcome,d.training_eligible,j.state AS job_state FROM samples s "
                "LEFT JOIN decisions d ON d.request_id=s.request_id AND d.is_current=1 "
                "LEFT JOIN jobs j ON j.id=(SELECT id FROM jobs WHERE request_id=s.request_id "
                "ORDER BY id DESC LIMIT 1) WHERE " + " AND ".join(clauses) +
                " ORDER BY s.timestamp DESC LIMIT ?", values
            )]

    def stats(self) -> dict:
        with self._lock:
            scalar = lambda sql: int(self.connection.execute(sql).fetchone()[0])
            outcomes = {row[0]: int(row[1]) for row in self.connection.execute(
                "SELECT outcome,COUNT(*) FROM decisions WHERE is_current=1 GROUP BY outcome"
            )}
            return {
                "samples": scalar("SELECT COUNT(*) FROM samples"),
                "annotations": scalar("SELECT COUNT(*) FROM annotations"),
                "analyses": scalar("SELECT COUNT(*) FROM analyses"),
                "reviewed": scalar("SELECT COUNT(*) FROM decisions WHERE is_current=1"),
                "training_eligible": scalar(
                    "SELECT COUNT(*) FROM decisions WHERE is_current=1 AND training_eligible=1"
                ),
                "pending_jobs": scalar("SELECT COUNT(*) FROM jobs WHERE state='pending'"),
                "failed_jobs": scalar("SELECT COUNT(*) FROM jobs WHERE state='failed'"),
                "sources": scalar("SELECT COUNT(*) FROM source_files"),
                "outcomes": outcomes,
            }

    def approved_records(self, locked_eval: Path = DEFAULT_LOCKED_EVAL) -> list[dict]:
        excluded = locked_transcript_fingerprints(locked_eval)
        with self._lock:
            rows = self.connection.execute(
                "SELECT s.*,a.id AS annotation_id,a.text AS label,a.origin,a.provider,a.model AS label_model,"
                "a.reasoning_effort,a.prompt_sha256,a.created_at AS label_created_at,d.outcome,"
                "d.context_dependency,d.decided_at FROM decisions d JOIN annotations a ON a.id=d.annotation_id "
                "JOIN samples s ON s.request_id=d.request_id WHERE d.is_current=1 "
                "AND d.training_eligible=1 ORDER BY s.timestamp,s.request_id"
            )
            records = []
            for row in rows:
                if row["transcript_fingerprint"] in excluded:
                    continue
                analysis = self.connection.execute(
                    "SELECT * FROM analyses WHERE annotation_id=? ORDER BY id DESC LIMIT 1",
                    (row["annotation_id"],),
                ).fetchone()
                record = {
                    "request_id": row["request_id"],
                    "timestamp": row["timestamp"],
                    "model_used_for_label": row["label_model"],
                    "label_provider": row["provider"] or row["origin"],
                    "label_provider_metadata": {
                        "provider": row["provider"] or row["origin"],
                        "model": row["label_model"],
                        "reasoning_effort": row["reasoning_effort"],
                        "prompt_sha256": row["prompt_sha256"],
                    },
                    "original_model": row["production_model"],
                    "raw_request_json": row["raw_request_json"],
                    "original_response": row["production_output"],
                    "label": row["label"],
                    "annotation_registry": {
                        "annotation_id": row["annotation_id"],
                        "origin": row["origin"],
                        "label_created_at": row["label_created_at"],
                    },
                    "manually_reviewed": True,
                    "manual_review": {
                        "status": "edited" if row["outcome"] == "human_edit" else "approved",
                        "timestamp": row["decided_at"],
                        "outcome": row["outcome"],
                        "context_dependency": row["context_dependency"],
                    },
                }
                if analysis:
                    record["live_analysis"] = self._decode_analysis(dict(analysis))
                records.append(record)
            return records

    def export_approved(self, output: Path = DEFAULT_EXPORT,
                        locked_eval: Path = DEFAULT_LOCKED_EVAL) -> dict:
        records = self.approved_records(locked_eval)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        manifest = {
            "created_at": utc_now(), "records": len(records),
            "output": str(output), "sha256": file_sha256(output),
            "database": str(self.path), "database_schema": SCHEMA_VERSION,
            "locked_eval": str(locked_eval),
            "locked_eval_sha256": file_sha256(locked_eval) if locked_eval.is_file() else None,
        }
        manifest_path = output.with_name(output.stem + "-manifest.json")
        temp_manifest = manifest_path.with_suffix(".json.tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        with self._lock:
            self.audit("export.refreshed", None, manifest)
        return manifest

    def create_release(self, name: str, directory: Path,
                       locked_eval: Path = DEFAULT_LOCKED_EVAL) -> dict:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Release directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "approved.jsonl"
        manifest = self.export_approved(output, locked_eval)
        records = self.approved_records(locked_eval)
        memberships = [
            {"request_id": row["request_id"],
             "annotation_id": row["annotation_registry"]["annotation_id"],
             "role": "train"}
            for row in records
        ]
        membership_path = directory / "membership.jsonl"
        with membership_path.open("w", encoding="utf-8") as stream:
            for row in memberships:
                stream.write(json.dumps(row) + "\n")
        release_manifest = manifest | {
            "name": name,
            "membership_sha256": file_sha256(membership_path),
        }
        release_manifest_path = directory / "manifest.json"
        release_manifest_path.write_text(json.dumps(release_manifest, indent=2) + "\n")
        digest = file_sha256(release_manifest_path)
        with self._lock:
            cursor = self.connection.execute(
                "INSERT INTO dataset_releases(name,created_at,selection_json,manifest_sha256,records,output_path) "
                "VALUES(?,?,?,?,?,?)",
                (name, utc_now(), json_dump({"training_eligible": True, "locked_eval_excluded": True}),
                 digest, len(records), str(output)),
            )
            release_id = int(cursor.lastrowid)
            for row in memberships:
                self.connection.execute(
                    "INSERT OR IGNORE INTO release_memberships(release_id,annotation_id,role) VALUES(?,?,'train')",
                    (release_id, row["annotation_id"]),
                )
            self.audit("release.created", None, release_manifest | {"release_id": release_id})
        return release_manifest | {"release_id": release_id}

    def mirror_audit_events(self, path: Path = DEFAULT_AUDIT_MIRROR) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        last_id = 0
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        last_id = max(last_id, int(json.loads(line)["id"]))
        with self._lock:
            rows = list(self.connection.execute(
                "SELECT * FROM audit_events WHERE id>? ORDER BY id", (last_id,)
            ))
        if not rows:
            return 0
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return len(rows)

    def backup(self, directory: Path = DEFAULT_BACKUPS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = directory / f"annotations-{stamp}.sqlite3"
        with self._lock:
            target = sqlite3.connect(destination)
            try:
                self.connection.backup(target)
            finally:
                target.close()
        self._prune_backups(directory)
        return destination

    @staticmethod
    def _prune_backups(directory: Path) -> None:
        backups = sorted(directory.glob("annotations-*.sqlite3"), reverse=True)
        keep = set(backups[:20])
        daily: dict[str, Path] = {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for path in backups:
            try:
                timestamp = datetime.strptime(
                    path.stem.removeprefix("annotations-").split(".")[0], "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                keep.add(path)
                continue
            if timestamp >= cutoff:
                daily.setdefault(timestamp.strftime("%Y%m%d"), path)
        keep.update(daily.values())
        for path in backups:
            if path not in keep:
                path.unlink()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield line_number, record
