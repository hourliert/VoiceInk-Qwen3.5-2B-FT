#!/usr/bin/env python3
"""LAN website for reviewing recent VoiceInk proxy requests as they happen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm_cli import (  # noqa: E402
    LABEL_SCHEMA,
    SCORE_DIMENSIONS,
    call_llm,
    parse_json_response,
)
from common.mlflow_tracking import add_mlflow_args, start_mlflow_run  # noqa: E402
from labeling.annotation_store import (  # noqa: E402
    AnnotationStore,
    DEFAULT_AUDIT_MIRROR,
    DEFAULT_BACKUPS,
    DEFAULT_DATABASE,
    DEFAULT_EXPORT,
    DEFAULT_LOCKED_EVAL,
    file_sha256,
    sha256_text,
)
from labeling.label import build_prompt  # noqa: E402

DEFAULT_LOG = ROOT / "logs" / "voiceink_proxy_requests.jsonl"
LABEL_PROMPT = ROOT / "src" / "labeling" / "judge_prompt.codex.txt"
EVALUATOR_PROMPT = ROOT / "src" / "labeling" / "live_evaluator_prompt.codex.txt"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "low"


def scores_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            key: {"type": "integer", "minimum": 1, "maximum": 5}
            for key in SCORE_DIMENSIONS
        },
        "required": list(SCORE_DIMENSIONS),
        "additionalProperties": False,
    }


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "label_validation": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pass", "fail"]},
                "type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["status", "type", "reason"],
            "additionalProperties": False,
        },
        "output_a": scores_schema(),
        "output_b": scores_schema(),
        "context_analysis": {
            "type": "object",
            "properties": {
                "output_a": {"type": "string", "minLength": 1},
                "output_b": {"type": "string", "minLength": 1},
            },
            "required": ["output_a", "output_b"],
            "additionalProperties": False,
        },
        "score_analysis": {
            "type": "object",
            "properties": {
                "output_a": {"type": "string", "minLength": 1},
                "output_b": {"type": "string", "minLength": 1},
            },
            "required": ["output_a", "output_b"],
            "additionalProperties": False,
        },
        "pairwise": {
            "type": "object",
            "properties": {
                "preference": {
                    "type": "string", "enum": ["output_a", "output_b", "tie"]
                },
                "confidence": {
                    "type": "string", "enum": ["low", "medium", "high"]
                },
                "material_difference": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["preference", "confidence", "material_difference", "reason"],
            "additionalProperties": False,
        },
    },
    "required": [
        "label_validation", "output_a", "output_b", "context_analysis",
        "score_analysis", "pairwise",
    ],
    "additionalProperties": False,
}


def read_recent_log(path: Path, limit: int = 200) -> list[dict]:
    if not path.is_file():
        return []
    records: deque[dict] = deque(maxlen=limit)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("path") == "/v1/chat/completions"
                and record.get("request_json_valid")
                and record.get("response_json_valid")
                and record.get("status_code") == 200
            ):
                records.append(record)
    return list(records)


class BestEffortTracking:
    """Keep labeling operational when the optional MLflow service is unavailable."""

    def __init__(self, delegate=None):
        self.delegate = delegate


    @property
    def run_id(self):
        return self.delegate.run_id if self.delegate is not None else None

    def __getattr__(self, name):
        if self.delegate is None:
            return lambda *args, **kwargs: None
        attribute = getattr(self.delegate, name)
        if not callable(attribute):
            return attribute

        def best_effort(*args, **kwargs):
            try:
                return attribute(*args, **kwargs)
            except Exception as exc:
                print(f"MLflow {name} warning: {exc}", file=sys.stderr)
                return None

        return best_effort


class AnalysisWorker:
    def __init__(self, store: AnnotationStore, args: argparse.Namespace):
        self.store = store
        self.args = args
        self.label_template = LABEL_PROMPT.read_text(encoding="utf-8")
        self.evaluator_template = EVALUATOR_PROMPT.read_text(encoding="utf-8")
        self.label_prompt_sha = sha256_text(self.label_template)
        self.evaluator_prompt_sha = sha256_text(self.evaluator_template)
        self.queue: queue.Queue[int] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="luna-review")
        self.store.recover_jobs()
        for job in self.store.pending_jobs():
            self.queue.put(int(job["id"]))
        self.thread.start()

    def cache_key(self, sample: dict) -> str:
        parts = [
            sample["request_sha256"], sample["response_sha256"], self.args.luna_model,
            self.args.luna_reasoning_effort, self.label_prompt_sha,
            self.evaluator_prompt_sha,
        ]
        return sha256_text("\n".join(parts))

    def enqueue(self, request_id: str) -> dict:
        sample = self.store.sample(request_id)
        if not sample:
            raise KeyError(request_id)
        job = self.store.create_or_get_job(request_id, self.cache_key(sample))
        if job["state"] == "pending":
            pending_ids = {item["id"] for item in self.store.pending_jobs()}
            if job["id"] in pending_ids:
                self.queue.put(int(job["id"]))
        return job

    def retry(self, request_id: str) -> dict:
        job = self.store.retry_job(request_id)
        if not job:
            return self.enqueue(request_id)
        self.queue.put(int(job["id"]))
        return job

    def _job(self, job_id: int) -> dict | None:
        with self.store._lock:
            row = self.store.connection.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job_id = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self._job(job_id)
            if not job or job["state"] != "pending":
                self.queue.task_done()
                continue
            for attempt in range(int(job["attempts"]) + 1, 4):
                self.store.update_job(job_id, "running", attempts=attempt, error="")
                print(f"Luna job {job_id}: starting attempt {attempt}/3", flush=True)
                try:
                    self._analyze(job)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"Luna job {job_id}: {error}", file=sys.stderr, flush=True)
                    final = attempt >= 3
                    self.store.update_job(job_id, "failed" if final else "running",
                                          attempts=attempt, error=error)
                    if final:
                        with self.store._lock:
                            self.store.audit("analysis.failed", job["request_id"], {
                                "job_id": job_id, "attempts": attempt, "error": error,
                            })
                    else:
                        time.sleep(2 ** attempt)
            self.queue.task_done()

    def _analyze(self, job: dict) -> None:
        sample = self.store.sample(job["request_id"])
        if not sample:
            raise KeyError(job["request_id"])
        tracking_args = SimpleNamespace(
            mlflow_enabled=self.args.mlflow_enabled,
            mlflow_tracking_uri=self.args.mlflow_tracking_uri,
            mlflow_experiment=self.args.mlflow_experiment,
            mlflow_run_name=None,
            mlflow_run_id=None,
        )
        try:
            tracking = BestEffortTracking(start_mlflow_run(
            tracking_args,
            run_name=f"live-review-{job['request_id'][:12]}",
            run_kind="labeling.live-review",
            params={
                "request_sha256": sample["request_sha256"],
                "response_sha256": sample["response_sha256"],
                "label_model": self.args.luna_model,
                "evaluator_model": self.args.luna_model,
                "reasoning_effort": self.args.luna_reasoning_effort,
                "label_prompt_sha256": self.label_prompt_sha,
                "evaluator_prompt_sha256": self.evaluator_prompt_sha,
            },
                tags={"voiceink.request_id_hash": sha256_text(job["request_id"])},
            ))
        except Exception as exc:
            print(f"MLflow start warning: {exc}", file=sys.stderr)
            tracking = BestEffortTracking()
        self.store.update_job(job["id"], "running", mlflow_run_id=tracking.run_id or "")
        try:
            tracking.register_prompt(
                name="voiceink-live-labeler", template=self.label_template,
                sha256=self.label_prompt_sha,
            )
            tracking.register_prompt(
                name="voiceink-live-reference-free-evaluator",
                template=self.evaluator_template, sha256=self.evaluator_prompt_sha,
            )
            record = {
                "raw_request_json": sample["raw_request_json"],
                "request_id": sample["request_id"],
            }
            label_prompt = build_prompt(record, self.label_template)
            if label_prompt is None:
                raise ValueError("Unable to build label prompt")
            print(f"Luna job {job['id']}: running labeler", flush=True)
            started = time.perf_counter()
            label_raw = call_llm(
                label_prompt, provider="codex", model=self.args.luna_model,
                reasoning_effort=self.args.luna_reasoning_effort,
                output_schema=LABEL_SCHEMA, timeout=600,
            )
            label = parse_json_response(label_raw)["label"].strip()
            label_ms = (time.perf_counter() - started) * 1000
            annotation_id = self.store.create_annotation(
                sample["request_id"], label, origin="luna", provider="codex",
                model=self.args.luna_model,
                reasoning_effort=self.args.luna_reasoning_effort,
                prompt_sha256=self.label_prompt_sha,
                metadata={"session": "ephemeral", "job_id": job["id"]},
            )

            a_is_production = random.Random(job["cache_key"]).random() < 0.5
            output_a = sample["production_output"] if a_is_production else label
            output_b = label if a_is_production else sample["production_output"]
            evaluator_prompt = self.evaluator_template.format(
                custom_vocabulary=sample["custom_vocabulary"] or "(none provided)",
                window_context=sample["window_context"] or "(empty)",
                clipboard_context=sample["clipboard_context"] or "(empty)",
                raw_transcript=sample["transcript"], proposed_label=label,
                output_a=output_a, output_b=output_b,
            )
            print(f"Luna job {job['id']}: label ready; running evaluator", flush=True)
            started = time.perf_counter()
            analysis_raw = call_llm(
                evaluator_prompt, provider="codex", model=self.args.luna_model,
                reasoning_effort=self.args.luna_reasoning_effort,
                output_schema=ANALYSIS_SCHEMA, timeout=600,
            )
            parsed = parse_json_response(analysis_raw)
            evaluator_ms = (time.perf_counter() - started) * 1000
            result = self._normalize_evaluation(parsed, a_is_production)
            analysis_id = self.store.create_analysis(
                sample["request_id"], annotation_id, job["cache_key"], result,
                provider="codex", model=self.args.luna_model,
                reasoning_effort=self.args.luna_reasoning_effort,
                prompt_sha256=self.evaluator_prompt_sha,
                blinded_assignment="a=production" if a_is_production else "a=proposal",
            )
            tracking.log_metrics({
                "label_latency_ms": label_ms,
                "evaluator_latency_ms": evaluator_ms,
                "label_validation_pass": result["label_validation"]["status"] == "pass",
                "material_difference": result["pairwise"]["material_difference"],
                **{f"production.{k}": v for k, v in result["production_scores"].items()},
                **{f"proposal.{k}": v for k, v in result["proposal_scores"].items()},
            })
            tracking.set_tags({
                "voiceink.validation_status": result["label_validation"]["status"],
                "voiceink.preference": result["pairwise"]["preference"],
                "voiceink.confidence": result["pairwise"]["confidence"],
            })
            self.store.update_job(job["id"], "completed", annotation_id=annotation_id,
                                  analysis_id=analysis_id, error="")
            print(f"Luna job {job['id']}: completed", flush=True)
            tracking.finish()
        except Exception:
            tracking.finish("FAILED")
            raise

    @staticmethod
    def _normalize_evaluation(parsed: dict, a_is_production: bool) -> dict:
        if a_is_production:
            production_key, proposal_key = "output_a", "output_b"
        else:
            production_key, proposal_key = "output_b", "output_a"
        preference = parsed["pairwise"]["preference"]
        if preference == "tie":
            mapped = "tie"
        elif preference == production_key:
            mapped = "production"
        else:
            mapped = "proposal"
        return {
            "label_validation": parsed["label_validation"],
            "production_scores": parsed[production_key],
            "proposal_scores": parsed[proposal_key],
            "context_analysis": {
                "production": parsed["context_analysis"][production_key],
                "proposal": parsed["context_analysis"][proposal_key],
            },
            "score_analysis": {
                "production": parsed["score_analysis"][production_key],
                "proposal": parsed["score_analysis"][proposal_key],
            },
            "pairwise": parsed["pairwise"] | {"preference": mapped},
        }


class LiveReviewHandler(BaseHTTPRequestHandler):
    server_version = "VoiceInkLiveReview/1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(HTML_PAGE)
            return
        if parsed.path == "/api/recent":
            self.server.sync_recent()
            query = parse_qs(parsed.query)
            limit = min(int(query.get("limit", ["10"])[0]), 50)
            self._json(self.server.store.recent(limit))
            return
        if parsed.path == "/api/history":
            query = parse_qs(parsed.query)
            self._json(self.server.store.history(
                query=query.get("q", [""])[0], status=query.get("status", [""])[0],
                limit=int(query.get("limit", ["50"])[0]),
                before=query.get("before", [""])[0],
            ))
            return
        if parsed.path == "/api/stats":
            self._json(self.server.store.stats())
            return
        if parsed.path == "/api/health":
            self._json(self.server.health())
            return
        prefix = "/api/samples/"
        if parsed.path.startswith(prefix):
            request_id = unquote(parsed.path[len(prefix):])
            sample = self.server.store.sample(request_id)
            if sample is None:
                self._error(404, "Sample not found")
            else:
                self._json(sample)
            return
        self._error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "samples"]:
            self._error(404, "Not found")
            return
        request_id = unquote(parts[2])
        action = parts[3]
        try:
            if action == "analyze":
                self._json(self.server.worker.enqueue(request_id))
            elif action == "retry":
                self._json(self.server.worker.retry(request_id))
            elif action == "decision":
                self._json(self.server.save_decision(request_id, self._body()))
            else:
                self._error(404, "Not found")
        except KeyError:
            self._error(404, "Sample not found")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        parsed = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, value: str) -> None:
        body = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def log_message(self, format: str, *args) -> None:
        if args and str(args[0]).startswith("POST"):
            super().log_message(format, *args)


class LiveReviewServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, store: AnnotationStore, worker: AnalysisWorker,
                 args: argparse.Namespace):
        super().__init__(address, LiveReviewHandler)
        self.store = store
        self.worker = worker
        self.args = args
        self.last_export: dict | None = None
        self.last_backup: str = ""

    def sync_recent(self) -> int:
        count = 0
        for record in read_recent_log(self.args.log_file):
            count += self.store.import_proxy_record(record)
        return count

    def cohort_queue(self, name: str = "", limit: int = 100) -> list[dict]:
        if name:
            cohort = self.store.connection.execute(
                "SELECT * FROM canonical_cohorts WHERE name=?", (name,)
            ).fetchone()
        else:
            cohort = self.store.connection.execute(
                "SELECT * FROM canonical_cohorts WHERE state!='released' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not cohort:
            return []
        rows = self.store.connection.execute(
            "SELECT s.request_id,s.timestamp,s.transcript,s.production_output,m.split,m.stratum,"
            "m.audit_selected,d.outcome,d.reviewer,j.state job_state FROM canonical_cohort_members m "
            "JOIN samples s ON s.request_id=m.request_id "
            "LEFT JOIN decisions d ON d.request_id=s.request_id AND d.is_current=1 "
            "LEFT JOIN jobs j ON j.id=(SELECT id FROM jobs WHERE request_id=s.request_id ORDER BY id DESC LIMIT 1) "
            "WHERE m.cohort_id=? ORDER BY CASE WHEN m.split='acceptance' THEN 0 ELSE 1 END,"
            "m.audit_selected DESC,s.timestamp DESC LIMIT ?",
            (cohort["id"], min(max(limit, 1), 500)),
        )
        return [dict(row) | {"cohort": cohort["name"]} for row in rows]

    def save_decision(self, request_id: str, body: dict) -> dict:
        outcome = str(body.get("outcome", ""))
        context_dependency = str(body.get("context_dependency", "unknown"))
        if context_dependency not in {"unknown", "none", "window", "clipboard", "vocabulary"}:
            raise ValueError("Invalid context_dependency")
        sample = self.store.sample(request_id)
        if not sample:
            raise KeyError(request_id)
        annotation_id = None
        if outcome == "accept_luna":
            luna = [item for item in sample["annotations"] if item["origin"] == "luna"]
            if not luna:
                raise ValueError("Luna label is not ready")
            annotation_id = int(luna[-1]["id"])
        elif outcome == "human_edit":
            text = str(body.get("label", "")).strip()
            if not text:
                raise ValueError("Edited label cannot be empty")
            luna = [item for item in sample["annotations"] if item["origin"] == "luna"]
            annotation_id = self.store.create_annotation(
                request_id, text, origin="human",
                supersedes_id=int(luna[-1]["id"]) if luna else None,
                metadata={"reviewer": "human"},
            )
        elif outcome == "production_correct":
            if not sample["production_output"].strip():
                raise ValueError("Production output is empty")
            annotation_id = self.store.create_annotation(
                request_id, sample["production_output"], origin="production",
                model=sample["production_model"], metadata={"approved_by": "human"},
            )
        elif outcome != "exclude":
            raise ValueError("Invalid outcome")
        decision_id = self.store.decide(
            request_id, outcome, annotation_id=annotation_id,
            context_dependency=context_dependency,
        )
        self.store.mirror_audit_events(self.args.audit_mirror)
        self.last_export = self.store.export_approved(
            self.args.approved_export, self.args.locked_eval
        )
        self.last_backup = str(self.store.backup(self.args.backup_dir))
        self._tag_mlflow_outcome(sample.get("job"), outcome)
        return {
            "ok": True, "decision_id": decision_id, "annotation_id": annotation_id,
            "export": self.last_export, "backup": self.last_backup,
        }

    def _tag_mlflow_outcome(self, job: dict | None, outcome: str) -> None:
        if not job or not job.get("mlflow_run_id") or not self.args.mlflow_enabled:
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(self.args.mlflow_tracking_uri)
            mlflow.tracking.MlflowClient().set_tag(
                job["mlflow_run_id"], "voiceink.human_outcome", outcome
            )
        except Exception as exc:
            print(f"MLflow outcome tag warning: {exc}", file=sys.stderr)

    def health(self) -> dict:
        export_fresh = self.args.approved_export.is_file()
        mirror_fresh = self.args.audit_mirror.is_file()
        return {
            "status": "ok" if self.store.integrity_check() == "ok" else "degraded",
            "database": str(self.store.path),
            "database_integrity": self.store.integrity_check(),
            "proxy_log": {"path": str(self.args.log_file), "readable": self.args.log_file.is_file()},
            "approved_export": {"path": str(self.args.approved_export), "exists": export_fresh},
            "audit_mirror": {"path": str(self.args.audit_mirror), "exists": mirror_fresh},
            "last_backup": self.last_backup,
            "worker_alive": self.worker.thread.is_alive(),
            "stats": self.store.stats(),
            "mlflow_tracking_uri": self.args.mlflow_tracking_uri if self.args.mlflow_enabled else None,
        }


HTML_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoiceInk Live Review</title>
<style>
:root{color-scheme:dark;--bg:#0c0d12;--panel:#151720;--line:#292d3b;--muted:#9298aa;--text:#eef0f6;--accent:#7c9cff;--good:#61d095;--bad:#ff718b}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header{position:sticky;top:0;z-index:2;background:#10121a;border-bottom:1px solid var(--line);padding:14px 20px;display:flex;gap:16px;align-items:center}h1{font-size:17px;margin:0}button,input,select,textarea{font:inherit;color:inherit;background:#1b1e29;border:1px solid var(--line);border-radius:7px;padding:8px}button{cursor:pointer}button.primary{background:#3658d4}button.good{background:#176a48}button.bad{background:#742a3b}.muted{color:var(--muted)}nav{display:flex;gap:6px}.tab.active{border-color:var(--accent)}main{max-width:1500px;margin:auto;padding:18px}.toolbar{display:flex;gap:8px;margin-bottom:12px}.list{display:grid;gap:8px}.row{background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:9px;cursor:pointer;display:grid;grid-template-columns:150px 1fr 120px;gap:12px}.row:hover{border-color:#465071}.snippet{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.badge{font-size:11px;color:var(--muted)}.detail{display:none}.detail.open{display:block}.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}.card h3{margin:0;padding:10px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}pre,.body{margin:0;padding:14px;white-space:pre-wrap;word-break:break-word;max-height:440px;overflow:auto}textarea{width:100%;min-height:220px;border:0;border-radius:0;padding:14px}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.scores{display:grid;grid-template-columns:1fr 1fr;gap:12px}.status{padding:10px;border-radius:7px;background:#222633}.ok{color:var(--good)}.fail{color:var(--bad)}details{margin-top:12px}.stats{margin-left:auto;color:var(--muted)}@media(max-width:800px){.columns,.scores{grid-template-columns:1fr}.row{grid-template-columns:1fr}.stats{display:none}}
</style></head><body>
<header><h1>VoiceInk Live Review</h1><nav><button class="tab active" data-view="recent">Recent</button><button class="tab" data-view="history">History</button><button class="tab" data-view="cohort">Bootstrap Queue</button></nav><span class="stats" id="stats"></span></header>
<main><section id="browser"><div class="toolbar"><input id="search" placeholder="Search history"><select id="filter"><option value="">All</option><option value="pending">Pending</option><option value="reviewed">Reviewed</option><option value="human_edit">Edited</option><option value="exclude">Excluded</option></select><button id="refresh">Refresh</button></div><div class="list" id="list"></div></section>
<section class="detail" id="detail"><button onclick="closeDetail()">← Back</button><h2 id="title"></h2><div class="status" id="job"></div>
<div class="columns"><div class="card"><h3>Raw transcript</h3><pre id="raw"></pre></div><div class="card"><h3>Production output</h3><pre id="production"></pre></div></div>
<div class="columns" style="margin-top:12px"><div class="card"><h3>Luna proposal</h3><pre id="luna"></pre></div><div class="card"><h3>Final label editor</h3><textarea id="editor"></textarea></div></div>
<div class="actions"><button class="good" onclick="decide('accept_luna')">Accept Luna</button><button class="primary" onclick="decide('human_edit')">Save My Edit</button><button onclick="decide('production_correct')">Production Was Correct</button><button class="bad" onclick="decide('exclude')">Exclude</button><select id="context"><option value="unknown">Context: unknown</option><option value="none">No context needed</option><option value="window">Window context</option><option value="clipboard">Clipboard</option><option value="vocabulary">Vocabulary</option></select><button onclick="retry()">Retry analysis</button></div>
<div class="card"><h3>Independent Luna evaluator</h3><div class="body" id="evaluation"></div></div>
<details><summary>Window context</summary><pre id="window"></pre></details><details><summary>Clipboard context</summary><pre id="clipboard"></pre></details><details><summary>Custom vocabulary</summary><pre id="vocab"></pre></details></section></main>
<script>
let view='recent', current=null, poll=null; const $=id=>document.getElementById(id); const esc=s=>(s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function api(path,options){const r=await fetch(path,options);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
async function load(){const q=view==='recent'?'/api/recent?limit=10':view==='cohort'?'/api/cohort?limit=200':`/api/history?q=${encodeURIComponent($('search').value)}&status=${encodeURIComponent($('filter').value)}`;const rows=await api(q);$('list').innerHTML=rows.map(r=>`<div class="row" onclick="openSample('${r.request_id}')"><span>${new Date(r.timestamp).toLocaleString()}</span><span class="snippet">${esc(r.transcript)}</span><span class="badge">${esc(r.outcome||r.job_state||'new')}</span></div>`).join('')||'<p class="muted">No samples.</p>';const s=await api('/api/stats');$('stats').textContent=`${s.samples} samples · ${s.reviewed} reviewed · ${s.training_eligible} approved`}
async function openSample(id,updateUrl=true){if(updateUrl){const u=new URL(location.href);u.searchParams.set("id",id);history.pushState({},"",u)}current=id;$('browser').style.display='none';$('detail').classList.add('open');await api(`/api/samples/${encodeURIComponent(id)}/analyze`,{method:'POST'});await loadDetail();clearInterval(poll);poll=setInterval(loadDetail,1800)}
async function loadDetail(){if(!current)return;const s=await api(`/api/samples/${encodeURIComponent(current)}`);$('title').textContent=`Review ${s.request_id.slice(0,12)}…`;$('raw').textContent=s.transcript;$('production').textContent=s.production_output;$('window').textContent=s.window_context;$('clipboard').textContent=s.clipboard_context||'(empty)';$('vocab').textContent=s.custom_vocabulary||'(empty)';const luna=[...s.annotations].reverse().find(a=>a.origin==='luna');$('luna').textContent=luna?.text||'Waiting for Luna…';if(!$('editor').dataset.touched)$('editor').value=luna?.text||s.production_output;$('job').innerHTML=`Analysis: <b>${esc(s.job?.state||'not started')}</b>${s.job?.error?' · <span class="fail">'+esc(s.job.error)+'</span>':''}`;const a=s.analyses.at(-1);$('evaluation').innerHTML=a?renderEval(a):'<span class="muted">Evaluator pending…</span>';if(s.job?.state==='completed'||s.job?.state==='failed')clearInterval(poll)}
function renderEval(a){const v=a.raw_result.label_validation,p=a.raw_result.pairwise;return `<p>Label validation: <b class="${v.status==='pass'?'ok':'fail'}">${esc(v.status)}</b> ${esc(v.type)} ${esc(v.reason)}</p><p>Preference: <b>${esc(p.preference)}</b> · ${esc(p.confidence)} confidence · material=${p.material_difference}</p><p>${esc(p.reason)}</p><div class="scores"><pre>Production\n${esc(JSON.stringify(a.production_scores,null,2))}\n\n${esc(a.score_analysis.production)}</pre><pre>Proposal\n${esc(JSON.stringify(a.proposal_scores,null,2))}\n\n${esc(a.score_analysis.proposal)}</pre></div>`}
async function decide(outcome){const body={outcome,context_dependency:$('context').value,label:$('editor').value};await api(`/api/samples/${encodeURIComponent(current)}/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});closeDetail();await load()}
async function retry(){await api(`/api/samples/${encodeURIComponent(current)}/retry`,{method:'POST'});await loadDetail()}
function closeDetail(updateUrl=true){if(updateUrl){const u=new URL(location.href);u.searchParams.delete("id");history.pushState({},"",u)}current=null;clearInterval(poll);$('detail').classList.remove('open');$('browser').style.display='block';delete $('editor').dataset.touched}
function syncNavigation(updateUrl=false){document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active",b.dataset.view===view));$("search").style.display=view==="history"?"block":"none";$("filter").style.display=view==="history"?"block":"none";if(updateUrl){const u=new URL(location.href);u.searchParams.delete("id");if(view!=="recent")u.searchParams.set("view",view);else u.searchParams.delete("view");history.pushState({},"",u)}}
async function navigate(nextView){view=nextView;closeDetail(false);syncNavigation(true);await load()}
$("editor").addEventListener("input",()=>{$("editor").dataset.touched="1"});$("refresh").onclick=load;$("search").onkeydown=e=>{if(e.key==="Enter")load()};$("filter").onchange=load;document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>navigate(b.dataset.view));const initialUrl=new URL(location.href);view=["history","cohort"].includes(initialUrl.searchParams.get("view"))?initialUrl.searchParams.get("view"):"recent";syncNavigation(false);load();const initialId=initialUrl.searchParams.get("id");if(initialId)openSample(initialId,false);window.addEventListener("popstate",()=>{const u=new URL(location.href);view=["history","cohort"].includes(u.searchParams.get("view"))?u.searchParams.get("view"):"recent";syncNavigation(false);const id=u.searchParams.get("id");if(id)openSample(id,false);else{closeDetail(false);load()}});setInterval(()=>{if(!current&&view==="recent")load()},10000);
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live VoiceInk labeling and review website")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--approved-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--audit-mirror", type=Path, default=DEFAULT_AUDIT_MIRROR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUPS)
    parser.add_argument("--locked-eval", type=Path, default=DEFAULT_LOCKED_EVAL)
    parser.add_argument("--luna-model", default=DEFAULT_MODEL)
    parser.add_argument("--luna-reasoning-effort", default=DEFAULT_EFFORT)
    add_mlflow_args(parser, default_experiment="voiceink-labeling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    store = AnnotationStore(args.database)
    worker = AnalysisWorker(store, args)
    server = LiveReviewServer((args.host, args.port), store, worker, args)
    imported = server.sync_recent()
    store.mirror_audit_events(args.audit_mirror)
    print(f"VoiceInk live review: http://{args.host}:{args.port}")
    print(f"Registry: {args.database} ({store.stats()['samples']} samples; synced {imported})")
    if args.host not in {"127.0.0.1", "localhost"}:
        print("WARNING: no authentication; private transcripts and screen context are visible on the LAN")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop_event.set()
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
