#!/usr/bin/env python3
"""VoiceInk control plane: labeling, dataset lifecycle, and MLflow navigation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "src" / "control_plane" / "static"
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import add_mlflow_args  # noqa: E402
from control_plane.contracts import DecisionResponse, HealthResponse, ReadinessResponse  # noqa: E402
from control_plane.ingest import ProxyLogIngester  # noqa: E402
from control_plane.maintenance import MaintenanceWorker  # noqa: E402
from control_plane.state import ControlPlaneState  # noqa: E402
from data.manage import (  # noqa: E402
    DEFAULT_RELEASE_ROOT, DEFAULT_SYSTEM_PROMPT, create_release, release_preview,
)
from data.schema import migrate  # noqa: E402
from labeling.annotation_store import (  # noqa: E402
    AnnotationStore,
    DEFAULT_AUDIT_MIRROR,
    DEFAULT_BACKUPS,
    DEFAULT_DATABASE,
    DEFAULT_EXPORT,
    DEFAULT_LOCKED_EVAL,
    sha256_text,
)
from labeling.live_review_server import (  # noqa: E402
    AnalysisWorker,
    DEFAULT_EFFORT,
    DEFAULT_LOG,
    DEFAULT_MODEL,
)

CONTROL_PLANE_API_VERSION = "2"

PAGE_PATHS = {
    "/", "/review/recent", "/review/queue", "/review/history",
    "/data/cohorts", "/data/releases", "/models", "/runs/training",
    "/runs/evaluations", "/system", "/system/docs",
}


class ControlPlaneWorker(AnalysisWorker):
    """Existing durable Luna worker with bounded concurrency and cohort-fresh keys."""

    def __init__(self, store: AnnotationStore, args: argparse.Namespace):
        super().__init__(store, args)
        self.threads = [self.thread]
        for index in range(1, max(1, args.luna_parallel)):
            thread = threading.Thread(
                target=self._run, daemon=True, name=f"luna-review-{index + 1}"
            )
            thread.start()
            self.threads.append(thread)

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def enqueue_for_cohort(self, request_id: str, cohort: str) -> dict:
        sample = self.store.sample(request_id)
        if not sample:
            raise KeyError(request_id)
        cache_key = sha256_text(self.cache_key(sample) + "\ncohort:" + cohort)
        job = self.store.create_or_get_job(request_id, cache_key)
        if job["state"] == "pending":
            self.queue.put(int(job["id"]))
        return job


class ControlPlaneHandler(BaseHTTPRequestHandler):
    server_version = "VoiceInkControlPlane/1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in {"/api/health", "/api/v1/health"}:
            self._json(self.server.health())
            return
        if path in {"/api/ready", "/api/v1/readiness"}:
            self._json(self.server.readiness())
            return
        if path.startswith("/api/v1/"):
            self._api_get(path, parse_qs(parsed.query))
            return
        if path.startswith("/api/"):
            self._legacy_get(path, parse_qs(parsed.query))
            return
        if path.startswith("/assets/"):
            self._static(path.lstrip("/"), cache_control="public, max-age=31536000, immutable")
            return
        if path == "/" and parsed.query:
            query = parse_qs(parsed.query)
            sample_id = query.get("id", [""])[0]
            view = query.get("view", ["recent"])[0]
            if sample_id:
                self._redirect(f"/samples/{quote(sample_id, safe='')}")
            else:
                target = {
                    "history": "/review/history", "cohort": "/review/queue",
                    "recent": "/review/recent",
                }.get(view, "/")
                self._redirect(target)
            return
        if (
            path in PAGE_PATHS
            or path.startswith("/samples/")
            or path.startswith("/data/cohorts/")
            or path.startswith("/data/releases/")
            or path.startswith("/models/")
            or path.startswith("/runs/")
        ):
            self._static("index.html", cache_control="no-store")
            return
        self._error(404, "Not found")

    def _api_get(self, path: str, query: dict[str, list[str]]) -> None:
        reader = self.server.reader
        try:
            if path == "/api/v1/stats":
                self._json(self.server.store.stats())
            elif path == "/api/v1/health":
                self._json(self.server.health())
            elif path == "/api/v1/overview":
                self._json(reader.overview())
            elif path == "/api/v1/system":
                self._json(reader.system())
            elif path == "/api/v1/samples":
                scope = query.get("scope", ["history"])[0]
                limit = int(query.get("limit", ["50"])[0])
                if scope == "recent":
                    items = self.server.store.recent(min(limit, 50))
                else:
                    items = self.server.store.history(
                        query=query.get("q", [""])[0],
                        status=query.get("status", [""])[0],
                        limit=limit, before=query.get("before", [""])[0],
                    )
                self._json({"items": items})
            elif path.startswith("/api/v1/samples/"):
                request_id = unquote(path.removeprefix("/api/v1/samples/"))
                sample = self.server.store.sample(request_id)
                self._json(sample) if sample else self._error(404, "Sample not found")
            elif path == "/api/v1/active-cohort":
                item = reader.active_cohort()
                self._json(item) if item else self._error(404, "No active cohort")
            elif path == "/api/v1/cohorts":
                self._json({"items": reader.cohorts()})
            elif path.startswith("/api/v1/cohorts/") and path.endswith("/queue"):
                name = unquote(path.removeprefix("/api/v1/cohorts/").removesuffix("/queue"))
                self._json(reader.cohort_queue(
                    name, review=query.get("review", ["all"])[0],
                    split=query.get("split", [""])[0],
                    stratum=query.get("stratum", [""])[0],
                    query=query.get("q", [""])[0],
                    limit=int(query.get("limit", ["200"])[0]),
                    offset=int(query.get("offset", ["0"])[0]),
                ))
            elif path.startswith("/api/v1/cohorts/") and path.endswith("/release-preview"):
                name = unquote(path.removeprefix("/api/v1/cohorts/").removesuffix("/release-preview"))
                self._json(release_preview(self.server.store.connection, name))
            elif path.startswith("/api/v1/cohorts/"):
                name = unquote(path.removeprefix("/api/v1/cohorts/"))
                item = reader.cohort(name, refresh=query.get("refresh", ["0"])[0] == "1")
                self._json(item) if item else self._error(404, "Cohort not found")
            elif path == "/api/v1/releases":
                self._json({"items": reader.releases()})
            elif path.startswith("/api/v1/releases/"):
                item = reader.release(unquote(path.removeprefix("/api/v1/releases/")))
                self._json(item) if item else self._error(404, "Release not found")
            elif path == "/api/v1/training-profiles":
                self._json({"items": reader.profiles()})
            elif path == "/api/v1/models":
                self._json({"items": reader.models()})
            elif path.startswith("/api/v1/models/"):
                item = reader.model(unquote(path.removeprefix("/api/v1/models/")))
                self._json(item) if item else self._error(404, "Model not found")
            elif path == "/api/v1/runs":
                self._json(reader.runs(
                    query.get("category", ["all"])[0],
                    int(query.get("limit", ["30"])[0]),
                ))
            elif path.startswith("/api/v1/runs/"):
                item = reader.run(unquote(path.removeprefix("/api/v1/runs/")))
                self._json(item) if item else self._error(404, "MLflow run not found")
            elif path == "/api/v1/docs/canonical":
                self._text((ROOT / "docs" / "CANONICAL_PIPELINE.md").read_text(encoding="utf-8"))
            elif path == "/api/v1/docs/review":
                self._text((ROOT / "docs" / "LIVE_REVIEW.md").read_text(encoding="utf-8"))
            else:
                self._error(404, "Not found")
        except KeyError as exc:
            self._error(404, f"Not found: {exc}")
        except (ValueError, OSError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _legacy_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/recent":
            self._json(self.server.store.recent(int(query.get("limit", ["10"])[0])))
        elif path == "/api/history":
            self._json(self.server.store.history(
                query=query.get("q", [""])[0], status=query.get("status", [""])[0],
                limit=int(query.get("limit", ["50"])[0]),
                before=query.get("before", [""])[0],
            ))
        elif path == "/api/stats":
            self._json(self.server.store.stats())
        elif path == "/api/cohort":
            cohorts = self.server.reader.cohorts()
            if not cohorts:
                self._json([])
            else:
                result = self.server.reader.cohort_queue(
                    cohorts[0]["name"], limit=int(query.get("limit", ["200"])[0])
                )
                self._json(result["items"])
        elif path.startswith("/api/samples/"):
            sample = self.server.store.sample(unquote(path.removeprefix("/api/samples/")))
            self._json(sample) if sample else self._error(404, "Sample not found")
        else:
            self._error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._body()
        try:
            if path.startswith("/api/v1/samples/"):
                parts = path.split("/")
                if len(parts) != 6:
                    self._error(404, "Not found")
                    return
                request_id, action = unquote(parts[4]), parts[5]
                if action == "analysis-jobs":
                    if body.get("retry"):
                        value = self.server.worker.retry(request_id)
                    elif body.get("cohort"):
                        value = self.server.worker.enqueue_for_cohort(
                            request_id, str(body["cohort"])
                        )
                    else:
                        value = self.server.worker.enqueue(request_id)
                    self._json(value, 202)
                elif action == "decisions":
                    self._json(self.server.save_decision(request_id, body), 201)
                else:
                    self._error(404, "Not found")
            elif path.startswith("/api/v1/cohorts/") and path.endswith("/analysis-jobs"):
                name = unquote(path.removeprefix("/api/v1/cohorts/").removesuffix("/analysis-jobs"))
                self._json(self.server.queue_cohort(name), 202)
            elif path.startswith("/api/v1/cohorts/") and path.endswith("/releases"):
                name = unquote(path.removeprefix("/api/v1/cohorts/").removesuffix("/releases"))
                release_name = str(body.get("name", "")).strip()
                if not release_name:
                    raise ValueError("Release name is required")
                result = create_release(
                    self.server.store.connection, name, release_name,
                    self.server.args.release_root, self.server.args.system_prompt,
                    correction_cutoff=body.get("correction_cutoff"),
                    selection_sha256=body.get("selection_sha256"),
                )
                self.server.reader._cohort_cache.pop(name, None)
                self._json(result, 201)
            elif path.startswith("/api/samples/"):
                parts = path.strip("/").split("/")
                request_id, action = unquote(parts[2]), parts[3]
                if action == "analyze":
                    self._json(self.server.worker.enqueue(request_id))
                elif action == "retry":
                    self._json(self.server.worker.retry(request_id))
                elif action == "decision":
                    self._json(self.server.save_decision(request_id, body))
                else:
                    self._error(404, "Not found")
            else:
                self._error(404, "Not found")
        except KeyError:
            self._error(404, "Sample or cohort not found")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, relative_path: str, *, cache_control: str) -> None:
        target = (STATIC_ROOT / unquote(relative_path)).resolve()
        if not target.is_relative_to(STATIC_ROOT.resolve()) or not target.is_file():
            self._error(404, "Static asset not found")
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, value: str) -> None:
        body = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, value: str) -> None:
        body = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, target: str) -> None:
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def log_message(self, format: str, *args) -> None:
        if args and str(args[0]).startswith("POST"):
            super().log_message(format, *args)


class ControlPlaneServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, store: AnnotationStore, worker: ControlPlaneWorker,
                 args: argparse.Namespace):
        super().__init__(address, ControlPlaneHandler)
        self.store, self.worker, self.args = store, worker, args
        self.reader = ControlPlaneState(
            store, mlflow_tracking_uri=args.mlflow_tracking_uri
        )
        self.ingester = ProxyLogIngester(store, args.log_file)
        self.maintenance = MaintenanceWorker(
            store,
            approved_export=args.approved_export,
            audit_mirror=args.audit_mirror,
            backup_dir=args.backup_dir,
            locked_eval=args.locked_eval,
        )
        self.ingester.start()
        self.maintenance.start()

    def server_close(self) -> None:
        self.ingester.stop()
        self.maintenance.stop()
        super().server_close()

    def queue_cohort(self, name: str) -> dict:
        cohort = self.reader.cohort(name, refresh=True)
        if not cohort:
            raise KeyError(name)
        rows = self.store.connection.execute(
            "SELECT m.request_id FROM canonical_cohort_members m WHERE m.cohort_id=? "
            "ORDER BY m.request_id", (cohort["id"],)
        )
        queued = completed = failed = 0
        for row in rows:
            sample = self.store.sample(row["request_id"])
            if not sample:
                continue
            analysis = self.store.connection.execute(
                "SELECT 1 FROM analyses x JOIN annotations a ON a.id=x.annotation_id "
                "WHERE x.request_id=? AND x.created_at>=? "
                "AND x.model=? AND x.reasoning_effort=? AND x.prompt_sha256=? "
                "AND a.model=? AND a.reasoning_effort=? AND a.prompt_sha256=? LIMIT 1",
                (row["request_id"], cohort["created_at"],
                 self.args.luna_model, self.args.luna_reasoning_effort,
                 self.worker.evaluator_prompt_sha, self.args.luna_model,
                 self.args.luna_reasoning_effort, self.worker.label_prompt_sha),
            ).fetchone()
            if analysis:
                completed += 1
                continue
            active = self.store.connection.execute(
                "SELECT 1 FROM jobs WHERE request_id=? AND state IN ('pending', 'running') LIMIT 1",
                (row["request_id"],),
            ).fetchone()
            if active:
                queued += 1
                continue
            job = self.worker.enqueue_for_cohort(row["request_id"], name)
            if job["state"] == "failed":
                failed += 1
            else:
                queued += 1
        return {"cohort": name, "queued": queued, "completed": completed,
                "failed": failed, "parallel": len(self.worker.threads)}

    def save_decision(self, request_id: str, body: dict) -> DecisionResponse:
        outcome = str(body.get("outcome", ""))
        context = str(body.get("context_dependency", "unknown"))
        if context not in {"unknown", "none", "window", "clipboard", "vocabulary"}:
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
            context_dependency=context,
        )
        self.reader._cohort_cache.clear()
        maintenance_job_id = self.maintenance.schedule(decision_id)
        return {
            "ok": True,
            "decision_id": decision_id,
            "annotation_id": annotation_id,
            "maintenance": "scheduled",
            "maintenance_job_id": maintenance_job_id,
        }

    def health(self) -> HealthResponse:
        ingestion = self.ingester.status()
        maintenance = self.maintenance.status()
        workers_alive = sum(thread.is_alive() for thread in self.worker.threads)
        healthy = bool(
            workers_alive == len(self.worker.threads)
            and ingestion["running"]
            and not ingestion["error"]
            and maintenance["running"]
        )
        return {
            "api_version": CONTROL_PLANE_API_VERSION,
            "status": "ok" if healthy else "degraded",
            "worker_threads": len(self.worker.threads),
            "workers_alive": workers_alive,
            "luna_model": self.args.luna_model,
            "luna_reasoning_effort": self.args.luna_reasoning_effort,
            "queue_depth": self.worker.queue.qsize(),
            "ingestion": ingestion,
            "maintenance": maintenance,
            "mlflow_tracking_uri": self.args.mlflow_tracking_uri,
        }

    def readiness(self) -> ReadinessResponse:
        started = time.monotonic()
        try:
            self.store.connection.execute("SELECT 1").fetchone()
            database = True
            error = ""
        except Exception as exc:
            database = False
            error = f"{type(exc).__name__}: {exc}"
        ingestion = self.ingester.status()
        ready = bool(database and ingestion["initialized"] and not ingestion["error"])
        return {
            "status": "ready" if ready else "not-ready",
            "database": database,
            "ingestion_initialized": ingestion["initialized"],
            "ingestion_lag_bytes": ingestion["lag_bytes"],
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "error": error or ingestion["error"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--approved-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--audit-mirror", type=Path, default=DEFAULT_AUDIT_MIRROR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUPS)
    parser.add_argument("--locked-eval", type=Path, default=DEFAULT_LOCKED_EVAL)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--luna-model", default=DEFAULT_MODEL)
    parser.add_argument("--luna-reasoning-effort", default=DEFAULT_EFFORT)
    parser.add_argument("--luna-parallel", type=int, default=3)
    add_mlflow_args(parser, default_experiment="voiceink-labeling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (STATIC_ROOT / "index.html").is_file():
        raise RuntimeError(
            "Control-plane frontend is missing; run 'npm --prefix ui run build'"
        )
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    store = AnnotationStore(args.database)
    migrate(store.connection)
    worker = ControlPlaneWorker(store, args)
    server = ControlPlaneServer((args.host, args.port), store, worker, args)
    print(f"VoiceInk control plane: http://{args.host}:{args.port}")
    print(f"Registry: {args.database} ({store.stats()['samples']} samples)")
    print(f"Incremental ingester: {args.log_file}")
    print(f"Luna workers: {len(worker.threads)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
