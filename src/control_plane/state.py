"""Read models for the lightweight VoiceInk control plane.

The control plane deliberately does not become a second source of truth. It
joins the annotation registry, canonical manifests, model configuration, and
MLflow into small JSON-safe projections for the web UI.
"""

from __future__ import annotations

import configparser
import json
import socket
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    from data.cohort_review import cohort_queue_rows, cohort_status
except ModuleNotFoundError:  # Imported as src.control_plane during unit tests.
    from src.data.cohort_review import cohort_queue_rows, cohort_status

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_CONFIG = ROOT / "config" / "models.ini"
DEFAULT_PROFILES = ROOT / "config" / "training_profiles.toml"
DEFAULT_RELEASE_ROOT = ROOT / "datasets" / "releases"


def _iso_millis(value: int | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class ControlPlaneState:
    """Aggregate canonical state without copying private records elsewhere."""

    def __init__(self, store, *, mlflow_tracking_uri: str,
                 models_config: Path = DEFAULT_MODELS_CONFIG,
                 profiles: Path = DEFAULT_PROFILES,
                 release_root: Path = DEFAULT_RELEASE_ROOT):
        self.store = store
        self.mlflow_tracking_uri = mlflow_tracking_uri.rstrip("/")
        self.models_config = models_config
        self.profiles_path = profiles
        self.release_root = release_root
        self._cohort_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _cohort_status(self, name: str, *, refresh: bool = False) -> dict[str, Any]:
        cached = self._cohort_cache.get(name)
        if cached and not refresh and time.monotonic() - cached[0] < 10:
            return cached[1]
        value = cohort_status(self.store.connection, name)
        self._cohort_cache[name] = (time.monotonic(), value)
        return value

    def cohorts(self) -> list[dict[str, Any]]:
        rows = list(self.store.connection.execute(
            "SELECT * FROM canonical_cohorts ORDER BY id DESC"
        ))
        return [dict(row) | {"status": self._cohort_status(row["name"])} for row in rows]

    def active_cohort(self) -> dict[str, Any] | None:
        row = self.store.connection.execute(
            "SELECT id,name,state,created_at FROM canonical_cohorts "
            "WHERE state != 'released' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def cohort(self, name: str, *, refresh: bool = False) -> dict[str, Any] | None:
        row = self.store.connection.execute(
            "SELECT * FROM canonical_cohorts WHERE name=?", (name,)
        ).fetchone()
        if not row:
            return None
        return dict(row) | {"status": self._cohort_status(name, refresh=refresh)}

    def cohort_queue(self, name: str, *, review: str = "all", split: str = "",
                     stratum: str = "", query: str = "", limit: int = 200,
                     offset: int = 0) -> dict[str, Any]:
        _, rows = cohort_queue_rows(self.store.connection, name)
        items: list[dict[str, Any]] = []
        for item in rows:
            if split and item["split"] != split:
                continue
            if stratum and item["stratum"] != stratum:
                continue
            if query and query.casefold() not in (
                item["request_id"] + " " + item["transcript"]
            ).casefold():
                continue
            if review == "required" and not item["review_pending"]:
                continue
            if review == "unanalyzed" and item["fresh_luna"]:
                continue
            if review == "failed" and item["job_state"] != "failed":
                continue
            if review == "reviewed" and not item["human_reviewed"]:
                continue
            items.append(item)
        if review == "required":
            items.sort(key=lambda item: (
                not item["fresh_luna"],
                bool(item["audit_selected"]),
                item["split"] != "acceptance",
                item["timestamp"],
                item["request_id"],
            ))
        else:
            items.sort(key=lambda item: (
                item["job_state"] != "failed",
                item["fresh_luna"],
                not item["review_pending"],
                item["split"] != "acceptance",
                not item["audit_selected"],
                item["timestamp"],
                item["request_id"],
            ))
        bounded_offset = max(offset, 0)
        bounded_limit = min(max(limit, 1), 500)
        return {
            "cohort": name,
            "total": len(items),
            "offset": bounded_offset,
            "limit": bounded_limit,
            "items": items[bounded_offset:bounded_offset + bounded_limit],
        }

    def releases(self) -> list[dict[str, Any]]:
        rows = self.store.connection.execute(
            "SELECT r.*,c.name cohort_name,p.name parent_name FROM canonical_releases r "
            "JOIN canonical_cohorts c ON c.id=r.cohort_id "
            "LEFT JOIN canonical_releases p ON p.id=r.parent_release_id ORDER BY r.id DESC"
        )
        result = []
        for row in rows:
            item = dict(row)
            manifest_path = Path(item["manifest_path"])
            item["manifest_exists"] = manifest_path.is_file()
            item["manifest"] = _json_file(manifest_path)
            result.append(item)
        return result

    def release(self, name: str) -> dict[str, Any] | None:
        return next((item for item in self.releases() if item["name"] == name), None)

    def profiles(self) -> list[dict[str, Any]]:
        if not self.profiles_path.is_file():
            return []
        values = tomllib.loads(self.profiles_path.read_text(encoding="utf-8"))
        return [{"name": name, **profile} for name, profile in
                values.get("profiles", {}).items()]

    def models(self) -> list[dict[str, Any]]:
        parser = configparser.ConfigParser(interpolation=None)
        raw = self.models_config.read_text(encoding="utf-8")
        parser.read_string("[control-plane-meta]\n" + raw)
        result = []
        for name in parser.sections():
            if name in {"*", "control-plane-meta"} or "VoiceInk" not in name:
                continue
            path = Path(parser[name].get("model", ""))
            if "/canonical/" in str(path):
                state = "canonical"
            elif "/legacy/" in str(path):
                state = "legacy"
            else:
                state = "experimental"
            result.append({
                "name": name, "path": str(path), "exists": path.is_file(),
                "state": state, "context": parser[name].get("c", parser["*"].get("c", "")),
                "load_on_startup": parser[name].getboolean("load-on-startup", fallback=False),
                "bytes": path.stat().st_size if path.is_file() else 0,
            })
        result.sort(key=lambda item: (
            {"canonical": 0, "experimental": 1, "legacy": 2}[item["state"]],
            item["name"],
        ))
        return result

    def model(self, name: str) -> dict[str, Any] | None:
        item = next((value for value in self.models() if value["name"] == name), None)
        if not item:
            return None
        promotions = [dict(row) for row in self.store.connection.execute(
            "SELECT p.*,r.name release_name FROM model_promotions p "
            "JOIN canonical_releases r ON r.id=p.release_id "
            "WHERE p.model_name=? ORDER BY p.id DESC", (name,)
        )]
        return item | {"promotions": promotions}

    def _mlflow_runs(self, experiment_name: str, limit: int = 20) -> tuple[list[dict], str]:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return [], ""
            runs = client.search_runs(
                [experiment.experiment_id], order_by=["attributes.start_time DESC"],
                max_results=min(max(limit, 1), 100),
            )
            result = []
            for run in runs:
                tags = dict(run.data.tags)
                metrics = dict(run.data.metrics)
                params = dict(run.data.params)
                preferred_metrics = (
                    "eval_loss", "train_loss", "overall_score",
                    "baseline_overall", "candidate_overall",
                    "candidate_win_rate", "latency_p50_ms", "latency_p90_ms",
                )
                summary_metrics = {
                    key: metrics[key] for key in preferred_metrics if key in metrics
                }
                result.append({
                    "run_id": run.info.run_id,
                    "experiment_id": experiment.experiment_id,
                    "experiment": experiment_name,
                    "name": tags.get("mlflow.runName", run.info.run_id[:8]),
                    "status": run.info.status,
                    "start_time": _iso_millis(run.info.start_time),
                    "end_time": _iso_millis(run.info.end_time),
                    "kind": tags.get("voiceink.run_kind", ""),
                    "release": tags.get("voiceink.dataset_release", params.get("dataset_release", "")),
                    "profile": tags.get("voiceink.training_profile", ""),
                    "baseline": tags.get("voiceink.baseline_model", params.get("baseline_model", "")),
                    "candidate": tags.get("voiceink.candidate_model", params.get("candidate_model", "")),
                    "metrics": summary_metrics,
                    "mlflow_url": (
                        f"{self.mlflow_tracking_uri}/#/experiments/"
                        f"{experiment.experiment_id}/runs/{run.info.run_id}"
                    ),
                })
            return result, ""
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

    def runs(self, category: str = "all", limit: int = 30) -> dict[str, Any]:
        experiments = {
            "training": ["voiceink-training"],
            "evaluations": ["voiceink-evaluation", "voiceink-benchmarks"],
            "all": ["voiceink-training", "voiceink-evaluation", "voiceink-benchmarks"],
        }.get(category, ["voiceink-training", "voiceink-evaluation"])
        result: list[dict] = []
        errors = []
        for experiment in experiments:
            rows, error = self._mlflow_runs(experiment, limit)
            result.extend(rows)
            if error:
                errors.append(f"{experiment}: {error}")
        result.sort(key=lambda item: item["start_time"], reverse=True)
        return {"items": result[:limit], "errors": errors,
                "mlflow_url": self.mlflow_tracking_uri}

    def run(self, run_id: str) -> dict[str, Any] | None:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            run = client.get_run(run_id)
            experiment = client.get_experiment(run.info.experiment_id)
            tags = dict(run.data.tags)
            return {
                "run_id": run_id, "experiment_id": run.info.experiment_id,
                "experiment": experiment.name if experiment else "",
                "name": tags.get("mlflow.runName", run_id[:8]),
                "status": run.info.status,
                "start_time": _iso_millis(run.info.start_time),
                "end_time": _iso_millis(run.info.end_time),
                "metrics": dict(run.data.metrics), "params": dict(run.data.params),
                "tags": tags,
                "mlflow_url": f"{self.mlflow_tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run_id}",
            }
        except Exception:
            return None

    @staticmethod
    def _http_health(url: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            with urlopen(url, timeout=1.5) as response:
                ok = 200 <= response.status < 400
            return {"ok": ok, "latency_ms": round((time.monotonic() - started) * 1000)}
        except (OSError, URLError) as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _tcp_health(host: str, port: int) -> dict[str, Any]:
        started = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=1.5):
                pass
            return {"ok": True, "latency_ms": round((time.monotonic() - started) * 1000)}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

    def system(self) -> dict[str, Any]:
        services = {
            "llama-server": self._http_health("http://127.0.0.1:8002/health"),
            "voiceink-proxy": self._tcp_health("127.0.0.1", 8001),
            "mlflow": self._http_health(f"{self.mlflow_tracking_uri}/health"),
            "control-plane": {"ok": True},
        }
        gpu: dict[str, Any] = {}
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu = {
                "name": pynvml.nvmlDeviceGetName(handle),
                "memory_used": memory.used, "memory_total": memory.total,
                "gpu_utilization": utilization.gpu,
            }
            pynvml.nvmlShutdown()
        except Exception as exc:
            gpu = {"available": False, "error": str(exc)}
        return {"services": services, "gpu": gpu,
                "all_healthy": all(item["ok"] for item in services.values())}

    def overview(self) -> dict[str, Any]:
        cohorts = self.cohorts()
        releases = self.releases()
        recent = self.store.recent(1)
        return {
            "stats": self.store.stats(),
            "active_cohort": next(
                (item for item in cohorts if item["state"] != "released"), None
            ),
            "latest_release": releases[0] if releases else None,
            "production_model": recent[0]["production_model"] if recent else "",
        }

