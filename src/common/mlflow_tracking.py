"""Shared, privacy-safe MLflow tracking for VoiceInk workflows."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import re
import subprocess
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://192.168.1.150:5000"
)


def add_mlflow_args(
    parser: ArgumentParser,
    *,
    default_experiment: str,
) -> None:
    """Add consistent MLflow controls to a CLI parser."""
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", default_experiment),
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Override the generated MLflow run name",
    )
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Resume and enrich an existing MLflow run instead of creating one",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="mlflow_enabled",
        action="store_false",
        help="Explicitly disable MLflow for this invocation",
    )
    parser.set_defaults(mlflow_enabled=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def dataset_metadata(role: str, path: Path, records: int) -> dict[str, Any]:
    """Describe a private dataset without uploading any examples."""
    return {
        "role": role,
        "path": display_path(path),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "raw_data_logged": False,
    }


def output_reference(role: str, path: Path, *, private: bool) -> dict[str, Any]:
    """Describe an output in place without copying it into MLflow artifacts."""
    reference: dict[str, Any] = {
        "role": role,
        "path": display_path(path),
        "exists": path.exists(),
        "private": private,
        "uploaded": False,
    }
    if path.is_file():
        reference["bytes"] = path.stat().st_size
        reference["sha256"] = file_sha256(path)
    elif path.is_dir():
        reference["files"] = sum(1 for item in path.rglob("*") if item.is_file())
    return reference


def _git_metadata() -> dict[str, str]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "git.commit": run("rev-parse", "HEAD"),
        "git.branch": run("branch", "--show-current"),
        "git.dirty": str(bool(run("status", "--porcelain"))).lower(),
    }


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten finite numeric leaves for MLflow metric logging."""
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            metrics.update(numeric_metrics(value, name))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[name] = float(value)
    return metrics


def mlflow_model_name(display_name: str) -> str:
    """Convert a production display name to MLflow's restricted model name."""
    name = re.sub(r"[/:.%\"']+", "-", display_name).strip("-")
    if not name:
        raise ValueError(
            f"Invalid empty MLflow model name derived from {display_name!r}"
        )
    return name


def resumed_param_updates(
    existing: dict[str, str], requested: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Split new params from immutable value changes on a resumed MLflow run."""
    additions = {key: value for key, value in requested.items() if key not in existing}
    changes = {
        key: {"original": existing[key], "resumed": str(value)}
        for key, value in requested.items()
        if key in existing and existing[key] != str(value)
    }
    return additions, changes


class MlflowRun:
    """Small lifecycle wrapper that marks abandoned Python runs as failed."""

    def __init__(
        self,
        module,
        run_id: str | None,
        run_name: str,
        datasets_by_role: dict[str, Any] | None = None,
    ):
        self._mlflow = module
        self.run_id = run_id
        self.run_name = run_name
        self._datasets_by_role = datasets_by_role or {}
        self._open = run_id is not None
        if self._open:
            atexit.register(self._fail_if_open)

    @property
    def enabled(self) -> bool:
        return self.run_id is not None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._open:
            return
        values = numeric_metrics(metrics)
        if values:
            self._mlflow.log_metrics(values, step=step, synchronous=True)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        if self._open:
            self._mlflow.log_dict(payload, artifact_file)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self._open:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self._open:
            self._mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def register_prompt(
        self,
        *,
        name: str,
        template: str,
        sha256: str,
        alias: str = "current",
    ) -> str | None:
        """Version a non-private prompt once and link it to this run."""
        if not self._open:
            return None
        client = self._mlflow.tracking.MlflowClient()
        existing_versions = (
            client.search_prompt_versions(name=name)
            if client.get_prompt(name) is not None
            else []
        )
        prompt = next(
            (
                version
                for version in existing_versions
                if (version.tags or {}).get("voiceink.sha256") == sha256
            ),
            None,
        )
        if prompt is None:
            prompt = self._mlflow.genai.register_prompt(
                name=name,
                template=template,
                commit_message=f"VoiceInk prompt {sha256[:12]}",
                tags={
                    "voiceink.sha256": sha256,
                    "voiceink.private": "false",
                },
            )
        self._mlflow.genai.set_prompt_alias(name, alias, int(prompt.version))
        client.link_prompt_version_to_run(self.run_id, prompt)
        uri = f"prompts:/{name}/{prompt.version}"
        self.set_tags({"voiceink.prompt_uri": uri})
        return uri

    def log_external_model(
        self,
        *,
        name: str,
        model_type: str,
        params: dict[str, Any],
        tags: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        dataset_role: str | None = None,
    ) -> str | None:
        """Create an MLflow 3 Logged Model without uploading local weights."""
        if not self._open:
            return None
        mlflow_name = mlflow_model_name(name)
        model = self._mlflow.create_external_model(
            name=mlflow_name,
            source_run_id=self.run_id,
            model_type=model_type,
            params={key: str(_param_value(value)) for key, value in params.items()},
            tags={
                "voiceink.weights_location": "local-only",
                "voiceink.binary_uploaded": "false",
                "voiceink.display_name": name,
                **{key: str(value) for key, value in (tags or {}).items()},
            },
        )
        values = numeric_metrics(metrics or {})
        if values:
            self._mlflow.log_metrics(
                values,
                model_id=model.model_id,
                dataset=self._datasets_by_role.get(dataset_role),
                synchronous=True,
            )
        self.set_tags({"voiceink.logged_model_id": model.model_id})
        return model.model_id

    def finish(self, status: str = "FINISHED") -> None:
        if not self._open:
            return
        active = self._mlflow.active_run()
        if active and active.info.run_id == self.run_id:
            self._mlflow.end_run(status=status)
        else:
            self._mlflow.tracking.MlflowClient().set_terminated(
                self.run_id, status=status
            )
        self._open = False

    def _fail_if_open(self) -> None:
        if self._open:
            try:
                self.finish("FAILED")
            except Exception:
                pass


def start_mlflow_run(
    args: Namespace,
    *,
    run_name: str,
    run_kind: str,
    params: dict[str, Any],
    datasets: Iterable[dict[str, Any]] = (),
    tags: dict[str, Any] | None = None,
) -> MlflowRun:
    """Start a run and log reproducibility metadata, never raw private rows."""
    resolved_name = args.mlflow_run_name or run_name
    if not args.mlflow_enabled:
        return MlflowRun(None, None, resolved_name)

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled but mlflow is not installed; install it "
            "or pass --no-mlflow"
        ) from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    active = mlflow.active_run()
    log_system_metrics = (
        run_kind.startswith("training.") and run_kind != "training.backfill"
    )
    requested_run_id = getattr(args, "mlflow_run_id", None)
    if requested_run_id:
        if active is not None:
            raise RuntimeError("--mlflow-run-id cannot be used inside an active MLflow run")
        run = mlflow.start_run(run_id=requested_run_id)
    else:
        run = mlflow.start_run(
            run_name=resolved_name,
            nested=active is not None,
            description=(
                f"VoiceInk {run_kind}; private rows and local model binaries are not uploaded."
            ),
            log_system_metrics=log_system_metrics,
        )

    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment
    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "false"
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "true"
    # Automatic GenAI tracing captures function inputs and outputs. VoiceInk
    # transcripts are private, so only aggregate evaluation metrics are logged.
    os.environ["MLFLOW_TRACING_ENABLED"] = "false"

    dataset_records = list(datasets)
    native_datasets: dict[str, Any] = {}
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset

    for metadata in dataset_records:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            source = resolve_dataset_source(metadata["path"])
        dataset = MetaDataset(
            source=source,
            name=(
                "voiceink-"
                + metadata["path"].replace("/", "-").replace(".", "-")
            ),
            # MLflow Dataset entities cap digests at 36 characters. The full
            # SHA-256 remains in metadata/datasets.json and the input tags.
            digest=metadata["sha256"][:36],
        )
        native_datasets[metadata["role"]] = dataset
        mlflow.log_input(
            dataset,
            context=metadata["role"],
            tags={
                "voiceink.records": str(metadata["records"]),
                "voiceink.bytes": str(metadata["bytes"]),
                "voiceink.sha256": metadata["sha256"],
                "voiceink.raw_data_logged": "false",
            },
        )

    handle = MlflowRun(
        mlflow,
        run.info.run_id,
        resolved_name,
        datasets_by_role=native_datasets,
    )

    parameter_values = {
        key: _param_value(value) for key, value in params.items()
    }
    resumed_changes: dict[str, dict[str, str]] = {}
    if requested_run_id:
        existing_params = mlflow.get_run(run.info.run_id).data.params
        parameter_values, resumed_changes = resumed_param_updates(
            existing_params, parameter_values
        )
    if parameter_values:
        mlflow.log_params(parameter_values)
    run_tags = {
        "voiceink.run_kind": run_kind,
        "voiceink.mlflow_experience": (
            "model-training" if run_kind.startswith("training.") else "genai"
        ),
        "voiceink.raw_data_logged": "false",
        "voiceink.tracing_enabled": "false",
        "voiceink.system_metrics_enabled": str(log_system_metrics).lower(),
        "voiceink.resumed": str(bool(requested_run_id)).lower(),
        **_git_metadata(),
        **(tags or {}),
    }
    if resumed_changes:
        run_tags["voiceink.resumed_param_changes"] = json.dumps(
            resumed_changes, sort_keys=True
        )
    handle.set_tags(run_tags)
    handle.log_dict(
        {
            "privacy": "Fingerprints and counts only; raw dataset rows are not logged.",
            "datasets": dataset_records,
        },
        "metadata/datasets.json",
    )
    print(
        f"MLflow: {args.mlflow_experiment}/{resolved_name} "
        f"({run.info.run_id}) -> {args.mlflow_tracking_uri}"
    )
    return handle
