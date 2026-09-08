"""Typed JSON contracts shared by control-plane handlers and tests."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ErrorResponse(TypedDict):
    error: str


class IngestionResponse(TypedDict):
    source_path: str
    running: bool
    initialized: bool
    source_inode: int
    byte_offset: int
    source_bytes: int
    line_number: int
    imported_records: int
    malformed_records: int
    lag_bytes: int
    last_ingested_at: str
    error: str


class MaintenanceResponse(TypedDict):
    running: bool
    pending: int
    last_completed_at: str
    last_backup: str
    last_export: str
    error: str


class HealthResponse(TypedDict):
    api_version: str
    status: str
    worker_threads: int
    workers_alive: int
    luna_model: str
    luna_reasoning_effort: str
    queue_depth: int
    ingestion: IngestionResponse
    maintenance: MaintenanceResponse
    mlflow_tracking_uri: str


class ReadinessResponse(TypedDict):
    status: str
    database: bool
    ingestion_initialized: bool
    ingestion_lag_bytes: int
    latency_ms: float
    error: str


class DecisionResponse(TypedDict):
    ok: bool
    decision_id: int
    annotation_id: int | None
    maintenance: str
    maintenance_job_id: int


class QueueResponse(TypedDict):
    cohort: str
    total: int
    offset: int
    limit: int
    items: list[dict[str, Any]]


class CollectionResponse(TypedDict):
    items: list[dict[str, Any]]
    next_cursor: NotRequired[str]
