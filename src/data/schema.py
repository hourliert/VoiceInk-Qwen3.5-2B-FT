"""Schema v2 for canonical VoiceInk data cohorts and immutable releases."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_cohorts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'selected',
    seed INTEGER NOT NULL,
    specification_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS canonical_cohort_members (
    cohort_id INTEGER NOT NULL REFERENCES canonical_cohorts(id) ON DELETE CASCADE,
    request_id TEXT NOT NULL REFERENCES samples(request_id),
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    split TEXT NOT NULL CHECK(split IN ('train','validation','acceptance')),
    stratum TEXT NOT NULL,
    audit_selected INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(cohort_id, request_id),
    UNIQUE(cohort_id, annotation_id)
);
CREATE INDEX IF NOT EXISTS canonical_cohort_split_idx
ON canonical_cohort_members(cohort_id, split, stratum);
CREATE TABLE IF NOT EXISTS canonical_releases (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    cohort_id INTEGER NOT NULL REFERENCES canonical_cohorts(id),
    parent_release_id INTEGER REFERENCES canonical_releases(id),
    manifest_sha256 TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    records INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    sealed INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS canonical_release_members (
    release_id INTEGER NOT NULL REFERENCES canonical_releases(id) ON DELETE CASCADE,
    request_id TEXT NOT NULL REFERENCES samples(request_id),
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    split TEXT NOT NULL,
    stratum TEXT NOT NULL,
    PRIMARY KEY(release_id, request_id)
);
CREATE TABLE IF NOT EXISTS model_promotions (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    release_id INTEGER NOT NULL REFERENCES canonical_releases(id),
    mlflow_model_name TEXT NOT NULL DEFAULT '',
    mlflow_model_version TEXT NOT NULL DEFAULT '',
    regression_summary_sha256 TEXT NOT NULL,
    acceptance_summary_sha256 TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    promoted_at TEXT NOT NULL,
    UNIQUE(model_name, model_version)
);
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Apply additive schema-v2 objects without rewriting private records."""
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('canonical_version',?)",
        (str(SCHEMA_VERSION),),
    )
