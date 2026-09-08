"""Canonical, bounded, side-effect-free cohort review projections."""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from data.review_policy import is_fresh_human_decision, requires_human_review
    from labeling.annotation_store import normalize_transcript
except ModuleNotFoundError:  # Imported as src.control_plane during unit tests.
    from src.data.review_policy import is_fresh_human_decision, requires_human_review
    from src.labeling.annotation_store import normalize_transcript

EXPECTED_SPLITS = {"train": 900, "validation": 150, "acceptance": 150}
AUDIT_EXPANSION_MIN_REVIEWS = 20
AUDIT_EXPANSION_ERROR_RATE = 0.05


def _cohort(connection, name: str):
    row = connection.execute(
        "SELECT * FROM canonical_cohorts WHERE name=?",
        (name,),
    ).fetchone()
    if not row:
        raise KeyError(name)
    return row


def cohort_review_rows(connection, cohort_id: int, created_at: str) -> list[dict]:
    """Load all cohort review state in one set-based query."""
    rows = connection.execute(
        """
        WITH ranked_analysis AS (
            SELECT a.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.request_id ORDER BY a.id DESC
                   ) AS rank
            FROM analyses a
            JOIN canonical_cohort_members cm
              ON cm.request_id=a.request_id AND cm.cohort_id=?
            WHERE a.created_at>=? AND lower(a.model) LIKE '%luna%'
        ),
        latest_analysis AS (
            SELECT * FROM ranked_analysis WHERE rank=1
        ),
        ranked_job AS (
            SELECT j.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY j.request_id ORDER BY j.id DESC
                   ) AS rank
            FROM jobs j
            JOIN canonical_cohort_members cm
              ON cm.request_id=j.request_id AND cm.cohort_id=?
        ),
        latest_job AS (
            SELECT * FROM ranked_job WHERE rank=1
        )
        SELECT
            m.request_id,
            m.annotation_id,
            m.split,
            m.stratum,
            m.audit_selected,
            s.timestamp,
            s.transcript,
            s.production_model,
            s.production_output,
            s.transcript_fingerprint,
            d.id AS decision_id,
            d.annotation_id AS decision_annotation_id,
            d.outcome,
            d.reviewer,
            d.training_eligible,
            d.context_dependency,
            d.decided_at,
            j.id AS job_id,
            j.state AS job_state,
            j.error AS job_error,
            j.updated_at AS job_updated_at,
            a.id AS analysis_id,
            a.annotation_id AS analysis_annotation_id,
            a.validation_status,
            a.preference,
            a.confidence,
            a.material_difference,
            a.created_at AS analysis_created_at,
            proposal.text AS proposal_text,
            chosen.text AS decision_text
        FROM canonical_cohort_members m
        JOIN samples s ON s.request_id=m.request_id
        LEFT JOIN decisions d
          ON d.request_id=m.request_id AND d.is_current=1
        LEFT JOIN latest_job j ON j.request_id=m.request_id
        LEFT JOIN latest_analysis a ON a.request_id=m.request_id
        LEFT JOIN annotations proposal ON proposal.id=a.annotation_id
        LEFT JOIN annotations chosen ON chosen.id=d.annotation_id
        WHERE m.cohort_id=?
        """,
        (cohort_id, created_at, cohort_id, cohort_id),
    )
    return [dict(row) for row in rows]


def decorate_review_rows(rows: list[dict], created_at: str) -> tuple[list[dict], dict]:
    """Apply the canonical review policy without querying or mutating SQLite."""
    audit_stats: dict[str, list[int]] = {}
    for item in rows:
        decision = item if item["decision_id"] is not None else None
        if not item["audit_selected"] or not is_fresh_human_decision(
            decision, created_at
        ):
            continue
        if item["proposal_text"] is None or item["decision_text"] is None:
            continue
        values = audit_stats.setdefault(item["stratum"], [0, 0])
        values[0] += 1
        values[1] += int(
            normalize_transcript(item["proposal_text"])
            != normalize_transcript(item["decision_text"])
        )

    expanded = {
        stratum
        for stratum, (reviewed, errors) in audit_stats.items()
        if reviewed >= AUDIT_EXPANSION_MIN_REVIEWS
        and errors / reviewed > AUDIT_EXPANSION_ERROR_RATE
    }
    critical = 0
    for item in rows:
        analysis = item if item["analysis_id"] is not None else None
        decision = item if item["decision_id"] is not None else None
        human_reviewed = is_fresh_human_decision(decision, created_at)
        required = requires_human_review(
            item,
            analysis,
            expanded_stratum=item["stratum"] in expanded,
        )
        item.update(
            {
                "fresh_luna": analysis is not None,
                "human_reviewed": human_reviewed,
                "requires_human": required,
                "review_pending": required and not human_reviewed,
            }
        )
        critical += int(
            required
            and not human_reviewed
            and analysis is not None
            and item["validation_status"] not in {"valid", "pass", "approved"}
        )
    return rows, {
        "critical": critical,
        "expanded": expanded,
        "audit_stats": audit_stats,
    }


def cohort_status(connection, name: str) -> dict[str, Any]:
    """Return cohort readiness without changing cohort or member rows."""
    cohort = _cohort(connection, name)
    rows, policy = decorate_review_rows(
        cohort_review_rows(connection, int(cohort["id"]), cohort["created_at"]),
        cohort["created_at"],
    )
    split_counts = Counter(item["split"] for item in rows)
    stratum_counts = Counter((item["stratum"], item["split"]) for item in rows)
    fresh_luna = sum(item["fresh_luna"] for item in rows)
    human = sum(item["human_reviewed"] for item in rows)
    required = sum(item["requires_human"] for item in rows)
    required_done = sum(
        item["requires_human"] and item["human_reviewed"] for item in rows
    )
    ready = bool(
        len(rows) == 1200
        and fresh_luna == 1200
        and required_done == required
        and policy["critical"] == 0
        and dict(split_counts) == EXPECTED_SPLITS
    )
    return {
        "name": name,
        "state": "ready" if ready else "reviewing",
        "ready": ready,
        "members": len(rows),
        "fresh_luna": fresh_luna,
        "human_reviewed": human,
        "required_human": required,
        "required_human_done": required_done,
        "unresolved_critical": policy["critical"],
        "audit": {
            key: {
                "reviewed": value[0],
                "errors": value[1],
                "error_rate": value[1] / value[0],
            }
            for key, value in policy["audit_stats"].items()
        },
        "expanded_review_strata": sorted(policy["expanded"]),
        "splits": dict(split_counts),
        "strata": {
            f"{key[0]}/{key[1]}": value for key, value in stratum_counts.items()
        },
    }


def cohort_queue_rows(connection, name: str) -> tuple[dict, list[dict]]:
    cohort = _cohort(connection, name)
    rows, _ = decorate_review_rows(
        cohort_review_rows(connection, int(cohort["id"]), cohort["created_at"]),
        cohort["created_at"],
    )
    return dict(cohort), rows
