"""Canonical human-review and automatic-label selection policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PASSING_VALIDATIONS = frozenset({"valid", "pass", "approved"})


def is_fresh_human_decision(
    decision: Mapping[str, Any] | None, cohort_created_at: str
) -> bool:
    """Return whether a decision was explicitly made by a human for this cohort."""
    return bool(
        decision
        and decision["reviewer"] == "human"
        and (decision["decided_at"] or "") >= cohort_created_at
    )


def is_strong_luna_proposal(analysis: Mapping[str, Any] | None) -> bool:
    """Return whether Luna's proposal is safe to select without human review."""
    return bool(
        analysis
        and analysis["validation_status"] in PASSING_VALIDATIONS
        and analysis["confidence"] == "high"
        and analysis["preference"] == "proposal"
    )


def requires_human_review(
    member: Mapping[str, Any],
    analysis: Mapping[str, Any] | None,
    *,
    expanded_stratum: bool = False,
) -> bool:
    """Apply the single canonical escalation policy.

    High-confidence, validation-passing Luna wins are automatic even when the
    edit is material. Passing ties and high-confidence production wins keep the
    cohort-approved label. Validation-passing, non-material comparisons also keep that approved label,
    regardless of confidence. Explicit audits and genuinely
    ambiguous or unsafe proposal wins are reviewed by a human.
    """
    if member["audit_selected"] or expanded_stratum:
        return True
    if analysis is None:
        return False
    if (
        analysis["confidence"] == "high"
        and analysis["preference"] == "production"
    ):
        return False
    if analysis["validation_status"] in PASSING_VALIDATIONS:
        if not analysis["material_difference"]:
            return False
        if analysis["confidence"] == "high":
            return False
    return True


def resolved_annotation_id(
    member: Mapping[str, Any],
    analysis: Mapping[str, Any] | None,
    decision: Mapping[str, Any] | None,
    cohort_created_at: str,
) -> tuple[int, str]:
    """Resolve a cohort label without ever overriding a fresh human choice."""
    if is_fresh_human_decision(decision, cohort_created_at):
        annotation_id = decision["annotation_id"]
        if annotation_id is None:
            raise ValueError("A training-eligible human decision needs an annotation")
        return int(annotation_id), "human"
    if is_strong_luna_proposal(analysis):
        return int(analysis["annotation_id"]), "policy:luna"
    return int(member["annotation_id"]), "cohort:approved"
