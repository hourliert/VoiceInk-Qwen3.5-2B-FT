import json
import tempfile
import unittest
from pathlib import Path

from src.labeling.annotation_store import AnnotationStore, sha256_text
from src.labeling.live_review_server import (
    ANALYSIS_SCHEMA,
    EVALUATOR_PROMPT,
    HTML_PAGE,
    AnalysisWorker,
    BestEffortTracking,
    read_recent_log,
)
from src.labeling.migrate_annotations import import_file, inventory


def proxy_record(request_id: str = "request-1", transcript: str = "Okay, use Monza.") -> dict:
    request = {
        "model": "Qwen3.5-2B-VoiceInk-v3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "<SYSTEM_INSTRUCTIONS>Clean.</SYSTEM_INSTRUCTIONS>"
                    "<CURRENT_WINDOW_CONTEXT>Monza circuit</CURRENT_WINDOW_CONTEXT>"
                    "<CLIPBOARD_CONTEXT></CLIPBOARD_CONTEXT>"
                    "<CUSTOM_VOCABULARY>Monza</CUSTOM_VOCABULARY>"
                ),
            },
            {"role": "user", "content": f"<TRANSCRIPT>{transcript}</TRANSCRIPT>"},
        ],
    }
    return {
        "request_id": request_id,
        "timestamp": "2026-08-16T12:00:00+00:00",
        "path": "/v1/chat/completions",
        "request_json_valid": True,
        "response_json_valid": True,
        "status_code": 200,
        "duration_ms": 123.0,
        "model": "Qwen3.5-2B-VoiceInk-v3",
        "raw_request_json": json.dumps(request),
        "raw_response_json": "{}",
        "response_text": "Use Monza.",
    }


def evaluation_result() -> dict:
    scores = {
        "meaning_preservation": 5,
        "filler_removal": 5,
        "grammar_fluency": 5,
        "technical_accuracy": 5,
        "conciseness": 5,
        "instruction_following": 5,
    }
    return {
        "label_validation": {"status": "pass", "type": "", "reason": ""},
        "production_scores": scores,
        "proposal_scores": scores,
        "context_analysis": {"production": "None", "proposal": "None"},
        "score_analysis": {"production": "None", "proposal": "None"},
        "pairwise": {
            "preference": "tie", "confidence": "high",
            "material_difference": False, "reason": "Equivalent.",
        },
    }


class LiveReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(self.root / "annotations.sqlite3")
        self.addCleanup(self.store.close)
        self.assertTrue(self.store.import_proxy_record(proxy_record()))

    def test_annotations_and_human_edits_are_immutable_revisions(self) -> None:
        luna = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna", provider="codex",
            model="gpt-5.6-luna", prompt_sha256="label-prompt",
        )
        human = self.store.create_annotation(
            "request-1", "Please use Monza.", origin="human", supersedes_id=luna,
        )
        self.store.decide("request-1", "human_edit", annotation_id=human)

        sample = self.store.sample("request-1")

        self.assertEqual([row["text"] for row in sample["annotations"]], [
            "Use Monza.", "Please use Monza."
        ])
        self.assertEqual(sample["annotations"][1]["supersedes_id"], luna)
        self.assertEqual(sample["decision"]["annotation_id"], human)

    def test_exclude_supersedes_approval_without_deleting_label(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna"
        )
        self.store.decide("request-1", "accept_luna", annotation_id=annotation)
        self.store.decide("request-1", "exclude", annotation_id=None)

        sample = self.store.sample("request-1")

        self.assertEqual(len(sample["annotations"]), 1)
        self.assertEqual(sample["decision"]["outcome"], "exclude")
        self.assertFalse(sample["decision"]["training_eligible"])

    def test_analysis_and_job_cache_are_bound_to_exact_annotation(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna"
        )
        first = self.store.create_or_get_job("request-1", "cache")
        second = self.store.create_or_get_job("request-1", "cache")
        analysis = self.store.create_analysis(
            "request-1", annotation, "cache", evaluation_result(),
            provider="codex", model="gpt-5.6-luna", reasoning_effort="low",
            prompt_sha256="eval-prompt", blinded_assignment="a=production",
        )
        self.store.update_job(first["id"], "completed", annotation_id=annotation,
                              analysis_id=analysis)

        self.assertEqual(first["id"], second["id"])
        sample = self.store.sample("request-1")
        self.assertEqual(sample["job"]["state"], "completed")
        self.assertEqual(sample["analyses"][0]["annotation_id"], annotation)

    def test_export_is_atomic_compatible_and_excludes_locked_transcript(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna"
        )
        self.store.decide("request-1", "accept_luna", annotation_id=annotation)
        locked = self.root / "locked.jsonl"
        locked.write_text(json.dumps({
            "messages": [
                {"role": "system", "content": "Clean."},
                {"role": "user", "content": "<TRANSCRIPT>Different.</TRANSCRIPT>"},
                {"role": "assistant", "content": "Different."},
            ]
        }) + "\n")
        output = self.root / "approved.jsonl"

        manifest = self.store.export_approved(output, locked)
        record = json.loads(output.read_text().strip())

        self.assertEqual(manifest["records"], 1)
        self.assertEqual(record["label"], "Use Monza.")
        self.assertTrue(record["manually_reviewed"])
        self.assertIn("annotation_registry", record)

    def test_release_membership_matches_export(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna"
        )
        self.store.decide("request-1", "accept_luna", annotation_id=annotation)
        locked = self.root / "empty-locked.jsonl"
        locked.write_text("")

        result = self.store.create_release("release-v1", self.root / "release", locked)

        memberships = [json.loads(line) for line in
                       (self.root / "release" / "membership.jsonl").read_text().splitlines()]
        self.assertEqual(result["records"], 1)
        self.assertEqual(memberships[0]["annotation_id"], annotation)

    def test_audit_mirror_and_online_backup(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna"
        )
        self.store.decide("request-1", "accept_luna", annotation_id=annotation)

        mirrored = self.store.mirror_audit_events(self.root / "events.jsonl")
        backup = self.store.backup(self.root / "backups")

        self.assertGreater(mirrored, 0)
        self.assertTrue(backup.is_file())
        restored = AnnotationStore(backup)
        try:
            self.assertEqual(restored.integrity_check(), "ok")
            self.assertEqual(restored.stats()["reviewed"], 1)
        finally:
            restored.close()

    def test_recent_log_filters_invalid_and_non_chat_rows(self) -> None:
        log = self.root / "proxy.jsonl"
        rows = [proxy_record(), proxy_record("bad") | {"status_code": 500},
                proxy_record("health") | {"path": "/health"}]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))

        recent = read_recent_log(log)

        self.assertEqual([row["request_id"] for row in recent], ["request-1"])

    def test_blinded_result_maps_back_to_named_outputs(self) -> None:
        scores_a = {key: 5 for key in evaluation_result()["production_scores"]}
        scores_b = scores_a | {"technical_accuracy": 3}
        parsed = {
            "label_validation": {"status": "pass", "type": "", "reason": ""},
            "output_a": scores_a, "output_b": scores_b,
            "context_analysis": {"output_a": "A context", "output_b": "B context"},
            "score_analysis": {"output_a": "A score", "output_b": "B score"},
            "pairwise": {
                "preference": "output_a", "confidence": "high",
                "material_difference": True, "reason": "A is better.",
            },
        }

        result = AnalysisWorker._normalize_evaluation(parsed, a_is_production=False)

        self.assertEqual(result["pairwise"]["preference"], "proposal")
        self.assertEqual(result["production_scores"]["technical_accuracy"], 3)
        self.assertEqual(result["context_analysis"]["proposal"], "A context")

    def test_live_evaluator_is_reference_free_and_schema_is_strict(self) -> None:
        prompt = EVALUATOR_PROMPT.read_text(encoding="utf-8")

        self.assertNotIn("HUMAN-REVIEWED REFERENCE", prompt)
        self.assertIn("PROPOSED LABEL", prompt)
        self.assertFalse(ANALYSIS_SCHEMA["additionalProperties"])

    def test_mlflow_failures_are_best_effort(self) -> None:
        class BrokenTracking:
            run_id = "run-1"

            def log_metrics(self, *_args, **_kwargs):
                raise ConnectionError("offline")

        tracking = BestEffortTracking(BrokenTracking())

        self.assertEqual(tracking.run_id, "run-1")
        self.assertIsNone(tracking.log_metrics({"score": 1}))
        self.assertIsNone(BestEffortTracking().register_prompt(name="test"))

    def test_retry_resets_exhausted_attempts(self) -> None:
        job = self.store.create_or_get_job("request-1", "cache-key")
        self.store.update_job(job["id"], "failed", attempts=3, error="boom")

        retried = self.store.retry_job("request-1")

        self.assertEqual(retried["state"], "pending")
        self.assertEqual(retried["attempts"], 0)
        self.assertEqual(retried["error"], "")

    def test_recovery_resets_exhausted_pending_job(self) -> None:
        job = self.store.create_or_get_job("request-1", "cache-key")
        self.store.update_job(job["id"], "pending", attempts=3, error="stuck")

        recovered = self.store.recover_jobs()
        current = self.store.pending_jobs()[0]

        self.assertEqual(recovered, 1)
        self.assertEqual(current["attempts"], 0)

    def test_review_url_preserves_sample_id(self) -> None:
        self.assertIn("searchParams.set(\"id\",id)", HTML_PAGE)
        self.assertIn("searchParams.get(\"id\")", HTML_PAGE)
        self.assertIn("popstate", HTML_PAGE)
        self.assertIn("async function navigate(nextView)", HTML_PAGE)
        self.assertIn("closeDetail(false);syncNavigation(true)", HTML_PAGE)
        self.assertIn("searchParams.set(\"view\",view)", HTML_PAGE)
        self.assertIn("data-view=\"cohort\"", HTML_PAGE)

    def test_legacy_import_is_idempotent_and_retains_provenance(self) -> None:
        labeled = self.root / "labeled.jsonl"
        source = proxy_record()
        labeled.write_text(json.dumps({
            "request_id": source["request_id"], "timestamp": source["timestamp"],
            "raw_request_json": source["raw_request_json"],
            "original_response": source["response_text"],
            "original_model": source["model"], "label": "Use Monza.",
            "label_provider": "codex", "model_used_for_label": "gpt-5.6-luna",
            "manually_reviewed": True,
        }) + "\n")
        meta = inventory(labeled)

        first = import_file(self.store, labeled, meta)
        second = import_file(self.store, labeled, meta)

        self.assertEqual(first["imported_files"], 1)
        self.assertEqual(second["already_imported_files"], 1)
        self.assertEqual(self.store.stats()["sources"], 1)


if __name__ == "__main__":
    unittest.main()
