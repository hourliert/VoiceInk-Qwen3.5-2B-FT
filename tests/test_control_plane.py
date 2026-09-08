import json
import tempfile
import unittest
from pathlib import Path

from src.control_plane.state import ControlPlaneState
from src.data.schema import migrate
from src.labeling.annotation_store import AnnotationStore

from tests.test_live_review import evaluation_result, proxy_record


class ControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(self.root / "annotations.sqlite3")
        self.addCleanup(self.store.close)
        migrate(self.store.connection)
        self.store.import_proxy_record(proxy_record())
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="historical"
        )
        self.store.decide(
            "request-1", "production_correct", annotation_id=annotation
        )
        self.store.connection.execute(
            "UPDATE decisions SET decided_at=? WHERE request_id=?",
            ("2025-01-01T00:00:00+00:00", "request-1"),
        )
        cursor = self.store.connection.execute(
            "INSERT INTO canonical_cohorts(name,state,seed,specification_json,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?)",
            ("cohort-v1", "reviewing", 1, "{}", "2026-01-01T00:00:00+00:00",
             "2026-01-01T00:00:00+00:00"),
        )
        self.store.connection.execute(
            "INSERT INTO canonical_cohort_members(cohort_id,request_id,annotation_id,"
            "split,stratum,audit_selected) VALUES(?,?,?,?,?,?)",
            (cursor.lastrowid, "request-1", annotation, "acceptance",
             "technical-risk", 0),
        )
        models = self.root / "models.ini"
        model = self.root / "model.gguf"
        model.write_bytes(b"gguf")
        models.write_text(
            "version = 1\n\n[*]\nc = 16384\n\n"
            f"[VoiceInk-Production]\nmodel = {model}\n"
        )
        profiles = self.root / "profiles.toml"
        profiles.write_text(
            "[profiles.test]\nmode='sft'\ntrainer='qwen'\n"
        )
        self.state = ControlPlaneState(
            self.store, mlflow_tracking_uri="http://127.0.0.1:9",
            models_config=models, profiles=profiles,
            release_root=self.root / "releases",
        )

    def test_acceptance_is_not_blanket_human_review_before_luna(self) -> None:
        result = self.state.cohort_queue("cohort-v1", review="required")

        self.assertEqual(result["total"], 0)

    def test_deterministic_audit_requires_human_review(self) -> None:
        self.store.connection.execute(
            "UPDATE canonical_cohort_members SET audit_selected=1"
        )
        self.state._cohort_cache.clear()

        result = self.state.cohort_queue("cohort-v1", review="required")

        self.assertEqual(result["total"], 1)
        self.assertTrue(result["items"][0]["requires_human"])

    def test_queue_tracks_fresh_analysis_and_human_review(self) -> None:
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="luna", provider="codex",
            model="gpt-5.6-luna", prompt_sha256="prompt",
            created_at="2026-02-01T00:00:00+00:00",
        )
        self.store.create_analysis(
            "request-1", annotation, "fresh-cache", evaluation_result(),
            provider="codex", model="gpt-5.6-luna", reasoning_effort="low",
            prompt_sha256="eval", blinded_assignment="a=production",
        )
        self.store.decide(
            "request-1", "accept_luna", annotation_id=annotation,
            reviewer="human",
        )
        self.state._cohort_cache.clear()

        item = self.state.cohort_queue("cohort-v1", review="reviewed")["items"][0]

        self.assertTrue(item["fresh_luna"])
        self.assertTrue(item["human_reviewed"])
        self.assertFalse(item["review_pending"])

    def test_queue_does_not_recompute_full_cohort_status(self) -> None:
        self.state._cohort_cache.clear()

        def unexpected_status(*args, **kwargs):
            self.fail("Queue lookup recomputed full cohort status")

        self.state._cohort_status = unexpected_status
        result = self.state.cohort_queue("cohort-v1", review="all")

        self.assertEqual(result["total"], 1)
        self.assertEqual(self.state.active_cohort()["name"], "cohort-v1")

    def test_llama_preset_and_react_routes_are_supported(self) -> None:
        models = self.state.models()
        app_source = (
            Path(__file__).resolve().parents[1] / "ui" / "src" / "App.tsx"
        ).read_text(encoding="utf-8")
        sample_source = (
            Path(__file__).resolve().parents[1]
            / "ui" / "src" / "pages" / "SamplePage.tsx"
        ).read_text(encoding="utf-8")
        styles = (
            Path(__file__).resolve().parents[1] / "ui" / "src" / "styles.css"
        ).read_text(encoding="utf-8")

        self.assertEqual(models[0]["name"], "VoiceInk-Production")
        self.assertTrue(models[0]["exists"])
        for route in (
            "/review/recent", "/review/queue", "/review/history",
            "/data/cohorts", "/data/releases", "/models",
            "/runs/training", "/runs/evaluations", "/system",
        ):
            self.assertIn(route, app_source)
        self.assertNotIn("setInterval(", app_source + sample_source)
        self.assertIn("refetchInterval", sample_source)
        self.assertIn('review: "required"', sample_source)
        self.assertIn("navigateAfterSave", sample_source)
        self.assertIn("padding: 10px 15px", styles)


if __name__ == "__main__":
    unittest.main()

