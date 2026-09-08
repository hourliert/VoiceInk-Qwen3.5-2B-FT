import json
import tempfile
import unittest
from pathlib import Path

from src.labeling.adjudicate import build_approval_manifest
from src.labeling.calibrate_validator import build_fixture, score_fixture
from src.labeling.label import load_id_prefixes
from src.labeling.select_strategic import qwen_risk, select_manifest
from src.labeling.triage import classify_record, triage_records
from src.labeling.validate import LabeledDataset as ValidationDataset


class LabelingStrategyTests(unittest.TestCase):
    def write_text(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(text)
        return Path(temporary.name)

    def calibration_source(self, request_id: str = "source") -> dict:
        request = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<SYSTEM_INSTRUCTIONS>Clean.</SYSTEM_INSTRUCTIONS>"
                        "<CURRENT_WINDOW_CONTEXT>VoiceInk</CURRENT_WINDOW_CONTEXT>"
                        "<CLIPBOARD_CONTEXT></CLIPBOARD_CONTEXT>"
                        "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "<TRANSCRIPT>Okay, so we should deploy this today. "
                        "Then verify the result.</TRANSCRIPT>"
                    ),
                },
            ]
        }
        return {
            "request_id": request_id,
            "raw_request_json": json.dumps(request),
            "reference_label": "We should deploy this today. Then verify the result.",
            "reference_manually_reviewed": True,
        }

    def test_load_id_prefixes_accepts_text_and_jsonl(self) -> None:
        path = self.write_text(
            "# pilot\nabc\n{\"request_id\": \"def\"}\nabc\n"
        )

        self.assertEqual(load_id_prefixes(path), ["abc", "def"])

    def test_build_validator_fixture_has_known_good_and_defects(self) -> None:
        fixture = build_fixture([self.calibration_source()], 2, 6)

        self.assertEqual(len(fixture), 8)
        self.assertEqual(
            [record["expected_validation"] for record in fixture].count("pass"), 2
        )
        self.assertEqual(
            {record["expected_failure_type"] for record in fixture[2:]},
            {
                "HALLUCINATION",
                "MEANING_ALTERED",
                "OVER_DELETION",
                "REPETITION",
                "BROKEN_OUTPUT",
                "UNDER_CLEANING",
            },
        )
        self.assertEqual(len({record["request_id"] for record in fixture}), 8)

    def test_validator_score_rejects_critical_false_pass(self) -> None:
        fixture = build_fixture([self.calibration_source()], 2, 6)
        for record in fixture:
            record["validation"] = {
                "status": record["expected_validation"],
                "type": record["expected_failure_type"],
            }
        self.assertTrue(score_fixture(fixture, 0.9, 0.15))

        hallucination = next(
            record for record in fixture
            if record["expected_failure_type"] == "HALLUCINATION"
        )
        hallucination["validation"] = {"status": "pass"}
        self.assertFalse(score_fixture(fixture, 0.8, 0.15))

    def test_qwen_risk_prioritizes_changed_numbers(self) -> None:
        safe_score, _ = qwen_risk("Use version 2 today.", "Use version 2 today.")
        risky_score, reasons = qwen_risk(
            "Use version 2 today.", "Use version 3 tomorrow."
        )

        self.assertGreater(risky_score, safe_score)
        self.assertTrue(any("numeric" in reason for reason in reasons))

    def test_strategic_manifest_has_non_overlapping_strata(self) -> None:
        pool = [
            {
                "request_id": str(index),
                "transcript_length": 2000 - index,
                "context_score": float(index % 3),
                "qwen_risk_score": float(index),
            }
            for index in range(12)
        ]

        manifest = select_manifest(
            pool,
            representative=2,
            context_rich=2,
            long_count=2,
            qwen_risk_count=2,
            long_threshold=1000,
            seed=42,
        )

        self.assertEqual(len(manifest), 8)
        self.assertEqual(len({record["request_id"] for record in manifest}), 8)
        self.assertEqual(
            {stratum: sum(record["stratum"] == stratum for record in manifest)
             for stratum in {record["stratum"] for record in manifest}},
            {"qwen-risk": 2, "long": 2, "context-rich": 2, "representative": 2},
        )

    def test_codex_prompts_require_filler_cleanup_review(self) -> None:
        label_prompt = Path("src/labeling/judge_prompt.codex.txt").read_text()
        validate_prompt = Path("src/labeling/validate_prompt.codex.txt").read_text()

        self.assertIn("required cleanup", label_prompt)
        self.assertIn("UNDER_CLEANING", validate_prompt)

    def test_codex_prompts_handle_long_garble_and_validator_consistency(self) -> None:
        label_prompt = Path("src/labeling/judge_prompt.codex.txt").read_text()
        validate_prompt = Path("src/labeling/validate_prompt.codex.txt").read_text()

        self.assertIn("unintelligible or grammatically broken", label_prompt)
        self.assertIn("scan every sentence", label_prompt)
        self.assertIn("entire transcript", validate_prompt)
        self.assertIn("self-contradictory failure", validate_prompt)

    def test_selective_review_marks_only_exact_ids(self) -> None:
        path = self.write_text(
            '{"request_id":"abc","label":"A"}\n'
            '{"request_id":"abcd","label":"B"}\n'
        )
        dataset = ValidationDataset(path)

        self.assertEqual(dataset.mark_reviewed_exact({"abc"}), 1)
        records = {record["request_id"]: record for record in dataset.records()}
        self.assertTrue(records["abc"]["manually_reviewed"])
        self.assertNotIn("manually_reviewed", records["abcd"])

    def test_selective_review_refuses_partial_approval(self) -> None:
        path = self.write_text('{"request_id":"abc","label":"A"}\n')
        dataset = ValidationDataset(path)

        with self.assertRaisesRegex(ValueError, "missing"):
            dataset.mark_reviewed_exact({"abc", "missing"})
        self.assertNotIn("manually_reviewed", dataset.records()[0])

    def test_approval_manifest_combines_passes_and_adjudication(self) -> None:
        records = [
            {"request_id": "pass", "validation": {"status": "pass"}},
            {"request_id": "false-fail", "validation": {"status": "fail"}},
            {"request_id": "exclude", "validation": {"status": "pass"}},
        ]

        approved = build_approval_manifest(
            records, {"false-fail"}, {"exclude"}
        )

        self.assertEqual(
            {record["request_id"] for record in approved},
            {"pass", "false-fail"},
        )

    def test_approval_manifest_fails_closed_on_unresolved_record(self) -> None:
        records = [
            {"request_id": "pass", "validation": {"status": "pass"}},
            {"request_id": "unresolved", "validation": {"status": "fail"}},
        ]

        with self.assertRaisesRegex(ValueError, "unresolved"):
            build_approval_manifest(records, set(), set())

    def test_triage_auto_approves_clean_validator_pass(self) -> None:
        record = self.calibration_source("clean")
        record["label"] = record["reference_label"]
        record["validation"] = {"status": "pass"}

        classification, reasons = classify_record(record)

        self.assertEqual(classification, "auto-approved")
        self.assertEqual(reasons, [])

    def test_triage_queues_suspicious_pass_and_failure(self) -> None:
        numeric = self.calibration_source("numeric")
        numeric["label"] = "We should deploy version 3 today."
        numeric["raw_request_json"] = numeric["raw_request_json"].replace(
            "deploy this", "deploy version 2"
        )
        numeric["validation"] = {"status": "pass"}
        failed = self.calibration_source("failed")
        failed["label"] = failed["reference_label"]
        failed["validation"] = {"status": "fail"}

        self.assertEqual(classify_record(numeric)[0], "human-review")
        self.assertEqual(classify_record(failed)[0], "human-review")

    def test_triage_migrates_unproven_legacy_review_flag(self) -> None:
        record = self.calibration_source("legacy")
        record["label"] = record["reference_label"]
        record["validation"] = {"status": "pass"}
        record["manually_reviewed"] = True

        counts = triage_records([record], True, "now")

        self.assertNotIn("manually_reviewed", record)
        self.assertEqual(record["auto_review"]["status"], "approved")
        self.assertEqual(counts["auto-approved"], 1)

    def test_labeller_prompt_does_not_contain_calibration_answers(self) -> None:
        prompt = Path("src/labeling/judge_prompt.codex.txt").read_text()

        for leaked_answer in (
            "revise the pram",
            "not a liar at the moment",
            "TEETH Online",
            "worse for you to dispatch",
        ):
            with self.subTest(leaked_answer=leaked_answer):
                self.assertNotIn(leaked_answer, prompt)


if __name__ == "__main__":
    unittest.main()
