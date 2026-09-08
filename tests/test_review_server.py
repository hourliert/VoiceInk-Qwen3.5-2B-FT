import json
import tempfile
import unittest
from pathlib import Path

from src.labeling.review_server import (
    HTML_PAGE,
    ReviewDatasets,
    get_review_records,
    highlighted_diff,
    load_review_manifest,
    suspicion_reasons,
    word_similarity,
)
from src.labeling.validate import LabeledDataset


def make_record(request_id: str, transcript: str, label: str,
                status: str = "pass") -> dict:
    request = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "<SYSTEM_INSTRUCTIONS>Clean.</SYSTEM_INSTRUCTIONS>"
                    "<CURRENT_WINDOW_CONTEXT>Context</CURRENT_WINDOW_CONTEXT>"
                ),
            },
            {
                "role": "user",
                "content": f"<TRANSCRIPT>{transcript}</TRANSCRIPT>",
            },
        ]
    }
    return {
        "request_id": request_id,
        "raw_request_json": json.dumps(request),
        "label": label,
        "validation": {"status": status},
    }


class ReviewServerTests(unittest.TestCase):
    def test_label_editor_always_expands_to_full_content(self) -> None:
        self.assertIn("overflow-y: hidden", HTML_PAGE)
        self.assertIn("autoSizeEditor.call(editor)", HTML_PAGE)
        self.assertIn("this.scrollHeight", HTML_PAGE)

    def write_records(self, records: list[dict]) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            for record in records:
                temporary.write(json.dumps(record) + "\n")
        return Path(temporary.name)

    def test_manifest_selects_validator_pass(self) -> None:
        path = self.write_records([
            make_record("selected", "Use version 2.", "Use version 2."),
            make_record("other", "Keep this.", "Keep this."),
        ])
        dataset = LabeledDataset(path)

        queued = get_review_records(
            dataset, selected={"selected": "manual audit"}
        )

        self.assertEqual([record["request_id"] for record in queued], ["selected"])
        self.assertEqual(queued[0]["queue_reason"], "manual audit")

    def test_suspicious_mode_includes_numeric_change(self) -> None:
        record = make_record("numeric", "Use version 2.", "Use version 3.")
        components = {
            "transcript": "Use version 2.",
        }

        self.assertIn("numeric tokens changed", suspicion_reasons(record, components))

    def test_similarity_uses_words_without_character_autojunk(self) -> None:
        raw = (
            "Okay, so it's working. I mean it's starting without errors. "
            "Now the real question is whether shadow mode will work today."
        )
        label = (
            "It's working. It's starting without errors. "
            "The real question is whether shadow mode will work today."
        )
        record = make_record("similar", raw, label)

        reasons = suspicion_reasons(record, {"transcript": raw})

        self.assertFalse(any("low raw/label similarity" in r for r in reasons))

    def test_highlighted_diff_escapes_and_marks_changes(self) -> None:
        rendered = highlighted_diff("Use <version> 2.", "Use version 3.")

        self.assertNotIn("<version>", rendered)
        self.assertIn("diff-remove", rendered)
        self.assertIn("diff-add", rendered)
        self.assertGreater(word_similarity("Use version 2.", "Use version 3."), 0)

    def test_review_record_contains_three_way_comparison(self) -> None:
        record = make_record("compare", "Use version 2.", "Use version 3.", "fail")
        record["original_response"] = "Use version two."
        dataset = LabeledDataset(self.write_records([record]))

        queued = get_review_records(dataset)

        self.assertIn("original_diff_html", queued[0])
        self.assertIn("label_diff_html", queued[0])
        self.assertIn("label_original_similarity", queued[0])

    def test_calibration_record_contains_optional_sonnet_comparison(self) -> None:
        record = make_record("audit", "Use version 2.", "Use version 3.", "fail")
        record["original_response"] = "Use version two."
        record["reference_label"] = "Use version 2."
        record["reference_model_used_for_label"] = "claude-sonnet-4-6"
        dataset = LabeledDataset(self.write_records([record]))

        queued = get_review_records(dataset)

        self.assertEqual(queued[0]["reference_label"], "Use version 2.")
        self.assertEqual(queued[0]["reference_model"], "claude-sonnet-4-6")
        self.assertIn("reference_diff_html", queued[0])
        self.assertEqual(queued[0]["reference_similarity"], 100)
        self.assertIn("label_reference_similarity", queued[0])
        self.assertIn("Sonnet reference", HTML_PAGE)
        self.assertIn("four-way", HTML_PAGE)

    def test_current_model_output_replaces_historical_qwen_in_audit(self) -> None:
        record = make_record("current", "Raw.", "Luna.", "fail")
        record["original_model"] = "Qwen3.5-4B"
        record["original_response"] = "Historical output."
        record["current_model"] = "Qwen3.5-2B-VoiceInk"
        record["current_model_response"] = "Current output."
        dataset = LabeledDataset(self.write_records([record]))

        queued = get_review_records(dataset)

        self.assertEqual(queued[0]["original_model"], "Qwen3.5-2B-VoiceInk")
        self.assertEqual(queued[0]["original_response"], "Current output.")

    def test_rejected_record_does_not_reappear(self) -> None:
        record = make_record("rejected", "Keep this.", "Keep this.", "fail")
        record["manual_review"] = {"status": "rejected"}
        dataset = LabeledDataset(self.write_records([record]))

        self.assertEqual(get_review_records(dataset), [])

    def test_load_review_manifest_retains_reason(self) -> None:
        path = self.write_records([
            {"request_id": "abc", "reason": "ambiguous correction"},
        ])

        self.assertEqual(
            load_review_manifest(path), {"abc": "ambiguous correction"}
        )

    def test_multiple_inputs_form_one_queue_and_preserve_owners(self) -> None:
        first = self.write_records([
            make_record("pilot", "Pilot raw.", "Pilot label.", "fail"),
        ])
        second = self.write_records([
            make_record("batch", "Batch raw.", "Batch label.", "fail"),
        ])

        datasets = ReviewDatasets([first, second])
        queued = get_review_records(datasets)

        self.assertEqual(
            [record["request_id"] for record in queued], ["pilot", "batch"]
        )
        self.assertIsNot(
            datasets.dataset_for("pilot"), datasets.dataset_for("batch")
        )

    def test_multiple_inputs_reject_duplicate_ids(self) -> None:
        first = self.write_records([
            make_record("duplicate", "First.", "First.", "fail"),
        ])
        second = self.write_records([
            make_record("duplicate", "Second.", "Second.", "fail"),
        ])

        with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
            ReviewDatasets([first, second])


if __name__ == "__main__":
    unittest.main()
