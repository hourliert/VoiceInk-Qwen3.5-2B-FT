import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from src.common.llm_cli import SCORE_DIMENSIONS
from src.eval.evaluate import (
    CANONICAL_EVAL,
    CANONICAL_EVAL_COUNT,
    DEFAULT_EVAL,
    aggregate,
    _model_metrics,
    load_eval_data,
    load_saved_outputs,
    message_text,
    messages_for_layout,
    parse_judge_response,
    select_sample_indices,
    validate_eval_corpus,
)
from src.labeling.label import load_logs, load_reference_labels


class ProviderWorkflowTests(unittest.TestCase):
    def write_jsonl(self, records: list[dict]) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            for record in records:
                temporary.write(json.dumps(record) + "\n")
        return Path(temporary.name)

    def test_default_eval_is_locked_canonical_440(self) -> None:
        self.assertEqual(DEFAULT_EVAL, CANONICAL_EVAL)
        samples = load_eval_data(DEFAULT_EVAL)
        self.assertEqual(len(samples), CANONICAL_EVAL_COUNT)
        validate_eval_corpus(DEFAULT_EVAL, len(samples))

    def test_noncanonical_full_eval_requires_explicit_override(self) -> None:
        path = self.write_jsonl([])
        with self.assertRaisesRegex(ValueError, "require canonical"):
            validate_eval_corpus(path, 0)
        validate_eval_corpus(path, 0, allow_noncanonical=True)

    def test_message_text_supports_qwen_and_lfm_content(self) -> None:
        self.assertEqual(message_text("plain"), "plain")
        self.assertEqual(
            message_text([{"type": "text", "text": "typed"}]), "typed"
        )

    def test_eval_loader_accepts_string_messages(self) -> None:
        path = self.write_jsonl([{
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "<TRANSCRIPT>Hello</TRANSCRIPT>"},
                {"role": "assistant", "content": "Hello."},
            ]
        }])

        sample = load_eval_data(path)[0]

        self.assertEqual(sample["raw_transcript"], "Hello")
        self.assertEqual(sample["gold_label"], "Hello.")
        self.assertEqual(sample["window_context"], "")
        self.assertEqual(sample["clipboard_context"], "")

    def test_eval_loader_extracts_model_context_for_judge(self) -> None:
        path = self.write_jsonl([{
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "<CURRENT_WINDOW_CONTEXT>electron-mocks.ts</CURRENT_WINDOW_CONTEXT>"
                        "<CLIPBOARD_CONTEXT>reportZonePlanResults</CLIPBOARD_CONTEXT>"
                        "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                        "<TRANSCRIPT>Electron Mox</TRANSCRIPT>"
                    ),
                },
                {"role": "assistant", "content": "electron-mocks.ts"},
            ]
        }])

        sample = load_eval_data(path)[0]

        self.assertEqual(sample["window_context"], "electron-mocks.ts")
        self.assertEqual(sample["clipboard_context"], "reportZonePlanResults")
        self.assertEqual(sample["custom_vocabulary"], "VoiceInk")

    def test_eval_loader_accepts_voiceink_production_layout(self) -> None:
        path = self.write_jsonl([{
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<SYSTEM_INSTRUCTIONS>system</SYSTEM_INSTRUCTIONS>"
                        "<CURRENT_WINDOW_CONTEXT>app.ts</CURRENT_WINDOW_CONTEXT>"
                        "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                    ),
                },
                {"role": "user", "content": "<TRANSCRIPT>Voice Inc</TRANSCRIPT>"},
                {"role": "assistant", "content": "VoiceInk"},
            ]
        }])

        sample = load_eval_data(path)[0]

        self.assertEqual(sample["system_text"], "system")
        self.assertEqual(sample["window_context"], "app.ts")
        self.assertEqual(sample["custom_vocabulary"], "VoiceInk")
        production = messages_for_layout(sample, "voiceink")
        prepared = messages_for_layout(sample, "prepared")
        self.assertIn("CURRENT_WINDOW_CONTEXT", production[0]["content"])
        self.assertNotIn("CURRENT_WINDOW_CONTEXT", production[1]["content"])
        self.assertNotIn("SYSTEM_INSTRUCTIONS", prepared[0]["content"])
        self.assertIn("CURRENT_WINDOW_CONTEXT", prepared[1]["content"])
        self.assertEqual(
            production,
            [
                {
                    "role": "system",
                    "content": (
                        "<SYSTEM_INSTRUCTIONS>system</SYSTEM_INSTRUCTIONS>"
                        "<CURRENT_WINDOW_CONTEXT>app.ts</CURRENT_WINDOW_CONTEXT>"
                        "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                    ),
                },
                {"role": "user", "content": "<TRANSCRIPT>Voice Inc</TRANSCRIPT>"},
            ],
        )

    def test_eval_loader_uses_last_transcript_tag_pair(self) -> None:
        path = self.write_jsonl([{
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "<CURRENT_WINDOW_CONTEXT>"
                        "example <TRANSCRIPT>\\s*(.*?)\\s*</TRANSCRIPT>"
                        "</CURRENT_WINDOW_CONTEXT>"
                        "<TRANSCRIPT>Actual dictated text.</TRANSCRIPT>"
                    ),
                },
                {"role": "assistant", "content": "Actual dictated text."},
            ]
        }])

        sample = load_eval_data(path)[0]

        self.assertEqual(sample["raw_transcript"], "Actual dictated text.")

    def test_prepared_prompt_tag_examples_are_not_mistaken_for_wrapper(self) -> None:
        path = self.write_jsonl([{
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Instructions may mention "
                        "<SYSTEM_INSTRUCTIONS>examples</SYSTEM_INSTRUCTIONS>."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "<CURRENT_WINDOW_CONTEXT>app.ts</CURRENT_WINDOW_CONTEXT>"
                        "<TRANSCRIPT>hello</TRANSCRIPT>"
                    ),
                },
                {"role": "assistant", "content": "Hello."},
            ]
        }])

        sample = load_eval_data(path)[0]

        self.assertTrue(sample["system_text"].startswith("Instructions may mention"))
        self.assertEqual(sample["window_context"], "app.ts")

    def test_select_sample_indices_preserves_requested_order(self) -> None:
        samples = [{"sample_index": index} for index in range(4)]

        selected = select_sample_indices(samples, "3, 1")

        self.assertEqual([sample["sample_index"] for sample in selected], [3, 1])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_sample_indices(samples, "1,1")
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_sample_indices(samples, "4")

    def test_judge_response_requires_every_integer_score(self) -> None:
        scores = {
            "meaning_preservation": 5,
            "filler_removal": 4,
            "grammar_fluency": 5,
            "technical_accuracy": 5,
            "conciseness": 4,
            "instruction_following": 5,
        }
        valid = json.dumps({"output_a": scores, "output_b": scores})
        missing = json.dumps({
            "output_a": scores,
            "output_b": {key: value for key, value in scores.items()
                         if key != "conciseness"},
        })

        self.assertIsNotNone(parse_judge_response(valid))
        self.assertIsNone(parse_judge_response(missing))

    def test_codex_judge_response_requires_context_analysis(self) -> None:
        scores = {dimension: 5 for dimension in SCORE_DIMENSIONS}
        without_analysis = json.dumps({"output_a": scores, "output_b": scores})
        with_analysis = json.dumps({
            "output_a": scores,
            "output_b": scores,
            "context_analysis": {
                "output_a": "Supported code change from the code-related window.",
                "output_b": "Retains the unsupported ASR wording cold change.",
            },
            "score_analysis": {
                "output_a": "No material scoring issue.",
                "output_b": "Technical accuracy: retains cold change.",
            },
        })

        self.assertIsNone(parse_judge_response(
            without_analysis, require_context_analysis=True
        ))
        self.assertIsNotNone(parse_judge_response(
            with_analysis, require_context_analysis=True
        ))

    def test_strict_judge_response_requires_pairwise_verdict(self) -> None:
        scores = {dimension: 5 for dimension in SCORE_DIMENSIONS}
        response = {
            "output_a": scores,
            "output_b": scores,
            "context_analysis": {
                "output_a": "No material context-sensitive issue.",
                "output_b": "No material context-sensitive issue.",
            },
            "score_analysis": {
                "output_a": "No material scoring issue.",
                "output_b": "No material scoring issue.",
            },
        }

        self.assertIsNone(parse_judge_response(
            json.dumps(response),
            require_context_analysis=True,
            require_pairwise=True,
        ))
        response["pairwise"] = {
            "preference": "tie",
            "confidence": "high",
            "material_difference": False,
            "reason": "The outputs are equivalent.",
        }
        self.assertIsNotNone(parse_judge_response(
            json.dumps(response),
            require_context_analysis=True,
            require_pairwise=True,
        ))

    def test_aggregate_reports_strict_pairwise_and_material_wins(self) -> None:
        scores = {dimension: 5 for dimension in SCORE_DIMENSIONS}
        judgments = [
            {
                "baseline_scores": scores,
                "candidate_scores": scores,
                "pairwise_preference": "candidate",
                "pairwise_confidence": "high",
                "pairwise_material_difference": True,
            },
            {
                "baseline_scores": scores,
                "candidate_scores": scores,
                "pairwise_preference": "tie",
                "pairwise_confidence": "medium",
                "pairwise_material_difference": False,
            },
        ]
        outputs = [{"duration_ms": 10}, {"duration_ms": 10}]

        summary = aggregate(
            judgments, outputs, outputs, "baseline", "candidate"
        )

        self.assertEqual(
            summary["judge_pairwise"]["preferences"],
            {"baseline": 0, "candidate": 1, "tie": 1},
        )
        self.assertEqual(
            summary["judge_pairwise"]["material_wins"],
            {"baseline": 0, "candidate": 1},
        )

    def test_saved_outputs_accept_completed_eval_files(self) -> None:
        path = self.write_jsonl([{
            "sample_index": 0,
            "raw_transcript": "Hello",
            "baseline_output": "Baseline",
            "candidate_output": "Candidate",
            "baseline_duration_ms": 10,
            "candidate_duration_ms": 5,
            "baseline_scores": {},
            "candidate_scores": {},
        }])

        saved_outputs = load_saved_outputs(path)
        outputs = saved_outputs["Hello"]

        self.assertEqual(outputs["baseline"]["text"], "Baseline")
        self.assertEqual(outputs["candidate"]["duration_ms"], 5)
        self.assertIs(outputs, saved_outputs[0])

    def test_mlflow_model_metrics_use_shared_names_for_both_models(self) -> None:
        summary = {
            "n_samples": 440,
            "baseline_avg_score": 95.2,
            "candidate_avg_score": 95.3,
            "baseline_latency": {"avg": 770, "p50": 268},
            "candidate_latency": {"avg": 760, "p50": 268},
            "per_dimension": {
                "meaning_preservation": {
                    "baseline_avg": 4.60, "candidate_avg": 4.59,
                },
            },
            "wins": {"baseline": 90, "candidate": 95},
            "judge_pairwise": {"material_wins": {"baseline": 62, "candidate": 71}},
        }

        baseline = _model_metrics(summary, "baseline")
        candidate = _model_metrics(summary, "candidate")

        self.assertEqual(set(baseline), set(candidate))
        self.assertEqual(baseline["quality"]["meaning_preservation"], 4.60)
        self.assertEqual(candidate["quality"]["meaning_preservation"], 4.59)
        self.assertEqual(baseline["latency"]["avg"], 770)
        self.assertEqual(candidate["latency"]["avg"], 760)

    def test_reference_loader_keeps_human_label_metadata(self) -> None:
        path = self.write_jsonl([{
            "request_id": "abc",
            "label": "Reviewed label",
            "manually_reviewed": True,
        }])

        self.assertEqual(load_reference_labels(path)["abc"]["label"], "Reviewed label")

    def test_log_loader_skips_malformed_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(
                '{"request_id":"good","request_json_valid":true,'
                '"response_json_valid":true}\n'
                '{"broken":\n',
                encoding="utf-8",
            )
            with redirect_stderr(StringIO()) as stderr:
                records = load_logs(path)

            self.assertEqual([record["request_id"] for record in records], ["good"])
            self.assertIn("skipped 1 malformed JSONL record", stderr.getvalue())

    def test_all_provider_prompts_render(self) -> None:
        substitutions = {
            "transcript": "raw",
            "custom_vocabulary": "VoiceInk",
            "clipboard_context": "clipboard",
            "window_context": "window",
            "label": "clean",
            "raw_transcript": "raw",
            "gold_label": "gold",
            "output_a": "A",
            "output_b": "B",
            "track": "Monza",
            "scenario": "QA",
            "target_words": 500,
            "seed": 1,
        }
        paths = [
            Path("src/labeling/judge_prompt.codex.txt"),
            Path("src/labeling/validate_prompt.codex.txt"),
            Path("src/eval/judge_prompt.codex.txt"),
            Path("src/synthetic/generator_prompt.codex.txt"),
        ]
        for path in paths:
            with self.subTest(path=path):
                rendered = path.read_text(encoding="utf-8").format(**substitutions)
                self.assertTrue(rendered.strip())

    def test_codex_eval_prompt_defines_every_score_dimension(self) -> None:
        prompt = Path("src/eval/judge_prompt.codex.txt").read_text(encoding="utf-8")
        for dimension in SCORE_DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.assertIn(dimension, prompt)


if __name__ == "__main__":
    unittest.main()
