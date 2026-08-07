import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from src.common.llm_cli import SCORE_DIMENSIONS
from src.eval.evaluate import (
    load_eval_data,
    load_saved_outputs,
    message_text,
    parse_judge_response,
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

    def test_saved_outputs_accept_completed_eval_files(self) -> None:
        path = self.write_jsonl([{
            "raw_transcript": "Hello",
            "baseline_output": "Baseline",
            "candidate_output": "Candidate",
            "baseline_duration_ms": 10,
            "candidate_duration_ms": 5,
            "baseline_scores": {},
            "candidate_scores": {},
        }])

        outputs = load_saved_outputs(path)["Hello"]

        self.assertEqual(outputs["baseline"]["text"], "Baseline")
        self.assertEqual(outputs["candidate"]["duration_ms"], 5)

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
