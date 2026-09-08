import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.run import resolved_command


class ReleaseEvaluationTests(unittest.TestCase):
    def args(self, include_regression=False):
        return argparse.Namespace(
            baseline="production", candidate="candidate",
            release_manifest=Path("release/manifest.json"),
            include_regression=include_regression, suite=None,
            judge_model="gpt-5.6-luna", judge_reasoning_effort="low",
            parallel=10, llama_host="127.0.0.1", llama_port=8002,
            output_dir=Path("results/canonical"),
        )

    @patch("src.eval.run.split_path", return_value=Path("release/acceptance.jsonl"))
    @patch("src.eval.run.load_manifest", return_value={"task": "sft"})
    def test_default_resolves_acceptance_without_historical_regression(
        self, _manifest, _split
    ):
        command = resolved_command(self.args())
        self.assertIn("release/acceptance.jsonl", command)
        self.assertIn("strict-v3", command)
        self.assertIn("codex", command)
        self.assertNotIn("strict.py", " ".join(command))

    @patch("src.eval.run.load_suite", return_value={
        "name": "engineering-100", "rubric": "strict-v3",
        "data": {"resolved_path": "/sealed/engineering-100.jsonl"},
    })
    @patch("src.eval.run.load_manifest", return_value={"task": "sft"})
    def test_named_suite_is_verified_and_resolved(self, _manifest, _suite):
        args = self.args()
        args.suite = "engineering-100"
        command = resolved_command(args)
        self.assertIn("/sealed/engineering-100.jsonl", command)
        self.assertIn("engineering-100", " ".join(command))

    @patch("src.eval.run.load_manifest", return_value={"task": "sft"})
    def test_regression_is_explicit_opt_in(self, _manifest):
        command = resolved_command(self.args(include_regression=True))
        self.assertIn("strict.py", " ".join(command))
        self.assertIn("run", command)


if __name__ == "__main__":
    unittest.main()
