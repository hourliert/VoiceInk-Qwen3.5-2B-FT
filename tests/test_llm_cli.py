import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.llm_cli import (
    LABEL_SCHEMA,
    call_llm,
    parse_json_response,
    provider_prompt_path,
    resolve_model,
)


class LlmCliTests(unittest.TestCase):
    def test_provider_defaults_and_prompt_paths(self) -> None:
        prompt = Path("src/labeling/judge_prompt.txt")

        self.assertEqual(resolve_model("claude", None), "claude-sonnet-4-6")
        self.assertEqual(resolve_model("codex", None), "gpt-5.6-luna")
        self.assertEqual(provider_prompt_path(prompt, "claude"), prompt)
        self.assertEqual(
            provider_prompt_path(prompt, "codex"),
            Path("src/labeling/judge_prompt.codex.txt"),
        )

    @patch("src.common.llm_cli.subprocess.run")
    def test_codex_is_ephemeral_isolated_schema_bound_and_uses_stdin(self, run) -> None:
        captured = {}

        def complete(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"label":"Clean transcript."}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = complete
        response = call_llm(
            "private transcript",
            provider="codex",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            output_schema=LABEL_SCHEMA,
        )

        command = captured["command"]
        self.assertEqual(parse_json_response(response), {"label": "Clean transcript."})
        self.assertNotIn("private transcript", command)
        self.assertIn("private transcript", captured["input"])
        self.assertIn("--ephemeral", command)
        self.assertIn("--output-schema", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-luna")
        self.assertIn('model_reasoning_effort="low"', command)
        self.assertEqual(command[-1], "-")
        self.assertNotEqual(Path(captured["cwd"]), Path.cwd())

    @patch("src.common.llm_cli.subprocess.run")
    def test_claude_support_remains_available(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["claude"], 0, "Legacy response\n", ""
        )

        response = call_llm(
            "prompt", provider="claude", model="claude-sonnet-4-6"
        )

        self.assertEqual(response, "Legacy response")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "claude")
        self.assertIn("--allowed-tools", command)


if __name__ == "__main__":
    unittest.main()
