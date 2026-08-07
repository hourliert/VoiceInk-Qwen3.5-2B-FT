import json
import tempfile
import unittest
from pathlib import Path

from src.training.finetune_lfm25 import load_conversations, render_conversations
from src.training.prepare_dataset import convert_record


def labeled_record() -> dict:
    request = {
        "model": "VoiceInk",
        "messages": [
            {
                "role": "system",
                "content": (
                    "<SYSTEM_INSTRUCTIONS>old prompt</SYSTEM_INSTRUCTIONS>\n"
                    "<CURRENT_WINDOW_CONTEXT>Terminal</CURRENT_WINDOW_CONTEXT>\n"
                    "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                ),
            },
            {
                "role": "user",
                "content": "<TRANSCRIPT>um test voice ink please</TRANSCRIPT>",
            },
        ],
    }
    return {
        "raw_request_json": json.dumps(request),
        "label": "Test VoiceInk, please.",
    }


class PrepareDatasetTests(unittest.TestCase):
    def test_default_format_preserves_qwen_text_blocks(self) -> None:
        converted = convert_record(labeled_record(), "new prompt")

        self.assertEqual(
            converted["messages"][0]["content"],
            [{"type": "text", "text": "new prompt"}],
        )

    def test_string_format_builds_lfm_conversation(self) -> None:
        converted = convert_record(labeled_record(), "new prompt", "string")

        self.assertEqual(
            [message["role"] for message in converted["messages"]],
            ["system", "user", "assistant"],
        )
        self.assertTrue(
            all(isinstance(message["content"], str)
                for message in converted["messages"])
        )
        self.assertIn("<CURRENT_WINDOW_CONTEXT>", converted["messages"][1]["content"])
        self.assertIn("<TRANSCRIPT>", converted["messages"][1]["content"])
        self.assertEqual(converted["messages"][2]["content"], "Test VoiceInk, please.")


class LfmTrainerInputTests(unittest.TestCase):
    def write_jsonl(self, records: list[dict]) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            for record in records:
                temporary.write(json.dumps(record) + "\n")
        return Path(temporary.name)

    def test_loader_accepts_string_conversations(self) -> None:
        record = convert_record(labeled_record(), "new prompt", "string")
        path = self.write_jsonl([record])

        self.assertEqual(load_conversations(path), [record])

    def test_loader_rejects_qwen_text_blocks(self) -> None:
        path = self.write_jsonl([convert_record(labeled_record(), "new prompt")])

        with self.assertRaisesRegex(ValueError, "requires non-empty string content"):
            load_conversations(path)

    def test_renderer_uses_model_chat_template_without_generation_prompt(self) -> None:
        record = convert_record(labeled_record(), "new prompt", "string")

        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.call = (messages, tokenize, add_generation_prompt)
                return "rendered chat"

        tokenizer = Tokenizer()
        rendered = render_conversations([record], tokenizer)

        self.assertEqual(rendered, [{"text": "rendered chat"}])
        self.assertEqual(tokenizer.call, (record["messages"], False, False))


if __name__ == "__main__":
    unittest.main()
