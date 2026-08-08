import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from src.training.finetune_lfm25 import (
    LFM25_12B_PROFILE,
    LFM25_26B_BASE_PROFILE,
    load_conversations,
    parse_args,
    render_conversations,
)
from src.training.finetune import (
    compute_fused_eval_loss,
    parse_args as parse_qwen_args,
)
from src.training.prepare_dataset import convert_record
from src.training.prepare_qwen_v2 import (
    deduplicate_strategic,
    proportional_allocation,
    select_length_stratified_holdout,
)
from src.training.show_distribution import is_training_approved


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
    def test_training_approval_accepts_manual_or_auto_review(self) -> None:
        self.assertTrue(is_training_approved({"manually_reviewed": True}))
        self.assertTrue(is_training_approved({
            "auto_review": {"status": "approved"},
        }))
        self.assertFalse(is_training_approved({}))

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

    def test_proportional_holdout_preserves_strategic_mix(self) -> None:
        self.assertEqual(
            proportional_allocation({
                "representative": 400,
                "context-rich": 200,
                "long": 200,
                "qwen-risk": 200,
            }, 100),
            {
                "representative": 40,
                "context-rich": 20,
                "long": 20,
                "qwen-risk": 20,
            },
        )

    def test_deduplication_prefers_human_edited_label(self) -> None:
        auto = labeled_record() | {
            "request_id": "auto",
            "auto_review": {"status": "approved"},
        }
        edited = labeled_record() | {
            "request_id": "edited",
            "manually_reviewed": True,
            "manual_review": {"status": "edited"},
        }

        kept, excluded = deduplicate_strategic([auto, edited])

        self.assertEqual([record["request_id"] for record in kept], ["edited"])
        self.assertEqual(excluded[0]["request_id"], "auto")

    def test_length_stratified_holdout_is_exact_and_reproducible(self) -> None:
        records = []
        strata = {}
        for index in range(20):
            record = labeled_record() | {
                "request_id": f"sample-{index:02d}",
                "label": "Clean.",
            }
            request = json.loads(record["raw_request_json"])
            request["messages"][1]["content"] = (
                f"<TRANSCRIPT>{'word ' * (index + 1)}</TRANSCRIPT>"
            )
            record["raw_request_json"] = json.dumps(request)
            records.append(record)
            strata[record["request_id"]] = "short" if index < 10 else "long"

        first = select_length_stratified_holdout(records, strata, 6, 42)
        second = select_length_stratified_holdout(records, strata, 6, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(sum(strata[request_id] == "short" for request_id in first), 3)


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

    def test_12b_profile_preserves_existing_defaults(self) -> None:
        args = parse_args(LFM25_12B_PROFILE, [])

        self.assertEqual(args.base_model, "LiquidAI/LFM2.5-1.2B-Instruct")
        self.assertEqual((args.r, args.lora_alpha), (16, 16))
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.lr, 2e-4)

    def test_26b_base_profile_uses_isolated_two_epoch_recipe(self) -> None:
        args = parse_args(LFM25_26B_BASE_PROFILE, [])

        self.assertEqual(args.base_model, "LiquidAI/LFM2.5-2.6B-Base")
        self.assertEqual((args.r, args.lora_alpha), (32, 32))
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.lr, 1e-4)
        self.assertIn("lfm25-2.6b-base", str(args.lora_dir))
        self.assertIn("lfm25-2.6b-base", str(args.output_dir))
        self.assertEqual(args.gguf_base.name, "LFM2.5-2.6B-VoiceInk")


class QwenTrainerProfileTests(unittest.TestCase):
    def test_v2_paths_and_recipe_can_be_isolated_from_production(self) -> None:
        args = parse_qwen_args([
            "--train", "datasets/qwen35-2b-voiceink-v2/train.jsonl",
            "--eval", "datasets/qwen35-2b-voiceink-v2/eval-regression-340.jsonl",
            "--epochs", "2",
            "--lora-dir", "training/qwen35-2b-voiceink-v2/lora",
            "--output-dir", "training/qwen35-2b-voiceink-v2/outputs",
            "--gguf-base", "models/Qwen3.5-2B-VoiceInk-v2",
            "--check-only",
        ])

        self.assertEqual(args.epochs, 2)
        self.assertEqual((args.r, args.lora_alpha), (32, 64))
        self.assertEqual(args.lr, 2e-4)
        self.assertIn("qwen35-2b-voiceink-v2", str(args.lora_dir))
        self.assertEqual(args.gguf_base.name, "Qwen3.5-2B-VoiceInk-v2")
        self.assertTrue(args.check_only)

    def test_eval_loss_does_not_request_unsloth_logits(self) -> None:
        calls = []

        class Loss:
            def mean(self):
                calls.append("mean")
                return self

            def detach(self):
                calls.append("detach")
                return "loss"

        class Trainer:
            def _prepare_inputs(self, inputs):
                calls.append(("prepare", inputs))
                return {"prepared": True}

            def compute_loss_context_manager(self):
                return nullcontext()

            def compute_loss(self, model, inputs, return_outputs):
                calls.append(("compute", inputs, return_outputs,
                              os.environ.get("UNSLOTH_RETURN_LOGITS")))
                return Loss()

        class Torch:
            @staticmethod
            def no_grad():
                return nullcontext()

        previous = os.environ.get("UNSLOTH_RETURN_LOGITS")
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        try:
            result = compute_fused_eval_loss(Trainer(), object(), {"raw": True}, Torch)
        finally:
            if previous is None:
                os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
            else:
                os.environ["UNSLOTH_RETURN_LOGITS"] = previous

        self.assertEqual(result, ("loss", None, None))
        self.assertIn(("compute", {"prepared": True}, False, "0"), calls)


if __name__ == "__main__":
    unittest.main()
