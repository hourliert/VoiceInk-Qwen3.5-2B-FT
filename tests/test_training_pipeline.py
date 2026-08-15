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
from src.training.finetune_lfm25_12b_v3 import parse_args as parse_lfm_v3_args
from src.training.finetune_lfm25_26b_v3 import parse_args as parse_lfm_26b_v3_args
from src.training.finetune import (
    align_step_interval,
    compute_fused_eval_loss,
    expected_training_steps,
    parse_args as parse_qwen_args,
)
from src.training.finetune_qwen35_08b import parse_args as parse_qwen_08b_args
from src.training.prepare_dataset import (
    content_text,
    convert_conversation_layout,
    convert_record,
)
from src.training.prepare_qwen_v2 import (
    deduplicate_strategic,
    proportional_allocation,
    select_length_stratified_holdout,
)
from src.training.prepare_lfm25_v3 import convert_conversation
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

        system = content_text(converted["messages"][0]["content"])
        user = content_text(converted["messages"][1]["content"])
        self.assertIn("<SYSTEM_INSTRUCTIONS>\nnew prompt", system)
        self.assertIn("<CURRENT_WINDOW_CONTEXT>Terminal", system)
        self.assertIn("<CUSTOM_VOCABULARY>VoiceInk", system)
        self.assertNotIn("CURRENT_WINDOW_CONTEXT", user)
        self.assertEqual(user, "<TRANSCRIPT>um test voice ink please</TRANSCRIPT>")

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
        self.assertIn("<CURRENT_WINDOW_CONTEXT>", converted["messages"][0]["content"])
        self.assertIn("<TRANSCRIPT>", converted["messages"][1]["content"])
        self.assertEqual(converted["messages"][2]["content"], "Test VoiceInk, please.")

    def test_prepared_layout_remains_available_for_legacy_experiments(self) -> None:
        converted = convert_record(
            labeled_record(), "new prompt", message_layout="prepared"
        )

        system = content_text(converted["messages"][0]["content"])
        user = content_text(converted["messages"][1]["content"])
        self.assertEqual(system, "new prompt")
        self.assertIn("<CURRENT_WINDOW_CONTEXT>", user)
        self.assertIn("<TRANSCRIPT>", user)

    def test_layout_conversion_preserves_context_transcript_and_label(self) -> None:
        prepared = convert_record(
            labeled_record(), "new prompt", message_layout="prepared"
        )

        production = convert_conversation_layout(prepared, "voiceink")

        system = content_text(production["messages"][0]["content"])
        user = content_text(production["messages"][1]["content"])
        assistant = content_text(production["messages"][2]["content"])
        self.assertIn("<CURRENT_WINDOW_CONTEXT>\nTerminal", system)
        self.assertIn("<CUSTOM_VOCABULARY>\nVoiceInk", system)
        self.assertEqual(user, "<TRANSCRIPT>\num test voice ink please\n</TRANSCRIPT>")
        self.assertEqual(assistant, "Test VoiceInk, please.")

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

    def test_lfm_v3_profile_uses_locked_samples_and_faster_batches(self) -> None:
        args = parse_lfm_v3_args(["--check-only"])

        self.assertEqual(str(args.train), "datasets/lfm25-v3/train.jsonl")
        self.assertEqual(
            str(args.eval), "datasets/lfm25-v3/eval-regression-340.jsonl"
        )
        self.assertEqual((args.batch_size, args.grad_accum), (4, 2))
        self.assertEqual(args.eval_batch_size, 4)
        self.assertTrue(args.load_best_model_at_end)
        self.assertIn("lfm25-1.2b-voiceink-v3", str(args.output_dir))
        self.assertEqual(args.gguf_base.name, "LFM2.5-1.2B-VoiceInk-v3")

    def test_lfm_26b_v3_profile_matches_locked_qwen_v3_recipe(self) -> None:
        args = parse_lfm_26b_v3_args(["--check-only"])

        self.assertEqual(args.base_model, "LiquidAI/LFM2.5-2.6B-Base")
        self.assertEqual(str(args.train), "datasets/lfm25-v3/train.jsonl")
        self.assertEqual(
            str(args.eval), "datasets/lfm25-v3/eval-regression-340.jsonl"
        )
        self.assertEqual(args.epochs, 1)
        self.assertEqual((args.r, args.lora_alpha), (32, 64))
        self.assertEqual((args.batch_size, args.grad_accum), (4, 2))
        self.assertEqual(args.batch_size * args.grad_accum, 8)
        self.assertEqual(args.eval_batch_size, 4)
        self.assertEqual(args.lr, 2e-4)
        self.assertIn("lfm25-2.6b-voiceink-v3", str(args.output_dir))
        self.assertEqual(args.gguf_base.name, "LFM2.5-2.6B-VoiceInk-v3")
        self.assertTrue(args.check_only)

    def test_lfm_v3_conversion_only_changes_content_representation(self) -> None:
        source = convert_record(labeled_record(), "new prompt")
        converted = convert_conversation(source, Path("source.jsonl"), 1)

        self.assertEqual(
            [message["role"] for message in converted["messages"]],
            ["system", "user", "assistant"],
        )
        for original, result in zip(source["messages"], converted["messages"]):
            self.assertEqual(content_text(original["content"]), result["content"])
            self.assertIsInstance(result["content"], str)


class QwenTrainerProfileTests(unittest.TestCase):
    def test_08b_profile_reuses_locked_v3_data_with_isolated_outputs(self) -> None:
        args = parse_qwen_08b_args(["--check-only"])

        self.assertEqual(args.base_model, "unsloth/Qwen3.5-0.8B")
        self.assertEqual(
            str(args.train), "datasets/qwen35-2b-voiceink-v3/train.jsonl"
        )
        self.assertEqual(
            str(args.eval),
            "datasets/qwen35-2b-voiceink-v3/eval-regression-340.jsonl",
        )
        self.assertEqual(args.epochs, 1)
        self.assertEqual((args.r, args.lora_alpha), (32, 64))
        self.assertEqual((args.batch_size, args.grad_accum), (4, 2))
        self.assertEqual(args.batch_size * args.grad_accum, 8)
        self.assertEqual(args.eval_batch_size, 4)
        self.assertEqual((args.eval_steps, args.save_steps), (50, 50))
        self.assertIn("qwen35-08b-voiceink-v1", str(args.lora_dir))
        self.assertEqual(args.gguf_base.name, "Qwen3.5-0.8B-VoiceInk-v1")
        self.assertTrue(args.check_only)

    def test_08b_profile_allows_smoke_output_overrides(self) -> None:
        args = parse_qwen_08b_args([
            "--max-steps", "1",
            "--lora-dir", "training/qwen35-08b-voiceink-v1/smoke-lora",
            "--output-dir", "training/qwen35-08b-voiceink-v1/smoke-outputs",
        ])

        self.assertEqual(args.max_steps, 1)
        self.assertIn("smoke-lora", str(args.lora_dir))
        self.assertIn("smoke-outputs", str(args.output_dir))

    def test_08b_profile_plans_496_steps_with_faster_micro_batches(self) -> None:
        args = parse_qwen_08b_args([])

        total = expected_training_steps(
            3962,
            args.batch_size,
            args.grad_accum,
            args.epochs,
            args.max_steps,
        )

        self.assertEqual(total, 496)
        self.assertEqual(align_step_interval(total, args.eval_steps), 62)

    def test_eval_interval_is_unchanged_when_terminal_step_already_aligns(self) -> None:
        self.assertEqual(align_step_interval(500, 50), 50)

    def test_max_steps_controls_terminal_eval_alignment(self) -> None:
        self.assertEqual(expected_training_steps(3962, 4, 2, 1, 7), 7)
        self.assertEqual(align_step_interval(7, 50), 7)

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
