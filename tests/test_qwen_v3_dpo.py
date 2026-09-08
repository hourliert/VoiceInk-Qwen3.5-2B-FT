import json
import unittest

from src.training.finetune_qwen35_2b_v3_dpo import profile_argv
from src.training.prepare_dataset import convert_record
from src.training.prepare_qwen_v3_preferences import (
    DEFAULT_REJECTION_MODEL,
    accept_judgment,
    candidate_record,
    prompt_digest,
)


def approved_label() -> dict:
    request = {
        "model": "VoiceInk",
        "messages": [
            {
                "role": "system",
                "content": (
                    "<SYSTEM_INSTRUCTIONS>old</SYSTEM_INSTRUCTIONS>\n"
                    "<CURRENT_WINDOW_CONTEXT>Terminal</CURRENT_WINDOW_CONTEXT>\n"
                    "<CUSTOM_VOCABULARY>VoiceInk</CUSTOM_VOCABULARY>"
                ),
            },
            {
                "role": "user",
                "content": "<TRANSCRIPT>update the modal please</TRANSCRIPT>",
            },
        ],
    }
    return {
        "request_id": "request-1",
        "raw_request_json": json.dumps(request),
        "label": "Update the model, please.",
        "auto_review": {"status": "approved", "policy": "strict"},
    }


def rejection(label: dict, prompt: str, **updates) -> dict:
    conversation = convert_record(label, prompt, "text-blocks")
    record = {
        "request_id": label["request_id"],
        "model": DEFAULT_REJECTION_MODEL,
        "response": "Update the modal, please.",
        "prompt_sha256": prompt_digest(conversation["messages"][:2]),
    }
    record.update(updates)
    return record


def scores(**updates) -> dict:
    result = {
        "meaning_preservation": 5,
        "instruction_following": 5,
        "filler_removal": 5,
        "grammar_fluency": 5,
        "technical_accuracy": 5,
        "conciseness": 5,
    }
    result.update(updates)
    return result


class QwenV3PreferenceTests(unittest.TestCase):
    def test_candidate_uses_fresh_v3_rejection_and_matching_prompt(self) -> None:
        label = approved_label()
        prompt = "Clean the transcript."

        candidate, reason = candidate_record(
            label,
            rejection(label, prompt),
            "qwen-risk",
            prompt,
            DEFAULT_REJECTION_MODEL,
            1,
            0.5,
        )

        self.assertEqual(reason, "candidate")
        self.assertEqual(candidate["rejection_model"], DEFAULT_REJECTION_MODEL)
        self.assertEqual(candidate["judge_sample"]["raw_transcript"],
                         "update the modal please")

    def test_candidate_rejects_stale_model_or_prompt(self) -> None:
        label = approved_label()
        prompt = "Clean the transcript."

        _, model_reason = candidate_record(
            label,
            rejection(label, prompt, model="Qwen3.5-2B-VoiceInk"),
            "qwen-risk", prompt, DEFAULT_REJECTION_MODEL, 1, 0.5,
        )
        _, prompt_reason = candidate_record(
            label,
            rejection(label, prompt, prompt_sha256="wrong"),
            "qwen-risk", prompt, DEFAULT_REJECTION_MODEL, 1, 0.5,
        )

        self.assertEqual(model_reason, "wrong-rejection-model")
        self.assertEqual(prompt_reason, "prompt-fingerprint-mismatch")

    def test_filter_requires_material_semantic_or_technical_gain(self) -> None:
        judgment = {
            "pairwise_preference": "chosen",
            "pairwise_confidence": "high",
            "pairwise_material_difference": True,
            "chosen_scores": scores(),
            "rejected_scores": scores(technical_accuracy=3),
        }
        accepted, reason = accept_judgment(judgment)
        self.assertTrue(accepted)
        self.assertEqual(reason, "selected")

        grammar_only = judgment | {
            "rejected_scores": scores(grammar_fluency=3),
        }
        accepted, reason = accept_judgment(grammar_only)
        self.assertFalse(accepted)
        self.assertEqual(reason, "no-semantic-or-technical-gain")

    def test_filter_rejects_ties_and_protected_regressions(self) -> None:
        tied = {
            "pairwise_preference": "tie",
            "pairwise_confidence": "high",
            "pairwise_material_difference": True,
            "chosen_scores": scores(),
            "rejected_scores": scores(meaning_preservation=3),
        }
        self.assertEqual(accept_judgment(tied), (False, "judge-tie"))

        regression = tied | {
            "pairwise_preference": "chosen",
            "chosen_scores": scores(technical_accuracy=4),
            "rejected_scores": scores(meaning_preservation=3, technical_accuracy=5),
        }
        self.assertEqual(
            accept_judgment(regression),
            (False, "protected-dimension-regression"),
        )

    def test_training_profile_is_v3_specific_and_isolated(self) -> None:
        arguments = profile_argv([])
        joined = " ".join(arguments)
        self.assertIn("models/Qwen3.5-2B-VoiceInk-v3", joined)
        self.assertIn("datasets/qwen35-2b-voiceink-dpo-v3/train.jsonl", joined)
        self.assertIn("Qwen3.5-2B-VoiceInk-v3-DPO-v1", joined)
        self.assertIn("qwen35-2b-voiceink-v3-dpo-v1", joined)


if __name__ == "__main__":
    unittest.main()
