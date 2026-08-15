import argparse
import tempfile
import unittest
from pathlib import Path

from src.common.mlflow_tracking import (
    add_mlflow_args,
    dataset_metadata,
    mlflow_model_name,
    numeric_metrics,
    output_reference,
    resumed_param_updates,
    start_mlflow_run,
)


class MlflowTrackingTests(unittest.TestCase):
    def test_cli_defaults_to_tracking_and_supports_explicit_opt_out(self) -> None:
        parser = argparse.ArgumentParser()
        add_mlflow_args(parser, default_experiment="voiceink-test")

        enabled = parser.parse_args([])
        disabled = parser.parse_args(["--no-mlflow"])

        self.assertTrue(enabled.mlflow_enabled)
        self.assertFalse(disabled.mlflow_enabled)
        self.assertEqual(enabled.mlflow_experiment, "voiceink-test")
        self.assertIsNone(enabled.mlflow_run_id)

    def test_private_dataset_metadata_has_fingerprint_not_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.jsonl"
            path.write_text('{"private":"do not log"}\n', encoding="utf-8")

            metadata = dataset_metadata("eval", path, 1)

        self.assertEqual(metadata["records"], 1)
        self.assertEqual(len(metadata["sha256"]), 64)
        self.assertFalse(metadata["raw_data_logged"])
        self.assertNotIn("private", metadata)

    def test_numeric_metrics_flattens_only_finite_numbers(self) -> None:
        metrics = numeric_metrics({
            "loss": 0.25,
            "nested": {"count": 440, "skip": float("nan")},
            "flag": True,
            "label": "tie",
        })

        self.assertEqual(metrics, {"loss": 0.25, "nested.count": 440.0})

    def test_disabled_run_does_not_import_or_contact_mlflow(self) -> None:
        args = argparse.Namespace(
            mlflow_enabled=False,
            mlflow_run_name=None,
        )

        run = start_mlflow_run(
            args,
            run_name="disabled",
            run_kind="training.test",
            params={},
        )

        self.assertFalse(run.enabled)
        self.assertIsNone(run.log_external_model(
            name="none", model_type="test", params={}
        ))
        run.finish()

    def test_output_reference_never_marks_private_output_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.jsonl"
            path.write_text("private details\n", encoding="utf-8")

            reference = output_reference("eval_details", path, private=True)

        self.assertTrue(reference["private"])
        self.assertFalse(reference["uploaded"])
        self.assertEqual(len(reference["sha256"]), 64)

    def test_disabled_external_model_accepts_production_display_name(self) -> None:
        args = argparse.Namespace(mlflow_enabled=False, mlflow_run_name=None)
        run = start_mlflow_run(
            args, run_name="disabled", run_kind="training.test", params={}
        )

        self.assertIsNone(run.log_external_model(
            name="LFM2.5-2.6B-VoiceInk-v3",
            model_type="voiceink-transcript-cleanup",
            params={},
        ))

    def test_production_model_name_is_sanitized_for_mlflow(self) -> None:
        self.assertEqual(
            mlflow_model_name("LFM2.5-2.6B-VoiceInk-v3"),
            "LFM2-5-2-6B-VoiceInk-v3",
        )
        with self.assertRaises(ValueError):
            mlflow_model_name(".../")

    def test_resumed_params_preserve_immutable_values(self) -> None:
        additions, changes = resumed_param_updates(
            {"resume": "", "samples": "440"},
            {"resume": "results/partial.jsonl", "samples": 440, "retry": True},
        )

        self.assertEqual(additions, {"retry": True})
        self.assertEqual(changes, {
            "resume": {"original": "", "resumed": "results/partial.jsonl"}
        })


if __name__ == "__main__":
    unittest.main()
