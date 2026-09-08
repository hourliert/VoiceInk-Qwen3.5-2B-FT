import hashlib
import json
import sys
import tempfile
import tomllib
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.consolidate_release import consolidate_rows, fingerprint
from data.legacy_layout import apply_plan, build_plan
from data.manage import BOOTSTRAP_SPEC, EXPECTED_SPLITS, select_bootstrap
from data.manifest import canonical_json, load_manifest, sha256_file
from data.schema import migrate
from data.seal_preference import seal
from eval.strict import verify_pair
from labeling.annotation_store import AnnotationStore


def sample(index, kind):
    transcript = f"ordinary dictated sentence {index}"
    label = transcript + "."
    window = ""
    if kind == "no-op":
        label = transcript
    elif kind == "long":
        transcript = " ".join([f"word{index}"] * 210)
        label = transcript + "."
    elif kind == "technical":
        transcript = f"Please update API QWEN_{index} version {index}.2"
    elif kind == "context":
        window = " ".join(["context"] * 810)
    return {"request_id": f"id-{index:05d}", "transcript": transcript,
            "label": label, "origin": "human", "window_context": window,
            "transcript_fingerprint": hashlib.sha256(transcript.encode()).hexdigest()}


def release(root):
    splits = {}
    for role, count in EXPECTED_SPLITS.items():
        path = root / f"{role}.jsonl"
        path.write_text("".join(json.dumps({"row": i}) + "\n" for i in range(count)))
        splits[role] = {"path": path.name, "records": count, "sha256": sha256_file(path)}
    prompt = root / "system-prompt.txt"
    prompt.write_text("Clean the transcript.")
    regression = root / "regression.jsonl"
    regression.write_text("".join(json.dumps({"row": i}) + "\n" for i in range(440)))
    manifest = {
        "schema": "voiceink.dataset-release/v2",
        "name": "test-v1",
        "sealed": True,
        "splits": splits,
        "system_prompt": {"path": prompt.name, "sha256": sha256_file(prompt)},
        "historical_regression": {
            "path": regression.name,
            "records": 440,
            "sha256": sha256_file(regression),
            "checkpoint_selection": False,
        },
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class CanonicalPipelineTests(unittest.TestCase):
    def test_consolidation_excludes_holdouts_and_reviewed_labels_override_legacy(self):
        def row(raw, label):
            return {"messages": [
                {"role": "user", "content": f"<TRANSCRIPT>{raw}</TRANSCRIPT>"},
                {"role": "assistant", "content": label},
            ]}

        legacy = [
            row("Keep me", "legacy"),
            row("  KEEP   ME  ", "duplicate"),
            row("Replace me", "old label"),
            row("Holdout", "must be excluded"),
        ]
        reviewed = [row("replace me", "reviewed label"), row("New sample", "new")]
        rows, report = consolidate_rows(
            legacy, reviewed, {fingerprint(row("holdout", "ignored"))}
        )

        self.assertEqual(len(rows), 3)
        labels = {fingerprint(item): item["messages"][-1]["content"] for item in rows}
        self.assertEqual(labels[fingerprint(row("replace me", "ignored"))], "reviewed label")
        self.assertEqual(report, {
            "legacy_duplicates_removed": 1,
            "legacy_holdout_rows_removed": 1,
            "reviewed_replacements": 1,
            "reviewed_additions": 1,
        })

    def test_bootstrap_selection_has_exact_fixed_shape(self):
        rows = []
        index = 0
        for kind, count in (("no-op", 140), ("long", 220), ("technical", 320),
                            ("context", 340), ("representative", 900)):
            for _ in range(count):
                rows.append(sample(index, kind))
                index += 1
        selected = select_bootstrap(rows, 20260817)
        self.assertEqual(len(selected), 1200)
        self.assertEqual(Counter(row["split"] for row in selected), EXPECTED_SPLITS)
        self.assertEqual(Counter(row["stratum"] for row in selected), {
            key: sum(value.values()) for key, value in BOOTSTRAP_SPEC.items()})
        self.assertEqual(len({row["transcript_fingerprint"] for row in selected}), 1200)

    def test_manifest_fails_closed_on_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = release(root)
            self.assertEqual(load_manifest(manifest)["name"], "test-v1")
            with (root / "validation.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_manifest(manifest)

    def test_strict_verifier_requires_both_complete_corpora(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = release(root)
            def summary(path, count):
                path.write_text(json.dumps({
                    "baseline_model": "baseline", "candidate_model": "candidate",
                    "n_samples": count, "baseline_avg_score": 95.0, "candidate_avg_score": 95.0,
                    "per_dimension": {
                        "meaning_preservation": {"baseline_avg": 4.7, "candidate_avg": 4.7, "p_value": 1.0},
                        "technical_accuracy": {"baseline_avg": 4.6, "candidate_avg": 4.6, "p_value": 1.0}},
                    "baseline_latency": {"p50": 100, "p90": 200},
                    "candidate_latency": {"p50": 105, "p90": 220}}))
                return path
            def details(path, count):
                row = {"candidate_scores": {"meaning_preservation": 5, "technical_accuracy": 5}}
                path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)))
                return path
            result = verify_pair(summary(root / "r.json", 440), summary(root / "a.json", 149),
                                 details(root / "r.jsonl", 440), details(root / "a.jsonl", 149),
                                 manifest)
            self.assertFalse(result["passed"])
            self.assertTrue(any("149/150" in failure for failure in result["failures"]))
            self.assertIn("regression_details_sha256", result["evidence"])
            self.assertIn("acceptance_details_sha256", result["evidence"])

    def test_schema_v2_is_additive(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AnnotationStore(Path(directory) / "annotations.sqlite3")
            migrate(store.connection)
            tables = {row[0] for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"canonical_cohorts", "canonical_releases", "model_promotions"} <= tables)
            self.assertEqual(store.integrity_check(), "ok")
            store.close()

    def test_manifest_verifies_every_declared_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = release(root)
            manifest = json.loads(manifest_path.read_text())
            text_splits = {}
            for role, metadata in manifest["splits"].items():
                source = root / metadata["path"]
                target = root / f"{role}-text.jsonl"
                target.write_bytes(source.read_bytes())
                text_splits[role] = {
                    "path": target.name,
                    "records": metadata["records"],
                    "sha256": sha256_file(target),
                }
            manifest["representations"] = {
                "text-blocks": manifest["splits"],
                "text": text_splits,
            }
            manifest.pop("content_sha256")
            manifest["content_sha256"] = hashlib.sha256(
                canonical_json(manifest).encode()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            load_manifest(manifest_path)
            with (root / "validation-text.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "text/validation"):
                load_manifest(manifest_path)

    def test_preference_release_is_sealed_and_parent_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_root = root / "parent"
            parent_root.mkdir()
            parent = release(parent_root)
            pair = {
                "request_id": "preference-1",
                "prompt": [{"role": "user", "content": [{"type": "text", "text": "raw"}]}],
                "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "good"}]}],
                "rejected": [{"role": "assistant", "content": [{"type": "text", "text": "bad"}]}],
            }
            train = root / "preference-train.jsonl"
            validation = root / "preference-validation.jsonl"
            train.write_text(json.dumps(pair) + "\n")
            validation.write_text(json.dumps(pair | {"request_id": "preference-2"}) + "\n")
            judgments = root / "judgments.jsonl"
            judgments.write_text(json.dumps({"request_id": "preference-1"}) + "\n")
            prompt = root / "judge-prompt.txt"
            prompt.write_text("Judge strictly.")
            sealed = seal(
                "preference-v1",
                train,
                validation,
                parent,
                judgments,
                prompt,
                root / "releases",
            )
            loaded = load_manifest(Path(sealed["manifest_path"]))
            self.assertEqual(loaded["task"], "preference")
            self.assertEqual(loaded["parent_release"]["name"], "test-v1")

    def test_legacy_model_rewrite_handles_overlapping_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "models" / "Qwen3.5-2B-VoiceInk-v3"
            legacy = root / "models" / "Qwen3.5-2B-VoiceInk-v3-DPO-v1_gguf"
            canonical.mkdir(parents=True)
            legacy.mkdir(parents=True)
            config = root / "config" / "models.ini"
            config.parent.mkdir()
            dpo_model = legacy / "model.gguf"
            config.write_text(f"[DPO]\nmodel = {dpo_model}\n")
            apply_plan(root, build_plan(root), allow_running_training=True)
            updated = config.read_text()
            self.assertIn("models/legacy/Qwen3.5-2B-VoiceInk-v3-DPO-v1_gguf", updated)
            self.assertNotIn("models/canonical/Qwen3.5-2B-VoiceInk-v3-DPO", updated)

    def test_training_profiles_write_only_to_canonical_namespaces(self):
        profiles = tomllib.loads(
            (ROOT / "config" / "training_profiles.toml").read_text(encoding="utf-8")
        )["profiles"]
        for profile in profiles.values():
            self.assertTrue(profile["output_root"].startswith("training/canonical/"))
            self.assertTrue(profile["model_root"].startswith("models/canonical/"))
        self.assertEqual(
            profiles["qwen35-2b-dpo"]["base_model"],
            "models/canonical/Qwen3.5-2B-VoiceInk-v3",
        )
        qwen_sft = profiles["qwen35-2b-sft"]
        self.assertEqual(qwen_sft["batch_size"], 1)
        self.assertEqual(qwen_sft["grad_accum"], 8)
        self.assertEqual(qwen_sft["eval_batch_size"], 1)
        self.assertEqual(qwen_sft["batch_size"] * qwen_sft["grad_accum"], 8)
        minicpm = profiles["minicpm5-2b-sft"]
        self.assertEqual(minicpm["trainer"], "minicpm")
        self.assertEqual(minicpm["base_model"], "openbmb/MiniCPM5-2B")
        self.assertEqual(minicpm["batch_size"] * minicpm["grad_accum"], 8)

    def test_legacy_layout_is_explicit_and_non_destructive_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (
                root / "datasets" / "old-data",
                root / "training" / "old-run",
                root / "training" / "qwen38-2b-voiceink-v1",
                root / "results" / "old-eval",
                root / "models" / "Qwen3.5-0.8B-VoiceInk-v1",
                root / "models" / "Qwen3.5-2B-VoiceInk-v3",
            ):
                path.mkdir(parents=True)
            (root / "datasets" / "labeled.jsonl.bak").write_text("private")
            moves = build_plan(root)
            targets = {Path(move["target"]).relative_to(root).as_posix() for move in moves}
            self.assertIn("datasets/legacy/old-data", targets)
            self.assertIn("datasets/legacy/root-files/labeled.jsonl.bak", targets)
            self.assertIn("training/legacy/old-run", targets)
            self.assertIn("training/canonical/qwen38-2b-voiceink-v1", targets)
            self.assertIn("results/legacy/old-eval", targets)
            self.assertIn("models/legacy/Qwen3.5-0.8B-VoiceInk-v1", targets)
            self.assertIn("models/canonical/Qwen3.5-2B-VoiceInk-v3", targets)
            self.assertTrue((root / "training" / "old-run").is_dir())


if __name__ == "__main__":
    unittest.main()
