"""Synthetic tests of pairing guards and per-document comparison statistics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("compare", Path(__file__).resolve().parents[1] / "scripts" / "compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directories = [Path(self.temp.name) / mode for mode in compare.MODES]
        scores = {"baseline": [1, 0, 0, 1], "fixed": [1, 1, 1, 0], "adaptive": [1, 0, 1, 1]}
        for mode, directory in zip(compare.MODES, self.directories):
            directory.mkdir()
            manifest = {
                "status": "completed", "task": "sciq", "limit": 4, "is_full_split": False,
                "guidance": {"mode": mode, "omega": 0.5, "omega_cap": 1.0, "early_loop": 1},
                "paper_config": {"paper_sciq_percent": {"baseline": 94.7, "fixed": 95.3, "adaptive": 95.2}},
                "batch_size": 1, "seed": 42, "chat_template": False,
                "provenance": {
                    "model": {"repo_id": "example/model", "revision": "model-v1", "model_code_sha256": "code-v1"},
                    "loaded_model_code_sha256": "code-v1", "source_sha256": {"guidance.py": "source-v1"}},
                "datasets": {"sciq": {"revision": "dataset-v1", "splits": {"test": {"rows": 1000, "fingerprint": "data-v1"}}}},
                "samples": {"sciq": 4},
            }
            result = {"results": {"sciq": {"acc,none": sum(scores[mode]) / 4, "acc_norm,none": 0.5}},
                      "n-samples": {"sciq": {"original": 1000, "effective": 4}}}
            rows = [{"doc_id": index, "doc_hash": f"doc-{index}", "prompt_hash": f"prompt-{index}",
                     "target_hash": f"target-{index}", "acc": value, "acc_norm": index % 2}
                    for index, value in enumerate(scores[mode])]
            self.write_json(directory / "manifest.json", manifest)
            self.write_json(directory / "results.json", result)
            self.write_rows(directory, rows)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def write_rows(directory, rows):
        (directory / "samples_sciq.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def mutate_manifest(self, callback):
        path = self.directories[1] / "manifest.json"
        manifest = json.loads(path.read_text())
        callback(manifest)
        self.write_json(path, manifest)

    def mutate_rows(self, callback):
        rows = [json.loads(line) for line in (self.directories[1] / "samples_sciq.jsonl").read_text().splitlines()]
        callback(rows)
        self.write_rows(self.directories[1], rows)

    def test_known_wins_losses_and_se(self):
        result = compare.compare_runs(self.directories)
        self.assertEqual(result["metrics"]["acc"]["percent"], {"baseline": 50, "fixed": 75, "adaptive": 75})
        fixed = result["metrics"]["acc"]["paired_vs_baseline"]["fixed"]
        self.assertEqual((fixed["wins"], fixed["losses"], fixed["ties"]), (2, 1, 1))
        self.assertEqual(fixed["delta_percentage_points"], 25)
        self.assertAlmostEqual(fixed["paired_se_percentage_points"], (11 / 48) ** 0.5 * 100)
        self.assertEqual(result["metrics"]["acc_norm"]["paired_vs_baseline"]["fixed"]["paired_se_percentage_points"], 0)
        self.assertIn("not a reproduction", result["interpretation"])
        self.assertIsNone(result["paper_targets"]["metric"])

    def test_reordered_rows_are_paired_by_id(self):
        self.mutate_rows(lambda rows: rows.reverse())
        result = compare.compare_runs(self.directories)
        self.assertEqual(result["metrics"]["acc"]["paired_vs_baseline"]["fixed"]["wins"], 2)

    def test_duplicate_doc_id_rejected(self):
        self.mutate_rows(lambda rows: rows.append(copy.deepcopy(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            compare.compare_runs(self.directories)

    def test_misaligned_doc_id_rejected(self):
        self.mutate_rows(lambda rows: rows[0].update(doc_id=99))
        with self.assertRaisesRegex(ValueError, "doc_id set"):
            compare.compare_runs(self.directories)

    def test_misaligned_prompt_rejected(self):
        self.mutate_rows(lambda rows: rows[0].update(prompt_hash="different"))
        with self.assertRaisesRegex(ValueError, "prompt_hash"):
            compare.compare_runs(self.directories)

    def test_missing_prompt_rejected(self):
        self.mutate_rows(lambda rows: rows[0].pop("prompt_hash"))
        with self.assertRaisesRegex(ValueError, "missing prompt_hash"):
            compare.compare_runs(self.directories)

    def test_model_revision_rejected(self):
        self.mutate_manifest(lambda value: value["provenance"]["model"].update(revision="other"))
        with self.assertRaisesRegex(ValueError, "provenance.model"):
            compare.compare_runs(self.directories)

    def test_effective_model_code_rejected(self):
        self.mutate_manifest(lambda value: value["provenance"].update(loaded_model_code_sha256="other"))
        with self.assertRaisesRegex(ValueError, "loaded model code hash"):
            compare.compare_runs(self.directories)

    def test_source_hash_mismatch_rejected(self):
        self.mutate_manifest(lambda value: value["provenance"]["source_sha256"].update(**{"guidance.py": "other"}))
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            compare.compare_runs(self.directories)

    def test_dataset_fingerprint_mismatch_rejected(self):
        self.mutate_manifest(lambda value: value["datasets"]["sciq"]["splits"]["test"].update(fingerprint="other"))
        with self.assertRaisesRegex(ValueError, "datasets"):
            compare.compare_runs(self.directories)

    def test_aggregate_disagreement_rejected(self):
        path = self.directories[1] / "results.json"
        result = json.loads(path.read_text())
        result["results"]["sciq"]["acc,none"] = 0.9
        self.write_json(path, result)
        with self.assertRaisesRegex(ValueError, "sample mean disagrees"):
            compare.compare_runs(self.directories)

    def test_subset_cannot_claim_full_split(self):
        for directory in self.directories:
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(limit=None, is_full_split=True)
            self.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "full split sample count"):
            compare.compare_runs(self.directories)


if __name__ == "__main__":
    unittest.main()
