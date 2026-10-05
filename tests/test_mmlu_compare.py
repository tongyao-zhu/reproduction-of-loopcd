"""Grouped benchmark evidence must retain subject identity and correct weighting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("compare_group", Path(__file__).resolve().parents[1] / "scripts/compare.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MMLUComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = []
        for mode, correct in [("baseline", [1, 0, 1]), ("fixed", [1, 1, 0]), ("adaptive", [1, 1, 1])]:
            path = Path(self.temp.name) / mode
            path.mkdir()
            self.runs.append(path)
            manifest = dict(status="completed", task="mmlu", limit=None, is_full_split=True,
                            guidance={"mode": mode}, paper_config={"mmlu_expected_subjects": 2},
                            batch_size=1, seed=42, chat_template=False,
                            provenance={"model": {"repo_id": "test/model", "revision": "v1", "model_code_sha256": "code"},
                                        "loaded_model_code_sha256": "code", "source_sha256": {"test": "hash"}},
                            datasets={}, samples={})
            result = {"results": {"mmlu": {"acc,none": sum(correct) / 3}}, "configs": {}, "n-samples": {}}
            for subject, indices in [("mmlu_a", [0, 1]), ("mmlu_b", [2])]:
                n = len(indices)
                manifest["datasets"][subject] = {"splits": {"test": {"rows": n, "fingerprint": subject}}}
                manifest["samples"][subject] = n
                result["configs"][subject] = {"test_split": "test"}
                result["n-samples"][subject] = {"original": n, "effective": n}
                result["results"][subject] = {"acc,none": sum(correct[i] for i in indices) / n}
                rows = [dict(doc_id=j, doc_hash=f"doc-{i}", prompt_hash=f"prompt-{i}", target_hash=f"gold-{i}", acc=correct[i])
                        for j, i in enumerate(indices)]
                (path / f"samples_{subject}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "results.json").write_text(json.dumps(result))

    def test_subject_ids_and_micro_weighting(self):
        summary = module.compare_runs(self.runs)
        self.assertEqual(summary["n_documents"], 3)
        self.assertTrue(summary["full_split_count_verified"])
        self.assertAlmostEqual(summary["metrics"]["acc"]["percent"]["baseline"], 200 / 3)
        self.assertEqual(summary["metrics"]["acc"]["paired_vs_baseline"]["fixed"]["wins"], 1)
        self.assertEqual(summary["metrics"]["acc"]["paired_vs_baseline"]["fixed"]["losses"], 1)

    def test_missing_subject_file_rejected(self):
        (self.runs[1] / "samples_mmlu_b.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "evidence file set"):
            module.compare_runs(self.runs)

    def test_unweighted_group_aggregate_rejected(self):
        path = self.runs[0] / "results.json"
        value = json.loads(path.read_text())
        value["results"]["mmlu"]["acc,none"] = 0.75
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "sample mean disagrees"):
            module.compare_runs(self.runs)
