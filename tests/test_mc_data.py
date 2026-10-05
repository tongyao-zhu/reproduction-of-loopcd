import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("check_mc_data", Path(__file__).resolve().parents[1] / "scripts/check_mc_data.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
REGISTRY = json.loads((Path(__file__).resolve().parents[1] / "configs/mc_datasets.json").read_text())


class MCDataTests(unittest.TestCase):
    def test_group_spec_does_not_override_subject_name(self):
        spec = module.build_task_spec("mmlu", REGISTRY)
        self.assertNotIn("dataset_name", spec)
        self.assertEqual(spec["dataset_kwargs"]["revision"], REGISTRY["tasks"]["mmlu"]["revision"])

    def test_native_path_and_config_are_preserved(self):
        spec = module.build_task_spec("arc_challenge", REGISTRY)
        self.assertEqual(spec["dataset_name"], "ARC-Challenge")
        self.assertEqual(spec["dataset_path"], "allenai/ai2_arc")
        self.assertEqual(module.build_task_spec("sciq", REGISTRY)["dataset_path"], "sciq")

    def test_nested_leaf_traversal(self):
        class Leaf:
            dataset = {}
        leaf = Leaf()
        self.assertEqual(list(module.iter_leaf_tasks({"group": {"inner": {"subject": leaf}}})), [("subject", leaf)])

    def test_offline_latest_cache_cannot_silently_replace_pin(self):
        class Config:
            metadata = {}
        class Data:
            cache_files = [{"filename": "/cache/other-revision/dataset.arrow"}]
        class Task:
            config = Config()
            dataset = {"test": Data()}
        with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
            module.validate_dataset_pin(Task(), REGISTRY["tasks"]["arc_easy"])

    def test_local_group_preserves_aggregation_and_hashes_sources(self):
        class Manager:
            task_index = {"mmlu": {"type": "group"}, "mmlu_stem_tasks": {"type": "tag"}, "mmlu_subject": {"type": "task"}}
            def _get_tasklist(self, name):
                return ["mmlu_subject"]
            def _get_config(self, name):
                if name == "mmlu":
                    return {"group": "mmlu", "task": ["mmlu_stem_tasks"], "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}]}
                return {"task": name, "dataset_path": "cais/mmlu", "dataset_name": "subject", "test_split": "test", "fewshot_split": "dev", "doc_to_text": "unchanged prompt"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "datasets--cais--mmlu" / "snapshots" / REGISTRY["tasks"]["mmlu"]["revision"] / "subject"
            snapshot.mkdir(parents=True)
            for split in ("dev", "test"):
                (snapshot / f"{split}-00000-of-00001.parquet").write_bytes(split.encode())
            spec = module.build_local_task_spec("mmlu", REGISTRY, Manager(), root / "cache", root)
            self.assertTrue(spec["aggregate_metric_list"][0]["weight_by_size"])
            leaf = spec["task"][0]
            self.assertEqual(leaf["doc_to_text"], "unchanged prompt")
            self.assertEqual(leaf["dataset_name"], "subject")
            self.assertEqual(len(leaf["metadata"]["loopcd_dataset"]["raw_files"]), 2)
            self.assertEqual(leaf["metadata"]["loopcd_dataset"]["raw_files"][0]["sha256"], module.sha256(snapshot / "dev-00000-of-00001.parquet"))


if __name__ == "__main__":
    unittest.main()
