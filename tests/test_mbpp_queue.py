"""Reject incomplete or unpaired generation before advancing to full MBPP."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_mbpp_suite as queue


class PairGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.problems = [{"task_id": "Mbpp/2", "prompt": "public"}, {"task_id": "Mbpp/3", "prompt": "another"}]
        self.mock = patch.object(queue, "load_problems", return_value=self.problems)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        for arm, (mode, loops, ref) in queue.ARMS.items():
            path = self.root / arm
            path.mkdir()
            config = {"task_ids": [p["task_id"] for p in self.problems], "limit": 2, "seed": 42,
                      "guidance": {"mode": mode, "total_loops": loops, "reference_loop": ref, "omega": .3},
                      "max_new_tokens": 2048, "native_context_length": 4096, "source": {"pin": "same"}}
            digest = queue.digest(config)
            manifest = {"status": "completed", "completed_samples": 2, "expected_samples": 2,
                        "is_full_split": False, "config": config, "config_hash": digest}
            (path / "manifest.json").write_text(json.dumps(manifest))
            rows = [{"task_id": p["task_id"], "config_hash": digest, "problem_sha256": queue.digest(p),
                     "prompt": p["prompt"], "prompt_sha256": queue.digest(p["prompt"]),
                     "prompt_token_ids": [65504, 12, 13], "seed": queue.task_seed(p["task_id"], 42),
                     "effective_max_new_tokens": 2048, "generated_tokens": 2,
                     "generated_token_ids": [3, 65505], "cap_hit": False} for p in self.problems]
            (path / "samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

    def mutate(self, callback):
        path = self.root / "hidden16/samples.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        callback(rows)
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))

    def test_valid_smoke_is_unscored_and_not_full(self):
        result = queue.verify_generation(self.root, "unused", 2, False)
        self.assertTrue(result["paired_prompts_tokens_data_seeds_caps"])
        self.assertFalse(result["generated_code_executed"])
        self.assertFalse(result["is_full_split"])

    def test_missing_or_duplicated_result_is_rejected(self):
        self.mutate(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "task IDs"):
            queue.verify_generation(self.root, "unused", 2, False)

    def test_prompt_token_change_is_rejected(self):
        self.mutate(lambda rows: rows[0]["prompt_token_ids"].append(999))
        with self.assertRaisesRegex(ValueError, "pairing"):
            queue.verify_generation(self.root, "unused", 2, False)

    def test_seed_change_is_rejected(self):
        self.mutate(lambda rows: rows[0].update(seed=123))
        with self.assertRaisesRegex(ValueError, "seed"):
            queue.verify_generation(self.root, "unused", 2, False)

    def test_smoke_cannot_be_reported_as_full(self):
        with self.assertRaisesRegex(ValueError, "378"):
            queue.verify_generation(self.root, "unused", 2, True)


if __name__ == "__main__":
    unittest.main()
