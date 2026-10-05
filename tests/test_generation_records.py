"""Resume and stopping semantics must not change the paired benchmark."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("generate_humaneval", Path(__file__).resolve().parents[1] / "scripts/generate_humaneval.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class GenerationRecordsTests(unittest.TestCase):
    def test_seed_is_keyed_by_task_not_iteration(self):
        forward = {task: generation.task_seed(task, 42) for task in ["HumanEval/1", "HumanEval/2"]}
        backward = {task: generation.task_seed(task, 42) for task in ["HumanEval/2", "HumanEval/1"]}
        self.assertEqual(forward, backward)
        self.assertNotEqual(forward["HumanEval/1"], forward["HumanEval/2"])

    def test_first_stop_wins_and_function_definitions_are_preserved(self):
        code = "def helper():\n    return 1\n\ndef target():\n    return helper()\n```\nextra\nprint(2)"
        trimmed, stop = generation.trim_stops(code)
        self.assertEqual(stop, "\n```\n")
        self.assertIn("def target()", trimmed)
        self.assertNotIn("extra", trimmed)

    def test_resume_rejects_duplicates_and_config_changes(self):
        problem = {"task_id": "HumanEval/0", "prompt": "def f():"}
        row = {"task_id": problem["task_id"], "config_hash": "config", "problem_sha256": generation.digest(problem)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(generation.read_completed(path, "config", {problem["task_id"]: problem})), 1)
            with self.assertRaises(ValueError):
                generation.read_completed(path, "changed", {problem["task_id"]: problem})
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                generation.read_completed(path, "config", {problem["task_id"]: problem})

    def test_resume_rejects_changed_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(json.dumps({"task_id": "HumanEval/0", "config_hash": "c", "problem_sha256": "wrong"}) + "\n")
            with self.assertRaises(ValueError):
                generation.read_completed(path, "c", {"HumanEval/0": {"prompt": "changed"}})

    def test_recovery_preserves_only_partial_tail_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            valid = b'{"task_id":"HumanEval/0"}\n'
            path.write_bytes(valid + b'{"broken":"\xe4')
            recovery = generation.repair_partial_tail(path)
            self.assertEqual(path.read_bytes(), valid)
            self.assertEqual(Path(recovery["partial_tail_backup"]).read_bytes(), b'{"broken":"\xe4')
            path.write_bytes(b'{"valid":1}')
            generation.repair_partial_tail(path)
            self.assertEqual(path.read_bytes(), b'{"valid":1}\n')
            path.write_bytes(b'{broken}\n')
            self.assertIsNone(generation.repair_partial_tail(path))
            self.assertEqual(path.read_bytes(), b'{broken}\n')


if __name__ == "__main__":
    unittest.main()
