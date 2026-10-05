import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("launch_huginn_mc_suite", Path(__file__).resolve().parents[1] / "scripts/launch_huginn_mc_suite.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FakeQueue(module.HuginnQueue):
    def __init__(self, args, **kwargs):
        self.commands = []
        self.fail_name = None
        self.missing_compare_output = False
        self.command_lock = threading.Lock()
        super().__init__(args, **kwargs)

    def _execute(self, command, log_path, environment, on_started):
        output = Path(command[command.index("--output") + 1])
        with self.command_lock:
            self.commands.append(output.name)
        Path(log_path).write_text("CPU fake child process\n")
        on_started(999999)
        if output.name == self.fail_name:
            return 1
        if "compare_huginn.py" in command[1]:
            if self.missing_compare_output:
                return 0
            smoke = output.name.startswith("smoke_")
            candidate = "hidden32" if "hidden32" in output.name else "hidden16"
            task = output.name.removeprefix("smoke_").removesuffix(f"_{candidate}_comparison.json")
            output.write_text(json.dumps({
                "task": task, "is_full_split": not smoke, "n_documents": 2 if smoke else 1000,
                "validation": {"paired": True, "model_code_data_prompts_equal": True, "sample_aggregate_consistency": True,
                               "native_initialization_stream": {"requests": 8, "trace_sha256": "verified"}},
                "baseline": {"guidance": {"mode": "baseline", "total_loops": 32}},
                "candidate": {"guidance": {"mode": "hidden", "total_loops": 32 if candidate == "hidden32" else 16, "reference_loop": 6, "omega": .5}},
            }))
        else:
            output.mkdir()
        return 0


class HuginnQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(output=self.root / "queue", model=self.root / "model", gpus=["0", "1", "2"],
                                       wait_for_matrix=None, dataset_cache=None, poll_seconds=.001)

    def queue(self, **kwargs):
        return FakeQueue(self.args, root=self.root, gpu_probe=kwargs.get("gpu_probe", lambda gpu: {"ready": True, "memory_mib": 0, "compute_pids": []}))

    def test_all_three_smokes_and_both_comparisons_precede_full_runs(self):
        queue = self.queue()
        self.assertEqual(queue.run(), 0)
        self.assertEqual(queue.state["status"], "completed")
        self.assertEqual(len(queue.state["jobs"]), 24)
        self.assertEqual(len(queue.state["comparisons"]), 16)
        first_full = min(index for index, name in enumerate(queue.commands) if not name.startswith("smoke_"))
        self.assertEqual(first_full, 5)
        for arm in module.ARMS:
            names = [name for name in queue.commands if name.endswith("_" + arm) and not name.startswith("smoke_")]
            self.assertEqual(names, [f"{task}_{arm}" for task in module.TASKS])
        persisted = json.loads((queue.output / "matrix.json").read_text())
        self.assertEqual(persisted["status"], "completed")
        self.assertFalse((queue.output / "matrix.json.tmp").exists())

    def test_smoke_failure_blocks_every_full_run(self):
        queue = self.queue()
        queue.fail_name = "smoke_sciq_hidden32"
        self.assertEqual(queue.run(), 1)
        self.assertEqual(queue.state["smoke_gate"]["status"], "failed")
        self.assertFalse(any(not name.startswith("smoke_") for name in queue.commands))

    def test_comparison_failure_is_fail_closed(self):
        queue = self.queue()
        queue.fail_name = "smoke_sciq_hidden16_comparison.json"
        self.assertEqual(queue.run(), 1)
        self.assertFalse(any(not name.startswith("smoke_") for name in queue.commands))

    def test_success_exit_without_comparison_evidence_is_fail_closed(self):
        queue = self.queue()
        queue.missing_compare_output = True
        self.assertEqual(queue.run(), 1)
        self.assertFalse(any(not name.startswith("smoke_") for name in queue.commands))

    def test_failed_full_arm_does_not_block_unrelated_tasks(self):
        queue = self.queue()
        queue.fail_name = "arc_easy_hidden32"
        self.assertEqual(queue.run(), 1)
        self.assertEqual(queue.state["comparisons"]["arc_easy_hidden16_comparison"]["status"], "blocked_by_failed_arm")
        self.assertEqual(queue.state["comparisons"]["mmlu_hidden16_comparison"]["status"], "completed")
        self.assertEqual(queue.state["jobs"]["mmlu_hidden32"]["status"], "completed")

    def test_busy_gpu_wait_is_recorded_before_launch(self):
        calls = {gpu: 0 for gpu in self.args.gpus}
        def probe(gpu):
            calls[gpu] += 1
            return {"ready": calls[gpu] > 1, "memory_mib": 2048 if calls[gpu] == 1 else 0, "compute_pids": []}
        queue = self.queue(gpu_probe=probe)
        self.assertEqual(queue.run(), 0)
        self.assertTrue(all(worker["wait_checks"] == 9 for worker in queue.state["workers"].values()))

    def test_existing_output_directory_is_not_overwritten(self):
        self.args.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.queue()

    def test_predecessor_uses_gpu_mapping_and_all_expected_tasks(self):
        path = self.root / "ouro.json"
        state = {"gpus": {"baseline": "2", "fixed": "1", "adaptive": "0"}, "tasks": sorted(module.OURO_TASKS), "jobs": {}}
        for task in module.OURO_TASKS:
            state["jobs"][f"{task}_adaptive"] = {"status": "completed"}
        path.write_text(json.dumps(state))
        self.assertTrue(module.dependency_status(path, "0")["ready"])
        self.assertFalse(module.dependency_status(path, "2")["ready"])
        state["jobs"]["mmlu_adaptive"]["status"] = "running"
        path.write_text(json.dumps(state))
        self.assertFalse(module.dependency_status(path, "0")["ready"])
        state["jobs"]["mmlu_adaptive"]["status"] = "failed"
        path.write_text(json.dumps(state))
        self.assertTrue(module.dependency_status(path, "0")["ready"])

    def test_gpu_requires_both_low_memory_and_no_process(self):
        with patch.object(module.subprocess, "check_output", side_effect=["GPU-a, 0\n", "GPU-a, 123\n"]):
            self.assertFalse(module.gpu_status("0")["ready"])
        with patch.object(module.subprocess, "check_output", side_effect=["GPU-a, 1000\n", ""]):
            self.assertFalse(module.gpu_status("0")["ready"])
        with patch.object(module.subprocess, "check_output", side_effect=["GPU-a, 999\n", "GPU-b, 123\n"]):
            self.assertTrue(module.gpu_status("0")["ready"])


if __name__ == "__main__":
    unittest.main()
