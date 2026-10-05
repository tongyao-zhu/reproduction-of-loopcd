import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("launch_huginn_r16_suite", Path(__file__).resolve().parents[1] / "scripts/launch_huginn_r16_suite.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FakeQueue(module.R16Queue):
    def __init__(self, *args, **kwargs):
        self.commands = []
        self.command_lock = threading.Lock()
        self.fail_name = None
        self.missing_comparison = False
        self.bad_full_count = False
        super().__init__(*args, **kwargs)

    def _execute(self, command, log, environment, on_started):
        output = Path(command[command.index("--output") + 1])
        with self.command_lock:
            self.commands.append((output.name, command, environment.get("CUDA_VISIBLE_DEVICES")))
        Path(log).write_text("CPU fixture only\n")
        on_started(999999)
        if output.name == self.fail_name:
            return 1
        if any(Path(arg).name == "compare_huginn.py" for arg in command):
            if self.missing_comparison:
                return 0
            smoke = output.name.startswith("smoke_")
            candidate = "half16" if "half16" in output.name else "hidden16"
            task = output.name.removeprefix("smoke_").removesuffix(f"_{candidate}_comparison.json")
            output.write_text(json.dumps({
                "task": task, "n_documents": 2 if smoke else 1000, "is_full_split": not smoke,
                "full_split_count_verified": not smoke and not self.bad_full_count,
                "validation": {"paired": True, "model_code_data_prompts_equal": True, "sample_aggregate_consistency": True,
                               "native_initialization_stream": {"requests": 8, "trace_sha256": "fixture-native-stream"}},
                "baseline": {"path": command[command.index("--baseline") + 1],
                             "guidance": {"mode": "baseline", "total_loops": 16, "reference_loop": 7, "omega": .5}},
                "candidate": {"path": command[command.index("--candidate") + 1],
                              "guidance": {"mode": "hidden", "total_loops": 16, "reference_loop": 6 if candidate == "half16" else 7, "omega": .5}},
            }))
        else:
            output.mkdir()
            (output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        return 0


class R16QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "frozen"
        self.source.mkdir()
        for name in ("scripts/evaluate_huginn.py", "scripts/compare_huginn.py", "configs/huginn_mc.json"):
            p = self.source / name
            p.parent.mkdir(exist_ok=True)
            p.write_text("{}\n")
        self.pins = {str(p.relative_to(self.source)): module.sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.old = self.root / "old"
        for smoke, task in [(True, "sciq")] + [(False, t) for t in module.TASKS]:
            directory = self.old / module.R16Queue.name(task, "hidden16", smoke)
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps({
                "status": "completed", "task": task, "limit": 2 if smoke else None, "is_full_split": not smoke,
                "protocol": "half-depth", "seed": 42, "provenance": {"source_sha256": self.pins},
                "guidance": {"mode": "hidden", "total_loops": 16, "reference_loop": 6, "omega": .5},
            }))
        self.args = argparse.Namespace(output=self.root / "new", source_root=self.source, source_commit="fixture-commit",
                                       half_depth_suite=self.old, model=self.root / "model", gpus=["0", "1"],
                                       dataset_cache=None, poll_seconds=.001, seed=42, smoke_only=False)
        self.source_mock = patch.object(module, "verify_frozen_source", return_value=self.pins)
        self.source_mock.start()
        self.addCleanup(self.source_mock.stop)

    def queue(self, **kwargs):
        return FakeQueue(self.args, gpu_probe=kwargs.get("gpu_probe", lambda _: {"ready": True}))

    def test_two_smokes_and_both_strict_pairs_before_full_suite(self):
        before = module.inventory(self.old)
        q = self.queue()
        self.assertEqual(q.run(), 0)
        self.assertTrue(q.state["full_suite_verified"])
        self.assertEqual(len(q.state["jobs"]), 16)
        self.assertEqual(len(q.state["comparisons"]), 16)
        names = [x[0] for x in q.commands]
        self.assertEqual(min(i for i, n in enumerate(names) if not n.startswith("smoke_")), 4)
        gpu_commands = [x for x in q.commands if x[2]]
        self.assertEqual({x[2] for x in gpu_commands}, {"0", "1"})
        for _, command, _ in gpu_commands:
            self.assertEqual(command[command.index("--loops") + 1], "16")
            self.assertEqual(command[command.index("--reference-loop") + 1], "7")
            self.assertEqual(command[command.index("--protocol") + 1], "standard")
            self.assertEqual(command[2], str(self.source.resolve() / "scripts/evaluate_huginn.py"))
        self.assertEqual(module.inventory(self.old), before)
        self.assertEqual(json.loads((q.output / "matrix.json").read_text())["status"], "completed")

    def test_failed_smoke_blocks_all_full_evaluations(self):
        q = self.queue()
        q.fail_name = "smoke_sciq_hidden16"
        self.assertEqual(q.run(), 1)
        self.assertEqual(q.state["smoke_gate"]["status"], "failed")
        self.assertTrue(all(n.startswith("smoke_") for n, _, _ in q.commands))

    def test_missing_compare_output_does_not_release_gate(self):
        q = self.queue()
        q.missing_comparison = True
        self.assertEqual(q.run(), 1)
        self.assertTrue(all(n.startswith("smoke_") for n, _, _ in q.commands))

    def test_changed_reused_evidence_fails_gate(self):
        q = self.queue()
        (q.old_directory("sciq", True) / "new.json").write_text("{}")
        self.assertEqual(q.run(), 1)
        self.assertIn("changed", q.state["comparisons"]["smoke_sciq_half16_comparison"]["error"])

    def test_full_failure_stops_that_worker(self):
        q = self.queue()
        q.fail_name = "piqa_hidden16"
        self.assertEqual(q.run(), 1)
        names = [n for n, _, _ in q.commands]
        self.assertNotIn("arc_challenge_hidden16", names)
        self.assertIn("mmlu_baseline16", names)
        self.assertEqual(q.state["jobs"]["mmlu_hidden16"]["status"], "blocked_by_failed_worker")
        self.assertEqual(q.state["comparisons"]["mmlu_half16_comparison"]["status"], "completed")

    def test_full_comparison_requires_verified_split_count(self):
        q = self.queue()
        q.bad_full_count = True
        self.assertEqual(q.run(), 1)
        self.assertFalse(q.state["full_suite_verified"])

    def test_source_change_after_startup_stops_before_any_execution(self):
        q = self.queue()
        (self.source / "scripts/evaluate_huginn.py").write_text("changed")
        self.assertEqual(q.run(), 1)
        self.assertEqual(q.commands, [])

    def test_smoke_only_is_not_marked_full_suite(self):
        self.args.smoke_only = True
        q = self.queue()
        self.assertEqual(q.run(), 0)
        self.assertFalse(q.state["full_suite_verified"])
        self.assertEqual(len(q.commands), 4)

    def test_fresh_output_and_paired_old_source_are_required(self):
        self.args.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.queue()
        self.args.output.rmdir()
        p = self.old / "sciq_hidden16/manifest.json"
        m = json.loads(p.read_text())
        m["provenance"]["source_sha256"] = {"wrong": "hash"}
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "source/seed"):
            self.queue()

    def test_busy_gpu_is_rechecked(self):
        seen = {g: 0 for g in self.args.gpus}
        def probe(g):
            seen[g] += 1
            return {"ready": seen[g] > 1}
        q = self.queue(gpu_probe=probe)
        self.assertEqual(q.run(), 0)
        self.assertEqual(seen, {"0": 9, "1": 9})


class FrozenSourceTests(unittest.TestCase):
    def test_git_content_and_extra_source_files_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("scripts/evaluate_huginn.py", "scripts/compare_huginn.py", "configs/huginn_mc.json"):
                p = root / name
                p.parent.mkdir(exist_ok=True)
                p.write_text("{}\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(len(module.verify_frozen_source(root, commit)), 3)
            (root / "scripts/extra.py").write_text("extra")
            with self.assertRaisesRegex(ValueError, "file set"):
                module.verify_frozen_source(root, commit)
            (root / "scripts/extra.py").unlink()
            (root / "scripts/evaluate_huginn.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Git content"):
                module.verify_frozen_source(root, commit)


if __name__ == "__main__":
    unittest.main()
