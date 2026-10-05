"""Synthetic complete split tests: these fixtures contain no executable answers."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_mbpp as c


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompareMbppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_path = self.root / "data.jsonl"
        self.ids = [f"Mbpp/{i}" for i in range(377)] + ["Mbpp/793"]
        problems = [{"task_id": task, "prompt": f'"""Public problem {task}."""',
                     "entry_point": "synthetic", "canonical_solution": "NOT EXECUTABLE CANONICAL",
                     "base_input": [[0], [1]], "plus_input": {} if task == "Mbpp/793" else [[2], [3], [4]]}
                    for task in self.ids]
        write_rows(self.data_path, problems)
        self.source_map = {f"scripts/fixture{i}.py": "a" * 64 for i in range(32)}
        for key, value in {"DATA_SHA256": file_hash(self.data_path),
                           "EXPECTED_TEST_TOTALS": {"base": 756, "plus": 1131},
                           "SOURCE_MAP_SHA256": c.digest(self.source_map)}.items():
            handle = patch.object(c, key, value)
            handle.start()
            self.addCleanup(handle.stop)
        self.data = c.load_data(self.data_path)
        source = {"git_commit": c.GENERATION_COMMIT, "source_sha256": self.source_map,
                  "model": {"repo_id": "tomg-group-umd/huginn-0125", "revision": c.MODEL_REVISION,
                            "model_code_sha256": c.MODEL_CODE_SHA256,
                            "files": {name: {"sha256_original": value} for name, value in c.MODEL_FILE_PINS.items()}},
                  "loaded_model_code_sha256": c.MODEL_CODE_SHA256, "evalplus_version": "0.3.1",
                  "evalplus_prompt_sha256": "455b72b9fdc7aa7daa7afe54a31118ee3eec3d17c463cc911642777d6ae50a7a",
                  "evalplus_sanitize_sha256": "12fec16b93bfc4d9d9103f227b864dee3c510fe946539b567e6a7805de7d09b5",
                  "precision": "BF16 model; FP32 guidance/log_softmax", "attention": "sdpa", "arxiv": "2610.02185v1",
                  "python": "3.10.18", "cuda": "12.8", "gpu": "NVIDIA A100-SXM4-40GB",
                  "packages": {"torch": "2.9.0", "transformers": "4.54.1", "accelerate": "1.12.0", "datasets": "4.0.0", "lm_eval": "0.4.9.1"}}
        self.runs, self.scores = [], []
        for arm, guidance in c.ARM_SETTINGS.items():
            path = self.root / arm
            path.mkdir()
            self.runs.append(path)
            config = {**copy.deepcopy(c.PROTOCOL), "data_sha256": c.DATA_SHA256, "task_ids": self.ids,
                      "source": source, "guidance": guidance}
            rows = []
            for problem in problems:
                task = problem["task_id"]
                prompt = c.expected_prompt(problem)
                observation = {"mode": guidance["mode"], "total_loops": guidance["total_loops"],
                               "guidance_applied": guidance["mode"] == "hidden", "extra_coda_passes": 0,
                               "extra_lm_head_calls": 0, "initialization": "native_random_or_explicit_input_states"}
                if guidance["mode"] == "hidden":
                    observation.update(reference_loop=guidance["reference_loop"],
                        executed_physical_loops=list(range(1, guidance["total_loops"] + 1)),
                        coda_layer_calls=2, lm_head_calls=1,
                        combination_location="raw_recurrent_output_before_native_pre_coda_ln_f",
                        arithmetic="FP32 blend cast back to native hidden dtype before native output interface",
                        cache_semantics="native recurrent history; guided coda history; one update per layer and token")
                rows.append({"task_id": task, "config_hash": c.digest(config), "problem_sha256": c.digest(problem),
                             "prompt": prompt, "prompt_sha256": c.digest(prompt), "prompt_token_ids": [65504, 123],
                             "seed": (42 + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)) % 2**32,
                             "generated_token_ids": [12, 65505], "generated_tokens": 2,
                             "effective_max_new_tokens": 2048, "cap_hit": False, "stop_reason": "eos_token", "stop_string": None,
                             "elapsed_seconds": 0.25, "raw_generation": "RAW", "completion": "COMPLETION",
                             "solution": "NOT EXECUTABLE MODEL ANSWER", "adapter_observation": observation})
            write_rows(path / "samples.jsonl", rows)
            write(path / "manifest.json", {"status": "completed", "config": config, "config_hash": c.digest(config),
                  "is_full_split": True, "expected_samples": 378, "completed_samples": 378, "cap_hits": 0})
        evaluator = c.EVALUATOR_PINS["opt/site/evalplus/eval/__init__.py"]
        self.manifest = {"dataset": "MbppPlus-v0.2.0", "dataset_sha256": c.DATA_SHA256, "expected_tasks": 378,
                         "rootfs": "/synthetic/.sandbox/mbpp-v2/rootfs", "prepare_source_sha256": c.PREPARE_SHA256,
                         "scorer_source_sha256": c.RUNNER_SHA256, "identity": {"uid": 60002, "gid": 60002,
                         "selected_at": "2026-10-04T00:00:00Z", "selection": "fixture reserved"},
                         "immutable_sha256": {**c.EVALUATOR_PINS, "runner.py": c.RUNNER_SHA256,
                            "data/MbppPlus-v0.2.0.jsonl": c.DATA_SHA256, "usr/bin/python3": "a" * 64},
                         "evaluator_patch": {"applied": False, "reason": "MBPP uses unmodified EvalPlus 0.3.1",
                                             "upstream_sha256": evaluator, "runtime_sha256": evaluator},
                         "packages": {"numpy": "2.2.6", "psutil": "7.0.0", "evalplus": "0.3.1", "appdirs": "1.4.4", "tempdir": "0.7.1", "wget": "3.2"}}
        self.manifest_path = self.root / "sandbox_manifest.json"
        write(self.manifest_path, self.manifest)
        rows = [{"task_id": task, "sample_id": 0, "plus_passed": True,
                 **{suite: {"status": "pass", "passed": True, "tests": self.data["counts"][task][suite],
                            "details": [True] * self.data["counts"][task][suite]} for suite in ("base", "plus")}}
                for task in self.ids]
        self.canonical = {"status": "PASS", "exit_code": 0, "evaluation_complete": True,
            "manifest_sha256": file_hash(self.manifest_path), "dataset_sha256": c.DATA_SHA256,
            "scorer_source_sha256": c.RUNNER_SHA256, "evaluator_patch": self.manifest["evaluator_patch"],
            "identity": self.manifest["identity"], "sandbox": "/synthetic/.sandbox/mbpp-v2", "scorer": c.SCORER,
            "numpy": "2.2.6", "psutil": "7.0.0", "samples_sha256": None, "expected_rows": 378, "completed_rows": 378,
            "sample_validation": {"samples": 378, "expected_full_tasks": 378, "is_full_task_set": True},
            "safety": {"passed": True, "checks": {name: True for name in c.SAFETY_CHECKS}}, "rows": rows}
        self.canonical_path = self.root / "canonical378.json"
        write(self.canonical_path, self.canonical)
        cert = {"status": "PASS", "passed_tasks": 378, "expected_rows": 378, "completed_rows": 378,
                "all_base_plus_passed": True, "manifest_sha256": file_hash(self.manifest_path), "dataset_sha256": c.DATA_SHA256,
                "scorer_source_sha256": c.RUNNER_SHA256, "evaluator_patch": self.manifest["evaluator_patch"],
                "evidence_sha256": file_hash(self.canonical_path), "result_sha256": file_hash(self.canonical_path),
                "result_path": "/synthetic/canonical378.json", "task_ids": self.ids}
        for run in self.runs:
            path = self.root / (run.name + "-score.json")
            score = {**copy.deepcopy(self.canonical), "samples_sha256": file_hash(run / "samples.jsonl"), "canonical_validation": cert}
            write(path, score)
            self.scores.append(path)

    def compare(self):
        return c.compare_mbpp(self.runs, self.scores, self.data_path, self.canonical_path, self.manifest_path)

    def mutate_score(self, index, function):
        path = self.scores[index]
        record = json.loads(path.read_text())
        function(record)
        write(path, record)

    def mutate_generation(self, index, function):
        path = self.runs[index]
        manifest = json.loads((path / "manifest.json").read_text())
        rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
        function(manifest, rows)
        manifest["config_hash"] = c.digest(manifest["config"])
        for row in rows:
            row["config_hash"] = manifest["config_hash"]
        write(path / "manifest.json", manifest)
        write_rows(path / "samples.jsonl", rows)
        self.mutate_score(index, lambda score: score.update(samples_sha256=file_hash(path / "samples.jsonl")))

    def test_full378_and_legal_empty_plus(self):
        self.assertEqual(self.data["problems"]["Mbpp/793"]["plus_input"], {})
        result = self.compare()
        self.assertTrue(result["full_378_verified"])
        self.assertEqual(result["pairs"]["R32"]["metrics"]["plus"]["baseline_passed"], 378)
        self.assertEqual(result["scorer"]["test_totals"], {"base": 756, "plus": 1131})

    def test_empty_dict_exception_is_only_pinned_task_plus(self):
        rows = [json.loads(line) for line in self.data_path.read_text().splitlines()]
        rows[0]["plus_input"] = {}
        write_rows(self.data_path, rows)
        with patch.object(c, "DATA_SHA256", file_hash(self.data_path)):
            with self.assertRaisesRegex(ValueError, "suite inputs"):
                c.load_data(self.data_path)

    def test_wrong_answers_may_have_partial_details_and_plus_is_joint(self):
        def wrong(score):
            score["rows"][0]["base"].update(status="fail", passed=False, details=[True, False])
            score["rows"][0]["plus_passed"] = False
            score["rows"][1]["plus"].update(status="timeout", passed=False, details=[])
            score["rows"][1]["plus_passed"] = False
        self.mutate_score(0, wrong)
        result = self.compare()["pairs"]["R32"]["metrics"]
        self.assertEqual(result["base"]["wins"], 1)
        self.assertEqual(result["plus"]["wins"], 2)
        self.assertEqual(result["plus"]["baseline_passed"], 376)

    def test_failure_cannot_claim_joint_pass(self):
        self.mutate_score(0, lambda r: r["rows"][0]["base"].update(status="fail", passed=False, details=[]))
        with self.assertRaisesRegex(ValueError, "base AND"):
            self.compare()

    def test_pass_truncation_is_rejected(self):
        self.mutate_score(0, lambda r: r["rows"][0]["plus"].update(details=[]))
        with self.assertRaisesRegex(ValueError, "complete per-test"):
            self.compare()

    def test_zero_suite_requires_present_list_and_integer_count(self):
        for value in (None, {}, False):
            with self.subTest(value=value):
                record = copy.deepcopy(self.canonical)
                record["rows"][-1]["plus"]["details"] = value
                with self.assertRaises(ValueError):
                    c.validate_score_rows(record, self.data, canonical=True)
        record = copy.deepcopy(self.canonical)
        del record["rows"][-1]["plus"]["details"]
        with self.assertRaises(ValueError):
            c.validate_score_rows(record, self.data, canonical=True)
        record["rows"][-1]["plus"].update(details=[], tests=False)
        with self.assertRaises(ValueError):
            c.validate_score_rows(record, self.data, canonical=True)

    def test_canonical_evidence_is_rechecked_even_if_resigned(self):
        record = copy.deepcopy(self.canonical)
        record["rows"][0]["base"]["details"] = []
        write(self.canonical_path, record)
        for index in range(4):
            self.mutate_score(index, lambda r: r["canonical_validation"].update(evidence_sha256=file_hash(self.canonical_path), result_sha256=file_hash(self.canonical_path)))
        with self.assertRaisesRegex(ValueError, "complete per-test"):
            self.compare()

    def test_duplicate_score_ids_rejected(self):
        self.mutate_score(0, lambda r: r["rows"].__setitem__(1, r["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compare()

    def test_subset_and_duplicate_generation_rejected(self):
        self.mutate_generation(0, lambda m, r: (r.pop(), m.update(completed_samples=377)))
        with self.assertRaises(ValueError):
            self.compare()

    def test_complete_source_map_pin_rejects_resigned_change(self):
        self.mutate_generation(0, lambda m, r: m["config"]["source"]["source_sha256"].update({"scripts/fixture0.py": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "frozen generation source map"):
            self.compare()

    def test_seed_and_initialization_protocol_are_pinned(self):
        self.mutate_generation(0, lambda m, r: m["config"].update(initialization="zero", seed=43))
        with self.assertRaisesRegex(ValueError, "generation protocol"):
            self.compare()

    def test_prompt_leak_rejected_even_with_recomputed_hash(self):
        def leak(manifest, rows):
            rows[0]["prompt"] += "NOT EXECUTABLE CANONICAL"
            rows[0]["prompt_sha256"] = c.digest(rows[0]["prompt"])
        self.mutate_generation(0, leak)
        with self.assertRaisesRegex(ValueError, "public-only"):
            self.compare()

    def test_token_pairing_is_required(self):
        self.mutate_generation(1, lambda m, r: r[0].update(prompt_token_ids=[65504, 124]))
        with self.assertRaisesRegex(ValueError, "paired prompt_token_ids"):
            self.compare()

    def test_cap_accounting(self):
        def cap(manifest, rows):
            rows[0].update(generated_token_ids=[12] * 2048, generated_tokens=2048,
                           cap_hit=True, stop_reason="token_cap", stop_string=None)
            manifest["cap_hits"] = 1
        self.mutate_generation(1, cap)
        self.assertEqual(self.compare()["arms"]["hidden32"]["generation"]["cap_hits"], 1)

    def test_score_input_hash_is_required(self):
        self.mutate_score(0, lambda r: r.update(samples_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "sample bytes"):
            self.compare()

    def test_all_safety_checks_and_v2_runtime_are_required(self):
        self.mutate_score(0, lambda r: r["safety"]["checks"].pop("exec_blocked"))
        with self.assertRaisesRegex(ValueError, "11 isolation"):
            self.compare()
        manifest = copy.deepcopy(self.manifest)
        manifest["rootfs"] = "/synthetic/.sandbox/mbpp/rootfs"
        with self.assertRaisesRegex(ValueError, "mbpp-v2"):
            c.validate_runtime(manifest, self.data)

    def test_wrong_runner_and_certificate_hash_rejected(self):
        self.mutate_score(0, lambda r: r["canonical_validation"].update(evidence_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "canonical certificate"):
            self.compare()
        manifest = copy.deepcopy(self.manifest)
        manifest["scorer_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "scorer_source"):
            c.validate_runtime(manifest, self.data)


if __name__ == "__main__":
    unittest.main()
