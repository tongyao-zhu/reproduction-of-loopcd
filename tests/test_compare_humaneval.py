"""Synthetic scientific-integrity gates for paired HumanEval summaries."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("compare_humaneval", Path(__file__).resolve().parents[1] / "scripts/compare_humaneval.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class CompareHumanEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.make_fixtures(4, limit=4)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n")

    def make_fixtures(self, n, limit):
        self.runs = [self.root / arm for arm in compare.ARM_SETTINGS]
        self.scores = [self.root / (arm + "_scores.json") for arm in compare.ARM_SETTINGS]
        h = compare.digest
        patch = {"reason": "find_zero progress", "scope": "bookkeeping only", "upstream_sha256": h("upstream"),
                 "patched_sha256": h("patched"), "diff": "two bookkeeping assignments\n"}
        patch["diff_sha256"] = compare.hashlib.sha256(patch["diff"].encode()).hexdigest()
        canonical = {"status": "PASS", "manifest_sha256": h("runtime"), "passed_tasks": 164,
                     "expected_rows": 164, "completed_rows": 164, "all_base_plus_passed": True,
                     "dataset_sha256": compare.DATA_SHA256, "scorer_source_sha256": h("scorer"),
                     "evidence_sha256": h("canonical evidence"), "evaluator_patch": patch}
        base = {"baseline32": [True, False, False, True], "hidden32": [True, True, True, False],
                "baseline16": [False, False, True, True], "hidden16": [False, True, True, True]}
        extra = {"baseline32": [True, True, True, False], "hidden32": [True, True, False, True],
                 "baseline16": [True] * 4, "hidden16": [True, False, True, True]}
        for directory, score_path, (arm, guidance) in zip(self.runs, self.scores, compare.ARM_SETTINGS.items()):
            directory.mkdir(exist_ok=True)
            ids = [f"HumanEval/{i}" for i in range(n)]
            config = {"guidance": guidance, "benchmark": "HumanEvalPlus-v0.1.10", "data_sha256": compare.DATA_SHA256,
                      "do_sample": False, "limit": limit, "task_ids": ids, "max_new_tokens": 8,
                      "native_context_length": 4096, "seed": 42, "seed_rule": "stable task hash",
                      "instruction": "instruction", "response_prefix": "assistant code block",
                      "prompt_builder": "official builder without extra BOS", "initialization": "native Gaussian",
                      "cache": "fresh full cache", "stops": ["\n```\n"], "eos_token_id": [65505, 65508],
                      "pad_token_id": 65509,
                      "source": {"model": {"repo_id": "tomg-group-umd/huginn-0125", "revision": compare.MODEL_REVISION,
                                           "model_code_sha256": h("model")},
                                 "loaded_model_code_sha256": h("model"), "source_sha256": {"runner.py": h("runner")},
                                 "evalplus_prompt_sha256": h("prompt source"), "evalplus_sanitize_sha256": h("sanitize source"),
                                 "packages": {"torch": "2.9.0", "transformers": "4.54.1"}, "evalplus_version": "0.3.1"}}
            config_hash = h(config)
            generated, evaluated = [], []
            for index, task in enumerate(ids):
                cap = index % 4 == 3
                prompt = "Prompt " + task
                observation = {"mode": guidance["mode"], "total_loops": guidance["total_loops"],
                               "extra_coda_passes": 0, "extra_lm_head_calls": 0,
                               "guidance_applied": guidance["mode"] == "hidden"}
                if guidance["mode"] == "hidden":
                    observation.update(reference_loop=guidance["reference_loop"],
                                       executed_physical_loops=list(range(1, guidance["total_loops"] + 1)),
                                       lm_head_calls=1, coda_layer_calls=2)
                seed = (42 + int(compare.hashlib.sha256(task.encode()).hexdigest()[:8], 16)) % (2**32)
                generated.append({"task_id": task, "config_hash": config_hash, "problem_sha256": h(task),
                                  "prompt": prompt, "prompt_sha256": h(prompt), "prompt_token_ids": [65504, 10 + index],
                                  "seed": seed, "generated_token_ids": [19] * 8 if cap else [19, 65505],
                                  "generated_tokens": 8 if cap else 2, "effective_max_new_tokens": 8,
                                  "stop_reason": "token_cap" if cap else "eos_token", "stop_string": None,
                                  "cap_hit": cap, "elapsed_seconds": 1.0 + index,
                                  "solution": "def example():\n    return 1", "completion": "code", "raw_generation": "code",
                                  "adapter_observation": observation})
                row = {"task_id": task, "sample_id": 0}
                for suite, values in (("base", base), ("plus", extra)):
                    passed = values[arm][index % 4]
                    row[suite] = {"status": "pass" if passed else "fail", "passed": passed,
                                  "details": [True, True] if passed else [False], "tests": 2}
                    if index == 32:
                        row[suite].update(upstream_status="fail", upstream_details=[])
                row["plus_passed"] = row["base"]["passed"] and row["plus"]["passed"]
                evaluated.append(row)
            manifest = {"status": "completed", "config": config, "config_hash": config_hash,
                        "is_full_split": limit is None, "expected_samples": n, "completed_samples": n,
                        "cap_hits": sum(row["cap_hit"] for row in generated)}
            self.write(directory / "manifest.json", manifest)
            self.write_rows(directory, generated)
            score = {"status": "PASS", "exit_code": 0, "evaluation_complete": True,
                     "samples_sha256": compare.sha256(directory / "samples.jsonl"), "dataset_sha256": compare.DATA_SHA256,
                     "manifest_sha256": h("runtime"), "scorer_source_sha256": h("scorer"),
                     "expected_rows": n, "completed_rows": n, "rows": evaluated,
                     "safety": {"passed": True, "checks": {name: True for name in compare.SAFETY_CHECKS}},
                     "evaluator_patch": patch, "canonical_validation": canonical,
                     "scorer": "EvalPlus 0.3.1 with audited private bookkeeping repair", "numpy": "1.26", "psutil": "6.0"}
            self.write(score_path, score)

    @staticmethod
    def write_rows(directory, rows):
        (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def compare(self):
        return compare.compare_humaneval(self.runs, self.scores)

    def mutate_score(self, function, index=1):
        value = compare.read_json(self.scores[index])
        function(value)
        self.write(self.scores[index], value)

    def mutate_generation(self, function, index=1):
        path = self.runs[index] / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        function(rows)
        self.write_rows(self.runs[index], rows)
        self.mutate_score(lambda score: score.update(samples_sha256=compare.sha256(path)), index=index)

    def mutate_manifest(self, function, index=1, resign=False):
        path = self.runs[index] / "manifest.json"
        value = compare.read_json(path)
        function(value)
        if resign:
            value["config_hash"] = compare.digest(value["config"])
            self.mutate_generation(lambda rows: [row.update(config_hash=value["config_hash"]) for row in rows], index=index)
        self.write(path, value)

    def test_known_paired_counts_plus_requires_base_and_extended(self):
        result = self.compare()
        base = result["pairs"]["R32"]["metrics"]["base"]
        self.assertEqual((base["baseline_pass_at_1_percent"], base["hidden_pass_at_1_percent"]), (50, 75))
        self.assertEqual((base["wins"], base["losses"], base["ties"]), (2, 1, 1))
        self.assertEqual(base["delta_percentage_points"], 25)
        plus = result["pairs"]["R32"]["metrics"]["plus"]
        self.assertEqual((plus["baseline_pass_at_1_percent"], plus["hidden_pass_at_1_percent"]), (25, 50))
        self.assertEqual((plus["wins"], plus["losses"]), (1, 0))
        self.assertEqual(result["arms"]["baseline32"]["generation"]["cap_hits"], 1)
        self.assertEqual(result["arms"]["baseline32"]["generation"]["generated_tokens_total"], 14)
        self.assertFalse(result["full_164_verified"])
        self.assertIn("Debug subset", result["interpretation"])

    def test_full_164_distinct_tasks_accepted(self):
        self.make_fixtures(164, limit=None)
        result = self.compare()
        self.assertTrue(result["full_164_verified"])
        self.assertEqual(result["n_tasks"], 164)
        self.assertIsNotNone(result["arms"]["hidden16"]["find_zero_upstream_evidence"])

    def test_reordered_score_rows_are_joined_by_task(self):
        self.mutate_score(lambda score: score["rows"].reverse())
        self.assertEqual(self.compare()["pairs"]["R32"]["metrics"]["base"]["wins"], 2)

    def test_config_hash_recomputed(self):
        self.mutate_manifest(lambda manifest: manifest["config"].update(seed=99))
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self.compare()

    def test_changed_source_rejected_even_with_valid_config_hash(self):
        self.mutate_manifest(lambda m: m["config"]["source"]["source_sha256"].update(**{"runner.py": compare.digest("changed")}), resign=True)
        with self.assertRaisesRegex(ValueError, "protocol/model/source"):
            self.compare()

    def test_running_generation_rejected(self):
        self.mutate_manifest(lambda m: m.update(status="running"))
        with self.assertRaisesRegex(ValueError, "generation status"):
            self.compare()

    def test_subset_cannot_claim_full_164(self):
        self.mutate_manifest(lambda m: m.update(is_full_split=True))
        with self.assertRaisesRegex(ValueError, "Full-split"):
            self.compare()

    def test_missing_generated_task_rejected(self):
        self.mutate_generation(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "complete generated task set"):
            self.compare()

    def test_duplicate_generated_task_rejected(self):
        self.mutate_generation(lambda rows: rows.append(copy.deepcopy(rows[0])))
        with self.assertRaisesRegex(ValueError, "Duplicate or unknown generation"):
            self.compare()

    def test_prompt_token_pairing_rejected(self):
        self.mutate_generation(lambda rows: rows[0]["prompt_token_ids"].append(55))
        with self.assertRaisesRegex(ValueError, "prompt_token_ids"):
            self.compare()

    def test_seed_rule_checked(self):
        self.mutate_generation(lambda rows: rows[0].update(seed=17))
        with self.assertRaisesRegex(ValueError, "initialization seed"):
            self.compare()

    def test_stale_or_swapped_scoring_input_rejected(self):
        self.mutate_score(lambda score: score.update(samples_sha256=compare.digest("stale")))
        with self.assertRaisesRegex(ValueError, "score/input samples"):
            self.compare()

    def test_failed_scoring_process_cannot_yield_accuracy(self):
        self.mutate_score(lambda score: score.update(status="FAIL"))
        with self.assertRaisesRegex(ValueError, "scoring process status"):
            self.compare()

    def test_partial_scoring_cannot_yield_accuracy(self):
        self.mutate_score(lambda score: score.update(evaluation_complete=False))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.compare()

    def test_missing_scored_task_rejected(self):
        self.mutate_score(lambda score: score["rows"].pop())
        with self.assertRaisesRegex(ValueError, "complete scored task set"):
            self.compare()

    def test_duplicate_scored_task_rejected(self):
        self.mutate_score(lambda score: score["rows"].append(copy.deepcopy(score["rows"][0])))
        with self.assertRaisesRegex(ValueError, "Duplicate or unknown scored"):
            self.compare()

    def test_unsafe_scorer_rejected(self):
        self.mutate_score(lambda score: score["safety"]["checks"].update(network_blocked=False))
        with self.assertRaisesRegex(ValueError, "isolation"):
            self.compare()

    def test_unvalidated_canonical_suite_rejected(self):
        self.mutate_score(lambda score: score["canonical_validation"].update(all_base_plus_passed=False))
        with self.assertRaisesRegex(ValueError, "canonical base/plus"):
            self.compare()

    def test_canonical_runtime_must_match_actual_scoring(self):
        self.mutate_score(lambda score: score["canonical_validation"].update(manifest_sha256=compare.digest("other")))
        with self.assertRaisesRegex(ValueError, "canonical/scoring"):
            self.compare()

    def test_scorer_patch_diff_hash_checked(self):
        self.mutate_score(lambda score: score["evaluator_patch"].update(diff="different patch"))
        with self.assertRaisesRegex(ValueError, "patch diff hash"):
            self.compare()

    def test_pass_requires_all_test_details(self):
        self.mutate_score(lambda score: score["rows"][0]["base"].update(details=[True]))
        with self.assertRaisesRegex(ValueError, "complete passing test details"):
            self.compare()

    def test_plus_flag_cannot_ignore_base_failure(self):
        self.mutate_score(lambda score: score["rows"][3].update(plus_passed=True))
        with self.assertRaisesRegex(ValueError, "base AND extended"):
            self.compare()

    def test_infrastructure_error_status_is_not_scored_as_wrong_answer(self):
        self.mutate_score(lambda score: score["rows"][0]["base"].update(status="worker_error", passed=False))
        with self.assertRaisesRegex(ValueError, "infrastructure errors"):
            self.compare()

    def test_guidance_application_evidence_required(self):
        self.mutate_generation(lambda rows: rows[0]["adapter_observation"].update(guidance_applied=False))
        with self.assertRaisesRegex(ValueError, "not applied"):
            self.compare()

    def test_native_model_code_binding_required(self):
        self.mutate_manifest(lambda m: m["config"]["source"].update(loaded_model_code_sha256=compare.digest("wrong code")), resign=True)
        with self.assertRaisesRegex(ValueError, "loaded model code"):
            self.compare()

    def test_find_zero_original_evidence_required(self):
        self.make_fixtures(164, limit=None)
        self.mutate_score(lambda score: score["rows"][32]["base"].pop("upstream_status"))
        with self.assertRaisesRegex(ValueError, "upstream_status"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
