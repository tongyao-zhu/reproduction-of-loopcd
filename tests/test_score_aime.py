"""AIME scoring tests use fixed strings and synthetic records, never code."""
import copy
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_aime as s


class ExtractorTests(unittest.TestCase):
    def test_fixed_boundaries(self):
        self.assertEqual(len(s.extractor_boundaries()), 25)

    def test_last_malformed_command_never_falls_back(self):
        for final in (r"\boxed", r"\boxed{", r"\boxed{1000}", r"\boxed{\text{25}}"):
            with self.subTest(final=final):
                self.assertFalse(s.extract_answer(r"\boxed{25} " + final)["parsed"])

    def test_box_matching_and_outer_whitespace(self):
        self.assertEqual(s.extract_answer("x \\boxed \n{ 025\t} ignored 9")["value"], 25)
        self.assertEqual(s.extract_answer(r"\boxed{{25}}")["error"], "invalid_ascii_integer")
        self.assertEqual(s.extract_answer(r"\boxed{\text{25}")["error"], "unclosed_brace")

    def test_no_evaluation_or_number_fallback(self):
        for text in ("25", r"\boxed{2**5}", r"\boxed{__import__('os').system('x')}", r"\boxed{25%1000}"):
            self.assertFalse(s.extract_answer(text)["parsed"])

    def test_ascii_digits_only(self):
        for digits in ("１２", "١٢", "²", "1_0", "1 0", "1e2", "-0", "+0", "0000"):
            self.assertFalse(s.extract_answer("\\boxed{" + digits + "}")["parsed"])

    def test_source_degree_exception_not_applied_to_prediction(self):
        self.assertFalse(s.extract_answer(r"\boxed{336^\circ}")["parsed"])
        self.assertEqual(s.extract_answer(r"\boxed{336}")["value"], 336)

    def test_wrong_type_is_infrastructure_error(self):
        with self.assertRaises(ValueError):
            s.extract_answer(None)


class EstimatorTests(unittest.TestCase):
    def test_estimator_boundaries(self):
        self.assertEqual(s.pass_at_k(0, 10), 0)
        self.assertEqual(s.pass_at_k(1, 1), 1 / 16)
        self.assertEqual(s.pass_at_k(1, 10), 10 / 16)
        for count in range(7, 17):
            self.assertEqual(s.pass_at_k(count, 10), 1)

    def test_estimator_is_not_first_ten_success(self):
        self.assertEqual(s.pass_at_k(1, 10), .625)
        self.assertNotEqual(s.pass_at_k(1, 10), 1)

    def test_bad_estimator_counts_rejected(self):
        for count in (-1, 17, True, 1.0):
            with self.assertRaises(ValueError):
                s.pass_at_k(count, 1)
        with self.assertRaises(ValueError):
            s.pass_at_k(1, 17)

    def fixture(self, correct_samples):
        answers, rows = {}, []
        for task in s.task_ids(2024):
            answers[task] = {"task_id": task, "gold_int": 25, "gold_raw": "025", "question_sha256": "a" * 64}
            for sample in range(16):
                rows.append({"task_id": task, "sample_id": sample, "seed": s.sample_seed(task, sample),
                             "completion": r"\boxed{025}" if sample in correct_samples else "No box",
                             "cap_hit": sample == 15, "generated_tokens": 8192 if sample == 15 else 10,
                             "stop_reason": "max_new_tokens" if sample == 15 else "eos", "elapsed_seconds": 1,
                             "peak_allocated_bytes": 10, "peak_reserved_bytes": 20})
        return rows, {"answers": answers}

    def test_complete_equal_weight_problem_statistics(self):
        rows, data = self.fixture({15})
        result = s.summarize_arm(rows, data, 2024)
        self.assertEqual(result["correct_samples"], 30)
        self.assertEqual(result["pass_at_1_percent"], 6.25)
        self.assertEqual(result["pass_at_10_percent"], 62.5)
        self.assertEqual(result["cap_correct"], 30)
        self.assertEqual(result["parse_failures"], {"no_boxed": 450})

    def test_missing_duplicate_or_reordered_samples_fail(self):
        rows, data = self.fixture({15})
        malformed = [rows[:-1], rows[:1] + rows[:-1], [rows[1], rows[0]] + rows[2:]]
        for records in malformed:
            with self.assertRaises(ValueError):
                s.summarize_arm(records, data, 2024)

    def test_paired_problem_bootstrap_and_sample_wins(self):
        rows, data = self.fixture(set())
        baseline = s.summarize_arm(rows, data, 2024)
        rows, data = self.fixture({15})
        candidate = s.summarize_arm(rows, data, 2024)
        result = s.compare_pair(baseline, candidate)
        self.assertEqual(result["sample_wins"], 30)
        self.assertEqual(result["sample_losses"], 0)
        self.assertEqual(result["sample_ties"], 450)
        self.assertEqual(result["metrics"]["pass_at_10"]["delta_percentage_points"], 62.5)
        self.assertEqual(result["metrics"]["pass_at_10"]["paired_problem_bootstrap_95_percentile"], [62.5, 62.5])


class DataGateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.protocol = self.root / "data/aime/protocol-v1.json"
        self.data_root = self.root / "data/aime/prepared-v1"
        if not self.protocol.is_file() or not (self.data_root / "manifest.json").is_file():
            self.skipTest("Private pinned AIME artifacts are not present; extractors/synthetic estimators still tested")

    def test_actual_all60_gold_and_handwritten_boundaries(self):
        data = s.load_scoring_data(self.protocol, self.data_root)
        gate = s.canonical_gate(data)
        self.assertEqual(gate["canonical_tasks"], 60)
        self.assertEqual(gate["boundary_cases_passed"], 25)
        self.assertTrue(gate["all_correct"])
        self.assertFalse(gate["model_answers_executed"])

    def test_data_tamper_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for name in (*s.DATA_PINS, "manifest.json"):
                (destination / name).write_bytes((self.data_root / name).read_bytes())
            path = destination / "aime2024.answers.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"gold_int":204', b'"gold_int":205', 1))
            with self.assertRaisesRegex(ValueError, "pinned aime2024.answers"):
                s.load_scoring_data(self.protocol, destination)


class FullGenerationTests(DataGateTests):
    def setUp(self):
        super().setUp()
        self.data = s.load_scoring_data(self.protocol, self.data_root)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.runs = []
        model = "ByteDance/Ouro-1.4B-Thinking"
        identity = {"repo_id": model, "revision": self.data["protocol"]["models"][model]["revision"],
                    "model_provenance_sha256": "a" * 64, "model_code_sha256": "b" * 64,
                    "config_sha256": "c" * 64, "tokenizer_files_sha256": {name: "d" * 64 for name in
                       ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json")}}
        source = {"git_commit": "f" * 40, "loaded_model_code_sha256": "b" * 64, "attention": "sdpa", "python": "3.10.18",
                  "cuda": "12.8", "gpu": "Synthetic CPU fixture", "precision": "BF16 model; FP32 guidance before native sampling",
                  "packages": {"torch": "2.9.0", "transformers": "4.54.1", "accelerate": "1.12.0"},
                  "model": {key: identity[key] for key in ("repo_id", "revision", "model_code_sha256")},
                  "source_sha256": {name: "e" * 64 for name in ("scripts/generate_aime.py", "src/loopcd_repro/aime_protocol.py",
                       "src/loopcd_repro/ouro.py", "src/loopcd_repro/guidance.py", "src/loopcd_repro/runtime.py")}}
        for arm in s.ARMS:
            folder = self.output / arm
            folder.mkdir()
            self.runs.append(folder)
            config = {"protocol_id": self.data["protocol"]["protocol_id"], "protocol_sha256": s.PROTOCOL_SHA256,
                      "dataset": {"year": 2024, "questions_sha256": s.DATA_PINS["aime2024.questions.jsonl"],
                                  "manifest_sha256": s.MANIFEST_SHA256, "task_ids": s.task_ids(2024)},
                      "samples_per_problem": 16, "generation_config": {**self.data["protocol"]["sampling"], **s.GENERATION_DEFAULTS},
                      "prompt_policy": self.data["protocol"]["prompt"], "execution_policy": self.data["protocol"]["execution"],
                      "model_identity": identity, "source": source, "is_full_split": True,
                      "debug": None,
                      "guidance": {"mode": arm, "omega": .5, "omega_cap": 1.0, "early_loop": 1, "total_loops": 4}}
            paired = {key: value for key, value in config.items() if key != "guidance"}
            rows = []
            for task in s.task_ids(2024):
                messages, prompt = s.expected_prompt(self.data["questions"][task], self.data["protocol"])
                for sample in range(16):
                    text = "\\boxed{" + str(self.data["answers"][task]["gold_int"]) + "}"
                    observation = {"mode": arm, "guidance_applied": arm != "baseline", "native_exit_at_step": 3,
                                   "extra_lm_head_calls": 0 if arm == "baseline" else 1}
                    if arm != "baseline":
                        observation.update(executed_source_indices=[0, 1, 2, 3], early_loop=1, already_normalized_identity=True)
                    rows.append({"schema_version": 1, "arm": arm, "task_id": task, "sample_id": sample,
                        "seed": s.sample_seed(task, sample), "question_sha256": self.data["questions"][task]["question_sha256"],
                        "config_hash": s.hash_json(config), "paired_config_hash": s.hash_json(paired),
                        "generation_config_sha256": s.hash_json(config["generation_config"]), "effective_max_new_tokens": 8192,
                        "messages": messages, "messages_sha256": s.hash_json(messages), "prompt": prompt, "prompt_sha256": s.text_hash(prompt),
                        "prompt_token_ids": [1, 100, 2, 1, 200, 2, 1, 300], "prompt_token_ids_sha256": s.hash_json([1, 100, 2, 1, 200, 2, 1, 300]),
                        "generated_token_ids": [50, 2], "generated_token_ids_sha256": s.hash_json([50, 2]), "generated_tokens": 2,
                        "raw_generation": text + "<|im_end|>", "raw_generation_sha256": s.text_hash(text + "<|im_end|>"),
                        "completion": text, "completion_sha256": s.text_hash(text), "stop_reason": "eos", "cap_hit": False,
                        "elapsed_seconds": 1.0, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                        "adapter_observation": observation,
                        "execution_observation": {"forward_calls": 2, "loop_calls": 8, "head_calls": 2 if arm == "baseline" else 4,
                            "observed_loop_pattern_valid": True, "expected_head_calls_per_forward": 1 if arm == "baseline" else 2,
                            "cache_type": "UniversalTransformerCache", "cache_slots": 96, "fresh_cache_initial_length": 0,
                            "final_cache_length": 9}})
            manifest = {"schema_version": 1, "kind": "aime_generation", "status": "completed", "is_full_split": True,
                        "expected_samples": 480, "completed_samples": 480, "config": config, "config_hash": s.hash_json(config),
                        "paired_config_hash": s.hash_json(paired), "gpu_smoke_sha256": "f" * 64, "cap_hits": 0}
            self.save(folder, manifest, rows)

    def save(self, folder, manifest, rows):
        (folder / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["samples_sha256"] = s.snapshot(folder / "samples.jsonl")[1]
        (folder / "manifest.json").write_text(json.dumps(manifest) + "\n")

    def mutate(self, arm, function):
        folder = self.runs[arm]
        manifest = json.loads((folder / "manifest.json").read_text())
        rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
        function(manifest, rows)
        manifest["config_hash"] = s.hash_json(manifest["config"])
        manifest["paired_config_hash"] = s.hash_json({key: value for key, value in manifest["config"].items() if key != "guidance"})
        for row in rows:
            row["config_hash"] = manifest["config_hash"]
            row["paired_config_hash"] = manifest["paired_config_hash"]
        self.save(folder, manifest, rows)

    def score(self):
        return s.score_aime(self.runs, self.protocol, self.data_root)

    def test_complete_three_arm_end_to_end(self):
        result = self.score()
        self.assertTrue(result["summary"]["full_480_verified"])
        self.assertEqual(result["summary"]["arms"]["baseline"]["correct_samples"], 480)
        self.assertEqual(result["summary"]["pairs"]["fixed"]["sample_ties"], 480)
        self.assertEqual(result["summary"]["scorer_source_sha256"], s.snapshot(s.__file__)[1])

    def test_cli_fresh_output_binds_all_score_files(self):
        output = self.output / "scores"
        argv = ["score_aime.py", "--runs", *map(str, self.runs), "--protocol", str(self.protocol),
                "--data-root", str(self.data_root), "--output", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            s.main()
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["canonical_gate_file_sha256"], s.snapshot(output / "canonical_gate.json")[1])
        for arm in s.ARMS:
            self.assertEqual(summary["arms"][arm]["score_file_sha256"], s.snapshot(output / (arm + ".json"))[1])
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                s.main()
        self.assertEqual(error.exception.code, 2)

    def test_duplicate_missing_or_out_of_order_rejected(self):
        self.mutate(0, lambda m, rows: rows.__setitem__(1, rows[0]))
        with self.assertRaisesRegex(ValueError, "sample sample_id"):
            self.score()

    def test_failed_run_is_not_model_error(self):
        self.mutate(0, lambda m, rows: m.update(status="failed", error="OOM"))
        with self.assertRaisesRegex(ValueError, "manifest status"):
            self.score()

    def test_partial_cannot_claim_full(self):
        self.mutate(0, lambda m, rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "full sample count"):
            self.score()

    def test_changed_seed_rejected(self):
        self.mutate(1, lambda m, rows: rows[0].update(seed=42))
        with self.assertRaisesRegex(ValueError, "sample seed"):
            self.score()

    def test_changed_source_cannot_be_attributed_to_guidance(self):
        self.mutate(1, lambda m, rows: m["config"]["source"]["source_sha256"].update({"scripts/generate_aime.py": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "source/runtime/protocol"):
            self.score()

    def test_wrong_guidance_cap_rejected(self):
        self.mutate(2, lambda m, rows: m["config"]["guidance"].update(omega_cap=1.5))
        with self.assertRaisesRegex(ValueError, "guidance configuration"):
            self.score()

    def test_prompt_contamination_rejected_after_rehash(self):
        def corrupt(manifest, rows):
            rows[0]["prompt"] += "gold=204"
            rows[0]["prompt_sha256"] = s.text_hash(rows[0]["prompt"])
        self.mutate(0, corrupt)
        with self.assertRaisesRegex(ValueError, "public-only prompt"):
            self.score()

    def test_native_loop_and_cache_observations_are_required(self):
        self.mutate(0, lambda m, rows: rows[0]["execution_observation"].update(loop_calls=7))
        with self.assertRaisesRegex(ValueError, "native execution loop_calls"):
            self.score()

    def test_eos_cannot_be_called_cap(self):
        self.mutate(0, lambda m, rows: rows[0].update(cap_hit=True, stop_reason="max_new_tokens"))
        with self.assertRaisesRegex(ValueError, "stop reason"):
            self.score()

    def test_token_hash_and_cross_sample_prompt_are_checked(self):
        def corrupt(manifest, rows):
            rows[0]["prompt_token_ids"][1] = 101
            rows[0]["prompt_token_ids_sha256"] = s.hash_json(rows[0]["prompt_token_ids"])
        self.mutate(0, corrupt)
        with self.assertRaisesRegex(ValueError, "deterministic prompt"):
            self.score()

    def test_resigned_debug_protocol_rejected(self):
        self.mutate(0, lambda m, rows: m["config"].update(debug={"limit": None, "samples": None, "max_new_tokens": 256}))
        with self.assertRaisesRegex(ValueError, "debug settings"):
            self.score()

    def test_complete_sampling_defaults_cannot_be_omitted(self):
        self.mutate(0, lambda m, rows: m["config"]["generation_config"].pop("suppress_tokens"))
        with self.assertRaisesRegex(ValueError, "complete explicit GenerationConfig"):
            self.score()

    def test_actual_generator_configuration_construction_without_model(self):
        # Exercise generate_aime.run's actual protocol/config construction up
        # to initialize_states. Intercept before any model-generation call.
        audit_path = self.root / "results/aime_tokenizer_audit_20261004.json"
        if not audit_path.is_file():
            self.skipTest("Native tokenizer CPU audit unavailable")
        audit = json.loads(audit_path.read_text())["models"]["Ouro-1.4B-Thinking"]["rows"]
        by_prompt = {row["prompt"]: row["prompt_token_ids"] for row in audit}
        by_messages = {s.hash_json(row["messages"]): row for row in audit}
        class Tokenizer:
            def apply_chat_template(self, messages, **options):
                row = by_messages[s.hash_json(messages)]
                return row["prompt_token_ids"] if options["tokenize"] else row["prompt"]
            def encode(self, text, add_special_tokens):
                if add_special_tokens is not False:
                    raise AssertionError("Extra special tokens")
                return by_prompt[text]
        class Configuration:
            def __init__(self, **values):
                self.values = {**s.GENERATION_DEFAULTS, **values}
            def validate(self):
                pass
            def to_dict(self):
                return copy.deepcopy(self.values)
        class Captured(Exception):
            pass
        sample_config = json.loads((self.runs[0] / "manifest.json").read_text())["config"]
        capture = {}
        def capture_states(output, configs, prompts, *rest):
            capture.update(configs=configs, prompts=prompts)
            raise Captured()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.GenerationConfig = Configuration
        # The package __init__ imports torch for its public adapters; this
        # config-only test loads only the actual stdlib helper/runtime modules.
        package = types.ModuleType("loopcd_repro")
        package.__path__ = [str(self.root / "src/loopcd_repro")]
        with patch.dict(sys.modules, {"loopcd_repro": package}):
            generator = importlib.import_module("generate_aime")
            runtime = importlib.import_module("loopcd_repro.runtime")
        args = types.SimpleNamespace(protocol=self.protocol, data_root=self.data_root, model=self.output / "no-model",
             year=2024, debug_limit=None, debug_samples=None, debug_max_new_tokens=None, device="cuda:0",
             smoke=self.output / "no-smoke-executed", output=self.output / "no-generation", resume=False)
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers, "loopcd_repro": package,
                                     "loopcd_repro.runtime": runtime}), \
             patch.object(generator, "read_model_identity", return_value=sample_config["model_identity"]), \
             patch.object(runtime, "load_ouro", return_value=(object(), Tokenizer())), \
             patch.object(runtime, "provenance", return_value=copy.deepcopy(sample_config["source"])), \
             patch.object(generator, "validate_smoke"), \
             patch.object(generator, "initialize_states", side_effect=capture_states):
            with self.assertRaises(Captured):
                generator.run(args)
        self.assertEqual(len(capture["prompts"]), 30)
        for arm, config in capture["configs"].items():
            self.assertEqual(s.validate_configuration(config, self.data), (2024, arm))
            self.assertIsNone(config["debug"])
            self.assertEqual(s.hash_json(config["generation_config"]), s.GENERATION_CONFIG_SHA256)
        for task, prompt in capture["prompts"].items():
            messages, expected = s.expected_prompt(self.data["questions"][task], self.data["protocol"])
            self.assertEqual(prompt["messages"], messages)
            self.assertEqual(prompt["prompt"], expected)
            self.assertEqual(prompt["messages_sha256"], s.hash_json(messages))
            self.assertEqual(prompt["prompt_sha256"], s.text_hash(expected))
        self.assertFalse((self.output / "no-generation").exists())


if __name__ == "__main__":
    unittest.main()
