"""CPU contracts for registered data, prompt identity, resume, and GPU evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loopcd_repro import aime_protocol as ap


def fixture_protocol():
    return {"prompt": {"system": "You are a helpful assistant.", "joiner": "\n\n", "user_suffix": r"Put answer in \boxed{}.",
                        "apply_chat_template": {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}},
            "sampling": {"max_new_tokens": 8192}, "execution": {"max_prompt_plus_output_budget": 32768}}


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, **kwargs):
        if tokenize:
            return [1, 7, 2, 1, 8, 2, 1, 9]
        return messages[0]["content"] + messages[1]["content"] + "<|im_start|>assistant\n<think>\n"

    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("No second set of special tokens")
        return [1, 7, 2, 1, 8, 2, 1, 9]


def prompt_fixture():
    q = {"task_id": "AIME2024-I-01", "question": "question", "question_sha256": ap.hash_text("question")}
    return ap.build_prompt(Tokenizer(), q, fixture_protocol())


def config_fixture(mode="baseline", count=2):
    return {"guidance": {"mode": mode, "omega": .5, "omega_cap": 1., "early_loop": 1, "total_loops": 4},
            "samples_per_problem": count, "generation_config": {"max_new_tokens": 2},
            "model_identity": {"repo_id": "ByteDance/Ouro-1.4B-Thinking"}, "is_full_split": False}


def sample_fixture(config=None, sample_id=0, prompt=None):
    config, prompt = config or config_fixture(), prompt or prompt_fixture()
    mode = config["guidance"]["mode"]
    heads = 1 if mode == "baseline" else 2
    observation = {"mode": mode, "native_exit_at_step": 3, "guidance_applied": mode != "baseline",
                   "extra_lm_head_calls": 0 if mode == "baseline" else 1,
                   "executed_source_indices": [0, 1, 2, 3], "early_loop": 1, "already_normalized_identity": True}
    return {"schema_version": 1, **prompt, "arm": mode, "sample_id": sample_id,
            "config_hash": ap.hash_json(config), "paired_config_hash": ap.hash_json(ap.paired_config(config)),
            "seed": ap.sample_seed(prompt["task_id"], sample_id),
            "generation_config_sha256": ap.hash_json(config["generation_config"]), "effective_max_new_tokens": 2,
            "generated_token_ids": [7, 2], "generated_token_ids_sha256": ap.hash_json([7, 2]),
            "generated_tokens": 2, "raw_generation": "hello<|im_end|>",
            "raw_generation_sha256": ap.hash_text("hello<|im_end|>"), "completion": "hello", "completion_sha256": ap.hash_text("hello"),
            "stop_reason": "eos", "cap_hit": False, "elapsed_seconds": 1., "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 12, "adapter_observation": observation,
            "execution_observation": {"forward_calls": 2, "loop_calls": 8, "head_calls": 2 * heads,
                "observed_loop_pattern_valid": True, "expected_head_calls_per_forward": heads,
                "cache_type": "UniversalTransformerCache", "cache_slots": 96, "fresh_cache_initial_length": 0,
                "final_cache_length": len(prompt["prompt_token_ids"]) + 1}}


def smoke_fixture():
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_aime import GENERATION_DEFAULTS
    sampling = {"do_sample": True, "temperature": 1.0, "top_p": 0.7, "top_k": 0,
                "typical_p": 1.0, "min_p": None, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "num_beams": 1,
                "num_return_sequences": 1, "max_new_tokens": 8192, "min_new_tokens": 0,
                "eos_token_id": 2, "pad_token_id": 2, "bos_token_id": 1,
                "forced_bos_token_id": None, "forced_eos_token_id": None,
                "use_cache": True, "cache_implementation": None}
    generation = {**sampling, **GENERATION_DEFAULTS}
    fixture = prompt_fixture()
    identity = {"repo_id": "ByteDance/Ouro-1.4B-Thinking", "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d", "model_code_sha256": "c"}
    source = {"loaded_model_code_sha256": "c", "source_sha256": {"x": "y"}, "git_commit": "a" * 40,
              "python": "test", "packages": {"transformers": "4.54.1"}, "cuda": "test", "gpu": "test",
              "precision": "BF16 model; FP32 guidance before native sampling", "attention": "sdpa"}

    def observation(calls, heads=1, initial=0, final=None):
        return {"forward_calls": calls, "loop_calls": calls * 4, "head_calls": calls * heads,
                "observed_loop_pattern_valid": True, "expected_head_calls_per_forward": heads,
                "cache_type": "UniversalTransformerCache", "cache_slots": 96,
                "fresh_cache_initial_length": initial,
                "final_cache_length": len(fixture["prompt_token_ids"]) + calls - 1 if final is None else final}

    def short(ids, cap, forced=None):
        return {"generated_token_ids": ids, **ap.stop_metadata(ids, cap), "seed": 42,
                "generation_config": {**generation, "max_new_tokens": cap},
                "forced_fixture_token": forced, "execution_observation": observation(len(ids))}

    report = {"schema_version": 1, "kind": "aime_gpu_smoke", "status": "PASS",
            "checks": [{"name": n, "passed": True} for n in sorted(ap.REQUIRED_SMOKE_CHECKS)],
            "bindings": ap.smoke_bindings(identity, source), "source": source, "fixture": fixture,
            "short_generation": {name: short([7, 2], 32) for name in ("native", "baseline", "fixed_zero", "adaptive_zero")},
            "prompt_audit": [{"task_id": task, "tokens": 471, "prompt_sha256": "p", "prompt_token_ids_sha256": "t"}
                             for task in ap.expected_task_ids(2024) + ap.expected_task_ids(2025)],
            "resource_gate": {"status": "PASS", "scope": "native_cache_prefill_plus_one_cached_decode", "max_new_tokens": 8192,
                              "max_prompt_tokens_all60": 471, "target_prefill_tokens": 8663, "actual_prefill_tokens": 8663,
                              "final_cache_length": 8664, "measured_autoregressive_tokens": 1, "cache_slots": 96,
                              "peak_allocated_bytes": 100, "peak_reserved_bytes": 120, "prefill_chunk_tokens": 512,
                              "prefill_execution_observation": observation(17, final=8663),
                              "decode_execution_observation": observation(1, heads=2, initial=8663, final=8664),
                              "final_guidance_observation": sample_fixture(config_fixture("adaptive"))["adapter_observation"]}}
    for row in report["checks"]:
        if row["name"] == "eos_stop":
            row["fixture"] = short([2], 3, 2)
        elif row["name"] == "cap_stop":
            row["fixture"] = short([7, 7, 7], 3, 7)
    return report


class ProtocolTests(unittest.TestCase):
    def test_seeds_match_registration_all_960(self):
        path = ROOT / "data/aime/protocol-v1.json"
        if not path.exists():
            self.skipTest("Private registration not installed")
        protocol = ap.load_protocol(path)
        seeds = [ap.sample_seed(row["task_id"], row["sample_id"]) for row in protocol["seed"]["records"]]
        self.assertEqual(seeds, [row["seed"] for row in protocol["seed"]["records"]])
        self.assertEqual(len(set(seeds)), 960)

    def test_invalid_seed_identity(self):
        for task, sample in (("unknown", 0), ("AIME2024-I-01", 16), ("AIME2024-I-01", True)):
            with self.assertRaises(ValueError):
                ap.sample_seed(task, sample)

    def test_preserves_three_native_chat_delimiters(self):
        prompt = prompt_fixture()
        self.assertEqual(prompt["prompt_token_ids"].count(1), 3)
        self.assertEqual(prompt["prompt_token_ids"].count(2), 2)
        self.assertEqual(prompt["messages_sha256"], ap.hash_json(prompt["messages"]))

    def test_reject_extra_bos_or_budget_shrink(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *a, **k: [1] + Tokenizer().encode(*a, **k)
        with self.assertRaisesRegex(ValueError, "extra special"):
            ap.build_prompt(tokenizer, {"task_id": "x", "question": "x", "question_sha256": "x"}, fixture_protocol())
        protocol = fixture_protocol()
        protocol["execution"]["max_prompt_plus_output_budget"] = 8192
        with self.assertRaisesRegex(ValueError, "budget"):
            ap.build_prompt(Tokenizer(), {"task_id": "x", "question": "x", "question_sha256": "x"}, protocol)

    def test_stop_rules_and_unregistered_early_stop(self):
        self.assertEqual(ap.stop_metadata([7, 2], 2), {"stop_reason": "eos", "cap_hit": False})
        self.assertEqual(ap.stop_metadata([7, 7], 2), {"stop_reason": "max_new_tokens", "cap_hit": True})
        for ids in ([], [7], [2, 7], [7, 7, 7], [True, 2]):
            with self.assertRaises(ValueError):
                ap.stop_metadata(ids, 2)

    def test_paired_hash_removes_only_guidance(self):
        a, b = config_fixture("baseline"), config_fixture("adaptive")
        self.assertEqual(ap.hash_json(ap.paired_config(a)), ap.hash_json(ap.paired_config(b)))
        b["generation_config"]["max_new_tokens"] = 3
        self.assertNotEqual(ap.hash_json(ap.paired_config(a)), ap.hash_json(ap.paired_config(b)))

    def test_valid_records_and_execution_variants(self):
        for mode in ap.ARMS:
            config = config_fixture(mode)
            ap.validate_sample(sample_fixture(config), config, prompt_fixture())

    def test_record_mutations_rejected(self):
        config = config_fixture()
        for key, value in (("seed", 1), ("completion", "edited"), ("prompt", "edited"), ("effective_max_new_tokens", 1),
                           ("generated_token_ids_sha256", "changed"), ("cap_hit", True), ("elapsed_seconds", float("nan"))):
            row = sample_fixture(config)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ap.validate_sample(row, config, prompt_fixture())
        for key, value in (("loop_calls", 4), ("head_calls", 4), ("fresh_cache_initial_length", 1), ("final_cache_length", 1)):
            row = sample_fixture(config)
            row["execution_observation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ap.validate_sample(row, config, prompt_fixture())


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_resume_requires_ordered_prefix(self):
        path = self.root / "samples.jsonl"
        config = config_fixture()
        prompts = {prompt_fixture()["task_id"]: prompt_fixture()}
        good = [sample_fixture(config, i) for i in range(2)]
        for rows in (good[:1], good):
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertEqual(len(ap.read_completed(path, config, prompts)), len(rows))
        for rows in ([good[1]], good[::-1], [good[0], good[0]]):
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "ordered prefix"):
                ap.read_completed(path, config, prompts)

    def test_tail_recovery_preserves_bytes_and_never_repairs_middle(self):
        path = self.root / "samples.jsonl"
        raw = b'{"valid":1}\n{"partial":"\xe4'
        path.write_bytes(raw)
        result = ap.repair_partial_tail(path)
        self.assertEqual((self.root / result["backup"]).read_bytes(), raw)
        self.assertEqual(path.read_bytes(), b'{"valid":1}\n')
        path.write_bytes(b'{"valid":1}')
        ap.repair_partial_tail(path)
        self.assertEqual(path.read_bytes(), b'{"valid":1}\n')
        for raw in (b'broken\n{"valid":1}\n', b'{"bad":\n'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                ap.repair_partial_tail(path)
            self.assertEqual(path.read_bytes(), raw)

    def test_data_loader_never_opens_gold(self):
        private = ROOT / "data/aime/prepared-v1"
        if not private.exists():
            self.skipTest("Private prepared data unavailable")
        original = Path.read_bytes
        opened = []
        def track(path):
            opened.append(path.name)
            if "answers" in path.name or "parquet" in path.name:
                raise AssertionError("Gold file access")
            return original(path)
        with patch.object(Path, "read_bytes", track):
            self.assertEqual(len(ap.load_questions(private, 2024)), 30)
            self.assertEqual(len(ap.load_questions(private, 2025)), 30)
        self.assertEqual(set(opened), {"manifest.json", "aime2024.questions.jsonl", "aime2025.questions.jsonl"})

    def test_smoke_rejects_empty_false_or_short_capacity_gate(self):
        path = self.root / "smoke.json"
        report = smoke_fixture()
        path.write_text(json.dumps(report))
        self.assertEqual(ap.validate_smoke(path), report)
        cases = []
        changed = deepcopy(report); changed["checks"] = []; cases.append(changed)
        changed = deepcopy(report); changed["checks"][0]["passed"] = False; cases.append(changed)
        changed = deepcopy(report); changed["resource_gate"]["actual_prefill_tokens"] = 32; cases.append(changed)
        changed = deepcopy(report); changed["resource_gate"]["cache_slots"] = 192; cases.append(changed)
        changed = deepcopy(report); changed["prompt_audit"] = changed["prompt_audit"][:-1]; cases.append(changed)
        changed = deepcopy(report); changed["bindings"]["source_sha256"] = {}; cases.append(changed)
        changed = deepcopy(report); changed["short_generation"]["fixed_zero"]["execution_observation"]["head_calls"] = 4; cases.append(changed)
        changed = deepcopy(report); changed["short_generation"]["baseline"]["generation_config"]["top_k"] = 50; cases.append(changed)
        changed = deepcopy(report); changed["resource_gate"]["decode_execution_observation"]["forward_calls"] = 2; cases.append(changed)
        changed = deepcopy(report); changed["resource_gate"]["prefill_execution_observation"]["final_cache_length"] = 100; cases.append(changed)
        changed = deepcopy(report); changed["source"]["precision"] = "changed"; cases.append(changed)
        for changed in cases:
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                ap.validate_smoke(path)

    def test_smoke_binds_current_loaded_source(self):
        path = self.root / "smoke.json"
        report = smoke_fixture()
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            ap.validate_smoke(path, {**report["bindings"], "source_sha256": {"x": "edited"}})


if __name__ == "__main__":
    unittest.main()
