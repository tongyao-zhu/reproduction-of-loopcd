"""CPU checks of interleaved-run persistence and native observation contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import generate_aime as gen
import gpu_smoke_aime as smoke
from test_aime_protocol import ap, config_fixture, prompt_fixture, sample_fixture


class Handle:
    def __init__(self, collection, item):
        self.collection, self.item = collection, item

    def remove(self):
        self.collection.remove(self.item)


class HookModule:
    def __init__(self):
        self.pre, self.post = [], []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.pre.append(hook)
        return Handle(self.pre, hook)

    def register_forward_hook(self, hook):
        self.post.append(hook)
        return Handle(self.post, hook)


class UniversalTransformerCache:
    def __init__(self):
        self.length = 0
        self.max_cache_size = 96

    def get_seq_length(self):
        return self.length


class ObservationTests(unittest.TestCase):
    def fixture(self):
        model = HookModule()
        model.model = SimpleNamespace(layers=[HookModule()])
        model.lm_head = HookModule()
        return model, UniversalTransformerCache()

    def forward(self, model, cache, sequence, heads=1):
        for hook in model.pre:
            hook(model, (), {"past_key_values": cache})
        for step in sequence:
            for hook in model.model.layers[0].pre:
                hook(model.model.layers[0], (), {"current_ut": step})
        for _ in range(heads):
            for hook in model.lm_head.post:
                hook(model.lm_head, (), None)
        cache.length += 1
        for hook in model.post:
            hook(model, (), SimpleNamespace(past_key_values=cache))

    def test_counts_native_and_guided_heads_without_retaining_states(self):
        for mode, heads in (("native", 1), ("baseline", 1), ("fixed_zero", 1), ("adaptive_zero", 1), ("fixed", 2), ("adaptive", 2)):
            model, cache = self.fixture()
            with gen.observe_execution(model, cache, mode) as record:
                self.forward(model, cache, [0, 1, 2, 3], heads)
                self.forward(model, cache, [0, 1, 2, 3], heads)
            self.assertEqual(record["loop_calls"], 8)
            self.assertEqual(record["head_calls"], 2 * heads)
            self.assertEqual(record["final_cache_length"], 2)
            self.assertEqual(model.pre + model.post + model.lm_head.post + model.model.layers[0].pre, [])

    def test_bad_loop_or_head_restores_hooks(self):
        for sequence, heads in (([0, 1, 2], 1), ([0, 1, 3, 2], 1), ([0, 1, 2, 3], 2)):
            model, cache = self.fixture()
            with self.assertRaises(ValueError):
                with gen.observe_execution(model, cache, "baseline"):
                    self.forward(model, cache, sequence, heads)
            self.assertFalse(model.pre or model.post or model.lm_head.post or model.model.layers[0].pre)

    def test_native_settings_restore_after_exception(self):
        model = SimpleNamespace(early_exit_step=0, config=SimpleNamespace(early_exit_threshold=1.))
        with self.assertRaises(RuntimeError):
            with smoke.native_fixed_depth(model):
                self.assertEqual(model.early_exit_step, 3)
                self.assertIsNone(model.early_exit_threshold)
                raise RuntimeError("fixture")
        self.assertEqual(model.early_exit_step, 0)
        self.assertFalse(hasattr(model, "early_exit_threshold"))
        self.assertEqual(model.config.early_exit_threshold, 1.)
        self.assertFalse(hasattr(model.config, "early_exit_step"))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.smoke = self.root / "smoke.json"
        self.smoke.write_text('{"fixture":true}\n')
        self.prompts = {prompt_fixture()["task_id"]: prompt_fixture()}
        self.configs = {mode: config_fixture(mode) for mode in ap.ARMS}

    def initialize(self, resume=False, configs=None):
        return gen.initialize_states(self.root / "output", configs or self.configs, self.prompts, resume, self.smoke, "commit")

    def test_initializes_all_arms_with_paired_hash_and_gate_binding(self):
        states = self.initialize()
        self.assertEqual(set(states), set(ap.ARMS))
        self.assertEqual(len({state["manifest"]["paired_config_hash"] for state in states.values()}), 1)
        for state in states.values():
            self.assertEqual(state["manifest"]["expected_samples"], 2)
            self.assertEqual(state["manifest"]["gpu_smoke_sha256"], ap.sha256_file(self.smoke))
        with self.assertRaises(ValueError):
            self.initialize()

    def test_resume_partial_prefix_and_configuration_guard(self):
        states = self.initialize()
        for arm, state in states.items():
            row = sample_fixture(self.configs[arm])
            (state["folder"] / "samples.jsonl").write_text(json.dumps(row) + '\n{"task_id":')
        resumed = self.initialize(True)
        self.assertTrue(all(len(state["done"]) == 1 for state in resumed.values()))
        self.assertTrue(all(state["manifest"]["resume_history"][-1]["recovery"] for state in resumed.values()))
        changed = deepcopy(self.configs)
        changed["fixed"]["generation_config"]["max_new_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "frozen configuration"):
            self.initialize(True, changed)

    def test_completed_run_is_not_rewritten_or_repaired(self):
        states = self.initialize()
        for arm, state in states.items():
            path = state["folder"] / "samples.jsonl"
            path.write_text("\n".join(json.dumps(sample_fixture(self.configs[arm], i)) for i in range(2)) + "\n")
            state["manifest"].update(status="completed", completed_samples=2, samples_sha256=ap.sha256_file(path))
            ap.atomic_json(state["folder"] / "manifest.json", state["manifest"])
        before = {arm: (state["folder"] / "manifest.json").read_bytes() for arm, state in states.items()}
        self.initialize(True)
        self.assertEqual(before, {arm: (state["folder"] / "manifest.json").read_bytes() for arm, state in states.items()})
        corrupted = states["fixed"]["folder"] / "samples.jsonl"
        corrupted.write_bytes(corrupted.read_bytes() + b'{"tail":')
        raw = corrupted.read_bytes()
        with self.assertRaisesRegex(ValueError, "Completed samples changed"):
            self.initialize(True)
        self.assertEqual(corrupted.read_bytes(), raw)

    def test_resume_rejects_missing_arm_before_repair(self):
        self.initialize()
        (self.root / "output/fixed/manifest.json").unlink()
        with self.assertRaises(ValueError):
            self.initialize(True)


class GenerationConfigTests(unittest.TestCase):
    def test_registered_generation_configuration(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("Transformers is available in the remote CPU environment")
        path = ROOT / "data/aime/protocol-v1.json"
        if not path.exists():
            self.skipTest("Private registered protocol unavailable")
        protocol = ap.load_protocol(path)
        config = gen.generation_config(protocol).to_dict()
        for key, value in protocol["sampling"].items():
            self.assertEqual(config[key], value)
        if transformers.__version__ == "4.54.1":
            self.assertEqual(ap.hash_json(config), "d7492806bea88786419a34ad23baccae8dbdc4689fe041efaebd64f21e48b71e")
        self.assertEqual(gen.generation_config(protocol, 32).max_new_tokens, 32)


if __name__ == "__main__":
    unittest.main()
