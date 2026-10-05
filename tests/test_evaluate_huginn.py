import importlib.util
import io
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_huginn", ROOT / "scripts/evaluate_huginn.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
CONFIG = json.loads((ROOT / "configs/huginn_mc.json").read_text())


class HuginnProtocolTests(unittest.TestCase):
    def test_paper_zero_shot_across_all_tasks(self):
        self.assertEqual(len(CONFIG["tasks"]), 7)
        self.assertEqual(set(CONFIG["tasks"].values()), {0})

    def test_standard_and_half_depth_references_are_distinct(self):
        self.assertEqual(module.resolve_protocol(CONFIG, "hidden", loops=16)["reference_loop"], 7)
        self.assertEqual(module.resolve_protocol(CONFIG, "hidden", loops=32)["reference_loop"], 6)
        half = module.resolve_protocol(CONFIG, "hidden", protocol="half-depth")
        self.assertEqual((half["total_loops"], half["reference_loop"], half["omega"]), (16, 6, 0.5))
        self.assertEqual(module.resolve_protocol(CONFIG, "baseline", protocol="half-depth")["total_loops"], 32)

    def test_half_depth_protocol_rejects_wrong_depth(self):
        with self.assertRaisesRegex(ValueError, "requires baseline R32 or hidden R16"):
            module.resolve_protocol(CONFIG, "hidden", protocol="half-depth", loops=32)

    def test_custom_reference_is_labeled(self):
        self.assertFalse(module.resolve_protocol(CONFIG, "hidden", loops=32, reference_loop=7)["paper_guidance_parameters"])

    def test_no_silent_truncation_and_exact_boundary(self):
        module.validate_requests([(None, [1] * 4096, [2])])
        with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
            module.validate_requests([(None, [1] * 4097, [2])])

    def test_identical_streams_match_despite_arm_depth_differences(self):
        def arm():
            audit = module.RequestAudit(io.StringIO())
            audit.add_plan([(None, [2, 3], [4]), (None, [1], [2])])
            for tokens in ([2, 3], [1]):
                init = {"rng_before": {"cpu": "before"}, "rng_after": {"cpu": "after"},
                        "shape": [1, len(tokens), 16], "dtype": "bfloat16", "first_16_values_sha256": "same-noise"}
                audit.record(tokens, init["rng_before"], init["rng_after"], [init])
            return audit.summary(require_complete=True)
        self.assertEqual(arm(), arm())

    def test_wrong_request_order_rejected(self):
        audit = module.RequestAudit(io.StringIO())
        audit.add_plan([(None, [2, 3], [4]), (None, [1], [2])])
        with self.assertRaisesRegex(ValueError, "forward order"):
            audit.record([1], {}, {}, [{}])

    def test_extra_randomness_rejected(self):
        audit = module.RequestAudit(io.StringIO())
        audit.add_plan([(None, [1], [2])])
        with self.assertRaisesRegex(ValueError, "outside native"):
            audit.record([1], {"cpu": "a"}, {"cpu": "c"}, [{"rng_before": {"cpu": "a"}, "rng_after": {"cpu": "b"}}])

    def test_missing_initialization_rejected(self):
        audit = module.RequestAudit(io.StringIO())
        audit.add_plan([(None, [1], [2])])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit.record([1], {}, {}, [])

    def test_incomplete_audit_rejected(self):
        audit = module.RequestAudit(io.StringIO())
        audit.add_plan([(None, [1], [2])])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit.summary(require_complete=True)

    @unittest.skipUnless(os.environ.get("LOOPCD_HUGINN_TEST_MODEL"), "Optional tiny native CPU gate needs the prepared model's code/tokenizer")
    def test_native_hflm_three_arm_random_streams_match(self):
        import torch
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from lm_eval.models.huggingface import HFLM
        from loopcd_repro.huginn import HuginnHiddenConfig, HuginnHiddenGuidance

        path = os.environ["LOOPCD_HUGINN_TEST_MODEL"]
        cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", path, local_files_only=True)
        config = cls.config_class(n_embd=32, n_heads=4, n_layers=6, block_size=64, vocab_size=31,
                                  padding_multiple=1, intermediate_size=64, n_layers_in_prelude=2,
                                  n_layers_in_recurrent_block=2, n_layers_in_coda=2, mean_recurrence=32,
                                  mean_backprop_depth=0, pad_token_id=30, bos_token_id=1, eos_token_id=2,
                                  torch_dtype="float32")
        model = cls(config).eval()  # Only tiny random CPU weights, never checkpoint weights.
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        requests = [(("a", "b"), [3, 4, 5], [6]), (("c", "d"), [7, 8], [9, 10]), (("e", "f"), [11], [12])]
        traces, summaries = [], []
        for mode, loops in (("baseline", 32), ("hidden", 32), ("hidden", 16)):
            stream = io.StringIO()
            audit = module.RequestAudit(stream)
            wrapper = module.make_audited_hflm(HFLM, torch, audit, loops)(
                pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=64,
                softmax_dtype=torch.float32, logits_cache=False, truncation=False)
            torch.manual_seed(42)
            with HuginnHiddenGuidance(model, HuginnHiddenConfig(mode, loops, 6, .5)):
                self.assertEqual(len(wrapper._loglikelihood_tokens(requests, disable_tqdm=True)), 3)
            self.assertNotIn("initialize_state", model.__dict__)
            traces.append(stream.getvalue())
            summaries.append(audit.summary(require_complete=True))
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], traces[2])
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0], summaries[2])


if __name__ == "__main__":
    unittest.main()
