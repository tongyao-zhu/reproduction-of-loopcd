"""Meaningful CPU checks for boundaries, duplicate execution and native scoring."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from loopcd_repro.parcae_mc import RequestAudit, digest, encode_pair, make_parcae_lm
from loopcd_repro.parcae_mc import request_identity

try:
    import torch
except ImportError:
    torch = None


class CharTokenizer:
    vocab_size = 64
    bos_id = eos_id = pad_id = None

    def encode(self, text, return_tensors=False):
        assert return_tensors is False
        return [ord(char) % self.vocab_size for char in text]


def request(context="abc", continuation=" de", candidate=0, doc_id=0):
    return SimpleNamespace(request_type="loglikelihood", repeats=1, task_name="tiny_parcae",
                           doc_id=doc_id, idx=candidate, args=(context, continuation))


class PairTests(unittest.TestCase):
    def test_whitespace_is_moved_without_losing_tokens(self):
        plan = encode_pair(CharTokenizer(), "abc \n", "de")
        self.assertEqual(plan["input_ids"], CharTokenizer().encode("abc \nd"))
        self.assertEqual(plan["continuation_ids"], CharTokenizer().encode(" \nde"))
        self.assertEqual(plan["pairing"]["original_context_length"], 3)

    def test_exact_registered_truncation_and_candidate_retention(self):
        plan = encode_pair(CharTokenizer(), "a" * 2050, "bc")
        self.assertEqual(len(plan["input_ids"]), 2048)
        self.assertEqual(plan["input_ids"][-1], ord("b") % 64)
        self.assertEqual(plan["continuation_ids"], [ord("b") % 64, ord("c") % 64])
        self.assertEqual(plan["pairing"]["left_truncated_tokens"], 3)

    def test_no_special_token_fallback_and_no_broken_boundaries(self):
        for context, continuation in (("", "x"), (" ", "x"), ("x", ""), ("a", "b" * 2049)):
            with self.subTest(context=context[:4], continuation=continuation[:4]), self.assertRaises(ValueError):
                encode_pair(CharTokenizer(), context, continuation)
        class Broken(CharTokenizer):
            def encode(self, text, return_tensors=False):
                return [1] if text == "a" else [2, 3]
        with self.assertRaisesRegex(ValueError, "non-prefix"):
            encode_pair(Broken(), "a", "b")

    def test_same_arguments_distinct_identities_are_planned_separately(self):
        audit = RequestAudit(io.StringIO())
        audit.plan([request(candidate=0), request(candidate=1)], CharTokenizer())
        self.assertEqual(len(audit.planned), 2)
        self.assertEqual(audit.planned[0]["arguments_sha256"], audit.planned[1]["arguments_sha256"])
        with self.assertRaises(ValueError):
            RequestAudit(io.StringIO()).plan([request(), request()], CharTokenizer())

    def test_external_prompt_mismatch_is_rejected_before_execution(self):
        audit = RequestAudit(io.StringIO(), {("tiny_parcae", 0, 0): {"wrong": True}})
        with self.assertRaisesRegex(ValueError, "differs"):
            audit.plan([request()], CharTokenizer())
        self.assertEqual(audit.count, 0)

    def test_record_detects_reseed_and_incomplete_execution(self):
        stream = io.StringIO()
        audit = RequestAudit(stream)
        audit.plan([request(candidate=0), request(candidate=1)], CharTokenizer())
        initial = {"rng_before": {"cpu": "a"}, "rng_after": {"cpu": "b"}}
        audit.record(audit.planned[0], initial["rng_before"], initial["rng_after"], initial, {}, -1., False)
        with self.assertRaisesRegex(ValueError, "reset"):
            audit.record(audit.planned[1], initial["rng_before"], initial["rng_after"], initial, {}, -1., False)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit.summary(require_complete=True)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_registered_arm_shots_cannot_silently_change(self):
        from evaluate_parcae import validate_configuration
        config = json.loads((ROOT / "configs/parcae_1_3b_mc.json").read_text())
        validate_configuration(config)
        config["tasks"]["arc_easy"] = 8
        with self.assertRaises(ValueError):
            validate_configuration(config)

    def test_harness_integer_subtypes_normalize_without_accepting_booleans(self):
        from enum import IntEnum
        class Index(IntEnum):
            FIRST = 0
        req = request(doc_id=Index.FIRST)
        self.assertIs(type(request_identity(req)["doc_id"]), int)
        req.doc_id = True
        with self.assertRaises(ValueError):
            request_identity(req)


@unittest.skipUnless(torch is not None, "Torch is not available in this local interpreter")
class ScoringTests(unittest.TestCase):
    def setUp(self):
        from test_parcae import Native
        class CompleteNative(Native):
            def forward_for_generation(self, *args, **kwargs):
                # Parent oracle's prelude is Identity, but must still be called.
                self.transformer.prelude[0](torch.zeros(1))
                return super().forward_for_generation(*args, **kwargs)
        class SmallTokenizer(CharTokenizer):
            vocab_size = 13
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.model = CompleteNative().eval()
        self.tokenizer = SmallTokenizer()
        self.config = json.loads((ROOT / "configs/parcae_1_3b_mc.json").read_text())

    def run_arm(self, arm, requests):
        stream = io.StringIO()
        audit = RequestAudit(stream)
        lm = make_parcae_lm(object)(self.model, self.tokenizer, self.config["arms"][arm], audit)
        torch.manual_seed(42)
        answers = lm.loglikelihood(requests)
        return answers, [json.loads(line) for line in stream.getvalue().splitlines()], audit.summary(True)

    def test_native_teacher_forced_oracle_scores_every_candidate_token(self):
        req = request()
        plan = encode_pair(self.tokenizer, *req.args)
        torch.manual_seed(42)
        logits = self.model.forward_for_generation(torch.tensor([plan["input_ids"]]), num_steps=8, past_key_values=None)["logits"]
        values = torch.log_softmax(logits.float(), -1)[0].detach()
        start = len(plan["input_ids"]) - len(plan["continuation_ids"])
        expected = sum(float(values[start + i, token]) for i, token in enumerate(plan["continuation_ids"]))
        answers, rows, summary = self.run_arm("baseline8", [req])
        self.assertAlmostEqual(answers[0][0], expected, places=5)
        self.assertEqual(rows[0]["pairing"]["continuation_length"], 3)
        self.assertTrue(summary["complete"])

    def test_all_seven_arms_keep_same_rng_inputs_and_initialization_stream(self):
        requests = [request(candidate=0), request(candidate=1), request("longer", " xy", doc_id=1)]
        evidence = {}
        for arm in self.config["arms"]:
            answers, rows, summary = self.run_arm(arm, requests)
            evidence[arm] = summary["pairing_sha256"]
            self.assertEqual(len(answers), 3)
            self.assertEqual(rows[0]["pairing"]["arguments_sha256"], rows[1]["pairing"]["arguments_sha256"])
            self.assertNotEqual(rows[0]["pairing"]["initialization"]["first_16_values_sha256"], rows[1]["pairing"]["initialization"]["first_16_values_sha256"])
            self.assertEqual(rows[0]["pairing"]["rng_after"], rows[1]["pairing"]["rng_before"])
            self.assertFalse("initialize_state" in self.model.__dict__)
        self.assertEqual(len(set(evidence.values())), 1)

    def test_forward_error_restores_observation_hooks_and_initializer(self):
        native = self.model.lm_head.forward
        def fail(*args):
            raise RuntimeError("deliberate head failure")
        self.model.lm_head.forward = fail
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.run_arm("fixed8", [request()])
        self.model.lm_head.forward = native
        self.assertNotIn("initialize_state", self.model.__dict__)
        self.assertNotIn("core_block_forward", self.model.__dict__)
        self.assertFalse(self.model.transformer.C._forward_hooks)
        self.assertFalse(self.model.transformer.C._forward_pre_hooks)


@unittest.skipUnless(torch is not None and os.environ.get("PARCAE_NATIVE_SOURCE"), "Optional fixed native source CPU integration")
class NativeHarnessIntegration(unittest.TestCase):
    def test_real_native_tiny_model_with_real_lm_eval_task_scoring(self):
        """Random tiny native model; no checkpoint or GPU/certificate is used."""
        sys.path.insert(0, os.environ["PARCAE_NATIVE_SOURCE"])
        from receval.models.parcae import ModelingParcae
        from parcae_lm.models.parcae import ParcaeConfig
        from lm_eval import simple_evaluate
        from lm_eval.api.model import LM
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable
        from evaluate_parcae import write_json
        from datasets import Dataset
        torch.set_num_threads(1)
        config = ParcaeConfig(n_embd=32, intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=64, padded_vocab_size=64, block_size=2048,
            recurrent_embedding_dimension=32, recurrent_intermediation_embedding_dimension=64,
            recurrent_num_attention_heads=4, n_layers_in_prelude=8, n_layers_in_recurrent_block=8,
            n_layers_in_coda=8, mean_recurrence=8, mean_backprop_depth=8,
            attn_impl="sdpa", use_fused_head="pytorch", state_init="like-init")
        torch.manual_seed(123)
        model = ModelingParcae(config).to(dtype=torch.bfloat16).eval()
        arms = json.loads((ROOT / "configs/parcae_1_3b_mc.json").read_text())["arms"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "test.parquet"
            Dataset.from_dict({"question": ["ab", "ab"], "choices": [["x", "x"], ["y", "z"]], "answer": [0, 1]}).to_parquet(data)
            spec = {"task": "tiny_parcae", "dataset_path": "parquet",
                    "dataset_kwargs": {"data_files": {"test": [str(data)]}, "cache_dir": str(tmp / "cache")},
                    "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
                    "doc_to_text": "{{question}}", "doc_to_choice": "choices", "doc_to_target": "answer",
                    "metric_list": [{"metric": metric, "aggregation": "mean", "higher_is_better": True} for metric in ("acc", "acc_norm")]}
            # TaskManager pops task from dict overrides. Register the native
            # YAML first, as all seven formal tasks are registered in practice.
            (tmp / "tiny.yaml").write_text(json.dumps(spec))
            hashes = []
            manager = TaskManager(include_path=str(tmp), include_defaults=False)
            group = {"group": "tiny_group", "task": [spec], "aggregate_metric_list": [
                {"metric": "acc", "weight_by_size": True, "aggregation": "mean"}]}
            for arm in arms:
                audit = RequestAudit(io.StringIO())
                lm = make_parcae_lm(LM)(model, CharTokenizer(), arms[arm], audit)
                torch.manual_seed(42)
                import copy
                result = simple_evaluate(model=lm, tasks=[copy.deepcopy(group)], task_manager=manager, num_fewshot=0, batch_size=1,
                    bootstrap_iters=0, log_samples=True, random_seed=42, numpy_random_seed=42,
                    torch_random_seed=None, fewshot_random_seed=42, apply_chat_template=False)
                self.assertEqual(len(result["samples"]["tiny_parcae"]), 2)
                self.assertEqual(audit.count, 4)
                self.assertEqual(result["n-samples"]["tiny_parcae"], {"original": 2, "effective": 2})
                self.assertEqual(result["groups"]["tiny_group"]["acc,none"], result["results"]["tiny_parcae"]["acc,none"])
                write_json(tmp / (arm + ".json"), result, handle_non_serializable)
                persisted = json.loads((tmp / (arm + ".json")).read_text())
                self.assertEqual(len(persisted["samples"]["tiny_parcae"]), 2)
                hashes.append(audit.summary(True)["pairing_sha256"])
            self.assertEqual(len(set(hashes)), 1)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
