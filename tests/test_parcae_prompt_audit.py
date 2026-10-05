"""CPU-only audits of candidate boundaries and the two actual input policies."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_parcae_prompts.py"
spec = importlib.util.spec_from_file_location("parcae_prompt_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PairAuditTests(unittest.TestCase):
    def test_short_native_extra_unused_token_is_distinguished(self):
        item = audit.pair_views("abc", "de", lambda text: list(map(ord, text)), 8)
        self.assertEqual(item["native"]["input_tokens"], 5)
        self.assertEqual(item["hf"]["input_tokens"], 4)
        self.assertEqual(item["native"]["scoring_positions"], [2, 4])
        self.assertEqual(item["hf"]["scoring_positions"], [2, 4])
        self.assertTrue(item["comparison"]["effective_predictor_inputs_equal"])
        self.assertFalse(item["comparison"]["actual_model_inputs_equal"])

    def test_2048_boundary_and_one_more_token(self):
        encode = lambda text: list(map(ord, text))
        exact = audit.pair_views("a" * 2047, "b", encode)
        overflow = audit.pair_views("a" * 2048, "b", encode)
        self.assertEqual(exact["native"]["left_dropped"], 0)
        self.assertEqual(overflow["native"]["left_dropped"], 1)
        self.assertEqual(overflow["hf"]["left_dropped"], 0)
        self.assertTrue(overflow["comparison"]["hf_retains_one_extra_context_token"])
        self.assertTrue(overflow["native"]["scored_labels_match_input"])
        self.assertEqual(overflow["native"]["input_tokens"], overflow["hf"]["input_tokens"])
        self.assertNotEqual(overflow["native"]["scoring_positions"], overflow["hf"]["scoring_positions"])

    def test_many_dropped_tokens_preserve_full_candidate(self):
        item = audit.pair_views("abcdefghijklmnop", "qr", lambda text: list(map(ord, text)), 6)
        self.assertEqual(item["native"]["left_dropped"], 12)
        self.assertEqual(item["hf"]["left_dropped"], 11)
        self.assertTrue(item["native"]["all_labels_have_predictors"])
        self.assertTrue(item["hf"]["all_labels_have_predictors"])
        self.assertTrue(item["comparison"]["hf_retains_one_extra_context_token"])

    def test_empty_context_has_no_invented_bos_or_eot(self):
        item = audit.pair_views("", "abc", lambda text: list(map(ord, text)))
        self.assertTrue(item["empty_context"])
        self.assertTrue(item["hf"]["empty_context_requires_undefined_eot"])
        self.assertFalse(item["hf"]["all_labels_have_predictors"])
        self.assertFalse(item["native"]["all_labels_have_predictors"])
        self.assertEqual(item["raw_full_tokens"], 3)

    def test_long_or_empty_candidate_is_exposed(self):
        item = audit.pair_views("a", "bcdefgh", lambda text: list(map(ord, text)), 4)
        self.assertFalse(item["native"]["all_labels_have_predictors"])
        self.assertFalse(item["hf"]["all_labels_have_predictors"])
        empty = audit.pair_views("a", "", lambda text: list(map(ord, text)))
        self.assertTrue(empty["empty_continuation"])
        self.assertFalse(empty["native"]["scored_labels_match_input"])

    def test_native_rstrip_can_change_text_while_hf_preserves_it(self):
        mapping = {"a ": [1, 2], "a b": [1, 3], "a": [1], "ab": [1, 4]}
        item = audit.pair_views("a ", "b", mapping.__getitem__)
        self.assertTrue(item["native"]["retry_rstrip"])
        self.assertTrue(item["native"]["full_text_changed"])
        self.assertTrue(item["native"]["boundary_prefix_valid"])
        self.assertFalse(item["comparison"]["continuation_ids_equal"])
        self.assertFalse(item["comparison"]["full_tokenization_equal"])

    def test_unresolved_bpe_boundary_is_not_silently_marked_valid(self):
        mapping = {"a": [1], "ab": [2]}
        item = audit.pair_views("a", "b", mapping.__getitem__)
        self.assertFalse(item["native"]["boundary_prefix_valid"])
        self.assertFalse(item["hf"]["boundary_prefix_valid"])
        self.assertTrue(item["native"]["retry_rstrip"])
        self.assertFalse(item["native"]["all_labels_have_predictors"])

    def test_summary_counts_every_request_and_pins_shots(self):
        rows = [audit.pair_views("abc", "d", lambda text: list(map(ord, text)), cap) for cap in (3, 4)]
        result = audit.summarize(rows)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["counts"]["native_truncated"], 1)
        self.assertEqual(result["counts"]["hf_truncated"], 0)
        self.assertEqual(audit.SHOTS["arc_easy"], 0)
        self.assertEqual(audit.SHOTS["arc_challenge"], 25)
        self.assertEqual(audit.SHOTS["mmlu"], 5)
        self.assertEqual(sum(audit.EXPECTED_DOCS.values()), 31737)

    def test_piqa_explicit_pinned_filename_map_preserves_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = "a" * 40
            snapshot = root / "datasets--baber--piqa" / "snapshots" / pin
            snapshot.mkdir(parents=True)
            for split in ("train", "validation", "test"):
                (snapshot / ("piqa_" + split + ".parquet")).write_bytes(split.encode())
            registry = {"tasks": {"piqa": {"dataset_path": "baber/piqa", "revision": pin}}}
            class Manager:
                def _get_config(self, name):
                    return {"task": name, "dataset_path": "baber/piqa", "doc_to_text": "{{goal}}",
                            "doc_to_choice": ["{{sol1}}", "{{sol2}}"], "validation_split": "validation"}
            spec = audit.piqa_spec(registry, Manager(), root / "cache", root)
            self.assertEqual(set(spec["dataset_kwargs"]["data_files"]), {"train", "validation", "test"})
            self.assertEqual(spec["doc_to_text"], "{{goal}}")
            self.assertEqual(spec["validation_split"], "validation")
            self.assertEqual(spec["metadata"]["loopcd_dataset"]["revision"], pin)
            self.assertEqual(len(spec["metadata"]["loopcd_dataset"]["raw_files"]), 3)


if __name__ == "__main__":
    unittest.main()
