"""Evidence-only Parcae comparison guards, with complete audited synthetic corpora."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/compare_parcae.py"
SPEC = importlib.util.spec_from_file_location("compare_parcae", PATH)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_bytes(b"".join(c.canonical(row) + b"\n" for row in rows))


class Fixture:
    def __init__(self, root, limit=4, task="sciq", arms=None):
        self.root, self.limit = Path(root), limit
        self.source_root = self.root / "release"
        sources = ["scripts/compare_parcae.py", "scripts/evaluate_parcae.py", "scripts/prepare_parcae.py",
                   "scripts/audit_parcae_prompts.py", "src/loopcd_repro/parcae.py", "src/loopcd_repro/parcae_mc.py",
                   "src/loopcd_repro/guidance.py", "src/loopcd_repro/runtime.py", "configs/parcae_1_3b_mc.json", "configs/mc_datasets.json"]
        for name in sources:
            destination = self.source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((PATH.parents[1] / name).read_bytes())
        (self.source_root / "docs").mkdir()
        (self.source_root / "docs/parcae_mc_protocol.md").write_text("Registered fixture protocol\n")
        source_hashes = {name: c.sha256(self.source_root / name) for name in sources}
        paper_config = c.read_json(self.source_root / "configs/parcae_1_3b_mc.json")
        prepared = {"repo_id": c.MODEL_REPO, "revision": c.MODEL_REVISION,
                    "source": {"git_commit": c.OFFICIAL_SOURCE, "git_tree_sha1": c.SOURCE_TREE,
                               "files": {"receval/models/parcae.py": {"sha256": h("model"), "bytes": 10}}},
                    "files": {name: {"sha256": value, "bytes": 10} for name, value in c.MODEL_FILES.items()}}
        prepared_sha = hashlib.sha256((json.dumps(prepared, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        raw_source = {"fixture": True, "sha256": h("dataset")}
        self.audit = self.root / "audit"
        self.audit.mkdir()
        records, self.examples = [], []
        for doc_id in range(c.DOCS[task]):
            doc = {"question": f"Question {doc_id}", "answer": 0}
            # Two identical candidate requests must still be independently scored.
            arguments = [[f"Question {doc_id}?", " same" if candidate < 2 else f" answer{candidate}"] for candidate in range(4)]
            for candidate, args in enumerate(arguments):
                token = candidate if candidate >= 2 else 0
                hf = {"boundary_prefix_valid": True, "all_labels_have_predictors": True,
                      "context_sha256": h([doc_id, "context"]), "full_sha256": h([doc_id, token, "full"]),
                      "input_sha256": h([doc_id, token, "input"]), "continuation_sha256": h([token, "cont"]),
                      "context_tokens": 2, "full_tokens": 4, "input_tokens": 3, "continuation_tokens": 2,
                      "left_dropped": 0}
                records.append({"task": task, "leaf": task, "doc_id": doc_id, "candidate": candidate,
                                "doc_sha256": c.hash_json(doc), "arguments_sha256": c.hash_json(args),
                                "context_sha256": c.text_hash(args[0]), "continuation_sha256": c.text_hash(args[1]),
                                "empty_context": False, "empty_continuation": False, "hf": hf})
            self.examples.append({"doc_id": doc_id, "doc": doc, "target": 0, "arguments": arguments,
                                  "doc_hash": h([doc_id, "doc"]), "prompt_hash": h([doc_id, "prompt"]),
                                  "target_hash": h("target"), "filter": "none", "metrics": ["acc", "acc_norm"]})
        with gzip.open(self.audit / (task + ".jsonl.gz"), "wt") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
        self.records = {(r["leaf"], r["doc_id"], r["candidate"]): r for r in records}
        harness = {"lm_eval/" + p: {"sha256": h(p), "bytes": 100} for p in ["evaluator.py", "api/task.py", "utils.py"]}
        write_json(self.audit / "harness_files.json", harness)
        task_meta = {name: {"documents": c.DOCS[name], "requests": c.REQUESTS[name]} for name in c.SHOTS}
        task_meta[task].update(leaves={task: {"documents": c.DOCS[task], "requests": c.REQUESTS[task], "dataset_source": raw_source}},
                                 records={"file": (task + ".jsonl.gz"), "sha256": c.sha256(self.audit / (task + ".jsonl.gz"))})
        report = {"status": "PASS", "registered_hf_policy_ready": True, "seed": 42, "shots": c.SHOTS,
                  "max_length": 2048, "total_documents": sum(c.DOCS.values()), "total_requests": sum(c.REQUESTS.values()),
                  "tasks": task_meta, "model_identity": {"repo_id": c.MODEL_REPO, "revision": c.MODEL_REVISION,
                  "files": c.MODEL_FILES, "source_tree_sha1": c.SOURCE_TREE},
                  "model_full_hash_verified_before_and_after": True, "cuda_initialized": False,
                  "harness": {"commit": c.HARNESS_COMMIT, "files_sha256": c.hash_json(harness)},
                  "model_manifest_sha256": prepared_sha, "registry_sha256": source_hashes["configs/mc_datasets.json"]}
        write_json(self.audit / "result.json", report)
        self.paths = []
        for arm in (arms or c.ARMS):
            p = self.root / arm
            p.mkdir()
            self.paths.append(p)
            guidance = copy.deepcopy(c.GUIDANCE[arm])
            n = c.DOCS[task] if limit is None else limit
            responses, samples, trace = {}, [], []
            for doc_id in range(n):
                row = copy.deepcopy(self.examples[doc_id])
                row["filtered_resps"] = [[-float(doc_id + candidate + 1), candidate == 0] for candidate in range(4)]
                row["resps"] = [[r] for r in row["filtered_resps"]]
                values = {"baseline8": [1, 0, 0, 1], "fixed8": [1, 1, 0, 0], "adaptive8": [0, 0, 0, 1],
                          "hidden8": [1, 1, 1, 1], "baseline4": [0, 0, 0, 1], "fixed4": [0, 1, 0, 1],
                          "hidden4": [1, 0, 0, 1]}[arm]
                row["acc"] = values[doc_id % 4]
                row["acc_norm"] = (doc_id % 2)
                samples.append(row)
                for candidate in range(4):
                    index = len(trace)
                    pair = c._pairing_fields(self.records[(task, doc_id, candidate)])
                    before, after = {"cpu": h("cpu"), "model_device": h(index)}, {"cpu": h("cpu"), "model_device": h(index + 1)}
                    pair.update(rng_before=before, rng_after=after, initialization={"rng_before": before, "rng_after": after,
                                "shape": [1, 3, 1024], "dtype": "torch.bfloat16", "first_16_values_sha256": h([index, "first"]),
                                "last_16_values_sha256": h([index, "last"])})
                    enabled = guidance["mode"] != "baseline"
                    readouts = 2 if guidance["mode"] in ("fixed", "adaptive") else 1
                    observation = {"mode": guidance["mode"], "guidance_applied": enabled,
                                   "total_loops": guidance["total_loops"], "readout_passes": readouts}
                    if enabled:
                        observation.update(reference_loop=1, executed_source_indices=list(range(guidance["total_loops"])),
                                           combination_location="before_C" if guidance["mode"] == "hidden" else "after_complete_native_readout", cache=None)
                    trace.append({"index": index, "pairing": pair,
                                  "observation": {"guidance": observation, "call_counts": {"initializations": 1, "prelude": 8, "core_layers": 8 * guidance["total_loops"], "projection": readouts, "coda": 8 * readouts, "norm": readouts, "head": readouts}},
                                  "loglikelihood": row["filtered_resps"][candidate][0], "is_greedy": row["filtered_resps"][candidate][1]})
            write_rows(p / ("samples_" + task + ".jsonl"), samples)
            write_rows(p / "request_trace.jsonl", trace)
            result = {"results": {task: {"acc,none": sum(x["acc"] for x in samples) / n,
                                            "acc_norm,none": sum(x["acc_norm"] for x in samples) / n}},
                      "configs": {task: {"test_split": c.SPLITS[task], "num_fewshot": 0}},
                      "n-shot": {task: 0}, "n-samples": {task: {"original": c.DOCS[task], "effective": n}},
                      "versions": {task: 1}, "higher_is_better": {task: {"acc": True, "acc_norm": True}}}
            write_json(p / "results.json", result)
            manifest = {"status": "completed", "task": task, "arm": arm, "guidance": guidance,
                        "protocol": "parcae_mc_v1", "paper_config": paper_config,
                        "paper_config_sha256": source_hashes["configs/parcae_1_3b_mc.json"],
                        "registry_sha256": source_hashes["configs/mc_datasets.json"],
                        "protocol_sha256": c.sha256(self.source_root / "docs/parcae_mc_protocol.md"),
                        "harness_commit": c.HARNESS_COMMIT, "harness_files_sha256": c.hash_json(harness),
                        "add_bos": False, "truncation": "left_keep_2049_then_remove_last_token",
                        "request_order": "native_harness_order_no_sort_no_dedup",
                        "seed_reset": "once_after_model_loading_before_harness_requests; no per-request reseeding",
                        "model_full_hash_verified_before_and_after": True, "execution_device_type": "cuda",
                        "gpu_smoke_sha256": h("real gpu gate"),
                        "tokenizer": {"bos_id": None, "eos_id": None, "pad_id": None, "vocab_size": 32768, "add_special_tokens": False},
                        "loading": {"official_requested_strict": False, "effective_strict": True, "loaded_keys": 225,
                                    "missing_keys": [], "unexpected_keys": [], "attention": "sdpa", "native_model_code_sha256": h("model")},
                        "batch_size": 1, "seed": 42, "num_fewshot": 0, "chat_template": False, "max_length": 2048,
                        "use_cache": False, "logits_cache": False, "limit": limit, "is_full_split": limit is None,
                        "prompt_audit": {"path": str(self.audit), "result_sha256": c.sha256(self.audit / "result.json"),
                                         "records_sha256": c.sha256(self.audit / (task + ".jsonl.gz")), "records_file": (task + ".jsonl.gz"),
                                         "harness_files_sha256": c.hash_json(harness), "task_documents": c.DOCS[task], "task_requests": c.REQUESTS[task]},
                        "native_initialization": {"recurrent_dimension": 1024, "method": "like-init: randn then trunc_normal_",
                                                  "std": .02, "embedding_scale": 1., "per_request_reseed": False,
                                                  "one_initialization_per_scored_request": True},
                        "provenance": {"git_commit": "a" * 40, "source_sha256": source_hashes,
                                       "model": {**prepared, "model_code_sha256": h("model")},
                                       "loaded_model_code_sha256": h("model"), "packages": {"torch": "2.9.0", "transformers": "4.54.1", "accelerate": "1.12.0", "datasets": "4.0.0", "lm_eval": "0.4.9.1"},
                                       "attention": "sdpa", "arxiv": "2610.02185v1",
                                       "precision": "BF16 model; FP32 logits guidance; FP32 hidden blend cast to BF16 before native readout; FP32 log_softmax"},
                        "harness_files": harness, "datasets": {task: {"revision": c.REVISIONS[task],
                                           "raw_source": raw_source, "splits": {c.SPLITS[task]: {"rows": c.DOCS[task], "fingerprint": "fixed-fingerprint"}}}},
                        "samples": {task: n}, "results": result["results"], "request_audit": {}}
            write_json(p / "manifest.json", manifest)
            self.rehash_trace(p)

    @staticmethod
    def rehash_trace(path):
        rows = [json.loads(line) for line in (path / "request_trace.jsonl").read_text().splitlines()]
        write_rows(path / "request_trace.jsonl", rows)
        m = c.read_json(path / "manifest.json")
        m["request_audit"] = {"complete": True, "request_count": len(rows), "planned_requests": len(rows), "truncated_requests": 0,
                              "pairing_sha256": hashlib.sha256(b"".join(c.canonical(r["pairing"]) + b"\n" for r in rows)).hexdigest(),
                              "trace_sha256": c.sha256(path / "request_trace.jsonl")}
        m["evidence_files"] = {p.name: c.sha256(p) for p in path.iterdir() if p.name != "manifest.json"}
        write_json(path / "manifest.json", m)


class CompareParcaeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)
        self.paths = self.fixture.paths

    def compare(self):
        return c.compare_parcae(self.paths, self.fixture.audit, self.fixture.source_root)

    def mutate_manifest(self, fn, arm=1):
        p = self.paths[arm] / "manifest.json"
        x = c.read_json(p)
        fn(x)
        write_json(p, x)

    def mutate_trace(self, fn, rehash=True, arm=1):
        p = self.paths[arm] / "request_trace.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        fn(rows)
        write_rows(p, rows)
        if rehash:
            self.fixture.rehash_trace(p.parent)

    def mutate_samples(self, fn, arm=1):
        p = self.paths[arm] / "samples_sciq.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        fn(rows)
        write_rows(p, rows)

    def test_seven_arm_end_to_end_preserves_negative_results_and_duplicates(self):
        result = self.compare()
        self.assertFalse(result["is_full_split"])
        self.assertEqual(result["n_requests"], 16)  # duplicate strings were not deduplicated
        self.assertEqual(result["metrics"]["acc"]["percent"]["adaptive8"], 25.)
        pair = result["metrics"]["acc"]["paired_vs_baseline8"]["fixed8"]
        self.assertEqual((pair["wins"], pair["losses"], pair["ties"]), (1, 1, 2))
        self.assertAlmostEqual(pair["paired_se_percentage_points"], (1 / 6) ** .5 * 100)
        self.assertEqual(result["metrics"]["acc"]["paired_vs_baseline8"]["adaptive8"]["delta_percentage_points"], -25.)
        self.assertEqual(result["metrics"]["acc"]["auxiliary_paired_vs_baseline4"]["hidden4"]["delta_percentage_points"], 25.)
        self.assertEqual(self.compare()["comparer_source_sha256"], c.sha256(PATH))

    def test_arm_order_does_not_change_pairing(self):
        result = c.compare_parcae(list(reversed(self.paths)), self.fixture.audit, self.fixture.source_root)
        self.assertEqual(result["metrics"], self.compare()["metrics"])

    def test_full_split_is_verified_only_with_all_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, limit=None)
            result = c.compare_parcae(fixture.paths, fixture.audit, fixture.source_root)
            self.assertTrue(result["full_split_count_verified"])
            self.assertEqual((result["n_documents"], result["n_requests"]), (1000, 4000))

    def test_rejects_missing_or_duplicate_arm(self):
        with self.assertRaisesRegex(ValueError, "seven"):
            c.compare_parcae(self.paths[:6], self.fixture.audit, self.fixture.source_root)
        with self.assertRaisesRegex(ValueError, "Duplicate arm"):
            c.compare_parcae(self.paths[:6] + self.paths[:1], self.fixture.audit, self.fixture.source_root)

    def test_all_arms_cannot_claim_subset_is_full(self):
        for i in range(7):
            self.mutate_manifest(lambda m: m.update(limit=None, is_full_split=True), arm=i)
        with self.assertRaisesRegex(ValueError, "sample counts"):
            self.compare()

    def test_rejects_wrong_paper_parameters_even_when_all_arms_agree(self):
        for i in range(7):
            self.mutate_manifest(lambda m: m["guidance"].update(reference_loop=2), arm=i)
        with self.assertRaisesRegex(ValueError, "guidance parameters"):
            self.compare()

    def test_rejects_changed_shots(self):
        self.mutate_manifest(lambda m: m.update(num_fewshot=8))
        with self.assertRaisesRegex(ValueError, "num_fewshot"):
            self.compare()

    def test_rejects_wrong_depth_readout_counts(self):
        self.mutate_trace(lambda rows: rows[0]["observation"]["call_counts"].update(core_layers=31))
        with self.assertRaisesRegex(ValueError, "counts"):
            self.compare()

    def test_rejects_missing_duplicate_or_reordered_request_index(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(index=2)):
            with self.subTest(mutate=mutate):
                original = (self.paths[1] / "request_trace.jsonl").read_bytes()
                self.mutate_trace(mutate)
                with self.assertRaises(ValueError):
                    self.compare()
                (self.paths[1] / "request_trace.jsonl").write_bytes(original)
                self.fixture.rehash_trace(self.paths[1])

    def test_rejects_deduplicated_candidate_even_when_counts_and_hashes_updated(self):
        self.mutate_trace(lambda rows: rows.__delitem__(1))
        with self.assertRaisesRegex(ValueError, "request/sample candidate"):
            self.compare()

    def test_rejects_regenerated_trace_sha_mismatch(self):
        self.mutate_trace(lambda rows: rows[0]["pairing"]["initialization"].update(first_16_values_sha256=h("changed")), rehash=False)
        with self.assertRaisesRegex(ValueError, "hash"):
            self.compare()

    def test_rejects_unpaired_native_state_even_with_self_consistent_hashes(self):
        self.mutate_trace(lambda rows: rows[0]["pairing"]["initialization"].update(first_16_values_sha256=h("changed")))
        with self.assertRaisesRegex(ValueError, "ordered stream"):
            self.compare()

    def test_rejects_rng_reseed_between_requests(self):
        def mutate(rows):
            rows[1]["pairing"]["rng_before"]["model_device"] = h("reseed")
            rows[1]["pairing"]["initialization"]["rng_before"]["model_device"] = h("reseed")
        self.mutate_trace(mutate)
        with self.assertRaisesRegex(ValueError, "continuous RNG"):
            self.compare()

    def test_rejects_initialization_shape(self):
        self.mutate_trace(lambda rows: rows[0]["pairing"]["initialization"].update(shape=[1, 4, 1024]))
        with self.assertRaisesRegex(ValueError, "initialization shape"):
            self.compare()

    def test_rejects_different_token_inputs_even_if_prompt_strings_match(self):
        self.mutate_trace(lambda rows: rows[0]["pairing"].update(input_ids_sha256=h("wrong")))
        with self.assertRaisesRegex(ValueError, "audited request input"):
            self.compare()

    def test_rejects_unregistered_truncation(self):
        self.mutate_trace(lambda rows: rows[0]["pairing"].update(left_truncated_tokens=1))
        with self.assertRaisesRegex(ValueError, "audited request left"):
            self.compare()

    def test_rejects_trace_response_not_equal_sample(self):
        self.mutate_trace(lambda rows: rows[0].update(loglikelihood=-999.))
        with self.assertRaisesRegex(ValueError, "likelihood response"):
            self.compare()

    def test_rejects_wrong_raw_document_or_prompt(self):
        self.mutate_samples(lambda rows: rows[0]["doc"].update(question="wrong"))
        with self.assertRaisesRegex(ValueError, "audited document"):
            self.compare()

    def test_rejects_sample_drop_or_shuffle(self):
        self.mutate_samples(lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, "ordered complete"):
            self.compare()

    def test_rejects_wrong_or_nonbinary_sample_metric(self):
        self.mutate_samples(lambda rows: rows[0].update(acc=0.5))
        with self.assertRaisesRegex(ValueError, "Non-binary"):
            self.compare()

    def test_rejects_aggregate_change(self):
        p = self.paths[1] / "results.json"
        x = c.read_json(p)
        x["results"]["sciq"]["acc,none"] = .75
        write_json(p, x)
        self.mutate_manifest(lambda m: m.update(results=x["results"]))
        with self.assertRaisesRegex(ValueError, "Sample mean"):
            self.compare()

    def test_rejects_source_or_harness_provenance_change(self):
        self.mutate_manifest(lambda m: m["provenance"]["source_sha256"].update(**{"extra.py": h("other")}))
        with self.assertRaisesRegex(ValueError, "frozen source"):
            self.compare()

    def test_rejects_prompt_audit_bytes_change(self):
        with (self.fixture.audit / "sciq.jsonl.gz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "records SHA"):
            self.compare()

    def test_exact_integer_moment_se_and_singleton(self):
        left = {i: {"metrics": {"acc": value}} for i, value in enumerate([0] * 77 + [1] * 24 + [0] * 1071)}
        right = {i: {"metrics": {"acc": value}} for i, value in enumerate([1] * 77 + [0] * 24 + [0] * 1071)}
        self.assertEqual(c.paired_metrics(left, right, "acc")["paired_se_percentage_points"], 0.8476241942667028)
        self.assertIsNone(c.paired_metrics({0: left[0]}, {0: right[0]}, "acc")["paired_se_percentage_points"])

    def test_rejects_changed_frozen_source_bytes(self):
        (self.fixture.source_root / "scripts/evaluate_parcae.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError, "frozen source"):
            self.compare()

    def test_rejects_wrong_protocol_or_model_or_load_gate(self):
        path = self.paths[1] / "manifest.json"
        original = path.read_bytes()
        for mutate in (lambda m: m.update(protocol_sha256=h("wrong")),
                       lambda m: m["loading"].update(effective_strict=False),
                       lambda m: m["provenance"]["model"]["files"]["config.json"].update(sha256=h("wrong")),
                       lambda m: m.update(execution_device_type="cpu"),
                       lambda m: m.update(gpu_smoke_sha256=h("different GPU gate"))):
            with self.subTest(mutate=mutate):
                self.mutate_manifest(mutate)
                with self.assertRaises(ValueError):
                    self.compare()
                path.write_bytes(original)

    def test_rejects_boolean_depth_or_count(self):
        self.mutate_manifest(lambda m: m["guidance"].update(reference_loop=True))
        with self.assertRaisesRegex(ValueError, "guidance reference_loop"):
            self.compare()

    def test_rejects_evidence_file_hash_mismatch(self):
        self.mutate_manifest(lambda m: m["evidence_files"].update(**{"results.json": h("wrong")}))
        with self.assertRaisesRegex(ValueError, "evidence file hashes"):
            self.compare()

    def test_rejects_incorrect_truncation_summary(self):
        self.mutate_manifest(lambda m: m["request_audit"].update(truncated_requests=1))
        with self.assertRaisesRegex(ValueError, "truncation count"):
            self.compare()

    def test_actual_request_audit_writer_contract(self):
        # Import the evaluator's stdlib audit module, without loading a model,
        # torch, a tokenizer, or the evaluation harness.
        helper_path = PATH.parents[1] / "src/loopcd_repro/parcae_mc.py"
        spec = importlib.util.spec_from_file_location("parcae_mc_test_contract", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for path in self.paths:
            original = [json.loads(line) for line in (path / "request_trace.jsonl").read_text().splitlines()]
            with (path / "request_trace.jsonl").open("w") as stream:
                audit = helper.RequestAudit(stream)
                audit.planned = []
                for row in original:
                    pair = row["pairing"]
                    meta = helper.audited_pairing(self.fixture.records[(pair["task"], pair["doc_id"], pair["candidate"])])
                    self.assertEqual(meta, c._pairing_fields(self.fixture.records[(pair["task"], pair["doc_id"], pair["candidate"])]))
                    audit.planned.append(meta)
                for row, meta in zip(original, audit.planned):
                    pair = row["pairing"]
                    audit.record(meta, pair["rng_before"], pair["rng_after"], pair["initialization"],
                                 row["observation"], row["loglikelihood"], row["is_greedy"])
            m = c.read_json(path / "manifest.json")
            m["request_audit"] = audit.summary(require_complete=True)
            m["evidence_files"]["request_trace.jsonl"] = c.sha256(path / "request_trace.jsonl")
            write_json(path / "manifest.json", m)
        self.assertTrue(self.compare()["all_seven_arms_verified"])

    def test_cli_fresh_only(self):
        out = Path(self.tmp.name) / "comparison.json"
        cmd = [sys.executable, str(PATH), "--runs", *map(str, self.paths), "--prompt-audit", str(self.fixture.audit), "--source-root", str(self.fixture.source_root), "--output", str(out)]
        first = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        old = out.read_bytes()
        second = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(out.read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
