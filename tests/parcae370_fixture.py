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

PATH = Path(__file__).resolve().parents[1] / "scripts/compare_parcae370.py"
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
    def __init__(self, root, limit=4, task="arc_challenge", arms=None):
        self.root, self.limit = Path(root), limit
        self.source_root = self.root / "release"
        sources = ["scripts/compare_parcae370.py", "scripts/evaluate_parcae370.py", "scripts/prepare_parcae370.py",
                   "scripts/audit_parcae_prompts.py", "src/loopcd_repro/parcae.py", "src/loopcd_repro/parcae370_mc.py",
                   "src/loopcd_repro/guidance.py", "src/loopcd_repro/runtime.py", "configs/parcae_370m_arc.json", "configs/mc_datasets.json"]
        for name in sources:
            destination = self.source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((PATH.parents[1] / name).read_bytes())
        (self.source_root / "docs").mkdir()
        (self.source_root / "docs/parcae370_arc_protocol.md").write_text("Registered fixture protocol\n")
        source_hashes = {name: c.sha256(self.source_root / name) for name in sources}
        paper_config = c.read_json(self.source_root / "configs/parcae_370m_arc.json")
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
            arguments = [[f"Question {doc_id}?", " same" if candidate < 2 else f" answer{candidate}"] for candidate in range(3 if doc_id == c.DOCS[task]-1 else 4)]
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
        report = {"profile": "parcae370_arc_challenge_v1", "status": "PASS", "registered_hf_policy_ready": True, "seed": 42, "shots": c.SHOTS,
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
                row["filtered_resps"] = [[-float(doc_id + candidate + 1), candidate == 0] for candidate in range(len(row["arguments"]))]
                row["resps"] = [[r] for r in row["filtered_resps"]]
                values = {"baseline8": [1, 0, 0, 1], "fixed8": [1, 1, 0, 0], "adaptive8": [0, 0, 0, 1],
                          "hidden8": [1, 1, 1, 1], "baseline4": [0, 0, 0, 1], "fixed4": [0, 1, 0, 1],
                          "hidden4": [1, 0, 0, 1]}[arm]
                row["acc"] = values[doc_id % 4]
                row["acc_norm"] = (doc_id % 2)
                samples.append(row)
                for candidate in range(len(row["arguments"])):
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
                                  "observation": {"guidance": observation, "call_counts": {"initializations": 1, "prelude": 4, "core_layers": 4 * guidance["total_loops"], "projection": readouts, "coda": 4 * readouts, "norm": readouts, "head": readouts}},
                                  "loglikelihood": row["filtered_resps"][candidate][0], "is_greedy": row["filtered_resps"][candidate][1]})
            write_rows(p / ("samples_" + task + ".jsonl"), samples)
            write_rows(p / "request_trace.jsonl", trace)
            result = {"results": {task: {"acc,none": sum(x["acc"] for x in samples) / n,
                                            "acc_norm,none": sum(x["acc_norm"] for x in samples) / n}},
                      "configs": {task: {"test_split": c.SPLITS[task], "num_fewshot": 25}},
                      "n-shot": {task: 25}, "n-samples": {task: {"original": c.DOCS[task], "effective": n}},
                      "versions": {task: 1}, "higher_is_better": {task: {"acc": True, "acc_norm": True}}}
            write_json(p / "results.json", result)
            manifest = {"status": "completed", "task": task, "arm": arm, "guidance": guidance,
                        "protocol": "parcae370_arc_v1", "paper_config": paper_config,
                        "paper_config_sha256": source_hashes["configs/parcae_370m_arc.json"],
                        "registry_sha256": source_hashes["configs/mc_datasets.json"],
                        "protocol_sha256": c.sha256(self.source_root / "docs/parcae370_arc_protocol.md"),
                        "harness_commit": c.HARNESS_COMMIT, "harness_files_sha256": c.hash_json(harness),
                        "add_bos": False, "truncation": "left_keep_2049_then_remove_last_token",
                        "request_order": "native_harness_order_no_sort_no_dedup",
                        "seed_reset": "once_after_model_loading_before_harness_requests; no per-request reseeding",
                        "model_full_hash_verified_before_and_after": True, "execution_device_type": "cuda",
                        "gpu_smoke_sha256": h("real gpu gate"),
                        "tokenizer": {"bos_id": None, "eos_id": None, "pad_id": None, "vocab_size": 32768, "add_special_tokens": False},
                        "loading": {"official_requested_strict": False, "effective_strict": True, "loaded_keys": 117,
                                    "missing_keys": [], "unexpected_keys": [], "attention": "sdpa", "native_model_code_sha256": h("model")},
                        "batch_size": 1, "seed": 42, "num_fewshot": 25, "chat_template": False, "max_length": 2048,
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


