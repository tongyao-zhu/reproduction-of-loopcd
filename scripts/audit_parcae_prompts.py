"""Offline full-split Parcae prompt/token audit. Never loads a model or scores it."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SHOTS = {"sciq": 0, "piqa": 0, "arc_challenge": 25, "arc_easy": 0,
         "hellaswag": 0, "winogrande": 0, "mmlu": 5}
EXPECTED_DOCS = {"sciq": 1000, "piqa": 1838, "arc_challenge": 1172,
                 "arc_easy": 2376, "hellaswag": 10042, "winogrande": 1267, "mmlu": 14042}
HARNESS_COMMIT = "7ddb2b1e4bc819292ff56f334c572d1fc77dec28"


def hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_views(context, continuation, encode, max_length=2048):
    """Mirror official LMEvalModel and causal TemplateLM boundaries separately.

    HF path has no implicit EOT/BOS fallback: the real native wrapper has none.
    Both views expose missing predictors instead of silently accepting them.
    """
    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("Request must contain two strings")
    raw_ctx, raw_full = list(encode(context)), list(encode(context + continuation))
    native_context = context
    retried = raw_ctx != raw_full[:len(raw_ctx)]
    if retried:
        native_context = context.rstrip()
    nc, nf = list(encode(native_context)), list(encode(native_context + continuation))
    nt = nf[len(nc):]
    native_overflow = max(0, len(nf) - max_length)
    native_input = nf[native_overflow:]
    native_context_kept = nc[native_overflow:]
    native_start = len(native_context_kept) - 1
    native_positions = list(range(native_start, native_start + len(nt)))
    native_predictors_valid = bool(nt) and all(0 <= pos < len(native_input) - 1 for pos in native_positions)
    native_labels_match = (native_predictors_valid and
                           [native_input[pos + 1] for pos in native_positions] == nt)

    # TemplateLM._encode_pair moves all trailing whitespace to continuation.
    spaces = len(context) - len(context.rstrip())
    hc_text = context[:-spaces] if spaces else context
    ht_text = context[-spaces:] + continuation if spaces else continuation
    hc, hf = list(encode(hc_text)), list(encode(hc_text + ht_text))
    ht = hf[len(hc):]
    joined = hc + ht
    hf_overflow = max(0, len(joined) - (max_length + 1))
    hf_input = joined[-(max_length + 1):][:-1]
    hf_start = len(hf_input) - len(ht)
    hf_positions = list(range(hf_start, len(hf_input)))
    hf_predictors_valid = bool(hc) and bool(ht) and len(ht) <= len(hf_input) and len(ht) <= max_length
    native_effective = native_input[:-1]
    return {
        "context_chars": len(context), "continuation_chars": len(continuation),
        "empty_context": not context, "empty_continuation": not continuation,
        "trailing_context_whitespace": spaces,
        "raw_context_tokens": len(raw_ctx), "raw_full_tokens": len(raw_full),
        "native": {"context_tokens": len(nc), "full_tokens": len(nf), "continuation_tokens": len(nt),
                   "retry_rstrip": retried, "boundary_prefix_valid": nc == nf[:len(nc)],
                   "full_text_changed": native_context + continuation != context + continuation,
                   "left_dropped": native_overflow, "input_tokens": len(native_input),
                   "context_tokens_kept": len(native_context_kept),
                   "all_labels_have_predictors": native_predictors_valid,
                   "scored_labels_match_input": native_labels_match,
                   "context_sha256": hash_json(nc), "full_sha256": hash_json(nf),
                   "continuation_sha256": hash_json(nt), "input_sha256": hash_json(native_input),
                   "scoring_positions": [native_start, native_start + len(nt)]},
        "hf": {"context_tokens": len(hc), "full_tokens": len(joined), "continuation_tokens": len(ht),
               "boundary_prefix_valid": hc == hf[:len(hc)], "left_dropped": hf_overflow,
               "input_tokens": len(hf_input), "all_labels_have_predictors": hf_predictors_valid,
               "empty_context_requires_undefined_eot": not hc,
               "context_sha256": hash_json(hc), "full_sha256": hash_json(joined),
               "continuation_sha256": hash_json(ht), "input_sha256": hash_json(hf_input),
               "scoring_positions": [hf_start, len(hf_input)]},
        "comparison": {"continuation_ids_equal": nt == ht,
                       "full_tokenization_equal": nf == joined,
                       "actual_model_inputs_equal": native_input == hf_input,
                       "effective_predictor_inputs_equal": native_effective == hf_input,
                       "hf_retains_one_extra_context_token": (
                           nt == ht and native_overflow > 0 and len(hf_input) == len(native_effective) + 1
                           and hf_input[1:] == native_effective)},
    }


def summarize(records):
    records = list(records)
    counter = Counter()
    for item in records:
        for flag in ("empty_context", "empty_continuation"):
            counter[flag] += bool(item[flag])
        counter["trailing_context_whitespace"] += bool(item["trailing_context_whitespace"])
        for method in ("native", "hf"):
            view = item[method]
            counter[method + "_truncated"] += view["left_dropped"] > 0
            counter[method + "_invalid_boundary"] += not view["boundary_prefix_valid"]
            counter[method + "_unscoreable_labels"] += not view["all_labels_have_predictors"]
        counter["native_changed_text"] += item["native"]["full_text_changed"]
        counter["native_label_input_mismatch"] += not item["native"]["scored_labels_match_input"]
        for flag, value in item["comparison"].items():
            counter[flag] += bool(value)
    lengths = sorted(item["raw_full_tokens"] for item in records)
    return {"requests": len(records), "counts": dict(counter),
            "full_token_lengths": {"min": min(lengths), "max": max(lengths),
                                   "p50": lengths[(len(lengths) - 1) // 2],
                                   "p95": lengths[int((len(lengths) - 1) * .95)],
                                   "p99": lengths[int((len(lengths) - 1) * .99)]},
            "max_continuation_tokens": {name: max(item[name]["continuation_tokens"] for item in records)
                                        for name in ("native", "hf")},
            "max_left_dropped": {name: max(item[name]["left_dropped"] for item in records)
                                  for name in ("native", "hf")}}


def git(root, *args):
    return subprocess.check_output(["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})


def verify_harness(reference, installed_root):
    from prepare_parcae import stream_fingerprint
    commit = git(reference, "rev-parse", "HEAD").decode().strip()
    dirty = git(reference, "status", "--porcelain=v1", "--untracked-files=no").decode()
    if commit != HARNESS_COMMIT or dirty:
        raise ValueError("Harness reference must be the clean previously used commit")
    entries = git(reference, "ls-tree", "-r", "-z", "HEAD", "lm_eval").split(b"\0")
    records = {}
    for entry in filter(None, entries):
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        name = os.fsdecode(raw_name)
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError("Only regular harness source files are supported")
        source, installed = reference / name, installed_root / Path(name).relative_to("lm_eval")
        observed = stream_fingerprint(source, git_blob=True)
        expected = observed["sha256"]
        if observed["git_blob_sha1"] != object_id or sha256(installed) != expected:
            raise ValueError(f"Installed harness differs from pinned source: {name}")
        records[name] = {"sha256": expected, "bytes": source.stat().st_size}
    installed_names = {"lm_eval/" + path.relative_to(installed_root).as_posix()
                       for path in installed_root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if installed_names != set(records):
        raise ValueError(f"Installed harness file set differs: {sorted(installed_names.symmetric_difference(records))[:12]}")
    return {"commit": commit, "tracked_dirty": dirty, "reference": str(reference),
            "installed_root": str(installed_root), "files": records,
            "files_sha256": hash_json(records), "version": importlib.metadata.version("lm_eval")}


def piqa_spec(registry, manager, cache_dir, hub_cache):
    """Pinned PIQA uses piqa_<split>.parquet, unlike the generic shard names."""
    entry = registry["tasks"]["piqa"]
    snapshot = hub_cache / ("datasets--" + entry["dataset_path"].replace("/", "--")) / "snapshots" / entry["revision"]
    config = copy.deepcopy(manager._get_config("piqa"))
    if config["dataset_path"] != entry["dataset_path"]:
        raise ValueError("PIQA native dataset identity changed")
    data_files, raw_files = {}, []
    for split in ("train", "validation", "test"):
        path = snapshot / ("piqa_" + split + ".parquet")
        data_files[split] = [str(path.absolute())]
        raw_files.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    config["dataset_path"] = "parquet"
    config["dataset_kwargs"] = {"data_files": data_files, "cache_dir": str(cache_dir.resolve())}
    config["metadata"] = {**config.get("metadata", {}), "loopcd_dataset": {
        "loader": "local_parquet_from_pinned_hub_snapshot", "repo_id": entry["dataset_path"],
        "revision": entry["revision"], "config": None, "raw_files": raw_files,
        "note": "PIQA raw filename-to-split mapping only; native task templates and processing preserved."}}
    return config


def audit(args, *, profile=None):
    # Explicit profiles are limited to the independently pinned 370M Figure1b row.
    if profile is None:
        from prepare_parcae import verify_prepared
        shots, expected_docs = SHOTS, EXPECTED_DOCS
        source_names = ("audit_parcae_prompts.py", "check_mc_data.py", "prepare_parcae.py")
    elif profile == "parcae370_arc_challenge_v1":
        from prepare_parcae370 import verify_prepared
        shots, expected_docs = {"arc_challenge": 25}, {"arc_challenge": 1172}
        source_names = ("audit_parcae_prompts.py", "audit_parcae370_prompts.py",
                        "check_mc_data.py", "prepare_parcae370.py")
    else:
        raise ValueError("Unknown pinned Parcae audit profile")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not sys.dont_write_bytecode:
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1")
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {"status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Full-split prompt/token audit only; no model load, inference, scores, or protocol changes.",
              "seed": 42, "shots": shots, "max_length": 2048, "cpu_only": True,
              "script_sha256": sha256(Path(__file__)), "registry_sha256": sha256(args.registry), "tasks": {}}
    report["project_source"] = {
        "git_commit_observed": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "files": {name: sha256(ROOT / "scripts" / name) for name in source_names}}
    write = lambda: (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write()
    try:
        from check_mc_data import build_local_task_spec, iter_leaf_tasks, validate_dataset_pin
        before = verify_prepared(args.model)
        report["model_manifest_sha256"] = sha256(args.model / "model_provenance.json")
        report["model_identity"] = {"repo_id": before["repo_id"], "revision": before["revision"],
                                    "files": {name: record["sha256"] for name, record in before["files"].items()},
                                    "source_tree_sha1": before["source"]["git_tree_sha1"]}
        sys.path.insert(0, str(args.model / "source"))
        import torch
        import numpy as np
        import lm_eval
        from lm_eval.tasks import TaskManager
        from parcae_lm.tokenizer import Tokenizer
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(4)
        random.seed(42)
        np.random.seed(42)
        report["harness"] = verify_harness(args.harness_reference, Path(lm_eval.__file__).parent)
        (args.output / "harness_files.json").write_text(json.dumps(report["harness"].pop("files"), indent=2) + "\n")
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ("lm_eval", "datasets", "accelerate", "torch", "transformers", "tokenizers")}
        if report["packages"] != {"lm_eval": "0.4.9.1", "datasets": "4.0.0", "accelerate": "1.12.0",
                                   "torch": "2.9.0", "transformers": "4.54.1", "tokenizers": "0.21.4"}:
            raise ValueError("Unexpected runtime versions")
        tokenizer = Tokenizer.from_directory(args.model)
        report["tokenizer"] = {"bos_id": tokenizer.bos_id, "eos_id": tokenizer.eos_id,
                               "pad_id": tokenizer.pad_id, "vocab_size": tokenizer.vocab_size,
                               "encode": "native encode(return_tensors=False), no added BOS/EOS/PAD"}
        if any(report["tokenizer"][key] is not None for key in ("bos_id", "eos_id", "pad_id")):
            raise ValueError("Tokenizer native special-token properties changed")
        @lru_cache(maxsize=4096)
        def encode(text):
            ids = tuple(tokenizer.encode(text, return_tensors=False))
            if any(type(token) is not int or not 0 <= token < tokenizer.vocab_size for token in ids):
                raise ValueError("Tokenizer returned an invalid token ID")
            return ids
        registry = json.loads(args.registry.read_text())
        manager = TaskManager()
        for name, shot in shots.items():
            entry = registry["tasks"][name]
            spec = (piqa_spec(registry, manager, args.dataset_cache, args.hub_cache) if name == "piqa"
                    else build_local_task_spec(name, registry, manager, args.dataset_cache, args.hub_cache))
            tree = manager.load_task_or_group([spec])
            task_records, leaves, examples = [], {}, []
            total_docs = 0
            records_path = args.output / (name + ".jsonl.gz")
            with gzip.open(records_path, "xt", encoding="utf-8") as stream:
                for leaf_name, task in iter_leaf_tasks(tree):
                    validate_dataset_pin(task, entry)
                    if (task.config.test_split or task.config.validation_split) != entry["eval_split"]:
                        raise ValueError("Unexpected evaluation split")
                    task.set_config("num_fewshot", shot)
                    task.set_fewshot_seed(42)
                    docs = task.eval_docs
                    seen_prompts, seen_docs = defaultdict(list), defaultdict(list)
                    leaf_records, doc_counts, overlap, duplicate_candidates = [], [], [], []
                    document_issues = Counter()
                    sampled = []
                    if shot:
                        native_sample = task.sampler.sample
                        def observe_sample(n, original=native_sample):
                            values = original(n)
                            sampled[:] = values
                            return values
                        task.sampler.sample = observe_sample
                    for doc_id, doc in enumerate(docs):
                        sampled.clear()
                        context = task.fewshot_context(doc, num_fewshot=shot, apply_chat_template=False,
                                                       gen_prefix=task.doc_to_prefix(doc))
                        requests = task.construct_requests(doc=doc, ctx=context,
                                    metadata=(task.config.task, doc_id, task.config.repeats), apply_chat_template=False)
                        if not isinstance(requests, list):
                            requests = [requests]
                        selected = [item for item in sampled if item != doc][:shot]
                        if len(selected) != shot:
                            raise ValueError(f"{leaf_name}/{doc_id}: wrong observed shot count")
                        current_question = task.doc_to_text(doc)
                        matching = [i for i, item in enumerate(selected) if task.doc_to_text(item) == current_question]
                        if matching:
                            overlap.append({"doc_id": doc_id, "selected_fewshot_positions": matching})
                        seen_docs[hash_json(doc)].append(doc_id)
                        seen_prompts[hash_json([list(request.args) for request in requests])].append(doc_id)
                        doc_counts.append(len(requests))
                        full_hashes = []
                        doc_views = []
                        for request in requests:
                            if request.request_type != "loglikelihood" or len(request.args) != 2:
                                raise ValueError("Only two-string loglikelihood candidates expected")
                            context_text, continuation = request.args
                            view = pair_views(context_text, continuation, encode)
                            record = {"task": name, "leaf": leaf_name, "doc_id": doc_id,
                                      "candidate": request.idx, "doc_sha256": hash_json(doc),
                                      "context_sha256": text_hash(context_text), "continuation_sha256": text_hash(continuation),
                                      "arguments_sha256": hash_json(list(request.args)),
                                      "fewshot_docs_sha256": [hash_json(item) for item in selected], **view}
                            full_hashes.append(view["hf"]["full_sha256"])
                            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            leaf_records.append(view)
                            doc_views.append(view)
                            if (len(examples) < 12 and (view["native"]["left_dropped"] or
                                    not view["native"]["boundary_prefix_valid"] or view["native"]["retry_rstrip"])):
                                examples.append({**record, "context_tail": context_text[-250:], "continuation": continuation})
                        if len(set(full_hashes)) != len(full_hashes):
                            duplicate_candidates.append({"doc_id": doc_id, "candidate_full_token_hashes": full_hashes})
                        for method in ("native", "hf"):
                            document_issues[method + "_truncated"] += any(item[method]["left_dropped"] > 0 for item in doc_views)
                            document_issues[method + "_invalid_boundary"] += any(not item[method]["boundary_prefix_valid"] for item in doc_views)
                            document_issues[method + "_unscoreable_labels"] += any(not item[method]["all_labels_have_predictors"] for item in doc_views)
                    leaf_summary = summarize(leaf_records)
                    leaf_summary.update({"documents": len(docs), "candidate_counts": dict(Counter(doc_counts)),
                                         "identical_document_groups": [ids for ids in seen_docs.values() if len(ids) > 1],
                                         "identical_request_groups": [ids for ids in seen_prompts.values() if len(ids) > 1],
                                         "sampled_fewshot_exact_question_overlaps": overlap,
                                         "duplicate_tokenized_candidates": duplicate_candidates,
                                         "document_issue_counts": dict(document_issues),
                                         "dataset_source": task.config.metadata["loopcd_dataset"],
                                         "fewshot_sampler": type(task.sampler).__name__ if shot else None,
                                         "native_metrics": [item["metric"] for item in task.config.metric_list],
                                         "multiple_input": task.multiple_input})
                    leaves[leaf_name] = leaf_summary
                    task_records.extend(leaf_records)
                    total_docs += len(docs)
                    print(json.dumps({"task": name, "leaf": leaf_name, "documents": len(docs),
                                      "requests": len(leaf_records)}), flush=True)
            if total_docs != expected_docs[name] or len(leaves) != entry.get("expected_leaf_tasks", 1):
                raise ValueError(f"Incomplete split: {name}: {total_docs}, {len(leaves)}")
            report["tasks"][name] = {**summarize(task_records), "documents": total_docs,
                                      "leaves": leaves, "examples": examples,
                                      "document_issue_counts": dict(sum((Counter(leaf["document_issue_counts"])
                                                                         for leaf in leaves.values()), Counter())),
                                      "records": {"file": records_path.name, "sha256": sha256(records_path)}}
            write()
        after = verify_prepared(args.model)
        if before != after or report["model_manifest_sha256"] != sha256(args.model / "model_provenance.json"):
            raise ValueError("Prepared model changed during audit")
        report["model_full_hash_verified_before_and_after"] = True
        report["cuda_initialized"] = torch.cuda.is_initialized()
        assert not report["cuda_initialized"]
        report["registered_hf_policy_ready"] = all(
            task["counts"]["hf_invalid_boundary"] == 0 and task["counts"]["hf_unscoreable_labels"] == 0
            and task["counts"]["empty_context"] == 0 and task["counts"]["empty_continuation"] == 0
            for task in report["tasks"].values())
        report["total_documents"] = sum(task["documents"] for task in report["tasks"].values())
        report["total_requests"] = sum(task["requests"] for task in report["tasks"].values())
        if profile is not None:
            if report["total_documents"] != 1172 or report["total_requests"] != 4687:
                raise ValueError("Incomplete 370M ARC-C audit")
            report["profile"] = profile
        report["status"] = "PASS"
        report["elapsed_seconds"] = time.time() - started
        report["interpretation"] = [
            "PASS means audit integrity and full split coverage; it does not approve either truncation policy.",
            "Official input includes the final candidate token (unused to score it); HF-style input removes it.",
            "When full length exceeds 2048, HF-style 2049-then[:-1] can retain one additional leading context token.",
            "Different model input lengths can alter native random state initialization even if causal predictor tokens match.",
            "No implicit BOS/EOT/PAD is supplied. Any empty context or unscoreable candidate requires a protocol decision.",
            "Fewshot overlap compares exact doc_to_text strings in actually sampled demonstrations, not semantic similarity.",
            "MMLU native candidates are answer letters; equivalence to unpublished Apple task templates is unresolved.",
            "No documents or candidates were dropped; no shot count or metric was changed based on results.",
        ]
        write()
        return report
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.time() - started)
        write()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--harness-reference", required=True, type=Path)
    parser.add_argument("--hub-cache", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--registry", default=ROOT / "configs/mc_datasets.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
