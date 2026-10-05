"""Strict, standard-library-only comparison of all seven registered Parcae MC arms.

Reads evidence only; never imports model, tokenizer, harness, or solution code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS = ("baseline8", "fixed8", "adaptive8", "hidden8", "baseline4", "fixed4", "hidden4")
SHOTS = {"sciq": 0, "piqa": 0, "arc_challenge": 25, "arc_easy": 0,
         "hellaswag": 0, "winogrande": 0, "mmlu": 5}
DOCS = {"sciq": 1000, "piqa": 1838, "arc_challenge": 1172, "arc_easy": 2376,
        "hellaswag": 10042, "winogrande": 1267, "mmlu": 14042}
REQUESTS = {"sciq": 4000, "piqa": 3676, "arc_challenge": 4687, "arc_easy": 9501,
            "hellaswag": 40168, "winogrande": 2534, "mmlu": 56168}
SPLITS = {name: "validation" if name in ("piqa", "hellaswag", "winogrande") else "test" for name in SHOTS}
REVISIONS = {"sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
             "piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
             "arc_challenge": "210d026faf9955653af8916fad021475a3f00453",
             "arc_easy": "210d026faf9955653af8916fad021475a3f00453",
             "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
             "winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
             "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe"}
MODEL_REPO = "SandyResearch/parcae-1.3b"
MODEL_REVISION = "517f8e6add927e3d73e15beec13a5b500537b9cc"
HARNESS_COMMIT = "7ddb2b1e4bc819292ff56f334c572d1fc77dec28"
OFFICIAL_SOURCE = "69284c13746e849104f738d6d1a347b1f457df76"
MODEL_FILES = {"pytorch_model.bin": "1e5d08f8a23820900ed31180b17625f3ea0c272f795f79975c88dfba5a21a56f",
               "config.json": "6e6b3ac71f6b5e24ddc9499649c02ffdb8970a8a75d9086ed76c785e4b8bffa0",
               "tokenizer.json": "e0021e26057088d68047dbe6e77e2ba1c9fe9ae45bae3df9d67d48c82405ea77"}
SOURCE_TREE = "1e284593583eaf4403dad3bd5054ea81eb0cb7ac"
GUIDANCE = {arm: {"mode": arm[:-1], "total_loops": int(arm[-1]), "reference_loop": 1,
                  "omega": 1.0 if arm == "hidden8" else .75 if arm == "hidden4" else .5,
                  "omega_cap": 1.0} for arm in ARMS}
METRICS = ("acc", "acc_norm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right, name):
    require(left == right, f"Mismatched {name}")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hash_json(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(value):
    require(isinstance(value, str), "Expected string for text hash")
    return hashlib.sha256(value.encode()).hexdigest()


def valid_hash(value, name, length=64):
    require(isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value), f"Invalid {name} hash")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite {name}")
    return value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_bytes())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_prompt_audit(directory, task):
    """Validate the full independently constructed corpus, retaining every identity."""
    directory = Path(directory).resolve()
    report = read_json(directory / "result.json")
    require(report["status"] == "PASS" and report["registered_hf_policy_ready"] is True,
            "Prompt audit has not passed")
    same(report["seed"], 42, "prompt audit seed")
    same(report["shots"], SHOTS, "prompt audit shots")
    same(report["max_length"], 2048, "prompt audit context limit")
    same(report["total_documents"], sum(DOCS.values()), "prompt audit full document count")
    same(report["total_requests"], sum(REQUESTS.values()), "prompt audit full candidate count")
    same(sorted(report["tasks"]), sorted(SHOTS), "prompt audit task set")
    same(report["model_identity"], {"repo_id": MODEL_REPO, "revision": MODEL_REVISION,
         "files": MODEL_FILES, "source_tree_sha1": SOURCE_TREE}, "prompt audit model identity")
    require(report["model_full_hash_verified_before_and_after"] is True and report["cuda_initialized"] is False,
            "Prompt audit integrity/CPU gate missing")
    same(report["harness"]["commit"], HARNESS_COMMIT, "prompt audit harness commit")
    harness_files = read_json(directory / "harness_files.json")
    require(bool(harness_files), "Empty harness file evidence")
    same(hash_json(harness_files), report["harness"]["files_sha256"], "harness file-map SHA")
    for record in harness_files.values():
        valid_hash(record["sha256"], "harness source")
        integer(record["bytes"], "harness source bytes")
    task_report = report["tasks"][task]
    same(task_report["documents"], DOCS[task], "prompt audit task documents")
    same(task_report["requests"], REQUESTS[task], "prompt audit task requests")
    same(task_report["records"]["file"], task + ".jsonl.gz", "prompt audit records filename")
    records_path = directory / task_report["records"]["file"]
    same(sha256(records_path), task_report["records"]["sha256"], "prompt audit records SHA")
    leaves = task_report["leaves"]
    require(len(leaves) == (57 if task == "mmlu" else 1), "Incomplete prompt audit leaves")
    require(all(k.startswith("mmlu_") for k in leaves) if task == "mmlu" else list(leaves) == [task], "Invalid prompt audit leaf names")
    records, ids = {}, {leaf: {} for leaf in leaves}
    with gzip.open(records_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank prompt audit row")
            row = json.loads(line)
            same(row["task"], task, "audited task")
            leaf = row["leaf"]
            require(leaf in leaves, "Unexpected audited leaf")
            doc_id, candidate = integer(row["doc_id"], "audited doc ID"), integer(row["candidate"], "audited candidate")
            identity = (leaf, doc_id, candidate)
            require(identity not in records, "Duplicate audited candidate identity")
            hf = row["hf"]
            require(hf["boundary_prefix_valid"] is True and hf["all_labels_have_predictors"] is True,
                    "Invalid audited token boundary")
            require(row["empty_context"] is False and row["empty_continuation"] is False,
                    "Empty audited request")
            records[identity] = row
            ids[leaf].setdefault(doc_id, []).append(candidate)
    same(len(records), REQUESTS[task], "audited candidate coverage")
    for leaf, documents in ids.items():
        original = integer(leaves[leaf]["documents"], "audited leaf document count", 1)
        same(sorted(documents), list(range(original)), "audited document coverage")
        for candidates in documents.values():
            same(sorted(candidates), list(range(len(candidates))), "audited candidate indices")
        same(sum(map(len, documents.values())), leaves[leaf]["requests"], "audited leaf candidate count")
    same(sum(len(v) for v in ids.values()), DOCS[task], "audited task document coverage")
    return {"directory": directory, "report": report, "task": task, "records": records, "leaves": leaves,
            "harness_files": harness_files, "result_sha256": sha256(directory / "result.json"),
            "records_sha256": task_report["records"]["sha256"]}


def _aggregate(value, values, name):
    finite(value, name)
    expected = float(Fraction(sum(values), len(values)))
    # Weighted harness groups may differ by a few final floating-point ulps.
    require(abs(value - expected) <= 8 * math.ulp(expected), f"Sample mean disagrees with {name}")


def _rng(value):
    require(isinstance(value, dict) and set(value) == {"cpu", "model_device"}, "Expected CPU/CUDA RNG state hashes")
    for key, h in value.items():
        valid_hash(h, "RNG " + key)


def _pairing_fields(audited):
    hf = audited["hf"]
    return {"task": audited["leaf"], "doc_id": audited["doc_id"], "candidate": audited["candidate"],
            "arguments_sha256": audited["arguments_sha256"], "context_sha256": audited["context_sha256"],
            "continuation_text_sha256": audited["continuation_sha256"],
            "context_tokens_sha256": hf["context_sha256"], "full_tokens_sha256": hf["full_sha256"],
            "input_ids_sha256": hf["input_sha256"], "continuation_ids_sha256": hf["continuation_sha256"],
            "original_context_length": hf["context_tokens"], "original_full_length": hf["full_tokens"],
            "input_length": hf["input_tokens"], "continuation_length": hf["continuation_tokens"],
            "left_truncated_tokens": hf["left_dropped"]}


def _trace(directory, manifest, context, expected_responses):
    audit = manifest["request_audit"]
    require(audit["complete"] is True, "Request audit incomplete")
    expected_count = len(expected_responses)
    same(integer(audit["request_count"], "request count", 1), expected_count, "request/sample candidate count")
    same(audit["planned_requests"], expected_count, "planned request count")
    ph, th, raw_sha, seen = hashlib.sha256(), hashlib.sha256(), hashlib.sha256(), set()
    previous_after = None
    truncated = 0
    max_input = 0
    init_config = manifest["native_initialization"]
    dimensions = integer(init_config["recurrent_dimension"], "recurrent dimension", 1)
    path = directory / "request_trace.jsonl"
    before_stat = path.stat()
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            raw_sha.update(line)
            row = json.loads(line)
            same(row["index"], index, "request trace index")
            pair = row["pairing"]
            key = (pair["task"], pair["doc_id"], pair["candidate"])
            require(key in expected_responses and key not in seen, "Missing, duplicate or unexpected trace candidate identity")
            seen.add(key)
            expected = _pairing_fields(context["records"][key])
            same(set(pair), set(expected) | {"rng_before", "rng_after", "initialization"}, "pairing schema")
            for name, value in expected.items():
                same(pair[name], value, "audited request " + name)
            length = integer(pair["input_length"], "input length", 1)
            require(length <= 2048 and 0 < pair["continuation_length"] <= length, "Invalid model input/continuation lengths")
            same(length, min(pair["original_full_length"], 2049) - 1, "causal input length")
            same(pair["left_truncated_tokens"], max(0, pair["original_full_length"] - 2049), "left truncation")
            for name in ("rng_before", "rng_after"):
                _rng(pair[name])
            if previous_after is not None:
                same(pair["rng_before"], previous_after, "continuous RNG request stream")
            previous_after = pair["rng_after"]
            initial = pair["initialization"]
            same(initial["rng_before"], pair["rng_before"], "RNG before initialization")
            same(initial["rng_after"], pair["rng_after"], "RNG after initialization")
            same(initial["shape"], [1, length, dimensions], "initialization shape")
            same(initial["dtype"], "torch.bfloat16", "initialization dtype")
            valid_hash(initial["first_16_values_sha256"], "initialization prefix")
            valid_hash(initial["last_16_values_sha256"], "initialization suffix")
            require(pair["rng_before"]["model_device"] != pair["rng_after"]["model_device"], "Native random initialization did not advance GPU RNG")
            same(pair["rng_before"]["cpu"], pair["rng_after"]["cpu"], "unexpected CPU random draws")
            guidance = manifest["guidance"]
            observation = row["observation"]
            # Exact count schema is registered by the evaluator below.
            _validate_observation(observation, guidance)
            finite(row["loglikelihood"], "trace loglikelihood")
            require(type(row["is_greedy"]) is bool, "Invalid trace greedy flag")
            same([row["loglikelihood"], row["is_greedy"]], expected_responses[key], "trace/sample likelihood response")
            ph.update(canonical(pair) + b"\n")
            th.update(canonical(row) + b"\n")
            truncated += pair["left_truncated_tokens"] > 0
            max_input = max(max_input, length)
    after_stat = path.stat()
    same((before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns),
         (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns), "stable request trace file")
    same(seen, set(expected_responses), "complete request trace identities")
    same(ph.hexdigest(), audit["pairing_sha256"], "pairing stream hash")
    same(th.hexdigest(), audit["trace_sha256"], "trace stream hash")
    same(raw_sha.hexdigest(), audit["trace_sha256"], "canonical raw trace bytes")
    same(audit["truncated_requests"], truncated, "audited truncation count")
    return {"requests": expected_count, "pairing_sha256": ph.hexdigest(), "trace_sha256": th.hexdigest(),
            "truncated_requests": truncated, "max_input_tokens": max_input}


def _validate_observation(observation, guidance):
    # Kept separate so the exact evaluator schema remains explicit and testable.
    same(observation["guidance"]["mode"], guidance["mode"], "observed guidance mode")
    depth = guidance["total_loops"]
    enabled = guidance["mode"] != "baseline"
    readouts = 2 if guidance["mode"] in ("fixed", "adaptive") else 1
    expected = {"initializations": 1, "prelude": 8, "core_layers": 8 * depth,
                "projection": readouts, "coda": 8 * readouts, "norm": readouts, "head": readouts}
    for key, value in observation["call_counts"].items():
        integer(value, "call count " + key)
    same(observation["call_counts"], expected, "observed recurrence/readout counts")
    obs = observation["guidance"]
    same(obs["guidance_applied"], enabled, "observed guidance enabled")
    same(obs["total_loops"], depth, "observed depth")
    same(obs["readout_passes"], readouts, "observed readouts")
    if enabled:
        same(obs["reference_loop"], 1, "observed reference loop")
        same(obs["executed_source_indices"], list(range(depth)), "observed recurrence indices")
        same(obs["combination_location"], "before_C" if guidance["mode"] == "hidden" else "after_complete_native_readout", "guidance location")
        same(obs["cache"], None, "guidance cache")


def read_parcae_run(directory, prompt_audit=None, source_root=None):
    """Read one complete arm; all candidate occurrences remain separate."""
    directory = Path(directory).resolve()
    source_root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    m = read_json(directory / "manifest.json")
    require(m["status"] == "completed", "Run is not completed")
    task, arm = m["task"], m["arm"]
    require(task in SHOTS and arm in ARMS, "Unregistered task/arm")
    for key in ("total_loops", "reference_loop"):
        integer(m["guidance"][key], "guidance " + key, 1)
    for key in ("omega", "omega_cap"):
        finite(m["guidance"][key], "guidance " + key)
    same(m["guidance"], GUIDANCE[arm], "registered seven-arm guidance parameters")
    same(m["protocol"], "parcae_mc_v1", "registered protocol")
    for key, value in {"batch_size": 1, "seed": 42, "num_fewshot": SHOTS[task], "chat_template": False,
                       "max_length": 2048, "use_cache": False, "logits_cache": False}.items():
        require(type(m[key]) is type(value), "Incorrect type: " + key)
        same(m[key], value, key)
    limit = m["limit"]
    if limit is not None:
        integer(limit, "smoke limit", 1)
    require(type(m["is_full_split"]) is bool and m["is_full_split"] == (limit is None), "Inconsistent full/smoke split")
    if isinstance(prompt_audit, dict):
        context = prompt_audit
        same(context["task"], task, "prompt audit task context")
    else:
        context = load_prompt_audit(prompt_audit or m["prompt_audit"]["path"], task)
    audit_pin = m["prompt_audit"]
    for key, value in {"result_sha256": context["result_sha256"], "records_sha256": context["records_sha256"],
                       "records_file": task + ".jsonl.gz", "harness_files_sha256": context["report"]["harness"]["files_sha256"],
                       "task_documents": DOCS[task], "task_requests": REQUESTS[task]}.items():
        same(audit_pin[key], value, "prompt audit pin " + key)
    native = m["native_initialization"]
    require(native["per_request_reseed"] is False and native["one_initialization_per_scored_request"] is True,
            "Native initialization policy changed")
    require(isinstance(native["method"], str) and "trunc_normal" in native["method"], "Missing native like-init policy")
    require(finite(native["std"], "initialization std") > 0 and finite(native["embedding_scale"], "embedding scale") > 0,
            "Invalid native initialization scale")
    provenance = m["provenance"]
    required_sources = {"scripts/compare_parcae.py", "scripts/evaluate_parcae.py",
                        "scripts/prepare_parcae.py", "scripts/audit_parcae_prompts.py",
                        "src/loopcd_repro/parcae.py", "src/loopcd_repro/parcae_mc.py",
                        "src/loopcd_repro/guidance.py", "src/loopcd_repro/runtime.py",
                        "configs/parcae_1_3b_mc.json", "configs/mc_datasets.json"}
    observed_source = {str(p.relative_to(source_root)): sha256(p)
                       for folder in ("src", "scripts", "configs") for p in sorted((source_root / folder).rglob("*"))
                       if p.is_file() and p.suffix in (".py", ".json", ".yaml")}
    require(required_sources <= set(provenance["source_sha256"]), "Missing required frozen source hashes")
    same(provenance["source_sha256"], observed_source, "frozen source file set/bytes")
    same(provenance["source_sha256"]["scripts/compare_parcae.py"], sha256(Path(__file__)), "executing comparer source")
    same(m["paper_config_sha256"], observed_source["configs/parcae_1_3b_mc.json"], "paper config source hash")
    paper = read_json(source_root / "configs/parcae_1_3b_mc.json")
    same(m["paper_config"], paper, "frozen paper configuration")
    for key, value in {"arms": GUIDANCE, "tasks": SHOTS, "expected_documents": DOCS, "protocol": "parcae_mc_v1",
                       "repo_id": MODEL_REPO, "revision": MODEL_REVISION, "native_source_revision": OFFICIAL_SOURCE,
                       "harness_commit": HARNESS_COMMIT, "max_length": 2048, "seed": 42,
                       "truncation": "left_keep_2049_then_remove_last_token", "add_bos": False}.items():
        same(paper[key], value, "registered paper config " + key)
    same(m["protocol_sha256"], sha256(source_root / "docs/parcae_mc_protocol.md"), "protocol document hash")
    same(m["registry_sha256"], observed_source["configs/mc_datasets.json"], "data registry source hash")
    same(m["registry_sha256"], context["report"]["registry_sha256"], "audited data registry")
    same(m["harness_commit"], HARNESS_COMMIT, "runtime harness commit")
    same(m["harness_files"], context["harness_files"], "runtime/prompt harness file map")
    same(m["harness_files_sha256"], hash_json(m["harness_files"]), "runtime harness map hash")
    for key, value in {"add_bos": False, "truncation": "left_keep_2049_then_remove_last_token",
                       "request_order": "native_harness_order_no_sort_no_dedup",
                       "seed_reset": "once_after_model_loading_before_harness_requests; no per-request reseeding"}.items():
        same(m[key], value, key)
    require(m["model_full_hash_verified_before_and_after"] is True, "Missing model byte integrity gate")
    same(m["execution_device_type"], "cuda", "verified CUDA execution")
    valid_hash(m["gpu_smoke_sha256"], "real GPU gate")
    same(m["tokenizer"], {"bos_id": None, "eos_id": None, "pad_id": None,
         "vocab_size": 32768, "add_special_tokens": False}, "native tokenizer policy")
    for name, value in provenance["source_sha256"].items():
        valid_hash(value, "project source " + name)
    valid_hash(provenance["git_commit"], "frozen project commit", 40)
    model = provenance["model"]
    same(model["repo_id"], MODEL_REPO, "pinned model ID")
    same(model["revision"], MODEL_REVISION, "pinned model revision")
    valid_hash(provenance["loaded_model_code_sha256"], "loaded model code")
    same(model["model_code_sha256"], provenance["loaded_model_code_sha256"], "loaded/declared model code")
    same(model["source"]["git_commit"], OFFICIAL_SOURCE, "official model source commit")
    same(model["source"]["git_tree_sha1"], SOURCE_TREE, "official model source tree")
    same(model["source"]["files"]["receval/models/parcae.py"]["sha256"], model["model_code_sha256"], "loaded native forward source")
    same({k: v["sha256"] for k, v in model["files"].items()}, MODEL_FILES, "prepared checkpoint/tokenizer/config bytes")
    prepared = {k: v for k, v in model.items() if k != "model_code_sha256"}
    prepared_sha = hashlib.sha256((json.dumps(prepared, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    same(prepared_sha, context["report"]["model_manifest_sha256"], "prepared model manifest/prompt audit binding")
    loading = m["loading"]
    for key, value in {"official_requested_strict": False, "effective_strict": True, "loaded_keys": 225,
                       "missing_keys": [], "unexpected_keys": [], "attention": "sdpa",
                       "native_model_code_sha256": model["model_code_sha256"]}.items():
        same(loading[key], value, "strict checkpoint load " + key)
    for name, version in {"torch": "2.9.0", "transformers": "4.54.1", "accelerate": "1.12.0",
                          "datasets": "4.0.0", "lm_eval": "0.4.9.1"}.items():
        same(provenance["packages"][name], version, "runtime package " + name)
    same(provenance["attention"], "sdpa", "attention backend")
    same(provenance["arxiv"], "2610.02185v1", "target paper")
    same(provenance["precision"], "BF16 model; FP32 logits guidance; FP32 hidden blend cast to BF16 before native readout; FP32 log_softmax", "precision")
    result = read_json(directory / "results.json")
    same(m["results"], result["results"], "manifest/results aggregates")
    leaves = context["leaves"]
    same(sorted(m["datasets"]), sorted(leaves), "dataset leaf set")
    same(sorted(m["samples"]), sorted(leaves), "manifest sample leaf set")
    sample_paths = {p.name for p in directory.glob("samples_*.jsonl")}
    same(sample_paths, {"samples_" + leaf + ".jsonl" for leaf in leaves}, "sample file set")
    metrics = ("acc",) if task in ("mmlu", "winogrande") else METRICS
    documents, responses, artifacts = {}, {}, {}
    for leaf in sorted(leaves):
        count = leaves[leaf]["documents"]
        expected_count = count if limit is None else min(count, limit)
        dataset = m["datasets"][leaf]
        same(dataset["revision"], REVISIONS[task], "dataset revision")
        same(dataset["raw_source"], leaves[leaf]["dataset_source"], "audited raw dataset source")
        same(dataset["splits"][SPLITS[task]]["rows"], count, "dataset split size")
        require(bool(dataset["splits"][SPLITS[task]]["fingerprint"]), "Missing dataset fingerprint")
        config = result["configs"][leaf]
        same(config.get("test_split") or config.get("validation_split"), SPLITS[task], "harness evaluation split")
        same(config["num_fewshot"], SHOTS[task], "harness configured shots")
        same(result["n-shot"][leaf], SHOTS[task], "harness actual shots")
        same(result["n-samples"][leaf], {"original": count, "effective": expected_count}, "harness full/smoke sample counts")
        path = directory / ("samples_" + leaf + ".jsonl")
        found = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                require(bool(line.strip()), "Blank sample row")
                row = json.loads(line)
                doc_id = integer(row["doc_id"], "sample doc ID")
                key = (leaf, doc_id)
                require(key not in documents, "Duplicate sample document identity")
                found.append(doc_id)
                arguments = row["arguments"]
                require(isinstance(arguments, list) and bool(arguments), "Missing candidate arguments")
                same(row["filter"], "none", "sample filter")
                same(sorted(row["metrics"]), sorted(metrics), "sample metric set")
                same(len(row["resps"]), len(arguments), "sample response count")
                same(len(row["filtered_resps"]), len(arguments), "filtered response count")
                for candidate, args in enumerate(arguments):
                    identity = (leaf, doc_id, candidate)
                    require(identity in context["records"], "Unexpected sample candidate")
                    audited = context["records"][identity]
                    same(hash_json(row["doc"]), audited["doc_sha256"], "audited document content")
                    require(isinstance(args, list) and len(args) == 2 and all(isinstance(v, str) for v in args), "Invalid candidate arguments")
                    same(hash_json(args), audited["arguments_sha256"], "audited candidate arguments")
                    same(text_hash(args[0]), audited["context_sha256"], "audited context text")
                    same(text_hash(args[1]), audited["continuation_sha256"], "audited continuation text")
                    response = row["filtered_resps"][candidate]
                    require(isinstance(response, list) and len(response) == 2 and type(response[1]) is bool, "Invalid likelihood response")
                    finite(response[0], "sample likelihood")
                    same(row["resps"][candidate], [response], "unfiltered single request response")
                    responses[identity] = response
                require((leaf, doc_id, len(arguments)) not in context["records"], "Dropped candidate occurrence")
                for field in ("doc_hash", "prompt_hash", "target_hash"):
                    valid_hash(row[field], field)
                values = {}
                for metric in metrics:
                    value = row[metric]
                    require(type(value) in (int, float, bool) and value in (0, 1), "Non-binary per-document metric")
                    values[metric] = int(value)
                documents[key] = {"metrics": values, "identity_sha256": hash_json({k: row[k] for k in (
                    "doc", "target", "arguments", "doc_hash", "prompt_hash", "target_hash")})}
        same(found, list(range(expected_count)), "ordered complete sample document IDs")
        same(m["samples"][leaf], expected_count, "manifest sample count")
        for metric in metrics:
            _aggregate(result["results"][leaf][metric + ",none"], [documents[(leaf, i)]["metrics"][metric] for i in found], leaf + "/" + metric)
        artifacts[path.name] = sha256(path)
    for metric in metrics:
        _aggregate(result["results"][task][metric + ",none"], [r["metrics"][metric] for r in documents.values()], task + "/" + metric)
    expected_request_ids = {key for key in context["records"] if (key[0], key[1]) in documents}
    same(set(responses), expected_request_ids, "all original candidate occurrences")
    trace = _trace(directory, m, context, responses)
    for name in ("results.json", "request_trace.jsonl"):
        artifacts[name] = sha256(directory / name)
    same(m["evidence_files"], artifacts, "completed output evidence file hashes")
    artifacts["manifest.json"] = sha256(directory / "manifest.json")
    return {"path": str(directory), "manifest": m, "result": result, "arm": arm, "task": task,
            "documents": documents, "trace": trace, "artifacts": artifacts, "context": context}


def paired_metrics(left, right, metric):
    changes = [right[key]["metrics"][metric] - row["metrics"][metric] for key, row in left.items()]
    n = len(changes)
    wins, losses = changes.count(1), changes.count(-1)
    # Exact rational moments avoid the Python 3.9/3.10 statistics variance difference.
    variance = (Fraction(wins + losses) - Fraction((wins - losses) ** 2, n)) / (n - 1) if n > 1 else None
    return {"delta_percentage_points": 100 * float(Fraction(wins - losses, n)), "wins": wins,
            "losses": losses, "ties": n - wins - losses,
            "paired_se_percentage_points": 100 * math.sqrt(float(variance)) / math.sqrt(n) if variance is not None else None}


def compare_parcae(directories, prompt_audit=None, source_root=None):
    require(len(directories) == 7, "Exactly seven registered arms are required")
    first = read_parcae_run(directories[0], prompt_audit, source_root)
    runs = {first["arm"]: first}
    for path in directories[1:]:
        run = read_parcae_run(path, first["context"], source_root)
        require(run["arm"] not in runs, "Duplicate arm")
        runs[run["arm"]] = run
    same(set(runs), set(ARMS), "complete seven-arm set")
    base = runs["baseline8"]
    paired_keys = ("task", "limit", "is_full_split", "protocol", "paper_config", "batch_size", "seed", "num_fewshot",
                   "chat_template", "max_length", "use_cache", "logits_cache", "native_initialization", "datasets", "harness_files",
                   "protocol_sha256", "paper_config_sha256", "registry_sha256", "gpu_smoke_sha256", "loading", "tokenizer", "execution_device_type")
    for arm, run in runs.items():
        for key in paired_keys:
            same(run["manifest"][key], base["manifest"][key], key)
        same(run["manifest"]["provenance"], base["manifest"]["provenance"], "provenance (model/source/environment)")
        for key in ("configs", "versions", "n-shot", "higher_is_better"):
            same(run["result"][key], base["result"][key], "harness results " + key)
        same(set(run["documents"]), set(base["documents"]), "document identities")
        for identity in base["documents"]:
            same(run["documents"][identity]["identity_sha256"], base["documents"][identity]["identity_sha256"], "document/prompt/target")
        same(run["trace"]["pairing_sha256"], base["trace"]["pairing_sha256"], "actual request/token/init/RNG ordered stream")
        same(run["trace"]["requests"], base["trace"]["requests"], "request count")
    result_metrics = {}
    for metric in METRICS:
        if metric not in next(iter(base["documents"].values()))["metrics"]:
            result_metrics[metric] = {"available": False, "reason": "Metric is not emitted by this task"}
            continue
        result_metrics[metric] = {"available": True,
            "percent": {arm: 100 * float(Fraction(sum(row["metrics"][metric] for row in runs[arm]["documents"].values()), len(base["documents"]))) for arm in ARMS},
            "correct_counts": {arm: sum(row["metrics"][metric] for row in runs[arm]["documents"].values()) for arm in ARMS},
            "paired_vs_baseline8": {arm: paired_metrics(base["documents"], runs[arm]["documents"], metric) for arm in ARMS if arm != "baseline8"},
            "auxiliary_paired_vs_baseline4": {arm: paired_metrics(runs["baseline4"]["documents"], runs[arm]["documents"], metric) for arm in ("fixed4", "hidden4")}}
    full = base["manifest"]["is_full_split"]
    return {"schema_version": 1, "status": "PASS", "created_at": datetime.now(timezone.utc).isoformat(),
            "task": base["task"], "n_documents": len(base["documents"]), "n_requests": base["trace"]["requests"],
            "is_full_split": full, "limit": base["manifest"]["limit"], "full_split_count_verified": full,
            "all_seven_arms_verified": True, "metrics": result_metrics,
            "runs": {arm: {"path": runs[arm]["path"], "guidance": runs[arm]["manifest"]["guidance"],
                            "files_sha256": runs[arm]["artifacts"], "request_audit": runs[arm]["trace"]} for arm in ARMS},
            "comparer_source_sha256": sha256(Path(__file__)),
            "validation": {"paper_arm_parameters": True, "complete_candidate_occurrences": True,
                           "external_prompt_audit": True, "model_source_data_harness_protocol": True,
                           "ordered_native_initialization_rng_stream": True, "likelihood_response_binding": True,
                           "sample_aggregate_consistency": True},
            "prompt_audit": {k: v for k, v in base["manifest"]["prompt_audit"].items() if k != "path"},
            "uncertainty_note": "Descriptive paired sample-SD/sqrt(n) in percentage points, computed from exact integer moments. Single seed; no model/seed/protocol uncertainty included.",
            "interpretation": "Full split paired measurement; no automatic paper-match verdict or elapsed-time FLOP claim." if full else "Matched subset smoke only; not a full benchmark result.",
            "half_depth_note": "R4 versus baseline8 is the registered half-depth comparison; baseline4 and its fixed4/hidden4 comparisons are auxiliary same-depth controls. All arms retain identical actual input and native initialization streams."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=7, type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path)
    parser.add_argument("--source-root", type=Path, help="Frozen source tree; defaults to this script's repository")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Refusing to overwrite comparison evidence")
        result = compare_parcae(args.runs, args.prompt_audit, args.source_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "PASS", "task": result["task"], "n_documents": result["n_documents"],
                      "is_full_split": result["is_full_split"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
