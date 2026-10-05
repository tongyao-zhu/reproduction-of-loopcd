"""Independently verify full paired MBPP generation and isolated scoring JSON.

No model, evaluator, canonical solution, or generated program is imported or
executed. Dataset and source pins identify the audited 2026-10-04 protocol.
The actual canonical evidence and runtime manifest are required, not merely
an embedded claim that canonical validation succeeded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics


EXPECTED_TASKS = 378
EXPECTED_TEST_TOTALS = {"base": 1174, "plus": 39841}
DATA_SHA256 = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
DATA_URL = "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
GENERATION_COMMIT = "d04ccabf6797a68bb9ca7f964a6f0acb27f2155a"
# Digest of all 32 .py/.json/.yaml files under src/scripts/configs at this commit.
SOURCE_MAP_SHA256 = "a9ed22b52b7415449425edf05377d19152f9574487f6babc04fa30b65212473d"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MODEL_CODE_SHA256 = "2bdcbf5e59a9a7d5509a8bbdd75583cc774b49be2f95a628b6486358951f246a"
RUNNER_SHA256 = "054e6996df65eaf1426da8f867086e75fc9e37d6ce9aeb56f296041301ff0e23"
PREPARE_SHA256 = "e320cb53d5f47933338680703d777deb0f1b0be0fc9fb68fdbde8e3a1bb14bd6"
SCORER = "EvalPlus 0.3.1 unmodified MBPP evaluator and official deserialized inputs"
EVALUATOR_PINS = {
    "opt/site/evalplus/eval/__init__.py": "76857b678cddca08dcaf54d7927b9a77826715cf657654ca5baf6c2b267f4c34",
    "opt/site/evalplus/eval/_special_oracle.py": "92b126b907ee493121b55de06f6a34058b6e18adc8cf1c48737eedcd83f24cdd",
    "opt/site/evalplus/data/mbpp.py": "186e09b7b14dcf12ed259dfca1d6644a1c266c1bbac41c917fd3c0449a734b24",
    "opt/site/evalplus/gen/util/__init__.py": "81a7f0ee32dda4b5794f9645cc1199dc4b5ad8f187c6bd36fca32892ac2b8eef",
}
MODEL_FILE_PINS = {
    "tokenizer.json": "9cc201a5061b70aba0d227ef9766fcfe21e0989e0c0ba442b4d7732c4de12308",
    "tokenizer_config.json": "ff9e171e16f200e7865c27fa38e661722848ab665af1610d5bb2b62502d6cdcf",
    "config.json": "e9fe79df06a783ca33a76038c59a715b6a62f15c4a5b17b9681e697fea46c79c",
    "generation_config.json": "3ed2b71c39d566df6ab667a2b843322cbafd2dd7dace7a27bc39dc7c3c3627ee",
}
ARM_SETTINGS = {
    "baseline32": {"mode": "baseline", "total_loops": 32, "reference_loop": 7, "omega": 0.3},
    "hidden32": {"mode": "hidden", "total_loops": 32, "reference_loop": 7, "omega": 0.3},
    "baseline16": {"mode": "baseline", "total_loops": 16, "reference_loop": 6, "omega": 0.3},
    "hidden16": {"mode": "hidden", "total_loops": 16, "reference_loop": 6, "omega": 0.3},
}
SAFETY_CHECKS = {"uid_nonroot", "groups_empty", "capabilities_empty", "no_new_privs",
                 "host_read_blocked", "host_write_blocked", "network_blocked", "fork_blocked",
                 "exec_blocked", "signal_blocked", "chroot_blocked"}
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
STOPS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n```\n"]
PROTOCOL = {
    "benchmark": "MbppPlus-v0.2.0", "data_url": DATA_URL, "expected_full_tasks": EXPECTED_TASKS,
    "protocol": "paper Table 3(b) Huginn LoopCD-Hidden; B.2/B.3 and Table A4",
    "metric": "greedy pass@1; base and base+extended separately",
    "max_new_tokens_source": "reconstruction choice; paper does not specify",
    "data_version_source": "EvalPlus 0.3.1 default; exact paper version unspecified",
    "seed": 42, "seed_rule": "(seed+int(sha256(task_id)[:8],16)) % 2**32; reset before every arm/problem",
    "max_new_tokens": 2048, "native_context_length": 4096, "limit": None, "do_sample": False,
    "eos_token_id": [65505, 65508], "pad_token_id": 65509, "stops": STOPS,
    "instruction": INSTRUCTION, "response_prefix": RESPONSE,
    "prompt_builder": "EvalPlus make_raw_chat_prompt; tokenize add_special_tokens=False",
    "tokenization_policy": "one template BOS; same reconstruction choice as HumanEval",
    "official_hf_default_difference": "EvalPlus HF encode default prepends another BOS to Huginn template; deliberately not replicated",
    "tokenization_gate": "default encode == [BOS] + single-BOS ids, checked for every task",
    "initialization": "native truncated Gaussian, init_scale=1; never zero",
    "cache": "native full HuginnDynamicCache; fresh for every arm/problem",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def problem_digest(problem):
    """Match immutable generator hashing of the SHA-pinned official raw data.

    Mbpp/404 includes 15 signed infinities in plus_input. The generator used
    Python JSON's Infinity spelling. Configurations and scores remain strict.
    This function is used only on problems from load_data's whole-file SHA gate.
    """
    return hashlib.sha256(json.dumps(problem, sort_keys=True, ensure_ascii=False, allow_nan=True).encode()).hexdigest()


def snapshot(path):
    raw = Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def json_snapshot(path):
    raw, checksum = snapshot(path)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, checksum


def require(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing required evidence: {key}")
    return mapping[key]


def same(actual, expected, name):
    # JSON comparison retains bool/int distinctions (True must not count as 1).
    if isinstance(actual, set) and isinstance(expected, set):
        actual, expected = sorted(actual), sorted(expected)
    if digest(actual) != digest(expected):
        raise ValueError(f"Inconsistent or unpaired {name}")


def valid_hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256 evidence: {name}")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {name}")
    return value


def token_ids(value, name):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < 65536 for x in value):
        raise ValueError(f"Invalid Huginn token IDs: {name}")


def load_data(path):
    raw, checksum = snapshot(path)
    same(checksum, DATA_SHA256, "pinned MBPP dataset SHA256")
    problems = {}
    for line in raw.splitlines():
        row = json.loads(line)
        task = require(row, "task_id")
        if not isinstance(task, str) or re.fullmatch(r"Mbpp/[0-9]+", task) is None or task in problems:
            raise ValueError("Duplicate or invalid dataset task ID")
        if not isinstance(require(row, "prompt"), str) or not row["prompt"].strip():
            raise ValueError("Dataset lacks public prompt")
        for suite in ("base", "plus"):
            inputs = require(row, suite + "_input")
            # The exact pinned JSON artifact stores Mbpp/793's empty extended
            # inputs as {}; the official loader deserializes that to []. Keep
            # raw problem bytes/shape for generation's problem digest.
            known_empty = task == "Mbpp/793" and suite == "plus" and inputs == {} and type(inputs) is dict
            if not isinstance(inputs, list) and not known_empty:
                raise ValueError("Dataset lacks suite inputs")
        problems[task] = row
    same(len(problems), EXPECTED_TASKS, "full 378-task dataset")
    ids = sorted(problems, key=lambda task: int(task.split("/")[1]))
    counts = {task: {suite: len(problems[task][suite + "_input"]) for suite in ("base", "plus")} for task in ids}
    same({suite: sum(c[suite] for c in counts.values()) for suite in ("base", "plus")},
         EXPECTED_TEST_TOTALS, "pinned test totals")
    return {"path": str(Path(path).resolve()), "sha256": checksum, "problems": problems, "task_ids": ids, "counts": counts}


def expected_prompt(problem):
    # Pinned Huginn template + official EvalPlus make_raw_chat_prompt, expressed
    # as string operations only. Hidden tests/canonical answers never enter it.
    return ("<|begin_text|><|begin_header|>user<|end_header|>\n\n" + INSTRUCTION
            + "\n```\n" + problem["prompt"].strip()
            + "\n```<|end_turn|><|begin_header|>Huginn<|end_header|>\n\n" + RESPONSE + "\n```python\n")


def validate_source(source):
    same(require(source, "git_commit"), GENERATION_COMMIT, "frozen generation commit")
    hashes = require(source, "source_sha256")
    if not isinstance(hashes, dict) or len(hashes) != 32:
        raise ValueError("Expected all 32 frozen generation source hashes")
    for path, checksum in hashes.items():
        valid_hash(checksum, path)
    same(digest(hashes), SOURCE_MAP_SHA256, "complete frozen generation source map")
    model = require(source, "model")
    for key, value in (("repo_id", "tomg-group-umd/huginn-0125"), ("revision", MODEL_REVISION),
                       ("model_code_sha256", MODEL_CODE_SHA256)):
        same(require(model, key), value, "model " + key)
    for path, checksum in MODEL_FILE_PINS.items():
        same(require(require(require(model, "files"), path), "sha256_original"), checksum, "model " + path)
    for key, value in {
        "loaded_model_code_sha256": MODEL_CODE_SHA256, "evalplus_version": "0.3.1",
        "evalplus_prompt_sha256": "455b72b9fdc7aa7daa7afe54a31118ee3eec3d17c463cc911642777d6ae50a7a",
        "evalplus_sanitize_sha256": "12fec16b93bfc4d9d9103f227b864dee3c510fe946539b567e6a7805de7d09b5",
        "precision": "BF16 model; FP32 guidance/log_softmax", "attention": "sdpa", "arxiv": "2610.02185v1",
        "python": "3.10.18", "cuda": "12.8", "gpu": "NVIDIA A100-SXM4-40GB",
        "packages": {"torch": "2.9.0", "transformers": "4.54.1", "accelerate": "1.12.0", "datasets": "4.0.0", "lm_eval": "0.4.9.1"},
    }.items():
        same(require(source, key), value, "generation source " + key)


def read_generation(directory, data):
    directory = Path(directory).resolve()
    manifest, manifest_hash = json_snapshot(directory / "manifest.json")
    for key, value in {"status": "completed", "is_full_split": True,
                       "expected_samples": EXPECTED_TASKS, "completed_samples": EXPECTED_TASKS}.items():
        same(require(manifest, key), value, "generation " + key)
    if manifest.get("error"):
        raise ValueError("Generation recorded an error")
    config = require(manifest, "config")
    checksum = valid_hash(require(manifest, "config_hash"), "config_hash")
    same(checksum, digest(config), "configuration hash")
    same(set(config), set(PROTOCOL) | {"data_sha256", "task_ids", "source", "guidance"}, "complete configuration fields")
    for key, value in PROTOCOL.items():
        same(require(config, key), value, "generation protocol " + key)
    same(config["data_sha256"], data["sha256"], "generation/data SHA256")
    same(config["task_ids"], data["task_ids"], "full ordered task IDs")
    matches = [name for name, settings in ARM_SETTINGS.items() if digest(config["guidance"]) == digest(settings)]
    if len(matches) != 1:
        raise ValueError("Unexpected guidance arm")
    arm = matches[0]
    validate_source(config["source"])
    raw, samples_hash = snapshot(directory / "samples.jsonl")
    rows = {}
    for line in raw.splitlines():
        row = json.loads(line)
        task = require(row, "task_id")
        if task not in data["problems"] or task in rows:
            raise ValueError("Duplicate or unknown generation task ID")
        same(task, data["task_ids"][len(rows)], "ordered generation task prefix")
        same(require(row, "config_hash"), checksum, "sample configuration hash")
        same(require(row, "problem_sha256"), problem_digest(data["problems"][task]), "sample/pinned problem hash")
        same(require(row, "prompt"), expected_prompt(data["problems"][task]), "public-only rendered prompt")
        same(require(row, "prompt_sha256"), digest(row["prompt"]), "prompt hash")
        ids = require(row, "prompt_token_ids")
        token_ids(ids, "prompt")
        if ids[0] != 65504 or ids.count(65504) != 1 or 65509 in ids:
            raise ValueError("Prompt requires exactly one leading BOS and no padding")
        seed = (config["seed"] + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)) % 2**32
        same(require(row, "seed"), seed, "per-task native initialization seed")
        generated = require(row, "generated_token_ids")
        token_ids(generated, "generated")
        same(require(row, "generated_tokens"), len(generated), "generated token count")
        cap = min(config["max_new_tokens"], 4096 - len(ids))
        same(require(row, "effective_max_new_tokens"), cap, "native-context token budget")
        if not 1 <= len(generated) <= cap:
            raise ValueError("Completion exceeds its token budget")
        reason, stop = require(row, "stop_reason"), require(row, "stop_string")
        if reason not in {"stop_string", "eos_token", "token_cap"}:
            raise ValueError("Unknown stop reason")
        same(require(row, "cap_hit"), reason == "token_cap", "cap flag")
        if reason == "token_cap":
            same(len(generated), cap, "cap-hit length")
            if generated[-1] in config["eos_token_id"]:
                raise ValueError("EOS termination mislabeled as token cap")
        if reason == "eos_token" and generated[-1] not in config["eos_token_id"]:
            raise ValueError("EOS reason without native EOS token")
        if (reason == "stop_string" and stop not in STOPS) or (reason != "stop_string" and stop is not None):
            raise ValueError("Invalid stop-string evidence")
        for key in ("raw_generation", "completion", "solution"):
            if not isinstance(require(row, key), str):
                raise ValueError("Invalid generated text field " + key)
        if any(text in row["completion"] for text in STOPS):
            raise ValueError("Completion contains an untrimmed configured stop string")
        elapsed = require(row, "elapsed_seconds")
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid generation elapsed time")
        obs, guidance = require(row, "adapter_observation"), config["guidance"]
        for key, value in {"mode": guidance["mode"], "total_loops": guidance["total_loops"],
                           "extra_coda_passes": 0, "extra_lm_head_calls": 0,
                           "guidance_applied": guidance["mode"] == "hidden",
                           "initialization": "native_random_or_explicit_input_states"}.items():
            same(require(obs, key), value, "adapter " + key)
        if guidance["mode"] == "hidden":
            for key, value in {"reference_loop": guidance["reference_loop"],
                               "executed_physical_loops": list(range(1, guidance["total_loops"] + 1)),
                               "coda_layer_calls": 2, "lm_head_calls": 1,
                               "combination_location": "raw_recurrent_output_before_native_pre_coda_ln_f",
                               "arithmetic": "FP32 blend cast back to native hidden dtype before native output interface",
                               "cache_semantics": "native recurrent history; guided coda history; one update per layer and token"}.items():
                same(require(obs, key), value, "hidden adapter " + key)
        rows[task] = row
    same(list(rows), data["task_ids"], "complete ordered 378-task generation")
    same(require(manifest, "cap_hits"), sum(row["cap_hit"] for row in rows.values()), "manifest cap hits")
    return {"path": str(directory), "manifest": manifest, "manifest_sha256": manifest_hash,
            "samples_sha256": samples_hash, "config": config, "arm": arm, "rows": rows}


def validate_generation_pairs(generations):
    if not isinstance(generations, list) or len(generations) != 4:
        raise ValueError("Exactly four verified generation records are required")
    arms = {record["arm"]: record for record in generations}
    same(set(arms), set(ARM_SETTINGS), "four unique generation arms")
    first = arms["baseline32"]
    common = {key: value for key, value in first["config"].items() if key != "guidance"}
    for generation in generations:
        same({key: value for key, value in generation["config"].items() if key != "guidance"}, common,
             "paired model/source/generation protocol")
        same(set(generation["rows"]), set(first["rows"]), "paired task set")
        for task, row in first["rows"].items():
            for key in ("problem_sha256", "prompt", "prompt_sha256", "prompt_token_ids", "seed", "effective_max_new_tokens"):
                same(generation["rows"][task][key], row[key], task + " paired " + key)
    return common


def validate_runtime(manifest, data):
    for key, value in {"dataset": "MbppPlus-v0.2.0", "dataset_sha256": data["sha256"],
                       "expected_tasks": EXPECTED_TASKS, "prepare_source_sha256": PREPARE_SHA256,
                       "scorer_source_sha256": RUNNER_SHA256}.items():
        same(require(manifest, key), value, "sandbox manifest " + key)
    root = PurePosixPath(require(manifest, "rootfs"))
    if not root.is_absolute() or root.parts[-3:] != (".sandbox", "mbpp-v2", "rootfs") or ".." in root.parts:
        raise ValueError("Expected the dedicated .sandbox/mbpp-v2/rootfs runtime")
    identity = require(manifest, "identity")
    same(require(identity, "uid"), 60002, "v2 dedicated UID")
    same(require(identity, "gid"), 60002, "v2 dedicated GID")
    for key in ("selected_at", "selection"):
        if not isinstance(require(identity, key), str) or not identity[key]:
            raise ValueError("Missing identity reservation evidence")
    immutable = require(manifest, "immutable_sha256")
    if not isinstance(immutable, dict):
        raise ValueError("Missing immutable runtime hashes")
    for path, checksum in immutable.items():
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid immutable runtime path")
        valid_hash(checksum, path)
    for path, checksum in {**EVALUATOR_PINS, "runner.py": RUNNER_SHA256,
                           "data/MbppPlus-v0.2.0.jsonl": data["sha256"]}.items():
        same(require(immutable, path), checksum, "immutable " + path)
    valid_hash(require(immutable, "usr/bin/python3"), "sandbox Python")
    evaluator = EVALUATOR_PINS["opt/site/evalplus/eval/__init__.py"]
    same(require(manifest, "evaluator_patch"), {"applied": False, "reason": "MBPP uses unmodified EvalPlus 0.3.1",
                                               "upstream_sha256": evaluator, "runtime_sha256": evaluator}, "unmodified official evaluator")
    same(require(manifest, "packages"), {"numpy": "2.2.6", "psutil": "7.0.0", "evalplus": "0.3.1",
                                         "appdirs": "1.4.4", "tempdir": "0.7.1", "wget": "3.2"}, "sandbox packages")


def validate_score_metadata(report, context):
    manifest = context["manifest"]
    for key, value in {"status": "PASS", "exit_code": 0, "evaluation_complete": True,
                       "manifest_sha256": context["manifest_sha256"], "dataset_sha256": context["data"]["sha256"],
                       "scorer_source_sha256": RUNNER_SHA256, "evaluator_patch": manifest["evaluator_patch"],
                       "identity": manifest["identity"], "sandbox": str(PurePosixPath(manifest["rootfs"]).parent),
                       "scorer": SCORER, "numpy": manifest["packages"]["numpy"], "psutil": manifest["packages"]["psutil"],
                       "expected_rows": EXPECTED_TASKS, "completed_rows": EXPECTED_TASKS,
                       "sample_validation": {"samples": EXPECTED_TASKS, "expected_full_tasks": EXPECTED_TASKS, "is_full_task_set": True}}.items():
        same(require(report, key), value, "scoring " + key)
    if report.get("error"):
        raise ValueError("Scoring infrastructure recorded an error")
    safety = require(report, "safety")
    same(require(safety, "passed"), True, "isolation self-test")
    checks = require(safety, "checks")
    same(checks, {name: True for name in SAFETY_CHECKS}, "all 11 isolation checks")


def validate_score_rows(report, data, canonical=False):
    records = require(report, "rows")
    if not isinstance(records, list) or len(records) != EXPECTED_TASKS:
        raise ValueError("Expected exactly 378 scoring rows")
    rows = {}
    for row in records:
        task = require(row, "task_id")
        if task not in data["problems"] or task in rows:
            raise ValueError("Duplicate or unknown scoring task ID")
        same(require(row, "sample_id"), 0, "single greedy sample ID")
        for name in ("base", "plus"):
            suite = require(row, name)
            status = require(suite, "status")
            if status not in {"pass", "fail", "timeout"}:
                raise ValueError("Unknown task status; scorer errors are not incorrect answers")
            count = integer(require(suite, "tests"), "suite test count")
            same(count, data["counts"][task][name], "pinned per-task test count")
            details = require(suite, "details")
            if not isinstance(details, list) or any(type(x) is not bool for x in details) or len(details) > count:
                raise ValueError("Invalid per-test details")
            complete_pass = len(details) == count and all(details)
            if status == "pass" and not complete_pass:
                raise ValueError("Passing suite lacks complete per-test evidence")
            same(require(suite, "passed"), status == "pass" and complete_pass, "suite pass flag")
            if canonical and (status != "pass" or not complete_pass):
                raise ValueError("Canonical suite did not pass every pinned test")
        same(require(row, "plus_passed"), row["base"]["passed"] and row["plus"]["passed"], "MBPP+ base AND extended")
        rows[task] = row
    same(set(rows), set(data["task_ids"]), "complete unique scored task set")
    return rows


def load_canonical(canonical_path, sandbox_manifest_path, data):
    manifest, manifest_hash = json_snapshot(sandbox_manifest_path)
    validate_runtime(manifest, data)
    evidence, evidence_hash = json_snapshot(canonical_path)
    context = {"manifest": manifest, "manifest_sha256": manifest_hash, "data": data,
               "evidence": evidence, "evidence_sha256": evidence_hash,
               "canonical_path": str(Path(canonical_path).resolve()),
               "manifest_path": str(Path(sandbox_manifest_path).resolve())}
    validate_score_metadata(evidence, context)
    same(require(evidence, "samples_sha256"), None, "canonical-only scorer input")
    context["rows"] = validate_score_rows(evidence, data, canonical=True)
    return context


def read_scores(path, generation, canonical_context):
    report, checksum = json_snapshot(path)
    validate_score_metadata(report, canonical_context)
    same(require(report, "samples_sha256"), generation["samples_sha256"], "score/input sample bytes SHA256")
    cert = require(report, "canonical_validation")
    expected = {"status": "PASS", "passed_tasks": EXPECTED_TASKS, "expected_rows": EXPECTED_TASKS,
                "completed_rows": EXPECTED_TASKS, "all_base_plus_passed": True,
                "manifest_sha256": canonical_context["manifest_sha256"], "dataset_sha256": canonical_context["data"]["sha256"],
                "scorer_source_sha256": RUNNER_SHA256, "evaluator_patch": canonical_context["manifest"]["evaluator_patch"],
                "evidence_sha256": canonical_context["evidence_sha256"], "result_sha256": canonical_context["evidence_sha256"],
                "task_ids": canonical_context["data"]["task_ids"]}
    for key, value in expected.items():
        same(require(cert, key), value, "canonical certificate " + key)
    result_path = require(cert, "result_path")
    if not isinstance(result_path, str) or not PurePosixPath(result_path).is_absolute():
        raise ValueError("Canonical certificate lacks its recorded result path")
    rows = validate_score_rows(report, canonical_context["data"])
    same(set(rows), set(generation["rows"]), "scored/generated task IDs")
    return {"path": str(Path(path).resolve()), "sha256": checksum, "report": report, "rows": rows}


def paired_metric(baseline, candidate, suite):
    value = lambda row: int(row["base"]["passed"] if suite == "base" else row["plus_passed"])
    differences = {task: value(candidate[task]) - value(baseline[task]) for task in baseline}
    vals = list(differences.values())
    return {"baseline_passed": sum(value(row) for row in baseline.values()),
            "hidden_passed": sum(value(row) for row in candidate.values()),
            "baseline_pass_at_1_percent": 100 * statistics.mean(value(row) for row in baseline.values()),
            "hidden_pass_at_1_percent": 100 * statistics.mean(value(row) for row in candidate.values()),
            "delta_percentage_points": 100 * statistics.mean(vals),
            "wins": sum(x > 0 for x in vals), "losses": sum(x < 0 for x in vals), "ties": sum(x == 0 for x in vals),
            "win_task_ids": [task for task, delta in differences.items() if delta > 0],
            "loss_task_ids": [task for task, delta in differences.items() if delta < 0],
            "paired_se_percentage_points": 100 * statistics.stdev(vals) / math.sqrt(len(vals))}


def generation_stats(rows):
    lengths = sorted(row["generated_tokens"] for row in rows.values())
    caps = [row["effective_max_new_tokens"] for row in rows.values()]
    elapsed = [row["elapsed_seconds"] for row in rows.values()]
    return {"cap_hits": sum(row["cap_hit"] for row in rows.values()),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows.values())),
            "generated_tokens_total": sum(lengths), "generated_tokens_mean": statistics.mean(lengths),
            "generated_tokens_median": statistics.median(lengths), "generated_tokens_p95_nearest_rank": lengths[math.ceil(.95 * len(lengths)) - 1],
            "effective_cap_min": min(caps), "effective_cap_max": max(caps),
            "generation_seconds_total": sum(elapsed), "generation_seconds_mean": statistics.mean(elapsed)}


def compare_mbpp(run_paths, score_paths, data_path, canonical_path, sandbox_manifest_path):
    if len(run_paths) != 4 or len(score_paths) != 4:
        raise ValueError("Exactly four full generation directories and matching score JSONs are required")
    data = load_data(data_path)
    generations = [read_generation(path, data) for path in run_paths]
    common = validate_generation_pairs(generations)
    context = load_canonical(canonical_path, sandbox_manifest_path, data)
    arms = {gen["arm"]: {"generation": gen, "scores": read_scores(path, gen, context)} for gen, path in zip(generations, score_paths)}
    first_report = arms["baseline32"]["scores"]["report"]
    for arm in arms.values():
        same(arm["scores"]["report"]["canonical_validation"], first_report["canonical_validation"], "paired canonical certificates")
    pairs = {}
    for depth in (32, 16):
        baseline, hidden = f"baseline{depth}", f"hidden{depth}"
        pairs[f"R{depth}"] = {"baseline": baseline, "candidate": hidden,
                              "metrics": {suite: paired_metric(arms[baseline]["scores"]["rows"], arms[hidden]["scores"]["rows"], suite) for suite in ("base", "plus")}}
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": "MbppPlus-v0.2.0", "n_tasks": EXPECTED_TASKS, "is_full_split": True, "full_378_verified": True,
            "protocol_sha256": digest(common), "common_protocol": common, "pairs": pairs,
            "metric_definition": {"base": "One greedy completion passes all base tests", "plus": "The same completion passes all base AND all extended tests"},
            "interpretation": "Full 378-task paired greedy measurement under the disclosed reconstruction protocol; not an automatic paper-reproduction verdict.",
            "validation": {"config_hashes_recomputed": True, "frozen_source_pins_verified": True,
                           "public_only_prompts_reconstructed": True, "problem_prompt_tokens_seed_paired": True,
                           "score_input_hashes_verified": True, "complete_unique_tasks_and_scores": True,
                           "actual_canonical_per_test_evidence_verified": True, "all_11_isolation_checks_passed": True},
            "scorer": {**{key: first_report[key] for key in ("scorer", "manifest_sha256", "dataset_sha256", "scorer_source_sha256", "evaluator_patch", "canonical_validation")},
                       "canonical_evidence_path": context["canonical_path"], "canonical_evidence_sha256": context["evidence_sha256"],
                       "sandbox_manifest_path": context["manifest_path"], "test_totals": EXPECTED_TEST_TOTALS},
            "arms": {name: {"generation_path": arm["generation"]["path"], "guidance": arm["generation"]["config"]["guidance"],
                            "generation_manifest_sha256": arm["generation"]["manifest_sha256"], "samples_sha256": arm["generation"]["samples_sha256"],
                            "scores_path": arm["scores"]["path"], "scores_sha256": arm["scores"]["sha256"],
                            "generation": generation_stats(arm["generation"]["rows"]),
                            "scorer_status_counts": {suite: dict(Counter(row[suite]["status"] for row in arm["scores"]["rows"].values())) for suite in ("base", "plus")}}
                     for name, arm in arms.items()},
            "limitations": ["JSON audit uses pinned tokenizer/source provenance and paired token IDs; it does not re-tokenize or decode with model libraries.",
                            "Sandbox safety records attest chroot/seccomp checks; the sandbox shares the host kernel and is not a VM.",
                            "Paired SE describes task variation, not seed/protocol uncertainty. Different completion lengths prevent timing alone from supporting FLOP/speedup claims."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, type=Path, required=True)
    parser.add_argument("--scores", nargs=4, type=Path, required=True, help="Scores in the same order as --runs")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--canonical-evidence", type=Path, required=True)
    parser.add_argument("--sandbox-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite comparison evidence")
    try:
        result = compare_mbpp(args.runs, args.scores, args.data, args.canonical_evidence, args.sandbox_manifest)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "full_378_verified": True, "pairs": result["pairs"]}))


if __name__ == "__main__":
    main()
