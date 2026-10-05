"""Validate four paired Huginn HumanEval runs and hash-bound isolated scores.

Only completed generation and fully validated scoring evidence can yield
pass@1. HumanEval+ requires both the base and extended suites to pass. A
single greedy completion per task is measured; no pass@k estimator is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


DATA_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
ARM_SETTINGS = {
    "baseline32": {"mode": "baseline", "total_loops": 32, "reference_loop": 7, "omega": 0.3},
    "hidden32": {"mode": "hidden", "total_loops": 32, "reference_loop": 7, "omega": 0.3},
    "baseline16": {"mode": "baseline", "total_loops": 16, "reference_loop": 6, "omega": 0.3},
    "hidden16": {"mode": "hidden", "total_loops": 16, "reference_loop": 6, "omega": 0.3},
}
SAFETY_CHECKS = {
    "uid_nonroot", "groups_empty", "capabilities_empty", "no_new_privs",
    "host_read_blocked", "host_write_blocked", "network_blocked", "fork_blocked",
    "exec_blocked", "signal_blocked", "chroot_blocked",
}
SCORE_STATUSES = {"pass", "fail", "timeout"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # Match generate_humaneval.py's serialized-configuration contract exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing required evidence: {key}")
    return mapping[key]


def same(a, b, name):
    if a != b:
        raise ValueError(f"Inconsistent or unpaired {name}")


def valid_hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256 evidence: {name}")
    return value


def positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {name}")


def tokens(value, name):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < 65536 for x in value):
        raise ValueError(f"Invalid Huginn token IDs: {name}")


def read_generation(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    same(require(manifest, "status"), "completed", "generation status")
    config = require(manifest, "config")
    config_hash = valid_hash(require(manifest, "config_hash"), "config_hash")
    same(config_hash, digest(config), "configuration hash")
    guidance = require(config, "guidance")
    matches = [name for name, settings in ARM_SETTINGS.items() if guidance == settings]
    if len(matches) != 1:
        raise ValueError("Unexpected HumanEval guidance settings; expected the four paper-configured Huginn arms")
    arm = matches[0]
    same(require(config, "benchmark"), "HumanEvalPlus-v0.1.10", "benchmark")
    same(require(config, "data_sha256"), DATA_SHA256, "pinned dataset")
    if require(config, "do_sample") is not False:
        raise ValueError("This pass@1 comparison expects one greedy completion per task")
    limit = require(config, "limit")
    if limit is not None:
        positive_integer(limit, "limit")
    full = require(manifest, "is_full_split")
    if type(full) is not bool or full != (limit is None):
        raise ValueError("Full-split claim disagrees with explicit debug limit")
    expected_ids = [f"HumanEval/{i}" for i in range(164 if full else min(limit, 164))]
    same(require(config, "task_ids"), expected_ids, "expected task IDs")
    same(require(manifest, "expected_samples"), len(expected_ids), "expected generation count")
    same(require(manifest, "completed_samples"), len(expected_ids), "completed generation count")
    positive_integer(require(config, "max_new_tokens"), "max_new_tokens")
    same(require(config, "native_context_length"), 4096, "native context limit")
    positive_integer(require(config, "seed"), "seed", minimum=0)
    for key in ("seed_rule", "instruction", "response_prefix", "prompt_builder", "initialization", "cache"):
        if not isinstance(require(config, key), str) or not config[key]:
            raise ValueError(f"Missing generation protocol field {key}")
    if not isinstance(require(config, "stops"), list) or not config["stops"]:
        raise ValueError("Missing stop protocol")
    same(require(config, "eos_token_id"), [65505, 65508], "native EOS IDs")
    same(require(config, "pad_token_id"), 65509, "native pad ID")
    source = require(config, "source")
    model = require(source, "model")
    same(require(model, "repo_id"), "tomg-group-umd/huginn-0125", "model family")
    same(require(model, "revision"), MODEL_REVISION, "model revision")
    valid_hash(require(model, "model_code_sha256"), "model code")
    same(require(source, "loaded_model_code_sha256"), model["model_code_sha256"], "loaded model code")
    hashes = require(source, "source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Generation source hashes are required")
    for key, value in hashes.items():
        valid_hash(value, key)
    for key in ("evalplus_prompt_sha256", "evalplus_sanitize_sha256"):
        valid_hash(require(source, key), key)
    require(source, "packages")
    require(source, "evalplus_version")
    path = directory / "samples.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            raise ValueError("Unexpected empty generation record")
        row = json.loads(line)
        task = require(row, "task_id")
        if task not in expected_ids or task in rows:
            raise ValueError("Duplicate or unknown generation task ID")
        same(require(row, "config_hash"), config_hash, "sample configuration hash")
        valid_hash(require(row, "problem_sha256"), "problem")
        prompt = require(row, "prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Empty prompt")
        same(require(row, "prompt_sha256"), digest(prompt), "prompt hash")
        tokens(require(row, "prompt_token_ids"), "prompt")
        if row["prompt_token_ids"].count(65504) != 1 or 65509 in row["prompt_token_ids"]:
            raise ValueError("Expected exactly one BOS and an unpadded prompt")
        expected_seed = (config["seed"] + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)) % (2**32)
        same(require(row, "seed"), expected_seed, "per-task initialization seed")
        tokens(require(row, "generated_token_ids"), "completion")
        positive_integer(require(row, "generated_tokens"), "generated_tokens")
        same(row["generated_tokens"], len(row["generated_token_ids"]), "generated token count")
        cap = min(config["max_new_tokens"], 4096 - len(row["prompt_token_ids"]))
        same(require(row, "effective_max_new_tokens"), cap, "per-task token cap")
        if cap < 1 or row["generated_tokens"] > cap:
            raise ValueError("Completion exceeds its token budget")
        reason = require(row, "stop_reason")
        if reason not in {"stop_string", "eos_token", "token_cap"}:
            raise ValueError("Unknown generation stop reason")
        if type(require(row, "cap_hit")) is not bool or row["cap_hit"] != (reason == "token_cap"):
            raise ValueError("Inconsistent cap-hit evidence")
        if reason == "token_cap":
            same(row["generated_tokens"], cap, "cap-hit token count")
        if reason == "eos_token" and row["generated_token_ids"][-1] not in config["eos_token_id"]:
            raise ValueError("EOS stop lacks an EOS token")
        stop = require(row, "stop_string")
        if (reason == "stop_string" and stop not in config["stops"]) or (reason != "stop_string" and stop is not None):
            raise ValueError("Inconsistent stop-string evidence")
        elapsed = require(row, "elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid generation elapsed time")
        for key in ("solution", "completion", "raw_generation"):
            if not isinstance(require(row, key), str):
                raise ValueError(f"Missing generated text field {key}")
        observation = require(row, "adapter_observation")
        for key, expected in (("mode", guidance["mode"]), ("total_loops", guidance["total_loops"]),
                              ("extra_coda_passes", 0), ("extra_lm_head_calls", 0)):
            same(require(observation, key), expected, "adapter " + key)
        if require(observation, "guidance_applied") is not (guidance["mode"] == "hidden"):
            raise ValueError("Recorded guidance was not applied as configured")
        if guidance["mode"] == "hidden":
            same(require(observation, "reference_loop"), guidance["reference_loop"], "observed reference loop")
            same(require(observation, "executed_physical_loops"), list(range(1, guidance["total_loops"] + 1)), "executed loops")
            same(require(observation, "lm_head_calls"), 1, "single native head")
            same(require(observation, "coda_layer_calls"), 2, "single native coda")
        rows[task] = row
    same(set(rows), set(expected_ids), "complete generated task set")
    same(require(manifest, "cap_hits"), sum(row["cap_hit"] for row in rows.values()), "manifest cap hits")
    return {"path": str(directory), "manifest": manifest, "config": config, "arm": arm,
            "rows": rows, "samples_sha256": sha256(path), "manifest_sha256": sha256(directory / "manifest.json")}


def read_scores(path, generation):
    report = read_json(path)
    same(require(report, "status"), "PASS", "scoring process status")
    same(require(report, "exit_code"), 0, "scoring process exit")
    if require(report, "evaluation_complete") is not True or report.get("error"):
        raise ValueError("Incomplete or failed scorer evidence cannot yield pass@1")
    same(require(report, "samples_sha256"), generation["samples_sha256"], "score/input samples SHA256")
    same(require(report, "dataset_sha256"), generation["config"]["data_sha256"], "scored dataset")
    for key in ("manifest_sha256", "scorer_source_sha256"):
        valid_hash(require(report, key), key)
    safety = require(report, "safety")
    checks = require(safety, "checks")
    if require(safety, "passed") is not True or not SAFETY_CHECKS.issubset(checks) or any(value is not True for value in checks.values()):
        raise ValueError("Scoring isolation checks are incomplete or failed")
    patch = require(report, "evaluator_patch")
    for key in ("upstream_sha256", "patched_sha256", "diff_sha256"):
        valid_hash(require(patch, key), key)
    same(hashlib.sha256(require(patch, "diff").encode()).hexdigest(), patch["diff_sha256"], "scorer patch diff hash")
    canonical = require(report, "canonical_validation")
    same(require(canonical, "status"), "PASS", "canonical validation")
    for key in ("manifest_sha256", "dataset_sha256", "scorer_source_sha256", "evaluator_patch"):
        same(require(canonical, key), report[key], "canonical/scoring " + key)
    for key in ("passed_tasks", "expected_rows", "completed_rows"):
        same(require(canonical, key), 164, "canonical " + key)
    if require(canonical, "all_base_plus_passed") is not True:
        raise ValueError("Scorer has not passed the complete canonical base/plus validation")
    valid_hash(require(canonical, "evidence_sha256"), "canonical evidence")
    rows = {}
    expected_ids = set(generation["rows"])
    for row in require(report, "rows"):
        task = require(row, "task_id")
        if task not in expected_ids or task in rows:
            raise ValueError("Duplicate or unknown scored task ID")
        if type(require(row, "sample_id")) is not int or row["sample_id"] != 0:
            raise ValueError("Expected one greedy score (sample_id=0) per task")
        for name in ("base", "plus"):
            suite = require(row, name)
            status = require(suite, "status")
            if status not in SCORE_STATUSES:
                raise ValueError("Unknown scorer task status; infrastructure errors are not wrong answers")
            if type(require(suite, "passed")) is not bool or suite["passed"] != (status == "pass"):
                raise ValueError("Inconsistent scorer pass flag")
            positive_integer(require(suite, "tests"), "suite test count")
            details = require(suite, "details")
            if not isinstance(details, list) or any(type(value) is not bool for value in details) or len(details) > suite["tests"]:
                raise ValueError("Invalid per-test scoring evidence")
            if suite["passed"] and (len(details) != suite["tests"] or not all(details)):
                raise ValueError("Passing suite lacks complete passing test details")
            if task == "HumanEval/32":
                if require(suite, "upstream_status") not in SCORE_STATUSES or not isinstance(require(suite, "upstream_details"), list):
                    raise ValueError("Missing original find_zero scorer evidence")
        if type(require(row, "plus_passed")) is not bool or row["plus_passed"] != (row["base"]["passed"] and row["plus"]["passed"]):
            raise ValueError("HumanEval+ requires base AND extended suites to pass")
        rows[task] = row
    same(set(rows), expected_ids, "complete scored task set")
    for key in ("expected_rows", "completed_rows"):
        same(require(report, key), len(rows), "scorer " + key)
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "report": report, "rows": rows}


def paired_metric(baseline, candidate, suite):
    value = lambda row: int(row["base"]["passed"] if suite == "base" else row["plus_passed"])
    differences = {task: value(candidate[task]) - value(baseline[task]) for task in baseline}
    vals = list(differences.values())
    return {"baseline_pass_at_1_percent": 100 * statistics.mean(value(row) for row in baseline.values()),
            "hidden_pass_at_1_percent": 100 * statistics.mean(value(row) for row in candidate.values()),
            "delta_percentage_points": 100 * statistics.mean(vals),
            "wins": sum(x > 0 for x in vals), "losses": sum(x < 0 for x in vals), "ties": sum(x == 0 for x in vals),
            "win_task_ids": [task for task, delta in differences.items() if delta > 0],
            "loss_task_ids": [task for task, delta in differences.items() if delta < 0],
            "paired_se_percentage_points": 100 * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None}


def generation_stats(rows):
    lengths = sorted(row["generated_tokens"] for row in rows.values())
    elapsed = [row["elapsed_seconds"] for row in rows.values()]
    caps = [row["effective_max_new_tokens"] for row in rows.values()]
    return {"cap_hits": sum(row["cap_hit"] for row in rows.values()),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows.values())),
            "generated_tokens_total": sum(lengths), "generated_tokens_mean": statistics.mean(lengths),
            "generated_tokens_median": statistics.median(lengths),
            "generated_tokens_p95_nearest_rank": lengths[math.ceil(.95 * len(lengths)) - 1],
            "effective_cap_min": min(caps), "effective_cap_max": max(caps),
            "generation_seconds_total": sum(elapsed), "generation_seconds_mean": statistics.mean(elapsed)}


def compare_humaneval(directories, score_paths):
    if len(directories) != 4 or len(score_paths) != 4:
        raise ValueError("Exactly four generation directories and corresponding scorer JSONs are required")
    arms = {}
    for directory, scores in zip(directories, score_paths):
        generation = read_generation(directory)
        arm = generation["arm"]
        if arm in arms:
            raise ValueError("Duplicate generation arm")
        arms[arm] = {"generation": generation, "scores": read_scores(scores, generation)}
    same(set(arms), set(ARM_SETTINGS), "four Huginn arms")
    first = arms["baseline32"]
    common = {key: value for key, value in first["generation"]["config"].items() if key != "guidance"}
    for arm in arms.values():
        generation, scored = arm["generation"], arm["scores"]
        same({key: value for key, value in generation["config"].items() if key != "guidance"}, common,
             "generation protocol/model/source except guidance")
        for task, row in first["generation"]["rows"].items():
            for key in ("problem_sha256", "prompt", "prompt_sha256", "prompt_token_ids", "seed", "effective_max_new_tokens"):
                same(row[key], generation["rows"][task][key], task + " " + key)
        for key in ("manifest_sha256", "dataset_sha256", "scorer_source_sha256", "evaluator_patch", "canonical_validation", "scorer", "numpy", "psutil"):
            same(require(scored["report"], key), require(first["scores"]["report"], key), "paired scorer " + key)
        for task, row in first["scores"]["rows"].items():
            for suite in ("base", "plus"):
                same(row[suite]["tests"], scored["rows"][task][suite]["tests"], task + " suite test count")
    pairs = {}
    for depth in (32, 16):
        baseline, hidden = f"baseline{depth}", f"hidden{depth}"
        pairs[f"R{depth}"] = {"baseline": baseline, "candidate": hidden,
                              "metrics": {suite: paired_metric(arms[baseline]["scores"]["rows"], arms[hidden]["scores"]["rows"], suite)
                                          for suite in ("base", "plus")}}
    rows = first["generation"]["rows"]
    full = first["generation"]["manifest"]["is_full_split"]
    report = first["scores"]["report"]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": "HumanEvalPlus-v0.1.10", "n_tasks": len(rows), "is_full_split": full,
            "limit": common["limit"], "full_164_verified": full and len(rows) == 164,
            "protocol_sha256": digest(common), "common_protocol": common,
            "interpretation": "Full 164-task paired greedy measurement; not an automatic paper-reproduction verdict." if full else
                              "Debug subset only; these pass@1 values are not full HumanEval benchmark results.",
            "metric_definition": {"base": "One greedy completion passes all base tests",
                                  "plus": "The same completion passes all base AND all extended tests"},
            "validation": {"config_hashes_recomputed": True, "model_source_protocol_paired": True,
                           "problem_prompt_tokens_seed_paired": True, "score_input_hashes_verified": True,
                           "complete_tasks_and_scores": True, "isolation_and_canonical_checks_passed": True},
            "scorer": {key: report[key] for key in ("scorer", "manifest_sha256", "dataset_sha256", "scorer_source_sha256", "evaluator_patch", "canonical_validation")},
            "arms": {name: {"generation_path": arm["generation"]["path"],
                            "guidance": arm["generation"]["config"]["guidance"],
                            "generation_manifest_sha256": arm["generation"]["manifest_sha256"],
                            "samples_sha256": arm["generation"]["samples_sha256"],
                            "scores_path": arm["scores"]["path"], "scores_sha256": arm["scores"]["sha256"],
                            "generation": generation_stats(arm["generation"]["rows"]),
                            "scorer_status_counts": {suite: dict(Counter(row[suite]["status"] for row in arm["scores"]["rows"].values())) for suite in ("base", "plus")},
                            "find_zero_upstream_evidence": arm["scores"]["rows"].get("HumanEval/32")}
                     for name, arm in arms.items()},
            "pairs": pairs,
            "uncertainty_note": "Paired SE describes variation across tasks, not seed/protocol uncertainty. Generation time includes different completion lengths and is not evidence of a FLOP or speedup claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, type=Path, required=True)
    parser.add_argument("--scores", nargs=4, type=Path, required=True, help="Scorer JSONs in the same order as --runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite comparison evidence; choose a fresh output path")
    try:
        result = compare_humaneval(args.runs, args.scores)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n_tasks": result["n_tasks"], "pairs": result["pairs"]}))


if __name__ == "__main__":
    main()
