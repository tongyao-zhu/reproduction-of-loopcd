"""Compare two Huginn MC arms with explicit native-initialization pairing gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from compare import METRICS, metric_values, read_run, same


def trace_evidence(directory, manifest):
    audit = manifest["request_audit"]
    if not audit["complete"] or audit["request_count"] != audit["planned_requests"] or not audit["request_count"]:
        raise ValueError("Native initialization audit is incomplete")
    digest = hashlib.sha256()
    count = 0
    with (Path(directory) / "request_trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            same(row["index"], count, "request trace index")
            same(row["rng_before"], row["initialization"]["rng_before"], "RNG before native initialization")
            same(row["rng_after"], row["initialization"]["rng_after"], "RNG after native initialization")
            same(row["initialization"]["scale"], 1.0, "native initialization scale")
            if not row["input_ids_sha256"] or not row["initialization"]["first_16_values_sha256"]:
                raise ValueError("Missing token or initialization evidence")
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            count += 1
    same(count, audit["request_count"], "trace/manifest request count")
    return {"requests": count, "trace_sha256": digest.hexdigest(), "summary": audit}


def compare_huginn(baseline_path, candidate_path):
    baseline = read_run(baseline_path, allowed_modes=("baseline", "hidden"))
    candidate = read_run(candidate_path, allowed_modes=("baseline", "hidden"))
    same(baseline["mode"], "baseline", "baseline mode")
    left, right = baseline["manifest"], candidate["manifest"]
    same(left["provenance"]["model"]["repo_id"], "tomg-group-umd/huginn-0125", "Huginn model")
    for key in ("task", "limit", "is_full_split", "paper_config", "batch_size", "seed", "chat_template",
                "datasets", "num_fewshot", "max_length", "use_cache", "logits_cache", "native_initialization", "harness_files"):
        same(left[key], right[key], key)
    for key in ("model", "source_sha256", "loaded_model_code_sha256", "packages", "precision", "attention", "arxiv"):
        same(left["provenance"][key], right["provenance"][key], "provenance." + key)
    for key in ("configs", "versions", "n-shot", "higher_is_better"):
        same(baseline["result"].get(key), candidate["result"].get(key), "results." + key)
    same(set(baseline["rows"]), set(candidate["rows"]), "document set")
    for identity, row in baseline["rows"].items():
        for key in ("doc_hash", "prompt_hash", "target_hash"):
            same(row[key], candidate["rows"][identity][key], f"{identity}: {key}")
    evidence = [trace_evidence(path, manifest) for path, manifest in ((baseline_path, left), (candidate_path, right))]
    same(evidence[0], evidence[1], "actual request order and native Gaussian initialization stream")
    metrics = {}
    for metric in METRICS:
        a, b = metric_values(baseline, metric), metric_values(candidate, metric)
        if a is None and b is None:
            metrics[metric] = {"available": False}
            continue
        if a is None or b is None:
            raise ValueError("A metric is missing from one arm")
        differences = [b[identity] - a[identity] for identity in a]
        metrics[metric] = {"available": True,
                           "percent": {"baseline": 100 * statistics.mean(a.values()), "candidate": 100 * statistics.mean(b.values())},
                           "delta_percentage_points": 100 * statistics.mean(differences),
                           "wins": sum(value > 0 for value in differences), "losses": sum(value < 0 for value in differences),
                           "ties": sum(value == 0 for value in differences),
                           "paired_se_percentage_points": 100 * statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None}
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "task": left["task"], "n_documents": len(baseline["rows"]), "is_full_split": left["is_full_split"],
            "full_split_count_verified": all(run["full_count_verified"] for run in (baseline, candidate)),
            "baseline": {"path": str(baseline_path), "guidance": left["guidance"]},
            "candidate": {"path": str(candidate_path), "guidance": right["guidance"]},
            "validation": {"paired": True, "native_initialization_stream": evidence[0],
                           "model_code_data_prompts_equal": True, "sample_aggregate_consistency": True},
            "metrics": metrics,
            "interpretation": "Matched single-seed measurement, not a complete paper reproduction. Different recurrence budgets are explicitly recorded. No inference of FLOP savings from elapsed time."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare_huginn(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"task": result["task"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
