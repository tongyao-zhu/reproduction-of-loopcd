"""Validate and compare three paired lm-eval runs using only the standard library."""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import statistics


MODES = ("baseline", "fixed", "adaptive")
METRICS = ("acc", "acc_norm")


def require(mapping, key, context):
    if key not in mapping:
        raise ValueError(f"{context}: missing {key}")
    return mapping[key]


def read_json(path):
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def same(left, right, name):
    if left != right:
        raise ValueError(f"Runs are not paired: mismatched {name}")


def read_mmlu_run(directory, manifest, result):
    """Combine subject-level evidence while preserving the weighted group metric."""
    leaves = sorted(manifest["datasets"])
    expected = manifest["paper_config"].get("mmlu_expected_subjects", 57)
    if len(leaves) != expected or any(not name.startswith("mmlu_") for name in leaves):
        raise ValueError(f"MMLU requires all {expected} subject datasets")
    same(set(manifest["samples"]), set(leaves), "MMLU sample subject set")
    found = {p.stem.removeprefix("samples_") for p in directory.glob("samples_*.jsonl")}
    same(found, set(leaves), "MMLU evidence file set")
    combined, complete = {}, True
    for name in leaves:
        rows = [json.loads(line) for line in (directory / f"samples_{name}.jsonl").read_text().splitlines() if line.strip()]
        ids = set()
        for row in rows:
            doc_id = require(row, "doc_id", name)
            identity = json.dumps([name, doc_id])
            if identity in ids:
                raise ValueError(f"duplicate doc_id in {name}")
            ids.add(identity)
            for key in ("doc_hash", "prompt_hash", "target_hash"):
                if not isinstance(require(row, key, name), str) or not row[key]:
                    raise ValueError(f"Missing {key} in {name}")
            combined[identity] = row
        same(len(rows), manifest["samples"][name], f"{name} manifest sample count")
        counts = require(result["n-samples"], name, "MMLU subject n-samples")
        same(len(rows), counts["effective"], f"{name} effective sample count")
        if manifest["is_full_split"]:
            same(len(rows), counts["original"], f"{name} full split sample count")
            split = result["configs"][name].get("test_split") or result["configs"][name]["validation_split"]
            same(len(rows), manifest["datasets"][name]["splits"][split]["rows"], f"{name} dataset split count")
        else:
            complete = False
            if len(rows) > manifest["limit"]:
                raise ValueError(f"{name}: subset exceeds limit")
        if not rows:
            raise ValueError(f"{name}: no subject samples")
        if not math.isclose(statistics.mean(row["acc"] for row in rows), result["results"][name]["acc,none"], abs_tol=1e-8):
            raise ValueError(f"{name}: sample mean disagrees with subject aggregate")
    return {"path": str(directory), "mode": manifest["guidance"]["mode"], "manifest": manifest,
            "result": result, "aggregate": result["results"]["mmlu"], "rows": combined,
            "full_count_verified": complete}


def read_run(directory, allowed_modes=MODES):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"{directory}: run must have status completed")
    for key in ("task", "limit", "is_full_split", "guidance", "paper_config",
                "batch_size", "seed", "chat_template", "provenance", "datasets"):
        require(manifest, key, str(directory))
    limit = manifest["limit"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError(f"{directory}: limit must be a positive integer or null")
    if not isinstance(manifest["is_full_split"], bool) or manifest["is_full_split"] != (limit is None):
        raise ValueError(f"{directory}: inconsistent limit/is_full_split")
    mode = require(manifest["guidance"], "mode", "guidance")
    if mode not in allowed_modes:
        raise ValueError(f"{directory}: unexpected guidance mode {mode!r}")
    provenance = manifest["provenance"]
    model = require(provenance, "model", "provenance")
    for key in ("repo_id", "revision", "model_code_sha256"):
        if not require(model, key, "provenance.model"):
            raise ValueError(f"{directory}: empty model {key}")
    loaded_code = require(provenance, "loaded_model_code_sha256", "provenance")
    same(model["model_code_sha256"], loaded_code, "declared and loaded model code hash")
    if not require(provenance, "source_sha256", "provenance"):
        raise ValueError(f"{directory}: source hashes are required")
    if not manifest["datasets"]:
        raise ValueError(f"{directory}: dataset provenance is required")
    for dataset in manifest["datasets"].values():
        splits = require(dataset, "splits", "dataset")
        if not splits:
            raise ValueError(f"{directory}: dataset split fingerprints are required")
        for split in splits.values():
            if not require(split, "fingerprint", "dataset split"):
                raise ValueError(f"{directory}: empty dataset fingerprint")
            require(split, "rows", "dataset split")
    task = manifest["task"]
    result = read_json(directory / "results.json")
    aggregate = require(require(result, "results", "results.json"), task, "task results")
    if task == "mmlu":
        return read_mmlu_run(directory, manifest, result)
    rows = {}
    with (directory / f"samples_{task}.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = require(row, "doc_id", f"sample line {line_number}")
            if isinstance(doc_id, bool) or not isinstance(doc_id, (str, int)):
                raise ValueError(f"{directory}: invalid doc_id at line {line_number}")
            identity = json.dumps(doc_id, sort_keys=True)
            if identity in rows:
                raise ValueError(f"{directory}: duplicate doc_id {doc_id!r}")
            for key in ("doc_hash", "prompt_hash", "target_hash"):
                if not isinstance(require(row, key, f"sample {doc_id}"), str) or not row[key]:
                    raise ValueError(f"{directory}: empty or invalid {key} for doc_id {doc_id!r}")
            rows[identity] = row
    if not rows:
        raise ValueError(f"{directory}: no sample rows")
    if limit is not None and len(rows) > limit:
        raise ValueError(f"{directory}: sample count exceeds limit")
    if task in manifest.get("samples", {}):
        same(manifest["samples"][task], len(rows), "manifest sample count")
    counts = result.get("n-samples", {}).get(task)
    full_count_verified = False
    if counts:
        same(require(counts, "effective", "n-samples"), len(rows), "effective sample count")
        if manifest["is_full_split"]:
            same(require(counts, "original", "n-samples"), len(rows), "full split sample count")
            full_count_verified = True
    config = result.get("configs", {}).get(task, {})
    evaluation_split = config.get("test_split") or config.get("validation_split")
    if evaluation_split and task in manifest["datasets"]:
        splits = manifest["datasets"][task]["splits"]
        if evaluation_split in splits and manifest["is_full_split"]:
            same(splits[evaluation_split]["rows"], len(rows), "dataset evaluation split count")
            full_count_verified = True
    return {"path": str(directory), "mode": mode, "manifest": manifest,
            "result": result, "aggregate": aggregate, "rows": rows,
            "full_count_verified": full_count_verified}


def metric_values(run, metric):
    aggregate = run["aggregate"]
    keys = [key for key in aggregate if key == metric or key.startswith(metric + ",")]
    if len(keys) > 1:
        raise ValueError(f"{run['mode']}: ambiguous result filters for {metric}: {keys}")
    present = [metric in row for row in run["rows"].values()]
    if not any(present) and not keys:
        return None
    if not all(present) or not keys:
        raise ValueError(f"{run['mode']}: incomplete sample/aggregate metric {metric}")
    if keys[0] not in (metric, metric + ",none"):
        raise ValueError(f"{run['mode']}: only the unfiltered metric is supported: {keys[0]}")
    values = {}
    for identity, row in run["rows"].items():
        value = row[metric]
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value) or value not in (0, 1):
            raise ValueError(f"{run['mode']}: {metric} must be binary per document")
        values[identity] = float(value)
    reported = aggregate[keys[0]]
    if not isinstance(reported, (int, float)) or not math.isfinite(reported):
        raise ValueError(f"{run['mode']}: invalid aggregate {metric}")
    if not math.isclose(statistics.mean(values.values()), reported, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"{run['mode']}: sample mean disagrees with aggregate {metric}")
    return values


def compare_runs(directories):
    if len(directories) != 3:
        raise ValueError("Exactly three runs are required: baseline, fixed, adaptive")
    runs = {}
    for directory in directories:
        run = read_run(directory)
        if run["mode"] in runs:
            raise ValueError(f"Duplicate guidance mode: {run['mode']}")
        runs[run["mode"]] = run
    if set(runs) != set(MODES):
        raise ValueError("The runs must contain baseline, fixed, and adaptive")
    baseline = runs["baseline"]
    base_manifest = baseline["manifest"]
    for mode in MODES[1:]:
        candidate = runs[mode]
        manifest = candidate["manifest"]
        for key in ("task", "limit", "is_full_split", "paper_config", "batch_size", "seed",
                    "chat_template", "datasets"):
            same(base_manifest[key], manifest[key], key)
        for key in ("model", "source_sha256", "loaded_model_code_sha256"):
            same(base_manifest["provenance"][key], manifest["provenance"][key], f"provenance.{key}")
        for key in ("packages", "precision", "attention", "arxiv"):
            same(base_manifest["provenance"].get(key), manifest["provenance"].get(key), f"provenance.{key}")
        same(base_manifest.get("harness_files"), manifest.get("harness_files"), "harness_files")
        for key in ("configs", "versions", "n-shot", "higher_is_better"):
            same(baseline["result"].get(key), candidate["result"].get(key), f"results.{key}")
        same(set(baseline["rows"]), set(candidate["rows"]), "doc_id set")
        for identity, row in baseline["rows"].items():
            for key in ("doc_hash", "prompt_hash", "target_hash"):
                same(row[key], candidate["rows"][identity][key], f"{key} for doc_id {row['doc_id']!r}")
    metrics = {}
    for metric in METRICS:
        values = {mode: metric_values(runs[mode], metric) for mode in MODES}
        if all(value is None for value in values.values()):
            metrics[metric] = {"available": False, "reason": "Metric is not emitted by this task"}
            continue
        if any(value is None for value in values.values()):
            raise ValueError(f"Metric {metric} is missing in some runs")
        paired = {}
        for mode in MODES[1:]:
            differences = [values[mode][identity] - values["baseline"][identity]
                           for identity in baseline["rows"]]
            count = len(differences)
            paired[mode] = {
                "delta_percentage_points": 100 * statistics.mean(differences),
                "wins": sum(value > 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "paired_se_percentage_points": (
                    100 * statistics.stdev(differences) / math.sqrt(count) if count > 1 else None
                ),
            }
        metrics[metric] = {"available": True,
                           "percent": {mode: 100 * statistics.mean(values[mode].values()) for mode in MODES},
                           "paired_vs_baseline": paired}
    paper_targets = base_manifest["paper_config"].get(f"paper_{base_manifest['task']}_percent")
    full_split = base_manifest["is_full_split"]
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": base_manifest["task"], "n_documents": len(baseline["rows"]),
        "limit": base_manifest["limit"], "is_full_split": full_split,
        "full_split_count_verified": full_split and all(run["full_count_verified"] for run in runs.values()),
        "interpretation": (
            "Full-split paired measurement; this alone does not establish paper reproduction."
            if full_split else "Subset smoke test only; not a reproduction of the paper's benchmark result."
        ),
        "runs": {mode: {"path": runs[mode]["path"], "guidance": runs[mode]["manifest"]["guidance"]}
                 for mode in MODES},
        "validation": {"paired": True, "model_and_code": True, "source_hashes": True,
                       "datasets": True, "document_prompt_target_hashes": True,
                       "sample_means_match_aggregates": True},
        "metrics": metrics,
        "uncertainty_note": "Paired normal-approximation SE is sample SD of per-document differences / sqrt(n), in percentage points. It is descriptive, not a significance claim; it excludes model/seed and protocol uncertainty.",
        "paper_targets": {"percent": paper_targets, "metric": None,
                          "source": base_manifest["paper_config"].get("source"),
                          "note": "Paper targets are separate reference values. The paper does not unambiguously specify acc versus acc_norm here; no automatic target match or reproduction verdict is computed."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True,
                        metavar=("BASELINE", "FIXED", "ADAPTIVE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = compare_runs(args.runs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "task": summary["task"],
                      "n_documents": summary["n_documents"], "metrics": summary["metrics"]}))


if __name__ == "__main__":
    main()
