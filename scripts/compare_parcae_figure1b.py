"""Figure1b Parcae1.3 HellaSwag baseline8/adaptive8 strict comparison only."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
from compare_parcae import read_parcae_run,require,same,paired_metrics,sha256,METRICS
ARMS=('baseline8','adaptive8')


def compare_pair(directories, prompt_audit=None, source_root=None):
    require(len(directories) == 2, "Exactly baseline8/adaptive8 are required")
    first = read_parcae_run(directories[0], prompt_audit, source_root)
    runs = {first["arm"]: first}
    for path in directories[1:]:
        run = read_parcae_run(path, first["context"], source_root)
        require(run["arm"] not in runs, "Duplicate arm")
        runs[run["arm"]] = run
    same(set(runs), set(ARMS), "complete Figure1b pair")
    base = runs["baseline8"]
    same(base["task"], "hellaswag", "Figure1b Parcae1.3 task")
    require(base["manifest"]["limit"] in (None, 2), "Only smoke2 or full split")
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
            "paired_vs_baseline8": {arm: paired_metrics(base["documents"], runs[arm]["documents"], metric) for arm in ARMS if arm != "baseline8"}}
    full = base["manifest"]["is_full_split"]
    return {"schema_version": 1, "status": "PASS", "created_at": datetime.now(timezone.utc).isoformat(),
            "task": base["task"], "n_documents": len(base["documents"]), "n_requests": base["trace"]["requests"],
            "is_full_split": full, "limit": base["manifest"]["limit"], "full_split_count_verified": full,
            "both_figure1b_arms_verified": True, "metrics": result_metrics,
            "runs": {arm: {"path": runs[arm]["path"], "guidance": runs[arm]["manifest"]["guidance"],
                            "files_sha256": runs[arm]["artifacts"], "request_audit": runs[arm]["trace"]} for arm in ARMS},
            "comparer_source_sha256": sha256(Path(__file__)),
            "validation": {"paper_arm_parameters": True, "complete_candidate_occurrences": True,
                           "external_prompt_audit": True, "model_source_data_harness_protocol": True,
                           "ordered_native_initialization_rng_stream": True, "likelihood_response_binding": True,
                           "sample_aggregate_consistency": True},
            "prompt_audit": {k: v for k, v in base["manifest"]["prompt_audit"].items() if k != "path"},
            "uncertainty_note": "Descriptive paired sample-SD/sqrt(n) in percentage points, computed from exact integer moments. Single seed; no model/seed/protocol uncertainty included.",
            "interpretation": "Full split paired measurement; no automatic paper-match verdict or elapsed-time FLOP claim." if full else "Matched subset smoke only; not a full benchmark result."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs=2,required=True)
    p.add_argument('--prompt-audit',type=Path)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),'Fresh comparison output required')
    result=compare_pair(a.runs,a.prompt_audit,a.source_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','task','n_documents','is_full_split')}))

if __name__=='__main__':main()
