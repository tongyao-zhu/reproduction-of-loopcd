"""One paper-configured paired-evaluation arm using the native lm-eval tasks."""
import argparse
import copy
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate_model_identity(config, model_record):
    """Never label another checkpoint as the requested paper model."""
    for key, expected in (("repo_id", config["model"]), ("revision", config["revision"])):
        if model_record.get(key) != expected:
            raise ValueError(f"Model {key} differs from selected paper configuration")
    if (config["total_loops"], config["early_loop"]) != (4, 1):
        raise ValueError("This evaluator supports the paper's Ouro R4/h1 protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Ouro-1.4B")
    parser.add_argument("--paper-config", choices=("ouro_1_4b_mc", "ouro_2_6b_mc"), default="ouro_1_4b_mc")
    parser.add_argument("--task", choices=("sciq", "piqa", "arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu"), default="sciq")
    parser.add_argument("--mode", choices=("baseline", "fixed", "adaptive"), required=True)
    parser.add_argument("--limit", type=int, help="Debug subset only; omit for full split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-cache", type=Path, default=ROOT / ".cache/datasets")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive number of documents")
    config = json.loads((ROOT / "configs" / (args.paper_config + ".json")).read_text())
    validate_model_identity(config, json.loads((Path(args.model) / "model_provenance.json").read_text()))
    # Exclusive directory creation prevents silently mixing or overwriting runs.
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    from loopcd_repro.runtime import load_ouro, provenance, sha256

    guidance = GuidanceConfig(mode=args.mode, omega=config["fixed_omega"],
                              omega_cap=config["adaptive_cap"], early_loop=config["early_loop"])
    manifest = {
        "status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": args.task, "limit": args.limit, "is_full_split": args.limit is None,
        "guidance": asdict(guidance), "paper_config": config,
        "batch_size": 1, "seed": 42, "chat_template": False,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    def save_manifest():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    save_manifest()
    start = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model, tokenizer = load_ouro(args.model, args.device)
        model.config.use_cache = False  # independent teacher-forced likelihood requests
        manifest["provenance"] = provenance(args.model, model)
        validate_model_identity(config, manifest["provenance"]["model"])
        if manifest["provenance"]["loaded_model_code_sha256"] != manifest["provenance"]["model"]["model_code_sha256"]:
            raise ValueError("Loaded model source differs from prepared private source")
        import lm_eval
        harness_root = Path(lm_eval.__file__).resolve().parent
        manifest["harness_source_root"] = str(harness_root)
        manifest["harness_files"] = {str(p.relative_to(harness_root)): sha256(p) for p in
            [harness_root / "models/huggingface.py", harness_root / "evaluator.py",
             harness_root / "api/task.py", harness_root / "utils.py"]}
        # Pin all native tasks, including MMLU's full group topology.
        from check_mc_data import build_task_spec, build_local_task_spec
        registry = json.loads((ROOT / "configs/mc_datasets.json").read_text())
        entry = registry["tasks"][args.task]
        if entry["num_fewshot"] != config["tasks"][args.task]:
            raise ValueError("Dataset registry and paper shot counts disagree")
        manager = TaskManager()
        task_spec = (build_local_task_spec(args.task, registry, manager, args.dataset_cache)
                     if entry.get("group") else build_task_spec(args.task, registry))
        task_dict = manager.load_task_or_group([copy.deepcopy(task_spec)])
        objects = []
        manifest["datasets"] = {}
        def flatten_tasks(mapping):
            for name, task in mapping.items():
                if isinstance(task, dict):
                    yield from flatten_tasks(task)
                else:
                    yield name, task

        for name, task in flatten_tasks(task_dict):
            if not hasattr(task, "dataset"):
                raise ValueError(f"Unsupported task object: {name}")
            task.EVAL_HARNESS_NAME = name
            objects.append(task)
            raw_source = (task.config.metadata or {}).get("loopcd_dataset")
            if raw_source is None:
                for data in task.dataset.values():
                    if not data.cache_files or any(entry["revision"] not in Path(f["filename"]).parts for f in data.cache_files):
                        raise ValueError(f"Loaded dataset cache does not match pinned revision for {name}")
            elif raw_source["revision"] != entry["revision"] or not raw_source["raw_files"]:
                raise ValueError("Local parquet data has invalid revision provenance")
            manifest["datasets"][name] = {
                "revision": entry["revision"], "raw_source": raw_source,
                "splits": {split: {"rows": len(data), "fingerprint": data._fingerprint,
                    "cache_files": [{"name": Path(f["filename"]).name, "sha256": sha256(f["filename"])} for f in data.cache_files]}
                    for split, data in task.dataset.items()},
            }
        if entry.get("expected_leaf_tasks") and len(objects) != entry["expected_leaf_tasks"]:
            raise ValueError("Missing MMLU subjects")
        class NoTruncationHFLM(HFLM):
            def _loglikelihood_tokens(self, requests, *positional, **kwargs):
                for _, context, continuation in requests:
                    if len(context) + len(continuation) > self.max_length + 1:
                        raise ValueError("A request exceeds the recorded context budget; refusing silent truncation")
                return super()._loglikelihood_tokens(requests, *positional, **kwargs)
        lm = NoTruncationHFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                  max_length=8192, softmax_dtype=torch.float32, trust_remote_code=True)
        save_manifest()
        with OuroGuidance(model, guidance, total_loops=config["total_loops"]):
            result = simple_evaluate(
                model=lm, tasks=[copy.deepcopy(task_spec)] if entry.get("group") else objects, task_manager=manager,
                num_fewshot=config["tasks"][args.task], limit=args.limit,
                batch_size=1, bootstrap_iters=0, log_samples=True,
                random_seed=42, numpy_random_seed=42, torch_random_seed=42,
                fewshot_random_seed=42, apply_chat_template=False,
            )
        samples = result.pop("samples", {})
        for task_name, rows in samples.items():
            with (args.output / f"samples_{task_name}.jsonl").open("w") as out:
                for row in rows:
                    out.write(json.dumps(row, default=handle_non_serializable) + "\n")
        (args.output / "results.json").write_text(json.dumps(result, indent=2, default=handle_non_serializable) + "\n")
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - start,
                        results=result["results"], samples={k: len(v) for k, v in samples.items()})
        print(json.dumps({"output": str(args.output), "results": result["results"]}, default=str))
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        save_manifest()


if __name__ == "__main__":
    main()
