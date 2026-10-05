"""Fresh-only two-arm Parcae-370M ARC-C evaluator under the registered protocol."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loopcd_repro.parcae370_mc import (MAX_LENGTH, PRECISION, RequestAudit, digest,
                                  load_parcae, make_parcae_lm, read_prompt_audit, sha256, validate_gpu_gate)


def write_json(path, value, serializer=None):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serializer, allow_nan=False) + "\n")


def validate_configuration(config):
    expected = {'baseline8': {'mode': 'baseline', 'total_loops': 8, 'reference_loop': 1, 'omega': 0.5, 'omega_cap': 1.0}, 'adaptive8': {'mode': 'adaptive', 'total_loops': 8, 'reference_loop': 1, 'omega': 0.5, 'omega_cap': 1.0}}
    if config["arms"] != expected or config["tasks"] != {"arc_challenge": 25}:
        raise ValueError("Only fixed Figure1b 370M ARC-C arms/shots")
    for key, value in {"protocol": "parcae370_arc_v1", "seed": 42, "batch_size": 1,
                       "max_length": 2048, "use_cache": False, "logits_cache": False,
                       "chat_template": False, "add_bos": False,
                       "truncation": "left_keep_2049_then_remove_last_token"}.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected registered setting: {key}")


def run(args):
    if not sys.dont_write_bytecode:
        raise RuntimeError("Require PYTHONDONTWRITEBYTECODE=1 to preserve native source integrity")
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config_path = ROOT / "configs/parcae_370m_arc.json"
    registry_path = ROOT / "configs/mc_datasets.json"
    protocol_path = ROOT / "docs/parcae370_arc_protocol.md"
    config = json.loads(config_path.read_text())
    validate_configuration(config)
    if args.arm not in config["arms"] or args.task not in config["tasks"]:
        raise ValueError("Unknown registered arm/task")
    if args.limit is not None and (type(args.limit) is not int or args.limit != 2):
        raise ValueError("Debug limit must be a positive integer")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task, "arm": args.arm, "limit": args.limit, "is_full_split": args.limit is None,
        "guidance": config["arms"][args.arm], "protocol": config["protocol"], "paper_config": config,
        "paper_config_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path),
        "registry_sha256": sha256(registry_path), "batch_size": 1, "seed": 42,
        "num_fewshot": config["tasks"][args.task], "chat_template": False,
        "max_length": MAX_LENGTH, "use_cache": False, "logits_cache": False,
        "add_bos": False, "truncation": config["truncation"], "request_order": "native_harness_order_no_sort_no_dedup",
        "seed_reset": "once_after_model_loading_before_harness_requests; no per-request reseeding",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    save = lambda: write_json(args.output / "manifest.json", manifest)
    save()
    audit = None
    try:
        import torch
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.api.model import LM
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable
        from audit_parcae_prompts import verify_harness, piqa_spec
        from check_mc_data import build_local_task_spec, iter_leaf_tasks, validate_dataset_pin
        from prepare_parcae370 import verify_prepared
        from loopcd_repro.runtime import provenance
        torch.set_num_threads(4)
        report, expected, binding = read_prompt_audit(args.prompt_audit, args.task, args.limit)
        manifest["prompt_audit"] = binding
        harness_root = Path(lm_eval.__file__).resolve().parent
        harness = verify_harness(args.harness_reference, harness_root)
        if harness["commit"] != config["harness_commit"] or harness["files_sha256"] != binding["harness_files_sha256"]:
            raise ValueError("Harness differs from full-split prompt audit")
        manifest["harness_commit"] = harness["commit"]
        manifest["harness_files_sha256"] = harness["files_sha256"]
        manifest["harness_files"] = harness["files"]
        registry = json.loads(registry_path.read_text())
        if report["registry_sha256"] != manifest["registry_sha256"] or report["shots"] != config["tasks"]:
            raise ValueError("Dataset pins/shots differ from completed prompt audit")
        manager = TaskManager()
        entry = registry["tasks"][args.task]
        spec = (piqa_spec(registry, manager, args.dataset_cache, args.hub_cache) if args.task == "piqa"
                else build_local_task_spec(args.task, registry, manager, args.dataset_cache, args.hub_cache))
        tree = manager.load_task_or_group([copy.deepcopy(spec)])
        objects, datasets = [], {}
        for name, task in iter_leaf_tasks(tree):
            validate_dataset_pin(task, entry)
            if (task.config.test_split or task.config.validation_split) != entry["eval_split"]:
                raise ValueError("Unexpected native evaluation split")
            task.EVAL_HARNESS_NAME = name
            objects.append(task)
            datasets[name] = {"revision": entry["revision"], "raw_source": task.config.metadata["loopcd_dataset"],
                "splits": {split: {"rows": len(data), "fingerprint": data._fingerprint,
                           "cache_files": [{"name": Path(item["filename"]).name, "sha256": sha256(item["filename"])}
                                           for item in data.cache_files]} for split, data in task.dataset.items()}}
            if len(task.eval_docs) != report["tasks"][args.task]["leaves"][name]["documents"]:
                raise ValueError("Full evaluation split changed after prompt audit")
            if task.config.metadata["loopcd_dataset"] != report["tasks"][args.task]["leaves"][name]["dataset_source"]:
                raise ValueError("Raw dataset bytes/source changed after prompt audit")
        if len(objects) != entry.get("expected_leaf_tasks", 1):
            raise ValueError("Missing native leaf tasks")
        manifest["datasets"] = datasets
        gate = validate_gpu_gate(args.smoke, args.model, ROOT)
        manifest["gpu_smoke_sha256"] = gate["sha256"]
        manifest["gpu_smoke_path"] = gate["path"]
        model, tokenizer, loading = load_parcae(args.model, args.device)
        if args.limit is None and model.device.type != "cuda":
            raise ValueError("Full benchmark runs require the verified CUDA path")
        manifest["execution_device_type"] = model.device.type
        manifest["loading"] = {key: value for key, value in loading.items() if key != "prepared"}
        prepared = loading["prepared"]
        if (prepared["repo_id"] != config["repo_id"] or prepared["revision"] != config["revision"]
                or prepared["tokenizer"]["revision"] != config["tokenizer_revision"]
                or prepared["source"]["git_commit"] != config["native_source_revision"]
                or {name: value["sha256"] for name, value in prepared["files"].items()} != report["model_identity"]["files"]):
            raise ValueError("Prepared model differs from fixed protocol/prompt audit")
        manifest["tokenizer"] = {"bos_id": tokenizer.bos_id, "eos_id": tokenizer.eos_id,
                                 "pad_id": tokenizer.pad_id, "vocab_size": tokenizer.vocab_size,
                                 "add_special_tokens": False}
        manifest["native_initialization"] = {
            "method": "native like-init: randn then trunc_normal +/-3std then embedding scale",
            "std": float(model.config.init.get_std("embedding")), "embedding_scale": float(model.emb_scale),
            "recurrent_dimension": model.config.recurrent_embedding_dimension,
            "per_request_reseed": False, "one_initialization_per_scored_request": True,
        }
        manifest["provenance"] = provenance(args.model)
        manifest["provenance"]["model"]["model_code_sha256"] = loading["native_model_code_sha256"]
        manifest["provenance"]["loaded_model_code_sha256"] = loading["native_model_code_sha256"]
        manifest["provenance"]["precision"] = PRECISION
        manifest["provenance"]["attention"] = "sdpa"
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        save()
        with (args.output / "request_trace.jsonl").open("x") as trace:
            audit = RequestAudit(trace, expected)
            lm = make_parcae_lm(LM)(model, tokenizer, manifest["guidance"], audit)
            # Native loader constructs/randomizes weights before strict loading.
            # The only evaluation seed reset happens after all that work.
            torch.manual_seed(42)
            result = simple_evaluate(
                model=lm, tasks=[copy.deepcopy(spec)] if entry.get("group") else objects,
                task_manager=manager, num_fewshot=config["tasks"][args.task], limit=args.limit,
                batch_size=1, device=args.device, use_cache=None, cache_requests=False,
                bootstrap_iters=0, log_samples=True, random_seed=42, numpy_random_seed=42,
                torch_random_seed=None, fewshot_random_seed=42, apply_chat_template=False,
            )
            manifest["request_audit"] = audit.summary(require_complete=True)
        samples = result.pop("samples", {})
        if set(samples) != set(datasets):
            raise ValueError("Native scoring returned an unexpected leaf set")
        for name, rows in samples.items():
            full_count = report["tasks"][args.task]["leaves"][name]["documents"]
            count = full_count if args.limit is None else min(args.limit, full_count)
            if len(rows) != count or {row["doc_id"] for row in rows} != set(range(count)):
                raise ValueError("Native sample coverage is incomplete")
            with (args.output / f"samples_{name}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, default=handle_non_serializable, allow_nan=False) + "\n")
        write_json(args.output / "results.json", result, handle_non_serializable)
        if verify_prepared(args.model) != prepared:
            raise ValueError("Prepared model changed during evaluation")
        current = provenance(args.model)
        if current["source_sha256"] != manifest["provenance"]["source_sha256"]:
            raise ValueError("Evaluation source changed during execution")
        manifest.update(status="completed", results=result["results"],
                        samples={name: len(rows) for name, rows in samples.items()},
                        model_full_hash_verified_before_and_after=True)
        manifest["evidence_files"] = {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.name != "manifest.json"}
        if model.device.type == "cuda":
            manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
            manifest["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(model.device)
        print(json.dumps({"status": "completed", "arm": args.arm, "task": args.task,
                          "samples": manifest["samples"], "request_audit": manifest["request_audit"]}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if audit is not None:
            manifest["request_audit"] = audit.summary()
        manifest["elapsed_seconds"] = time.monotonic() - started
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=("arc_challenge",), required=True)
    parser.add_argument("--arm", choices=("baseline8", "adaptive8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--harness-reference", type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
