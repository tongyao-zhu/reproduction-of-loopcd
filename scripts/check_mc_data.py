"""CPU-only, offline preflight of pinned native MC tasks and cached datasets.

Native specs keep the original lm-eval prompts and transformations. The explicit
local-parquet fallback is needed when raw Hub files exist but datasets cannot
resolve that repository offline; it preserves native group definitions and
records the SHA256 of every raw input file in each task's metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(task, registry):
    value = registry["tasks"][task]
    if not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise ValueError(f"{task}: dataset revision must be a full commit SHA")
    return value


def build_task_spec(task, registry, cache_dir=None):
    """Return a fresh native task/group spec; TaskManager may mutate its input."""
    entry = _entry(task, registry)
    kwargs = {"revision": entry["revision"]}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).resolve())
    spec = {"task": task, "dataset_path": entry["dataset_path"], "dataset_kwargs": kwargs}
    if not entry.get("group"):
        spec["dataset_name"] = entry["config"]
    return spec


def load_pinned_task_specs(task, registry, cache_dir=None):
    """List wrapper for TaskManager.load_task_or_group or simple_evaluate."""
    return [build_task_spec(task, registry, cache_dir)]


def iter_leaf_tasks(tree):
    """Yield (native task name, Task) while retaining group nodes in the tree."""
    for name, value in tree.items():
        if isinstance(value, dict):
            yield from iter_leaf_tasks(value)
        elif hasattr(value, "dataset"):
            yield str(name), value
        else:
            raise TypeError(f"Unexpected task tree value at {name}: {type(value)}")


def validate_dataset_pin(task, entry):
    """Check effective data, since offline datasets may choose latest cached data."""
    source = (task.config.metadata or {}).get("loopcd_dataset")
    if source is not None:
        if source.get("repo_id") != entry["dataset_path"] or source.get("revision") != entry["revision"]:
            raise ValueError("Local raw dataset provenance disagrees with registry")
        if not source.get("raw_files"):
            raise ValueError("Local raw dataset file hashes are missing")
        return
    for split, data in task.dataset.items():
        if not data.cache_files:
            raise ValueError(f"{split}: native dataset cache paths are required to verify the revision")
        for item in data.cache_files:
            if entry["revision"] not in Path(item["filename"]).parts:
                raise ValueError(f"{split}: effective native dataset cache does not match pinned revision: {item['filename']}")


def build_local_task_spec(task, registry, manager, cache_dir, hub_cache=None):
    """Resolve existing pinned snapshot parquet only, preserving native prompts.

    This performs no download and no model work. Only dataset construction in
    the explicitly supplied dedicated cache may write files later. MMLU's
    original nested groups, aliases, aggregate rules, and subject configs are
    retained; a tag expands to its native leaf tasks in native order.
    """
    entry = _entry(task, registry)
    if hub_cache is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub_cache = HF_HUB_CACHE
    snapshot = Path(hub_cache) / ("datasets--" + entry["dataset_path"].replace("/", "--")) / "snapshots" / entry["revision"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned raw dataset snapshot is missing: {snapshot}")

    def expand(name):
        index = manager.task_index[name]
        if index["type"] == "tag":
            return [item for child in manager._get_tasklist(name) for item in expand(child)]
        config = copy.deepcopy(manager._get_config(name))
        if index["type"] == "group":
            config["task"] = [item for child in config["task"] for item in expand(child)]
            return [config]
        if index["type"] != "task":
            raise ValueError(f"Unsupported native task index type: {index['type']}")
        if config["dataset_path"] != entry["dataset_path"]:
            raise ValueError(f"{name}: native dataset path differs from pinned registry")
        subset = config.get("dataset_name")
        raw_dir = snapshot / subset if subset else snapshot / "data"
        if not raw_dir.is_dir():
            raw_dir = snapshot
        files = sorted(raw_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{name}: no cached parquet in {raw_dir}")
        data_files = {}
        source_files = []
        for path in files:
            split = path.name.split("-", 1)[0]
            # Keep the snapshot's .parquet filename: resolving the symlink to a
            # suffixless Hub blob can confuse extension-based data discovery.
            data_files.setdefault(split, []).append(str(path.absolute()))
            source_files.append({"path": str(path.relative_to(snapshot)), "sha256": sha256(path), "bytes": path.stat().st_size})
        for split in (config.get("test_split"), config.get("validation_split"),
                      config.get("training_split"), config.get("fewshot_split")):
            if split and split not in data_files:
                raise FileNotFoundError(f"{name}: required split {split!r} is not cached")
        config["dataset_path"] = "parquet"
        config["dataset_kwargs"] = {"data_files": data_files, "cache_dir": str(Path(cache_dir).resolve())}
        config["metadata"] = {**config.get("metadata", {}), "loopcd_dataset": {
            "loader": "local_parquet_from_pinned_hub_snapshot", "repo_id": entry["dataset_path"],
            "revision": entry["revision"], "config": subset, "raw_files": source_files,
            "note": "Native task prompt, processor, split selection, metrics, and group aggregation retained.",
        }}
        return [config]

    result = expand(task)
    if len(result) != 1:
        raise ValueError("Top-level task must resolve to one native task or group")
    return result[0]


def task_metadata(name, task, entry, context_samples=2, tokenizer=None, seed=42):
    validate_dataset_pin(task, entry)
    task.set_config("num_fewshot", entry["num_fewshot"])
    task.set_fewshot_seed(seed)
    docs = task.eval_docs
    expected_split = task.config.test_split or task.config.validation_split
    if expected_split != entry["eval_split"]:
        raise ValueError(f"{name}: unexpected native evaluation split {expected_split}")
    contexts = []
    for doc_id in range(min(context_samples, len(docs))):
        context = task.fewshot_context(docs[doc_id], num_fewshot=entry["num_fewshot"], apply_chat_template=False)
        parts = context if isinstance(context, list) else [context]
        if not all(isinstance(part, str) for part in parts):
            raise TypeError(f"{name}: unexpected fewshot context type")
        record = {"doc_id": doc_id, "sha256": hashlib.sha256(json.dumps(context, ensure_ascii=False).encode()).hexdigest(),
                  "characters": [len(part) for part in parts]}
        if tokenizer is not None:
            record["tokens"] = [len(tokenizer.encode(part, add_special_tokens=False)) for part in parts]
        contexts.append(record)
    return {
        "task": name, "native_dataset_path": entry["dataset_path"], "revision": entry["revision"],
        "config": task.config.dataset_name, "num_fewshot": entry["num_fewshot"],
        "eval_split": expected_split, "n_eval_documents": len(docs),
        "fewshot_split": task.config.fewshot_split or task.config.training_split or task.config.validation_split,
        "metrics": [item["metric"] for item in task.config.metric_list],
        "splits": {split: {"rows": len(data), "fingerprint": data._fingerprint,
                           "cache_files": [{"name": Path(item["filename"]).name, "sha256": sha256(item["filename"])}
                                           for item in data.cache_files]}
                   for split, data in task.dataset.items()},
        "raw_source": task.config.metadata.get("loopcd_dataset"),
        "first_contexts": contexts,
        "context_note": "Only the first context_samples documents were inspected; these lengths are not corpus maxima.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu"])
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/mc_datasets.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/datasets")
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="Optional existing local tokenizer; never downloads")
    parser.add_argument("--context-samples", type=int, default=2)
    parser.add_argument("--local-parquet", action="store_true", help="Use only existing raw snapshots for every task")
    args = parser.parse_args()
    if args.context_samples < 0:
        parser.error("--context-samples must be nonnegative")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from lm_eval.tasks import TaskManager
    registry = json.loads(args.registry.read_text())
    manager = TaskManager()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    output = {"offline": True, "cpu_only": True, "seed": 42, "tasks": {}}
    for name in args.tasks:
        entry = _entry(name, registry)
        if args.local_parquet or entry.get("group"):
            spec = build_local_task_spec(name, registry, manager, args.cache_dir, args.hub_cache)
        else:
            spec = build_task_spec(name, registry)
        tree = manager.load_task_or_group([spec])
        leaves = {leaf_name: task_metadata(leaf_name, task, entry, args.context_samples, tokenizer)
                  for leaf_name, task in iter_leaf_tasks(tree)}
        if "expected_leaf_tasks" in entry and len(leaves) != entry["expected_leaf_tasks"]:
            raise ValueError(f"{name}: unexpected number of leaf tasks: {len(leaves)}")
        output["tasks"][name] = {"leaf_count": len(leaves), "n_eval_documents": sum(item["n_eval_documents"] for item in leaves.values()),
                                 "leaves": leaves}
        print(json.dumps({"task": name, "leaves": len(leaves), "n_eval_documents": output["tasks"][name]["n_eval_documents"]}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
