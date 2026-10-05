"""Audit a saved native 5-shot MMLU run against raw data; no model execution.

Run in the original environment with --run and a NEW --output JSON path.
Imports only task definitions and CPU data readers. It does not construct
datasets, change caches, load model weights, or edit a frozen release.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = "c30699e8356da336a370243923dbaf21066bb9fe"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_record(path):
    path = Path(path).absolute()
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def audit(run, source_root=None):
    # Set before importing lm_eval, which imports torch transitively.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    import pyarrow.parquet as pq
    from jinja2 import Environment, StrictUndefined
    from lm_eval.tasks import TaskManager

    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "results.json").read_text())
    manager = TaskManager()
    environment = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    errors, checks, counts, subjects = [], Counter(), Counter(), {}
    files = {}

    def check(category, passed, detail):
        checks[category] += 1
        if not passed:
            errors.append({"check": category, "detail": detail})

    def record(path, expected=None):
        item = file_record(path)
        if expected is not None:
            item["expected_sha256"] = expected
            check("recorded_file_hash", item["sha256"] == expected, item["path"])
        files[item["path"]] = item
        return item

    record(__file__)
    record(run / "manifest.json")
    record(run / "results.json")
    check("completed_native_full_run", manifest.get("status") == "completed"
          and manifest.get("task") == "mmlu" and manifest.get("limit") is None
          and manifest.get("is_full_split") is True and manifest.get("chat_template") is False,
          "Expected completed full native MMLU without chat template")
    check("baseline_arm", manifest.get("guidance", {}).get("mode") == "baseline",
          "This audit is scoped to the saved baseline arm")

    matrix_path = run.parent / "matrix.json"
    if source_root is None:
        matrix = json.loads(matrix_path.read_text())
        source_root = matrix["source_root"]
        record(matrix_path)
    source_root = Path(source_root).resolve()
    frozen_sources = {}
    for relative, expected in manifest["provenance"]["source_sha256"].items():
        frozen_sources[relative] = record(source_root / relative, expected)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    check("frozen_source_commit", source_commit == manifest["provenance"]["git_commit"], source_commit)

    harness_root = Path(importlib.util.find_spec("lm_eval").origin).resolve().parent
    check("harness_location", str(harness_root) == manifest["harness_source_root"], str(harness_root))
    for relative, expected in manifest["harness_files"].items():
        record(harness_root / relative, expected)
    native_files = sorted(path for path in (harness_root / "tasks/mmlu/default").rglob("*")
                          if path.is_file() and (path.suffix == ".yaml" or path.name.endswith("_yaml")))
    for path in native_files:
        record(path)
    for relative in ("tasks/__init__.py", "tasks/manager.py", "api/samplers.py"):
        path = harness_root / relative
        if path.is_file():
            record(path)
    continuation_path = harness_root / "tasks/mmlu/continuation/_continuation_template_yaml"
    record(continuation_path)
    harness_git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=harness_root.parent, text=True).strip()
    harness_status = subprocess.check_output(
        ["git", "status", "--short", "lm_eval/tasks/mmlu", "lm_eval/api/task.py",
         "lm_eval/api/samplers.py", "lm_eval/models/huggingface.py"],
        cwd=harness_root.parent, text=True).strip()
    check("harness_relevant_git_clean", not harness_status, harness_status)

    group_definitions = {}

    def native_leaves(name):
        kind = manager.task_index[name]["type"]
        if kind == "tag":
            return [leaf for child in manager._get_tasklist(name) for leaf in native_leaves(child)]
        config = manager._get_config(name)
        if kind == "group":
            group_definitions[name] = config
            return [leaf for child in config["task"] for leaf in native_leaves(child)]
        return [name]

    leaves = native_leaves("mmlu")
    check("native_leaf_set", len(leaves) == 57 and len(set(leaves)) == 57
          and set(result["configs"]) == set(leaves), "Saved/native subject membership")
    for name, group in group_definitions.items():
        check("native_group_aggregation", group.get("aggregate_metric_list") == [{"metric": "acc", "weight_by_size": True}], name)
        children = []
        for child in group["task"]:
            children += manager._get_tasklist(child) if manager.task_index[child]["type"] == "tag" else [child]
        check("saved_group_membership", set(result["group_subtasks"][name]) == set(children), name)

    all_devs, all_tests = {}, []
    expected_sample_files = {f"samples_{name}.jsonl" for name in leaves}
    check("sample_file_set", {path.name for path in run.glob("samples_*.jsonl")} == expected_sample_files, "57 expected sample files")
    for name, cfg in sorted(result["configs"].items()):
        native = manager._get_config(name)
        allowed_overrides = {"dataset_path", "dataset_kwargs", "metadata", "num_fewshot"}
        for key, value in native.items():
            if key not in allowed_overrides:
                check("native_config_field", cfg.get(key) == value, [name, key])
        check("shot_split_sampler", cfg["num_fewshot"] == 5 and cfg["test_split"] == "test"
              and cfg["fewshot_split"] == "dev" and cfg["fewshot_config"] == {"sampler": "first_n"}, name)
        check("delimiters", cfg["target_delimiter"] == " " and cfg["fewshot_delimiter"] == "\n\n", name)
        provenance = cfg["metadata"]["loopcd_dataset"]
        check("raw_provenance", provenance["repo_id"] == "cais/mmlu" and provenance["revision"] == PIN
              and provenance == manifest["datasets"][name]["raw_source"], name)
        expected_raw = {item["path"]: item for item in provenance["raw_files"]}
        raw_paths, raw = {}, {}
        for split, paths in cfg["dataset_kwargs"]["data_files"].items():
            raw_paths[split] = []
            for path_string in paths:
                path = Path(path_string)
                check("raw_snapshot_revision", PIN in path.parts, path_string)
                relative = str(Path(*path.parts[path.parts.index(PIN) + 1:]))
                expected = expected_raw.pop(relative)
                item = record(path, expected["sha256"])
                check("raw_file_size", item["bytes"] == expected["bytes"], path_string)
                raw_paths[split].append(item)
            raw[split] = pq.read_table(paths).to_pylist()
            check("raw_split_count", len(raw[split]) == manifest["datasets"][name]["splits"][split]["rows"], [name, split])
        check("raw_source_file_set", not expected_raw, name)
        check("five_dev_rows", len(raw["dev"]) == 5, name)
        for doc in raw["dev"]:
            all_devs.setdefault(doc["question"].strip(), []).append(name)
        for idx, doc in enumerate(raw["test"]):
            all_tests.append((name, idx, doc["question"].strip()))
        template = environment.from_string(native["doc_to_text"])
        render = lambda doc: template.render(**doc)
        prefix = native["description"] + "\n\n".join(
            render(doc) + " " + native["doc_to_choice"][doc["answer"]]
            for doc in raw["dev"][:5]) + "\n\n"
        sample_path = run / f"samples_{name}.jsonl"
        sample_file = record(sample_path)
        raw_lines = sample_path.read_bytes()
        check("sample_file_complete_line", raw_lines.endswith(b"\n"), name)
        docs = [json.loads(line) for line in raw_lines.splitlines()]
        check("saved_test_count", len(docs) == len(raw["test"]) == manifest["samples"][name], name)
        seen, correct = set(), 0
        for sample in docs:
            idx = sample["doc_id"]
            seen.add(idx)
            target = raw["test"][idx]
            check("doc_equals_raw_test", sample["doc"] == target, [name, idx])
            check("target_equals_raw_answer", sample["target"] == target["answer"], [name, idx])
            expected_prompt = prefix + render(target)
            expected_arguments = [[expected_prompt, " " + letter] for letter in native["doc_to_choice"]]
            check("all_request_arguments", sample["arguments"] == expected_arguments, [name, idx])
            check("unanswered_current_question", expected_prompt.endswith("Answer:"), [name, idx])
            reconstructed_acc = float(max(range(4), key=lambda i: sample["filtered_resps"][i][0]) == target["answer"])
            check("saved_per_doc_metric", sample["acc"] == reconstructed_acc, [name, idx])
            correct += reconstructed_acc
        check("complete_unique_doc_ids", seen == set(range(len(raw["test"]))) and len(seen) == len(docs), name)
        check("subject_aggregate", abs(result["results"][name]["acc,none"] - correct / len(docs)) < 1e-12, name)
        counts.update(subjects=1, samples=len(docs), dev_rows=len(raw["dev"]), correct=int(correct))
        subjects[name] = {"samples": len(docs), "correct": int(correct), "dev_rows": len(raw["dev"]),
                          "prompt_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                          "sample_file": sample_file, "raw_files": raw_paths}

    overlaps = [{"test_subject": name, "test_doc_id": idx, "dev_subjects": all_devs[question],
                 "question_sha256": hashlib.sha256(question.encode()).hexdigest()}
                for name, idx, question in all_tests if question in all_devs]
    same_subject = [entry for entry in overlaps if entry["test_subject"] in entry["dev_subjects"]]
    check("no_same_subject_dev_test_question_overlap", not same_subject, same_subject)
    check("total_full_split_size", counts["samples"] == 14042 and counts["subjects"] == 57 and counts["dev_rows"] == 285, dict(counts))
    accuracy = counts["correct"] / counts["samples"]
    check("overall_aggregate", abs(result["results"]["mmlu"]["acc,none"] - accuracy) < 1e-12, accuracy)
    # Completed runs and their inputs must remain unchanged throughout inspection.
    for item in files.values():
        check("file_unchanged_during_audit", digest(item["path"]) == item["sha256"], item["path"])

    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed", "cpu_only": True,
        "run": str(run), "source_root": str(source_root), "source_commit": source_commit,
        "harness_root": str(harness_root), "harness_git_commit": harness_git,
        "harness_relevant_git_status": harness_status,
        "dataset": {"repo_id": "cais/mmlu", "revision": PIN},
        "scope": "All saved baseline documents, four request argument pairs per document, and all 57 native subjects; reconstructed from pinned raw test/dev parquet without loading a model.",
        "counts": dict(counts), "accuracy_percent": accuracy * 100,
        "checks_executed": dict(checks), "error_count": len(errors), "errors": errors,
        "same_subject_dev_test_question_overlap_count": len(same_subject),
        "cross_subject_dev_test_question_overlaps": overlaps,
        "native_group_definitions": group_definitions, "subjects": subjects,
        "files": files,
        "continuation_template_read_only": {"path": str(continuation_path), "text": continuation_path.read_text(),
                                             "evaluated": False},
        "limitations": [
            "Audits only baseline raw samples. Cross-arm prompt/provenance alignment is handled by the separate paired comparison.",
            "Checks saved un-tokenized requests and recorded likelihood metrics; does not rerun inference, inspect attention causality, or validate token-level model outputs.",
            "Raw parquet hashes are compared with run provenance. Cached Arrow bytes, model weight bytes, and training-data contamination are not re-audited here.",
            "Native task files are read and hashed at audit time; not every task-definition file was independently hashed when the original run started.",
            "Question overlap uses exact equality after stripping outer whitespace; it does not detect paraphrases or near duplicates.",
            "No author LoopCD task YAML was available; equivalence to its Table 2 MMLU protocol remains unresolved.",
            "The continuation template is inspected only. No continuation evaluation or parameter search is performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Otherwise read the run parent's matrix.json")
    parser.add_argument("--output", type=Path, required=True, help="New file; refuses overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new audit artifact path")
    report = audit(args.run, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "counts", "accuracy_percent", "error_count")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
