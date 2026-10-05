"""Gate paired MBPP generation; this launcher never executes generated code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generate_mbpp import ARMS, digest, load_problems, task_seed
from launch_huginn_r16_suite import gpu_status, verify_frozen_source, sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_generation(path, data, count, full):
    tasks = load_problems(data)[:count]
    ids = [x["task_id"] for x in tasks]
    if full and count != 378:
        raise ValueError("Full MBPP requires 378 tasks")
    common = None
    paired = {}
    evidence = {}
    for arm, (mode, loops, reference) in ARMS.items():
        directory = Path(path) / arm
        m = json.loads((directory / "manifest.json").read_text())
        c = m["config"]
        if (m["status"], m["completed_samples"], m["expected_samples"], m["is_full_split"]) != ("completed", count, count, full):
            raise ValueError("Incomplete generation manifest")
        if c["task_ids"] != ids or c["limit"] != (None if full else count):
            raise ValueError("Incorrect task subset")
        if c["guidance"] != {"mode": mode, "total_loops": loops, "reference_loop": reference, "omega": .3}:
            raise ValueError("Guidance differs from the paper")
        if digest(c) != m["config_hash"]:
            raise ValueError("Configuration hash mismatch")
        remaining = {k: v for k, v in c.items() if k != "guidance"}
        if common is not None and remaining != common:
            raise ValueError("Unpaired generation configurations")
        common = remaining
        rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        if [r["task_id"] for r in rows] != ids:
            raise ValueError("Incomplete, duplicate, or unordered task IDs")
        for problem, row in zip(tasks, rows):
            if row["config_hash"] != m["config_hash"] or row["problem_sha256"] != digest(problem):
                raise ValueError("Sample data/config hash mismatch")
            if row["prompt_sha256"] != digest(row["prompt"]) or row["seed"] != task_seed(row["task_id"], c["seed"]):
                raise ValueError("Sample prompt or seed evidence invalid")
            tokens = row["prompt_token_ids"]
            if tokens[0] != 65504 or tokens.count(65504) != 1:
                raise ValueError("Expected one explicit BOS from the chat template")
            expected_cap = min(c["max_new_tokens"], c["native_context_length"] - len(tokens))
            if row["effective_max_new_tokens"] != expected_cap or row["generated_tokens"] != len(row["generated_token_ids"]):
                raise ValueError("Token count/cap evidence invalid")
            if not 0 < row["generated_tokens"] <= expected_cap:
                raise ValueError("Generation length exceeds configured limit")
            fields = {key: row[key] for key in ("problem_sha256", "prompt", "prompt_token_ids", "prompt_sha256", "seed", "effective_max_new_tokens")}
            identity = row["task_id"]
            if identity in paired and paired[identity] != fields:
                raise ValueError("Prompt/token/seed/cap pairing failed")
            paired[identity] = fields
        evidence[arm] = {"samples_sha256": sha256(directory / "samples.jsonl"), "manifest_sha256": sha256(directory / "manifest.json"),
                         "n": len(rows), "cap_hits": sum(row["cap_hit"] for row in rows)}
    return {"validated_at": now(), "is_full_split": full, "tasks_per_arm": count, "arms": evidence,
            "paired_prompts_tokens_data_seeds_caps": True, "generated_code_executed": False,
            "scoring_status": "pending independent MBPP canonical378 isolation gate"}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    output = args.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pins = verify_frozen_source(ROOT, commit)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "starting", "source_commit": commit, "source_root": str(ROOT), "source_sha256": pins,
             "started_at": now(), "gpu": args.gpu, "jobs": {}, "smoke_gate": "pending", "model_code_execution": "forbidden in this launcher"}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": args.gpu,
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_MODULES_CACHE": str(output / ".cache/modules"),
           "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
    def save():
        state["updated_at"] = now()
        tmp = output / "matrix.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(output / "matrix.json")
    save()
    try:
        for phase, count, full in (("smoke", 2, False), ("full", 378, True)):
            command = [sys.executable, "-u", str(ROOT / "scripts/generate_mbpp.py"), "--model", str(args.model.resolve()),
                       "--data", str(args.data.resolve()), "--output", str(output / phase)]
            if not full:
                command += ["--limit", str(count)]
            state["jobs"][phase] = {"status": "waiting_for_gpu", "command": command}
            while True:
                device = gpu_status(args.gpu)
                state["jobs"][phase]["device"] = device
                save()
                if device["ready"]:
                    break
                time.sleep(15)
            if verify_frozen_source(ROOT, commit) != pins:
                raise ValueError("Source changed before GPU job")
            with (output / (phase + ".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="running")
                state["jobs"][phase].update(status="running", pid=child.pid, started_at=now())
                save()
                code = child.wait()
            state["jobs"][phase].update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
            save()
            if code or verify_frozen_source(ROOT, commit) != pins:
                raise RuntimeError("Generation failed or source changed; retained outputs")
            evidence = verify_generation(output / phase, args.data, count, full)
            (output / (phase + "_verification.json")).write_text(json.dumps(evidence, indent=2) + "\n")
            if not full:
                state["smoke_gate"] = "passed"
            save()
        state["status"] = "completed_generation_unscored"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["finished_at"] = now()
        save()


if __name__ == "__main__":
    main()
