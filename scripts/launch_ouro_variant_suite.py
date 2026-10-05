"""Run a new Ouro scale on one idle GPU after numerical and paired smoke gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_huginn_mc_suite import gpu_status
from compare import compare_runs
from launch_huginn_r16_suite import verify_frozen_source, sha256

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sciq", "piqa", "arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu")
MODES = ("baseline", "fixed", "adaptive")


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_comparison(result, task, smoke):
    if result["task"] != task or result["is_full_split"] != (not smoke):
        raise ValueError("Unexpected task or full-split status")
    if smoke and result["n_documents"] != 2:
        raise ValueError("Smoke gate requires two paired SciQ documents")
    if not smoke and not result["full_split_count_verified"]:
        raise ValueError("Complete dataset was not verified")
    if not result["validation"] or not all(x is True for x in result["validation"].values()):
        raise ValueError("Pairing or provenance failed")
    for mode in MODES:
        guidance = result["runs"][mode]["guidance"]
        if (guidance["mode"], guidance["early_loop"], guidance["omega"], guidance["omega_cap"]) != (mode, 1, .5, 1.0):
            raise ValueError("Guidance differs from fixed paper parameters")


class Queue:
    def __init__(self, args):
        self.args = args
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.source_hashes = verify_frozen_source(ROOT, commit)
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        self.env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": args.gpu,
                    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                    "HF_MODULES_CACHE": str(ROOT / ".cache/modules"), "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        self.state = {"status": "starting", "started_at": now(), "gpu": args.gpu,
                      "source_root": str(ROOT), "source_commit": commit, "source_sha256": self.source_hashes,
                      "model": str(args.model.resolve()), "paper_config": args.paper_config,
                      "tasks": list(TASKS), "resource_policy": "Before every GPU job: no compute PID and memory below 1000 MiB",
                      "smoke_gate": {"status": "pending"}, "jobs": {}, "comparisons": {}}
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        tmp = self.output / "matrix.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.output / "matrix.json")

    def execute(self, name, command):
        actual = {str(p.relative_to(ROOT)): sha256(p) for folder in ("src", "scripts", "configs")
                  for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml")}
        if actual != self.source_hashes:
            raise ValueError("Frozen source changed after startup")
        self.state["jobs"][name] = {"status": "waiting_for_device", "command": command}
        while True:
            device = gpu_status(self.args.gpu)
            self.state["jobs"][name]["device"] = device
            self.save()
            if device["ready"]:
                break
            time.sleep(self.args.poll_seconds)
        with (self.output / "logs" / (name + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            self.state["status"] = "running"
            self.state["jobs"][name].update(status="running", pid=process.pid, started_at=now())
            self.save()
            code = process.wait()
        self.state["jobs"][name].update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
        self.save()
        if code != 0:
            raise RuntimeError(f"{name} failed with exit code {code}; retained output, no automatic retry")

    def task(self, task, smoke=False):
        paths = []
        for mode in MODES:
            name = ("smoke_" if smoke else "") + task + "_" + mode
            path = self.output / name
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"),
                       "--model", str(self.args.model.resolve()), "--paper-config", self.args.paper_config,
                       "--task", task, "--mode", mode, "--output", str(path),
                       "--dataset-cache", str(self.args.dataset_cache.resolve())]
            if smoke:
                command += ["--limit", "2"]
            self.execute(name, command)
            paths.append(path)
        result = compare_runs(paths)
        validate_comparison(result, task, smoke)
        name = ("smoke_" if smoke else "") + task
        with (self.output / (name + "_comparison.json")).open("x") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
        self.state["comparisons"][name] = {"status": "completed", "n_documents": result["n_documents"], "is_full_split": not smoke}
        self.save()

    def run(self):
        try:
            self.execute("gpu_oracle", [sys.executable, "-u", str(ROOT / "scripts/gpu_smoke.py"),
                         "--model", str(self.args.model.resolve()), "--output", str(self.output / "gpu_oracle.json")])
            report = json.loads((self.output / "gpu_oracle.json").read_text())
            if report["status"] != "PASS" or len(report["checks"]) < 80 or not all(c["passed"] for c in report["checks"]):
                raise ValueError("Real-checkpoint numerical oracle did not pass all checks")
            self.task("sciq", smoke=True)
            self.state["smoke_gate"] = {"status": "passed", "finished_at": now(), "numerical_checks": len(report["checks"])}
            self.save()
            for task in TASKS:
                self.task(task)
            self.state["status"] = "completed"
        except Exception as error:
            self.state.update(status="failed", error=repr(error))
            if self.state["smoke_gate"]["status"] != "passed":
                self.state["smoke_gate"].update(status="failed", error=repr(error))
            raise
        finally:
            self.state["finished_at"] = now()
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--paper-config", choices=("ouro_1_4b_mc", "ouro_2_6b_mc"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("poll-seconds must be at least one")
    Queue(args).run()


if __name__ == "__main__":
    main()
