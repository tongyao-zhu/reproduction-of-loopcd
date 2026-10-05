"""Queue three Huginn MC arms behind Ouro, with a mandatory paired smoke gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sciq", "piqa", "arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu")
OURO_TASKS = {"arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu"}
ARMS = ("baseline32", "hidden32", "hidden16")
TERMINAL = {"completed", "failed", "cancelled", "canceled", "skipped", "blocked_by_gate_failure"}


def now():
    return datetime.now(timezone.utc).isoformat()


def dependency_status(matrix_path, gpu):
    """Read the predecessor's own GPU mapping; never infer completion from age."""
    if matrix_path is None:
        return {"ready": True, "reason": "No predecessor matrix requested"}
    try:
        matrix = json.loads(Path(matrix_path).read_text())
        modes = [name for name, device in matrix["gpus"].items() if str(device) == str(gpu)]
        if len(modes) != 1:
            return {"ready": False, "reason": "GPU has no unique predecessor mode"}
        tasks = matrix["tasks"]
        if not isinstance(tasks, list) or len(tasks) != len(set(tasks)) or not OURO_TASKS.issubset(tasks):
            return {"ready": False, "reason": "Predecessor task list is missing the five expected Ouro tasks"}
        statuses = {task: matrix.get("jobs", {}).get(f"{task}_{modes[0]}", {}).get("status", "missing") for task in tasks}
        return {"ready": all(status in TERMINAL for status in statuses.values()),
                "mode": modes[0], "task_statuses": statuses,
                "reason": "All predecessor jobs terminal" if all(status in TERMINAL for status in statuses.values()) else "Waiting for predecessor jobs"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"ready": False, "reason": f"Cannot verify predecessor matrix: {error}"}


def gpu_status(gpu):
    """A device is available only below 1000 MiB and with no compute process."""
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        rows = [row for row in text.splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError("GPU query did not resolve one device")
        uuid, used = [value.strip() for value in rows[0].split(",")]
        memory_mib = int(used)
        applications = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=20)
        pids = []
        for row in applications.splitlines():
            if not row.strip():
                continue
            device, pid = [value.strip() for value in row.split(",")]
            if device == uuid:
                pids.append(int(pid))
        return {"ready": memory_mib < 1000 and not pids, "memory_mib": memory_mib,
                "compute_pids": sorted(pids), "uuid": uuid}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"ready": False, "reason": f"Cannot verify idle GPU: {error}"}


def validate_comparison(path, task, candidate_arm, smoke):
    """An exit code alone is insufficient to release the smoke barrier."""
    result = json.loads(Path(path).read_text())
    if result.get("task") != task or result.get("is_full_split") != (not smoke):
        raise ValueError("Comparison task or full-split flag differs from requested run")
    expected_count = 2 if smoke else None
    if not isinstance(result.get("n_documents"), int) or result["n_documents"] < 1:
        raise ValueError("Comparison has no document evidence")
    if expected_count is not None and result["n_documents"] != expected_count:
        raise ValueError("Smoke comparison must contain exactly two SciQ documents")
    validation = result["validation"]
    for key in ("paired", "model_code_data_prompts_equal", "sample_aggregate_consistency"):
        if validation.get(key) is not True:
            raise ValueError(f"Comparison validation is missing: {key}")
    evidence = validation["native_initialization_stream"]
    if evidence.get("requests", 0) < 1 or not evidence.get("trace_sha256"):
        raise ValueError("Comparison has no native initialization trace evidence")
    baseline = result["baseline"]["guidance"]
    candidate = result["candidate"]["guidance"]
    if baseline.get("mode") != "baseline" or baseline.get("total_loops") != 32:
        raise ValueError("Unexpected baseline guidance configuration")
    expected_loops = 32 if candidate_arm == "hidden32" else 16
    if (candidate.get("mode"), candidate.get("total_loops"), candidate.get("reference_loop"), candidate.get("omega")) != ("hidden", expected_loops, 6, 0.5):
        raise ValueError("Unexpected candidate guidance configuration")
    return {"n_documents": result["n_documents"], "is_full_split": result["is_full_split"],
            "native_trace_sha256": evidence["trace_sha256"]}


class HuginnQueue:
    def __init__(self, args, root=ROOT, gpu_probe=gpu_status):
        self.args, self.root, self.gpu_probe = args, Path(root).resolve(), gpu_probe
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.gate_decided = threading.Event()
        self.cancelled = threading.Event()
        self.processes = {}
        self.environment = {**os.environ, "PYTHONPATH": str(self.root / "src"), "HF_HUB_OFFLINE": "1",
                            "HF_DATASETS_OFFLINE": "1", "HF_MODULES_CACHE": str(self.root / ".cache/modules"),
                            "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        self.state = {
            "schema_version": 1, "status": "waiting", "started_at": now(), "source_root": str(self.root),
            "source_commit": commit, "tasks": list(TASKS), "gpus": dict(zip(ARMS, args.gpus)),
            "wait_for_matrix": str(Path(args.wait_for_matrix).resolve()) if args.wait_for_matrix else None,
            "resource_policy": "Predecessor jobs terminal; GPU memory <1000 MiB; no compute PID; checked before every job",
            "smoke_gate": {"status": "waiting", "task": "sciq", "limit": 2},
            "workers": {arm: {"gpu": gpu, "status": "waiting", "wait_checks": 0} for arm, gpu in zip(ARMS, args.gpus)},
            "jobs": {}, "comparisons": {},
        }
        for arm in ARMS:
            self.state["jobs"][self.job_name("sciq", arm, True)] = {"status": "pending", "phase": "smoke", "task": "sciq", "arm": arm}
            for task in TASKS:
                self.state["jobs"][self.job_name(task, arm)] = {"status": "pending", "phase": "full", "task": task, "arm": arm}
        self._save()

    @staticmethod
    def job_name(task, arm, smoke=False):
        return ("smoke_" if smoke else "") + f"{task}_{arm}"

    def _save(self):
        # All mutations and writes hold self.lock (initial construction is single-threaded).
        self.state["updated_at"] = now()
        temporary = self.output / "matrix.json.tmp"
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.output / "matrix.json")
        if hasattr(self, "condition"):
            with self.condition:
                self.condition.notify_all()

    def cancel(self):
        self.cancelled.set()
        self.gate_decided.set()
        with self.condition:
            self.state["cancellation_requested_at"] = now()
            self._save()

    def _gate_failed(self):
        return self.state["smoke_gate"]["status"] in ("failed", "cancelled")

    def _wait_for_device(self, arm, phase):
        gpu = self.state["gpus"][arm]
        while not self.cancelled.is_set():
            with self.lock:
                if self._gate_failed():
                    return False
            dependency = dependency_status(self.args.wait_for_matrix, gpu)
            availability = self.gpu_probe(gpu) if dependency["ready"] else {"ready": False, "reason": "Predecessor is not terminal"}
            with self.lock:
                worker = self.state["workers"][arm]
                worker.update(status="ready" if dependency["ready"] and availability["ready"] else "waiting_for_device",
                              phase=phase, waiting={"dependency": dependency, "gpu": availability}, checked_at=now())
                worker["wait_checks"] += 1
                self._save()
            if dependency["ready"] and availability["ready"]:
                return True
            event = self.gate_decided if phase == "smoke" else self.cancelled
            event.wait(self.args.poll_seconds)
        return False

    def _execute(self, command, log_path, environment, on_started):
        """Override in CPU tests; production only controls children it starts."""
        with Path(log_path).open("x") as log:
            process = subprocess.Popen(command, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            with self.lock:
                self.processes[process.pid] = process
            on_started(process.pid)
            try:
                while True:
                    if self.cancelled.is_set():
                        process.terminate()
                        try:
                            return process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            return process.wait()
                    try:
                        return process.wait(timeout=min(self.args.poll_seconds, 10))
                    except subprocess.TimeoutExpired:
                        with self.lock:
                            self.state["last_child_poll_at"] = now()
                            self._save()
            finally:
                with self.lock:
                    self.processes.pop(process.pid, None)

    def _run_evaluation(self, task, arm, smoke=False):
        name = self.job_name(task, arm, smoke)
        mode = "baseline" if arm == "baseline32" else "hidden"
        command = [sys.executable, "-u", str(self.root / "scripts/evaluate_huginn.py"), "--model", str(Path(self.args.model).resolve()),
                   "--task", task, "--mode", mode, "--loops", "16" if arm == "hidden16" else "32",
                   "--reference-loop", "6", "--omega", "0.5", "--output", str(self.output / name)]
        if arm == "hidden16":
            command += ["--protocol", "half-depth"]
        if smoke:
            command += ["--limit", "2"]
        if self.args.dataset_cache:
            command += ["--dataset-cache", str(Path(self.args.dataset_cache).resolve())]
        with self.lock:
            self.state["status"] = "running"
            self.state["jobs"][name].update(status="starting", command=command, started_at=now())
            self.state["workers"][arm].update(status="running", current_job=name)
            self._save()

        def started(pid):
            with self.lock:
                self.state["jobs"][name].update(status="running", pid=pid, gpu=self.state["gpus"][arm])
                self._save()

        try:
            code = self._execute(command, self.output / "logs" / f"{name}.log",
                                 {**self.environment, "CUDA_VISIBLE_DEVICES": self.state["gpus"][arm]}, started)
            status = "cancelled" if self.cancelled.is_set() else "completed" if code == 0 else "failed"
            details = {"exit_code": code}
        except Exception as error:
            status, details = "failed", {"error": repr(error)}
        with self.lock:
            self.state["jobs"][name].update(status=status, finished_at=now(), **details)
            self._save()
        return status == "completed"

    def _compare_pair(self, task, candidate, smoke=False):
        name = ("smoke_" if smoke else "") + f"{task}_{candidate}_comparison"
        output = self.output / f"{name}.json"
        command = [sys.executable, str(self.root / "scripts/compare_huginn.py"),
                   "--baseline", str(self.output / self.job_name(task, "baseline32", smoke)),
                   "--candidate", str(self.output / self.job_name(task, candidate, smoke)), "--output", str(output)]
        with self.lock:
            self.state["comparisons"][name] = {"status": "running", "task": task, "candidate": candidate, "smoke": smoke, "started_at": now()}
            self._save()
        try:
            code = self._execute(command, self.output / "logs" / f"{name}.log",
                                 {**self.environment, "CUDA_VISIBLE_DEVICES": ""}, lambda pid: None)
            if code != 0:
                raise RuntimeError(f"compare_huginn exited with code {code}")
            evidence = validate_comparison(output, task, candidate, smoke)
            result = {"status": "completed", "exit_code": code, "output": str(output), "evidence": evidence}
        except Exception as error:
            result = {"status": "failed", "error": repr(error)}
        with self.lock:
            self.state["comparisons"][name].update(**result, finished_at=now())
            self._save()
        return result["status"] == "completed"

    def _compare_ready_tasks(self):
        # Claim each task once under the lock; comparisons run outside it.
        ready = []
        with self.lock:
            for task in TASKS:
                if task in self.state.setdefault("task_comparison_claims", {}):
                    continue
                states = [self.state["jobs"][self.job_name(task, arm)]["status"] for arm in ARMS]
                if not all(status in TERMINAL for status in states):
                    continue
                self.state["task_comparison_claims"][task] = now()
                if all(status == "completed" for status in states):
                    ready.append(task)
                else:
                    for candidate in ARMS[1:]:
                        self.state["comparisons"][f"{task}_{candidate}_comparison"] = {"status": "blocked_by_failed_arm", "task": task}
            self._save()
        for task in ready:
            for candidate in ARMS[1:]:
                self._compare_pair(task, candidate)

    def _worker(self, arm):
        try:
            if not self._wait_for_device(arm, "smoke"):
                return
            self._run_evaluation("sciq", arm, smoke=True)
            with self.lock:
                self.state["workers"][arm].update(status="waiting_for_smoke_gate")
                self._save()
            self.gate_decided.wait()
            with self.lock:
                allowed = self.state["smoke_gate"]["status"] == "passed"
            if not allowed or self.cancelled.is_set():
                return
            for task in TASKS:
                if not self._wait_for_device(arm, "full"):
                    return
                self._run_evaluation(task, arm)
                self._compare_ready_tasks()
            with self.lock:
                self.state["workers"][arm].update(status="completed", finished_at=now())
                self._save()
        except Exception as error:
            with self.lock:
                self.state["workers"][arm].update(status="failed", error=repr(error), finished_at=now())
                for job in self.state["jobs"].values():
                    if job["arm"] == arm and job["status"] not in TERMINAL:
                        job.update(status="failed", error="Worker failed", finished_at=now())
                self._save()

    def run(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self._worker, arm) for arm in ARMS]
            with self.condition:
                while not self.cancelled.is_set():
                    smoke = [self.state["jobs"][self.job_name("sciq", arm, True)]["status"] for arm in ARMS]
                    if any(status in TERMINAL and status != "completed" for status in smoke) or all(status == "completed" for status in smoke):
                        break
                    self.condition.wait(timeout=self.args.poll_seconds)
            if self.cancelled.is_set():
                passed = False
            else:
                with self.lock:
                    passed = all(self.state["jobs"][self.job_name("sciq", arm, True)]["status"] == "completed" for arm in ARMS)
                    self.state["smoke_gate"]["status"] = "validating" if passed else "failed"
                    self._save()
                if passed:
                    # Evaluate both pairs, retaining failure evidence from either.
                    pairs = [self._compare_pair("sciq", candidate, smoke=True) for candidate in ARMS[1:]]
                    passed = all(pairs)
            with self.lock:
                self.state["smoke_gate"].update(status="cancelled" if self.cancelled.is_set() else "passed" if passed else "failed", decided_at=now())
                self._save()
            self.gate_decided.set()
            for future in futures:
                future.result()
        with self.lock:
            for job in self.state["jobs"].values():
                if job["status"] not in TERMINAL:
                    job.update(status="cancelled" if self.cancelled.is_set() else "blocked_by_gate_failure", finished_at=now())
            self._save()
        self._compare_ready_tasks()
        with self.lock:
            completed = self.state["smoke_gate"]["status"] == "passed" and all(job["status"] == "completed" for job in self.state["jobs"].values())
            completed = completed and len(self.state["comparisons"]) == 16 and all(value["status"] == "completed" for value in self.state["comparisons"].values())
            self.state.update(status="cancelled" if self.cancelled.is_set() else "completed" if completed else "failed", finished_at=now())
            self._save()
        return 0 if completed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpus", nargs=3, default=["0", "1", "2"], metavar=("BASELINE32", "HIDDEN32", "HIDDEN16"))
    parser.add_argument("--wait-for-matrix", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if len(set(args.gpus)) != 3 or args.poll_seconds <= 0:
        parser.error("Three distinct GPUs and positive polling interval are required")
    queue = HuginnQueue(args)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: queue.cancel())
    code = queue.run()
    print(json.dumps({"output": str(queue.output), "status": queue.state["status"], "smoke_gate": queue.state["smoke_gate"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
