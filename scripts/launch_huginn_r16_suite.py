"""Run Table 3 R16 arms and strictly compare existing Table A8 h6 evidence.

The launcher has its own release. Evaluation and comparison use an explicitly
pinned, older source tree so that reused runs retain identical source provenance.
No existing run is resumed, copied, relabelled, or overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b88c8cc90c15765f88a11acebf76ed4137935a5c"
TASKS = ("sciq", "piqa", "arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu")
ARMS = ("baseline16", "hidden16")
CANDIDATES = ("hidden16", "half16")
TERMINAL = {"completed", "failed", "cancelled", "blocked_by_failed_worker", "blocked_by_gate_failure", "skipped_smoke_only"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): {"sha256": sha256(p), "size_bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_frozen_source(root, expected_commit):
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise ValueError(f"Source HEAD {actual} differs from expected {expected_commit}")
    tracked = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "ls-tree", "-r", "--name-only", expected_commit], text=True).splitlines()
    expected = {p for p in tracked if p.split("/")[0] in ("src", "scripts", "configs") and Path(p).suffix in (".py", ".json", ".yaml")}
    found = {str(p.relative_to(root)) for folder in ("src", "scripts", "configs") for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".json", ".yaml")}
    if found != expected or not {"scripts/evaluate_huginn.py", "scripts/compare_huginn.py", "configs/huginn_mc.json"}.issubset(found):
        raise ValueError("Frozen source file set differs from the expected Git tree")
    hashes = {}
    for name in sorted(found):
        content = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "show", f"{expected_commit}:{name}"])
        digest = sha256(root / name)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Frozen source differs from Git content: {name}")
        hashes[name] = digest
    return hashes


def gpu_status(gpu):
    try:
        rows = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()
        if len(rows) != 1:
            raise ValueError("GPU query did not identify one device")
        uuid, memory = [s.strip() for s in rows[0].split(",")]
        applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=20)
        pids = [int(row.split(",")[1]) for row in applications.splitlines() if row.strip() and row.split(",")[0].strip() == uuid]
        return {"ready": int(memory) < 1000 and not pids, "memory_mib": int(memory), "compute_pids": pids, "uuid": uuid}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"ready": False, "error": repr(error)}


def validate_comparison(path, task, candidate, smoke):
    result = json.loads(Path(path).read_text())
    if result.get("task") != task or result.get("is_full_split") is not (not smoke):
        raise ValueError("Comparison task/full-split mismatch")
    if smoke:
        if result.get("n_documents") != 2:
            raise ValueError("Smoke must pair exactly two SciQ documents")
    elif result.get("full_split_count_verified") is not True:
        raise ValueError("Full split counts were not verified")
    if type(result.get("n_documents")) is not int or result["n_documents"] < 1:
        raise ValueError("Comparison has no document evidence")
    for key in ("paired", "model_code_data_prompts_equal", "sample_aggregate_consistency"):
        if result.get("validation", {}).get(key) is not True:
            raise ValueError(f"Missing strict comparison check: {key}")
    trace = result["validation"]["native_initialization_stream"]
    if trace.get("requests", 0) < 1 or not trace.get("trace_sha256"):
        raise ValueError("Missing native initialization evidence")
    for role, mode, reference in (("baseline", "baseline", 7), ("candidate", "hidden", 7 if candidate == "hidden16" else 6)):
        g = result[role]["guidance"]
        if (g.get("mode"), g.get("total_loops"), g.get("reference_loop"), g.get("omega")) != (mode, 16, reference, 0.5):
            raise ValueError(f"Incorrect {role} R16 guidance")
    return {"n_documents": result["n_documents"], "is_full_split": result["is_full_split"], "trace_sha256": trace["trace_sha256"]}


class R16Queue:
    def __init__(self, args, gpu_probe=gpu_status):
        if len(args.gpus) != 2 or len(set(args.gpus)) != 2 or args.poll_seconds <= 0:
            raise ValueError("Two distinct GPUs and a positive polling interval are required")
        self.args, self.gpu_probe = args, gpu_probe
        self.source = Path(args.source_root).resolve()
        self.output = Path(args.output).resolve()
        self.old_suite = Path(args.half_depth_suite).resolve()
        self.source_hashes = verify_frozen_source(self.source, args.source_commit)
        self.launcher_hash = sha256(__file__)
        self.old_pins = {}
        for smoke, task in [(True, "sciq")] + [(False, task) for task in TASKS]:
            directory = self.old_directory(task, smoke)
            m = json.loads((directory / "manifest.json").read_text())
            g = m["guidance"]
            if (m.get("status"), m.get("task"), m.get("limit"), m.get("is_full_split")) != ("completed", task, 2 if smoke else None, not smoke):
                raise ValueError(f"Reused run is incomplete or has wrong split: {directory}")
            if (g.get("mode"), g.get("total_loops"), g.get("reference_loop"), g.get("omega")) != ("hidden", 16, 6, .5):
                raise ValueError(f"Reused run does not use Table A8 h6/.5: {directory}")
            if m["provenance"]["source_sha256"] != self.source_hashes or m.get("seed") != args.seed:
                raise ValueError(f"Reused source/seed differs from frozen evaluator: {directory}")
            if m.get("protocol") != "half-depth":
                raise ValueError(f"Reused manifest is not marked half-depth: {directory}")
            self.old_pins[str(directory)] = inventory(directory)
        # Fresh-only output; initialization never touches the reused suite.
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.gate_decided = threading.Event()
        self.cancelled = threading.Event()
        self.abort = threading.Event()
        self.stop_worker = {arm: threading.Event() for arm in ARMS}
        self.environment = {**os.environ, "PYTHONPATH": str(self.source / "src"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                            "HF_MODULES_CACHE": str(self.output / ".cache/modules"), "TOKENIZERS_PARALLELISM": "false",
                            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        try:
            launcher_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            launcher_commit = None
        self.state = {"schema_version": 1, "status": "waiting", "started_at": now(), "tasks": list(TASKS),
                      "source_root": str(self.source), "source_commit": args.source_commit, "source_sha256": self.source_hashes,
                      "launcher": {"path": str(Path(__file__).resolve()), "sha256": self.launcher_hash, "release": str(ROOT), "commit": launcher_commit},
                      "gpus": dict(zip(ARMS, args.gpus)), "seed": args.seed, "smoke_only": args.smoke_only,
                      "resource_policy": "Before every GPU job: memory below 1000 MiB and no compute PID; never kill foreign processes",
                      "half_depth_suite": str(self.old_suite), "reused_run_files": self.old_pins,
                      "reused_comparison_meaning": "R16 baseline versus Table A8 h6 at R16 is an auxiliary same-depth comparison, not the original R32-to-R16 Figure 4 comparison",
                      "smoke_gate": {"status": "waiting", "task": "sciq", "limit": 2},
                      "workers": {a: {"status": "waiting", "gpu": g} for a, g in zip(ARMS, args.gpus)}, "jobs": {}, "comparisons": {}}
        for arm in ARMS:
            for smoke, task in [(True, "sciq")] + [(False, t) for t in TASKS]:
                self.state["jobs"][self.name(task, arm, smoke)] = {"status": "pending", "task": task, "arm": arm, "phase": "smoke" if smoke else "full"}
        self._save()

    @staticmethod
    def name(task, arm, smoke=False):
        return ("smoke_" if smoke else "") + f"{task}_{arm}"

    def old_directory(self, task, smoke=False):
        return self.old_suite / self.name(task, "hidden16", smoke)

    def _save(self):
        with self.condition:
            self.state["updated_at"] = now()
            temporary = self.output / "matrix.json.tmp"
            temporary.write_text(json.dumps(self.state, indent=2) + "\n")
            temporary.replace(self.output / "matrix.json")
            self.condition.notify_all()

    def cancel(self):
        self.cancelled.set()
        self.gate_decided.set()

    def _assert_sources(self):
        found = {str(p.relative_to(self.source)): sha256(p) for folder in ("src", "scripts", "configs")
                 for p in (self.source / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml")}
        if found != self.source_hashes or sha256(__file__) != self.launcher_hash:
            self.abort.set()
            raise ValueError("Pinned source or launcher changed after startup")

    def _wait_device(self, arm):
        while not self.cancelled.is_set() and not self.abort.is_set() and not self.stop_worker[arm].is_set():
            with self.lock:
                if self.state["smoke_gate"]["status"] == "failed":
                    return False
            available = self.gpu_probe(self.state["gpus"][arm])
            with self.lock:
                self.state["workers"][arm].update(status="ready" if available["ready"] else "waiting_for_device", checked_at=now(), device=available)
                self._save()
            if available["ready"]:
                return True
            self.cancelled.wait(self.args.poll_seconds)
        return False

    def _execute(self, command, log, environment, on_started):
        with Path(log).open("x") as stream:
            child = subprocess.Popen(command, cwd=self.source, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            on_started(child.pid)
            while True:
                if self.cancelled.is_set() or self.abort.is_set():
                    child.terminate()
                    try:
                        return child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        return child.wait()
                try:
                    return child.wait(timeout=min(10, self.args.poll_seconds))
                except subprocess.TimeoutExpired:
                    with self.lock:
                        self.state["last_child_poll_at"] = now()
                        self._save()

    def _evaluate(self, task, arm, smoke=False):
        self._assert_sources()
        name = self.name(task, arm, smoke)
        command = [sys.executable, "-u", str(self.source / "scripts/evaluate_huginn.py"), "--model", str(Path(self.args.model).resolve()),
                   "--task", task, "--mode", "baseline" if arm == "baseline16" else "hidden", "--loops", "16",
                   "--reference-loop", "7", "--omega", "0.5", "--protocol", "standard", "--seed", str(self.args.seed), "--output", str(self.output / name)]
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
            code = self._execute(command, self.output / "logs" / f"{name}.log", {**self.environment, "CUDA_VISIBLE_DEVICES": self.state["gpus"][arm]}, started)
            self._assert_sources()
            manifest = json.loads((self.output / name / "manifest.json").read_text()) if code == 0 else {}
            if code == 0 and manifest.get("status") != "completed":
                raise ValueError("Successful process did not write a completed manifest")
            result = {"status": "completed" if code == 0 else "failed", "exit_code": code}
        except Exception as error:
            result = {"status": "failed", "error": repr(error)}
        with self.lock:
            self.state["jobs"][name].update(**result, finished_at=now())
            self._save()
        return result["status"] == "completed"

    def _compare(self, task, candidate, smoke=False):
        name = self.name(task, candidate, smoke) + "_comparison"
        baseline = self.output / self.name(task, "baseline16", smoke)
        target = self.output / self.name(task, "hidden16", smoke) if candidate == "hidden16" else self.old_directory(task, smoke)
        output = self.output / f"{name}.json"
        command = [sys.executable, str(self.source / "scripts/compare_huginn.py"), "--baseline", str(baseline), "--candidate", str(target), "--output", str(output)]
        with self.lock:
            self.state["comparisons"][name] = {"status": "running", "task": task, "candidate": candidate, "smoke": smoke, "command": command, "started_at": now()}
            self._save()
        try:
            self._assert_sources()
            inputs = {str(p): inventory(p) for p in (baseline, target)}
            if candidate == "half16" and inputs[str(target)] != self.old_pins[str(target)]:
                raise ValueError("Reused evidence changed after startup")
            code = self._execute(command, self.output / "logs" / f"{name}.log", {**self.environment, "CUDA_VISIBLE_DEVICES": ""}, lambda pid: None)
            self._assert_sources()
            if code != 0:
                raise ValueError(f"Strict comparator exited with code {code}")
            if any(inventory(Path(p)) != pins for p, pins in inputs.items()):
                raise ValueError("Comparison inputs changed while comparing")
            evidence = validate_comparison(output, task, candidate, smoke)
            actual = json.loads(output.read_text())
            if Path(actual["baseline"]["path"]).resolve() != baseline or Path(actual["candidate"]["path"]).resolve() != target:
                raise ValueError("Comparator reported unexpected input paths")
            result = {"status": "completed", "exit_code": code, "output": str(output), "sha256": sha256(output), "input_files": inputs, "evidence": evidence}
        except Exception as error:
            result = {"status": "failed", "error": repr(error)}
            if not smoke:
                self.stop_worker["hidden16" if candidate == "hidden16" else "baseline16"].set()
        with self.lock:
            self.state["comparisons"][name].update(**result, finished_at=now())
            self._save()
        return result["status"] == "completed"

    def _compare_ready(self):
        ready = []
        with self.lock:
            for task in TASKS:
                for candidate in CANDIDATES:
                    name = self.name(task, candidate) + "_comparison"
                    if name in self.state["comparisons"]:
                        continue
                    needed = [self.state["jobs"][self.name(task, "baseline16")]["status"]]
                    if candidate == "hidden16":
                        needed.append(self.state["jobs"][self.name(task, "hidden16")]["status"])
                    if all(s in TERMINAL for s in needed):
                        self.state["comparisons"][name] = {"status": "claimed" if all(s == "completed" for s in needed) else "blocked_by_failed_arm", "task": task}
                        if all(s == "completed" for s in needed):
                            ready.append((task, candidate))
            self._save()
        for task, candidate in ready:
            self._compare(task, candidate)

    def _worker(self, arm):
        try:
            if not self._wait_device(arm):
                return
            if not self._evaluate("sciq", arm, True):
                self.stop_worker[arm].set()
            self.gate_decided.wait()
            if self.state["smoke_gate"]["status"] != "passed" or self.args.smoke_only:
                return
            for task in TASKS:
                if not self._wait_device(arm):
                    break
                if not self._evaluate(task, arm):
                    self.stop_worker[arm].set()
                self._compare_ready()
        except Exception as error:
            self.stop_worker[arm].set()
            with self.lock:
                self.state["workers"][arm]["error"] = repr(error)
        finally:
            with self.lock:
                status = "cancelled" if self.cancelled.is_set() else "failed" if self.stop_worker[arm].is_set() or self.abort.is_set() else "completed"
                self.state["workers"][arm].update(status=status, finished_at=now())
                for job in self.state["jobs"].values():
                    if job["arm"] == arm and job["status"] not in TERMINAL:
                        reason = "cancelled" if self.cancelled.is_set() else "skipped_smoke_only" if self.args.smoke_only and self.state["smoke_gate"]["status"] == "passed" else "blocked_by_gate_failure" if self.state["smoke_gate"]["status"] != "passed" else "blocked_by_failed_worker"
                        job.update(status=reason, finished_at=now())
                self._save()

    def run(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self._worker, arm) for arm in ARMS]
            with self.condition:
                while not self.cancelled.is_set() and not self.abort.is_set():
                    smoke = [self.state["jobs"][self.name("sciq", a, True)]["status"] for a in ARMS]
                    if any(s in TERMINAL and s != "completed" for s in smoke) or all(s == "completed" for s in smoke):
                        break
                    self.condition.wait(timeout=self.args.poll_seconds)
            passed = not self.cancelled.is_set() and not self.abort.is_set() and all(self.state["jobs"][self.name("sciq", a, True)]["status"] == "completed" for a in ARMS)
            if passed:
                passed = all([self._compare("sciq", c, True) for c in CANDIDATES])
            with self.lock:
                self.state["smoke_gate"].update(status="passed" if passed else "failed", decided_at=now())
                self._save()
            self.gate_decided.set()
            for future in futures:
                future.result()
        if not self.args.smoke_only:
            self._compare_ready()
        expected_comparisons = 2 if self.args.smoke_only else 16
        complete = passed and len(self.state["comparisons"]) == expected_comparisons and all(c["status"] == "completed" for c in self.state["comparisons"].values())
        if not self.args.smoke_only:
            complete = complete and all(j["status"] == "completed" for j in self.state["jobs"].values())
        with self.lock:
            self.state.update(status="cancelled" if self.cancelled.is_set() else "completed" if complete else "failed", finished_at=now(), full_suite_verified=complete and not self.args.smoke_only)
            self._save()
        return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--half-depth-suite", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"], metavar=("BASELINE16", "HIDDEN16"))
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    queue = R16Queue(args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda number, frame: queue.cancel())
    code = queue.run()
    print(json.dumps({"output": str(queue.output), "status": queue.state["status"], "full_suite_verified": queue.state["full_suite_verified"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
