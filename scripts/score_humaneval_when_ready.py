"""Wait for four complete paired generations, then score and compare serially.

This coordinator belongs to an immutable release. The isolated scorer lives
in the root project because its runtime manifest binds that exact location;
its source hash is pinned when this coordinator starts and rechecked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ARMS = ("baseline32", "hidden32", "baseline16", "hidden16")
RELEASE = Path(__file__).resolve().parent.parent


class GenerationFailed(RuntimeError):
    pass


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_hashes(pinned):
    for filename, expected in pinned.items():
        path = Path(filename)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Pinned source or input changed: {filename}")


def generation_snapshot(directory, allow_subset=False):
    """Read manifests first; only parse sample files once completion is asserted."""
    summary, manifests = {}, {}
    for arm in ARMS:
        path = directory / arm / "manifest.json"
        if not path.is_file():
            summary[arm] = {"status": "manifest_missing"}
            continue
        manifest = json.loads(path.read_text())
        status = manifest.get("status")
        summary[arm] = {key: manifest.get(key) for key in
                        ("status", "expected_samples", "completed_samples", "updated_at", "is_full_split")}
        if status in {"failed", "error", "cancelled", "canceled", "interrupted"}:
            raise GenerationFailed(f"{arm} generation ended with {status}: {manifest.get('error')}")
        if status not in {"running", "completed", "queued", "waiting", "initializing"}:
            raise ValueError(f"Unexpected generation status for {arm}: {status!r}")
        if not allow_subset and manifest.get("is_full_split") is not True:
            raise ValueError(f"{arm} is a debug subset; --allow-subset is required")
        manifests[arm] = manifest
    if len(manifests) != 4 or any(m["status"] != "completed" for m in manifests.values()):
        return False, summary, {}
    pins, shared_ids = {}, None
    for arm, manifest in manifests.items():
        config = manifest.get("config", {})
        ids = config.get("task_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid expected task IDs: {arm}")
        if not allow_subset and (ids != [f"HumanEval/{i}" for i in range(164)] or config.get("limit") is not None):
            raise ValueError(f"{arm} is not the complete fixed 164-task split")
        if manifest.get("completed_samples") != len(ids) or manifest.get("expected_samples") != len(ids):
            raise ValueError(f"Completed manifest has incomplete counts: {arm}")
        path = directory / arm / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        observed = [row.get("task_id") for row in rows]
        if len(observed) != len(ids) or len(set(observed)) != len(observed) or set(observed) != set(ids):
            raise ValueError(f"Completed sample file has missing, duplicated, or unknown tasks: {arm}")
        if shared_ids is not None and ids != shared_ids:
            raise ValueError("Four arms do not contain the same ordered task split")
        shared_ids = ids
        for name in ("manifest.json", "samples.jsonl"):
            filename = directory / arm / name
            pins[str(filename)] = sha(filename)
        summary[arm]["verified_unique_samples"] = len(rows)
    return True, summary, pins


def load_comparer(path):
    spec = importlib.util.spec_from_file_location("_frozen_humaneval_comparer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(command, output, label, state, save_state, timeout=None):
    stdout_path, stderr_path = output / f"{label}.stdout.log", output / f"{label}.stderr.log"
    entry = {"label": label, "command": [str(value) for value in command], "started_at": timestamp(),
             "stdout": str(stdout_path), "stderr": str(stderr_path), "returncode": None}
    state["commands"].append(entry)
    save_state()
    started = time.monotonic()
    try:
        with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
            completed = subprocess.run(entry["command"], stdout=stdout, stderr=stderr, timeout=timeout)
        entry["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"{label} failed with exit code {completed.returncode}; inspect {stderr_path}")
    finally:
        entry.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        save_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new directory for status, score JSONs, logs, and comparison")
    parser.add_argument("--sandbox-project", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly permit debug subsets; never label them full results")
    parser.add_argument("--score-timeout", type=int, default=1800, help="Outer sandbox wall-time limit per arm")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.score_timeout <= 0:
        parser.error("poll-seconds must be in [1,60] and score-timeout must be positive")
    args.generation, args.output = args.generation.resolve(), args.output.resolve()
    args.sandbox_project = args.sandbox_project.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "status.json"
    state = {"status": "waiting", "phase": "waiting", "started_at": timestamp(),
             "generation": str(args.generation), "output": str(args.output),
             "sandbox_project": str(args.sandbox_project), "release": str(RELEASE),
             "allow_subset": args.allow_subset, "poll_seconds": args.poll_seconds,
             "score_timeout": args.score_timeout, "commands": []}

    def save_state():
        state["updated_at"] = timestamp()
        atomic_json(state_path, state)

    save_state()
    try:
        scorer = args.sandbox_project / "scripts/run_eval_sandbox.py"
        comparer_path = RELEASE / "scripts/compare_humaneval.py"
        source_pins = {str(path): sha(path) for path in (Path(__file__).resolve(), scorer, comparer_path)}
        state["source_sha256"] = source_pins
        save_state()
        while True:
            verify_hashes(source_pins)
            ready, summary, input_pins = generation_snapshot(args.generation, args.allow_subset)
            state["generation_snapshot"] = summary
            save_state()
            if ready:
                break
            time.sleep(args.poll_seconds)

        # Run the frozen comparison module's strict generation validation
        # before spending time scoring. It verifies provenance and protocol.
        state.update(status="validating", phase="validating", generation_sha256=input_pins)
        save_state()
        comparer = load_comparer(comparer_path)
        generations = {arm: comparer.read_generation(args.generation / arm) for arm in ARMS}
        if not args.allow_subset and any(not item["manifest"]["is_full_split"] or len(item["rows"]) != 164 for item in generations.values()):
            raise ValueError("Strict generation validation did not establish four full 164-task runs")
        pins = {**source_pins, **input_pins}
        state.update(status="scoring", phase="scoring")
        save_state()
        scores = []
        for arm in ARMS:
            verify_hashes(pins)
            score = args.output / f"{arm}.json"
            if score.exists():
                raise FileExistsError(f"Refusing to overwrite score evidence: {score}")
            state["active_arm"] = arm
            execute(["/usr/bin/python3", scorer, "--samples", args.generation / arm / "samples.jsonl",
                     "--output", score, "--timeout", str(args.score_timeout)], args.output,
                    "score_" + arm, state, save_state)
            verify_hashes(pins)
            comparer.read_scores(score, generations[arm])
            scores.append(score)
            state.setdefault("completed_arms", []).append(arm)
            save_state()
        state.update(status="comparing", phase="comparing", active_arm=None)
        save_state()
        verify_hashes(pins)
        comparison = args.output / "comparison.json"
        execute([sys.executable, comparer_path, "--runs", *[args.generation / arm for arm in ARMS],
                 "--scores", *scores, "--output", comparison], args.output,
                "compare", state, save_state, timeout=120)
        verify_hashes(pins)
        result = json.loads(comparison.read_text())
        if not args.allow_subset and (not result.get("full_164_verified") or result.get("n_tasks") != 164):
            raise ValueError("Comparison failed to establish the complete 164-task benchmark")
        state.update(status="completed", phase="completed", finished_at=timestamp(),
                     comparison=str(comparison), comparison_sha256=sha(comparison),
                     n_tasks=result["n_tasks"], full_164_verified=result["full_164_verified"],
                     score_sha256={str(path): sha(path) for path in scores})
        save_state()
        print(json.dumps({"status": "completed", "comparison": str(comparison),
                          "n_tasks": state["n_tasks"], "full_164_verified": state["full_164_verified"]}), flush=True)
    except BaseException as error:
        previous_phase = state["phase"]
        state.update(status="generation_failed" if isinstance(error, GenerationFailed) else
                     "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     failed_phase=previous_phase, phase="stopped", finished_at=timestamp(),
                     error=repr(error), traceback=traceback.format_exc())
        save_state()
        print(json.dumps({"status": state["status"], "failed_phase": previous_phase,
                          "error": str(error), "status_file": str(state_path)}), flush=True)
        raise


if __name__ == "__main__":
    main()
