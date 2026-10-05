"""Run a frozen three-arm MC matrix and compare each task after all arms finish."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu"])
    p.add_argument("--gpus", nargs=3, default=["0", "1", "2"])
    p.add_argument("--dataset-cache", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if len(set(args.gpus)) != 3:
        p.error("Three distinct GPUs are required")
    for gpu in args.gpus:
        used = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if used > 1000:
            raise RuntimeError(f"GPU {gpu} is busy ({used} MiB)")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_MODULES_CACHE=str(root / ".cache/modules"), TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
    modes = ["baseline", "fixed", "adaptive"]
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "source_root": str(root), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
             "gpus": dict(zip(modes, args.gpus)), "tasks": args.tasks, "jobs": {}, "comparisons": {}}
    lock = threading.Lock()
    def save():
        tmp = args.output / "matrix.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.output / "matrix.json")
    save()
    def worker(mode, gpu):
        worker_env = {**env, "CUDA_VISIBLE_DEVICES": gpu}
        for task in args.tasks:
            name = f"{task}_{mode}"
            cmd = [sys.executable, "-u", str(root / "scripts/evaluate.py"), "--model", str(args.model.resolve()),
                   "--task", task, "--mode", mode, "--output", str(args.output / name)]
            if args.dataset_cache:
                cmd += ["--dataset-cache", str(args.dataset_cache.resolve())]
            with (args.output / "logs" / f"{name}.log").open("w") as log:
                process = subprocess.Popen(cmd, cwd=root, env=worker_env, stdout=log, stderr=subprocess.STDOUT)
                with lock:
                    state["jobs"][name] = {"status": "running", "pid": process.pid, "gpu": gpu,
                                           "started_at": datetime.now(timezone.utc).isoformat()}
                    save()
                code = process.wait()
            with lock:
                state["jobs"][name].update(status="completed" if code == 0 else "failed", exit_code=code,
                                           finished_at=datetime.now(timezone.utc).isoformat())
                save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, mode, gpu) for mode, gpu in zip(modes, args.gpus)]
        for future in futures:
            future.result()
    for task in args.tasks:
        if not all(state["jobs"][f"{task}_{mode}"]["status"] == "completed" for mode in modes):
            state["comparisons"][task] = {"status": "blocked_by_failed_arm"}
            continue
        cmd = [sys.executable, str(root / "scripts/compare.py"), "--runs"]
        cmd += [str(args.output / f"{task}_{mode}") for mode in modes]
        cmd += ["--output", str(args.output / f"{task}_comparison.json")]
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        (args.output / "logs" / f"{task}_compare.log").write_text(result.stdout + result.stderr)
        state["comparisons"][task] = {"status": "completed" if result.returncode == 0 else "failed", "exit_code": result.returncode}
    passed = all(x["status"] == "completed" for x in state["jobs"].values()) and all(x["status"] == "completed" for x in state["comparisons"].values())
    state.update(status="completed" if passed else "failed", finished_at=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(state), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
