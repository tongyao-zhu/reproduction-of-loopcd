"""Run official EvalPlus workers in a verified private Linux chroot."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import grp
import hashlib
import importlib.util
import json
import os
import pwd
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import types
import uuid


PROJECT = Path(__file__).resolve().parent.parent
SANDBOX = PROJECT / ".sandbox/evalplus"
COMMON_DENIED = (
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg",
    "execve", "execveat", "ptrace", "process_vm_readv", "process_vm_writev",
    "mount", "umount2", "pivot_root", "chroot", "unshare", "setns",
    "open_by_handle_at", "name_to_handle_at", "bpf", "perf_event_open",
    "userfaultfd", "io_uring_setup", "io_uring_enter", "io_uring_register",
    "init_module", "finit_module", "delete_module", "kexec_load", "reboot",
    "keyctl", "add_key", "request_key", "mknod", "mknodat", "capset",
    "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid",
    "setfsuid", "setfsgid", "setgroups", "setpgid", "setsid", "clone3",
)
WORKER_DENIED = ("clone", "fork", "vfork", "kill", "tkill", "tgkill", "pidfd_open", "pidfd_send_signal", "pidfd_getfd")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_seccomp(worker=False):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
        raise OSError("Cannot establish no_new_privs")
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in COMMON_DENIED + (WORKER_DENIED if worker else ()):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0):
                raise RuntimeError(f"Cannot deny syscall {name}")
        # clone3 is denied; trusted coordinator's ordinary fork remains usable.
        # Every untrusted worker adds a filter denying clone/fork entirely.
        if lib.seccomp_load(context):
            raise RuntimeError("seccomp_load failed")
    finally:
        lib.seccomp_release(context)


def capability_bits():
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    if ctypes.CDLL(None).capget(ctypes.byref(header), ctypes.byref(data)):
        raise RuntimeError("Cannot verify cleared capabilities")
    return [int(getattr(item, field)) for item in data for field in ("effective", "permitted", "inheritable")]


def inside_selftest(pipe, host_canary, uid, gid):
    try:
        install_seccomp(worker=True)
        import socket
        checks = {"uid_nonroot": os.getuid() == uid and uid != 0 and os.getgid() == gid,
                  "groups_empty": os.getgroups() == [], "capabilities_empty": not any(capability_bits()),
                  "no_new_privs": ctypes.CDLL(None).prctl(39, 0, 0, 0, 0) == 1}
        operations = {
            "host_read_blocked": lambda: open(host_canary).read(),
            "host_write_blocked": lambda: open(host_canary + ".write", "w"),
            "network_blocked": lambda: socket.socket(),
            "fork_blocked": lambda: os.fork(),
            "exec_blocked": lambda: os.execv("/usr/bin/python3", ["python3", "-c", "pass"]),
            "signal_blocked": lambda: os.kill(os.getpid(), 0),
            "chroot_blocked": lambda: os.chroot("/"),
        }
        for name, operation in operations.items():
            try:
                operation()
            except OSError as error:
                checks[name] = error.errno in (errno.EPERM, errno.EACCES, errno.ENOENT)
            else:
                checks[name] = False
        pipe.send({"checks": checks, "passed": all(checks.values())})
    except BaseException as error:
        pipe.send({"passed": False, "error": repr(error)})
    finally:
        pipe.close()


def inside(config_path):
    config = json.loads(Path(config_path).read_text())
    if os.geteuid() != config["uid"] or not os.geteuid() or os.getgroups() or any(capability_bits()):
        raise RuntimeError("Privilege drop not established")
    sys.path.insert(0, "/opt/site")
    import multiprocessing
    import numpy  # preload native extensions before seccomp
    import psutil
    import evalplus.eval as evaluator
    from evalplus.gen.util import trusted_exec
    # Bypass only data/__init__.py's unrelated EvalPerf/datasets import. The
    # official HumanEval loader and data utilities themselves are unchanged.
    data_package = types.ModuleType("evalplus.data")
    data_package.__path__ = ["/opt/site/evalplus/data"]
    sys.modules["evalplus.data"] = data_package
    from evalplus.data.humaneval import get_human_eval_plus
    problems = get_human_eval_plus()
    original = evaluator.unsafe_execute

    def guarded_execute(*args, **kwargs):
        install_seccomp(worker=True)
        return original(*args, **kwargs)

    evaluator.unsafe_execute = guarded_execute
    upstream_spec = importlib.util.spec_from_file_location("_evalplus_upstream", "/audit/eval_init.original.py")
    upstream = importlib.util.module_from_spec(upstream_spec)
    upstream_spec.loader.exec_module(upstream)
    upstream_unsafe = upstream.unsafe_execute

    def guarded_upstream(*args, **kwargs):
        install_seccomp(worker=True)
        return upstream_unsafe(*args, **kwargs)

    upstream.unsafe_execute = guarded_upstream
    multiprocessing.set_start_method("fork", force=True)
    install_seccomp()
    receive, send = multiprocessing.Pipe(duplex=False)
    child = multiprocessing.Process(target=inside_selftest, args=(send, config["host_canary"], config["uid"], config["gid"]))
    child.start()
    send.close()
    if not receive.poll(15):
        child.kill()
        child.join()
        raise RuntimeError("Isolation self-test timed out")
    safety = receive.recv()
    child.join(5)
    if child.is_alive():
        child.kill()
        child.join()
        raise RuntimeError("Isolation self-test child did not exit")
    if not safety.get("passed"):
        raise RuntimeError(f"Isolation self-test failed: {safety}")
    result = {"status": "RUNNING", "safety": safety,
              "scorer": "EvalPlus 0.3.1 with private find_zero progress bookkeeping repair; mathematical predicate unchanged",
              "evaluator_patch": config["evaluator_patch"],
              "numpy": numpy.__version__, "psutil": psutil.__version__, "rows": [],
              "evaluation_complete": False}
    if config["self_test"]:
        status, details = evaluator.untrusted_check(
            "humaneval", "def add(a, b):\n    return a + b\n", [[1, 2], [-3, 4]],
            "add", [3, 1], 0, [0.001, 0.001], fast_check=False)
        result["benign_fixture"] = {"status": status, "details": [bool(value) for value in details]}
        if status != evaluator.PASS or not all(details):
            raise RuntimeError(f"Official benign EvalPlus fixture failed: {result['benign_fixture']}")
        result["canonical_fixtures"] = []
        for task_id in ("HumanEval/0", "HumanEval/32"):
            problem = problems[task_id]
            code = problem["prompt"] + problem["canonical_solution"]
            row = {"task_id": task_id, "entry_point": problem["entry_point"]}
            for suite in ("base", "plus"):
                expected, durations = trusted_exec(code, problem[suite + "_input"], problem["entry_point"], record_time=True)
                status, details = evaluator.untrusted_check("humaneval", code, problem[suite + "_input"],
                    problem["entry_point"], expected, problem["atol"], durations, fast_check=False)
                row[suite] = {"status": status, "details_returned": len(details), "tests": len(problem[suite + "_input"])}
                if status != evaluator.PASS:
                    raise RuntimeError(f"Canonical fixture failed: {row}")
                if task_id == "HumanEval/32":
                    raw_status, raw_details = upstream.untrusted_check("humaneval", code, problem[suite + "_input"],
                        problem["entry_point"], expected, problem["atol"], durations, fast_check=False)
                    row[suite]["upstream_status"] = raw_status
                    row[suite]["upstream_details_returned"] = len(raw_details)
                    bad_status, _ = evaluator.untrusted_check("humaneval", "def find_zero(xs):\n    return float('nan')\n", problem[suite + "_input"],
                        problem["entry_point"], expected, problem["atol"], durations, fast_check=False)
                    row[suite]["invalid_root_status"] = bad_status
                    if bad_status == evaluator.PASS:
                        raise RuntimeError("find_zero invalid root incorrectly passes")
            result["canonical_fixtures"].append(row)
    else:
        samples = ([{"task_id": task_id, "solution": problem["prompt"] + problem["canonical_solution"]}
                    for task_id, problem in problems.items()] if config["canonical_all"] else
                   [json.loads(line) for line in Path(config["samples"]).read_text().splitlines() if line])
        result["expected_rows"] = len(samples)
        expected = {}
        for sample in samples:
            problem = problems[sample["task_id"]]
            task_id = sample["task_id"]
            if task_id not in expected:
                expected[task_id] = {suite: trusted_exec(
                    problem["prompt"] + problem["canonical_solution"], problem[suite + "_input"],
                    problem["entry_point"], record_time=True) for suite in ("base", "plus")}
            row = {"task_id": task_id, "sample_id": sample.get("sample_id", 0)}
            for suite in ("base", "plus"):
                answers, durations = expected[task_id][suite]
                status, details = evaluator.untrusted_check(
                    "humaneval", sample["solution"], problem[suite + "_input"], problem["entry_point"],
                    answers, problem["atol"], durations, fast_check=False)
                row[suite] = {"status": status, "passed": status == evaluator.PASS,
                              "details": [bool(value) for value in details], "tests": len(problem[suite + "_input"])}
                if task_id == "HumanEval/32":
                    raw_status, raw_details = upstream.untrusted_check(
                        "humaneval", sample["solution"], problem[suite + "_input"], problem["entry_point"],
                        answers, problem["atol"], durations, fast_check=False)
                    row[suite]["upstream_status"] = raw_status
                    row[suite]["upstream_details"] = [bool(value) for value in raw_details]
            row["plus_passed"] = row["base"]["passed"] and row["plus"]["passed"]
            result["rows"].append(row)
            Path(config["result"]).write_text(json.dumps(result) + "\n")
    result["status"] = "PASS"
    result["completed_rows"] = len(result["rows"])
    result["evaluation_complete"] = True
    if config["canonical_all"] and (len(result["rows"]) != 164 or not all(row["plus_passed"] for row in result["rows"])):
        result["status"] = "FAIL"
        result["error"] = "Canonical full-suite self-check did not pass every task"
    Path(config["result"]).write_text(json.dumps(result, indent=2) + "\n")


def verify_runtime(base):
    manifest_path = base / "manifest.json"
    if (base / "PREPARATION_FAILED").exists():
        raise RuntimeError("Sandbox preparation failed")
    manifest = json.loads(manifest_path.read_text())
    root = base / "rootfs"
    for relative, expected in manifest["immutable_sha256"].items():
        path = root / relative
        if path.resolve() != path.absolute() or not path.is_relative_to(root):
            raise RuntimeError(f"Sandbox path is not a direct private runtime file: {relative}")
        for parent in [path.parent, *path.parents]:
            if parent == root.parent:
                break
            if parent.stat().st_uid != 0 or parent.stat().st_mode & 0o022:
                raise RuntimeError(f"Sandbox runtime directory is writable: {parent}")
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != 0 or path.stat().st_mode & 0o022 or sha(path) != expected:
            raise RuntimeError(f"Sandbox runtime integrity failed: {relative}")
    if sha(root / "runner.py") != sha(Path(__file__)):
        raise RuntimeError("Sandbox runner differs from current audited source")
    return root, sha(manifest_path)


def child_setup(root, cpu_seconds, uid, gid):
    os.chroot(root)
    os.chdir("/")
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    if any(capability_bits()) or ctypes.CDLL(None).prctl(38, 1, 0, 0, 0):
        raise RuntimeError("Cannot drop privileges/no_new_privs")
    for kind, value in ((resource.RLIMIT_AS, 5 * 1024**3), (resource.RLIMIT_CPU, cpu_seconds),
                        (resource.RLIMIT_NPROC, 32), (resource.RLIMIT_NOFILE, 128),
                        (resource.RLIMIT_FSIZE, 32 * 1024**2), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (value, value))


def verify_identity(identity):
    uid, gid = identity["uid"], identity["gid"]
    if not (60000 <= uid < 65000 and gid == uid):
        raise RuntimeError("Unexpected dedicated sandbox identity")
    try:
        pwd.getpwuid(uid)
    except KeyError:
        pass
    else:
        raise RuntimeError("Sandbox UID has acquired a host account")
    try:
        grp.getgrgid(gid)
    except KeyError:
        pass
    else:
        raise RuntimeError("Sandbox GID has acquired a host group")
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in path.read_text().splitlines():
                if line.startswith(("Uid:", "Gid:")) and uid in [int(x) for x in line.split()[1:]]:
                    raise RuntimeError(f"Dedicated sandbox identity is already active: {path.parent.name}")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return uid, gid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--canonical-all", action="store_true", help="Evaluate all 164 pinned canonical solutions as a scorer self-check")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.inside:
        inside(args.inside)
        return
    if os.geteuid() != 0 or not sys.platform.startswith("linux"):
        raise RuntimeError("The chroot launcher requires Linux root; never run evaluator directly")
    if not args.output or args.output.exists() or args.timeout <= 0:
        raise ValueError("A fresh --output and a positive --timeout are required")
    # Keep this descriptor alive until process exit; close_fds prevents its
    # inheritance by sandbox children. Concurrent same-UID runs fail closed.
    launch_lock = os.open(SANDBOX / "launch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root, manifest_hash = verify_runtime(SANDBOX)
    manifest = json.loads((SANDBOX / "manifest.json").read_text())
    identity = manifest["identity"]
    uid, gid = verify_identity(identity)
    if not args.self_test:
        safety = json.loads((SANDBOX / "safety.json").read_text())
        if safety.get("manifest_sha256") != manifest_hash or safety.get("status") != "PASS":
            raise RuntimeError("A passing safety self-test for this runtime is required")
        if not args.canonical_all:
            canonical_validation = json.loads((SANDBOX / "canonical_validation.json").read_text())
            if canonical_validation.get("manifest_sha256") != manifest_hash or canonical_validation.get("status") != "PASS" or canonical_validation.get("passed_tasks") != 164:
                raise RuntimeError("All 164 canonical solutions must pass under this exact sandbox/scorer before model scoring")
        if not args.canonical_all and (not args.samples or not args.samples.is_file()):
            raise ValueError("--samples is required for evaluation")
        rows = ([{"task_id": "canonical", "solution": "canonical"}] if args.canonical_all else
                [json.loads(line) for line in args.samples.read_text().splitlines() if line])
        if not rows or any(not isinstance(r.get("solution"), str) or not isinstance(r.get("task_id"), str) for r in rows):
            raise ValueError("Expected nonempty JSONL task_id/solution rows")
        keys = [(r["task_id"], r.get("sample_id", 0)) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate task/sample key")
    run_id = uuid.uuid4().hex
    working = root / "work" / run_id
    working.mkdir(mode=0o755)
    output_dir = working / "out"
    output_dir.mkdir(mode=0o700)
    os.chown(output_dir, uid, gid)
    config = {"self_test": args.self_test, "canonical_all": args.canonical_all, "evaluator_patch": manifest["evaluator_patch"],
              "uid": uid, "gid": gid, "host_canary": str(SANDBOX / "host_canary"),
              "samples": f"/work/{run_id}/samples.jsonl", "result": f"/work/{run_id}/out/result.json"}
    if args.samples:
        shutil.copyfile(args.samples, working / "samples.jsonl")
        (working / "samples.jsonl").chmod(0o444)
    (working / "config.json").write_text(json.dumps(config))
    (working / "config.json").chmod(0o444)
    working.chmod(0o555)
    environment = {"PATH": "/usr/bin", "HOME": "/home/sandbox", "TMPDIR": "/tmp",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "HUMANEVAL_OVERRIDE_PATH": "/data/HumanEvalPlus-v0.1.10.jsonl"}
    started = time.monotonic()
    report = {"status": "FAIL", "manifest_sha256": manifest_hash, "run_id": run_id,
              "samples_sha256": sha(args.samples) if args.samples else None,
              "dataset_sha256": manifest["dataset_sha256"],
              "scorer_source_sha256": sha(Path(__file__)),
              "evaluation_complete": False,
              "identity": identity,
              "limitations": ["Linux chroot/seccomp shares the host kernel; not a VM", "Dedicated UID has no account; process conflicts checked before launch", "Private find_zero bookkeeping repair; upstream task32 scores also retained"]}
    logs = SANDBOX / "logs"
    logs.mkdir(exist_ok=True)
    process = None
    stdout_path, stderr_path = logs / (run_id + ".stdout"), logs / (run_id + ".stderr")
    try:
        # Regular output files are bounded by the child's RLIMIT_FSIZE. Pipes
        # with communicate() would allow unbounded hostile output in host RAM.
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "/runner.py", "--inside", f"/work/{run_id}/config.json"],
                                       env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       close_fds=True, start_new_session=True, preexec_fn=lambda: child_setup(str(root), args.timeout, uid, gid))
            process.wait(timeout=args.timeout)
        report["exit_code"] = process.returncode
        if (output_dir / "result.json").exists():
            child_report = json.loads((output_dir / "result.json").read_text())
            report.update(child_report)
        if process.returncode != 0:
            report.update(status="FAIL", evaluation_complete=False, error="Sandbox coordinator exited unsuccessfully; any rows are partial")
        if not args.self_test and not args.canonical_all:
            report["canonical_validation"] = canonical_validation
        if (SANDBOX / "host_canary.write").exists():
            report.update(status="FAIL", error="Host write canary unexpectedly created")
    except subprocess.TimeoutExpired:
        report["error"] = "Outer wall-time limit exceeded"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for key, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 12000))
                    report[key] = stream.read(12000).decode(errors="replace")
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "PASS" and args.self_test:
        (SANDBOX / "safety.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "PASS" and args.canonical_all:
        (SANDBOX / "canonical_validation.json").write_text(json.dumps({
            "status": "PASS", "manifest_sha256": manifest_hash, "passed_tasks": 164,
            "expected_rows": 164, "completed_rows": 164, "all_base_plus_passed": True,
            "evidence_sha256": sha(args.output), "result_sha256": sha(args.output), "result_path": str(args.output.resolve()),
            "dataset_sha256": manifest["dataset_sha256"], "scorer_source_sha256": sha(Path(__file__)),
            "evaluator_patch": manifest["evaluator_patch"]}, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
