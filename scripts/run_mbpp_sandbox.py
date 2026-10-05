"""Run official MBPPPlus EvalPlus workers in a verified private Linux chroot."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import grp
import hashlib
import json
import os
import pwd
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import types
import uuid


PROJECT = Path(__file__).resolve().parent.parent
SANDBOX = PROJECT / ".sandbox/mbpp"
DATA_SHA256 = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
EXPECTED_TASKS = 378
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
SAFETY_CHECKS = {"uid_nonroot", "groups_empty", "capabilities_empty", "no_new_privs",
                 "host_read_blocked", "host_write_blocked", "network_blocked", "fork_blocked",
                 "exec_blocked", "signal_blocked", "chroot_blocked"}


def sandbox_location(value):
    """Allow fresh MBPP runtime versions, never HumanEval's root or symlinks."""
    path = Path(value).absolute()
    if path != path.resolve() or path.name not in {"mbpp", "mbpp-v2"} or path.parent.name != ".sandbox":
        raise ValueError("MBPP sandbox must be a direct, non-symlink .sandbox/mbpp or .sandbox/mbpp-v2 directory")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_task_ids(path):
    if sha(path) != DATA_SHA256:
        raise ValueError("MBPPPlus artifact differs from the pinned SHA256")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    ids = [row["task_id"] for row in rows]
    if len(ids) != EXPECTED_TASKS or len(set(ids)) != EXPECTED_TASKS:
        raise ValueError("Expected all 378 unique MBPPPlus tasks")
    if any(not x.startswith("Mbpp/") or not x[5:].isdigit() for x in ids):
        raise ValueError("Unexpected MBPPPlus task ID")
    return sorted(ids, key=lambda x: int(x.split("/")[1]))


def validate_samples(rows, task_ids):
    """Parse only: model source is never imported or executed on the host."""
    if not rows:
        raise ValueError("Empty model samples")
    seen = set()
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in task_ids:
            raise ValueError("Unknown task ID in model samples")
        if task_id in seen:
            raise ValueError("Duplicate task ID; this protocol permits one greedy sample per task")
        if not isinstance(row.get("solution"), str):
            raise ValueError("Expected a string solution")
        if type(row.get("sample_id", 0)) is not int or row.get("sample_id", 0) != 0:
            raise ValueError("Expected greedy sample_id 0")
        seen.add(task_id)
    return {"samples": len(rows), "expected_full_tasks": EXPECTED_TASKS,
            "is_full_task_set": seen == set(task_ids)}


def suite_record(status, details, tests, pass_status="pass"):
    details = [bool(value) for value in details]
    return {"status": status,
            "passed": status == pass_status and len(details) == tests and all(details),
            "details": details, "tests": tests}


def read_child_report(path):
    # A sandbox-created absolute symlink must never be followed by the host.
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise RuntimeError("Sandbox result must be a direct regular file")
    return json.loads(path.read_text())


def verify_safety_record(record):
    safety = record.get("safety", {})
    checks = safety.get("checks", {})
    if (record.get("exit_code") != 0 or safety.get("passed") is not True
            or set(checks) != SAFETY_CHECKS or any(value is not True for value in checks.values())):
        raise RuntimeError("All 11 isolation checks and successful coordinator exit are required")


def verify_safety_gate(record, manifest_hash, evaluator_info):
    verify_safety_record(record)
    required = {"status": "PASS", "manifest_sha256": manifest_hash, "evaluation_complete": True,
                "dataset_sha256": DATA_SHA256, "scorer_source_sha256": sha(Path(__file__)),
                "evaluator_patch": evaluator_info}
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError("A passing safety self-test for this exact runtime is required")
    benign = record.get("benign_fixture", {})
    if benign != suite_record("pass", [True, True], 2):
        raise RuntimeError("Official benign fixture has not passed completely")


def verify_canonical_gate(gate, manifest_hash, task_ids, expected_counts, evaluator_info):
    required = {"status": "PASS", "manifest_sha256": manifest_hash,
                "passed_tasks": EXPECTED_TASKS, "expected_rows": EXPECTED_TASKS,
                "completed_rows": EXPECTED_TASKS, "all_base_plus_passed": True,
                "dataset_sha256": DATA_SHA256, "scorer_source_sha256": sha(Path(__file__)),
                "task_ids": task_ids}
    if any(gate.get(key) != value for key, value in required.items()):
        raise RuntimeError("All 378 canonical solutions must pass under this exact sandbox/scorer before model scoring")
    if gate.get("evaluator_patch") != evaluator_info or evaluator_info.get("applied") is not False:
        raise RuntimeError("Canonical gate does not identify an unmodified evaluator")
    path = Path(gate.get("result_path", ""))
    if (not path.is_file() or sha(path) != gate.get("evidence_sha256")
            or gate.get("result_sha256") != gate.get("evidence_sha256")):
        raise RuntimeError("Canonical evidence is missing or its SHA256 changed")
    result = json.loads(path.read_text())
    verify_safety_record(result)
    if result.get("evaluator_patch") != evaluator_info:
        raise RuntimeError("Canonical evidence identifies a different evaluator")
    for key in ("status", "manifest_sha256", "dataset_sha256", "scorer_source_sha256",
                "expected_rows", "completed_rows"):
        if result.get(key) != required[key]:
            raise RuntimeError(f"Canonical evidence disagrees with its gate: {key}")
    rows = result.get("rows", [])
    if (result.get("evaluation_complete") is not True or len(rows) != EXPECTED_TASKS
            or sorted(row.get("task_id", "") for row in rows) != sorted(task_ids)):
        raise RuntimeError("Canonical evidence does not cover the complete unique task set")
    for row in rows:
        if row.get("plus_passed") is not True:
            raise RuntimeError("Canonical base+extended failure in evidence")
        for suite in ("base", "plus"):
            record = row.get(suite, {})
            details = record.get("details")
            tests = record.get("tests")
            if (record.get("passed") is not True or record.get("status") != "pass"
                    or not isinstance(details, list) or type(tests) is not int
                    or len(details) != tests
                    or tests != expected_counts[row["task_id"]][suite]
                    or any(value is not True for value in details)):
                raise RuntimeError("Canonical test details are incomplete or failed")


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
    if Path(__file__).resolve() != Path("/runner.py"):
        raise RuntimeError("The private --inside entry point must only run after chroot at /runner.py")
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
    # official MBPP loader and data utilities themselves are unchanged.
    data_package = types.ModuleType("evalplus.data")
    data_package.__path__ = ["/opt/site/evalplus/data"]
    sys.modules["evalplus.data"] = data_package
    from evalplus.data.mbpp import get_mbpp_plus
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    problems = get_mbpp_plus()
    expected_task_ids = config["task_ids"]
    if set(problems) != set(expected_task_ids) or len(problems) != EXPECTED_TASKS:
        raise RuntimeError("Official MBPP loader differs from the complete pinned task set")

    def groundtruth(problem, suite):
        return trusted_exec(problem["prompt"] + problem["canonical_solution"],
                            problem[suite + "_input"], problem["entry_point"], record_time=True,
                            output_not_none=problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS)
    original = evaluator.unsafe_execute

    def guarded_execute(*args, **kwargs):
        install_seccomp(worker=True)
        return original(*args, **kwargs)

    evaluator.unsafe_execute = guarded_execute
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
              "scorer": "EvalPlus 0.3.1 unmodified MBPP evaluator and official deserialized inputs",
              "evaluator_patch": config["evaluator_patch"],
              "numpy": numpy.__version__, "psutil": psutil.__version__, "rows": [],
              "evaluation_complete": False}
    if config["self_test"]:
        status, details = evaluator.untrusted_check(
            "mbpp", "def add(a, b):\n    return a + b\n", [[1, 2], [-3, 4]],
            "add", [3, 1], 0, [0.001, 0.001], fast_check=False)
        result["benign_fixture"] = suite_record(status, details, 2, evaluator.PASS)
        if not result["benign_fixture"]["passed"]:
            raise RuntimeError(f"Official benign EvalPlus fixture failed: {result['benign_fixture']}")
        result["canonical_fixtures"] = []
        # Representative set-equality and output-not-None oracle cases.
        for task_id in ("Mbpp/2", "Mbpp/737"):
            problem = problems[task_id]
            code = problem["prompt"] + problem["canonical_solution"]
            row = {"task_id": task_id, "entry_point": problem["entry_point"]}
            for suite in ("base", "plus"):
                expected, durations = groundtruth(problem, suite)
                status, details = evaluator.untrusted_check("mbpp", code, problem[suite + "_input"],
                    problem["entry_point"], expected, problem["atol"], durations, fast_check=False)
                row[suite] = suite_record(status, details, len(problem[suite + "_input"]), evaluator.PASS)
                if not row[suite]["passed"]:
                    raise RuntimeError(f"Canonical fixture failed: {row}")
            result["canonical_fixtures"].append(row)
    else:
        samples = ([{"task_id": task_id, "solution": problem["prompt"] + problem["canonical_solution"]}
                    for task_id, problem in problems.items()] if config["canonical_all"] else
                   [json.loads(line) for line in Path(config["samples"]).read_text().splitlines() if line])
        result["expected_rows"] = len(samples)
        result["sample_validation"] = validate_samples(samples, expected_task_ids)
        expected = {}
        for sample in samples:
            problem = problems[sample["task_id"]]
            task_id = sample["task_id"]
            if task_id not in expected:
                expected[task_id] = {suite: groundtruth(problem, suite) for suite in ("base", "plus")}
            row = {"task_id": task_id, "sample_id": sample.get("sample_id", 0)}
            for suite in ("base", "plus"):
                answers, durations = expected[task_id][suite]
                status, details = evaluator.untrusted_check(
                    "mbpp", sample["solution"], problem[suite + "_input"], problem["entry_point"],
                    answers, problem["atol"], durations, fast_check=False)
                row[suite] = suite_record(status, details, len(problem[suite + "_input"]), evaluator.PASS)
            row["plus_passed"] = row["base"]["passed"] and row["plus"]["passed"]
            result["rows"].append(row)
            Path(config["result"]).write_text(json.dumps(result) + "\n")
    result["status"] = "PASS"
    result["completed_rows"] = len(result["rows"])
    result["evaluation_complete"] = True
    if config["canonical_all"] and (len(result["rows"]) != 378 or not all(row["plus_passed"] for row in result["rows"])):
        result["status"] = "FAIL"
        result["error"] = "Canonical full-suite self-check did not pass every task"
    Path(config["result"]).write_text(json.dumps(result, indent=2) + "\n")


def verify_runtime(base):
    manifest_path = base / "manifest.json"
    if (base / "PREPARATION_FAILED").exists():
        raise RuntimeError("Sandbox preparation failed")
    manifest = json.loads(manifest_path.read_text())
    root = base / "rootfs"
    required_files = {"runner.py", "usr/bin/python3", "data/MbppPlus-v0.2.0.jsonl",
                      "opt/site/evalplus/eval/__init__.py", "opt/site/evalplus/eval/_special_oracle.py",
                      "opt/site/evalplus/data/mbpp.py", "opt/site/evalplus/gen/util/__init__.py"}
    if not required_files <= set(manifest["immutable_sha256"]):
        raise RuntimeError("Sandbox runtime manifest omits required immutable files")
    if manifest.get("evaluator_patch", {}).get("applied") is not False:
        raise RuntimeError("MBPP requires the unmodified official evaluator")
    evaluator_hash = manifest["immutable_sha256"]["opt/site/evalplus/eval/__init__.py"]
    if any(manifest["evaluator_patch"].get(key) != evaluator_hash for key in ("upstream_sha256", "runtime_sha256")):
        raise RuntimeError("MBPP evaluator provenance disagrees with its immutable source")
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--canonical-all", action="store_true", help="Evaluate all 378 pinned canonical solutions as a scorer self-check")
    mode.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--sandbox", type=Path, default=SANDBOX, help="Independent .sandbox/mbpp or .sandbox/mbpp-v2 runtime; usable from a frozen release")
    args = parser.parse_args()
    if args.inside:
        inside(args.inside)
        return
    sandbox = sandbox_location(args.sandbox)
    if os.geteuid() != 0 or not sys.platform.startswith("linux"):
        raise RuntimeError("The chroot launcher requires Linux root; never run evaluator directly")
    if not args.output or args.output.exists() or args.timeout <= 0:
        raise ValueError("A fresh --output and a positive --timeout are required")
    # Keep this descriptor alive until process exit; close_fds prevents its
    # inheritance by sandbox children. Concurrent same-UID runs fail closed.
    launch_lock = os.open(sandbox / "launch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root, manifest_hash = verify_runtime(sandbox)
    manifest = json.loads((sandbox / "manifest.json").read_text())
    if manifest.get("dataset_sha256") != DATA_SHA256 or manifest.get("expected_tasks") != EXPECTED_TASKS:
        raise RuntimeError("Sandbox is not the pinned complete MBPPPlus runtime")
    task_ids = pinned_task_ids(root / "data/MbppPlus-v0.2.0.jsonl")
    expected_counts = {row["task_id"]: {suite: len(row[suite + "_input"]) for suite in ("base", "plus")}
                       for row in (json.loads(line) for line in (root / "data/MbppPlus-v0.2.0.jsonl").read_text().splitlines())}
    identity = manifest["identity"]
    uid, gid = verify_identity(identity)
    samples_bytes = None
    if not args.self_test:
        safety = json.loads((sandbox / "safety.json").read_text())
        verify_safety_gate(safety, manifest_hash, manifest["evaluator_patch"])
        if not args.canonical_all:
            canonical_validation = json.loads((sandbox / "canonical_validation.json").read_text())
            verify_canonical_gate(canonical_validation, manifest_hash, task_ids, expected_counts, manifest["evaluator_patch"])
        if not args.canonical_all and (not args.samples or not args.samples.is_file()):
            raise ValueError("--samples is required for evaluation")
        samples_bytes = args.samples.read_bytes() if args.samples else None
        rows = ([{"task_id": "canonical", "solution": "canonical"}] if args.canonical_all else
                [json.loads(line) for line in samples_bytes.splitlines() if line])
        if not rows or any(not isinstance(r.get("solution"), str) or not isinstance(r.get("task_id"), str) for r in rows):
            raise ValueError("Expected nonempty JSONL task_id/solution rows")
        if not args.canonical_all:
            validate_samples(rows, task_ids)
    run_id = uuid.uuid4().hex
    working = root / "work" / run_id
    working.mkdir(mode=0o755)
    output_dir = working / "out"
    output_dir.mkdir(mode=0o700)
    os.chown(output_dir, uid, gid)
    config = {"task_ids": task_ids, "self_test": args.self_test, "canonical_all": args.canonical_all, "evaluator_patch": manifest["evaluator_patch"],
              "uid": uid, "gid": gid, "host_canary": str(sandbox / "host_canary"),
              "samples": f"/work/{run_id}/samples.jsonl", "result": f"/work/{run_id}/out/result.json"}
    if samples_bytes is not None:
        (working / "samples.jsonl").write_bytes(samples_bytes)
        (working / "samples.jsonl").chmod(0o444)
    (working / "config.json").write_text(json.dumps(config))
    (working / "config.json").chmod(0o444)
    working.chmod(0o555)
    environment = {"PATH": "/usr/bin", "HOME": "/home/sandbox", "TMPDIR": "/tmp",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "MBPP_OVERRIDE_PATH": "/data/MbppPlus-v0.2.0.jsonl"}
    started = time.monotonic()
    report = {"status": "FAIL", "manifest_sha256": manifest_hash, "run_id": run_id,
              "sandbox": str(sandbox), "progress_path": str(output_dir / "result.json"),
              "samples_sha256": hashlib.sha256(samples_bytes).hexdigest() if samples_bytes is not None else None,
              "dataset_sha256": manifest["dataset_sha256"],
              "scorer_source_sha256": sha(Path(__file__)),
              "evaluation_complete": False,
              "identity": identity,
              "limitations": ["Linux chroot/seccomp shares the host kernel; not a VM", "Dedicated UID has no account; process conflicts checked before launch", "Official MBPP special oracles retained; no evaluator patches"]}
    logs = sandbox / "logs"
    logs.mkdir(exist_ok=True)
    process = None
    stdout_path, stderr_path = logs / (run_id + ".stdout"), logs / (run_id + ".stderr")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Sandbox launcher received signal {signum}; terminate its own process group")
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        # Regular output files are bounded by the child's RLIMIT_FSIZE. Pipes
        # with communicate() would allow unbounded hostile output in host RAM.
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "/runner.py", "--inside", f"/work/{run_id}/config.json"],
                                       env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       close_fds=True, start_new_session=True, preexec_fn=lambda: child_setup(str(root), args.timeout, uid, gid))
            report["coordinator_pid"] = process.pid
            print(json.dumps({"status": "RUNNING", "run_id": run_id, "coordinator_pid": process.pid,
                              "progress_path": report["progress_path"]}), flush=True)
            process.wait(timeout=args.timeout)
        report["exit_code"] = process.returncode
        if (output_dir / "result.json").exists():
            child_report = read_child_report(output_dir / "result.json")
            report.update(child_report)
        if process.returncode != 0:
            report.update(status="FAIL", evaluation_complete=False, error="Sandbox coordinator exited unsuccessfully; any rows are partial")
        if not args.self_test and not args.canonical_all:
            report["canonical_validation"] = canonical_validation
        if (sandbox / "host_canary.write").exists():
            report.update(status="FAIL", error="Host write canary unexpectedly created")
    except subprocess.TimeoutExpired:
        report["error"] = "Outer wall-time limit exceeded"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
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
        signal.signal(signal.SIGTERM, previous_sigterm)
    if report["status"] == "PASS" and args.self_test:
        (sandbox / "safety.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "PASS" and args.canonical_all:
        (sandbox / "canonical_validation.json").write_text(json.dumps({
            "status": "PASS", "manifest_sha256": manifest_hash, "passed_tasks": 378,
            "expected_rows": 378, "completed_rows": 378, "all_base_plus_passed": True,
            "evidence_sha256": sha(args.output), "result_sha256": sha(args.output), "result_path": str(args.output.resolve()),
            "dataset_sha256": manifest["dataset_sha256"], "scorer_source_sha256": sha(Path(__file__)),
            "evaluator_patch": manifest["evaluator_patch"], "task_ids": task_ids}, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
