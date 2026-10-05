"""Prepare an offline, private chroot for official MBPPPlus scoring; no execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import grp
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pwd
from pathlib import Path
import re
import shutil
import subprocess


DATA_SHA256 = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
PROJECT = Path(__file__).resolve().parents[1]
SANDBOX = PROJECT / ".sandbox/mbpp"


def sandbox_location(value):
    """Allow fresh MBPP runtime versions, never HumanEval's root or symlinks."""
    path = Path(value).absolute()
    if path != path.resolve() or path.name not in {"mbpp", "mbpp-v2"} or path.parent.name != ".sandbox":
        raise ValueError("MBPP sandbox must be a direct, non-symlink .sandbox/mbpp or .sandbox/mbpp-v2 directory")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_identity(sandbox_parent=None):
    used = set()
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in path.read_text().splitlines():
                if line.startswith(("Uid:", "Gid:")):
                    used.update(int(value) for value in line.split()[1:])
        except (FileNotFoundError, ProcessLookupError):
            continue
    # Reserve other project sandboxes' identities even while they are idle.
    for manifest in (sandbox_parent or PROJECT / ".sandbox").glob("*/manifest.json"):
        record = json.loads(manifest.read_text())
        identity = record.get("identity", {})
        used.update(identity[key] for key in ("uid", "gid") if key in identity)
    used.update(entry.pw_uid for entry in pwd.getpwall())
    used.update(entry.gr_gid for entry in grp.getgrall())
    for candidate in range(60000, 65000):
        if candidate not in used:
            return {"uid": candidate, "gid": candidate,
                    "selected_at": datetime.now(timezone.utc).isoformat(),
                    "selection": "no host account/group/process or other project sandbox identity; rechecked at launch"}
    raise RuntimeError("No unused dedicated sandbox UID/GID available")


def copy_file(source, root, destination=None):
    source = Path(source)
    destination = root / (destination or str(source).lstrip("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.resolve(), destination)
    return destination


def libraries(binary):
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, timeout=20)
    if result.returncode and "not a dynamic executable" not in result.stderr + result.stdout:
        raise RuntimeError(f"ldd failed for {binary}: {result.stderr}")
    if "not found" in result.stdout:
        raise RuntimeError(f"Missing library for {binary}: {result.stdout}")
    return [Path(value) for value in re.findall(r"(?:=>\s+)?(/[^\s()]+)", result.stdout)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sandbox", type=Path, default=SANDBOX, help="Independent .sandbox/mbpp or .sandbox/mbpp-v2 runtime; usable from a frozen release")
    args = parser.parse_args()
    sandbox = sandbox_location(args.sandbox)
    if os.geteuid() != 0:
        raise RuntimeError("Preparation needs root ownership for a non-writable runtime")
    if sandbox.exists():
        raise FileExistsError(f"Refusing to replace existing sandbox {sandbox}")
    if sha(args.dataset) != DATA_SHA256:
        raise ValueError("MbppPlus dataset hash is not the audited v0.2.0 artifact")
    if importlib.metadata.version("evalplus") != "0.3.1":
        raise ValueError("This sandbox is audited for EvalPlus 0.3.1 only")
    root = sandbox / "rootfs"
    root.mkdir(parents=True)
    try:
        identity = select_identity(sandbox.parent)
        copy_file("/usr/bin/python3", root, "usr/bin/python3")
        ignored = shutil.ignore_patterns("__pycache__", "test", "tests", "site-packages", "dist-packages", "ensurepip", "idlelib", "tkinter")
        shutil.copytree("/usr/lib/python3.10", root / "usr/lib/python3.10", ignore=ignored)
        versions = {}
        for name in ("numpy", "psutil", "evalplus", "appdirs", "tempdir", "wget"):
            origin = Path(importlib.util.find_spec(name).origin)
            package = origin.parent
            if origin.name == "__init__.py":
                shutil.copytree(package, root / "opt/site" / name, ignore=ignored)
            else:
                copy_file(origin, root, "opt/site/" + origin.name)
            companion = package.with_name(name + ".libs")
            if companion.is_dir():
                shutil.copytree(companion, root / "opt/site" / companion.name)
            versions[name] = importlib.metadata.version(name)
        evaluator_path = root / "opt/site/evalplus/eval/__init__.py"
        evaluator_patch = {"applied": False, "reason": "MBPP uses unmodified EvalPlus 0.3.1",
                           "upstream_sha256": sha(evaluator_path), "runtime_sha256": sha(evaluator_path)}
        # Resolve every extension dependency from its actual source location;
        # NumPy's wheel-local libraries retain their original relative layout.
        queue = [Path("/usr/bin/python3")]
        queue += list(Path("/usr/lib/python3.10/lib-dynload").glob("*.so"))
        for name in ("numpy", "psutil"):
            package = Path(importlib.util.find_spec(name).origin).parent
            queue += list(package.rglob("*.so"))
            queue += list(package.with_name(name + ".libs").glob("*.so*"))
        queue.append(Path("/usr/lib/x86_64-linux-gnu/libseccomp.so.2"))
        visited = set()
        while queue:
            binary = queue.pop()
            if str(binary) in visited:
                continue
            visited.add(str(binary))
            for library in libraries(binary):
                if not str(library).startswith(("/lib/", "/lib64/", "/usr/lib/")):
                    # Wheel-local dependencies already copied to /opt/site.
                    if ".libs/" not in str(library):
                        raise ValueError(f"Unexpected library outside system/wheel directories: {library}")
                    continue
                copy_file(library, root)
                queue.append(library)
        copy_file("/usr/lib/x86_64-linux-gnu/libseccomp.so.2", root)
        copy_file(args.dataset, root, "data/MbppPlus-v0.2.0.jsonl")
        copy_file(PROJECT / "scripts/run_mbpp_sandbox.py", root, "runner.py")
        # psutil.virtual_memory reads this synthetic, non-mounted file. No
        # real procfs/process metadata or host device tree is exposed.
        (root / "proc").mkdir()
        (root / "proc/meminfo").write_text("MemTotal: 4194304 kB\nMemFree: 4194304 kB\nMemAvailable: 4194304 kB\nBuffers: 0 kB\nCached: 0 kB\nSReclaimable: 0 kB\nShmem: 0 kB\nActive: 0 kB\nInactive: 0 kB\nSlab: 0 kB\n")
        for directory in ("tmp", "work", "dev/shm", "home/sandbox"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Unexpected sandbox symlink {path}")
            os.chown(path, 0, 0)
            executable = path == root / "usr/bin/python3" or path.name.startswith("ld-linux")
            path.chmod(0o555 if path.is_dir() or executable else 0o444)
        root.chmod(0o555)
        for directory in ("tmp", "dev/shm", "home/sandbox"):
            path = root / directory
            os.chown(path, identity["uid"], identity["gid"])
            path.chmod(0o700)
        # The canary is outside rootfs, but still inside the authorized new
        # sandbox directory. It contains no user data or credential.
        (sandbox / "host_canary").write_text("loopcd-sandbox-host-boundary-test\n")
        immutable = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
        manifest = {"status": "PREPARED_NOT_VALIDATED", "rootfs": str(root),
                    "dataset_sha256": DATA_SHA256, "dataset": "MbppPlus-v0.2.0", "expected_tasks": 378,
                    "packages": versions, "identity": identity,
                    "prepare_source_sha256": sha(Path(__file__)),
                    "scorer_source_sha256": sha(PROJECT / "scripts/run_mbpp_sandbox.py"),
                    "evaluator_patch": evaluator_patch,
                    "immutable_sha256": immutable,
                    "isolation": "private chroot; no mounts; UID drop; two-stage seccomp; resource limits"}
        (sandbox / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"status": manifest["status"], "sandbox": str(sandbox), "files": len(immutable)}))
    except BaseException:
        (sandbox / "PREPARATION_FAILED").write_text("Inspect partial private sandbox; never execute it.\n")
        raise


if __name__ == "__main__":
    main()
