"""CPU checks of fail-closed sandbox policy and pinned input validation."""
import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_untrusted_worker_policy_denies_process_and_network_escape():
    runner = module("run_eval_sandbox")
    common = set(runner.COMMON_DENIED)
    strict = common | set(runner.WORKER_DENIED)
    assert {"socket", "connect", "execve", "execveat", "chroot", "ptrace", "unshare", "setns", "process_vm_readv", "io_uring_setup"} <= common
    assert {"fork", "clone", "clone3", "kill", "pidfd_send_signal", "setsid", "setpgid"} <= strict


def test_partial_preparation_is_never_executable(tmp_path):
    runner = module("run_eval_sandbox")
    (tmp_path / "PREPARATION_FAILED").write_text("failed")
    with pytest.raises(RuntimeError, match="preparation failed"):
        runner.verify_runtime(tmp_path)


def test_runtime_manifest_detects_tampering(tmp_path):
    runner = module("run_eval_sandbox")
    root = tmp_path / "rootfs"
    root.mkdir()
    target = root / "runtime.py"
    target.write_text("original")
    manifest = {"immutable_sha256": {"runtime.py": runner.sha(target)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    target.write_text("changed")
    with pytest.raises(RuntimeError, match="Sandbox runtime"):
        runner.verify_runtime(tmp_path)


def test_dataset_pin_is_a_sha256_digest():
    prepare = module("prepare_eval_sandbox")
    assert len(prepare.DATA_SHA256) == 64
    assert int(prepare.DATA_SHA256, 16) > 0


def test_find_zero_patch_preserves_predicate_and_records_exact_delta(tmp_path):
    prepare = module("prepare_eval_sandbox")
    target = tmp_path / "opt/site/evalplus/eval/__init__.py"
    target.parent.mkdir(parents=True)
    original = '                            assert abs(_poly(*inp, out)) <= atol\n                            continue\n'
    target.write_text(original)
    record = prepare.patch_evalplus(tmp_path)
    assert (tmp_path / "audit/eval_init.original.py").read_text() == original
    assert 'assert abs(_poly(*inp, out)) <= atol' in target.read_text()
    assert target.read_text().replace('                            details[i] = True\n                            progress.value += 1\n', '') == original
    assert record["patched_sha256"] != record["upstream_sha256"]
    with pytest.raises(FileExistsError):
        prepare.patch_evalplus(tmp_path)


def test_find_zero_patch_refuses_unknown_source(tmp_path):
    prepare = module("prepare_eval_sandbox")
    target = tmp_path / "opt/site/evalplus/eval/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("unknown evaluator implementation")
    with pytest.raises(ValueError, match="Unrecognized"):
        prepare.patch_evalplus(tmp_path)
