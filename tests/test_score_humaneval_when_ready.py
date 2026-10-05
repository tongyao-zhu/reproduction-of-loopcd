import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("score_ready", Path(__file__).resolve().parents[1] / "scripts/score_humaneval_when_ready.py")
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


def write_run(root, arm, status="completed", n=164, full=True, duplicate=False):
    directory = root / arm
    directory.mkdir(parents=True)
    ids = [f"HumanEval/{i}" for i in range(n)]
    manifest = {"status": status, "is_full_split": full, "completed_samples": n,
                "expected_samples": n, "config": {"task_ids": ids, "limit": None if full else n}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    observed = ids[:-1] + [ids[0]] if duplicate else ids
    (directory / "samples.jsonl").write_text("".join(json.dumps({"task_id": task}) + "\n" for task in observed))


def test_missing_manifests_wait(tmp_path):
    ready, snapshot, pins = coordinator.generation_snapshot(tmp_path)
    assert not ready and not pins
    assert all(row["status"] == "manifest_missing" for row in snapshot.values())


def test_failed_generation_stops_even_if_other_manifests_missing(tmp_path):
    write_run(tmp_path, "baseline32", status="failed")
    with pytest.raises(coordinator.GenerationFailed):
        coordinator.generation_snapshot(tmp_path)


def test_all_four_completed_and_unique_required(tmp_path):
    for arm in coordinator.ARMS:
        write_run(tmp_path, arm, duplicate=arm == "hidden16")
    with pytest.raises(ValueError, match="duplicated"):
        coordinator.generation_snapshot(tmp_path)


def test_debug_requires_explicit_opt_in(tmp_path):
    for arm in coordinator.ARMS:
        write_run(tmp_path, arm, n=2, full=False)
    with pytest.raises(ValueError, match="allow-subset"):
        coordinator.generation_snapshot(tmp_path)
    ready, _, pins = coordinator.generation_snapshot(tmp_path, allow_subset=True)
    assert ready and len(pins) == 8


def test_complete_split_pins_inputs_and_detects_later_mutation(tmp_path):
    for arm in coordinator.ARMS:
        write_run(tmp_path, arm)
    ready, _, pins = coordinator.generation_snapshot(tmp_path)
    assert ready
    coordinator.verify_hashes(pins)
    (tmp_path / "hidden32/samples.jsonl").write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        coordinator.verify_hashes(pins)
