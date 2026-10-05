"""Checkpoint identity and paper-protocol guards for scale expansion."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import validate_model_identity
from launch_ouro_variant_suite import validate_comparison


@pytest.mark.parametrize("name", ["ouro_1_4b_mc", "ouro_2_6b_mc"])
def test_identity_rejects_wrong_scale_or_revision(name):
    config = json.loads((ROOT / "configs" / (name + ".json")).read_text())
    record = {"repo_id": config["model"], "revision": config["revision"]}
    validate_model_identity(config, record)
    for key in record:
        with pytest.raises(ValueError):
            validate_model_identity(config, {**record, key: "wrong"})


def evidence(smoke=False):
    return {"task": "sciq", "is_full_split": not smoke, "n_documents": 2 if smoke else 1000,
            "full_split_count_verified": not smoke, "validation": {"paired": True},
            "runs": {mode: {"guidance": {"mode": mode, "early_loop": 1, "omega": .5, "omega_cap": 1.0}}
                     for mode in ("baseline", "fixed", "adaptive")}}


def test_full_queue_gate_rejects_partial_or_unpaired_data():
    result = evidence()
    validate_comparison(result, "sciq", False)
    for key in ("full_split_count_verified", "is_full_split"):
        with pytest.raises(ValueError):
            validate_comparison({**result, key: False}, "sciq", False)
    bad = copy.deepcopy(result)
    bad["validation"]["paired"] = False
    with pytest.raises(ValueError):
        validate_comparison(bad, "sciq", False)


def test_smoke_rejects_wrong_count_or_tuned_parameters():
    result = evidence(True)
    validate_comparison(result, "sciq", True)
    with pytest.raises(ValueError):
        validate_comparison({**result, "n_documents": 1}, "sciq", True)
    result["runs"]["adaptive"]["guidance"]["omega_cap"] = .7
    with pytest.raises(ValueError):
        validate_comparison(result, "sciq", True)
