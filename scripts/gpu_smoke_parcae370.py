#!/usr/bin/env python3
"""Real Parcae-370M GPU oracle at short and 2048-token full-sequence lengths."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_parcae370 import verify_prepared, stream_fingerprint
from launch_huginn_r16_suite import gpu_status

CASES = (("baseline", .5, 1.), ("fixed", 0., 1.), ("adaptive", .5, 0.),
         ("hidden", 0., 1.), ("fixed", .5, 1.), ("adaptive", .5, 1.),
         ("hidden", 1., 1.), ("hidden", .75, 1.))


def validate_report(report):
    """Fail closed on incomplete, CPU-only, or altered gate certificates."""
    if report.get("schema_version") != 1 or report.get("status") != "PASS":
        raise ValueError("Incomplete Parcae GPU gate")
    if report.get("device_type") != "cuda" or report.get("model_repo") != "SandyResearch/parcae-370m":
        raise ValueError("Expected actual Parcae-370M CUDA execution")
    if report.get("lengths") != [3, 2048] or report.get("depths") != [4, 8]:
        raise ValueError("Missing required context lengths/depths")
    if report.get("loading") != [{"keys": 117, "missing": [], "unexpected": [], "strict": True}]:
        raise ValueError("Strict real-checkpoint loading not certified")
    records = report.get("records", [])
    expected = [(length, depth, mode, omega, cap) for length in (3, 2048)
                for depth in (4, 8) for mode, omega, cap in CASES]
    observed = [(r.get("length"), r.get("depth"), r.get("mode"), r.get("omega"), r.get("cap")) for r in records]
    if observed != expected:
        raise ValueError("Missing/duplicate/reordered numerical case")
    for record in records:
        if not all(record.get(key) is True for key in ("finite", "oracle_passed", "rng_identical", "initial_state_identical", "core_states_identical", "restored")):
            raise ValueError("Numerical, trajectory, initialization or restoration failure")
        enabled = record["mode"] != "baseline" and (record["cap"] != 0 if record["mode"] == "adaptive" else record["omega"] != 0)
        readouts = 2 if enabled and record["mode"] in {"fixed", "adaptive"} else 1
        counts = {"initialization": 1, "prelude": 4, "core": 4 * record["depth"], "C": readouts,
                  "coda": 4 * readouts, "norm": readouts, "head": readouts}
        if record.get("calls") != counts:
            raise ValueError("Wrong actual native layer/readout counts")
        if record.get("cache_is_none") is not True or record.get("dtype") != "torch.float32":
            raise ValueError("Unsupported cache/precision")
        error = record.get("max_abs_error")
        if isinstance(error, bool) or not isinstance(error, (int, float)) or not 0 <= error < float("inf"):
            raise ValueError("Invalid oracle error")
        scaled_error = record.get("maximum_tolerance_ratio")
        if isinstance(scaled_error, bool) or not isinstance(scaled_error, (int, float)) or not 0 <= scaled_error <= 1:
            raise ValueError("Pointwise oracle tolerance exceeded or unrecorded")
        if not enabled or record["mode"] in {"fixed", "hidden"}:
            if error != 0:
                raise ValueError("Exact-identity/fixed/hidden oracle was not exact")
        if not isinstance(record.get("max_memory_allocated_bytes"), int) or record["max_memory_allocated_bytes"] <= 0:
            raise ValueError("Missing actual GPU memory observation")
    if report.get("checks") != {
        "model_prepared_hashes_before_after": True, "sources_before_after": True,
        "strict_loading": True, "actual_gpu": True, "native_rng_and_trajectories": True,
        "short_and_context_limit": True, "all_32_cases": True,
    }:
        raise ValueError("Missing final source/model checks")
    return {"cases": 32, "maximum_input_tokens": 2048, "depths": [4, 8]}


def numerical_cases(model, tokens):
    """Also callable by CPU tests; only main() may certify a real GPU gate."""
    import torch
    from loopcd_repro.parcae import ParcaeGuidance, ParcaeGuidanceConfig
    device = tokens.device
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def rng():
        return {"cpu": digest(torch.get_rng_state()),
                "device": digest(torch.cuda.get_rng_state(device)) if device.type == "cuda" else None}
    def seed():
        torch.manual_seed(42)
    if any(name in model.__dict__ for name in ("initialize_state", "core_block_forward")) or model.transformer.C._forward_pre_hooks:
        raise ValueError("Numerical gate requires an unmodified native instance")
    native_init = model.initialize_state
    native_core = model.core_block_forward
    records = []
    for depth in (4, 8):
        states = []
        initial = []
        def capture_core(*args, **kwargs):
            value = native_core(*args, **kwargs)
            states.append(value.detach().clone())
            return value
        def capture_init(*args, **kwargs):
            value = native_init(*args, **kwargs)
            initial.append(digest(value))
            return value
        seed()
        model.initialize_state, model.core_block_forward = capture_init, capture_core
        try:
            with torch.inference_mode():
                baseline = model.forward_for_generation(tokens, num_steps=depth, past_key_values=None)["logits"]
        finally:
            del model.initialize_state, model.core_block_forward
        native_rng = rng()
        assert len(states) == depth and len(initial) == 1
        state_hashes = [digest(value) for value in states]
        def native_override(raw):
            calls = []
            def replace(module, args):
                calls.append(1)
                return (raw,) + args[1:]
            handle = model.transformer.C.register_forward_pre_hook(replace)
            seed()
            try:
                with torch.inference_mode():
                    value = model.forward_for_generation(tokens, num_steps=depth, past_key_values=None)["logits"]
            finally:
                handle.remove()
            assert calls == [1] and rng() == native_rng
            return value
        early = native_override(states[0])
        for mode, omega, cap in CASES:
            config = ParcaeGuidanceConfig(mode, depth, 1, omega, cap)
            if not config.enabled:
                expected = baseline
            elif mode == "hidden":
                blended = (states[-1].float() + omega * (states[-1].float() - states[0].float())).to(states[-1].dtype)
                expected = native_override(blended)
            elif mode == "fixed":
                expected = baseline.float() + omega * (baseline.float() - early.float())
            else:
                probabilities = baseline.double().softmax(-1).topk(2, dim=-1).values
                strength = cap * (1 - (probabilities[..., :1] - probabilities[..., 1:2]))
                expected = baseline.double() + strength * (baseline.double() - early.double())
            calls = {"initialization": 0, "prelude": 0, "core": 0, "C": 0, "coda": 0, "norm": 0, "head": 0}
            observed_initial, observed_states = [], []
            def observe_init(*args, **kwargs):
                value = native_init(*args, **kwargs)
                calls["initialization"] += 1
                observed_initial.append(digest(value))
                return value
            def observe_core(*args, **kwargs):
                value = native_core(*args, **kwargs)
                observed_states.append(digest(value))
                return value
            def count(name):
                def hook(*args):
                    calls[name] += 1
                return hook
            handles = []
            for name, modules in (("prelude", model.transformer.prelude), ("core", model.transformer.core_block), ("coda", model.transformer.coda)):
                handles += [layer.register_forward_hook(count(name)) for layer in modules]
            for name, module in (("C", model.transformer.C), ("norm", model.transformer.ln_f), ("head", model.lm_head)):
                handles.append(module.register_forward_hook(count(name)))
            model.initialize_state, model.core_block_forward = observe_init, observe_core
            seed()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            adapter = ParcaeGuidance(model, config)
            try:
                with torch.inference_mode():
                    output = adapter(tokens)
            finally:
                for handle in handles:
                    handle.remove()
                del model.initialize_state, model.core_block_forward
            logits = output["logits"]
            exact = not config.enabled or mode in {"fixed", "hidden"}
            torch.testing.assert_close(logits.double(), expected.double(), atol=0 if exact else 2e-5, rtol=0 if exact else 2e-6)
            readouts = 2 if config.enabled and mode in {"fixed", "adaptive"} else 1
            assert calls == {"initialization": 1, "prelude": len(model.transformer.prelude),
                             "core": len(model.transformer.core_block) * depth, "C": readouts,
                             "coda": len(model.transformer.coda) * readouts, "norm": readouts, "head": readouts}
            assert observed_initial == initial and observed_states == state_hashes and rng() == native_rng
            assert torch.isfinite(logits).all() and logits.dtype == torch.float32 and output["past_key_values"] is None
            assert not model.transformer.C._forward_pre_hooks
            records.append({
                "length": tokens.shape[1], "depth": depth, "mode": mode, "omega": omega, "cap": cap,
                "max_abs_error": float((logits.double() - expected.double()).abs().max()),
                "maximum_tolerance_ratio": float(((logits.double() - expected.double()).abs() / (2e-5 + 2e-6 * expected.double().abs())).max()),
                "oracle_passed": True, "finite": True, "rng_identical": True, "initial_state_identical": True,
                "core_states_identical": True, "restored": True, "cache_is_none": True,
                "dtype": str(logits.dtype), "calls": calls, "native_rng": native_rng,
                "initial_state_sha256": initial[0], "logits_sha256": digest(logits),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            })
        del states, baseline, early, expected, output, logits
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical_gpu.isdigit():
        raise ValueError("Select exactly one physical GPU in CUDA_VISIBLE_DEVICES")
    device_state = gpu_status(physical_gpu)
    if not device_state["ready"]:
        raise RuntimeError(f"GPU is not idle: {device_state}")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "running", "started_unix": time.time(),
             "scope": "Actual checkpoint GPU formula/RNG/layer-count gate, including a repeated 2048-token fixture; no benchmark score or throughput claim.",
             "physical_gpu": physical_gpu, "initial_gpu_state": device_state, "lengths": [3, 2048], "depths": [4, 8]}
    def save():
        (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        verify_prepared(args.model)
        state["model_provenance"] = stream_fingerprint(args.model / "model_provenance.json")
        state["sources"] = {str(path.relative_to(ROOT)): stream_fingerprint(path) for path in (
            Path(__file__), ROOT / "scripts/prepare_parcae370.py", ROOT / "src/loopcd_repro/parcae.py",
            ROOT / "src/loopcd_repro/guidance.py", ROOT / "scripts/launch_huginn_r16_suite.py")}
        sys.path.insert(0, str(args.model / "source"))
        import torch
        from receval.models.parcae import ModelingParcae
        from parcae_lm.attention_backends.flash_attention import HAS_FA3
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and not HAS_FA3
        state["device_type"] = "cuda"
        state["device"] = torch.cuda.get_device_name(0)
        state["model_repo"] = "SandyResearch/parcae-370m"
        state["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "einops")}
        loads = []
        class StrictNative(ModelingParcae):
            def load_state_dict(self, state_dict, strict=True, assign=False):
                result = super().load_state_dict(state_dict, strict=True, assign=assign)
                loads.append({"keys": len(state_dict), "missing": result.missing_keys, "unexpected": result.unexpected_keys, "strict": True})
                return result
        model = StrictNative.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
        assert model.config.state_init == "like-init" and model.config.block_size == 2048
        assert len(model.transformer.prelude) == len(model.transformer.core_block) == len(model.transformer.coda) == 4
        state["loading"] = loads
        state["records"] = []
        for length in state["lengths"]:
            tokens = torch.tensor([[452, 2903, 312] * ((length + 2) // 3)], dtype=torch.long, device="cuda:0")[:, :length].contiguous()
            state["records"].extend(numerical_cases(model, tokens))
            save()
        verify_prepared(args.model)
        assert state["sources"] == {name: stream_fingerprint(ROOT / name) for name in state["sources"]}
        state.update(status="PASS", elapsed_seconds=time.time() - state["started_unix"], checks={
            "model_prepared_hashes_before_after": True, "sources_before_after": True, "strict_loading": True,
            "actual_gpu": True, "native_rng_and_trajectories": True, "short_and_context_limit": True, "all_32_cases": True,
        })
        validate_report(state)
        save()
        (args.output / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    except BaseException as error:
        state.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
