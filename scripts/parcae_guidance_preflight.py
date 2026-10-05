#!/usr/bin/env python3
"""Short real-checkpoint CPU guidance oracle, not GPU or benchmark validation."""
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
from prepare_parcae import verify_prepared, stream_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This CPU gate requires CUDA_VISIBLE_DEVICES=''")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "scope": __doc__, "started_unix": time.time()}
    def save():
        (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        verify_prepared(args.model)
        state["sources"] = {str(path.relative_to(ROOT)): stream_fingerprint(path) for path in (
            Path(__file__), ROOT / "scripts/prepare_parcae.py",
            ROOT / "src/loopcd_repro/parcae.py", ROOT / "src/loopcd_repro/guidance.py",
        )}
        state["model_provenance"] = stream_fingerprint(args.model / "model_provenance.json")
        sys.path.insert(0, str(args.model / "source"))
        import torch
        from receval.models.parcae import ModelingParcae
        from loopcd_repro.parcae import ParcaeGuidance, ParcaeGuidanceConfig
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        torch.set_num_threads(4)
        state["python"] = sys.version
        state["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "einops", "numpy")}
        load_records = []
        class StrictNative(ModelingParcae):
            def load_state_dict(self, state_dict, strict=True, assign=False):
                value = super().load_state_dict(state_dict, strict=True, assign=assign)
                load_records.append({"keys": len(state_dict), "missing": value.missing_keys,
                                     "unexpected": value.unexpected_keys, "strict": True})
                return value
        model = StrictNative.from_pretrained(args.model, device="cpu", dtype=torch.bfloat16)
        state["loading"] = load_records
        assert len(load_records) == 1 and load_records[0]["keys"] == 225
        assert model.config.state_init == "like-init" and model.config.use_fused_head == "pytorch"
        tokens = torch.tensor([[452, 2903, 312]], dtype=torch.long)
        state["tokens"] = tokens.tolist()
        records = []
        def digest(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for depth in (4, 8):
            captured = []
            native_core = model.core_block_forward
            def capture(*args, **kwargs):
                value = native_core(*args, **kwargs)
                captured.append(value.detach().clone())
                return value
            torch.manual_seed(42)
            model.core_block_forward = capture
            try:
                with torch.inference_mode():
                    baseline = model.forward_for_generation(tokens, num_steps=depth, past_key_values=None)["logits"]
            finally:
                del model.core_block_forward
            assert len(captured) == depth
            native_rng = torch.get_rng_state()
            def native_readout_override(raw):
                # Independent oracle uses the complete original native method,
                # replacing C's input only. No adapter readout helper is called.
                count = []
                def substitute(module, args):
                    count.append(1)
                    return (raw,) + args[1:]
                handle = model.transformer.C.register_forward_pre_hook(substitute)
                torch.manual_seed(42)
                try:
                    with torch.inference_mode():
                        result = model.forward_for_generation(tokens, num_steps=depth, past_key_values=None)["logits"]
                finally:
                    handle.remove()
                assert count == [1] and torch.equal(native_rng, torch.get_rng_state())
                return result
            early = native_readout_override(captured[0])
            for mode, omega, cap in (("baseline", .5, 1.), ("fixed", 0., 1.),
                                     ("adaptive", .5, 0.), ("hidden", 0., 1.),
                                     ("fixed", .5, 1.), ("adaptive", .5, 1.),
                                     ("hidden", 1., 1.), ("hidden", .75, 1.)):
                config = ParcaeGuidanceConfig(mode, depth, 1, omega, cap)
                if not config.enabled:
                    expected = baseline
                elif mode == "hidden":
                    raw = (captured[-1].float() + omega * (captured[-1].float() - captured[0].float())).to(torch.bfloat16)
                    expected = native_readout_override(raw)
                else:
                    probs = baseline.double().softmax(-1).sort(descending=True).values
                    strength = cap * (1 - probs[..., :1] + probs[..., 1:2]) if mode == "adaptive" else omega
                    expected = baseline.double() + strength * (baseline.double() - early.double())
                    if mode == "fixed":
                        # Match the declared FP32 arithmetic, independently of
                        # apply_guidance; FP64 remains the adaptive oracle.
                        expected = baseline.float() + omega * (baseline.float() - early.float())
                counts = {"core": 0, "C": 0, "coda": 0, "head": 0}
                def count(name):
                    def hook(*args):
                        counts[name] += 1
                    return hook
                handles = [model.transformer.C.register_forward_hook(count("C")),
                           model.lm_head.register_forward_hook(count("head"))]
                handles += [layer.register_forward_hook(count("coda")) for layer in model.transformer.coda]
                handles += [layer.register_forward_hook(count("core")) for layer in model.transformer.core_block]
                torch.manual_seed(42)
                adapter = ParcaeGuidance(model, config)
                try:
                    with torch.inference_mode():
                        output = adapter(tokens)
                finally:
                    for handle in handles:
                        handle.remove()
                logits = output["logits"]
                error = float((logits.double() - expected.double()).abs().max())
                exact = mode == "hidden" or not config.enabled or mode == "fixed"
                torch.testing.assert_close(logits.double(), expected.double(), atol=0 if exact else 2e-5, rtol=0 if exact else 2e-6)
                assert torch.isfinite(logits).all() and output["past_key_values"] is None
                assert torch.equal(native_rng, torch.get_rng_state())
                readouts = 2 if config.enabled and mode in {"fixed", "adaptive"} else 1
                assert counts == {"core": 8 * depth, "C": readouts, "coda": 8 * readouts, "head": readouts}
                assert "core_block_forward" not in model.__dict__ and not model.transformer.C._forward_pre_hooks
                records.append({"depth": depth, "mode": mode, "omega": omega, "cap": cap,
                                "max_abs_error": error, "exact_required": exact,
                                "logits_sha256": digest(logits), "rng_sha256": digest(native_rng),
                                "calls": counts, "observation": adapter.last_observation})
                state["records"] = records
                save()
        assert len(records) == 16 and not torch.cuda.is_initialized()
        verify_prepared(args.model)
        state.update(status="PASS", elapsed_seconds=time.time() - state["started_unix"],
                     checks={"strict_checkpoint_load": True, "sixteen_real_forward_cases": True,
                             "independent_native_readout_oracle": True, "rng_identical": True,
                             "recurrence_and_readout_counts": True, "source_unchanged": True,
                             "cpu_only_no_benchmark_scores": True})
        save()
        (args.output / "result.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "cases": len(records), "output": str(args.output)}), flush=True)
    except Exception as error:
        state.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
