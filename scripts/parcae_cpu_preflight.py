#!/usr/bin/env python3
"""CPU-only checkpoint/interface gate; does not evaluate LoopCD or a benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from prepare_parcae import verify_prepared


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only gate")
    if sys.version_info < (3, 11):
        raise RuntimeError("The pinned official Parcae package requires Python >=3.11")
    model_path = args.model.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = {
        "status": "running",
        "scope": "Strict native checkpoint loading and short CPU forwards only; no LoopCD guidance, GPU gate, task scores, or context-length validation.",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model": str(model_path),
        "script_sha256": sha256(Path(__file__)),
        "preparation_script_sha256": sha256(Path(__file__).with_name("prepare_parcae.py")),
    }
    write_json(output / "status.json", state)
    try:
        verify_prepared(model_path)
        state["provenance_sha256"] = sha256(model_path / "model_provenance.json")
        state["phase"] = "loading_native_checkpoint"
        write_json(output / "status.json", state)
        sys.path.insert(0, str(model_path / "source"))
        import torch
        from receval.models.parcae import ModelingParcae
        from parcae_lm.tokenizer import Tokenizer
        from parcae_lm.attention_backends.flash_attention import HAS_FA3

        torch.set_num_threads(4)
        assert not torch.cuda.is_available()
        assert not torch.cuda.is_initialized()
        state["packages"] = {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "tokenizers", "numpy", "einops", "safetensors", "huggingface_hub"]
        }
        loading_records = []

        class StrictNativeParcae(ModelingParcae):
            def load_state_dict(self, state_dict, strict=True, assign=False):
                # Keep the official loader, but reject its otherwise-silent key mismatch.
                result = super().load_state_dict(state_dict, strict=True, assign=assign)
                loading_records.append({
                    "official_requested_strict": strict,
                    "effective_strict": True,
                    "loaded_keys": len(state_dict),
                    "missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys),
                })
                return result

        model = StrictNativeParcae.from_pretrained(model_path, device="cpu", dtype=torch.bfloat16)
        assert len(loading_records) == 1
        assert not loading_records[0]["missing_keys"] and not loading_records[0]["unexpected_keys"]
        assert model.config.state_init == "like-init"
        assert model.config.mean_recurrence == 8
        assert model.config.block_size == 2048
        assert model.config.padded_vocab_size == 32768
        assert len(model.transformer.prelude) == len(model.transformer.core_block) == len(model.transformer.coda) == 8
        assert all(p.device.type == "cpu" for p in model.parameters())
        state["loading"] = loading_records[0]
        state["parameter_count_unique"] = sum(p.numel() for p in model.parameters())
        state["loaded_at_seconds"] = time.time() - started
        state["attention"] = {"config": model.config.attn_impl, "has_fa3": HAS_FA3, "actual": "SDPA"}
        assert not HAS_FA3

        tokenizer = Tokenizer.from_directory(model_path)
        text = "The answer is"
        ids = tokenizer.encode(text, return_tensors=False)
        assert ids and len(ids) <= 8 and all(0 <= i < 32768 for i in ids)
        assert ids == tokenizer.processor.encode(text, add_special_tokens=False)
        raw_tokenizer = json.loads((model_path / "tokenizer.json").read_text())
        state["tokenizer"] = {
            "fixture": text, "token_ids": ids,
            "native_bos_id": tokenizer.bos_id,
            "native_eos_id": tokenizer.eos_id,
            "native_pad_id": tokenizer.pad_id,
            "raw_special_tokens": [v for v in raw_tokenizer.get("added_tokens", []) if v.get("special")],
            "note": "Native wrapper properties are recorded as observed; no BOS/EOS/PAD registration or prompt rule is changed by this gate.",
        }
        state["phase"] = "native_short_forward_checks"
        write_json(output / "status.json", state)
        tokens = torch.tensor([ids], dtype=torch.long)

        def tensor_sha(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def run(depth, seed, observe):
            record = {"depth": depth, "seed": seed, "observed": observe}
            torch.manual_seed(seed)
            record["rng_before"] = tensor_sha(torch.get_rng_state())
            native_init = model.initialize_state
            native_core = model.core_block_forward
            hooks = []
            counts = {"prelude": 0, "core_layers": 0, "projection": 0, "coda": 0, "norm": 0, "head": 0}
            states = []

            def count_hook(name):
                def hook(module, inputs, result):
                    counts[name] += 1
                return hook

            def observed_init(*a, **kw):
                value = native_init(*a, **kw)
                record["initial_state_sha256"] = tensor_sha(value)
                record["initial_state_nonzero"] = bool(torch.count_nonzero(value))
                return value

            def observed_core(*a, **kw):
                value = native_core(*a, **kw)
                states.append(tensor_sha(value))
                return value

            if observe:
                model.initialize_state = observed_init
                model.core_block_forward = observed_core
                for name, modules in [("prelude", model.transformer.prelude), ("core_layers", model.transformer.core_block), ("coda", model.transformer.coda)]:
                    hooks.extend(m.register_forward_hook(count_hook(name)) for m in modules)
                for name, module in [("projection", model.transformer.C), ("norm", model.transformer.ln_f), ("head", model.lm_head)]:
                    hooks.append(module.register_forward_hook(count_hook(name)))
            try:
                with torch.inference_mode():
                    result = model.forward_for_generation(tokens, num_steps=depth, past_key_values=None)
                logits = result["logits"]
                assert list(logits.shape) == [1, len(ids), 32768]
                assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
                assert result["past_key_values"] is None
                record.update(logits_sha256=tensor_sha(logits), rng_after=tensor_sha(torch.get_rng_state()), finite=True)
                if observe:
                    assert counts == {"prelude": 8, "core_layers": 8 * depth, "projection": 1, "coda": 8, "norm": 1, "head": 1}, counts
                    assert len(states) == depth and record["initial_state_nonzero"]
                    record.update(calls=counts, recurrent_state_sha256=states)
                return record
            finally:
                model.initialize_state = native_init
                model.core_block_forward = native_core
                for hook in hooks:
                    hook.remove()

        records = []
        for depth in (4, 8):
            native = run(depth, 42, False)
            observed = run(depth, 42, True)
            different = run(depth, 43, True)
            assert native["logits_sha256"] == observed["logits_sha256"]
            assert native["rng_before"] == observed["rng_before"] and native["rng_after"] == observed["rng_after"]
            assert observed["initial_state_sha256"] != different["initial_state_sha256"]
            records.extend([native, observed, different])
        assert records[1]["initial_state_sha256"] == records[4]["initial_state_sha256"]
        assert records[1]["rng_after"] == records[4]["rng_after"]
        assert not torch.cuda.is_initialized()
        state.update(status="PASS", phase="completed", native_forward_records=records, elapsed_seconds=time.time() - started)
        state["checks"] = {
            "prepared_files_sha256_verified": True,
            "strict_native_checkpoint_load": True,
            "all_parameters_cpu": True,
            "native_and_observed_logits_identical": True,
            "same_seed_same_initial_state_across_depths": True,
            "different_seed_changes_initial_state": True,
            "rng_stream_preserved_by_observation": True,
            "depth_and_readout_counts_exact": True,
            "cuda_not_initialized": True,
        }
        write_json(output / "result.json", state)
        write_json(output / "status.json", state)
        print(json.dumps({"status": "PASS", "output": str(output), "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
    except Exception as error:
        state.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.time() - started)
        write_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
