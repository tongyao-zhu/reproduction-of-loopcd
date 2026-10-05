"""Validate real Huginn-Hidden against independent raw-state/readout oracles.

Run on an idle, explicitly allocated GPU. The cached oracle intentionally
executes a second coda pass to overwrite only its current-token coda slots;
the adapter under test must execute one coda/head and retain native core KV.
This oracle overhead is a validation device, not part of Hidden inference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from loopcd_repro.huginn import HuginnHiddenConfig, HuginnHiddenGuidance, load_huginn
from loopcd_repro.runtime import provenance


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def check(report, name, passed, **details):
    report["checks"].append({"name": name, "passed": bool(passed), **details})
    if not passed:
        raise AssertionError(name)


def exact(report, name, actual, expected):
    shape_matches = actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    maximum = float((actual.float() - expected.float()).abs().max()) if shape_matches and finite else None
    check(report, name, shape_matches and finite and torch.equal(actual, expected),
          actual_shape=list(actual.shape), expected_shape=list(expected.shape),
          actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
          max_abs_diff=maximum, finite=finite)


@contextmanager
def observe(model, depth):
    trace = {"core_calls": 0, "adapter_calls": 0, "coda_calls": [], "head_calls": 0,
             "raw_states": {}, "initial_states": []}

    def adapter(module, inputs):
        if trace["adapter_calls"] % depth == 0:
            trace["initial_states"].append(inputs[0][..., :model.config.n_embd].detach().clone())
        trace["adapter_calls"] += 1

    def core(module, inputs, output):
        trace["core_calls"] += 1
        trace["raw_states"][trace["core_calls"]] = output.detach().clone()

    def coda(module, inputs):
        trace["coda_calls"].append(id(module))

    def head(module, inputs, output):
        trace["head_calls"] += 1

    handles = [model.transformer.adapter.register_forward_pre_hook(adapter),
               model.transformer.core_block[-1].register_forward_hook(core),
               model.lm_head.register_forward_hook(head)]
    handles.extend(layer.register_forward_pre_hook(coda) for layer in model.transformer.coda)
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def execution(report, name, model, trace, depth, forwards=1):
    check(report, name, trace["core_calls"] == trace["adapter_calls"] == depth * forwards
          and trace["head_calls"] == forwards
          and trace["coda_calls"] == [id(layer) for layer in model.transformer.coda] * forwards,
          recurrent_iterations=trace["core_calls"], expected_recurrent_iterations=depth * forwards,
          adapter_calls=trace["adapter_calls"], head_calls=trace["head_calls"],
          coda_layer_calls=len(trace["coda_calls"]), forwards=forwards)


def readout_oracle(model, final_raw, reference_raw, omega, *, cache=None, positions=None):
    """Independent Equation 2 and native post-recurrence interface.

    This does not invoke HuginnHiddenGuidance or its hooks. The two raw
    states come from an unmodified native trajectory observed separately.
    """
    guided_raw = (final_raw.float() + float(omega) * (final_raw.float() - reference_raw.float())).to(final_raw.dtype)
    hidden = model.transformer.ln_f(guided_raw)
    frequencies = model.freqs_cis[:, :hidden.shape[1]] if positions is None else model.freqs_cis[:, positions]
    for index, layer in enumerate(model.transformer.coda):
        hidden = layer(hidden, frequencies, torch.tensor(-(index + 1), dtype=torch.long), None, cache)
    return model.lm_head(model.transformer.ln_f(hidden)).float()


def cache_equal(report, name, left, right):
    metadata = lambda cache: {
        "type": type(cache).__qualname__, "seen_tokens": cache.get_seq_length(),
        "lookup_strategy": cache.lookup_strategy,
        "key_slots": sorted(cache.key_cache), "value_slots": sorted(cache.value_cache),
    }
    check(report, name + "/metadata", metadata(left) == metadata(right), metadata=metadata(left))
    tensors = 0
    failures = []
    for field in ("key_cache", "value_cache"):
        slots_left, slots_right = getattr(left, field), getattr(right, field)
        for slot in slots_left:
            if set(slots_left[slot]) != set(slots_right[slot]):
                failures.append(f"{field}/{slot}:positions")
                continue
            for position, a in slots_left[slot].items():
                b = slots_right[slot][position]
                tensors += 1
                if not torch.equal(a, b) or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    failures.append(f"{field}/{slot}/{position}")
    check(report, name + "/all_tensors", tensors > 0 and not failures,
          compared_tensors=tensors, unequal_slots=failures)


def restored(report, name, model, methods):
    check(report, name, all(getattr(model, key) == value for key, value in methods.items())
          and "forward" not in model.__dict__ and "generate" not in model.__dict__
          and getattr(model, "_loopcd_hidden_guidance", None) is None
          and not model.transformer.core_block[-1]._forward_hooks
          and not model.transformer.ln_f._forward_pre_hooks
          and not model.lm_head._forward_hooks
          and all(not layer._forward_pre_hooks for layer in model.transformer.coda))


@torch.inference_mode()
def run(args, report):
    if not torch.cuda.is_available():
        raise RuntimeError("The real checkpoint gate requires CUDA")
    torch.set_num_threads(4)
    model, tokenizer = load_huginn(args.model, args.device)
    report["provenance"] = provenance(args.model, model)
    report["provenance"]["precision"] = "BF16 native model; FP32 raw hidden blend cast back to BF16 before native ln_f/coda/head"
    methods = {name: getattr(model, name) for name in ("forward", "generate", "initialize_state")}
    ids = tokenizer("def add(a, b):\n    return", return_tensors="pt")["input_ids"].to(args.device)
    report["fixture"] = {"input_ids": ids.cpu().tolist(), "seed": 991,
                         "forced_token_rule": "repeat final two prompt tokens",
                         "scope": "one unpadded real-checkpoint sequence; all-token logits and cached generation"}
    baseline_states = {}
    for depth in (16, 32):
        torch.manual_seed(991)
        with observe(model, depth) as native_trace:
            native = model(ids, num_steps=depth, use_cache=False)
        native_rng = torch.cuda.get_rng_state(torch.device(args.device)).clone()
        baseline_states[depth] = {"logits": native.logits.detach().clone(),
                                  "raw": native_trace["raw_states"],
                                  "initial": native_trace["initial_states"][0]}
        execution(report, f"native/R{depth}/execution", model, native_trace, depth)
        check(report, f"native/R{depth}/all_token_logits", native.logits.shape[:2] == ids.shape,
              shape=list(native.logits.shape))
        initial = native_trace["initial_states"][0]
        check(report, f"native/R{depth}/nonzero_random_initialization", bool((initial != 0).any())
              and float(initial.float().std()) > 0, std=float(initial.float().std()),
              nonzero_fraction=float((initial != 0).float().mean()))
        # Calling the unchanged native initializer with the same shape/seed
        # must reproduce the h0 actually supplied to the recurrent adapter.
        torch.manual_seed(991)
        expected_initial = model.initialize_state(model.transformer.wte(ids))
        exact(report, f"native/R{depth}/initializer_identity", initial, expected_initial)
        for name, config in (
            ("baseline", HuginnHiddenConfig("baseline", depth, 6, .5)),
            ("zero", HuginnHiddenConfig("hidden", depth, 6, 0.0)),
        ):
            torch.manual_seed(991)
            with HuginnHiddenGuidance(model, config) as adapter:
                with observe(model, depth) as trace:
                    output = model(ids, use_cache=False)
                execution(report, f"{name}/R{depth}/execution", model, trace, depth)
                exact(report, f"{name}/R{depth}/seeded_logits", output.logits, native.logits)
                exact(report, f"{name}/R{depth}/initial_state", trace["initial_states"][0], initial)
                check(report, f"{name}/R{depth}/rng_consumption", torch.equal(
                    torch.cuda.get_rng_state(torch.device(args.device)), native_rng))
                check(report, f"{name}/R{depth}/no_extra_readout", adapter.last_observation["extra_lm_head_calls"] == 0)
            restored(report, f"{name}/R{depth}/restored", model, methods)

    settings = {
        "mc_R32_h6": HuginnHiddenConfig("hidden", 32, 6, .5),
        "mc_R16_h7": HuginnHiddenConfig("hidden", 16, 7, .5),
        "generation_R32_h7": HuginnHiddenConfig("hidden", 32, 7, .3),
        "generation_R16_h6": HuginnHiddenConfig("hidden", 16, 6, .3),
        "half_depth_R16_h6": HuginnHiddenConfig("hidden", 16, 6, .5),
    }
    report["settings"] = {name: asdict(config) for name, config in settings.items()}
    for name, config in settings.items():
        raw = baseline_states[config.total_loops]["raw"]
        oracle = readout_oracle(model, raw[config.total_loops], raw[config.reference_loop], config.omega)
        torch.manual_seed(991)
        with HuginnHiddenGuidance(model, config) as adapter:
            with observe(model, config.total_loops) as trace:
                output = model(ids, use_cache=False)
            execution(report, name + "/execution", model, trace, config.total_loops)
            exact(report, name + "/manual_all_positions", output.logits, oracle)
            exact(report, name + "/native_initialization", trace["initial_states"][0], baseline_states[config.total_loops]["initial"])
            exact(report, name + "/recurrence_unmodified", trace["raw_states"][config.total_loops], raw[config.total_loops])
            check(report, name + "/guidance_effect", not torch.equal(output.logits, baseline_states[config.total_loops]["logits"]))
            report.setdefault("observations", {})[name] = adapter.last_observation
        restored(report, name + "/restored", model, methods)

    # Independent cached oracle: run native recurrence once, then perform
    # the mathematical readout on captured raw states. Its extra coda pass
    # overwrites ONLY current-token negative-index coda slots with guided
    # values, keeping prior guided coda history and native core history.
    config = settings["half_depth_R16_h6"]
    cache_class = sys.modules[type(model).__module__].HuginnDynamicCache
    oracle_cache = cache_class(lookup_strategy="full")
    guided_cache = HuginnHiddenGuidance(model, config).new_cache()
    check(report, "new_cache/native_full", isinstance(guided_cache, cache_class)
          and guided_cache.lookup_strategy == "full" and guided_cache.get_seq_length() == 0)
    offset = 0
    for step, block in enumerate((ids, ids[:, -2:-1], ids[:, -1:])):
        positions = torch.arange(offset, offset + block.shape[1], device=args.device)
        call = dict(input_ids=block, use_cache=True, cache_position=positions)
        torch.manual_seed(710 + step)
        with observe(model, config.total_loops) as native_trace:
            model(**call, num_steps=config.total_loops, past_key_values=oracle_cache)
        oracle = readout_oracle(model, native_trace["raw_states"][config.total_loops],
                               native_trace["raw_states"][config.reference_loop], config.omega,
                               cache=oracle_cache, positions=positions)
        torch.manual_seed(710 + step)
        with HuginnHiddenGuidance(model, config):
            with observe(model, config.total_loops) as trace:
                output = model(**call, past_key_values=guided_cache)
            execution(report, f"cache/step{step}/execution", model, trace, config.total_loops)
            exact(report, f"cache/step{step}/manual_logits", output.logits, oracle)
            check(report, f"cache/step{step}/same_cache_object", output.past_key_values is guided_cache)
            cache_equal(report, f"cache/step{step}", guided_cache, oracle_cache)
        offset += block.shape[1]
        check(report, f"cache/step{step}/single_updates", guided_cache.get_seq_length() == offset
              and len(guided_cache.key_cache) == 2 + 4 * config.total_loops + 2
              and all(len(slot) == offset for slot in guided_cache.key_cache.values()),
              seen_tokens=guided_cache.get_seq_length(), expected_tokens=offset,
              cache_slots=len(guided_cache.key_cache))
        restored(report, f"cache/step{step}/restored", model, methods)

    for name, wrong in (
        ("guidance", HuginnHiddenConfig("baseline", 16, 6, .5)),
        ("depth", HuginnHiddenConfig("hidden", 32, 6, .5)),
        ("reference", HuginnHiddenConfig("hidden", 16, 7, .5)),
    ):
        rejected = False
        before = guided_cache.get_seq_length()
        try:
            with HuginnHiddenGuidance(model, wrong):
                model(ids[:, -1:], use_cache=True, past_key_values=guided_cache)
        except ValueError as error:
            rejected = "different or unknown" in str(error)
        check(report, "cache/reject_wrong_" + name, rejected and guided_cache.get_seq_length() == before)
        restored(report, "cache/reject_wrong_" + name + "/restored", model, methods)

    generation_config = settings["generation_R16_h6"]
    torch.manual_seed(1201)
    with HuginnHiddenGuidance(model, generation_config) as adapter:
        cache = adapter.new_cache()
        with observe(model, generation_config.total_loops) as trace:
            generated = model.generate(ids, max_new_tokens=3, min_new_tokens=3,
                                       do_sample=False, use_cache=True, past_key_values=cache,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                       return_dict_in_generate=True)
        execution(report, "generation/execution", model, trace, generation_config.total_loops, forwards=3)
        check(report, "generation/three_new_tokens", generated.sequences.shape == (1, ids.shape[1] + 3))
        check(report, "generation/native_cache_object", generated.past_key_values is cache
              and isinstance(cache, cache_class) and cache.get_seq_length() == ids.shape[1] + 2)
        check(report, "generation/guidance_was_applied", adapter.last_observation["guidance_applied"])
        report["generation"] = {"settings": asdict(generation_config),
                                "generated_token_ids": generated.sequences[:, ids.shape[1]:].cpu().tolist(),
                                "generated_text": tokenizer.decode(generated.sequences[0, ids.shape[1]:]),
                                "scope": "three-token cached integration test, not a benchmark result"}
    restored(report, "generation/restored", model, methods)
    torch.cuda.synchronize(torch.device(args.device))
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/huginn-0125")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/gpu_smoke_huginn.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke evidence: {args.output}")
    report = {"status": "RUNNING", "started_at": stamp(), "checks": [],
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "arguments": {key: str(value) for key, value in vars(args).items()},
              "scope": "real pinned Huginn checkpoint implementation gate; no benchmark or speedup claim"}
    start = time.monotonic()
    save(args.output, report)
    try:
        run(args, report)
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc())
        raise
    else:
        report["status"] = "PASS"
    finally:
        report.update(finished_at=stamp(), elapsed_seconds=time.monotonic() - start)
        save(args.output, report)
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                          "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
