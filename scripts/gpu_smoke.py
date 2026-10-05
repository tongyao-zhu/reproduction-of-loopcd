"""Check the real Ouro checkpoint against independent guidance/cache oracles."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from loopcd_repro.guidance import GuidanceConfig
from loopcd_repro.ouro import OuroGuidance
from loopcd_repro.runtime import load_ouro, provenance


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
    same_shape = actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    equal = same_shape and torch.equal(actual, expected)
    maximum = float((actual.float() - expected.float()).abs().max()) if same_shape and finite else None
    check(report, name, equal and finite,
          actual_shape=list(actual.shape), expected_shape=list(expected.shape),
          actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
          max_abs_diff=maximum, finite=finite)


@contextmanager
def observe(model):
    trace = {"loops": [], "head_calls": 0, "states": [], "norm_states": []}

    def loop(module, inputs, kwargs):
        trace["loops"].append(int(kwargs["current_ut"]))

    def head(module, inputs, output):
        trace["head_calls"] += 1

    def inner(module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise AssertionError("Unexpected native Ouro inner output")
        trace["states"] = output[1]

    def norm(module, inputs, output):
        trace["norm_states"].append(output)

    handles = [model.model.layers[0].register_forward_pre_hook(loop, with_kwargs=True),
               model.lm_head.register_forward_hook(head),
               model.model.register_forward_hook(inner),
               model.model.norm.register_forward_hook(norm)]
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def trace_check(report, name, trace, heads):
    check(report, name + "/execution", trace["loops"] == [0, 1, 2, 3]
          and trace["head_calls"] == heads, loops=trace["loops"],
          head_calls=trace["head_calls"], expected_head_calls=heads)
    check(report, name + "/normalized_states", len(trace["states"]) == 4
          and len(trace["norm_states"]) == 4
          and all(a is b for a, b in zip(trace["states"], trace["norm_states"])))


def manual_logits(final, early, config):
    """Independent Eq. 3/4 oracle; deliberately never calls apply_guidance."""
    if config.mode == "baseline" or (config.mode == "fixed" and config.omega == 0) or (
        config.mode == "adaptive" and config.omega_cap == 0
    ):
        return final
    z_r, z_1 = final.float(), early.float()
    if config.mode == "fixed":
        weight = float(config.omega)
    else:
        probabilities = torch.softmax(z_r, dim=-1)
        top_two = torch.topk(probabilities, 2, dim=-1).values
        weight = float(config.omega_cap) * (1.0 - (top_two[..., :1] - top_two[..., 1:2]))
    return z_r + weight * (z_r - z_1)


def settings(model):
    return [(target, name, hasattr(target, name), getattr(target, name, None))
            for target in (model, model.config)
            for name in ("early_exit_step", "early_exit_threshold")]


def restored(report, name, model, before, forward, had_instance_forward):
    attrs_ok = all(hasattr(target, key) == present and getattr(target, key, None) == value
                   for target, key, present, value in before)
    check(report, name, attrs_ok and model.forward == forward
          and ("forward" in model.__dict__) == had_instance_forward
          and getattr(model, "_loopcd_repro_guidance", None) is None)


def cache_equal(report, name, left, right):
    metadata = lambda cache: {
        "type": type(cache).__qualname__, "seen_tokens": cache._seen_tokens,
        "sequence_length": cache.get_seq_length(), "max_cache_size": cache.max_cache_size,
        "key_slots": len(cache.key_cache), "value_slots": len(cache.value_cache),
    }
    left_meta, right_meta = metadata(left), metadata(right)
    check(report, name + "/metadata", left_meta == right_meta, metadata=left_meta)
    compared, unequal = 0, []
    for attr in ("key_cache", "value_cache"):
        for index, (a, b) in enumerate(zip(getattr(left, attr), getattr(right, attr))):
            if a is None or b is None:
                if a is not b:
                    unequal.append(f"{attr}/{index}:missing")
            else:
                compared += 1
                if not torch.equal(a, b) or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    unequal.append(f"{attr}/{index}")
    check(report, name + "/all_kv_tensors", compared > 0 and not unequal,
          compared_tensors=compared, unequal_slots=unequal)


@torch.inference_mode()
def run(args, report):
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This checkpoint smoke requires a CUDA GPU")
    model, tokenizer = load_ouro(args.model, args.device)
    report["provenance"] = provenance(args.model, model)
    ids = tokenizer("Question: Water freezes at what temperature?\nAnswer:",
                    return_tensors="pt")["input_ids"].to(args.device)
    report["fixture"] = {"input_ids": ids.cpu().tolist(), "seed": 42,
                         "forced_token_rule": "repeat the final two prompt tokens",
                         "scope": "one unpadded sequence; all positions; prefill plus two forced cached steps"}
    before, forward = settings(model), model.forward
    had_instance_forward = "forward" in model.__dict__
    native_args = {"exit_at_step": 3, "exit_threshold": None, "use_weighted_exit": False}
    with observe(model) as trace:
        native = model(input_ids=ids, use_cache=False, logits_to_keep=0, **native_args)
    trace_check(report, "native/full", trace, 1)
    final = native.logits.clone()
    early = model.lm_head(trace["states"][0]).clone()
    check(report, "native/full_sequence_shape", final.shape[:2] == ids.shape,
          shape=list(final.shape))
    configurations = {
        "baseline": GuidanceConfig(mode="baseline"),
        "fixed_zero": GuidanceConfig(mode="fixed", omega=0.0),
        "adaptive_zero": GuidanceConfig(mode="adaptive", omega_cap=0.0),
        "fixed": GuidanceConfig(mode="fixed", omega=0.5),
        "adaptive": GuidanceConfig(mode="adaptive", omega_cap=1.0),
    }
    for name, config in configurations.items():
        with OuroGuidance(model, config, total_loops=4) as adapter:
            with observe(model) as seen:
                output = model(input_ids=ids, use_cache=False, logits_to_keep=0)
            trace_check(report, name + "/full", seen, 2 if config.enabled else 1)
            exact(report, name + "/manual_all_positions", output.logits,
                  manual_logits(final, early, config))
            report.setdefault("observations", {})[name] = adapter.last_observation
        restored(report, name + "/restore", model, before, forward, had_instance_forward)

    # Use a separately projected one-position native reference: BF16 GEMM
    # shape changes can change results, so full-vs-last is diagnostic only.
    with observe(model) as trace:
        last_native = model(input_ids=ids, use_cache=False, logits_to_keep=1, **native_args)
    trace_check(report, "native/last", trace, 1)
    last_early = model.lm_head(trace["states"][0][:, -1:, :])
    report["cross_shape_diagnostic"] = {
        "exact_assertion_required": False,
        "max_abs_diff": float((last_native.logits.float() - final[:, -1:, :].float()).abs().max()),
        "argmax_equal": bool(torch.equal(last_native.logits.argmax(-1), final[:, -1:, :].argmax(-1))),
    }
    for name in ("baseline", "fixed", "adaptive"):
        config = configurations[name]
        with OuroGuidance(model, config, total_loops=4):
            output = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            exact(report, name + "/last_same_shape", output.logits,
                  manual_logits(last_native.logits, last_early, config))
        restored(report, name + "/last_restore", model, before, forward, had_instance_forward)

    cache_class = sys.modules[type(model).__module__].UniversalTransformerCache
    slots = 4 * int(model.config.num_hidden_layers)
    blocks = [ids, ids[:, -2:-1], ids[:, -1:]]
    for name in ("fixed", "adaptive"):
        config = configurations[name]
        native_cache = cache_class(max_cache_size=slots)
        guided_cache = cache_class(max_cache_size=slots)
        offset = 0
        for step, block in enumerate(blocks):
            positions = torch.arange(offset, offset + block.shape[1], device=args.device)
            kwargs = {"input_ids": block, "use_cache": True, "cache_position": positions,
                      "logits_to_keep": 1}
            with observe(model) as trace:
                native_step = model(**kwargs, past_key_values=native_cache, **native_args)
            trace_check(report, f"{name}/cache{step}/native", trace, 1)
            native_cache = native_step.past_key_values
            early_step = model.lm_head(trace["states"][0][:, -1:, :])
            with OuroGuidance(model, config, total_loops=4):
                with observe(model) as seen:
                    guided_step = model(**kwargs, past_key_values=guided_cache)
                trace_check(report, f"{name}/cache{step}/guided", seen, 2)
            guided_cache = guided_step.past_key_values
            exact(report, f"{name}/cache{step}/logits", guided_step.logits,
                  manual_logits(native_step.logits, early_step, config))
            cache_equal(report, f"{name}/cache{step}", native_cache, guided_cache)
            restored(report, f"{name}/cache{step}/restore", model, before, forward, had_instance_forward)
            offset += block.shape[1]

    class IntendedFailure(Exception):
        pass

    try:
        with OuroGuidance(model, configurations["fixed"], total_loops=4):
            raise IntendedFailure("exercise context exception cleanup")
    except IntendedFailure:
        pass
    restored(report, "exception/restore", model, before, forward, had_instance_forward)
    torch.cuda.synchronize(torch.device(args.device))
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Ouro-1.4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/gpu_smoke.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke evidence: {args.output}")
    report = {"status": "RUNNING", "started_at": stamp(), "checks": [],
              "arguments": {key: str(value) for key, value in vars(args).items()}}
    started = time.monotonic()
    save(args.output, report)
    try:
        run(args, report)
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc())
        raise
    else:
        report["status"] = "PASS"
    finally:
        report.update(finished_at=stamp(), elapsed_seconds=time.monotonic() - started)
        save(args.output, report)
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                          "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
