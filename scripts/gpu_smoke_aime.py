"""Real Thinking checkpoint gate for registered AIME sampling and KV capacity.

The long gate prefills a repeated non-benchmark fixture into the native cache,
then executes one guided cached step. It does NOT claim an 8192-token sampled
trajectory, answer quality, or a long-run throughput/stability measurement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loopcd_repro.aime_protocol import (
    atomic_json, build_prompt, guidance_dict, hash_json, hash_text, load_protocol,
    load_questions, read_model_identity, smoke_bindings, stamp, stop_metadata, validate_execution, validate_smoke,
)
from generate_aime import generation_config, new_cache, observe_execution, set_seed


def check(report, name, passed, **details):
    report["checks"].append({"name": name, "passed": bool(passed), **details})
    if not passed:
        raise AssertionError(name)


@contextmanager
def native_fixed_depth(model):
    saved = [(target, key, hasattr(target, key), getattr(target, key, None))
             for target in (model, model.config)
             for key in ("early_exit_step", "early_exit_threshold")]
    for target in (model, model.config):
        target.early_exit_step, target.early_exit_threshold = 3, None
    try:
        yield
    finally:
        for target, key, exists, value in saved:
            if exists:
                setattr(target, key, value)
            else:
                delattr(target, key)


def sampled_fixture(model, tokenizer, prompt, protocol, device, settings=None, force_token=None, cap=32):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from loopcd_repro.ouro import OuroGuidance
    config = generation_config(protocol, cap)
    set_seed(42)
    cache = new_cache(model)
    ids = torch.tensor([prompt["prompt_token_ids"]], device=device)
    processors = None
    if force_token is not None:
        class ForceToken(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores.fill_(-float("inf"))
                scores[:, force_token] = 0
                return scores
        processors = LogitsProcessorList([ForceToken()])
    mode = "native" if settings is None else settings.mode
    if settings is not None and not settings.enabled and mode != "baseline":
        mode += "_zero"
    context = native_fixed_depth(model) if settings is None else OuroGuidance(model, settings, total_loops=4)
    with context, observe_execution(model, cache, mode) as observation, torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             past_key_values=cache, generation_config=config,
                             logits_to_keep=1, logits_processor=processors)
    generated = out[0, ids.shape[1]:].tolist()
    validate_execution(observation, "baseline" if settings is None or not settings.enabled else settings.mode,
                       ids.shape[1], len(generated), cache.max_cache_size)
    return {"generated_token_ids": generated, **stop_metadata(generated, cap), "seed": 42,
            "execution_observation": observation, "generation_config": config.to_dict(),
            "forced_fixture_token": force_token}


def long_cache_gate(model, tokenizer, protocol, maximum_prompt_length, settings, device):
    import torch
    from loopcd_repro.ouro import OuroGuidance
    target = maximum_prompt_length + protocol["sampling"]["max_new_tokens"]
    pattern = tokenizer.encode("This is a cache capacity check. ", add_special_tokens=False)
    if not pattern:
        raise ValueError("Empty capacity fixture")
    cache = new_cache(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with native_fixed_depth(model), observe_execution(model, cache, "native") as prefill_observation, torch.inference_mode():
        offset = 0
        while offset < target:
            size = min(512, target - offset)
            block = [pattern[(offset + i) % len(pattern)] for i in range(size)]
            ids = torch.tensor([block], device=device)
            out = model(input_ids=ids, attention_mask=torch.ones((1, offset + size), dtype=torch.long, device=device),
                        cache_position=torch.arange(offset, offset + size, device=device),
                        past_key_values=cache, use_cache=True, logits_to_keep=1,
                        exit_at_step=3, exit_threshold=None, use_weighted_exit=False)
            if out.past_key_values is not cache or not bool(torch.isfinite(out.logits).all()):
                raise ValueError("Nonfinite output or replaced cache during long prefill")
            offset += size
            if cache.get_seq_length() != offset:
                raise ValueError("Native cache prefill length diverged")
            del out, ids
    prefilling_length = cache.get_seq_length()
    with OuroGuidance(model, settings, total_loops=4) as adapter, observe_execution(model, cache, "adaptive") as decode_observation, torch.inference_mode():
        out = model(input_ids=torch.tensor([[pattern[0]]], device=device),
                    attention_mask=torch.ones((1, target + 1), dtype=torch.long, device=device),
                    cache_position=torch.tensor([target], device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    if not bool(torch.isfinite(out.logits).all()) or out.past_key_values is not cache:
        raise ValueError("Invalid full-budget guided cached decode")
    for slots in (cache.key_cache, cache.value_cache):
        if len(slots) != cache.max_cache_size or any(t is None or t.shape[2] != target + 1 for t in slots):
            raise ValueError("Incomplete long native KV slots")
    torch.cuda.synchronize(device)
    result = {"status": "PASS", "scope": "native_cache_prefill_plus_one_cached_decode",
              "max_new_tokens": 8192, "max_prompt_tokens_all60": maximum_prompt_length,
              "target_prefill_tokens": target, "actual_prefill_tokens": prefilling_length,
              "final_cache_length": cache.get_seq_length(), "cache_slots": cache.max_cache_size,
              "measured_autoregressive_tokens": 1, "prefill_chunk_tokens": 512,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "elapsed_seconds": time.monotonic() - started,
              "final_guidance_observation": adapter.last_observation,
              "prefill_execution_observation": prefill_observation,
              "decode_execution_observation": decode_observation,
              "limitation": "Repeated fixture native prefill and one real cached forward; no long sampled AIME trajectory or throughput claim."}
    del out, cache
    torch.cuda.empty_cache()
    return result


def run(args, report):
    import torch
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    from loopcd_repro.runtime import load_ouro, provenance
    from gpu_smoke import observe, manual_logits, cache_equal
    protocol = load_protocol(args.protocol)
    identity = read_model_identity(args.model, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("AIME smoke requires a coordinated CUDA device")
    model, tokenizer = load_ouro(args.model, args.device)
    source = provenance(args.model, model)
    source["precision"] = "BF16 model; FP32 guidance before native sampling"
    report["source"] = source
    report["bindings"] = smoke_bindings(identity, source)
    check(report, "protocol_and_identity", source["loaded_model_code_sha256"] == identity["model_code_sha256"])
    prompts = [build_prompt(tokenizer, q, protocol) for year in (2024, 2025) for q in load_questions(args.data_root, year)]
    report["prompt_audit"] = [{"task_id": p["task_id"], "prompt_sha256": p["prompt_sha256"],
                               "prompt_token_ids_sha256": p["prompt_token_ids_sha256"],
                               "tokens": len(p["prompt_token_ids"])} for p in prompts]
    check(report, "all60_prompts", len(prompts) == 60 and len({p["task_id"] for p in prompts}) == 60,
          max_prompt_tokens=max(len(p["prompt_token_ids"]) for p in prompts),
          native_tokenize_equal_to_render_encode=True)
    fixture_text = "Return a short greeting."
    fixture = build_prompt(tokenizer, {"task_id": "nonbenchmark-smoke", "question": fixture_text,
                                      "question_sha256": hash_text(fixture_text)}, protocol)
    report["fixture"] = fixture
    configs = {arm: GuidanceConfig(**{key: value for key, value in guidance_dict(arm, identity, protocol).items() if key != "total_loops"}) for arm in ("baseline", "fixed", "adaptive")}
    native = sampled_fixture(model, tokenizer, fixture, protocol, args.device)
    report["short_generation"] = {"native": native}
    for name, config in (("baseline", configs["baseline"]), ("fixed_zero", GuidanceConfig(mode="fixed", omega=0)),
                         ("adaptive_zero", GuidanceConfig(mode="adaptive", omega_cap=0))):
        observed = sampled_fixture(model, tokenizer, fixture, protocol, args.device, config)
        report["short_generation"][name] = observed
        check(report, "native_" + name + "_seed_parity", native["generated_token_ids"] == observed["generated_token_ids"]
              and native["stop_reason"] == observed["stop_reason"], seed=42)
    eos = sampled_fixture(model, tokenizer, fixture, protocol, args.device, configs["baseline"], force_token=2, cap=3)
    ordinary = next(token for token in tokenizer.encode(" hello", add_special_tokens=False) if token != 2)
    capped = sampled_fixture(model, tokenizer, fixture, protocol, args.device, configs["baseline"], force_token=ordinary, cap=3)
    check(report, "eos_stop", eos["stop_reason"] == "eos" and eos["generated_token_ids"] == [2], fixture=eos)
    check(report, "cap_stop", capped["cap_hit"] and len(capped["generated_token_ids"]) == 3, fixture=capped)
    ids = torch.tensor([fixture["prompt_token_ids"]], device=args.device)
    native_kwargs = {"exit_at_step": 3, "exit_threshold": None, "use_weighted_exit": False}
    with native_fixed_depth(model), torch.inference_mode():
        with observe(model) as trace:
            native_output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=0, **native_kwargs)
        final = native_output.logits.clone()
        early = model.lm_head(trace["states"][0]).clone()
        for mode in ("fixed", "adaptive"):
            config = configs[mode]
            with OuroGuidance(model, config, total_loops=4):
                guided = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=0)
            wanted = manual_logits(final, early, config)
            check(report, mode + "_formula_all_positions", torch.equal(guided.logits, wanted) and bool(torch.isfinite(wanted).all()),
                  shape=list(wanted.shape), cap=config.omega_cap,
                  max_abs_diff=float((guided.logits.float() - wanted.float()).abs().max()))
            native_cache, guided_cache = new_cache(model), new_cache(model)
            offset = 0
            for index, block in enumerate((ids, ids[:, -1:], ids[:, -1:])):
                arguments = {"input_ids": block, "attention_mask": torch.ones((1, offset + block.shape[1]), dtype=torch.long, device=args.device),
                             "cache_position": torch.arange(offset, offset + block.shape[1], device=args.device),
                             "use_cache": True, "logits_to_keep": 1}
                with observe(model) as seen:
                    native_step = model(**arguments, past_key_values=native_cache, **native_kwargs)
                earlier = model.lm_head(seen["states"][0][:, -1:, :])
                with OuroGuidance(model, config, total_loops=4):
                    guided_step = model(**arguments, past_key_values=guided_cache)
                oracle = manual_logits(native_step.logits, earlier, config)
                check(report, f"{mode}_cached_formula_{index}", torch.equal(guided_step.logits, oracle) and bool(torch.isfinite(oracle).all()))
                cache_equal(report, f"{mode}_cache_{index}", native_cache, guided_cache)
                offset += block.shape[1]
            check(report, mode + "_cache_oracle", native_cache.get_seq_length() == offset and guided_cache.get_seq_length() == offset)
            del native_cache, guided_cache, guided_step, native_step, oracle, earlier, guided, wanted
        del native_output, final, early, trace, seen
    # Free short-oracle tensors before measuring the long native-cache gate.
    del ids
    report["resource_gate"] = long_cache_gate(model, tokenizer, protocol, max(len(p["prompt_token_ids"]) for p in prompts), configs["adaptive"], args.device)
    check(report, "native_long_cache_capacity", report["resource_gate"]["status"] == "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite smoke evidence")
    report = {"schema_version": 1, "kind": "aime_gpu_smoke", "status": "RUNNING",
              "started_at": stamp(), "checks": [], "arguments": {k: str(v) for k, v in vars(args).items()}}
    atomic_json(args.output, report)
    started = time.monotonic()
    try:
        run(args, report)
        report.update(status="PASS", finished_at=stamp(), elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output, report)
        validate_smoke(args.output, report["bindings"])
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc(), finished_at=stamp(), elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output, report)
        raise
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
