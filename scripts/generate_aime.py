"""Registered Ouro-Thinking AIME sampling, interleaved by question/sample/arm.

Only question files are opened. No answers, raw datasets, judge, or scorer are
loaded here. Resume revalidates every completed record before generation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loopcd_repro.aime_protocol import (
    ARMS, DATA_MANIFEST_SHA256, PROTOCOL_ID, PROTOCOL_SHA256, QUESTIONS_SHA256,
    atomic_json, build_prompt, guidance_dict, hash_json, hash_text, load_protocol,
    load_questions, output_lock, paired_config, read_completed, read_model_identity,
    repair_partial_tail, sample_seed, sha256_file, smoke_bindings, stamp,
    stop_metadata, validate_execution, validate_generation_config, validate_sample, validate_smoke,
)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generation_config(protocol, max_new_tokens=None):
    from transformers import GenerationConfig
    values = dict(protocol["sampling"])
    if max_new_tokens is not None:
        values["max_new_tokens"] = max_new_tokens
    # Construct afresh: never inherit checkpoint suppressions, beams, or stops.
    result = GenerationConfig(**values, return_dict_in_generate=False,
                              output_scores=False, output_logits=False,
                              output_attentions=False, output_hidden_states=False)
    result.validate()
    resolved = result.to_dict()
    validate_generation_config(resolved, resolved["max_new_tokens"])
    return result


def new_cache(model):
    cls = sys.modules[type(model).__module__].UniversalTransformerCache
    cache = cls(max_cache_size=4 * int(model.config.num_hidden_layers))
    if cache.get_seq_length() != 0 or cache.key_cache or cache.value_cache:
        raise ValueError("Native cache did not start empty")
    return cache


@contextmanager
def observe_execution(model, cache, mode):
    heads = 1 if mode in ("baseline", "native", "fixed_zero", "adaptive_zero") else 2
    observation = {"forward_calls": 0, "loop_calls": 0, "head_calls": 0,
                   "observed_loop_pattern_valid": True, "expected_head_calls_per_forward": heads,
                   "cache_type": type(cache).__name__, "cache_slots": cache.max_cache_size,
                   "fresh_cache_initial_length": cache.get_seq_length()}
    current = {"loops": [], "heads": 0}

    def pre(module, inputs, kwargs):
        current["loops"], current["heads"] = [], 0
        if kwargs.get("past_key_values") is not cache:
            raise ValueError("Generation replaced the native cache")

    def loop(module, inputs, kwargs):
        current["loops"].append(int(kwargs["current_ut"]))

    def head(module, inputs, output):
        current["heads"] += 1

    def post(module, inputs, output):
        valid = current["loops"] == [0, 1, 2, 3] and current["heads"] == heads
        observation["observed_loop_pattern_valid"] &= valid
        observation["forward_calls"] += 1
        observation["loop_calls"] += len(current["loops"])
        observation["head_calls"] += current["heads"]
        if not valid or output.past_key_values is not cache:
            raise ValueError("Native recurrence, head count, or returned cache changed")

    handles = [model.register_forward_pre_hook(pre, with_kwargs=True),
               model.model.layers[0].register_forward_pre_hook(loop, with_kwargs=True),
               model.lm_head.register_forward_hook(head), model.register_forward_hook(post)]
    try:
        yield observation
    finally:
        for handle in handles:
            handle.remove()
        observation["final_cache_length"] = cache.get_seq_length()


def generation_row(model, tokenizer, prompt, sample_id, config, device):
    import torch
    from transformers import GenerationConfig
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    mode = config["guidance"]["mode"]
    settings = {key: value for key, value in config["guidance"].items() if key != "total_loops"}
    seed = sample_seed(prompt["task_id"], sample_id)
    set_seed(seed)
    inputs = torch.tensor([prompt["prompt_token_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(inputs)
    cache = new_cache(model)
    generation = GenerationConfig.from_dict(config["generation_config"])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with OuroGuidance(model, GuidanceConfig(**settings), total_loops=4) as adapter:
        with observe_execution(model, cache, mode) as observation, torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=mask,
                                    past_key_values=cache, generation_config=generation,
                                    logits_to_keep=1)
        adapter_observation = adapter.last_observation
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    if output.shape[0] != 1 or not torch.equal(output[0, :inputs.shape[1]], inputs[0]):
        raise ValueError("Generation returned a different prompt or batch")
    ids = output[0, inputs.shape[1]:].tolist()
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    completion = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    row = {"schema_version": 1, **prompt, "sample_id": sample_id, "arm": mode,
           "config_hash": hash_json(config), "paired_config_hash": hash_json(paired_config(config)),
           "seed": seed, "generation_config_sha256": hash_json(config["generation_config"]),
           "generated_token_ids": ids, "generated_token_ids_sha256": hash_json(ids),
           "generated_tokens": len(ids), "raw_generation": raw, "raw_generation_sha256": hash_text(raw),
           "completion": completion, "completion_sha256": hash_text(completion),
           "effective_max_new_tokens": generation.max_new_tokens,
           **stop_metadata(ids, generation.max_new_tokens), "elapsed_seconds": elapsed,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
           "adapter_observation": adapter_observation, "execution_observation": observation}
    validate_sample(row, config, prompt)
    del output, cache, inputs, mask
    return row


def initialize_states(output, configs, prompts, resume, smoke_path, runtime_git):
    """Validate every arm before mutating manifests or launching a new sample."""
    states = {}
    for arm, config in configs.items():
        folder = output / arm
        manifest_file = folder / "manifest.json"
        config_hash = hash_json(config)
        previous = None
        if folder.exists():
            if not resume or not manifest_file.is_file():
                raise ValueError("Existing arm requires --resume and a valid manifest")
            previous = json.loads(manifest_file.read_text())
            if (previous.get("config_hash") != config_hash or previous.get("config") != config
                    or previous.get("paired_config_hash") != hash_json(paired_config(config))
                    or previous.get("kind") != "aime_generation"):
                raise ValueError("Resume manifest differs from frozen configuration")
            if previous.get("status") == "completed" and previous.get("samples_sha256") != sha256_file(folder / "samples.jsonl"):
                raise ValueError("Completed samples changed; refusing repair or overwrite")
        elif resume:
            raise ValueError("Resume requires all three previously initialized arms")
        states[arm] = {"folder": folder, "config": config, "previous": previous}
    for arm, state in states.items():
        folder, config, previous = state["folder"], state["config"], state["previous"]
        recovery = repair_partial_tail(folder / "samples.jsonl") if resume and previous["status"] != "completed" else None
        done = read_completed(folder / "samples.jsonl", config, prompts)
        expected = len(prompts) * config["samples_per_problem"]
        if len(done) > expected or (previous and previous["status"] == "completed" and len(done) != expected):
            raise ValueError("Resume completion count differs from complete sample set")
        manifest = {"schema_version": 1, "kind": "aime_generation", "status": "running",
                    "is_full_split": config["is_full_split"], "expected_samples": expected,
                    "completed_samples": len(done), "config": config, "config_hash": hash_json(config),
                    "paired_config_hash": hash_json(paired_config(config)),
                    "gpu_smoke_sha256": sha256_file(smoke_path), "source_git_observed": runtime_git,
                    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "updated_at": stamp()}
        if previous:
            manifest["created_at"] = previous.get("created_at")
            manifest["resume_history"] = previous.get("resume_history", []) + [{"at": stamp(), "completed_samples": len(done), "recovery": recovery}]
        else:
            manifest["created_at"] = stamp()
        if previous and previous["status"] == "completed":
            manifest = previous  # Do not rewrite a valid finished arm during resume.
        state.update(done=done, manifest=manifest)
    for state in states.values():
        state["folder"].mkdir(parents=True, exist_ok=resume)
        if state["manifest"]["status"] != "completed":
            atomic_json(state["folder"] / "manifest.json", state["manifest"])
    return states


def run(args):
    import torch
    from loopcd_repro.runtime import load_ouro, provenance
    protocol = load_protocol(args.protocol)
    identity = read_model_identity(args.model, protocol)
    questions = load_questions(args.data_root, args.year)
    debug = {"limit": args.debug_limit, "samples": args.debug_samples, "max_new_tokens": args.debug_max_new_tokens}
    full = all(value is None for value in debug.values())
    for key, value, maximum in (("limit", args.debug_limit, 30), ("samples", args.debug_samples, 16), ("max_new_tokens", args.debug_max_new_tokens, 8192)):
        if value is not None and (value < 1 or value > maximum):
            raise ValueError(f"Invalid debug {key}")
    if args.debug_limit is not None:
        questions = questions[:args.debug_limit]
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("AIME production generation requires a coordinated CUDA device")
    model, tokenizer = load_ouro(args.model, args.device)
    source = provenance(args.model, model)
    runtime_git = source.get("git_commit")
    source["precision"] = "BF16 model; FP32 guidance before native sampling"
    if source["loaded_model_code_sha256"] != identity["model_code_sha256"]:
        raise ValueError("Loaded model code differs from prepared private source")
    validate_smoke(args.smoke, smoke_bindings(identity, source))
    prompts = {question["task_id"]: build_prompt(tokenizer, question, protocol) for question in questions}
    generation = generation_config(protocol, args.debug_max_new_tokens)
    common = {"protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
              "dataset": {"year": args.year, "questions_sha256": QUESTIONS_SHA256[args.year],
                          "manifest_sha256": DATA_MANIFEST_SHA256, "task_ids": list(prompts)},
              "samples_per_problem": args.debug_samples or 16, "generation_config": generation.to_dict(),
              "prompt_policy": protocol["prompt"], "execution_policy": protocol["execution"],
              "model_identity": identity, "source": source, "is_full_split": full, "debug": None if full else debug}
    configs = {arm: {**common, "guidance": guidance_dict(arm, identity, protocol)} for arm in ARMS}
    states = initialize_states(args.output, configs, prompts, args.resume, args.smoke, runtime_git)
    # Re-decode saved tokens on resume, not only their self-reported text hashes.
    for state in states.values():
        for row in state["done"].values():
            for key, skip in (("raw_generation", False), ("completion", True)):
                if tokenizer.decode(row["generated_token_ids"], skip_special_tokens=skip, clean_up_tokenization_spaces=False) != row[key]:
                    raise ValueError("Resume token IDs and decoded text disagree")
    try:
        for task_id, prompt in prompts.items():
            for sample_id in range(common["samples_per_problem"]):
                for arm in ARMS:
                    state = states[arm]
                    key = (task_id, sample_id)
                    if key in state["done"]:
                        continue
                    row = generation_row(model, tokenizer, prompt, sample_id, state["config"], args.device)
                    samples = state["folder"] / "samples.jsonl"
                    with samples.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    state["done"][key] = row
                    state["manifest"].update(completed_samples=len(state["done"]), updated_at=stamp())
                    atomic_json(state["folder"] / "manifest.json", state["manifest"])
                    print(json.dumps({"task_id": task_id, "sample_id": sample_id, "arm": arm,
                                      "completed": len(state["done"]), "tokens": row["generated_tokens"],
                                      "stop": row["stop_reason"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        for state in states.values():
            if state["manifest"]["status"] == "completed":
                continue
            state["manifest"].update(status="completed", completed_samples=len(state["done"]),
                                     samples_sha256=sha256_file(state["folder"] / "samples.jsonl"),
                                     cap_hits=sum(row["cap_hit"] for row in state["done"].values()), updated_at=stamp())
            atomic_json(state["folder"] / "manifest.json", state["manifest"])
    except BaseException as exc:
        for state in states.values():
            if state["manifest"]["status"] != "completed":
                state["manifest"].update(status="failed", error=repr(exc), updated_at=stamp())
                atomic_json(state["folder"] / "manifest.json", state["manifest"])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-limit", type=int)
    parser.add_argument("--debug-samples", type=int)
    parser.add_argument("--debug-max-new-tokens", type=int)
    args = parser.parse_args()
    if args.output.is_symlink() or (args.output.exists() and not args.resume):
        raise FileExistsError("Use a fresh run directory or explicit validated --resume")
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError("Resume output does not exist")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with output_lock(args.output):
        run(args)


if __name__ == "__main__":
    main()
