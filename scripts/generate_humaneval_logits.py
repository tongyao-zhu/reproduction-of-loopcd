"""Figure 1b R16 paired HumanEval generation after a bound real-checkpoint gate."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
STOPS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n```\n"]
ARMS = {"baseline16": ("baseline", 16, 1), "adaptive16": ("adaptive", 16, 1)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def task_seed(task_id, base_seed):
    return (base_seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)) % (2**32)


def trim_stops(text):
    found = [(text.find(stop), stop) for stop in STOPS if stop in text]
    if not found:
        return text, None
    position, stop = min(found, key=lambda pair: pair[0])
    return text[:position], stop


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="Debug subset only; never a full benchmark")
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    p.add_argument("--smoke", type=Path, help="Completed same-release two-task pair required for full generation")
    args = p.parse_args()
    from humaneval_logits_checks import validate_gate, audit_prompts, validate_rows, validate_pair, validate_model_files
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    if args.arms != list(ARMS) or args.max_new_tokens != 2048 or args.seed != 42 or args.limit not in (None, 2):
        p.error("Figure 1b protocol is fixed: two arms, seed42, 2048 tokens, full164 or explicit smoke2")
    if (args.limit is None) != (args.smoke is not None):
        p.error("Full run requires --smoke; smoke2 must not reuse another smoke")
    if args.device != "cuda:0" or os.getenv("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("Physical GPU binding mismatch")
    source_pins = frozen_tree(ROOT, args.source_commit)
    gate = json.loads(args.gate.read_text())
    validate_gate(gate, source_pins)
    model_pins = validate_model_files(args.model, gate["provenance"]["model"])
    state = gpu_status(args.gpu)
    if not state["ready"]:
        raise RuntimeError("GPU is occupied: " + repr(state))
    if len(set(args.arms)) != len(args.arms) or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        p.error("Arms must be unique and token/document budgets positive")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("HumanEvalPlus data does not match its pinned SHA256")
    problems = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(problems) != 164 or len({row["task_id"] for row in problems}) != 164:
        raise ValueError("Expected the complete, unique 164-task HumanEvalPlus dataset")
    problems.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    all_problems = list(problems)
    if args.limit is not None:
        problems = problems[:args.limit]
    tasks = {row["task_id"]: row for row in problems}
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    from loopcd_repro.huginn import load_huginn
    from loopcd_repro.huginn_logits import HuginnLogitsConfig, HuginnLogitsGuidance
    from loopcd_repro.runtime import provenance, sha256

    torch.set_num_threads(4)
    model, tokenizer = load_huginn(args.model, args.device)
    source = provenance(args.model, model)
    source["evalplus_version"] = importlib.metadata.version("evalplus")
    source["evalplus_prompt_sha256"] = sha256(inspect.getfile(make_raw_chat_prompt))
    source["evalplus_sanitize_sha256"] = sha256(inspect.getfile(sanitize))
    if model.config.block_size != 4096:
        raise ValueError("The pinned native context capacity changed")
    eos_ids = [65505, 65508]
    if model.generation_config.eos_token_id != eos_ids:
        raise ValueError("The checkpoint EOS configuration changed")
    validate_gate(gate, source_pins, source)
    audit = audit_prompts(all_problems, tokenizer, make_raw_chat_prompt)
    atomic_json(args.output / "prompt_audit.json", audit)
    binding = {"gate_sha256": sha256(args.gate), "gate_source_commit": gate["source_commit"],
               "generation_source_commit": args.source_commit,
               "unchanged_gate_source_sha256": gate["source_sha256"], "model_files_sha256": model_pins,
               "all_164_prompt_audit_sha256": digest(audit)}
    atomic_json(args.output / "gate_binding.json", binding)
    smoke_pair = validate_pair(args.smoke, all_problems, tokenizer, make_raw_chat_prompt, sanitize) if args.smoke else None
    states = {}
    for arm in args.arms:
        mode, loops, reference = ARMS[arm]
        guidance = HuginnLogitsConfig(mode, loops, reference, 0.2, 0.25)
        config = {"benchmark": "HumanEvalPlus-v0.1.10", "data_sha256": DATA_SHA256,
                  "guidance": asdict(guidance), "seed": args.seed,
                  "seed_rule": "(seed+int(sha256(task_id)[:8],16)) % 2**32; reset before every arm/problem",
                  "max_new_tokens": args.max_new_tokens, "native_context_length": 4096,
                  "limit": args.limit, "task_ids": list(tasks), "do_sample": False,
                  "eos_token_id": eos_ids, "pad_token_id": 65509,
                  "stops": STOPS, "instruction": INSTRUCTION, "response_prefix": RESPONSE,
                  "prompt_builder": "EvalPlus make_raw_chat_prompt; tokenize add_special_tokens=False",
                  "initialization": "native truncated Gaussian, init_scale=1; never zero",
                  "cache": "native full HuginnDynamicCache; separate reference coda history; fresh for every arm/problem",
                  "gate_binding": binding,
                  "source": source}
        config_hash = digest(config)
        folder = args.output / arm
        folder.mkdir()
        if smoke_pair:
            previous = dict(smoke_pair[arm]["config"])
            previous.update(limit=None, task_ids=list(tasks))
            if previous != config:
                raise ValueError("Smoke/full configuration differs beyond task subset")
        done = {}
        manifest_file = folder / "manifest.json"
        manifest = {"status": "running", "config": config, "config_hash": config_hash,
                    "is_full_split": args.limit is None, "expected_samples": len(tasks),
                    "completed_samples": len(done), "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(manifest_file, manifest)
        states[arm] = (guidance, config_hash, folder, done, manifest)

    class StopOnText(StoppingCriteria):
        def __init__(self, prompt_length):
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
            return any(stop in text for stop in STOPS)

    try:
        for problem in problems:
            task_id = problem["task_id"]
            prompt = make_raw_chat_prompt(problem["prompt"].strip() + "\n", INSTRUCTION, RESPONSE, tokenizer)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            cap = min(args.max_new_tokens, 4096 - len(token_ids))
            if cap < 1:
                raise ValueError(f"Prompt exceeds native context capacity: {task_id}")
            inputs = torch.tensor([token_ids], dtype=torch.long, device=args.device)
            seed = task_seed(task_id, args.seed)
            for arm, (guidance, config_hash, folder, done, manifest) in states.items():
                if task_id in done:
                    if done[task_id]["prompt_token_ids"] != token_ids or done[task_id]["seed"] != seed:
                        raise ValueError("Resume prompt or seed changed")
                    continue
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generation = GenerationConfig(
                    do_sample=False, max_new_tokens=cap, num_beams=1, num_return_sequences=1,
                    eos_token_id=eos_ids, bos_token_id=65504, pad_token_id=65509,
                    temperature=None, top_p=None, top_k=None, use_cache=True,
                    return_dict_in_generate=False,
                )
                started = time.monotonic()
                with HuginnLogitsGuidance(model, guidance) as adapter, torch.inference_mode():
                    output = model.generate(
                        input_ids=inputs, generation_config=generation, num_steps=guidance.total_loops,
                        past_key_values=adapter.new_cache(),
                        stopping_criteria=StoppingCriteriaList([StopOnText(len(token_ids))]),
                    )
                    observation = adapter.last_observation
                torch.cuda.synchronize()
                generated_ids = output[0, len(token_ids):].tolist()
                raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
                decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
                completion, stop = trim_stops(decoded)
                eos = bool(generated_ids and generated_ids[-1] in eos_ids)
                reason = "stop_string" if stop else "eos_token" if eos else "token_cap"
                if reason == "token_cap" and len(generated_ids) != cap:
                    raise ValueError("Generation stopped for an unrecorded reason")
                row = {"task_id": task_id, "config_hash": config_hash,
                       "problem_sha256": digest(problem), "prompt": prompt,
                       "prompt_token_ids": token_ids, "prompt_sha256": digest(prompt),
                       "seed": seed, "raw_generation": raw, "completion": completion,
                       "solution": sanitize(completion, entrypoint=problem["entry_point"]),
                       "generated_token_ids": generated_ids, "generated_tokens": len(generated_ids),
                       "effective_max_new_tokens": cap, "cap_hit": reason == "token_cap",
                       "stop_reason": reason, "stop_string": stop,
                       "elapsed_seconds": time.monotonic() - started, "adapter_observation": observation}
                validate_rows([row], manifest["config"], all_problems, tokenizer, make_raw_chat_prompt, sanitize, complete=False)
                with (folder / "samples.jsonl").open("a") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                done[task_id] = row
                manifest.update(completed_samples=len(done), updated_at=datetime.now(timezone.utc).isoformat())
                atomic_json(folder / "manifest.json", manifest)
                print(json.dumps({"task_id": task_id, "arm": arm, "completed": len(done),
                                  "tokens": len(generated_ids), "stop": reason,
                                  "seconds": row["elapsed_seconds"]}), flush=True)
        if frozen_tree(ROOT, args.source_commit) != source_pins:
            raise ValueError("Source changed during generation")
        for _, _, folder, done, manifest in states.values():
            validate_rows(list(done.values()), manifest["config"], all_problems, tokenizer, make_raw_chat_prompt, sanitize)
            manifest.update(status="completed", completed_samples=len(done),
                            cap_hits=sum(row["cap_hit"] for row in done.values()))
            atomic_json(folder / "manifest.json", manifest)
        checked = validate_pair(args.output, all_problems, tokenizer, make_raw_chat_prompt, sanitize)
        atomic_json(args.output / "pair_validation.json", {"status": "PASS", "n": len(tasks),
                    "full_164": args.limit is None, "arms": {arm: {"samples_sha256": sha256(args.output / arm / "samples.jsonl"),
                    "manifest_sha256": sha256(args.output / arm / "manifest.json")} for arm in checked}})
    except BaseException as exc:
        for _, _, folder, _, manifest in states.values():
            manifest.update(status="failed", error=repr(exc))
            atomic_json(folder / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
