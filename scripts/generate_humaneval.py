"""Paired native Huginn HumanEval(+) generation; never execute generated code."""
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
ARMS = {"baseline32": ("baseline", 32, 7), "hidden32": ("hidden", 32, 7),
        "baseline16": ("baseline", 16, 6), "hidden16": ("hidden", 16, 6)}


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


def read_completed(path, config_hash, tasks):
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        task_id = row["task_id"]
        if row["config_hash"] != config_hash or task_id not in tasks or task_id in completed:
            raise ValueError("Resume file has changed configuration, unknown task, or duplicate result")
        if row["problem_sha256"] != digest(tasks[task_id]):
            raise ValueError("Resume problem differs from the pinned data")
        completed[task_id] = row
    return completed


def repair_partial_tail(path):
    """Recover only an unterminated final record; preserve damaged bytes as evidence."""
    if not path.exists():
        return None
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return None
    boundary = content.rfind(b"\n") + 1
    tail = content[boundary:]
    try:
        json.loads(tail)
    except (ValueError, UnicodeDecodeError):
        backup = path.with_name(path.name + ".partial-" + hashlib.sha256(tail).hexdigest()[:16])
        backup.write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        return {"partial_tail_backup": str(backup), "removed_bytes": len(tail)}
    with path.open("ab") as stream:
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"completed_final_line_terminator": True}


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
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if len(set(args.arms)) != len(args.arms) or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        p.error("Arms must be unique and token/document budgets positive")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("HumanEvalPlus data does not match its pinned SHA256")
    problems = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(problems) != 164 or len({row["task_id"] for row in problems}) != 164:
        raise ValueError("Expected the complete, unique 164-task HumanEvalPlus dataset")
    problems.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    if args.limit is not None:
        problems = problems[:args.limit]
    tasks = {row["task_id"]: row for row in problems}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    from loopcd_repro.huginn import load_huginn, HuginnHiddenConfig, HuginnHiddenGuidance
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
    states = {}
    for arm in args.arms:
        mode, loops, reference = ARMS[arm]
        guidance = HuginnHiddenConfig(mode, loops, reference, 0.3)
        config = {"benchmark": "HumanEvalPlus-v0.1.10", "data_sha256": DATA_SHA256,
                  "guidance": asdict(guidance), "seed": args.seed,
                  "seed_rule": "(seed+int(sha256(task_id)[:8],16)) % 2**32; reset before every arm/problem",
                  "max_new_tokens": args.max_new_tokens, "native_context_length": 4096,
                  "limit": args.limit, "task_ids": list(tasks), "do_sample": False,
                  "eos_token_id": eos_ids, "pad_token_id": 65509,
                  "stops": STOPS, "instruction": INSTRUCTION, "response_prefix": RESPONSE,
                  "prompt_builder": "EvalPlus make_raw_chat_prompt; tokenize add_special_tokens=False",
                  "initialization": "native truncated Gaussian, init_scale=1; never zero",
                  "cache": "native full HuginnDynamicCache; fresh for every arm/problem",
                  "source": source}
        config_hash = digest(config)
        folder = args.output / arm
        folder.mkdir(exist_ok=args.resume)
        manifest_file = folder / "manifest.json"
        if manifest_file.exists():
            previous = json.loads(manifest_file.read_text())
            if previous["config_hash"] != config_hash:
                raise ValueError("Resume manifest does not match the current complete configuration")
        recovery = repair_partial_tail(folder / "samples.jsonl") if args.resume else None
        done = read_completed(folder / "samples.jsonl", config_hash, tasks)
        manifest = {"status": "running", "config": config, "config_hash": config_hash,
                    "is_full_split": args.limit is None, "expected_samples": len(tasks),
                    "completed_samples": len(done), "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        if recovery:
            manifest["resume_recovery"] = recovery
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
                with HuginnHiddenGuidance(model, guidance) as adapter, torch.inference_mode():
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
        for _, _, folder, done, manifest in states.values():
            manifest.update(status="completed", completed_samples=len(done),
                            cap_hits=sum(row["cap_hit"] for row in done.values()))
            atomic_json(folder / "manifest.json", manifest)
    except BaseException as exc:
        for _, _, folder, _, manifest in states.values():
            manifest.update(status="failed", error=repr(exc))
            atomic_json(folder / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
