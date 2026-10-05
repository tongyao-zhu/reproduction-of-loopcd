"""Score preregistered AIME integer text without executing any answer code.

The canonical gate checks all 60 pinned gold answers and fixed extractor
boundaries. Formal comparison requires three complete paired 30x16 runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


PROTOCOL_SHA256 = "1a908d00425016eb1bc3eb17b05267da19ab2d81b931abbbfadf0736ae08e380"
MANIFEST_SHA256 = "8c917d699556d66cd71fcd7f5a8ed13d79da8a6d213eaeef93798082a7c41d67"
EXTRACTOR_ID = "last-boxed-ascii-integer-v1"
DATA_PINS = {
    "aime2024.questions.jsonl": "48967ed2b5259955fb10f8c36b47ee93a86c3e850538d1209a415304bdd46e05",
    "aime2024.answers.jsonl": "d7a41e847d3ee8ab22897b6c6209b74638a3199a5f14d85c3b5b2d8675cacec0",
    "aime2025.questions.jsonl": "b5230a1ca58a5a6457b73f877cf24094496fe4703dfe9d37e15d9fb9684b51a9",
    "aime2025.answers.jsonl": "03f846b47b4d1d0aa37e27cfac4126a8cf968c0425cd107e87d53e5378b06dbe",
}
ARMS = ("baseline", "fixed", "adaptive")
# Observed on CPU in the existing Transformers 4.54.1 runtime before any AIME
# model generation: GenerationConfig(**registered_sampling, output*=False).
GENERATION_CONFIG_SHA256 = "d7492806bea88786419a34ad23baccae8dbdc4689fe041efaebd64f21e48b71e"
GENERATION_DEFAULTS = {
    "_from_model_config": False, "assistant_confidence_threshold": 0.4, "assistant_early_exit": None,
    "assistant_lookbehind": 10, "bad_words_ids": None, "begin_suppress_tokens": None,
    "cache_config": None, "constraints": None, "decoder_start_token_id": None, "disable_compile": False,
    "diversity_penalty": 0.0, "dola_layers": None, "early_stopping": False,
    "encoder_no_repeat_ngram_size": 0, "encoder_repetition_penalty": 1.0,
    "exponential_decay_length_penalty": None, "force_words_ids": None, "guidance_scale": None,
    "is_assistant": False, "length_penalty": 1.0, "low_memory": None, "max_length": 20,
    "max_matching_ngram_size": None, "max_time": None, "min_length": 0, "num_assistant_tokens": 20,
    "num_assistant_tokens_schedule": "constant", "num_beam_groups": 1,
    "output_attentions": False, "output_hidden_states": False, "output_logits": False, "output_scores": False,
    "penalty_alpha": None, "prefill_chunk_size": None, "prompt_lookup_num_tokens": None,
    "remove_invalid_values": False, "renormalize_logits": False, "return_dict_in_generate": False,
    "return_legacy_cache": None, "sequence_bias": None, "stop_strings": None, "suppress_tokens": None,
    "target_lookbehind": 10, "token_healing": False, "transformers_version": "4.54.1", "watermarking_config": None,
}


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot(path):
    raw = Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def json_snapshot(path):
    raw, sha = snapshot(path)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value, sha


def require(record, key):
    if not isinstance(record, dict) or key not in record:
        raise ValueError("Missing required evidence: " + key)
    return record[key]


def same(actual, expected, name):
    if hash_json(actual) != hash_json(expected):
        raise ValueError("Inconsistent " + name)


def valid_hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA256: " + name)
    return value


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid integer: " + name)
    return value


def task_ids(year):
    return [f"AIME{year}-{part}-{number:02d}" for part in ("I", "II") for number in range(1, 16)]


def sample_seed(task, sample):
    return int.from_bytes(hashlib.sha256(f"loopcd-aime-v1|42|{task}|{sample}".encode()).digest()[:4], "big")


def extract_answer(text):
    """The final literal \\boxed wins, including when malformed; no fallback."""
    if not isinstance(text, str):
        raise ValueError("Answer extraction requires generated text")
    start = text.rfind("\\boxed")
    result = {"parsed": False, "value": None, "error": None, "content": None,
              "command_offset": start if start >= 0 else None}
    if start < 0:
        result["error"] = "no_boxed"
        return result
    opening = start + len("\\boxed")
    while opening < len(text) and text[opening].isspace():
        opening += 1
    if opening >= len(text) or text[opening] != "{":
        result["error"] = "missing_open_brace"
        return result
    depth = 1
    closing = opening + 1
    while closing < len(text):
        if text[closing] == "{":
            depth += 1
        elif text[closing] == "}":
            depth -= 1
            if depth == 0:
                break
        closing += 1
    if depth:
        result["error"] = "unclosed_brace"
        return result
    content = text[opening + 1:closing].strip()
    result["content"] = content
    if re.fullmatch(r"[0-9]{1,3}", content) is None:
        result["error"] = "invalid_ascii_integer"
        return result
    result.update(parsed=True, value=int(content, 10))
    return result


def pass_at_k(correct, k, n=16):
    integer(n, "sample count", 1)
    integer(correct, "correct count", 0, n)
    integer(k, "k", 1, n)
    return 1.0 if n - correct < k else 1 - math.comb(n - correct, k) / math.comb(n, k)


def extractor_boundaries():
    # Fixed before inspecting any model answers. This is not a permissive math parser.
    cases = [
        (r"\boxed{025}", 25, None), (r"\boxed{0}", 0, None), (r"\boxed{999}", 999, None),
        ("\\boxed \t\n{ 007\n}", 7, None), (r"\boxed{1} then \boxed{002}", 2, None),
        (r"\boxed{1} then \boxed{bad}", None, "invalid_ascii_integer"),
        (r"\boxed{1} then \boxed", None, "missing_open_brace"),
        (r"\boxed{1} then \boxed{", None, "unclosed_brace"),
        ("The answer is 25", None, "no_boxed"), (r"\boxed{1000}", None, "invalid_ascii_integer"),
        (r"\boxed{-1}", None, "invalid_ascii_integer"), (r"\boxed{+1}", None, "invalid_ascii_integer"),
        (r"\boxed{1/2}", None, "invalid_ascii_integer"), (r"\boxed{1.0}", None, "invalid_ascii_integer"),
        (r"\boxed{336^\circ}", None, "invalid_ascii_integer"),
        (r"\boxed{\text{025}}", None, "invalid_ascii_integer"),
        (r"\boxed{{25}}", None, "invalid_ascii_integer"), (r"\boxed{1,2}", None, "invalid_ascii_integer"),
        (r"\boxed{１２}", None, "invalid_ascii_integer"), (r"\boxed{١٢}", None, "invalid_ascii_integer"),
        (r"\boxed{0000}", None, "invalid_ascii_integer"), (r"\boxed{}", None, "invalid_ascii_integer"),
        (r"\boxed{2+3}", None, "invalid_ascii_integer"),
        (r"\boxed{25} then truncated reasoning", 25, None),
        (r"\boxed{25} then \boxedness{2}", None, "missing_open_brace"),
    ]
    records = []
    for text, value, error in cases:
        parsed = extract_answer(text)
        same(parsed["value"], value, "extractor boundary value")
        same(parsed["error"], error, "extractor boundary error")
        same(parsed["parsed"], error is None, "extractor boundary status")
        records.append({"text": text, "expected_value": value, "expected_error": error, "result": parsed})
    return records


def load_scoring_data(protocol_path, data_root):
    protocol, protocol_sha = json_snapshot(protocol_path)
    same(protocol_sha, PROTOCOL_SHA256, "preregistered protocol SHA256")
    manifest, manifest_sha = json_snapshot(Path(data_root) / "manifest.json")
    same(manifest_sha, MANIFEST_SHA256, "registered data manifest")
    same(require(protocol, "data_manifest_sha256"), manifest_sha, "protocol/data manifest binding")
    same(require(manifest, "status"), "PASS", "data preparation status")
    questions, answers, files = {}, {}, {}
    for year in (2024, 2025):
        expected = task_ids(year)
        for kind, destination in (("questions", questions), ("answers", answers)):
            name = f"aime{year}.{kind}.jsonl"
            raw, sha = snapshot(Path(data_root) / name)
            same(sha, DATA_PINS[name], "pinned " + name)
            declaration = require(require(require(manifest, "datasets"), f"aime{year}"), "files")[kind]
            for key, value in {"file": name, "rows": 30, "bytes": len(raw), "sha256": sha}.items():
                same(require(declaration, key), value, "manifest " + name + " " + key)
            rows = [json.loads(line) for line in raw.splitlines()]
            same([require(row, "task_id") for row in rows], expected, "ordered unique " + kind)
            for row in rows:
                destination[row["task_id"]] = row
            files[name] = {"sha256": sha, "bytes": len(raw)}
    for task, question in questions.items():
        text = require(question, "question")
        if not isinstance(text, str) or not text:
            raise ValueError("Missing public question")
        same(require(question, "question_sha256"), text_hash(text), "question SHA256")
        if {"gold_int", "gold_raw", "solution", "answer"} & set(question):
            raise ValueError("Generation question file contains answer fields")
        answer = answers[task]
        same(require(answer, "question_sha256"), question["question_sha256"], "question/gold binding")
        integer(require(answer, "gold_int"), "gold", 0, 999)
        raw = require(answer, "gold_raw")
        if not isinstance(raw, str):
            raise ValueError("Gold source must be a string")
        changes = []
        if task == "AIME2025-II-05" and re.fullmatch(r"[0-9]{1,3}\^\\{1,2}circ", raw):
            raw = raw.split("^")[0]
            changes.append("remove_explicit_degree_suffix")
        if re.fullmatch(r"[0-9]{1,3}", raw) is None:
            raise ValueError("Noncanonical source gold")
        if len(raw) > 1 and raw[0] == "0":
            changes.append("remove_leading_zeros")
        same(answer["gold_int"], int(raw), "normalized source gold")
        same(require(answer, "normalization"), changes, "gold normalization event")
    expected_seeds = [{"task_id": task, "sample_id": sample, "seed": sample_seed(task, sample)}
                      for year in (2024, 2025) for task in task_ids(year) for sample in range(16)]
    same(require(require(protocol, "seed"), "records"), expected_seeds, "registered full seed table")
    return {"protocol": protocol, "protocol_sha256": protocol_sha, "manifest": manifest,
            "manifest_sha256": manifest_sha, "files": files, "questions": questions, "answers": answers,
            "data_root": str(Path(data_root).resolve())}


def canonical_gate(data):
    boundaries = extractor_boundaries()
    rows = []
    for task, answer in data["answers"].items():
        canonical = "\\boxed{" + str(answer["gold_int"]) + "}"
        parsed = extract_answer(canonical)
        same(parsed["value"], answer["gold_int"], "60-task canonical extraction")
        same(parsed["parsed"], True, "canonical parser status")
        rows.append({"task_id": task, "question_sha256": answer["question_sha256"],
                     "gold_sha256": hash_json(answer), "canonical": canonical, "parsed": parsed, "correct": True})
    same(len(rows), 60, "canonical full gold set")
    for correct, expected in ((0, (0.0, 0.0)), (1, (1 / 16, 10 / 16)), (7, (7 / 16, 1.0)), (16, (1.0, 1.0))):
        same([pass_at_k(correct, k) for k in (1, 10)], list(expected), "estimator boundary")
    return {"status": "PASS", "extractor_id": EXTRACTOR_ID, "canonical_tasks": 60, "all_correct": True,
            "scorer_source_sha256": snapshot(__file__)[1], "scorer_helpers": "Python standard library only",
            "protocol_sha256": data["protocol_sha256"], "data_manifest_sha256": data["manifest_sha256"],
            "data_files": data["files"], "rows": rows, "boundary_cases": boundaries,
            "boundary_cases_passed": len(boundaries), "model_answers_executed": False,
            "interpretation": "Parser/data self-check only; not a model benchmark result."}


def expected_prompt(question, protocol):
    policy = protocol["prompt"]
    messages = [{"role": "system", "content": policy["system"]},
                {"role": "user", "content": question["question"] + policy["joiner"] + policy["user_suffix"]}]
    rendered = "".join("<|im_start|>" + row["role"] + "\n" + row["content"] + "<|im_end|>\n" for row in messages)
    return messages, rendered + "<|im_start|>assistant\n<think>\n"


def validate_configuration(config, data):
    protocol = data["protocol"]
    for key, expected in {"protocol_id": protocol["protocol_id"], "protocol_sha256": data["protocol_sha256"],
                          "samples_per_problem": 16, "is_full_split": True,
                          "prompt_policy": protocol["prompt"], "execution_policy": protocol["execution"]}.items():
        same(require(config, key), expected, "generation config " + key)
    # A formal run cannot turn a debug budget into a full-score denominator.
    same(require(config, "debug"), None, "formal run debug settings")
    dataset = require(config, "dataset")
    year = integer(require(dataset, "year"), "year")
    if year not in (2024, 2025):
        raise ValueError("Unexpected AIME year")
    same(dataset, {"year": year, "questions_sha256": data["files"][f"aime{year}.questions.jsonl"]["sha256"],
                   "manifest_sha256": data["manifest_sha256"], "task_ids": task_ids(year)}, "generation data identity")
    identity = require(config, "model_identity")
    repo = require(identity, "repo_id")
    if repo not in protocol["models"]:
        raise ValueError("Expected a pinned Ouro-Thinking checkpoint")
    model_policy = protocol["models"][repo]
    same(require(identity, "revision"), model_policy["revision"], "Thinking checkpoint revision")
    for key in ("model_provenance_sha256", "model_code_sha256", "config_sha256"):
        valid_hash(require(identity, key), "model " + key)
    tokenizer = require(identity, "tokenizer_files_sha256")
    same(sorted(tokenizer), sorted(["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"]), "tokenizer file identities")
    for key, value in tokenizer.items():
        valid_hash(value, key)
    guidance = require(config, "guidance")
    mode = require(guidance, "mode")
    if mode not in ARMS:
        raise ValueError("Unknown AIME guidance arm")
    same(guidance, {"mode": mode, "omega": 0.5, "omega_cap": model_policy["adaptive_cap"],
                    "early_loop": 1, "total_loops": 4}, "paper R4/reference1/guidance configuration")
    generation = require(config, "generation_config")
    for key, expected in protocol["sampling"].items():
        same(require(generation, key), expected, "registered sampling " + key)
    same(hash_json(generation), GENERATION_CONFIG_SHA256, "complete explicit GenerationConfig including defaults")
    # Explicit defaults that can otherwise alter decoding despite matching T/p.
    neutral = {"bad_words_ids": None, "force_words_ids": None, "constraints": None,
               "suppress_tokens": None, "begin_suppress_tokens": None,
               "forced_decoder_ids": None, "sequence_bias": None,
               "stop_strings": None, "watermarking_config": None,
               "penalty_alpha": None, "dola_layers": None, "max_time": None,
               "exponential_decay_length_penalty": None, "guidance_scale": None,
               "encoder_repetition_penalty": 1.0, "encoder_no_repeat_ngram_size": 0,
               "num_beam_groups": 1, "diversity_penalty": 0.0,
               "remove_invalid_values": False, "renormalize_logits": False,
               "token_healing": False, "return_dict_in_generate": False}
    for key, expected in neutral.items():
        if key in generation:
            same(generation[key], expected, "neutral decoding " + key)
    source = require(config, "source")
    if not isinstance(require(source, "git_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", source["git_commit"]) is None:
        raise ValueError("Missing frozen generation Git commit")
    same(require(source, "loaded_model_code_sha256"), identity["model_code_sha256"], "loaded native model code")
    same(require(source, "attention"), "sdpa", "native attention")
    same(require(source, "precision"), "BF16 model; FP32 guidance before native sampling", "inference precision")
    for key in ("python", "cuda", "gpu", "precision"):
        if not isinstance(require(source, key), str) or not source[key]:
            raise ValueError("Missing frozen runtime " + key)
    packages = require(source, "packages")
    if not isinstance(packages, dict) or not {"torch", "transformers", "accelerate"} <= set(packages):
        raise ValueError("Incomplete runtime package provenance")
    for key, value in packages.items():
        if not isinstance(value, str) or not value:
            raise ValueError("Invalid package version " + key)
    model = require(source, "model")
    for key in ("repo_id", "revision", "model_code_sha256"):
        same(require(model, key), identity[key], "model/source " + key)
    sources = require(source, "source_sha256")
    if not isinstance(sources, dict) or not {"scripts/generate_aime.py", "src/loopcd_repro/aime_protocol.py",
        "src/loopcd_repro/ouro.py", "src/loopcd_repro/guidance.py", "src/loopcd_repro/runtime.py"} <= set(sources):
        raise ValueError("Missing generation/core source hashes")
    for key, value in sources.items():
        valid_hash(value, key)
    return year, mode


def validate_row(row, config, data, key):
    task, sample = key
    mode = config["guidance"]["mode"]
    common = {name: value for name, value in config.items() if name != "guidance"}
    for name, expected in {"schema_version": 1, "task_id": task, "sample_id": sample, "arm": mode,
                           "seed": sample_seed(task, sample), "config_hash": hash_json(config),
                           "paired_config_hash": hash_json(common), "question_sha256": data["questions"][task]["question_sha256"],
                           "generation_config_sha256": hash_json(config["generation_config"]), "effective_max_new_tokens": 8192}.items():
        same(require(row, name), expected, "sample " + name)
    messages, prompt = expected_prompt(data["questions"][task], data["protocol"])
    for name, expected in {"messages": messages, "messages_sha256": hash_json(messages),
                           "prompt": prompt, "prompt_sha256": text_hash(prompt)}.items():
        same(require(row, name), expected, "public-only prompt " + name)
    prompt_ids = require(row, "prompt_token_ids")
    generated = require(row, "generated_token_ids")
    for values in (prompt_ids, generated):
        if not isinstance(values, list) or not values or any(type(x) is not int or x < 0 for x in values):
            raise ValueError("Invalid token IDs")
    same(require(row, "prompt_token_ids_sha256"), hash_json(prompt_ids), "prompt token IDs hash")
    same(require(row, "generated_token_ids_sha256"), hash_json(generated), "output token IDs hash")
    same(require(row, "generated_tokens"), len(generated), "generated token count")
    if len(prompt_ids) + 8192 > 32768 or len(generated) > 8192:
        raise ValueError("Registered token budget exceeded")
    if 2 in generated[:-1]:
        raise ValueError("Generation continued after EOS")
    reason = "eos" if generated[-1] == 2 else "max_new_tokens" if len(generated) == 8192 else None
    if reason is None:
        raise ValueError("Generation stopped for an unregistered reason")
    same(require(row, "stop_reason"), reason, "stop reason")
    same(require(row, "cap_hit"), reason == "max_new_tokens", "cap hit")
    for name in ("raw_generation", "completion"):
        value = require(row, name)
        if not isinstance(value, str):
            raise ValueError("Missing generated-only text")
        same(require(row, name + "_sha256"), text_hash(value), "generated text " + name)
    elapsed = require(row, "elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid sample timing")
    allocated = integer(require(row, "peak_allocated_bytes"), "allocated bytes")
    reserved = integer(require(row, "peak_reserved_bytes"), "reserved bytes")
    if reserved < allocated:
        raise ValueError("Invalid memory observations")
    heads = 1 if mode == "baseline" else 2
    slots = 96 if config["model_identity"]["repo_id"] == "ByteDance/Ouro-1.4B-Thinking" else 192
    execution = require(row, "execution_observation")
    for name, value in {"forward_calls": len(generated), "loop_calls": len(generated) * 4,
                        "head_calls": len(generated) * heads, "expected_head_calls_per_forward": heads,
                        "observed_loop_pattern_valid": True, "cache_type": "UniversalTransformerCache",
                        "cache_slots": slots, "fresh_cache_initial_length": 0,
                        "final_cache_length": len(prompt_ids) + len(generated) - 1}.items():
        same(require(execution, name), value, "native execution " + name)
    adapter = require(row, "adapter_observation")
    for name, value in {"mode": mode, "guidance_applied": mode != "baseline", "native_exit_at_step": 3,
                        "extra_lm_head_calls": 0 if mode == "baseline" else 1}.items():
        same(require(adapter, name), value, "adapter " + name)
    if mode != "baseline":
        for name, value in {"executed_source_indices": [0, 1, 2, 3], "early_loop": 1, "already_normalized_identity": True}.items():
            same(require(adapter, name), value, "guided native trajectory " + name)


def read_generation(path, data):
    path = Path(path).resolve()
    manifest, manifest_sha = json_snapshot(path / "manifest.json")
    for name, expected in {"schema_version": 1, "kind": "aime_generation", "status": "completed",
                           "is_full_split": True, "expected_samples": 480, "completed_samples": 480}.items():
        same(require(manifest, name), expected, "full generation manifest " + name)
    if manifest.get("error"):
        raise ValueError("Generation reported infrastructure failure")
    valid_hash(require(manifest, "gpu_smoke_sha256"), "pre-generation GPU gate")
    config = require(manifest, "config")
    year, arm = validate_configuration(config, data)
    common = {key: value for key, value in config.items() if key != "guidance"}
    same(require(manifest, "config_hash"), hash_json(config), "manifest config hash")
    same(require(manifest, "paired_config_hash"), hash_json(common), "manifest paired config hash")
    raw, samples_sha = snapshot(path / "samples.jsonl")
    same(require(manifest, "samples_sha256"), samples_sha, "completed sample bytes hash")
    if not raw.endswith(b"\n"):
        raise ValueError("Generation has an unterminated final sample")
    rows = [json.loads(line) for line in raw.splitlines()]
    identities = [(task, sample) for task in task_ids(year) for sample in range(16)]
    same(len(rows), 480, "full sample count")
    for row, identity in zip(rows, identities):
        validate_row(row, config, data, identity)
    for index in range(0, 480, 16):
        for row in rows[index:index + 16]:
            same(row["prompt_token_ids"], rows[index]["prompt_token_ids"], "one deterministic prompt per problem")
    same(require(manifest, "cap_hits"), sum(row["cap_hit"] for row in rows), "manifest cap hits")
    return {"path": str(path), "manifest": manifest, "manifest_sha256": manifest_sha,
            "samples_sha256": samples_sha, "config": config, "common_config": common,
            "year": year, "arm": arm, "rows": rows}


def verify_paired_generations(generations):
    if len(generations) != 3:
        raise ValueError("Exactly three complete AIME generation arms are required")
    arms = {run["arm"]: run for run in generations}
    same(sorted(arms), sorted(ARMS), "unique three-arm set")
    baseline = arms["baseline"]
    for run in generations:
        same(run["common_config"], baseline["common_config"], "model/year/source/runtime/protocol except guidance")
        same(run["manifest"]["gpu_smoke_sha256"], baseline["manifest"]["gpu_smoke_sha256"], "paired pre-generation GPU gate")
        for first, row in zip(baseline["rows"], run["rows"]):
            for key in ("task_id", "sample_id", "seed", "question_sha256", "messages", "messages_sha256",
                        "prompt", "prompt_sha256", "prompt_token_ids", "prompt_token_ids_sha256",
                        "generation_config_sha256", "effective_max_new_tokens"):
                same(row[key], first[key], "paired sample " + key)
    return arms


def summarize_arm(rows, data, year):
    """Rows have already passed the complete generation identity validation."""
    expected = [(task, sample) for task in task_ids(year) for sample in range(16)]
    same([(row["task_id"], row["sample_id"]) for row in rows], expected, "full ordered scoring identities")
    scored, problems = [], []
    for row in rows:
        answer = data["answers"][row["task_id"]]
        parsed = extract_answer(row["completion"])
        scored.append({"task_id": row["task_id"], "sample_id": row["sample_id"], "seed": row["seed"],
                       "question_sha256": answer["question_sha256"], "gold_sha256": hash_json(answer),
                       "gold_int": answer["gold_int"], "completion_sha256": text_hash(row["completion"]),
                       "extraction": parsed, "correct": parsed["parsed"] and parsed["value"] == answer["gold_int"],
                       "cap_hit": row["cap_hit"], "generated_tokens": row["generated_tokens"]})
    for index, task in enumerate(task_ids(year)):
        records = scored[index * 16:(index + 1) * 16]
        correct = sum(row["correct"] for row in records)
        problems.append({"task_id": task, "n": 16, "c_i": correct,
                         "pass_at_1": pass_at_k(correct, 1), "pass_at_10": pass_at_k(correct, 10),
                         "parse_failures": sum(not row["extraction"]["parsed"] for row in records),
                         "cap_hits": sum(row["cap_hit"] for row in records)})
    return {"n_tasks": 30, "samples_per_problem": 16, "rows": scored, "problems": problems,
            "correct_samples": sum(row["correct"] for row in scored),
            "pass_at_1_percent": 100 * statistics.mean(row["pass_at_1"] for row in problems),
            "pass_at_10_percent": 100 * statistics.mean(row["pass_at_10"] for row in problems),
            "parse_failures": dict(Counter(row["extraction"]["error"] for row in scored if not row["extraction"]["parsed"])),
            "cap_hits": sum(row["cap_hit"] for row in rows),
            "cap_correct": sum(row["cap_hit"] and row["correct"] for row in scored),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "generated_tokens_total": sum(row["generated_tokens"] for row in rows),
            "elapsed_seconds_total": sum(row["elapsed_seconds"] for row in rows),
            "peak_allocated_bytes_max": max(row["peak_allocated_bytes"] for row in rows),
            "peak_reserved_bytes_max": max(row["peak_reserved_bytes"] for row in rows)}


def percentile(ordered, p):
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def compare_pair(baseline, candidate):
    same([x["task_id"] for x in baseline["problems"]], [x["task_id"] for x in candidate["problems"]], "paired problems")
    problems = []
    for before, after in zip(baseline["problems"], candidate["problems"]):
        problems.append({"task_id": before["task_id"], "baseline_c_i": before["c_i"], "candidate_c_i": after["c_i"],
                         "delta_pass_at_1": after["pass_at_1"] - before["pass_at_1"],
                         "delta_pass_at_10": after["pass_at_10"] - before["pass_at_10"]})
    sample_deltas = [int(after["correct"]) - int(before["correct"]) for before, after in zip(baseline["rows"], candidate["rows"])]
    rng = random.Random(42)
    bootstraps = {k: [] for k in (1, 10)}
    for _ in range(10000):
        indices = [rng.randrange(30) for _ in range(30)]
        for k in (1, 10):
            bootstraps[k].append(100 * sum(problems[index][f"delta_pass_at_{k}"] for index in indices) / 30)
    metrics = {}
    for k in (1, 10):
        values = sorted(bootstraps[k])
        metrics[f"pass_at_{k}"] = {"baseline_percent": baseline[f"pass_at_{k}_percent"],
            "candidate_percent": candidate[f"pass_at_{k}_percent"],
            "delta_percentage_points": 100 * statistics.mean(row[f"delta_pass_at_{k}"] for row in problems),
            "paired_problem_bootstrap_95_percentile": [percentile(values, .025), percentile(values, .975)]}
    return {"metrics": metrics, "problems": problems,
            "sample_wins": sum(x > 0 for x in sample_deltas), "sample_losses": sum(x < 0 for x in sample_deltas),
            "sample_ties": sum(x == 0 for x in sample_deltas),
            "bootstrap": {"replicates": 10000, "seed": 42, "unit": "problem", "samples_kept_together": 16,
                          "percentiles": [2.5, 97.5], "interpolation": "linear at (n-1)*p"}}


def score_aime(run_paths, protocol_path, data_root):
    data = load_scoring_data(protocol_path, data_root)
    gate = canonical_gate(data)
    arms = verify_paired_generations([read_generation(path, data) for path in run_paths])
    year = arms["baseline"]["year"]
    scored = {name: summarize_arm(run["rows"], data, year) for name, run in arms.items()}
    for name, record in scored.items():
        record.update(status="PASS", kind="aime_integer_text_scores", arm=name, year=year,
                      full_480_verified=True, samples_sha256=arms[name]["samples_sha256"],
                      generation_manifest_sha256=arms[name]["manifest_sha256"],
                      scorer_source_sha256=gate["scorer_source_sha256"],
                      protocol_sha256=data["protocol_sha256"], data_files=data["files"],
                      model_identity=arms[name]["config"]["model_identity"])
    summary = {"schema_version": 1, "kind": "aime_paired_text_scoring", "status": "PASS",
               "created_at": datetime.now(timezone.utc).isoformat(), "full_480_verified": True,
               "n_tasks": 30, "samples_per_problem": 16, "rows_per_arm": 480,
               "year": year, "model_identity": arms["baseline"]["config"]["model_identity"],
               "protocol_id": data["protocol"]["protocol_id"], "protocol_sha256": data["protocol_sha256"],
               "data_manifest_sha256": data["manifest_sha256"], "data_files": data["files"],
               "gpu_smoke_sha256": arms["baseline"]["manifest"]["gpu_smoke_sha256"],
               "extractor_id": EXTRACTOR_ID, "canonical_gate_sha256": hash_json(gate),
               "scorer_source_sha256": gate["scorer_source_sha256"], "scorer_helpers": gate["scorer_helpers"],
               "common_config": arms["baseline"]["common_config"], "paired_config_hash": hash_json(arms["baseline"]["common_config"]),
               "arms": {name: {"run_path": run["path"], "manifest_sha256": run["manifest_sha256"],
                               "samples_sha256": run["samples_sha256"], "guidance": run["config"]["guidance"],
                               **{key: value for key, value in scored[name].items() if key != "rows"}}
                        for name, run in arms.items()},
               "pairs": {name: compare_pair(scored["baseline"], scored[name]) for name in ("fixed", "adaptive")},
               "validation": {"all60_gold_canonical_gate": True, "complete_ordered_480_per_arm": True,
                              "source_runtime_config_paired": True, "public_prompts_reconstructed": True,
                              "token_ids_seed_question_hashes_paired": True, "native_r4_cache_observations_verified": True,
                              "gold_question_file_hashes_verified": True, "model_answers_executed": False},
               "limitations": ["This is the preregistered reconstruction protocol, not a claim that Apple supplied every generation setting.",
                                "Text/token hashes and pairing are verified without model libraries; this offline scorer does not re-tokenize prompts or decode output IDs.",
                                "The paired bootstrap resamples 30 problems with all 16 samples kept together. It does not measure seed or protocol uncertainty."]}
    return {"summary": summary, "canonical_gate": gate, "scores": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, help="Complete baseline, fixed and adaptive arm directories")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; never overwrites")
    parser.add_argument("--canonical-only", action="store_true", help="Check all gold and fixed boundaries without any model samples")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh evidence directory")
    if bool(args.runs) == args.canonical_only:
        parser.error("Choose either --runs with three directories or --canonical-only")
    try:
        if args.canonical_only:
            gate = canonical_gate(load_scoring_data(args.protocol, args.data_root))
            result = {"canonical_gate": gate}
        else:
            result = score_aime(args.runs, args.protocol, args.data_root)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    gate_path = args.output / "canonical_gate.json"
    gate_path.write_text(json.dumps(result["canonical_gate"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for arm, scores in result.get("scores", {}).items():
        path = args.output / (arm + ".json")
        path.write_text(json.dumps(scores, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        result["summary"]["arms"][arm]["score_file_sha256"] = snapshot(path)[1]
    if "summary" in result:
        # Publish the complete summary last, after every bound evidence file.
        result["summary"]["canonical_gate_file_sha256"] = snapshot(gate_path)[1]
        (args.output / "summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "full_480_verified": not args.canonical_only,
                      "canonical_tasks": result["canonical_gate"]["canonical_tasks"]}))


if __name__ == "__main__":
    main()
