"""Standard-library-only identities, records, and gates for registered AIME v1."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

PROTOCOL_ID = "ouro-thinking-aime-reconstruction-v1"
PROTOCOL_SHA256 = "1a908d00425016eb1bc3eb17b05267da19ab2d81b931abbbfadf0736ae08e380"
GENERATION_CONFIG_SHA256 = "d7492806bea88786419a34ad23baccae8dbdc4689fe041efaebd64f21e48b71e"
DATA_MANIFEST_SHA256 = "8c917d699556d66cd71fcd7f5a8ed13d79da8a6d213eaeef93798082a7c41d67"
QUESTIONS_SHA256 = {
    2024: "48967ed2b5259955fb10f8c36b47ee93a86c3e850538d1209a415304bdd46e05",
    2025: "b5230a1ca58a5a6457b73f877cf24094496fe4703dfe9d37e15d9fb9684b51a9",
}
ARMS = ("baseline", "fixed", "adaptive")
REQUIRED_SMOKE_CHECKS = {
    "protocol_and_identity", "all60_prompts", "native_baseline_seed_parity",
    "native_fixed_zero_seed_parity", "native_adaptive_zero_seed_parity",
    "fixed_formula_all_positions", "adaptive_formula_all_positions",
    "fixed_cache_oracle", "adaptive_cache_oracle", "eos_stop", "cap_stop",
    "native_long_cache_capacity",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def hash_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_protocol(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Protocol differs from the preregistered immutable v1 file")
    result = json.loads(raw)
    if result["protocol_id"] != PROTOCOL_ID or result["data_manifest_sha256"] != DATA_MANIFEST_SHA256:
        raise ValueError("Unexpected protocol identity")
    return result


def expected_task_ids(year):
    if type(year) is not int or year not in QUESTIONS_SHA256:
        raise ValueError("AIME year must be 2024 or 2025")
    return [f"AIME{year}-{part}-{number:02d}" for part in ("I", "II") for number in range(1, 16)]


def load_questions(data_root, year):
    """Read only the question file; gold-containing manifest is only byte-hashed."""
    ids = expected_task_ids(year)
    root = Path(data_root)
    if sha256_file(root / "manifest.json") != DATA_MANIFEST_SHA256:
        raise ValueError("AIME prepared manifest changed")
    raw = (root / f"aime{year}.questions.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUESTIONS_SHA256[year]:
        raise ValueError("AIME questions changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if [row["task_id"] for row in rows] != ids:
        raise ValueError("AIME question set/order changed")
    for row in rows:
        if set(row) != {"task_id", "dataset", "year", "part", "problem_number", "question", "question_sha256", "source"}:
            raise ValueError("Question file has unexpected or gold-bearing fields")
        if row["year"] != year or row["dataset"] != f"aime{year}" or not row["question"]:
            raise ValueError("Invalid question identity")
        if hash_text(row["question"]) != row["question_sha256"]:
            raise ValueError("Question hash mismatch")
    return rows


def sample_seed(task_id, sample_id):
    if type(sample_id) is not int or sample_id not in range(16):
        raise ValueError("sample_id must be 0..15")
    if task_id not in expected_task_ids(2024) + expected_task_ids(2025):
        raise ValueError("Unknown AIME task ID")
    return int.from_bytes(hashlib.sha256(f"loopcd-aime-v1|42|{task_id}|{sample_id}".encode()).digest()[:4], "big")


def build_prompt(tokenizer, question, protocol):
    policy = protocol["prompt"]
    messages = [{"role": "system", "content": policy["system"]},
                {"role": "user", "content": question["question"] + policy["joiner"] + policy["user_suffix"]}]
    text = tokenizer.apply_chat_template(messages, **policy["apply_chat_template"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Invalid tokenizer result")
    if not text.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("Pinned Thinking chat template did not enable thinking")
    # The native template itself uses BOS-like chat delimiters. Verify exact
    # tokenize=False/encode(False) equivalence, not a misleading BOS count.
    tokenized = tokenizer.apply_chat_template(messages, **{**policy["apply_chat_template"], "tokenize": True})
    if ids != tokenized:
        raise ValueError("Chat rendering introduced an extra special token")
    if len(ids) + protocol["sampling"]["max_new_tokens"] > protocol["execution"]["max_prompt_plus_output_budget"]:
        raise ValueError("Full prompt plus fixed output budget exceeds registered context budget")
    return {"task_id": question["task_id"], "question_sha256": question["question_sha256"],
            "messages": messages, "messages_sha256": hash_json(messages), "prompt": text, "prompt_sha256": hash_text(text),
            "prompt_token_ids": ids, "prompt_token_ids_sha256": hash_json(ids)}


def guidance_dict(mode, model_identity, protocol):
    if mode not in ARMS:
        raise ValueError("Unknown AIME guidance arm")
    return {"mode": mode, "omega": 0.5,
            "omega_cap": protocol["models"][model_identity["repo_id"]]["adaptive_cap"],
            "early_loop": 1, "total_loops": 4}


def read_model_identity(model_path, protocol):
    """Validate the prepared private snapshot without importing torch or a model."""
    path = Path(model_path)
    record = json.loads((path / "model_provenance.json").read_text())
    repo = record["repo_id"]
    if repo not in protocol["models"] or record["revision"] != protocol["models"][repo]["revision"]:
        raise ValueError("Model is not a registered Thinking checkpoint")
    pins = {
        "ByteDance/Ouro-1.4B-Thinking": (24, 2869336434, "44e42ec2ed97f49f20f87985688b6e1be9dd61bbfe9f8a349daac192126e2c55", "cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30"),
        "ByteDance/Ouro-2.6B-Thinking": (48, 5336011242, "ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a", "263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b"),
    }
    layers, weight_bytes, config_sha, native_sha = pins[repo]
    if sha256_file(path / "config.json") != config_sha or record["files"]["modeling_ouro.py"]["sha256_original"] != native_sha:
        raise ValueError("Prepared model config/native source identity changed")
    if (path / "model.safetensors").stat().st_size != weight_bytes:
        raise ValueError("Thinking weights missing or incomplete")
    code = sha256_file(path / "modeling_ouro.py")
    if code != record["model_code_sha256"]:
        raise ValueError("Prepared model source changed")
    configuration_sha = ("950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb" if layers == 24
                         else "ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6")
    if sha256_file(path / "configuration_ouro.py") != configuration_sha:
        raise ValueError("Prepared model configuration code changed")
    tokenizer_files = {}
    for name in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        actual = sha256_file(path / name)
        if actual != record["files"][name]["sha256_original"]:
            raise ValueError("Prepared tokenizer changed")
        tokenizer_files[name] = actual
    return {"repo_id": repo, "revision": record["revision"],
            "model_provenance_sha256": sha256_file(path / "model_provenance.json"),
            "model_code_sha256": code, "config_sha256": config_sha,
            "tokenizer_files_sha256": tokenizer_files}


def paired_config(config):
    return {key: value for key, value in config.items() if key != "guidance"}


def validate_generation_config(config, cap=8192):
    if type(cap) is not int or not 1 <= cap <= 8192 or config.get("max_new_tokens") != cap:
        raise ValueError("Unregistered generation token budget")
    if hash_json({**config, "max_new_tokens": 8192}) != GENERATION_CONFIG_SHA256:
        raise ValueError("Full GenerationConfig defaults differ from the verified Transformers 4.54.1 configuration")


def smoke_bindings(identity, source):
    return {"protocol_sha256": PROTOCOL_SHA256, "data_manifest_sha256": DATA_MANIFEST_SHA256,
            "questions_sha256": {str(year): value for year, value in QUESTIONS_SHA256.items()},
            "model_identity": identity, "source_sha256": source["source_sha256"],
            "git_commit": source.get("git_commit"),
            "loaded_model_code_sha256": source["loaded_model_code_sha256"],
            "runtime": {key: source[key] for key in ("python", "packages", "cuda", "gpu", "precision", "attention")}}


def validate_smoke(path, expected_bindings=None):
    report = json.loads(Path(path).read_text())
    if report.get("schema_version") != 1 or report.get("kind") != "aime_gpu_smoke" or report.get("status") != "PASS":
        raise ValueError("AIME GPU smoke is not complete and passing")
    checks = report.get("checks", [])
    names = [row.get("name") for row in checks]
    if not checks or len(set(names)) != len(names) or not REQUIRED_SMOKE_CHECKS <= set(names) or any(row.get("passed") is not True for row in checks):
        raise ValueError("Missing or failed AIME GPU smoke checks")
    bindings = report.get("bindings", {})
    if bindings.get("protocol_sha256") != PROTOCOL_SHA256 or bindings.get("data_manifest_sha256") != DATA_MANIFEST_SHA256:
        raise ValueError("Smoke protocol/data mismatch")
    if bindings.get("questions_sha256") != {str(k): v for k, v in QUESTIONS_SHA256.items()}:
        raise ValueError("Smoke question identity mismatch")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("Smoke model/runtime/source differs from this generation process")
    identity = bindings.get("model_identity", {})
    repo = identity.get("repo_id")
    models = {"ByteDance/Ouro-1.4B-Thinking": ("3aaa2224253a92ca45cf2e3d427c360e1ef9c93d", 96),
              "ByteDance/Ouro-2.6B-Thinking": ("f1edd81e7ac41355db670500ceaf204e0f73af68", 192)}
    if (repo not in models or identity.get("revision") != models[repo][0]
            or not bindings.get("source_sha256") or not bindings.get("runtime")
            or not identity.get("model_code_sha256")
            or bindings.get("loaded_model_code_sha256") != identity["model_code_sha256"]):
        raise ValueError("Incomplete or inconsistent smoke model/source binding")
    if smoke_bindings(identity, report.get("source", {})) != bindings:
        raise ValueError("Smoke source and derived runtime bindings disagree")
    audit = report.get("prompt_audit", [])
    if ([row.get("task_id") for row in audit] != expected_task_ids(2024) + expected_task_ids(2025)
            or any(type(row.get("tokens")) is not int or row["tokens"] < 1
                   or not row.get("prompt_sha256") or not row.get("prompt_token_ids_sha256") for row in audit)):
        raise ValueError("Smoke is missing the complete native prompt audit")
    fixture = report.get("fixture", {})
    if (not isinstance(fixture.get("prompt_token_ids"), list) or not fixture["prompt_token_ids"]
            or fixture.get("prompt_token_ids_sha256") != hash_json(fixture["prompt_token_ids"])
            or fixture.get("prompt_sha256") != hash_text(fixture.get("prompt", ""))):
        raise ValueError("Short generation fixture identity is incomplete")
    short = report.get("short_generation", {})
    if set(short) != {"native", "baseline", "fixed_zero", "adaptive_zero"}:
        raise ValueError("Missing native/baseline/zero sampled trajectories")
    for row in short.values():
        validate_generation_config(row.get("generation_config", {}), 32)
        ids = row.get("generated_token_ids")
        if not isinstance(ids, list) or row.get("forced_fixture_token") is not None or row.get("seed") != 42:
            raise ValueError("Short parity fixture is not unforced same-seed sampling")
        if any(row.get(key) != value for key, value in stop_metadata(ids, 32).items()):
            raise ValueError("Short sampled stopping metadata mismatch")
        validate_execution(row.get("execution_observation", {}), "baseline", len(fixture["prompt_token_ids"]), len(ids), models[repo][1])
        if ids != short["native"]["generated_token_ids"]:
            raise ValueError("Native/baseline/zero sampled tokens differ")
    by_name = {row["name"]: row for row in checks}
    for name, forced_eos in (("eos_stop", True), ("cap_stop", False)):
        row = by_name[name].get("fixture", {})
        validate_generation_config(row.get("generation_config", {}), 3)
        ids = row.get("generated_token_ids")
        if not isinstance(ids, list) or row.get("seed") != 42:
            raise ValueError("Missing controlled stopping fixture")
        forced = row.get("forced_fixture_token")
        if (forced_eos and (forced != 2 or ids != [2])) or (not forced_eos and (type(forced) is not int or forced == 2 or ids != [forced] * 3)):
            raise ValueError("Controlled EOS/cap fixture failed")
        if any(row.get(key) != value for key, value in stop_metadata(ids, 3).items()):
            raise ValueError("Controlled stopping fixture metadata mismatch")
        validate_execution(row.get("execution_observation", {}), "baseline", len(fixture["prompt_token_ids"]), len(ids), models[repo][1])
    gate = report.get("resource_gate", {})
    maximum = gate.get("max_prompt_tokens_all60", 0)
    if (gate.get("status") != "PASS" or gate.get("scope") != "native_cache_prefill_plus_one_cached_decode"
            or gate.get("max_new_tokens") != 8192 or type(maximum) is not int or maximum < 1
            or gate.get("target_prefill_tokens") != maximum + 8192
            or gate.get("actual_prefill_tokens") != maximum + 8192
            or gate.get("final_cache_length") != maximum + 8193
            or gate.get("measured_autoregressive_tokens") != 1
            or maximum != max(row["tokens"] for row in audit)
            or gate.get("cache_slots") != models[repo][1]
            or not isinstance(gate.get("peak_allocated_bytes"), int) or gate["peak_allocated_bytes"] <= 0
            or not isinstance(gate.get("peak_reserved_bytes"), int) or gate["peak_reserved_bytes"] < gate["peak_allocated_bytes"]):
        raise ValueError("Missing full-budget native-cache capacity evidence")
    target = gate["target_prefill_tokens"]
    chunk = gate.get("prefill_chunk_tokens")
    if chunk != 512:
        raise ValueError("Unexpected long-cache prefill schedule")
    for key, calls, heads, initial, final in (
        ("prefill_execution_observation", (target + chunk - 1) // chunk, 1, 0, target),
        ("decode_execution_observation", 1, 2, target, target + 1),
    ):
        expected = {"forward_calls": calls, "loop_calls": calls * 4, "head_calls": calls * heads,
                    "observed_loop_pattern_valid": True, "expected_head_calls_per_forward": heads,
                    "cache_type": "UniversalTransformerCache", "cache_slots": models[repo][1],
                    "fresh_cache_initial_length": initial, "final_cache_length": final}
        if any(gate.get(key, {}).get(field) != value for field, value in expected.items()):
            raise ValueError("Missing full-budget recurrence/head/cache execution counts")
    final_guidance = gate.get("final_guidance_observation", {})
    if (final_guidance.get("mode") != "adaptive" or final_guidance.get("guidance_applied") is not True
            or final_guidance.get("native_exit_at_step") != 3 or final_guidance.get("early_loop") != 1
            or final_guidance.get("executed_source_indices") != [0, 1, 2, 3]
            or final_guidance.get("extra_lm_head_calls") != 1
            or final_guidance.get("already_normalized_identity") is not True):
        raise ValueError("Long-cache decode did not observe registered adaptive guidance")
    return report


def stop_metadata(ids, cap, eos=2):
    if not ids or any(type(token) is not int or token < 0 for token in ids) or len(ids) > cap:
        raise ValueError("Invalid generated token sequence")
    if eos in ids[:-1]:
        raise ValueError("Generation continued after EOS")
    if ids[-1] == eos:
        return {"stop_reason": "eos", "cap_hit": False}
    if len(ids) == cap:
        return {"stop_reason": "max_new_tokens", "cap_hit": True}
    raise ValueError("Generation stopped for an unregistered reason")


def validate_execution(observation, mode, prompt_length, generated_tokens, cache_slots):
    calls = observation.get("forward_calls")
    heads = 1 if mode == "baseline" else 2
    expected = {"forward_calls": generated_tokens, "loop_calls": generated_tokens * 4,
                "head_calls": generated_tokens * heads, "observed_loop_pattern_valid": True,
                "expected_head_calls_per_forward": heads, "cache_type": "UniversalTransformerCache",
                "cache_slots": cache_slots, "fresh_cache_initial_length": 0,
                "final_cache_length": prompt_length + generated_tokens - 1}
    if type(calls) is not int or any(observation.get(key) != value for key, value in expected.items()):
        raise ValueError("Native recurrence/head/cache observations violate registered execution")


def validate_sample(row, config, prompt):
    mode = config["guidance"]["mode"]
    if row.get("schema_version") != 1 or row.get("arm") != mode:
        raise ValueError("Sample schema or arm mismatch")
    if row.get("config_hash") != hash_json(config) or row.get("paired_config_hash") != hash_json(paired_config(config)):
        raise ValueError("Sample configuration hash mismatch")
    sample = row.get("sample_id")
    if type(sample) is not int or sample not in range(config["samples_per_problem"]):
        raise ValueError("Unexpected sample index")
    for key, value in prompt.items():
        if row.get(key) != value:
            raise ValueError(f"Sample prompt identity changed: {key}")
    if row.get("seed") != sample_seed(row["task_id"], sample):
        raise ValueError("Sample seed mismatch")
    if row.get("generation_config_sha256") != hash_json(config["generation_config"]):
        raise ValueError("Sample generation config mismatch")
    cap = config["generation_config"]["max_new_tokens"]
    if row.get("effective_max_new_tokens") != cap:
        raise ValueError("Sample silently changed generation budget")
    ids = row.get("generated_token_ids")
    if not isinstance(ids, list) or row.get("generated_tokens") != len(ids):
        raise ValueError("Sample token counts disagree")
    if any(row.get(key) != value for key, value in stop_metadata(ids, cap).items()):
        raise ValueError("Sample stopping metadata disagrees with tokens")
    for key in ("raw_generation", "completion"):
        if not isinstance(row.get(key), str):
            raise ValueError("Missing generated text")
        if row.get(key + "_sha256") != hash_text(row[key]):
            raise ValueError("Generated text hash mismatch")
    if row.get("generated_token_ids_sha256") != hash_json(ids):
        raise ValueError("Generated token hash mismatch")
    for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid timing/memory metadata")
    layers = 24 if "1.4B" in config["model_identity"]["repo_id"] else 48
    validate_execution(row["execution_observation"], mode, len(prompt["prompt_token_ids"]), len(ids), 4 * layers)
    adapter = row.get("adapter_observation", {})
    if adapter.get("native_exit_at_step") != 3 or adapter.get("mode") != mode or adapter.get("guidance_applied") != (mode != "baseline") or adapter.get("extra_lm_head_calls") != (0 if mode == "baseline" else 1):
        raise ValueError("Adapter did not observe registered guidance")
    if mode != "baseline" and (adapter.get("executed_source_indices") != [0, 1, 2, 3] or adapter.get("early_loop") != 1 or adapter.get("already_normalized_identity") is not True):
        raise ValueError("Invalid early reference trajectory")


def read_completed(path, config, prompts):
    result = {}
    path = Path(path)
    if not path.exists():
        return result
    expected_order = [(task, sample) for task in prompts for sample in range(config["samples_per_problem"])]
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        key = (row.get("task_id"), row.get("sample_id"))
        if key[0] not in prompts or key in result or index >= len(expected_order) or key != expected_order[index]:
            raise ValueError("Unknown, duplicate, missing, or reordered generated sample; expected ordered prefix")
        validate_sample(row, config, prompts[key[0]])
        result[key] = row
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def repair_partial_tail(path):
    """Preserve malformed final bytes before truncation; never repair the middle."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return None
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        try:
            json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Interior or terminated JSON corruption; refusing recovery")
            backup = path.with_name(path.name + ".partial-" + hashlib.sha256(raw).hexdigest() + ".bak")
            if not backup.exists():
                with backup.open("xb") as handle:
                    handle.write(raw)
            elif backup.read_bytes() != raw:
                raise ValueError("Partial-tail backup collision")
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            return {"backup": backup.name, "original_sha256": hashlib.sha256(raw).hexdigest(), "removed_bytes": len(raw) - offset}
        offset += len(line)
    if not raw.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return None


@contextmanager
def output_lock(folder):
    import fcntl
    with (Path(folder) / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this AIME output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
