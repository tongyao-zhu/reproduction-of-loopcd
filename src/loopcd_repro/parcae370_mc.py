"""Parcae candidate likelihood scoring, without an HF model-type workaround.

All heavyweight imports are lazy. Tokenization and record-validation helpers
are usable with just the standard library; model execution uses the real LM API.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
from numbers import Integral
from pathlib import Path
import sys
import time

MAX_LENGTH = 2048
PRECISION = "BF16 model; FP32 logits guidance; FP32 hidden blend cast to BF16 before native readout; FP32 log_softmax"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def encode_pair(tokenizer, context, continuation, max_length=MAX_LENGTH):
    """Exact registered causal boundary handling, fail closed before scoring."""
    if not isinstance(context, str) or not isinstance(continuation, str) or not context or not continuation:
        raise ValueError("Nonempty string context and continuation are required; no EOT/BOS fallback")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Invalid context limit")
    trailing = len(context) - len(context.rstrip())
    context_text = context[:-trailing] if trailing else context
    continuation_text = context[-trailing:] + continuation if trailing else continuation
    context_ids = tokenizer.encode(context_text, return_tensors=False)
    full_ids = tokenizer.encode(context_text + continuation_text, return_tensors=False)
    for ids in (context_ids, full_ids):
        if not isinstance(ids, list) or any(type(token) is not int or not 0 <= token < tokenizer.vocab_size for token in ids):
            raise ValueError("Native tokenizer must return valid integer token lists")
    if not context_ids or context_ids != full_ids[:len(context_ids)]:
        raise ValueError("Empty or non-prefix tokenized context")
    continuation_ids = full_ids[len(context_ids):]
    if not continuation_ids or len(continuation_ids) > max_length:
        raise ValueError("Empty or overlong candidate")
    input_ids = full_ids[-(max_length + 1):][:-1]
    if not input_ids or len(continuation_ids) > len(input_ids):
        raise ValueError("Every candidate token needs an intact causal predictor")
    metadata = {
        "arguments_sha256": digest([context, continuation]),
        "context_sha256": text_digest(context), "continuation_text_sha256": text_digest(continuation),
        "context_tokens_sha256": digest(context_ids), "full_tokens_sha256": digest(full_ids),
        "input_ids_sha256": digest(input_ids), "continuation_ids_sha256": digest(continuation_ids),
        "original_context_length": len(context_ids), "original_full_length": len(full_ids),
        "input_length": len(input_ids), "continuation_length": len(continuation_ids),
        "left_truncated_tokens": max(0, len(full_ids) - (max_length + 1)),
    }
    return {"input_ids": input_ids, "continuation_ids": continuation_ids, "pairing": metadata}


def audited_pairing(row):
    """Map the completed all-task CPU audit schema to the execution schema."""
    view = row["hf"]
    return {
        "task": row["leaf"], "doc_id": row["doc_id"], "candidate": row["candidate"],
        "arguments_sha256": row["arguments_sha256"], "context_sha256": row["context_sha256"],
        "continuation_text_sha256": row["continuation_sha256"],
        "context_tokens_sha256": view["context_sha256"], "full_tokens_sha256": view["full_sha256"],
        "input_ids_sha256": view["input_sha256"], "continuation_ids_sha256": view["continuation_sha256"],
        "original_context_length": view["context_tokens"], "original_full_length": view["full_tokens"],
        "input_length": view["input_tokens"], "continuation_length": view["continuation_tokens"],
        "left_truncated_tokens": view["left_dropped"],
    }


def read_prompt_audit(directory, task, limit=None):
    directory = Path(directory).resolve(strict=True)
    report = json.loads((directory / "result.json").read_text())
    if report.get("status") != "PASS" or report.get("registered_hf_policy_ready") is not True:
        raise ValueError("A complete passed Parcae prompt audit is required")
    if report.get("total_documents") != 1172 or report.get("total_requests") != 4687:
        raise ValueError("Prompt audit is not the registered complete ARC-C audit")
    if task != "arc_challenge" or report.get("profile") != "parcae370_arc_challenge_v1" or report.get("shots") != {"arc_challenge": 25}:
        raise ValueError("Only the pinned 370M ARC-C25 audit is accepted")
    entry = report["tasks"][task]
    path = directory / entry["records"]["file"]
    if path.name != task + ".jsonl.gz" or path.parent != directory or sha256(path) != entry["records"]["sha256"]:
        raise ValueError("Prompt audit request evidence changed")
    expected, count = {}, 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["task"] != task or not row["hf"]["boundary_prefix_valid"] or not row["hf"]["all_labels_have_predictors"]:
                raise ValueError("Invalid audited request")
            count += 1
            if limit is None or row["doc_id"] < limit:
                identity = (row["leaf"], row["doc_id"], row["candidate"])
                if identity in expected:
                    raise ValueError("Repeated request identity in audit; repeated arguments remain allowed")
                expected[identity] = audited_pairing(row)
    if count != entry["requests"] or not expected:
        raise ValueError("Incomplete prompt audit evidence")
    binding = {"path": str(directory), "result_sha256": sha256(directory / "result.json"),
               "records_sha256": sha256(path), "records_file": path.name,
               "harness_files_sha256": report["harness"]["files_sha256"],
               "task_documents": entry["documents"], "task_requests": entry["requests"]}
    return report, expected, binding


def load_parcae(model_path, device="cuda:0"):
    """Pinned native loader, strict key matching, weights_only=True upstream."""
    import torch
    from prepare_parcae370 import verify_prepared
    path = Path(model_path).resolve(strict=True)
    prepared = verify_prepared(path)
    sys.path.insert(0, str(path / "source"))
    from receval.models.parcae import ModelingParcae
    from parcae_lm.tokenizer import Tokenizer
    from parcae_lm.attention_backends.flash_attention import HAS_FA3
    for module_name, relative in (("receval.models.parcae", "receval/models/parcae.py"),
                                   ("parcae_lm.tokenizer", "parcae_lm/tokenizer.py")):
        source = Path(importlib.import_module(module_name).__file__).resolve()
        if source != path / "source" / relative or sha256(source) != prepared["source"]["files"][relative]["sha256"]:
            raise ValueError("Imported native module differs from prepared source")
    records = []
    class StrictNativeParcae(ModelingParcae):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            result = super().load_state_dict(state_dict, strict=True, assign=assign)
            records.append({"official_requested_strict": strict, "effective_strict": True,
                            "loaded_keys": len(state_dict), "missing_keys": list(result.missing_keys),
                            "unexpected_keys": list(result.unexpected_keys)})
            return result
    model = StrictNativeParcae.from_pretrained(path, device=device, dtype=torch.bfloat16).eval()
    if len(records) != 1 or records[0]["loaded_keys"] != 117 or records[0]["missing_keys"] or records[0]["unexpected_keys"]:
        raise ValueError("Unexpected checkpoint key inventory")
    if (model.config.state_init != "like-init" or model.config.block_size != 2048
            or model.config.mean_recurrence != 8 or model.config.padded_vocab_size != 32768
            or any(len(getattr(model.transformer, name)) != 4 for name in ("prelude", "core_block", "coda"))
            or HAS_FA3 or any(parameter.dtype != torch.bfloat16 for parameter in model.parameters())):
        raise ValueError("Unexpected pinned native model configuration, dtype, or backend")
    tokenizer = Tokenizer.from_directory(path)
    if any(getattr(tokenizer, name) is not None for name in ("bos_id", "eos_id", "pad_id")):
        raise ValueError("Special-token registration differs from the native audit")
    return model, tokenizer, {**records[0], "prepared": prepared, "attention": "sdpa",
                              "native_model_code_sha256": prepared["source"]["files"]["receval/models/parcae.py"]["sha256"]}


def validate_gpu_gate(path, model_path, source_root):
    """Verify independent real-GPU checks and their exact model/code/runtime."""
    from gpu_smoke_parcae370 import validate_report
    path, model_path, source_root = Path(path), Path(model_path), Path(source_root)
    report = json.loads(path.read_text())
    result = validate_report(report)
    model_manifest = model_path / "model_provenance.json"
    if report["model_provenance"] != {"bytes": model_manifest.stat().st_size, "sha256": sha256(model_manifest)}:
        raise ValueError("GPU gate prepared model binding differs")
    required_sources = {"scripts/gpu_smoke_parcae370.py", "scripts/prepare_parcae370.py",
                        "src/loopcd_repro/parcae.py", "src/loopcd_repro/guidance.py",
                        "scripts/launch_huginn_r16_suite.py"}
    if set(report["sources"]) != required_sources:
        raise ValueError("GPU gate source file set changed")
    for name, expected in report["sources"].items():
        source = source_root / name
        if {"bytes": source.stat().st_size, "sha256": sha256(source)} != expected:
            raise ValueError(f"GPU gate source differs: {name}")
    if set(report["packages"]) != {"torch", "transformers", "numpy", "einops"}:
        raise ValueError("GPU gate package inventory changed")
    for name, version in report["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"GPU gate runtime differs: {name}")
    return {**result, "sha256": sha256(path), "path": str(path.resolve())}


def request_identity(request):
    if request.request_type != "loglikelihood" or request.repeats != 1:
        raise ValueError("Only once-per-candidate likelihood requests are supported")
    if (not isinstance(request.task_name, str) or
            any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in (request.doc_id, request.idx))):
        raise ValueError(f"A native leaf/document/candidate identity is required: "
                         f"{request.task_name!r}/{request.doc_id!r}/{request.idx!r} "
                         f"({type(request.task_name).__name__}/{type(request.doc_id).__name__}/{type(request.idx).__name__})")
    return {"task": request.task_name, "doc_id": int(request.doc_id), "candidate": int(request.idx)}


class RequestAudit:
    def __init__(self, stream, expected=None):
        self.stream, self.expected = stream, expected
        self.planned = None
        self.count = self.truncated = 0
        self.pairing_digest, self.trace_digest = hashlib.sha256(), hashlib.sha256()
        self.previous_rng = None

    def plan(self, requests, tokenizer):
        if self.planned is not None:
            raise ValueError("This task must produce one ordered likelihood batch")
        planned, seen = [], set()
        for request in requests:
            identity = request_identity(request)
            key = tuple(identity[field] for field in ("task", "doc_id", "candidate"))
            if key in seen:
                raise ValueError("Repeated candidate identity; argument duplicates must have distinct identities")
            seen.add(key)
            item = {**identity, **encode_pair(tokenizer, *request.args)["pairing"]}
            if self.expected is not None and item != self.expected.get(key):
                raise ValueError(f"Request differs from the full prompt audit: {key}")
            planned.append(item)
        if not planned or (self.expected is not None and seen != set(self.expected)):
            raise ValueError("Missing or extra audited request identities")
        self.planned = planned

    def record(self, metadata, before, after, initialization, observation, likelihood, greedy):
        if self.planned is None or self.count >= len(self.planned) or metadata != self.planned[self.count]:
            raise ValueError("Actual execution differs from prevalidated request order")
        if before != initialization["rng_before"] or after != initialization["rng_after"]:
            raise ValueError("Random draws occurred outside the native initializer")
        if self.previous_rng is not None and before != self.previous_rng:
            raise ValueError("Native random stream was reset or changed between requests")
        pairing = {**metadata, "rng_before": before, "rng_after": after, "initialization": initialization}
        row = {"index": self.count, "pairing": pairing, "observation": observation,
               "loglikelihood": likelihood, "is_greedy": greedy}
        payload = canonical(row) + b"\n"
        self.stream.write(payload.decode())
        self.stream.flush()
        self.trace_digest.update(payload)
        self.pairing_digest.update(canonical(pairing) + b"\n")
        self.previous_rng = after
        self.count += 1
        self.truncated += metadata["left_truncated_tokens"] > 0

    def summary(self, require_complete=False):
        complete = self.planned is not None and self.count == len(self.planned) and self.count > 0
        if require_complete and not complete:
            raise ValueError("Incomplete likelihood execution audit")
        return {"request_count": self.count, "planned_requests": len(self.planned or []),
                "complete": complete, "truncated_requests": self.truncated,
                "pairing_sha256": self.pairing_digest.hexdigest(), "trace_sha256": self.trace_digest.hexdigest()}


def make_parcae_lm(base_class):
    """Return a real LM subclass. Does not invoke HFLM or change model types."""
    import torch
    from .parcae import ParcaeGuidance, ParcaeGuidanceConfig

    def rng_snapshot(device):
        states = {"cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()}
        if device.type == "cuda":
            states["model_device"] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
        elif device.type != "cpu":
            raise ValueError("Only CPU or one CUDA device is supported")
        return states

    class ParcaeMC(base_class):
        def __init__(self, model, tokenizer, guidance, audit):
            super().__init__()
            self.model, self.tokenizer, self.audit = model, tokenizer, audit
            self.guidance = ParcaeGuidance(model, ParcaeGuidanceConfig(**guidance))
            self.device = next(model.parameters()).device
            self.batch_size, self.max_length = 1, MAX_LENGTH
            self.batch_sizes = {}
            self.config = {"name_or_path": "SandyResearch/parcae-370m"}
            if model.training:
                raise ValueError("Native model must be in eval mode")

        @property
        def tokenizer_name(self):
            return "parcae-tokenizer-6247b5d0592876b73660f34c2dd16c9db9e9045c-native-no-bos"

        def loglikelihood(self, requests):
            self.audit.plan(requests, self.tokenizer)
            answers = []
            for request in requests:
                plan = encode_pair(self.tokenizer, *request.args)
                metadata = {**request_identity(request), **plan["pairing"]}
                tokens = torch.tensor([plan["input_ids"]], dtype=torch.long, device=self.device)
                before = rng_snapshot(self.device)
                initializations, hooks = [], []
                counts = {"initializations": 0, "prelude": 0, "core_layers": 0,
                          "projection": 0, "coda": 0, "norm": 0, "head": 0}
                native = self.model.initialize_state
                existed, old = "initialize_state" in self.model.__dict__, self.model.__dict__.get("initialize_state")
                def observe_init(inputs):
                    init_before = rng_snapshot(self.device)
                    state = native(inputs)
                    flat = state.detach().flatten()
                    initializations.append({"rng_before": init_before, "rng_after": rng_snapshot(self.device),
                        "shape": list(state.shape), "dtype": str(state.dtype),
                        "first_16_values_sha256": digest(flat[:16].float().cpu().tolist()),
                        "last_16_values_sha256": digest(flat[-16:].float().cpu().tolist())})
                    counts["initializations"] += 1
                    return state
                def count_hook(name):
                    def callback(module, args, result):
                        counts[name] += 1
                    return callback
                self.model.initialize_state = observe_init
                for group, key in (("prelude", "prelude"), ("core_block", "core_layers"), ("coda", "coda")):
                    hooks.extend(layer.register_forward_hook(count_hook(key)) for layer in getattr(self.model.transformer, group))
                for layer, key in ((self.model.transformer.C, "projection"), (self.model.transformer.ln_f, "norm"), (self.model.lm_head, "head")):
                    hooks.append(layer.register_forward_hook(count_hook(key)))
                started = time.monotonic()
                try:
                    with torch.no_grad():
                        output = self.guidance(tokens, past_key_values=None)
                    after = rng_snapshot(self.device)
                finally:
                    for hook in hooks:
                        hook.remove()
                    if existed:
                        self.model.initialize_state = old
                    else:
                        self.model.__dict__.pop("initialize_state", None)
                passes = 2 if self.guidance.config.enabled and self.guidance.config.mode in ("fixed", "adaptive") else 1
                expected_counts = {"initializations": 1, "prelude": len(self.model.transformer.prelude),
                    "core_layers": len(self.model.transformer.core_block) * self.guidance.config.total_loops,
                    "projection": passes, "coda": len(self.model.transformer.coda) * passes,
                    "norm": passes, "head": passes}
                if counts != expected_counts or len(initializations) != 1 or output.get("past_key_values") is not None:
                    raise ValueError("Unexpected native call counts or cache usage")
                logits = output["logits"]
                if logits.ndim != 3 or logits.shape[:2] != tokens.shape or logits.shape[2] != self.tokenizer.vocab_size:
                    raise ValueError("Invalid native logits shape")
                selected = logits[0, -len(plan["continuation_ids"]):, :].float()
                if not torch.isfinite(selected).all():
                    raise ValueError("Non-finite scored logits")
                log_probs = torch.log_softmax(selected, dim=-1)
                labels = torch.tensor(plan["continuation_ids"], device=self.device, dtype=torch.long)
                value = float(log_probs.gather(-1, labels[:, None]).sum().item())
                greedy = bool(torch.equal(selected.argmax(-1), labels))
                if not math.isfinite(value):
                    raise ValueError("Non-finite candidate likelihood")
                observation = {"guidance": self.guidance.last_observation, "call_counts": counts,
                               "elapsed_seconds": time.monotonic() - started}
                self.audit.record(metadata, before, after, initializations[0], observation, value, greedy)
                answers.append((value, greedy))
                del output, logits, selected, log_probs, labels, tokens
            self.audit.summary(require_complete=True)
            return answers

        def loglikelihood_rolling(self, requests):
            raise ValueError("This evaluator only supports multiple-choice likelihood")

        def generate_until(self, requests):
            raise ValueError("This evaluator does not generate text")
    return ParcaeMC
