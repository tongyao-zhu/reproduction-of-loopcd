"""Native Huginn LoopCD-Hidden, combining raw loop outputs before ln_f/coda.

The output interface is ``ln_f -> coda -> ln_f -> lm_head``. Equation 2 is
applied *before* the first of those operations. Native Gaussian recurrent
initialization is retained; callers must control seeds (and use identical
input_states for cached/full-prefix numerical gates). No extra recurrence,
coda evaluation, or vocabulary projection is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Iterator, Literal

import torch


@dataclass(frozen=True)
class HuginnHiddenConfig:
    mode: Literal["baseline", "hidden"] = "baseline"
    total_loops: int = 32
    reference_loop: int = 6
    omega: float = 0.5

    def __post_init__(self):
        if self.mode not in {"baseline", "hidden"}:
            raise ValueError("Huginn Hidden supports baseline or hidden mode; no adaptive readout")
        for name in ("total_loops", "reference_loop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.reference_loop > self.total_loops:
            raise ValueError("reference_loop cannot exceed total_loops")
        if isinstance(self.omega, bool) or not isinstance(self.omega, (int, float)):
            raise TypeError("omega must be finite and nonnegative")
        if not math.isfinite(self.omega) or self.omega < 0:
            raise ValueError("omega must be finite and nonnegative")

    @property
    def enabled(self):
        return self.mode == "hidden" and self.omega != 0

    @property
    def cache_contract(self):
        # Coda history depends on guidance; recurrent cache layout depends on R.
        return (self.total_loops, self.reference_loop if self.enabled else None,
                float(self.omega) if self.enabled else 0.0)


def load_huginn(model_path, device="cuda:0"):
    """Load a prepared private checkpoint; do not modify sources or init mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .runtime import sha256

    path = Path(model_path).resolve()
    record_path = path / "model_provenance.json"
    if not record_path.is_file():
        raise ValueError("Run scripts/prepare_huginn.py first")
    record = json.loads(record_path.read_text())
    if record.get("repo_id") != "tomg-group-umd/huginn-0125":
        raise ValueError("Expected the pinned Huginn checkpoint")
    if sha256(path / "raven_modeling_minimal.py") != record["model_code_sha256"]:
        raise ValueError("Private Huginn model code differs from its preparation manifest")
    model = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, local_files_only=True, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


_MISSING = object()


class HuginnHiddenGuidance:
    """Temporarily guide native Huginn while retaining one native output pass.

    ``with HuginnHiddenGuidance(model, settings): model(...)`` supports
    teacher-forced full-token logits and ordinary native HF generation.
    Only unpadded batch size one and full native dynamic caches are validated.
    A populated cache may not be reused across different depths/guidance.
    ``input_states`` is accepted unchanged, allowing identical initial noise
    across forced-prefix comparisons. This adapter never resets global RNG.
    """

    def __init__(self, model, config: HuginnHiddenConfig | None = None):
        self.model = model
        self.config = config or HuginnHiddenConfig()
        if not hasattr(model, "transformer") or not hasattr(model, "iterate_forward"):
            raise TypeError("Expected native Huginn/Raven model")
        for name in ("core_block", "coda", "ln_f"):
            if name not in model.transformer:
                raise TypeError(f"Missing Huginn transformer.{name}")
        if not len(model.transformer.core_block) or not len(model.transformer.coda):
            raise ValueError("Huginn adapter requires a nonempty recurrent core and coda")
        self.last_observation = None
        self._active = False

    def __enter__(self):
        if self._active or getattr(self.model, "_loopcd_hidden_guidance", None) is not None:
            raise RuntimeError("Huginn guidance is already active")
        if self.model.training:
            raise ValueError("HuginnHiddenGuidance requires model.eval()")
        if getattr(self.model, "_loopcd_adapter_installed", False) or hasattr(self.model, "_original_forward"):
            raise RuntimeError("Another Huginn forward adapter is installed")
        if getattr(self.model.config, "test_time_noise", 0) != 0:
            raise ValueError("Additional test-time recurrent noise is outside this protocol")
        self._native_forward = self.model.forward
        self._original_instance_forward = self.model.__dict__.get("forward", _MISSING)
        adapter = self

        def replacement(this, input_ids=None, input_embeds=None, input_states=None,
                        attention_mask=None, position_ids=None, labels=None,
                        num_steps=None, past_key_values=None, output_details=None,
                        use_cache=False, cache_position=None, init_scale=1.0, **kwargs):
            arguments = dict(input_ids=input_ids, input_embeds=input_embeds,
                             input_states=input_states, attention_mask=attention_mask,
                             position_ids=position_ids, labels=labels, num_steps=num_steps,
                             past_key_values=past_key_values, use_cache=use_cache,
                             cache_position=cache_position, init_scale=init_scale, **kwargs)
            if output_details is not None:
                arguments["output_details"] = output_details
            return adapter._forward(arguments)

        self.model.forward = MethodType(replacement, self.model)
        self._original_instance_generate = self.model.__dict__.get("generate", _MISSING)
        self._native_generate = getattr(self.model, "generate", None)
        if self._native_generate is not None:
            def generate(this, *args, **kwargs):
                # Native special generators can call core_block_forward
                # directly, bypassing the model.forward observation hooks.
                unsupported = {"criterion", "exit_threshold", "exit_evaluator", "draft_steps",
                               "lookahead_for_draft", "verification_threshold", "continuous_compute"}
                if unsupported.intersection(kwargs):
                    raise ValueError("Only ordinary fixed-depth HF generation is supported by HuginnHiddenGuidance")
                return adapter._native_generate(*args, **kwargs)
            self.model.generate = MethodType(generate, self.model)
        self.model._loopcd_hidden_guidance = self
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._original_instance_forward is _MISSING:
            self.model.__dict__.pop("forward", None)
        else:
            self.model.forward = self._original_instance_forward
        if self._native_generate is not None:
            if self._original_instance_generate is _MISSING:
                self.model.__dict__.pop("generate", None)
            else:
                self.model.generate = self._original_instance_generate
        if getattr(self.model, "_loopcd_hidden_guidance", None) is self:
            delattr(self.model, "_loopcd_hidden_guidance")
        self._active = False
        return False

    @contextmanager
    def installed(self) -> Iterator["HuginnHiddenGuidance"]:
        with self:
            yield self

    def new_cache(self):
        cache_type = getattr(sys.modules[type(self.model).__module__], "HuginnDynamicCache")
        cache = cache_type(lookup_strategy="full")
        cache._loopcd_hidden_contract = self.config.cache_contract
        return cache

    def _validate(self, arguments):
        if self.model.training:
            raise ValueError("Huginn hidden guidance is inference-only")
        if arguments["labels"] is not None:
            raise ValueError("Score continuations using logits, without training labels")
        tokens = arguments["input_ids"]
        if tokens is None or tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("This Huginn protocol requires input_ids with batch size one")
        pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is not None and bool((tokens == pad_id).any()):
            raise ValueError("Native Huginn forward ignores padding masks; padded inputs are unsupported")
        mask = arguments["attention_mask"]
        if mask is not None and not bool(mask.all()):
            raise ValueError("Native Huginn forward ignores attention_mask; masking is unsupported")
        supplied_steps = arguments["num_steps"]
        if supplied_steps is not None:
            if isinstance(supplied_steps, torch.Tensor):
                if supplied_steps.numel() != 1 or supplied_steps.dtype not in (torch.int32, torch.int64):
                    raise ValueError("num_steps must be the configured single integer depth")
                supplied_steps = supplied_steps.item()
            if isinstance(supplied_steps, bool) or not isinstance(supplied_steps, int) or supplied_steps != self.config.total_loops:
                raise ValueError("num_steps differs from the configured fixed recurrent depth")
        arguments["num_steps"] = self.config.total_loops
        if arguments["init_scale"] != 1.0 and arguments["input_states"] is None:
            raise ValueError("Preserve native random initialization with init_scale=1.0")
        for name in ("continuous_compute", "criterion", "exit_threshold", "draft_steps"):
            if name in arguments:
                raise ValueError(f"{name} is outside fixed-depth Huginn guidance")
        cache = arguments["past_key_values"]
        if cache is not None and not arguments["use_cache"]:
            raise ValueError("A supplied cache requires use_cache=True; native Huginn otherwise still mutates it")
        if arguments["use_cache"]:
            if cache is None:
                cache = arguments["past_key_values"] = self.new_cache()
            if not isinstance(getattr(cache, "key_cache", None), dict) or getattr(cache, "lookup_strategy", None) != "full":
                raise TypeError("Only native full HuginnDynamicCache is supported")
            contract = getattr(cache, "_loopcd_hidden_contract", None)
            if contract is None and cache.get_seq_length() == 0:
                cache._loopcd_hidden_contract = self.config.cache_contract
            elif contract != self.config.cache_contract:
                raise ValueError("Cache history belongs to a different or unknown guidance/depth configuration")

    def _forward(self, arguments):
        self.last_observation = None
        self._validate(arguments)
        if not self.config.enabled:
            output = self._native_forward(**arguments)
            self.last_observation = {
                "mode": self.config.mode, "guidance_applied": False,
                "total_loops": self.config.total_loops, "extra_coda_passes": 0,
                "extra_lm_head_calls": 0, "initialization": "native_random_or_explicit_input_states",
            }
            return output

        reference = None
        final_raw = None
        completed = []
        norm_calls = 0
        coda_calls = []
        head_calls = []

        def capture_core(module, inputs, output):
            nonlocal reference, final_raw
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected a tensor after the final native recurrent layer")
            loop = len(completed) + 1
            completed.append(loop)
            if loop == self.config.reference_loop:
                reference = output.detach().clone()
            final_raw = output

        def guide_before_norm(module, inputs):
            nonlocal norm_calls
            norm_calls += 1
            if norm_calls == 1:
                if completed != list(range(1, self.config.total_loops + 1)):
                    raise RuntimeError(f"Unexpected native Huginn recurrence: {completed}")
                if reference is None or inputs[0] is not final_raw:
                    raise RuntimeError("Guidance must receive the raw completed-loop state before pre-coda normalization")
                final = inputs[0]
                blended = final.float() + float(self.config.omega) * (final.float() - reference.float())
                # The following coda remains in the original model precision.
                return (blended.to(final.dtype),) + inputs[1:]
            if norm_calls != 2:
                raise RuntimeError("Unexpected additional native Huginn output normalization")

        def observe_coda(module, inputs):
            coda_calls.append(id(module))

        def observe_head(module, inputs, output):
            head_calls.append(1)

        handles = []
        try:
            handles.append(self.model.transformer.core_block[-1].register_forward_hook(capture_core))
            handles.append(self.model.transformer.ln_f.register_forward_pre_hook(guide_before_norm))
            for layer in self.model.transformer.coda:
                handles.append(layer.register_forward_pre_hook(observe_coda))
            handles.append(self.model.lm_head.register_forward_hook(observe_head))
            output = self._native_forward(**arguments)
        finally:
            for handle in handles:
                handle.remove()
        if norm_calls != 2 or coda_calls != [id(layer) for layer in self.model.transformer.coda] or len(head_calls) != 1:
            raise RuntimeError("Expected exactly one native pre-coda norm, coda, final norm, and head")
        self.last_observation = {
            "mode": self.config.mode, "guidance_applied": True,
            "total_loops": self.config.total_loops, "reference_loop": self.config.reference_loop,
            "executed_physical_loops": completed, "coda_layer_calls": len(coda_calls),
            "lm_head_calls": 1, "extra_coda_passes": 0, "extra_lm_head_calls": 0,
            "combination_location": "raw_recurrent_output_before_native_pre_coda_ln_f",
            "arithmetic": "FP32 blend cast back to native hidden dtype before native output interface",
            "initialization": "native_random_or_explicit_input_states",
            "cache_semantics": "native recurrent history; guided coda history; one update per layer and token",
        }
        return output
