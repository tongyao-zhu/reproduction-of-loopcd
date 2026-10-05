"""Cache-free Parcae guidance around the pinned native recurrent forward.

The early state is the output of a complete core iteration, before C.
Logits guidance repeats the complete C/coda/norm/head readout; hidden
guidance blends before C and retains exactly one native readout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .guidance import GuidanceConfig, apply_guidance


@dataclass(frozen=True)
class ParcaeGuidanceConfig:
    mode: str = "baseline"
    total_loops: int = 8
    reference_loop: int = 1
    omega: float = 0.5
    omega_cap: float = 1.0

    def __post_init__(self):
        if self.mode not in {"baseline", "fixed", "adaptive", "hidden"}:
            raise ValueError("Unsupported Parcae guidance mode")
        for name in ("total_loops", "reference_loop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.reference_loop > self.total_loops:
            raise ValueError("reference_loop exceeds total_loops")
        for name in ("omega", "omega_cap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def enabled(self):
        return self.mode != "baseline" and (
            self.omega_cap != 0 if self.mode == "adaptive" else self.omega != 0
        )


_MISSING = object()


class ParcaeGuidance:
    """An explicit MC forward callable, without changing model generation.

    Only an unpadded batch of one and no cache/mask are supported. The
    native initializer and recurrent trajectory remain in charge of RNG.
    This object never reseeds, samples, truncates, or tokenizes inputs.
    """

    def __init__(self, model, config: ParcaeGuidanceConfig | None = None):
        self.model = model
        self.config = config or ParcaeGuidanceConfig()
        for name in ("forward_for_generation", "core_block_forward", "initialize_state"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"Expected native Parcae {name}")
        for name in ("C", "prelude", "core_block", "coda", "ln_f"):
            if not hasattr(model.transformer, name):
                raise TypeError(f"Missing transformer.{name}")
        if model.config.use_fused_head != "pytorch":
            raise ValueError("Only the pinned native pytorch head is supported")
        self.last_observation = None

    def _readout(self, hidden, tokens):
        """Pinned native output interface, without cache or extra recurrence."""
        model = self.model
        freqs = model.freqs_cis[:, :tokens.shape[1]]
        hidden = model.transformer.C(hidden)
        prelude = len(model.transformer.prelude)
        core = len(model.transformer.core_block)
        for index, block in enumerate(model.transformer.coda):
            key = str(prelude + core + index)
            embedding = model.value_embeds[key] if key in model.value_embeds else None
            value = embedding(tokens) if embedding is not None else None
            hidden = block(
                hidden, freqs, None, past_key_values=None,
                step_idx=torch.tensor(prelude + self.config.total_loops * core + index, dtype=torch.long),
                ve=value,
            )
        hidden = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden).float() * model.config.init.logit_scale
        softcap = getattr(model.config, "logit_softcap", None)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    @torch.no_grad()
    def __call__(self, input_ids, *, attention_mask=None, past_key_values=None):
        self.last_observation = None
        model, config = self.model, self.config
        if model.training:
            raise ValueError("Parcae guidance requires model.eval()")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected one unpadded token sequence")
        if input_ids.dtype != torch.long or not 0 < input_ids.shape[1] <= model.config.block_size:
            raise ValueError("Expected integer tokens within the native context limit")
        if attention_mask is not None or past_key_values is not None:
            raise ValueError("This MC adapter supports neither masks nor caches")
        if getattr(model, "_loopcd_parcae_active", False):
            raise RuntimeError("Another Parcae guidance call is already active")
        if not config.enabled:
            output = model.forward_for_generation(input_ids, num_steps=config.total_loops, past_key_values=None)
            self.last_observation = {
                "mode": config.mode, "guidance_applied": False,
                "total_loops": config.total_loops, "readout_passes": 1,
            }
            return output

        native_core = model.core_block_forward
        original_core = model.__dict__.get("core_block_forward", _MISSING)
        original_active = model.__dict__.get("_loopcd_parcae_active", _MISSING)
        reference, final = None, None
        executed = []
        projection_calls = 0
        model._loopcd_parcae_active = True

        def observe_core(*args, **kwargs):
            nonlocal reference, final
            step = kwargs.get("step")
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or int(step) != len(executed):
                raise RuntimeError("Unexpected native recurrence index")
            result = native_core(*args, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise TypeError("Expected completed native core state")
            executed.append(int(step))
            if len(executed) == config.reference_loop:
                reference = result.detach().clone()
            final = result
            return result

        def before_projection(module, args):
            nonlocal projection_calls
            projection_calls += 1
            if projection_calls != 1 or executed != list(range(config.total_loops)):
                raise RuntimeError("Expected one native C projection after all loops")
            if reference is None or args[0] is not final:
                raise RuntimeError("C must receive the completed raw recurrent state")
            if config.mode == "hidden":
                blended = final.float() + float(config.omega) * (final.float() - reference.float())
                return (blended.to(final.dtype),) + args[1:]

        hook = None
        try:
            model.core_block_forward = observe_core
            hook = model.transformer.C.register_forward_pre_hook(before_projection)
            output = model.forward_for_generation(input_ids, num_steps=config.total_loops, past_key_values=None)
            if projection_calls != 1 or executed != list(range(config.total_loops)):
                raise RuntimeError("Incomplete native recurrent/readout trajectory")
        finally:
            if hook is not None:
                hook.remove()
            if original_core is _MISSING:
                model.__dict__.pop("core_block_forward", None)
            else:
                model.core_block_forward = original_core
            if original_active is _MISSING:
                delattr(model, "_loopcd_parcae_active")
            else:
                model._loopcd_parcae_active = original_active

        if config.mode in {"fixed", "adaptive"}:
            early_logits = self._readout(reference, input_ids)
            output["logits"] = apply_guidance(output["logits"], early_logits, GuidanceConfig(
                mode=config.mode, omega=config.omega, omega_cap=config.omega_cap,
                early_loop=config.reference_loop,
            ))
        self.last_observation = {
            "mode": config.mode, "guidance_applied": True,
            "total_loops": config.total_loops, "reference_loop": config.reference_loop,
            "executed_source_indices": executed,
            "readout_passes": 1 if config.mode == "hidden" else 2,
            "combination_location": "before_C" if config.mode == "hidden" else "after_complete_native_readout",
            "cache": None,
        }
        return output
