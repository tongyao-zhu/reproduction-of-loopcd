"""Observe native Ouro recurrence and guide every requested logit position.

This adapter does not replace the recurrent forward pass or update caches
itself. It reads already normalized native early-loop states and performs
one extra LM-head projection. It implements the paper's *logit-space* Ouro
variant; hidden-state guidance is a separate, unclaimed experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch

from .guidance import GuidanceConfig, apply_guidance


_MISSING = object()


class OuroGuidance:
    """Install inference guidance temporarily on an unmodified Ouro model.

    Example::

        settings = GuidanceConfig(mode="adaptive", omega_cap=2.0)
        with OuroGuidance(model.eval(), settings, total_loops=4):
            output = model(input_ids, use_cache=False)  # all token positions
            generated = model.generate(input_ids, max_new_tokens=64)

    The native output policy is fixed to the last requested loop while the
    context is active, including baseline mode. Original exit settings and
    ``forward`` are restored afterwards, including if an exception occurs.
    The requested loop count must match the model's native configuration;
    this class never silently changes the recurrent compute budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: GuidanceConfig | None = None,
        *,
        total_loops: int = 4,
    ) -> None:
        if isinstance(total_loops, bool) or not isinstance(total_loops, int) or total_loops < 1:
            raise ValueError("total_loops must be a positive integer")
        self.model = model
        self.config = config if config is not None else GuidanceConfig()
        self.total_loops = total_loops
        if self.config.early_loop > total_loops:
            raise ValueError("early_loop cannot exceed total_loops")
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise TypeError("Expected native Ouro model.model and model.lm_head")
        for source in (model.config, model.model):
            if getattr(source, "total_ut_steps", None) != total_loops:
                raise ValueError("total_loops must match model.config and native model.total_ut_steps")
        self.last_observation: dict | None = None
        self._active = False
        self._saved_attributes: list[tuple[object, str, object]] = []

    def __enter__(self) -> "OuroGuidance":
        if self._active or getattr(self.model, "_loopcd_repro_guidance", None) is not None:
            raise RuntimeError("An OuroGuidance context is already active on this model")
        if self.model.training:
            raise ValueError("OuroGuidance requires model.eval()")
        if getattr(self.model, "_loopcd_adapter_installed", False):
            raise RuntimeError("An incompatible Ouro forward adapter is already active")
        self._native_forward = self.model.forward
        self._original_instance_forward = self.model.__dict__.get("forward", _MISSING)
        try:
            for target in (self.model, self.model.config):
                for name, value in (
                    ("early_exit_step", self.total_loops - 1),
                    ("early_exit_threshold", None),
                ):
                    self._saved_attributes.append((target, name, getattr(target, name, _MISSING)))
                    setattr(target, name, value)

            adapter = self

            # Preserve named inputs needed by Transformers generation's
            # signature inspection (especially attention_mask/use_cache).
            def replacement(
                this,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                cache_position=None,
                logits_to_keep=0,
                use_weighted_exit=False,
                exit_at_step=None,
                exit_threshold=None,
                **kwargs,
            ):
                return adapter._forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    use_weighted_exit=use_weighted_exit,
                    exit_at_step=exit_at_step,
                    exit_threshold=exit_threshold,
                    **kwargs,
                )

            self.model.forward = MethodType(replacement, self.model)
            self.model._loopcd_repro_guidance = self
            self._active = True
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._restore()
        return False

    @contextmanager
    def installed(self) -> Iterator["OuroGuidance"]:
        """Equivalent to using this object directly as a context manager."""
        with self:
            yield self

    def _restore(self) -> None:
        if hasattr(self, "_original_instance_forward"):
            if self._original_instance_forward is _MISSING:
                self.model.__dict__.pop("forward", None)
            else:
                self.model.forward = self._original_instance_forward
        for target, name, value in reversed(self._saved_attributes):
            if value is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, value)
        self._saved_attributes.clear()
        if getattr(self.model, "_loopcd_repro_guidance", None) is self:
            delattr(self.model, "_loopcd_repro_guidance")
        self._active = False

    @staticmethod
    def _select_positions(hidden: torch.Tensor, logits_to_keep) -> torch.Tensor:
        if isinstance(logits_to_keep, int) and not isinstance(logits_to_keep, bool):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be nonnegative")
            return hidden[:, slice(-logits_to_keep, None), :]
        if isinstance(logits_to_keep, torch.Tensor):
            if logits_to_keep.ndim != 1 or logits_to_keep.dtype not in (torch.int32, torch.int64):
                raise ValueError("logits_to_keep tensor must be one-dimensional integer indices")
            return hidden.index_select(1, logits_to_keep.to(hidden.device))
        raise TypeError("logits_to_keep must be a nonnegative integer or an index tensor")

    def _forward(self, **kwargs):
        self.last_observation = None
        if self.model.training:
            raise ValueError("OuroGuidance is inference-only; call model.eval()")
        if kwargs.get("labels") is not None:
            raise ValueError("Use output logits for likelihood scoring; labels invoke Ouro's weighted training exit")
        if kwargs.get("use_weighted_exit"):
            raise ValueError("Weighted exits are incompatible with fixed-loop guidance")
        if kwargs.get("exit_at_step") not in (None, self.total_loops - 1):
            raise ValueError("Native output must use the final configured loop")
        if kwargs.get("exit_threshold") is not None:
            raise ValueError("Threshold exits are incompatible with fixed-loop guidance")
        if (
            self.model.early_exit_step != self.total_loops - 1
            or self.model.early_exit_threshold is not None
        ):
            raise RuntimeError("Native Ouro exit settings changed during the guidance context")

        if not self.config.enabled:
            output = self._native_forward(**kwargs)
            self.last_observation = {
                "mode": self.config.mode,
                "guidance_applied": False,
                "extra_lm_head_calls": 0,
                "native_exit_at_step": self.total_loops - 1,
            }
            return output

        captured_states = []
        normalized_states = []
        executed_loops = []

        def capture_inner(module, inputs, output):
            if not isinstance(output, tuple) or len(output) != 3:
                raise TypeError("Expected native Ouro (outputs, normalized UT states, gates)")
            captured_states.append(output[1])

        def capture_norm(module, inputs, output):
            normalized_states.append(output)

        def capture_loop(module, inputs, named_inputs):
            executed_loops.append(int(named_inputs["current_ut"]))

        handles = []
        try:
            handles.append(self.model.model.register_forward_hook(capture_inner))
            handles.append(self.model.model.norm.register_forward_hook(capture_norm))
            handles.append(self.model.model.layers[0].register_forward_pre_hook(capture_loop, with_kwargs=True))
            output = self._native_forward(**kwargs)
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_states) != 1 or len(captured_states[0]) != self.total_loops:
            raise RuntimeError("Expected one complete native Ouro recurrent trajectory")
        if executed_loops != list(range(self.total_loops)):
            raise RuntimeError(f"Unexpected native recurrence indices: {executed_loops}")
        if len(normalized_states) != self.total_loops or not all(
            native is observed for native, observed in zip(captured_states[0], normalized_states)
        ):
            raise RuntimeError("Readouts are not the already normalized native loop outputs")

        final_logits = output[0] if isinstance(output, tuple) else output.logits
        early_hidden = captured_states[0][self.config.early_loop - 1]
        selected_hidden = self._select_positions(early_hidden, kwargs.get("logits_to_keep", 0))
        early_logits = self.model.lm_head(selected_hidden)
        guided = apply_guidance(final_logits, early_logits, self.config)
        if isinstance(output, tuple):
            output = (guided,) + output[1:]
        else:
            output.logits = guided
        self.last_observation = {
            "mode": self.config.mode,
            "guidance_applied": True,
            "extra_lm_head_calls": 1,
            "executed_source_indices": executed_loops,
            "early_loop": self.config.early_loop,
            "native_exit_at_step": self.total_loops - 1,
            "already_normalized_identity": True,
            "guided_logit_shape": list(guided.shape),
            "score_dtype": str(guided.dtype),
        }
        return output
