"""Logit-space guidance from arXiv:2610.02185, without plausibility filtering.

For final-loop logits ``z_R`` and early-loop logits ``z_1``, Equation 3 is
``z_R + omega * (z_R - z_1)``. Adaptive guidance obtains the top-two
probability margin from the *full, unmodified* final-loop softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


@dataclass(frozen=True)
class GuidanceConfig:
    """Explicit, serializable settings; loop numbers are one-based.

    ``omega`` controls fixed guidance and ``omega_cap`` controls adaptive
    guidance. The unused setting has no effect. ``baseline`` and a zero
    active coefficient both preserve the native logits object and dtype.
    """

    mode: Literal["baseline", "fixed", "adaptive"] = "baseline"
    omega: float = 1.0
    omega_cap: float = 1.0
    early_loop: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "fixed", "adaptive"}:
            raise ValueError("mode must be baseline, fixed, or adaptive")
        for name in ("omega", "omega_cap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.early_loop, bool) or not isinstance(self.early_loop, int):
            raise TypeError("early_loop must be a positive integer")
        if self.early_loop < 1:
            raise ValueError("early_loop uses one-based loop numbers")

    @property
    def enabled(self) -> bool:
        return (self.mode == "fixed" and self.omega != 0) or (
            self.mode == "adaptive" and self.omega_cap != 0
        )


def adaptive_strength(final_logits: torch.Tensor, omega_cap: float) -> torch.Tensor:
    """Return a separate FP32 strength for each batch/token position.

    The result has shape ``final_logits.shape[:-1] + (1,)``, so its final
    dimension broadcasts over the vocabulary. In particular this is not a
    two-class softmax over the two highest logits: all vocabulary entries
    contribute to the softmax normalization.
    """
    if isinstance(omega_cap, bool) or not isinstance(omega_cap, (int, float)):
        raise TypeError("omega_cap must be a finite nonnegative number")
    if not math.isfinite(omega_cap) or omega_cap < 0:
        raise ValueError("omega_cap must be finite and nonnegative")
    if final_logits.ndim < 1 or final_logits.shape[-1] < 2:
        raise ValueError("Adaptive guidance requires at least two vocabulary entries")
    top_two = final_logits.float().softmax(dim=-1).topk(2, dim=-1).values
    margin = top_two[..., :1] - top_two[..., 1:2]
    return float(omega_cap) * (1.0 - margin)


def apply_guidance(
    final_logits: torch.Tensor,
    early_logits: torch.Tensor | None,
    config: GuidanceConfig,
) -> torch.Tensor:
    """Apply guidance to every supplied position, using FP32 arithmetic.

    No log-softmax subtraction, vocabulary mask, renormalization, or
    temperature is applied here. Generation/scoring can consume the result
    as logits in the usual way. Zero guidance is an exact identity shortcut
    and does not require an early-loop readout.
    """
    if not config.enabled:
        return final_logits
    if early_logits is None:
        raise ValueError("Nonzero guidance requires early-loop logits")
    if final_logits.shape != early_logits.shape:
        raise ValueError("Final and early logits must have identical shapes")
    if final_logits.device != early_logits.device:
        raise ValueError("Final and early logits must be on the same device")
    final = final_logits.float()
    early = early_logits.float()
    omega = (
        float(config.omega)
        if config.mode == "fixed"
        else adaptive_strength(final, config.omega_cap)
    )
    return final + omega * (final - early)
