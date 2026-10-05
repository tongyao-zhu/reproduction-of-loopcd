import math

import pytest
import torch

from loopcd_repro.guidance import GuidanceConfig, adaptive_strength, apply_guidance


@pytest.mark.parametrize(
    "config",
    [GuidanceConfig(), GuidanceConfig(mode="fixed", omega=0), GuidanceConfig(mode="adaptive", omega_cap=0)],
)
def test_disabled_guidance_is_strict_object_and_dtype_identity(config):
    logits = torch.tensor([[[1.0, 2.0, -3.0]]], dtype=torch.bfloat16)
    assert apply_guidance(logits, None, config) is logits


def test_fixed_guidance_matches_equation_three_at_every_position_without_masking():
    final = torch.tensor([[[2.0, 0.0, -30.0], [1.0, 4.0, -2.0]]])
    early = torch.tensor([[[3.0, 2.0, -20.0], [2.0, 1.0, 3.0]]])
    actual = apply_guidance(final, early, GuidanceConfig(mode="fixed", omega=1.5))
    assert torch.equal(actual, torch.tensor([[[0.5, -3.0, -45.0], [-0.5, 8.5, -9.5]]]))
    assert torch.isfinite(actual).all()  # No plausibility cutoff at low-probability tokens.
    assert torch.equal(final, torch.tensor([[[2.0, 0.0, -30.0], [1.0, 4.0, -2.0]]]))


def test_adaptive_strength_uses_full_vocabulary_normalizer_and_native_final_logits():
    # exp(logits)=[4,2,1,1], so p1-p2=1/4 and omega=3/2.
    # A two-class softmax would instead produce omega=4/3.
    final = torch.tensor([[[math.log(4), math.log(2), 0.0, 0.0]]])
    early = torch.tensor([[[7.0, -3.0, 0.0, 8.0]]])
    strength = adaptive_strength(final, 2)
    torch.testing.assert_close(strength, torch.tensor([[[1.5]]]))
    actual = apply_guidance(final, early, GuidanceConfig(mode="adaptive", omega_cap=2))
    torch.testing.assert_close(actual, final + 1.5 * (final - early))


def test_adaptive_strength_is_per_token_and_per_batch():
    final = torch.tensor(
        [[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[math.log(4), math.log(2), 0.0], [0.0, 0.0, 0.0]]]
    )
    expected = torch.tensor([[[3.0], [0.0]], [[15.0 / 7.0], [3.0]]])
    torch.testing.assert_close(adaptive_strength(final, 3), expected)


def test_nonzero_guidance_calculates_in_float32():
    final = torch.tensor([[[1.0, 2.0, -3.0]]], dtype=torch.bfloat16)
    early = torch.tensor([[[2.0, -1.0, 0.0]]], dtype=torch.bfloat16)
    result = apply_guidance(final, early, GuidanceConfig(mode="fixed", omega=0.3))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, final.float() + 0.3 * (final.float() - early.float()))


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "masked"}, {"omega": -1}, {"omega_cap": float("nan")}, {"omega": float("inf")}, {"early_loop": 0}],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GuidanceConfig(**kwargs)


def test_active_guidance_requires_aligned_readouts():
    cfg = GuidanceConfig(mode="fixed")
    with pytest.raises(ValueError, match="requires early"):
        apply_guidance(torch.zeros(1, 2, 3), None, cfg)
    with pytest.raises(ValueError, match="identical shapes"):
        apply_guidance(torch.zeros(1, 2, 3), torch.zeros(1, 1, 3), cfg)
    with pytest.raises(ValueError, match="at least two"):
        adaptive_strength(torch.zeros(1, 2, 1), 1)
