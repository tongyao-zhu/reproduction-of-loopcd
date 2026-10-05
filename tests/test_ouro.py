"""CPU contract tests using an independently implemented recurrent model.

The fake layer uses causal prefix sums and a distinct cache slot per loop,
so cached/full comparisons detect both wrong recurrence and extra updates.
These checks complement, rather than replace, a real Ouro GPU gate.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from loopcd_repro import GuidanceConfig, OuroGuidance, apply_guidance


class FakeCache:
    def __init__(self):
        self.prefix = {}
        self.updates = {}


class CausalLayer(nn.Module):
    def forward(self, hidden, *, current_ut, past_key_value, use_cache):
        prefix = hidden.cumsum(dim=1)
        if use_cache:
            previous = past_key_value.prefix.get(current_ut, torch.zeros_like(hidden[:, :1]))
            prefix = prefix + previous
            past_key_value.prefix[current_ut] = prefix[:, -1:].clone()
            past_key_value.updates[current_ut] = past_key_value.updates.get(current_ut, 0) + 1
        return hidden + prefix * (0.1 * (current_ut + 1))


class NativeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_ut_steps = 4
        self.embed_tokens = nn.Embedding(11, 5)
        self.layers = nn.ModuleList([CausalLayer()])
        self.norm = nn.LayerNorm(5)
        self.last_states = None

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False, **kwargs):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = FakeCache()
        states, gates = [], []
        for current_ut in range(self.total_ut_steps):
            hidden = self.layers[0](
                hidden, current_ut=current_ut, past_key_value=past_key_values, use_cache=use_cache
            )
            hidden = self.norm(hidden)
            states.append(hidden)
            gates.append(hidden[..., :1] * 0)
        self.last_states = states
        output = SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)
        return output, states, gates


class CountingHead(nn.Linear):
    def __init__(self):
        super().__init__(5, 7, bias=False)
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return super().forward(hidden)


class NativeOuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(total_ut_steps=4, early_exit_threshold=0.8)
        self.model = NativeInner()
        self.lm_head = CountingHead()
        self.early_exit_step = 2
        self.early_exit_threshold = 0.8
        self.last_native_output = None
        self.last_native_logits = None

    def forward(self, input_ids=None, logits_to_keep=0, return_dict=True, exit_at_step=None, **kwargs):
        output, states, gates = self.model(input_ids=input_ids, **kwargs)
        selected = states[self.early_exit_step if exit_at_step is None else exit_at_step]
        if isinstance(logits_to_keep, int):
            selected = selected[:, -logits_to_keep:]
        else:
            selected = selected.index_select(1, logits_to_keep.to(selected.device))
        logits = self.lm_head(selected)
        self.last_native_logits = logits
        result = SimpleNamespace(logits=logits, past_key_values=output.past_key_values, marker="native")
        if not return_dict:
            result = (logits, output.past_key_values, "native")
        self.last_native_output = result
        return result


@pytest.fixture
def native():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        return NativeOuro().eval()


@pytest.mark.parametrize(
    "config",
    [GuidanceConfig(), GuidanceConfig(mode="fixed", omega=0), GuidanceConfig(mode="adaptive", omega_cap=0)],
)
def test_zero_guidance_exact_native_identity_and_one_head_call(native, config):
    tokens = torch.tensor([[1, 2, 3]])
    with OuroGuidance(native, config) as adapter:
        output = native(tokens)
        assert output is native.last_native_output
        assert output.logits is native.last_native_logits
        assert native.lm_head.calls == 1
        assert adapter.last_observation["extra_lm_head_calls"] == 0
        assert native.early_exit_step == 3
        assert not native.model._forward_hooks
        assert not native.model.norm._forward_hooks


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("positions", [0, 2, torch.tensor([2, 0])])
def test_guides_all_requested_positions_with_one_extra_head(native, mode, positions):
    config = GuidanceConfig(mode=mode, omega=1.3, omega_cap=2.5)
    with OuroGuidance(native, config) as adapter:
        output = native(torch.tensor([[1, 2, 3], [4, 5, 6]]), logits_to_keep=positions)
        assert native.lm_head.calls == 2
        early = native.model.last_states[0]
        early = early[:, -positions:] if isinstance(positions, int) else early.index_select(1, positions)
        early_logits = F.linear(early, native.lm_head.weight)
        expected = apply_guidance(native.last_native_logits, early_logits, config)
        torch.testing.assert_close(output.logits, expected)
        assert not torch.equal(output.logits[:, 0], native.last_native_logits[:, 0])
        assert adapter.last_observation["executed_source_indices"] == [0, 1, 2, 3]
        assert adapter.last_observation["already_normalized_identity"]
        assert not native.model._forward_hooks
        assert not native.model.norm._forward_hooks
        assert not native.model.layers[0]._forward_pre_hooks


@pytest.mark.parametrize("mode", ["baseline", "fixed", "adaptive"])
def test_cached_decoding_equals_full_prefix_and_preserves_cache_updates(native, mode):
    tokens = torch.tensor([[1, 2, 3, 4]])
    settings = GuidanceConfig(mode=mode, omega=0.7, omega_cap=1.2)
    with torch.no_grad(), OuroGuidance(native, settings):
        complete = native(tokens, use_cache=False).logits
        prefill = native(tokens[:, :2], use_cache=True)
        cache = prefill.past_key_values
        pieces = [prefill.logits]
        for index in range(2, tokens.shape[1]):
            decoded = native(tokens[:, index:index + 1], use_cache=True, past_key_values=cache)
            assert decoded.past_key_values is cache
            pieces.append(decoded.logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), complete, atol=2e-6, rtol=2e-6)
        assert cache.updates == {0: 3, 1: 3, 2: 3, 3: 3}


def test_context_restores_instance_and_config_attributes_after_exception(native):
    original_forward = native.forward
    assert not hasattr(native.config, "early_exit_step")
    with pytest.raises(RuntimeError, match="intentional"):
        with OuroGuidance(native, GuidanceConfig(mode="fixed")).installed():
            assert native.config.early_exit_step == 3
            raise RuntimeError("intentional")
    assert native.forward == original_forward
    assert "forward" not in native.__dict__
    assert native.early_exit_step == 2
    assert native.early_exit_threshold == 0.8
    assert native.config.early_exit_threshold == 0.8
    assert not hasattr(native.config, "early_exit_step")
    assert not hasattr(native, "_loopcd_repro_guidance")


def test_tuple_outputs_preserve_native_auxiliary_values(native):
    with OuroGuidance(native, GuidanceConfig(mode="fixed")):
        result = native(torch.tensor([[1, 2]]), return_dict=False, use_cache=True)
        assert isinstance(result, tuple)
        assert result[1] is native.last_native_output[1]
        assert result[2] == "native"
        assert not torch.equal(result[0], native.last_native_logits)


@pytest.mark.parametrize(
    "kwargs",
    [{"labels": torch.tensor([[1, 2]])}, {"exit_at_step": 0}, {"exit_threshold": 0.5}, {"use_weighted_exit": True}],
)
def test_rejects_training_or_nonfinal_exit_without_running_native(native, kwargs):
    with OuroGuidance(native, GuidanceConfig(mode="fixed")):
        with pytest.raises(ValueError):
            native(torch.tensor([[1, 2]]), **kwargs)
    assert native.lm_head.calls == 0


def test_rejects_wrong_loop_budget_and_nested_context(native):
    with pytest.raises(ValueError, match="must match"):
        OuroGuidance(native, total_loops=3)
    with OuroGuidance(native):
        with pytest.raises(RuntimeError, match="already active"):
            with OuroGuidance(native):
                pass
        assert native.early_exit_step == 3


def test_observation_hooks_removed_after_native_error(native):
    def explode(*args, **kwargs):
        raise RuntimeError("native failed")

    native.model.layers[0].forward = explode
    with OuroGuidance(native, GuidanceConfig(mode="fixed")):
        with pytest.raises(RuntimeError, match="native failed"):
            native(torch.tensor([[1, 2]]))
        assert not native.model._forward_hooks
        assert not native.model.norm._forward_hooks
        assert not native.model.layers[0]._forward_pre_hooks
