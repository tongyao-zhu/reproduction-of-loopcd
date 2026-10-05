"""Huginn inference contracts, optionally repeated on a tiny native Raven.

Set LOOPCD_HUGINN_TEST_MODEL to the prepared private model directory to
load only its Python/config classes and construct tiny random CPU weights.
No checkpoint weights or GPU memory are used by these tests.
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from loopcd_repro.huginn import HuginnHiddenConfig, HuginnHiddenGuidance


class HuginnDynamicCache:
    def __init__(self, lookup_strategy="full"):
        self.lookup_strategy = lookup_strategy
        self.key_cache, self.value_cache = {}, {}
        self._seen_tokens = 0

    def get_seq_length(self):
        return self._seen_tokens


class FakeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, hidden, frequencies, step_idx, mask, cache):
        index = int(step_idx)
        prefix = hidden.cumsum(dim=1)
        if cache is not None:
            slot = cache.key_cache.setdefault(index, {})
            if slot:
                prefix = prefix + torch.stack(list(slot.values()), dim=1).sum(dim=1, keepdim=True)
            offset = len(slot)
            for pos in range(hidden.shape[1]):
                assert offset + pos not in slot
                slot[offset + pos] = hidden[:, pos].detach().clone()
            if index == 0:
                cache._seen_tokens += hidden.shape[1]
        return hidden * 0.9 + torch.tanh(self.proj(prefix)) * 0.3


class FakeHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(mean_recurrence=32, pad_token_id=30, test_time_noise=0)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(31, 8), "prelude": nn.ModuleList([FakeBlock(8), FakeBlock(8)]),
            "adapter": nn.Linear(16, 8), "core_block": nn.ModuleList([FakeBlock(8), FakeBlock(8)]),
            "coda": nn.ModuleList([FakeBlock(8), FakeBlock(8)]), "ln_f": nn.LayerNorm(8),
        })
        self.lm_head = nn.Linear(8, 31, bias=False)
        self.register_buffer("freqs_cis", torch.ones(1, 128, 1))
        self.last_native_output = None

    def initialize_state(self, embeddings, scale=1.0):
        return torch.randn_like(embeddings) * scale * 0.2

    def iterate_forward(self, embeddings, input_states, num_steps, cache):
        hidden = self.initialize_state(embeddings) if input_states is None else input_states.clone()
        step_idx = 1
        for _ in range(num_steps):
            hidden = self.transformer.adapter(torch.cat((hidden, embeddings), dim=-1))
            for block in self.transformer.core_block:
                step_idx += 1
                hidden = block(hidden, self.freqs_cis, step_idx, None, cache)
        return self.transformer.ln_f(hidden)

    def forward(self, input_ids, input_embeds=None, input_states=None, num_steps=None,
                use_cache=False, past_key_values=None, **kwargs):
        embeddings = self.transformer.wte(input_ids) if input_embeds is None else input_embeds
        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()
        for index, layer in enumerate(self.transformer.prelude):
            embeddings = layer(embeddings, self.freqs_cis, index, None, past_key_values)
        hidden = self.iterate_forward(embeddings, input_states, num_steps or self.config.mean_recurrence, past_key_values)
        latents = hidden.detach().clone()
        for index, layer in enumerate(self.transformer.coda):
            hidden = layer(hidden, self.freqs_cis, -(index + 1), None, past_key_values)
        logits = self.lm_head(self.transformer.ln_f(hidden)).float()
        self.last_native_output = SimpleNamespace(logits=logits, latent_states=latents, past_key_values=past_key_values)
        return self.last_native_output


@pytest.fixture(params=["fake", "native"])
def model(request):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        if request.param == "fake":
            return FakeHuginn().eval()
        model_path = os.environ.get("LOOPCD_HUGINN_TEST_MODEL")
        if not model_path:
            pytest.skip("Set LOOPCD_HUGINN_TEST_MODEL to test tiny native Raven too")
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", model_path,
                                           local_files_only=True)
        config = cls.config_class(n_embd=32, n_heads=4, n_layers=6, block_size=64,
                                  vocab_size=31, padding_multiple=1, intermediate_size=64,
                                  n_layers_in_prelude=2, n_layers_in_recurrent_block=2,
                                  n_layers_in_coda=2, mean_recurrence=32, mean_backprop_depth=0,
                                  pad_token_id=30, bos_token_id=1, eos_token_id=2,
                                  torch_dtype="float32")
        return cls(config).eval()


def readout(model, raw):
    hidden = model.transformer.ln_f(raw)
    for index, layer in enumerate(model.transformer.coda):
        hidden = layer(hidden, model.freqs_cis[:, :raw.shape[1]], torch.tensor(-(index + 1)), None, None)
    return model.lm_head(model.transformer.ln_f(hidden)).float()


@pytest.mark.parametrize("settings", [HuginnHiddenConfig(), HuginnHiddenConfig(mode="hidden", omega=0)])
def test_zero_identity_matches_native_random_initialization(model, settings):
    tokens = torch.tensor([[3, 4, 5]])
    original_forward = model.forward
    torch.manual_seed(991)
    expected = model(tokens, num_steps=settings.total_loops).logits
    torch.manual_seed(991)
    with HuginnHiddenGuidance(model, settings) as adapter:
        actual = model(tokens)
        assert adapter.last_observation["extra_lm_head_calls"] == 0
        if isinstance(model, FakeHuginn):
            assert actual is model.last_native_output
    assert torch.equal(actual.logits, expected)
    assert model.forward == original_forward
    assert "forward" not in model.__dict__


@pytest.mark.parametrize("depth,reference,omega", [(32, 6, .5), (16, 7, .5), (32, 7, .3), (16, 6, .3), (16, 6, .5)])
def test_raw_hidden_formula_exact_depth_and_one_readout(model, depth, reference, omega):
    tokens = torch.tensor([[3, 4, 5]])
    states = []
    counts = {"core": 0, "coda": 0, "head": 0}
    capture = model.transformer.core_block[-1].register_forward_hook(lambda m, a, out: states.append(out.clone()))
    torch.manual_seed(123)
    model(tokens, num_steps=depth)
    capture.remove()
    expected = readout(model, (states[-1].float() + omega * (states[-1].float() - states[reference - 1].float())).to(states[-1].dtype))

    def count(name):
        def hook(module, args, output):
            counts[name] += 1
        return hook
    handles = [model.transformer.core_block[-1].register_forward_hook(count("core")),
               model.transformer.coda[0].register_forward_hook(count("coda")),
               model.lm_head.register_forward_hook(count("head"))]
    torch.manual_seed(123)
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("hidden", depth, reference, omega)) as adapter:
        actual = model(tokens)
        assert adapter.last_observation["executed_physical_loops"] == list(range(1, depth + 1))
        assert adapter.last_observation["extra_coda_passes"] == 0
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
    assert counts == {"core": depth, "coda": 1, "head": 1}
    assert actual.logits.shape[:2] == tokens.shape


@pytest.mark.parametrize("mode", ["baseline", "hidden"])
def test_fixed_initial_states_cache_matches_full_prefix_and_no_double_updates(model, mode):
    settings = HuginnHiddenConfig(mode, 16, 6, .5)
    tokens = torch.tensor([[3, 4, 5, 6]])
    initial = model.initialize_state(model.transformer.wte(tokens))
    with torch.no_grad(), HuginnHiddenGuidance(model, settings):
        full = model(tokens, input_states=initial).logits
        first = model(tokens[:, :2], input_states=initial[:, :2], use_cache=True)
        cache = first.past_key_values
        pieces = [first.logits]
        for position in (2, 3):
            output = model(tokens[:, position:position + 1], input_states=initial[:, position:position + 1],
                           use_cache=True, past_key_values=cache, cache_position=torch.tensor([position]))
            assert output.past_key_values is cache
            pieces.append(output.logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=2e-5, atol=2e-5)
        assert cache.get_seq_length() == 4
        assert len(cache.key_cache) == 2 + 16 * 2 + 2
        assert all(len(slot) == 4 for slot in cache.key_cache.values())


def test_cache_cannot_cross_guidance_or_depth(model):
    tokens = torch.tensor([[3, 4]])
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("baseline", 16, 6)):
        cache = model(tokens, use_cache=True).past_key_values
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("hidden", 16, 6)):
        with pytest.raises(ValueError, match="different or unknown"):
            model(tokens[:, :1], past_key_values=cache, use_cache=True)
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("baseline", 32, 6)):
        with pytest.raises(ValueError, match="different or unknown"):
            model(tokens[:, :1], past_key_values=cache, use_cache=True)


@pytest.mark.parametrize("invalid", [
    {"num_steps": 7}, {"num_steps": [16, 16]}, {"init_scale": 0.0},
    {"labels": torch.tensor([[1, 2]])}, {"attention_mask": torch.tensor([[1, 0]])},
])
def test_rejects_silent_protocol_changes(model, invalid):
    with HuginnHiddenGuidance(model):
        with pytest.raises(ValueError):
            model(torch.tensor([[3, 4]]), **invalid)


def test_hooks_and_forward_restored_on_error(model):
    original = model.forward
    def explode(module, args):
        raise RuntimeError("intentional coda error")
    external = model.transformer.coda[0].register_forward_pre_hook(explode)
    with pytest.raises(RuntimeError, match="intentional"):
        with HuginnHiddenGuidance(model, HuginnHiddenConfig("hidden", 16, 6)):
            model(torch.tensor([[3, 4]]))
    external.remove()
    assert model.forward == original
    assert not model.transformer.core_block[-1]._forward_hooks
    assert not model.transformer.ln_f._forward_pre_hooks
    assert not model.lm_head._forward_hooks
    assert all(not layer._forward_pre_hooks for layer in model.transformer.coda)


def test_guidance_retains_native_initialization_rng_and_rejects_padding(model):
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("hidden", 16, 6)):
        before = torch.random.get_rng_state().clone()
        model(torch.tensor([[3, 4]]))
        assert not torch.equal(before, torch.random.get_rng_state())
        with pytest.raises(ValueError, match="padded"):
            model(torch.tensor([[3, 30]]))
        with pytest.raises(ValueError, match="batch size one"):
            model(torch.tensor([[3, 4], [3, 4]]))


def test_native_generation_uses_hooks_and_blocks_bypass_generators(model):
    if isinstance(model, FakeHuginn):
        pytest.skip("Generation integration uses the actual native Raven class")
    original_generate = model.generate
    with HuginnHiddenGuidance(model, HuginnHiddenConfig("hidden", 16, 6)) as adapter:
        result = model.generate(torch.tensor([[3, 4]]), max_new_tokens=2, min_new_tokens=2,
                                do_sample=False, use_cache=True, pad_token_id=30, eos_token_id=2)
        assert result.shape == (1, 4)
        assert adapter.last_observation["executed_physical_loops"] == list(range(1, 17))
        assert adapter.last_observation["lm_head_calls"] == 1
        with pytest.raises(ValueError, match="ordinary fixed-depth"):
            model.generate(torch.tensor([[3, 4]]), continuous_compute=False)
    assert model.generate == original_generate
    assert "generate" not in model.__dict__


def test_prepare_rejects_wrong_original_without_creating_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raven_modeling_minimal.py").write_text("wrong original")
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_huginn.py"
    spec = importlib.util.spec_from_file_location("prepare_huginn_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="audited original"):
        module.prepare(source, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()
