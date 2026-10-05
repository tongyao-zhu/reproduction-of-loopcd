"""Independent causal toy oracle for the cache-free Parcae adapter."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from loopcd_repro.parcae import ParcaeGuidance, ParcaeGuidanceConfig


class Coda(nn.Module):
    def forward(self, hidden, freqs, mask, *, past_key_values, step_idx, ve):
        assert mask is None and past_key_values is None
        return torch.tanh(hidden + hidden.cumsum(1) * 0.03 + ve * 0.1 + int(step_idx) * 0.01)


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_size=32, use_fused_head="pytorch",
                                      init=SimpleNamespace(logit_scale=1.7), logit_softcap=0.8)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(13, 6), "prelude": nn.ModuleList([nn.Identity()]),
            "core_block": nn.ModuleList([nn.Identity(), nn.Identity()]),
            "C": nn.Linear(6, 6), "coda": nn.ModuleList([Coda(), Coda()]),
            "ln_f": nn.LayerNorm(6),
        })
        self.lm_head = nn.Linear(6, 13)
        self.value_embeds = nn.ModuleDict({"3": nn.Embedding(13, 6), "4": nn.Embedding(13, 6)})
        self.register_buffer("freqs_cis", torch.zeros(1, 32))
        self.states = []
        self.last_output = None

    def initialize_state(self, inputs):
        return torch.randn_like(inputs) * 0.2

    def core_block_forward(self, hidden, inputs, *, step):
        # The complete iteration modifies the last physical-layer result.
        # Observing core_block[-1] instead of this method would be wrong.
        for layer in self.transformer.core_block:
            hidden = layer(hidden)
        return torch.tanh(hidden + inputs * 0.2 + (int(step) + 1) * 0.07)

    def oracle_readout(self, hidden, tokens, depth):
        value = nn.functional.linear(hidden, self.transformer.C.weight, self.transformer.C.bias)
        for index in range(2):
            ve = self.value_embeds[str(3 + index)].weight[tokens]
            value = torch.tanh(value + value.cumsum(1) * 0.03 + ve * 0.1 + (1 + 2 * depth + index) * 0.01)
        value = nn.functional.layer_norm(value, (6,), self.transformer.ln_f.weight,
                                         self.transformer.ln_f.bias, self.transformer.ln_f.eps)
        logits = nn.functional.linear(value, self.lm_head.weight, self.lm_head.bias).float() * 1.7
        return 0.8 * torch.tanh(logits / 0.8)

    def forward_for_generation(self, input_ids, *, num_steps, past_key_values):
        assert past_key_values is None
        inputs = self.transformer.wte(input_ids)
        state = self.initialize_state(inputs)
        self.states = []
        for step in range(num_steps):
            state = self.core_block_forward(state, inputs, step=torch.tensor(step))
            self.states.append(state.detach().clone())
        state = self.transformer.C(state)
        for i, layer in enumerate(self.transformer.coda):
            state = layer(state, self.freqs_cis[:, :input_ids.shape[1]], None,
                          past_key_values=None, step_idx=torch.tensor(1 + 2 * num_steps + i),
                          ve=self.value_embeds[str(3 + i)](input_ids))
        logits = self.lm_head(self.transformer.ln_f(state)).float() * 1.7
        self.last_output = {"logits": 0.8 * torch.tanh(logits / 0.8), "past_key_values": None}
        return self.last_output


class ParcaeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = Native().eval()
        self.tokens = torch.tensor([[4, 2, 9, 1]])

    def test_zero_identity_and_rng(self):
        for mode in ("baseline", "fixed", "adaptive", "hidden"):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                native = self.model.forward_for_generation(self.tokens, num_steps=4, past_key_values=None)["logits"]
                rng = torch.get_rng_state()
                torch.manual_seed(42)
                adapter = ParcaeGuidance(self.model, ParcaeGuidanceConfig(mode=mode, total_loops=4, omega=0, omega_cap=0))
                output = adapter(self.tokens)
                self.assertIs(output, self.model.last_output)
                torch.testing.assert_close(output["logits"], native, atol=0, rtol=0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(adapter.last_observation["readout_passes"], 1)

    def test_independent_formulas_every_position_depth_and_rng(self):
        for depth in (4, 8):
            for reference in (1, depth):
                for mode in ("fixed", "adaptive", "hidden"):
                    with self.subTest(depth=depth, reference=reference, mode=mode):
                        torch.manual_seed(42)
                        native = self.model.forward_for_generation(self.tokens, num_steps=depth, past_key_values=None)["logits"].clone()
                        rng = torch.get_rng_state()
                        early, final = self.model.states[reference - 1], self.model.states[-1]
                        if mode == "hidden":
                            expected = self.model.oracle_readout(final + 0.75 * (final - early), self.tokens, depth)
                        else:
                            early_logits = self.model.oracle_readout(early, self.tokens, depth)
                            p = torch.softmax(native.double(), dim=-1).sort(descending=True).values
                            strength = 1.3 * (1 - p[..., :1] + p[..., 1:2]) if mode == "adaptive" else 0.75
                            expected = native.double() + strength * (native.double() - early_logits.double())
                        counts = {"C": 0, "head": 0, "coda": 0}
                        def count(key):
                            def hook(*args):
                                counts[key] += 1
                            return hook
                        handles = [self.model.transformer.C.register_forward_hook(count("C")),
                                   self.model.lm_head.register_forward_hook(count("head"))]
                        handles.extend(layer.register_forward_hook(count("coda")) for layer in self.model.transformer.coda)
                        torch.manual_seed(42)
                        adapter = ParcaeGuidance(self.model, ParcaeGuidanceConfig(mode, depth, reference, 0.75, 1.3))
                        output = adapter(self.tokens)["logits"]
                        for handle in handles:
                            handle.remove()
                        torch.testing.assert_close(output.double(), expected.double(), atol=2e-6, rtol=2e-6)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        readouts = 1 if mode == "hidden" else 2
                        self.assertEqual(counts, {"C": readouts, "head": readouts, "coda": readouts * 2})
                        self.assertNotIn("core_block_forward", self.model.__dict__)
                        self.assertFalse(self.model.transformer.C._forward_pre_hooks)
                        self.assertFalse(hasattr(self.model, "_loopcd_parcae_active"))

    def test_rejects_cache_mask_training_empty_long_and_batch(self):
        adapter = ParcaeGuidance(self.model)
        for kwargs in ({"past_key_values": object()}, {"attention_mask": torch.ones_like(self.tokens)}):
            with self.assertRaises(ValueError):
                adapter(self.tokens, **kwargs)
        for tokens in (self.tokens.repeat(2, 1), self.tokens[:, :0], torch.ones(1, 33, dtype=torch.long), self.tokens.float()):
            with self.assertRaises(ValueError):
                adapter(tokens)
        self.model.train()
        with self.assertRaises(ValueError):
            adapter(self.tokens)

    def test_restores_after_native_failure_and_preserves_existing_method(self):
        original = self.model.core_block_forward
        def broken(*args, **kwargs):
            raise RuntimeError("test failure")
        self.model.core_block_forward = broken
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            ParcaeGuidance(self.model, ParcaeGuidanceConfig(mode="hidden"))(self.tokens)
        self.assertIs(self.model.core_block_forward, broken)
        self.assertFalse(self.model.transformer.C._forward_pre_hooks)
        self.assertFalse(hasattr(self.model, "_loopcd_parcae_active"))
        self.model.core_block_forward = original

    def test_invalid_configs(self):
        for kwargs in ({"total_loops": True}, {"reference_loop": 0}, {"reference_loop": 9},
                       {"omega": float("nan")}, {"omega_cap": -1}, {"mode": "other"}):
            with self.assertRaises((ValueError, TypeError)):
                ParcaeGuidanceConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
