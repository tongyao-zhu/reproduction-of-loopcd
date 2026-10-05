"""Real native Qwen3 CPU fixtures; no pretrained model or generated code."""
import unittest
from dataclasses import replace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.masking_utils import create_causal_mask

from loopcd_repro.qwen_loop import QwenLoopConfig, QwenLoopCache, forward_loop


def model_fixture(dtype=torch.float32):
    torch.manual_seed(812)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, attention_dropout=0.0,
        tie_word_embeddings=True, use_sliding_window=False)
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config).to(dtype=dtype).eval()


@torch.no_grad()
def full_prefix_oracle(model, ids, config):
    """Separate no-cache reference using native mask and explicit state list.

    Does not call adapter helpers, logical slots or its cache implementation.
    """
    net = model.model
    state = net.embed_tokens(ids)
    positions = torch.arange(ids.shape[1])
    mask = create_causal_mask(config=net.config, input_embeds=state,
        attention_mask=None, cache_position=positions, past_key_values=None,
        position_ids=positions[None])
    positional = net.rotary_emb(state, positions[None])

    def apply(indices, value):
        for i in indices:
            value = net.layers[i](value, attention_mask=mask, position_ids=positions[None],
                past_key_value=None, use_cache=False, cache_position=positions,
                position_embeddings=positional)
        return value

    states = [apply(range(15), state)]
    for _ in range(config.loops):
        transformed = apply(range(15, 19), states[-1])
        states.append((1 - config.damping) * states[-1] + config.damping * transformed)
    strong = model.lm_head(net.norm(apply(range(19, 36), states[-1]))).float()
    weak = model.lm_head(net.norm(apply(range(19, 36), states[config.reference]))).float()
    return strong, weak


class QwenLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = model_fixture()
        cls.ids = torch.tensor([[1, 3, 8, 2, 16, 21, 6, 19, 4]], dtype=torch.long)

    def test_native_r1(self):
        cfg = QwenLoopConfig(loops=1, damping=1, guided=False)
        with torch.no_grad():
            native = self.model(self.ids, use_cache=False).logits.float()
        observed = forward_loop(self.model, self.ids, cfg, use_cache=False, last_token_only=False)
        torch.testing.assert_close(observed.logits, native, atol=2e-6, rtol=2e-5)
        self.assertEqual(observed.layer_calls, dict(prelude=15, core=4, strong_tail=17, reference_tail=0))

    def test_full_prefix_formula_and_reference(self):
        cfg = QwenLoopConfig()
        strong, weak = full_prefix_oracle(self.model, self.ids, cfg)
        out = forward_loop(self.model, self.ids, cfg, use_cache=False, last_token_only=False)
        for a, b in [(out.strong_logits,strong),(out.reference_logits,weak),(out.logits,strong+.3*(strong-weak))]:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        self.assertEqual(out.layer_calls, dict(prelude=15, core=32, strong_tail=17, reference_tail=17))
        self.assertFalse(torch.equal(strong, weak))

    def test_incremental_and_multi_token_chunks(self):
        for sizes in [(3,1,1,1,1,1,1), (4,2,3)]:
            cache = None
            consumed = 0
            for size in sizes:
                out = forward_loop(self.model, self.ids[:, consumed:consumed+size], cache=cache, last_token_only=False)
                consumed += size; cache = out.cache
                s, w = full_prefix_oracle(self.model, self.ids[:, :consumed], QwenLoopConfig())
                torch.testing.assert_close(out.logits, (s+.3*(s-w))[:, -size:], atol=3e-6, rtol=3e-5)
                self.assertEqual(cache.length, consumed)
                self.assertEqual(len(cache.kv.layers), 81)
                self.assertTrue(all(cache.kv.get_seq_length(i) == consumed for i in range(81)))

    def test_zero_guidance_and_no_strong_cache_mutation(self):
        base = QwenLoopConfig(guided=False)
        zero = QwenLoopConfig(omega=0)
        a = forward_loop(self.model, self.ids[:,:4], base)
        b = forward_loop(self.model, self.ids[:,:4], zero)
        torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
        for i in range(64):
            torch.testing.assert_close(a.cache.kv.layers[i].keys, b.cache.kv.layers[i].keys, atol=0, rtol=0)
            torch.testing.assert_close(a.cache.kv.layers[i].values, b.cache.kv.layers[i].values, atol=0, rtol=0)
        a2 = forward_loop(self.model,self.ids[:,4:],base,cache=a.cache)
        b2 = forward_loop(self.model,self.ids[:,4:],zero,cache=b.cache)
        torch.testing.assert_close(a2.logits,b2.logits,atol=0,rtol=0)
        self.assertNotEqual(b2.cache.kv.layers[47].keys.data_ptr(), b2.cache.kv.layers[64].keys.data_ptr())
        self.assertNotEqual(b2.cache.kv.layers[15].keys.data_ptr(), b2.cache.kv.layers[19].keys.data_ptr())

    def test_cache_guards(self):
        fresh=QwenLoopCache(self.model,QwenLoopConfig())
        fresh.validate(self.model,QwenLoopConfig())
        self.assertEqual(len(fresh.kv.layers),1)
        self.assertEqual(fresh.kv.get_seq_length(),0)
        out=forward_loop(self.model,self.ids[:,:3])
        with self.assertRaises(ValueError):forward_loop(self.model,self.ids[:,3:4],replace(QwenLoopConfig(),omega=.2),cache=out.cache)
        with self.assertRaises(ValueError):forward_loop(model_fixture(),self.ids[:,3:4],cache=out.cache)
        with self.assertRaises(ValueError):forward_loop(self.model,self.ids[:,3:4],cache=out.cache,use_cache=False)
        out.cache.kv.layers[16].crop(1)
        with self.assertRaises(ValueError):forward_loop(self.model,self.ids[:,3:4],cache=out.cache)

    def test_interrupted_forward_poisoned(self):
        out=forward_loop(self.model,self.ids[:,:3])
        def fail(*args):raise RuntimeError('deliberate fixture failure')
        handle=self.model.model.layers[18].register_forward_pre_hook(fail)
        try:
            with self.assertRaises(RuntimeError):forward_loop(self.model,self.ids[:,3:4],cache=out.cache)
        finally:handle.remove()
        self.assertTrue(out.cache.poisoned)
        with self.assertRaises(ValueError):forward_loop(self.model,self.ids[:,3:4],cache=out.cache)

    def test_bf16_and_long_chunk(self):
        model=model_fixture(torch.bfloat16)
        ids=torch.arange(65)[None] % 97
        cfg=QwenLoopConfig()
        whole=forward_loop(model,ids,cfg,use_cache=False)
        prefix=forward_loop(model,ids[:,:64],cfg)
        final=forward_loop(model,ids[:,64:],cfg,cache=prefix.cache)
        torch.testing.assert_close(whole.logits,final.logits,atol=.015,rtol=.025)
        s,w=full_prefix_oracle(model,ids,cfg)
        torch.testing.assert_close(whole.logits,(s+.3*(s-w))[:,-1:],atol=.015,rtol=.025)
        self.assertTrue(torch.isfinite(whole.logits).all())

    def test_no_weights_indices_or_rng_changed(self):
        before={n:p.clone() for n,p in self.model.state_dict().items()}
        rng=torch.random.get_rng_state().clone()
        forward_loop(self.model,self.ids)
        for n,p in self.model.state_dict().items():torch.testing.assert_close(p,before[n],atol=0,rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        self.assertEqual([x.self_attn.layer_idx for x in self.model.model.layers],list(range(36)))

    def test_invalid_scope(self):
        for kwargs in [dict(loops=0),dict(reference=9),dict(damping=0),dict(omega=float('nan'))]:
            with self.assertRaises(ValueError):QwenLoopConfig(**kwargs)
        for ids in [self.ids.repeat(2,1),self.ids[:,:0],self.ids.float()]:
            with self.assertRaises(ValueError):forward_loop(self.model,ids)


if __name__=='__main__':unittest.main()
