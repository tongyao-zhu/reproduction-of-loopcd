"""Independently reconstructed Qwen loop, with logical-depth KV slots.

See docs/qwen_mbpp_reconstruction.md: this cache policy is a preregistered
assumption, not a claim about Apple's unpublished wrapper. No monkey-patching
of native model/layers; no weights copied or changed. Batch-one unpadded only.
"""
from dataclasses import dataclass
import math

import torch
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class QwenLoopConfig:
    loops: int = 8
    damping: float = 0.125
    reference: int = 1
    omega: float = 0.3
    guided: bool = True

    def __post_init__(self):
        if type(self.loops) is not int or not 1 <= self.loops <= 8:
            raise ValueError('Invalid loop count')
        if type(self.reference) is not int or not 1 <= self.reference <= self.loops:
            raise ValueError('Invalid reference')
        if not math.isfinite(self.damping) or not 0 < self.damping <= 1:
            raise ValueError('Invalid damping')
        if not math.isfinite(self.omega) or self.omega < 0 or type(self.guided) is not bool:
            raise ValueError('Invalid guidance')

    @property
    def slots(self):
        return 15 + self.loops * 4 + 17 * (1 + int(self.guided))


class QwenLoopCache:
    def __init__(self, model, config):
        self.owner = model
        self.config = config
        self.kv = DynamicCache()
        self.length = 0
        self.poisoned = False

    def validate(self, model, config):
        if self.owner is not model or self.config != config or self.poisoned:
            raise ValueError('Cache owner/config changed or interrupted forward')
        # Pinned Transformers 4.54.1 creates one empty DynamicLayer eagerly.
        expected = config.slots if self.length else 1
        if len(self.kv.layers) != expected:
            raise ValueError('Unexpected cache slot inventory')
        if any(self.kv.get_seq_length(i) != self.length for i in range(expected)):
            raise ValueError('Unequal logical cache lengths')


class _SlotView:
    """Native Qwen attention writes its physical index into one logical slot."""
    def __init__(self, cache, physical, logical):
        self.cache, self.physical, self.logical = cache, physical, logical
        self.writes = 0

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        if layer_idx != self.physical or self.writes:
            raise ValueError('Unexpected layer index or repeated KV append')
        self.writes += 1
        return self.cache.update(keys, values, self.logical, cache_kwargs)


@dataclass
class QwenLoopOutput:
    logits: torch.Tensor
    strong_logits: torch.Tensor
    reference_logits: object
    cache: object
    layer_calls: dict


@torch.no_grad()
def forward_loop(model, input_ids, config=QwenLoopConfig(), *, cache=None,
                 use_cache=True, last_token_only=True):
    """One prefill or incremental chunk; fresh cache per problem and arm.

    Hidden arithmetic uses two dtype-preserving multiplications and addition:
    (1-damping)*u + damping*g(u). Guidance alone is computed in FP32.
    Cache failure poisons the state rather than allowing an unsafe retry.
    """
    if model.training or model.config.model_type != 'qwen3' or len(model.model.layers) != 36:
        raise ValueError('An eval-mode, 36-layer dense Qwen3 is required')
    if model.config._attn_implementation != 'sdpa' or model.config.use_sliding_window:
        raise ValueError('Only full-attention SDPA is validated')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1 or input_ids.dtype != torch.long:
        raise ValueError('Only nonempty, unpadded batch-one token IDs are supported')
    if not use_cache and cache is not None:
        raise ValueError('Cannot supply past state to a no-cache forward')
    if use_cache:
        cache = cache if cache is not None else QwenLoopCache(model, config)
        cache.validate(model, config)
    start = cache.length if cache is not None else 0
    end = start + input_ids.shape[1]
    if end > model.config.max_position_embeddings:
        raise ValueError('Context budget exceeded')
    inner = model.model
    hidden = inner.embed_tokens(input_ids)
    position = torch.arange(start, end, device=hidden.device)
    position_ids = position.unsqueeze(0)
    allowed = torch.arange(end, device=hidden.device)[None, :] <= position[:, None]
    mask = torch.zeros((input_ids.shape[1], end), device=hidden.device, dtype=hidden.dtype)
    mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
    mask = mask[None, None]
    rope = inner.rotary_emb(hidden, position_ids)
    calls = {'prelude': 0, 'core': 0, 'strong_tail': 0, 'reference_tail': 0}
    if cache is not None:
        cache.poisoned = True

    def layer(physical, logical, state, group):
        view = _SlotView(cache.kv, physical, logical) if cache is not None else None
        output = inner.layers[physical](state, attention_mask=mask, position_ids=position_ids,
            past_key_value=view, use_cache=use_cache, cache_position=position, position_embeddings=rope)
        if view is not None and view.writes != 1:
            raise ValueError('Native layer did not write exactly once')
        calls[group] += 1
        return output

    for i in range(15):
        hidden = layer(i, i, hidden, 'prelude')
    reference = None
    for step in range(config.loops):
        previous = hidden
        for offset in range(4):
            hidden = layer(15 + offset, 15 + step * 4 + offset, hidden, 'core')
        hidden = (1 - config.damping) * previous + config.damping * hidden
        if step + 1 == config.reference:
            reference = hidden

    def readout(state, offset, group):
        for i in range(17):
            state = layer(19 + i, offset + i, state, group)
        state = inner.norm(state[:, -1:] if last_token_only else state)
        return model.lm_head(state).float()

    strong = readout(hidden, 15 + 4 * config.loops, 'strong_tail')
    weak = readout(reference, 32 + 4 * config.loops, 'reference_tail') if config.guided else None
    logits = strong + config.omega * (strong - weak) if config.guided else strong
    if cache is not None:
        if len(cache.kv.layers) != config.slots or any(cache.kv.get_seq_length(i) != end for i in range(config.slots)):
            raise ValueError('Incomplete or duplicated logical cache update')
        cache.length = end
        cache.poisoned = False
    return QwenLoopOutput(logits, strong, weak, cache, calls)
