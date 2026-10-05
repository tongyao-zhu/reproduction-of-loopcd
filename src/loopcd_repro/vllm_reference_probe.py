"""Test-only subclass: native R1 oracle and forced-token KV comparisons.

Never register this class in a benchmark. It overrides sampling logits only
after recording the real outputs from the unchanged LoopCD adapter.
"""
import json
from pathlib import Path

import numpy as np
import torch
from vllm.forward_context import get_forward_context

from .vllm_ouro import LoopCDOuroForCausalLM


class ReferenceProbeOuro(LoopCDOuroForCausalLM):
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        control = self.trace / 'control.json' if self.trace else None
        request = json.loads(control.read_text()) if control and control.exists() else None
        saved = self.model.total_ut_steps
        if request and request.get('native_h1'):
            if request.get('native') is not True or request['guidance']['mode'] != 'baseline':
                raise ValueError('R1 oracle must use the unmodified native path')
            self.model.total_ut_steps = 1
        try:
            result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        finally:
            self.model.total_ut_steps = saved
        self._probe_kv = None
        self._probe_slots = None
        if request:
            context = get_forward_context()
            metadata = context.attn_metadata
            if not isinstance(metadata, dict):
                raise ValueError('Only inspected single-microbatch metadata supported')
            values, names, mappings = [], [], []
            for layer in self.model.layers:
                attention = layer.self_attn.attn[:1] if request.get('native_h1') else layer.self_attn.attn
                for attn in attention:
                    meta = metadata[attn.layer_name]
                    slots = meta.slot_mapping
                    if len(slots) != len(self._positions):
                        raise ValueError('Padding or speculative slots outside gate scope')
                    slot = int(slots[-1].item())
                    caches = attn.kv_cache
                    if not isinstance(caches, list) or len(caches) != 1:
                        raise ValueError('Unexpected virtual-engine cache layout')
                    cache = caches[0]
                    if cache.ndim != 5 or cache.shape[0] != 2 or slot < 0 or cache.dtype != torch.bfloat16:
                        raise ValueError('Uninspected KV layout/dtype/slot')
                    block, offset = divmod(slot, cache.shape[2])
                    values.append(cache[:, block, offset].clone())
                    names.append(attn.layer_name)
                    mappings.append(slot)
            # Capture actual paged-cache values after the native forward writes.
            self._probe_kv = torch.stack(values).contiguous().view(torch.uint16).cpu().numpy()
            self._probe_slots = dict(names=names, slot_mapping=mappings, positions=self._positions)
        return result

    def compute_logits(self, hidden_states):
        real = super().compute_logits(hidden_states)
        if not self._control:
            return real
        label = self._control['label']
        base = self.trace / (label + '__' + str(self._trace_index - 1).zfill(5))
        with Path(str(base) + '.kv.npy').open('xb') as stream:
            np.save(stream, self._probe_kv, allow_pickle=False)
        with Path(str(base) + '.kv.json').open('x') as stream:
            json.dump(self._probe_slots, stream)
        step = self._positions[-1] - (self._control['prompt_length'] - 1)
        if step < 0:  # chunked-prefill dummy logits are not sampled
            return real
        tokens = self._control['forced_tokens']
        if real.shape[0] != 1 or not 0 <= step < len(tokens):
            raise ValueError('Only one finite forced trajectory supported')
        forced = torch.full_like(real, float('-inf'))
        forced[0, tokens[step]] = 0
        return forced


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('OuroForCausalLM', 'loopcd_repro.vllm_reference_probe:ReferenceProbeOuro')
