"""Experimental eager/TP1 Ouro LoopCD adapter for pinned vLLM 0.13.0.

Native recurrence, weights and Attention objects remain unchanged. Packing
the two normalized states along the feature dimension lets vLLM select the
same token rows for both heads, including chunked prefill and decode. This
module is not authorized for benchmark migration merely by being importable.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from vllm.model_executor.models import ouro

from .guidance import GuidanceConfig, apply_guidance

NATIVE_SHA = '93e1c32b50d31e12ac41236a327490b28747635b31322b93fa9f3eea4ba127ab'


class LoopCDOuroForCausalLM(ouro.OuroForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        if hashlib.sha256(Path(ouro.__file__).read_bytes()).hexdigest() != NATIVE_SHA:
            raise ValueError('Unvalidated native Ouro implementation')
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError('Only TP1/PP1 validated scope')
        if not vllm_config.model_config.enforce_eager:
            raise ValueError('CUDA graphs/compilation not validated')
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
            raise ValueError('Quantization/LoRA/speculation not validated')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.model.total_ut_steps != 4:
            raise ValueError('Only fixed R4 supported')
        self.settings = GuidanceConfig(**json.loads(os.environ['LOOPCD_VLLM_SETTINGS']))
        if self.settings.early_loop != 1:
            raise ValueError('Only h1 supported')
        self.trace = Path(os.environ['LOOPCD_VLLM_GATE']) if os.environ.get('LOOPCD_VLLM_GATE') else None
        self._trace_index = 0
        self._control = None

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError('Intermediate pipeline tensors not supported')
        # Test-only control is read synchronously between completed generate
        # calls. Production settings are immutable and do not use this path.
        self._control = None
        config = self.settings
        if self.trace and (self.trace / 'control.json').exists():
            self._control = json.loads((self.trace / 'control.json').read_text())
            config = GuidanceConfig(**self._control['guidance'])
            if config.early_loop != 1:
                raise ValueError('Only h1 supported')
        self._active_config = config
        self._native = bool(self._control and self._control.get('native'))
        if self._native or not config.enabled:
            result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
            self._norm_calls = None
        else:
            observed = []
            def capture(module, args, output):
                # The subsequent fused layers can mutate their input/residual.
                # Clone only h1; the final state comes from native forward.
                observed.append(output[0].clone() if not observed else None)
            handle = self.model.norm.register_forward_hook(capture)
            try:
                final = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
            finally:
                handle.remove()
            if len(observed) != 4 or observed[0].shape != final.shape:
                raise RuntimeError('Unexpected native recurrence/normalized state')
            self._norm_calls = len(observed)
            result = torch.cat((final, observed[0]), dim=-1)
        if self._control:
            self._positions = positions.detach().cpu().tolist()
            self._input_ids = None if input_ids is None else input_ids.detach().cpu().tolist()
        return result

    def compute_logits(self, hidden_states):
        config = self._active_config
        early = None
        if self._native or not config.enabled:
            final = super().compute_logits(hidden_states)
            guided = final
        else:
            if hidden_states.shape[-1] != 2 * self.config.hidden_size:
                raise RuntimeError('Runner did not preserve packed states')
            final_state, early_state = hidden_states.split(self.config.hidden_size, dim=-1)
            final = super().compute_logits(final_state.contiguous())
            early = super().compute_logits(early_state.contiguous())
            guided = apply_guidance(final, early, config)
        if self._control:
            label = self._control['label']
            if not label.replace('_', '').isalnum():
                raise ValueError('Invalid trace label')
            dest = self.trace / (label + '__' + str(self._trace_index).zfill(5))
            self._trace_index += 1
            arrays = {'final': final, 'guided': guided}
            if early is not None:
                arrays['early'] = early
            with dest.with_suffix('.npz').open('xb') as stream:
                np.savez(stream, **{k: v.detach().float().cpu().numpy() for k, v in arrays.items()})
            slots = []
            for layer in self.model.layers:
                for attn in layer.self_attn.attn:
                    cache = attn.kv_cache
                    tensors = cache if isinstance(cache, (list, tuple)) else [cache]
                    slots.append({'object': id(attn), 'prefix': attn.layer_name,
                                  'cache': [{'ptr': t.data_ptr(), 'shape': list(t.shape), 'numel': t.numel(), 'element_size': t.element_size()} for t in tensors if torch.is_tensor(t)]})
            with dest.with_suffix('.json').open('x') as stream:
                json.dump({'control': self._control, 'positions': self._positions,
                           'input_ids': self._input_ids, 'norm_calls': self._norm_calls,
                           'slots': slots}, stream)
        return guided
