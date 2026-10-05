"""Diagnostic-only, transparent logit capture for installed vLLM Ouro.

This is not a LoopCD adapter. It calls the native compute_logits unchanged,
and writes float32 copies for fixed-prefix comparison. Eager, TP1 only.
"""
import json
import os
from pathlib import Path
import numpy as np
from vllm.model_executor.models.ouro import OuroForCausalLM


class TracedOuroForCausalLM(OuroForCausalLM):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        control = Path(os.environ['LOOPCD_VLLM_TRACE']) / 'control.json'
        if control.exists():
            label = json.loads(control.read_text())['label']
            if not label.replace('_','').isalnum():
                raise ValueError('Invalid diagnostic label')
            # The asynchronous scheduler can split one submitted batch. Preserve
            # each actual compute call, including its observed microbatch size.
            if not hasattr(self, '_loopcd_trace_counters'):
                self._loopcd_trace_counters = {}
            index = self._loopcd_trace_counters.get(label, 0)
            self._loopcd_trace_counters[label] = index + 1
            dest = control.parent / (label + '__call' + str(index).zfill(4) + '.npy')
            with dest.open('xb') as f:
                np.save(f, logits.detach().float().cpu().numpy(), allow_pickle=False)
        return logits
