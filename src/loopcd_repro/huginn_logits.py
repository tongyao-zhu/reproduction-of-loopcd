"""Huginn logit guidance with separate recurrent-reference coda history.

The strong native forward is unchanged. A raw recurrent reference is read
through ln_f, every native coda layer, ln_f and the head on the same prefix.
Its negative-index coda cache is distinct from the strong model's cache.
"""
from dataclasses import dataclass
import torch
from .guidance import GuidanceConfig, apply_guidance
from .huginn import HuginnHiddenConfig, HuginnHiddenGuidance


@dataclass(frozen=True)
class HuginnLogitsConfig:
    mode: str = 'adaptive'
    total_loops: int = 16
    reference_loop: int = 1
    omega: float = .2
    omega_cap: float = .25

    def __post_init__(self):
        HuginnHiddenConfig('baseline', self.total_loops, self.reference_loop, self.omega)
        GuidanceConfig(self.mode, self.omega, self.omega_cap, self.reference_loop)

    @property
    def logits_config(self):
        return GuidanceConfig(self.mode, self.omega, self.omega_cap, self.reference_loop)

    @property
    def enabled(self):
        return self.logits_config.enabled

    @property
    def cache_contract(self):
        return ('logits', self.total_loops, self.mode if self.enabled else 'baseline',
                self.reference_loop if self.enabled else None,
                self.omega if self.enabled and self.mode=='fixed' else 0.,
                self.omega_cap if self.enabled and self.mode=='adaptive' else 0.)


class HuginnLogitsGuidance(HuginnHiddenGuidance):
    """Reuse native fixed-depth input guards/restoration, never hidden blending."""
    def __init__(self, model, config=None):
        super().__init__(model, config or HuginnLogitsConfig())

    def _forward(self, arguments):
        self.last_observation = None
        self._validate(arguments)
        if arguments.get('output_details', {}).get('return_logits') is False:
            raise ValueError('Logit guidance requires returned native logits')
        if not self.config.enabled:
            result = self._native_forward(**arguments)
            self.last_observation = dict(mode=self.config.mode, guidance_applied=False,
                total_loops=self.config.total_loops, extra_coda_passes=0, extra_lm_head_calls=0)
            return result
        main_cache = arguments['past_key_values']
        weak_cache = None
        if main_cache is not None:
            length = main_cache.get_seq_length()
            weak_cache = getattr(main_cache, '_loopcd_reference_coda_cache', None)
            if weak_cache is None:
                if length:
                    raise ValueError('Populated strong cache has no reference-coda history')
                weak_cache = type(main_cache)(lookup_strategy='full')
                main_cache._loopcd_reference_coda_cache = weak_cache
            if weak_cache is main_cache or weak_cache.get_seq_length() != length:
                raise ValueError('Reference-coda cache is aliased or out of sync')
            if any(k >= 0 for k in weak_cache.key_cache):
                raise ValueError('Reference cache may contain only coda slots')
        reference, core_calls, first_coda = [], [], []
        coda_calls, head_calls, norm_calls = [], [], []
        def capture_core(module, inputs, output):
            core_calls.append(len(core_calls)+1)
            if len(core_calls)==self.config.reference_loop:
                reference.append(output.detach().clone())
        def capture_coda(module, inputs):
            if not first_coda:
                if len(inputs)!=5 or inputs[4] is not main_cache:
                    raise ValueError('Unexpected native coda interface')
                first_coda.append((inputs[1], inputs[3]))  # exact native rotary positions and mask
        handles=[self.model.transformer.core_block[-1].register_forward_hook(capture_core),
                 self.model.transformer.coda[0].register_forward_pre_hook(capture_coda),
                 self.model.transformer.ln_f.register_forward_hook(lambda *args: norm_calls.append(1)),
                 self.model.lm_head.register_forward_hook(lambda *args: head_calls.append(1))]
        handles += [layer.register_forward_pre_hook(lambda *args: coda_calls.append(1)) for layer in self.model.transformer.coda]
        try:
            result=self._native_forward(**arguments)
            layers=len(self.model.transformer.coda)
            if (core_calls != list(range(1,self.config.total_loops+1)) or len(reference)!=1 or
                len(first_coda)!=1 or len(norm_calls)!=2 or len(coda_calls)!=layers or len(head_calls)!=1):
                raise RuntimeError('Unexpected native recurrent/output trajectory')
            if result.past_key_values is not main_cache:
                raise ValueError('Native strong cache was replaced')
            if weak_cache is not None:
                # Native update advances the sequence clock only in prelude slot0.
                # The reference branch has no prelude: synchronize its clock after
                # the untouched main forward, then append only negative coda slots.
                weak_cache._seen_tokens = main_cache.get_seq_length()
            rng=torch.random.get_rng_state()
            device_rng=torch.cuda.get_rng_state(reference[0].device) if reference[0].is_cuda else None
            x=self.model.transformer.ln_f(reference[0])
            frequencies,mask=first_coda[0]
            for index,layer in enumerate(self.model.transformer.coda):
                x=layer(x,frequencies,torch.tensor(-(index+1),dtype=torch.long,device='cpu'),mask,weak_cache)
            early=self.model.lm_head(self.model.transformer.ln_f(x)).float()
            if not torch.equal(torch.random.get_rng_state(),rng):
                raise RuntimeError('Reference readout unexpectedly consumed RNG')
            if device_rng is not None and not torch.equal(torch.cuda.get_rng_state(reference[0].device),device_rng):
                raise RuntimeError('Reference readout unexpectedly consumed CUDA RNG')
            if len(norm_calls)!=4 or len(coda_calls)!=2*layers or len(head_calls)!=2:
                raise RuntimeError('Reference must traverse the complete native output interface once')
            if weak_cache is not None:
                expected=set(range(main_cache.get_seq_length()))
                for storage in (weak_cache.key_cache,weak_cache.value_cache):
                    if set(storage)!=set(range(-layers,0)) or any(set(slot)!=expected for slot in storage.values()):
                        raise RuntimeError('Incomplete or misindexed reference-coda cache')
            result.logits=apply_guidance(result.logits,early,self.config.logits_config)
            self.last_observation=dict(mode=self.config.mode,guidance_applied=True,total_loops=self.config.total_loops,
                reference_loop=self.config.reference_loop,executed_physical_loops=core_calls,
                extra_coda_passes=1,extra_lm_head_calls=1,lm_head_calls=2,coda_layer_calls=len(coda_calls),
                reference_cache_separate=weak_cache is None or weak_cache is not main_cache,
                reference_cache_length=weak_cache.get_seq_length() if weak_cache is not None else None,
                reference_rng_unchanged=True)
            return result
        finally:
            for handle in handles:handle.remove()
