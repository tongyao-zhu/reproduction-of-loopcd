"""Real Parcae-370M short CPU guidance oracle; not a GPU or benchmark result."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_parcae370 import verify_prepared, stream_fingerprint
from continue_aime_figure1b import frozen_tree
from gpu_smoke_parcae import numerical_cases, CASES

def validate_records(records):
    expected = [(3, depth, mode, omega, cap) for depth in (4, 8) for mode, omega, cap in CASES]
    if [(r.get('length'), r.get('depth'), r.get('mode'), r.get('omega'), r.get('cap')) for r in records] != expected:
        raise ValueError('Missing/duplicate/reordered 370M CPU cases')
    for r in records:
        for key in ('finite','oracle_passed','rng_identical','initial_state_identical','core_states_identical','restored','cache_is_none'):
            if r.get(key) is not True:
                raise ValueError('Unverified ' + key)
        enabled = r['mode'] != 'baseline' and (r['cap'] != 0 if r['mode'] == 'adaptive' else r['omega'] != 0)
        n = 2 if enabled and r['mode'] in ('fixed','adaptive') else 1
        if r.get('calls') != dict(initialization=1,prelude=4,core=4*r['depth'],C=n,coda=4*n,norm=n,head=n):
            raise ValueError('370M layer/readout counts differ')
        if r.get('dtype') != 'torch.float32' or r.get('max_memory_allocated_bytes') != 0 or r.get('max_memory_reserved_bytes') != 0:
            raise ValueError('Not the declared CPU FP32 result')
        for name, upper in [('max_abs_error', float('inf')), ('maximum_tolerance_ratio', 1)]:
            value = r.get(name)
            if type(value) not in (int,float) or not 0 <= value <= upper or value == float('inf'):
                raise ValueError('Invalid numerical error')
        if (not enabled or r['mode'] in ('fixed','hidden')) and r['max_abs_error'] != 0:
            raise ValueError('Required exact oracle differs')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not sys.dont_write_bytecode:
        raise ValueError('Require CUDA_VISIBLE_DEVICES empty and PYTHONDONTWRITEBYTECODE=1')
    pins=frozen_tree(ROOT,a.source_commit)
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',scope=__doc__,source_commit=a.source_commit,source_sha256=pins,started_unix=time.time())
    def save():
        (a.output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        prepared=verify_prepared(a.model)
        state['model_provenance']=stream_fingerprint(a.model/'model_provenance.json')
        sys.path.insert(0,str(a.model/'source'))
        import torch
        from receval.models.parcae import ModelingParcae
        from parcae_lm.attention_backends.flash_attention import HAS_FA3
        if torch.cuda.is_initialized() or torch.cuda.is_available() or HAS_FA3:
            raise ValueError('Require native CPU SDPA')
        torch.set_num_threads(4)
        loads=[]
        class StrictNative(ModelingParcae):
            def load_state_dict(self,state_dict,strict=True,assign=False):
                result=super().load_state_dict(state_dict,strict=True,assign=assign)
                loads.append(dict(keys=len(state_dict),missing=list(result.missing_keys),unexpected=list(result.unexpected_keys),strict=True))
                return result
        model=StrictNative.from_pretrained(a.model,device='cpu',dtype=torch.bfloat16).eval()
        if loads != [dict(keys=117,missing=[],unexpected=[],strict=True)]:
            raise ValueError('Unexpected strict checkpoint inventory')
        if any(len(getattr(model.transformer,name)) != 4 for name in ('prelude','core_block','coda')):
            raise ValueError('Not the fixed 370M physical layer profile')
        if (model.config.state_init != 'like-init' or model.config.block_size != 2048
                or model.config.mean_recurrence != 8 or model.config.padded_vocab_size != 32768
                or any(p.device.type != 'cpu' or p.dtype != torch.bfloat16 for p in model.parameters())):
            raise ValueError('Native model profile changed')
        state['loading']=loads
        state['packages']={name:importlib.metadata.version(name) for name in ('torch','transformers','einops','numpy')}
        state['records']=numerical_cases(model,torch.tensor([[452,2903,312]],dtype=torch.long))
        validate_records(state['records'])
        if verify_prepared(a.model) != prepared or frozen_tree(ROOT,a.source_commit) != pins or torch.cuda.is_initialized():
            raise ValueError('Model/source/CPU binding changed')
        state.update(status='PASS',elapsed_seconds=time.time()-state['started_unix'],cuda_initialized=False)
        save();(a.output/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());save();raise

if __name__=='__main__':
    main()
