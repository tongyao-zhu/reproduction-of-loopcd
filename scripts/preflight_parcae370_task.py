"""CPU-only real loader/likelihood interface checks, not benchmark scores."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from loopcd_repro.parcae370_mc import load_parcae,read_prompt_audit,make_parcae_lm,RequestAudit,sha256
from evaluate_parcae370 import validate_configuration

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.project.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not sys.dont_write_bytecode:raise ValueError('CPU-only required')
    pins=frozen_tree(ROOT,a.source_commit)
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from lm_eval.api.model import LM
    torch.set_num_threads(4)
    config=json.loads((ROOT/'configs/parcae_370m_arc.json').read_text());validate_configuration(config)
    report,expected,binding=read_prompt_audit(root/'results/runs/20261004-parcae370-cpu-checks/arc_prompts','arc_challenge')
    assert len(expected)==4687
    model,tok,loading=load_parcae(root/'models/Parcae-370M',device='cpu')
    rows={};proof={}
    for arm in ('baseline8','adaptive8'):
        stream=io.StringIO();audit=RequestAudit(stream)
        lm=make_parcae_lm(LM)(model,tok,config['arms'][arm],audit)
        requests=[SimpleNamespace(request_type='loglikelihood',repeats=1,task_name='fixture',doc_id=0,idx=i,args=('The answer is',x)) for i,x in enumerate((' A',' B'))]
        torch.manual_seed(42);values=lm.loglikelihood(requests)
        rows[arm]=[json.loads(x) for x in stream.getvalue().splitlines()]
        proof[arm]=dict(values=values,trace=rows[arm],audit=audit.summary(require_complete=True))
    assert [x['pairing'] for x in rows['baseline8']]==[x['pairing'] for x in rows['adaptive8']]
    assert all(x['observation']['call_counts']['core_layers']==32 for values in rows.values() for x in values)
    assert not torch.cuda.is_initialized() and frozen_tree(ROOT,a.source_commit)==pins
    result=dict(status='PASS',scope=__doc__,source_commit=a.source_commit,source_sha256=pins,
        loading={k:v for k,v in loading.items() if k!='prepared'},prompt_audit=binding,
        fixture=proof,all_4687_audited_requests_loaded=True,paired_rng_initialization=True,cuda_initialized=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
