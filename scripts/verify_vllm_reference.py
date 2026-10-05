"""Independent, read-only NumPy verification of native h1 and actual KV traces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(repo,folder,config_path):
    status_path=folder/'status.json';s=json.loads(status_path.read_text())
    assert s['status']=='completed' and all(c['passed'] for c in s['checks'])
    assert [g['label'] for g in s['groups']]==[f+'_'+m for f in ('short','long') for m in ('native','zero','h1','adaptive')]
    assert sha(config_path)==s['model_provenance']['files']['config.json']['sha256_original']
    config=json.loads(config_path.read_text());assert config==s['model_config'] and config['total_ut_steps']==4
    cap={'ByteDance/Ouro-1.4B-Thinking':1.0,'ByteDance/Ouro-2.6B-Thinking':1.5}[s['model_provenance']['repo_id']]
    assert cap==s['adaptive_cap']
    for name,digest in s['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',s['source_commit']+':'+name],cwd=repo)).hexdigest()==digest
    fixtures_path=folder/'fixtures.json';assert sha(fixtures_path)==s['fixtures_sha256']
    fixtures=json.loads(fixtures_path.read_text());prompts=dict(fixtures['fixtures']);forced=fixtures['forced_tokens'];assert len(forced)==8
    refs={};max_error=0.0;files=0;kv_comparisons=0
    for g in s['groups']:
        label=g['label'];fixture,method=label.rsplit('_',1);ids=prompts[fixture]
        assert g['prompt_length']==len(ids) and g['tokens']==forced
        for name,digest in g['files'].items():assert sha(folder/name)==digest;files+=1
        paths=sorted(folder.glob(label+'__*.npz'));assert {p.name for p in paths}=={n for n in g['files'] if n.endswith('.npz')}
        positions=[];input_ids=[];records=[]
        for path in paths:
            prefix=str(path.with_suffix(''));meta=json.loads(path.with_suffix('.json').read_text());km=json.loads(Path(prefix+'.kv.json').read_text())
            kv=np.load(prefix+'.kv.npy',allow_pickle=False)
            with np.load(path,allow_pickle=False) as array:z=array['final'].copy();guided=array['guided'].copy();early=array['early'].copy() if 'early' in array else None
            positions+=meta['positions'];input_ids+=meta['input_ids']
            c=meta['control'];assert c['label']==label and c['forced_tokens']==forced and c['prompt_length']==len(ids)
            assert c['native_h1']==(method=='h1') and c['native']==(method in ('native','h1'))
            assert z.shape==guided.shape==(1,config['vocab_size']) and np.isfinite(z).all() and np.isfinite(guided).all()
            wanted=[f'model.layers.{layer+ut*config["num_hidden_layers"]}.self_attn.attn' for layer in range(config['num_hidden_layers']) for ut in range(1 if method=='h1' else 4)]
            assert km['names']==wanted and km['positions']==meta['positions'] and len(km['slot_mapping'])==len(wanted)
            assert kv.shape[0]==len(wanted) and kv.shape[1]==2 and kv.dtype==np.uint16
            if method=='adaptive':
                assert c['guidance']==dict(mode='adaptive',omega_cap=cap) and meta['norm_calls']==4
                assert early is not None and early.shape==z.shape and np.isfinite(early).all()
                strong=z.astype('float64');weak=early.astype('float64');prob=np.exp(strong-strong.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True);top=np.sort(prob,axis=-1)[:,-2:]
                error=float(np.abs(guided-(strong+cap*(1-top[:,1:]+top[:,:1])*(strong-weak))).max());assert error<=1e-4;max_error=max(max_error,error)
            else:
                assert early is None and np.array_equal(z,guided)
                assert c['guidance']==(dict(mode='adaptive',omega_cap=0) if method=='zero' else dict(mode='baseline'))
            records.append(dict(z=z,early=early,kv=kv,names=wanted,pos=meta['positions']))
        assert positions==list(range(len(ids)+7)) and input_ids==ids+forced[:-1]
        assert len(records)==g['forward_calls']
        assert sum(r['pos'][-1]>=len(ids)-1 for r in records)==8
        refs[(fixture,method)]=records
        if method in ('zero','adaptive'):
            native=refs[(fixture,'native')];assert len(native)==len(records)
            for a,b in zip(records,native):
                assert a['pos']==b['pos'] and np.array_equal(a['z'],b['z']) and np.array_equal(a['kv'],b['kv']);kv_comparisons+=1
        if method=='adaptive':
            reference=refs[(fixture,'h1')];assert len(reference)==len(records)
            for a,b in zip(records,reference):
                assert a['pos']==b['pos'] and np.array_equal(a['early'],b['z'])
                indices=[a['names'].index(n) for n in b['names']];assert np.array_equal(a['kv'][indices],b['kv']);kv_comparisons+=1
    return dict(status_path=str(status_path.relative_to(repo)),status_sha256=sha(status_path),source_commit=s['source_commit'],source_files=len(s['source_sha256']),
        successful_checks=len(s['checks']),groups=8,raw_trace_files=files,kv_content_comparisons=kv_comparisons,formula_max_abs=max_error,
        h1_all_observed_logits_exact=True,native_zero_adaptive_forced_trajectory_strong_logits_exact=True,
        all_observed_last_written_paged_kv_exact=True,physical_layers=config['num_hidden_layers'],cap=cap,config_sha256=sha(config_path))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.repo
    cases=[('Ouro-1.4B-Thinking','20261004-vllm-reference-14','ouro14_config.json'),('Ouro-2.6B-Thinking','20261004-vllm-reference-26-gpu2','ouro26_config.json')]
    result=dict(status='reference_and_forced_cache_verified',verified_at=datetime.now(timezone.utc).isoformat(),models={m:verify(r,r/'results/runs'/run,r/'results/runs/20261004-heartbeat-1725'/config) for m,run,config in cases},
        benchmark_migration_accepted=False,limits=['Eager TP1, one sequence; short and synthetic 8384-token prefill then 8 forced tokens',
          'KV values compare the last position written in each forward, not every token in the entire cache',
          'Does not establish HF/vLLM distribution equivalence, sampling/8192-output or batch throughput'])
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))
