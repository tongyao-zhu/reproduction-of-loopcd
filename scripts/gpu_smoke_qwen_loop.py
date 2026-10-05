"""One claimed Qwen correctness gate after the existing Parcae GPU2 queue.

No MBPP generation or scoring. New output only; failed checks retain evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status

ROOT=Path(__file__).resolve().parents[1]
CPU_COMMIT='4ad500152c831f7b685f75cf1149d215b6ef5dd2'
PARCAE_COMMIT='01b49871e5e89cd8fe2cf63685de77df20b73dc3'
REVISION='1cfa9a7208912126459214e8b04321603b3df60c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def cpu_binding(project,commit):
    old=project/'releases/qwen-cpu-4ad5001'
    pins=frozen_tree(old,CPU_COMMIT);current=frozen_tree(ROOT,commit)
    if any(current.get(k)!=v for k,v in pins.items()):raise ValueError('CPU-validated computation changed')
    folder=project/'results/runs/20261004-qwen-cpu-checks-v2'
    report=json.loads((folder/'result.json').read_text())
    if (report['status']!='PASS' or report['tests_run']!=9 or report['errors'] or report['failures'] or report['skipped']
        or report['cuda_initialized'] or report['checkpoint_loaded'] or report['source_commit']!=CPU_COMMIT
        or report['source_sha256']!=pins or report['log_sha256']!=sha(folder/'tests.log')
        or report['tests_source_sha256']!=sha(old/'tests/test_qwen_loop.py')):
        raise ValueError('CPU certificate incomplete or changed')
    original=subprocess.check_output(['git','show',CPU_COMMIT+':tests/test_qwen_loop.py'],cwd=old)
    if original!=(old/'tests/test_qwen_loop.py').read_bytes():raise ValueError('CPU test source changed')
    return report,sha(folder/'result.json')


def numerical(model,report,device="cuda:0",long_length=2277):
    import torch
    from loopcd_repro.qwen_loop import QwenLoopConfig,forward_loop
    def check(name,condition,**extra):
        report['checks'].append(dict(name=name,passed=bool(condition),**extra))
        if not condition:raise ValueError('GPU check failed: '+name)
    def close(name,a,b,atol=.06,rtol=.02):
        check(name,torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
              and torch.allclose(a,b,atol=atol,rtol=rtol),max_abs_error=(a-b).abs().max().item(),atol=atol,rtol=rtol)
    ids=torch.tensor([[151643,100,200,300,400,500,600]],device=device) % model.config.vocab_size
    with torch.inference_mode():
        native=model(ids,use_cache=False).logits[:,-1:].float()
        one=forward_loop(model,ids,QwenLoopConfig(loops=1,damping=1,guided=False),use_cache=False)
        close('R1/unit_damping/native',one.logits,native)
        cfg=QwenLoopConfig();base=QwenLoopConfig(guided=False);zero=QwenLoopConfig(omega=0)
        whole=forward_loop(model,ids,cfg,use_cache=False,last_token_only=False)
        close('fixed_formula',whole.logits,whole.strong_logits+.3*(whole.strong_logits-whole.reference_logits),0,0)
        prefix=forward_loop(model,ids[:,:4],cfg)
        cache=prefix.cache
        for start,end in [(4,5),(5,7)]:
            chunk=forward_loop(model,ids[:,start:end],cfg,cache=cache,last_token_only=False)
            close('incremental/'+str(end),chunk.logits,whole.logits[:,start:end])
            check('KV_lengths/'+str(end),len(cache.kv.layers)==81 and all(cache.kv.get_seq_length(i)==end for i in range(81)))
        a=forward_loop(model,ids,base);b=forward_loop(model,ids,zero)
        close('zero_guidance',a.logits,b.logits,0,0)
        check('strong_cache_unchanged',all(torch.equal(a.cache.kv.layers[i].keys,b.cache.kv.layers[i].keys)
            and torch.equal(a.cache.kv.layers[i].values,b.cache.kv.layers[i].values) for i in range(64)))
        check('readout_counts',whole.layer_calls==dict(prelude=15,core=32,strong_tail=17,reference_tail=17))
        check('native_layer_indices',[l.self_attn.layer_idx for l in model.model.layers]==list(range(36)))
        del a,b,prefix,chunk,cache,whole,one,native
        torch.cuda.empty_cache()
        # Largest audited prompt plus full output budget, then one decode token.
        # Synthetic token fixture only, not a code sample or throughput estimate.
        length=long_length
        long_ids=((torch.arange(length,device=device)[None] % 1000)+100) % model.config.vocab_size
        for name,config in [('baseline',base),('fixed',cfg)]:
            state=None
            for start in range(0,length,256):
                out=forward_loop(model,long_ids[:,start:start+256],config,cache=state)
                state=out.cache
                if not torch.isfinite(out.logits).all():raise ValueError('Nonfinite long prefill')
            check('long/'+name+'/prefill',state.length==length)
            out=forward_loop(model,ids[:,:1],config,cache=state)
            check('long/'+name+'/decode',torch.isfinite(out.logits).all().item() and state.length==length+1)
            check('long/'+name+'/slots',len(state.kv.layers)==config.slots and all(state.kv.get_seq_length(i)==length+1 for i in range(config.slots)))
            del out,state
            torch.cuda.empty_cache()
    report['max_memory_allocated']=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,type=Path);p.add_argument('--source-commit',required=True)
    a=p.parse_args();r=a.project.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Only the registered GPU2 is supported')
    cpu,cpu_sha=cpu_binding(r,a.source_commit);pins=frozen_tree(ROOT,a.source_commit)
    out=r/'results/runs/20261004-qwen-gpu-gate'
    claims=r/'results/.qwen-claims';claims.mkdir(exist_ok=True)
    with (claims/'figure1b-mbpp-gpu-gate-v1.json').open('x') as f:json.dump({'pid':os.getpid(),'output':str(out),'commit':a.source_commit},f)
    out.mkdir(exist_ok=False)
    report={'status':'waiting_predecessor','pid':os.getpid(),'source_commit':a.source_commit,
            'source_sha256':pins,'cpu_result_sha256':cpu_sha,'checks':[],
            'scope':'correctness only; independent Qwen reconstruction; not MBPP result'}
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        tmp=out/'result.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'result.json')
    save()
    try:
        predecessor=r/'results/runs/20261004-parcae13-figure1b'
        while True:
            state=json.loads((predecessor/'status.json').read_text())
            if state['status']=='failed':raise ValueError('Parcae predecessor failed; inspect first')
            if state['status']=='completed':break
            time.sleep(30)
        old=r/'releases/parcae-figure1b-01b4987';frozen_tree(old,PARCAE_COMMIT)
        subprocess.run([sys.executable,str(old/'scripts/compare_parcae_figure1b.py'),
            '--runs',str(predecessor/'full_baseline8'),str(predecessor/'full_adaptive8'),
            '--source-root',str(old),'--output',str(out/'predecessor_pair.json')],check=True)
        pair=json.loads((out/'predecessor_pair.json').read_text())
        if pair['status']!='PASS' or pair['n_documents']!=10042 or not pair['is_full_split']:raise ValueError('Predecessor full pair required')
        report['predecessor_pair_sha256']=sha(out/'predecessor_pair.json')
        report['status']='waiting_gpu';save()
        while not gpu_status('2')['ready']:time.sleep(30)
        if cpu_binding(r,a.source_commit)[1]!=cpu_sha:raise ValueError('CPU evidence changed')
        resource=r/'results/runs/20261004-qwen-resource-audit/summary.json'
        audit=json.loads(resource.read_text())
        if audit['status']!='PASS' or audit['revision']!=REVISION or audit['model_id']!='Qwen/Qwen3-4B' or audit['prompt_count']!=378:
            raise ValueError('Resource audit missing or incorrect')
        snapshot=Path('/path/to/your/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots')/REVISION
        for name,pin in audit['files'].items():
            f=snapshot/name
            if f.stat().st_size!=pin['bytes'] or sha(f)!=pin['sha256']:raise ValueError('Checkpoint file changed: '+name)
        report['resource_sha256']=sha(resource);report['model_files']=audit['files']
        if not gpu_status('2')['ready']:raise ValueError('GPU became busy before model load')
        sys.path.insert(0,str(ROOT/'src'))
        import torch
        from transformers import AutoModelForCausalLM
        import transformers.models.qwen3.modeling_qwen3 as native
        import transformers.cache_utils as cache
        import transformers.masking_utils as masking
        import transformers.integrations.sdpa_attention as sdpa
        runtime={m.__name__:sha(inspect.getfile(m)) for m in (native,cache,masking,sdpa)}
        if runtime!=cpu['runtime_sha256'] or {n:importlib.metadata.version(n) for n in ('torch','transformers')}!=cpu['versions']:
            raise ValueError('Native runtime differs from CPU validation')
        report['runtime_sha256']=runtime;report['status']='running';save();torch.set_num_threads(2)
        model,info=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict checkpoint loading')
        model.eval();report['loading']=info
        numerical(model,report)
        if len(report['checks'])!=16 or not all(x['passed'] for x in report['checks']):raise ValueError('Incomplete GPU checks')
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed during GPU check')
        report['status']='PASS';save()
    except BaseException as exc:
        report['status']='FAIL';report['error']=repr(exc);save();raise


if __name__=='__main__':main()
