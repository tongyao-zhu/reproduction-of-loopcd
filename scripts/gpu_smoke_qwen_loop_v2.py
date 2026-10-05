"""Revised correctness evidence, preserving failed v1 unchanged.

Full-prefix/cache equivalence in FP32 at 1e-4 absolute/relative; BF16 compared
exactly to an independent graph with separate native caches and identical
chunk shapes. BF16 full-prefix shape sensitivity remains a reported limitation.
No threshold on v1 is relaxed and this process never generates benchmark code.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gpu_smoke_qwen_loop import sha,cpu_binding,numerical,REVISION
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from generate_humaneval import atomic_json
from qwen_stream_oracle import oracle_stream


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    r=a.project;pins=frozen_tree(ROOT,a.source_commit);cpu,cpu_sha=cpu_binding(r,a.source_commit)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or not gpu_status('2')['ready']:raise ValueError('GPU2 not free')
    failed=r/'results/runs/20261004-qwen-gpu-gate/result.json';v1=json.loads(failed.read_text())
    diagpath=r/'results/runs/20261004-qwen-cache-diagnostic/result.json';diag=json.loads(diagpath.read_text())
    if v1['status']!='FAIL' or diag['status']!='completed' or diag['failed_gate_sha256']!=sha(failed):raise ValueError('Failure/diagnosis binding missing')
    if frozen_tree(r/'releases/qwen-task-6c3ec3e',diag['source_commit'])!=diag['source_sha256']:raise ValueError('Diagnostic source changed')
    if not all(v['tight_fp32_allclose'] for k,v in diag['comparisons'].items() if k.startswith('torch.float32/')):raise ValueError('FP32 diagnosis failed')
    out=r/'results/runs/20261004-qwen-gpu-gate-v2';claim=r/'results/.qwen-claims/figure1b-mbpp-gpu-gate-v2.json'
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),commit=a.source_commit,output=str(out)),f)
    out.mkdir(exist_ok=False)
    report=dict(status='running',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,cpu_result_sha256=cpu_sha,
        failed_v1_sha256=sha(failed),diagnostic_sha256=sha(diagpath),checks=[],scope='revised correctness only; no benchmark',
        criteria='FP32 full-prefix 1e-4/1e-4; BF16 same-shape independent native-cache oracle exact; long BF16 capacity unchanged')
    def save():report['updated_at']=datetime.now(timezone.utc).isoformat();atomic_json(out/'result.json',report)
    def check(name,condition,**extras):
        report['checks'].append(dict(name=name,passed=bool(condition),**extras));save()
        if not condition:raise ValueError('Failed '+name)
    save()
    try:
        resource=r/'results/runs/20261004-qwen-resource-audit/summary.json';audit=json.loads(resource.read_text())
        if sha(resource)!=v1['resource_sha256']:raise ValueError('Resource changed')
        snapshot=Path('/path/to/your/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots')/REVISION
        for name,pin in audit['files'].items():
            if sha(snapshot/name)!=pin['sha256']:raise ValueError('Model changed')
        report.update(resource_sha256=sha(resource),model_files=audit['files'])
        import torch
        from transformers import AutoModelForCausalLM
        from loopcd_repro.qwen_loop import forward_loop,QwenLoopConfig
        import transformers.models.qwen3.modeling_qwen3 as native
        import transformers.cache_utils as cache
        import transformers.masking_utils as masking
        import transformers.integrations.sdpa_attention as sdpa
        runtime={m.__name__:sha(inspect.getfile(m)) for m in (native,cache,masking,sdpa)}
        if runtime!=cpu['runtime_sha256']:raise ValueError('Native runtime changed')
        report['runtime_sha256']=runtime
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not gpu_status('2')['ready']:raise ValueError('GPU became busy')
        model,info=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
            device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict load')
        model.eval();report['loading']=info;cfg=QwenLoopConfig()
        with torch.inference_mode():
            for fixture,values in [('original',[151643,100,200,300,400,500,600]),('different',[1000,15000,21,83,5000,8,319,20,44])]:
                ids=torch.tensor([values],device='cuda:0');state=None;kv=None
                for start,end in ((0,4),(4,5),(5,len(values))):
                    outp=forward_loop(model,ids[:,start:end],cfg,cache=kv,last_token_only=False);kv=outp.cache
                    expected,state=oracle_stream(model,ids[:,start:end],state)
                    check('bf16_oracle/'+fixture+'/'+str(end),torch.equal(outp.logits,expected),max_abs_error=float((outp.logits-expected).abs().max()))
                del outp,expected,state,kv
            # Run unchanged full-prefix/cache suite in FP32, tighten its short numeric assertions.
            model.float();fp={'checks':[]};numerical(model,fp,long_length=65)
            for item in fp['checks']:
                check('fp32/'+item['name'],item['passed'] and item.get('max_abs_error',0)<=.0001,original=item)
            model.bfloat16();torch.cuda.empty_cache()
            ids=torch.tensor([[100]],device='cuda:0');long=((torch.arange(2277,device='cuda:0')[None]%1000)+100)
            for arm,settings in [('baseline',QwenLoopConfig(guided=False)),('fixed',cfg)]:
                state=None
                for start in range(0,2277,256):
                    outp=forward_loop(model,long[:,start:start+256],settings,cache=state);state=outp.cache
                    if not torch.isfinite(outp.logits).all():raise ValueError('Long nonfinite')
                check('bf16_long/'+arm+'/prefill',state.length==2277)
                outp=forward_loop(model,ids,settings,cache=state)
                check('bf16_long/'+arm+'/decode',bool(torch.isfinite(outp.logits).all()) and state.length==2278)
                check('bf16_long/'+arm+'/slots',len(state.kv.layers)==settings.slots and all(state.kv.get_seq_length(i)==2278 for i in range(settings.slots)))
                del outp,state;torch.cuda.empty_cache()
        if len(report['checks'])!=28 or not all(c['passed'] for c in report['checks']):raise ValueError('Incomplete revised gate')
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed')
        report.update(status='PASS',max_memory_allocated=torch.cuda.max_memory_allocated());save()
    except BaseException as exc:report.update(status='FAIL',error=repr(exc));save();raise
if __name__=='__main__':main()
