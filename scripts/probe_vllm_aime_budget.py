"""Finite real-sampling/batch probe after a completed reference gate.

Original project T=1/top-p=.7/8192/EOS2 and paired seeds. Outputs are only
diagnostics and cannot be mixed with HF benchmark samples. No code execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','prompt-source','gate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins=frozen_tree(ROOT,a.source_commit)
    gate=json.loads(a.gate.read_text())
    if gate['status']!='completed' or len(gate['groups'])!=8 or not all(c['passed'] for c in gate['checks']):raise ValueError('Reference gate incomplete')
    for name in ('src/loopcd_repro/vllm_ouro.py','src/loopcd_repro/guidance.py'):
        if pins[name]!=gate['source_sha256'][name]:raise ValueError('Computation changed after reference gate')
    for g in gate['groups']:
        for name,digest in g['files'].items():
            if sha(a.gate.parent/name)!=digest:raise ValueError('Gate raw changed')
    a.output.mkdir(parents=True,exist_ok=False)
    s=dict(status='checking',purpose=__doc__,source_commit=a.source_commit,source_sha256=pins,pid=os.getpid(),gate_sha256=sha(a.gate),batches=[])
    def save():
        s['updated_at']=datetime.now(timezone.utc).isoformat();tmp=a.output/'status.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(a.output/'status.json')
    save()
    try:
        s['gpu_before']=gpu_status(a.gpu)
        if not s['gpu_before']['ready']:raise RuntimeError('GPU not free')
        provenance=json.loads((a.model/'model_provenance.json').read_text())
        if provenance!=gate['model_provenance']:raise ValueError('Model identity differs')
        s['model_provenance']=provenance;cap=gate['adaptive_cap']
        with a.prompt_source.open('rb') as stream:raw=[stream.readline() for _ in range(2)]
        if hashlib.sha256(raw[0]).hexdigest()!=gate['input_row_sha256']:raise ValueError('Input changed')
        rows=[json.loads(x) for x in raw]
        if len({(r['task_id'],r['sample_id']) for r in rows})!=2:raise ValueError('Duplicate probe samples')
        for r in rows:
            seed=int.from_bytes(hashlib.sha256(f"loopcd-aime-v1|42|{r['task_id']}|{r['sample_id']}".encode()).digest()[:4],'big')
            if r['seed']!=seed or len(r['prompt_token_ids'])+8192>9216:raise ValueError('Seed/length outside gate scope')
        s['input_rows']=[{k:r[k] for k in ('task_id','sample_id','seed','prompt_token_ids')} for r in rows]
        s['input_row_sha256']=[hashlib.sha256(x).hexdigest() for x in raw]
        os.environ.pop('LOOPCD_VLLM_GATE',None)
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
            HF_MODULES_CACHE=str(a.output/'hf_modules'),VLLM_CACHE_ROOT=str(a.output/'vllm_cache'),TRITON_CACHE_DIR=str(a.output/'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),VLLM_WORKER_MULTIPROC_METHOD='spawn',LOOPCD_VLLM_SETTINGS=json.dumps(dict(mode='baseline')))
        plugin=a.output/'plugins';dist=plugin/'loopcd_ouro-0.0.0.dist-info';dist.mkdir(parents=True)
        (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: loopcd-ouro\nVersion: 0.0.0\n')
        (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nloopcd_ouro = loopcd_repro.vllm_ouro_plugin:register\n')
        sys.path.insert(0,str(plugin));os.environ['PYTHONPATH']=str(plugin)+os.pathsep+str(ROOT/'src');os.environ['VLLM_PLUGINS']='loopcd_ouro'
        from vllm import LLM,SamplingParams
        s['versions']={n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}
        if s['versions']['vllm']!='0.13.0':raise ValueError('Version changed')
        s['status']='loading';save()
        model=LLM(model=str(a.model),tokenizer=str(a.model),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=9216,max_num_seqs=2,max_num_batched_tokens=4096,gpu_memory_utilization=.85,enable_prefix_caching=False,
            enforce_eager=True,seed=42,generation_config='vllm',
            worker_extension_cls='loopcd_repro.vllm_budget_runtime.BudgetWorkerExtension')
        tokenizer=model.get_tokenizer()
        if tokenizer.eos_token_id!=2:raise ValueError('Wrong EOS')
        for label,mode,coefficient,n in [('baseline_serial','baseline',0,1),('zero_serial','adaptive',0,1),
                ('adaptive_serial','adaptive',cap,1),('baseline_batch2','baseline',0,2),('adaptive_batch2','adaptive',cap,2)]:
            s['status']=label;save()
            configured=model.collective_rpc('loopcd_budget_configure',args=(mode,coefficient))
            params=[SamplingParams(n=1,temperature=1.0,top_p=.7,top_k=-1,min_p=0.0,repetition_penalty=1.0,
                presence_penalty=0.0,frequency_penalty=0.0,max_tokens=8192,min_tokens=0,seed=r['seed'],
                stop=None,stop_token_ids=[2],ignore_eos=False,skip_special_tokens=False) for r in rows[:n]]
            start=time.perf_counter()
            outputs=model.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows[:n]],params,use_tqdm=False)
            elapsed=time.perf_counter()-start
            counters=model.collective_rpc('loopcd_budget_finish');records=[]
            for r,o in zip(rows[:n],outputs):
                item=o.outputs[0];tokens=list(item.token_ids)
                if list(o.prompt_token_ids)!=r['prompt_token_ids'] or not tokens or len(tokens)>8192:raise ValueError('Invalid output')
                if tokens[-1]==2:
                    if 2 in tokens[:-1] or item.finish_reason!='stop':raise ValueError('EOS not respected')
                    reason='eos'
                elif len(tokens)==8192 and item.finish_reason=='length':reason='length'
                else:raise ValueError('Unexpected stopping condition')
                records.append(dict(task_id=r['task_id'],sample_id=r['sample_id'],seed=r['seed'],generated_token_ids=tokens,
                    raw_generation=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop_reason=reason))
            if len(records)!=n:raise ValueError('Incomplete batch')
            s['batches'].append(dict(label=label,mode=mode,cap=coefficient,n=n,seconds=elapsed,
                output_tokens=sum(len(r['generated_token_ids']) for r in records),configured=configured,counters=counters,
                sampling_parameters=[str(p) for p in params],outputs=records));save()
            if label=='zero_serial' and records[0]['generated_token_ids']!=s['batches'][0]['outputs'][0]['generated_token_ids']:
                raise ValueError('Native/zero full-budget same-seed sampling differs')
        for b in s['batches']:b['tokens_per_second']=b['output_tokens']/b['seconds']
        s['batch_first_matches_serial']={arm:s['batches'][idx]['outputs'][0]['generated_token_ids']==s['batches'][ref]['outputs'][0]['generated_token_ids'] for arm,idx,ref in [('baseline',3,0),('adaptive',4,2)]}
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed')
        s['status']='completed';s['benchmark_migration_accepted']=False;save()
    except Exception:s['status']='failed';s['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
