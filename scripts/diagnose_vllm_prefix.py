"""Compare HF/vLLM native Ouro logits on identical prefixes, not task scores.

Launch separately with --backend hf then vllm; no concurrent model residency.
Both write fresh outputs and preserve the prior speed probe as immutable input.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
PROBE_SHA='bc47d064d8e69c1ec625bd980bccdc94a8e4f5fe3206928363491b8f01b11cdb'
OURO_SHA='93e1c32b50d31e12ac41236a327490b28747635b31322b93fa9f3eea4ba127ab'
ENDS=(0,64,135,136,179,180,255)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','model','probe','prompt-source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',default='3')
    p.add_argument('--backend',choices=('hf','vllm'),required=True)
    a=p.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins=frozen_tree(ROOT,a.source_commit);a.output.mkdir(parents=True,exist_ok=False)
    s=dict(status='checking',backend=a.backend,pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,
           purpose='fixed-prefix native logits diagnostic; no LoopCD or benchmark scores',started_at=datetime.now(timezone.utc).isoformat(),records=[])
    def save():
        t=a.output/'status.tmp';s['updated_at']=datetime.now(timezone.utc).isoformat();t.write_text(json.dumps(s,indent=2)+'\n');t.replace(a.output/'status.json')
    save()
    try:
        g=gpu_status(a.gpu);s['gpu_before']=g
        if not g['ready']:raise RuntimeError('GPU occupied; diagnostic refused')
        if sha(a.probe)!=PROBE_SHA:raise RuntimeError('Prior diagnostic bytes changed')
        prior=json.loads(a.probe.read_text());raw=a.prompt_source.open('rb').readline()
        if hashlib.sha256(raw).hexdigest()!=prior['prompt_first_line_sha256']:raise RuntimeError('Prompt bytes changed')
        prompt=json.loads(raw)['prompt_token_ids'];trajectory=prior['batches'][1]['generated_token_ids'][0]
        fixtures=[dict(end=n,ids=prompt+trajectory[:n]) for n in ENDS]
        (a.output/'fixtures.json').write_text(json.dumps(fixtures)+'\n');s['fixtures_sha256']=sha(a.output/'fixtures.json');s['prior_sha256']=PROBE_SHA
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
            HF_MODULES_CACHE=str(a.output/'hf_modules'),VLLM_CACHE_ROOT=str(a.output/'vllm_cache'),
            TRITON_CACHE_DIR=str(a.output/'triton'),TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),
            VLLM_WORKER_MULTIPROC_METHOD='spawn',LOOPCD_VLLM_TRACE=str(a.output))
        import torch,numpy as np
        s['versions']={n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}
        s['prefill_batch_token_budget']=4096
        s['model_provenance']=json.loads((a.model/'model_provenance.json').read_text())
        s['status']='loading';save()
        if a.backend=='hf':
            from loopcd_repro.runtime import load_ouro
            from gpu_smoke_aime import native_fixed_depth
            from generate_aime import new_cache
            model,_=load_ouro(a.model)
            # Match the original production runtime, without changing its files.
            for f in fixtures:
                for kind in ('full','cached'):
                    ids=torch.tensor([f['ids']],device='cuda:0');label='hf_'+kind+'_'+str(f['end']);s['status']=label;save()
                    with native_fixed_depth(model),torch.inference_mode():
                        if kind=='full':out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1,exit_at_step=3,exit_threshold=None,use_weighted_exit=False)
                        else:
                            cache=new_cache(model)
                            out=model(input_ids=ids[:,:-1],attention_mask=torch.ones_like(ids[:,:-1]),past_key_values=cache,use_cache=True,logits_to_keep=1,exit_at_step=3,exit_threshold=None,use_weighted_exit=False)
                            out=model(input_ids=ids[:,-1:],attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True,logits_to_keep=1,cache_position=torch.tensor([ids.shape[1]-1],device='cuda:0'),exit_at_step=3,exit_threshold=None,use_weighted_exit=False)
                    arr=out.logits[:,-1].float().cpu().numpy();np.save(a.output/(label+'.npy'),arr,allow_pickle=False)
                    s['records'].append(dict(label=label,shape=list(arr.shape),sha256=sha(a.output/(label+'.npy'))));save()
                    del out,ids
                    if kind=='cached':del cache
        else:
            # A parent-only registry mutation is lost under spawn. Discover this
            # private entry point in every process without installing packages.
            plugin_path=a.output/'plugins';dist=plugin_path/'loopcd_trace-0.0.0.dist-info';dist.mkdir(parents=True)
            (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: loopcd-trace\nVersion: 0.0.0\n')
            (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nloopcd_trace = loopcd_repro.vllm_trace_plugin:register\n')
            sys.path.insert(0,str(plugin_path))
            os.environ['PYTHONPATH']=str(plugin_path)+os.pathsep+str(ROOT/'src')
            os.environ['VLLM_PLUGINS']='loopcd_trace'
            from vllm import LLM,SamplingParams
            from vllm.model_executor.models import ouro
            if s['versions']['vllm']!='0.13.0' or sha(Path(ouro.__file__))!=OURO_SHA:raise RuntimeError('Uninspected vLLM implementation')
            model=LLM(model=str(a.model),tokenizer=str(a.model),trust_remote_code=True,dtype='bfloat16',
                tensor_parallel_size=1,max_model_len=9216,gpu_memory_utilization=.85,max_num_seqs=4,
                max_num_batched_tokens=4096,enable_prefix_caching=False,enforce_eager=True,seed=42)
            for f in fixtures:
                for n in (1,4):
                    label='vllm_n'+str(n)+'_'+str(f['end']);s['status']=label;save()
                    (a.output/'control.json').write_text(json.dumps(dict(label=label)))
                    outputs=model.generate([dict(prompt_token_ids=f['ids']) for _ in range(n)],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,seed=42),use_tqdm=False)
                    (a.output/'control.json').unlink()
                    chunks=sorted(a.output.glob(label+'__call*.npy'))
                    if not chunks:raise RuntimeError('No worker logits captured')
                    arrays=[np.load(path,allow_pickle=False) for path in chunks]
                    arr=np.concatenate(arrays,axis=0)
                    with (a.output/(label+'.npy')).open('xb') as dest:np.save(dest,arr,allow_pickle=False)
                    if arr.shape!=(n,model.llm_engine.model_config.get_vocab_size()) or not np.isfinite(arr).all():raise RuntimeError('Invalid captured logits')
                    tokens=[list(x.outputs[0].token_ids) for x in outputs]
                    if sorted(tokens)!=sorted([[int(x.argmax())] for x in arr]):raise RuntimeError('Captured logits differ from actual greedy sampler')
                    s['records'].append(dict(label=label,shape=list(arr.shape),sha256=sha(a.output/(label+'.npy')),tokens=tokens,actual_microbatch_shapes=[list(x.shape) for x in arrays],trace_files={path.name:sha(path) for path in chunks}));save()
        if frozen_tree(ROOT,a.source_commit)!=pins:raise RuntimeError('Source changed')
        s['status']='completed';save()
    except Exception:
        s['status']='failed';s['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
