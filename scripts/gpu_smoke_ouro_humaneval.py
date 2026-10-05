"""Fresh real Ouro-2.6B gate for registered HumanEval generation and capacity."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from ouro_humaneval_protocol import model_identity,problems,build_prompt,generate_row,generation_config,sha,validate_execution,stopping_criteria
from generate_aime import new_cache,observe_execution,set_seed


def save(path,obj):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(path)


def run(args,report):
    import torch
    from loopcd_repro.runtime import load_ouro,provenance
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    from gpu_smoke import run as numerical,check,exact
    from transformers import LogitsProcessor,LogitsProcessorList
    identity=model_identity(args.model)
    report['model_files_sha256']=identity['files_sha256']
    # Original independent 80-check complete checkpoint oracle: all/last logits,
    # same-shape Eq3/4, native strong-cache tensors and restoration.
    args.device='cuda:0';numerical(args,report)
    torch.cuda.empty_cache();torch.set_num_threads(2)
    model,tokenizer=load_ouro(args.model,args.device)
    report['generation_provenance']=provenance(args.model,model)
    data=problems(args.data);prompts=[build_prompt(x,tokenizer) for x in data]
    report['prompt_audit']={'n':164,'max_tokens':max(len(p['prompt_token_ids']) for p in prompts),
                            'records':[{'task_id':p['task_id'],'prompt_sha256':p['prompt_sha256'],'tokens':len(p['prompt_token_ids'])} for p in prompts]}
    fixture={'task_id':'nonbenchmark-code-gate','prompt':'def add(a, b):\n    """Return the sum of two numbers."""\n','entry_point':'add'}
    prompt=build_prompt(fixture,tokenizer)
    rows={mode:generate_row(model,tokenizer,fixture,mode,args.device,cap=3) for mode in ('baseline','adaptive')}
    report['generation_rows']=rows
    for mode,row in rows.items():
        check(report,'code/'+mode+'/execution',row['generated_tokens']>=1 and row['adapter_observation']['guidance_applied']==(mode=='adaptive'))
    native_ids=torch.tensor([prompt['prompt_token_ids']],device=args.device)
    before=(model.forward,model.early_exit_step,model.early_exit_threshold)
    set_seed(prompt['seed']);cache=new_cache(model)
    with observe_execution(model,cache,'native') as obs,torch.inference_mode():
        out=model.generate(input_ids=native_ids,attention_mask=torch.ones_like(native_ids),past_key_values=cache,
                           generation_config=generation_config(3),logits_to_keep=1,exit_at_step=3,
                           stopping_criteria=stopping_criteria(tokenizer,native_ids.shape[1]))
    native_tokens=out[0,native_ids.shape[1]:].tolist();validate_execution(obs,'native',native_ids.shape[1],len(native_tokens))
    check(report,'code/native_baseline_tokens',native_tokens==rows['baseline']['generated_token_ids'])
    report['native_generation_tokens']=native_tokens
    for name,settings in [('fixed_zero',GuidanceConfig('fixed',omega=0)),('adaptive_zero',GuidanceConfig('adaptive',omega_cap=0))]:
        set_seed(prompt['seed']);cache=new_cache(model)
        with OuroGuidance(model,settings),torch.inference_mode():
            output=model.generate(input_ids=native_ids,attention_mask=torch.ones_like(native_ids),past_key_values=cache,
                                  generation_config=generation_config(3),logits_to_keep=1,
                                  stopping_criteria=stopping_criteria(tokenizer,native_ids.shape[1]))
        check(report,'code/'+name+'/tokens',output[0,native_ids.shape[1]:].tolist()==native_tokens)
    class Force(LogitsProcessor):
        def __init__(self,tokens):self.tokens=tokens;self.i=0
        def __call__(self,ids,scores):
            token=self.tokens[min(self.i,len(self.tokens)-1)];self.i+=1
            scores.fill_(-float('inf'));scores[:,token]=0;return scores
    controlled={}
    for name,tokens in {'eos':[0],'role_delimiter_not_eos':[2],'cap':[10],
                        'stop_string':tokenizer.encode('\nprint(',add_special_tokens=False)}.items():
        controlled[name]={}
        for mode in ('baseline','adaptive'):
            row=generate_row(model,tokenizer,fixture,mode,args.device,cap=8,processors=LogitsProcessorList([Force(tokens)]))
            controlled[name][mode]=row
            expected='eos_token' if name=='eos' else 'stop_string' if name=='stop_string' else 'token_cap'
            check(report,'code/controlled/'+name+'/'+mode,row['stop_reason']==expected)
    report['controlled_generation']=controlled
    del out,output,cache;torch.cuda.empty_cache()
    # Resource gate: repeat a nonbenchmark fixture up to the maximum registered
    # prompt + output budget, then one cached decode. No throughput claim.
    target=report['prompt_audit']['max_tokens']+2048
    pattern=prompt['prompt_token_ids'];tokens=(pattern*((target+len(pattern)-1)//len(pattern)))[:target]
    cache=new_cache(model)
    with OuroGuidance(model,GuidanceConfig('baseline')),observe_execution(model,cache,'baseline') as pre,torch.inference_mode():
        for start in range(0,target,512):
            block=tokens[start:start+512]
            output=model(input_ids=torch.tensor([block],device=args.device),attention_mask=torch.ones((1,start+len(block)),dtype=torch.long,device=args.device),
                         past_key_values=cache,use_cache=True,logits_to_keep=1,cache_position=torch.arange(start,start+len(block),device=args.device))
            if not torch.isfinite(output.logits).all():raise ValueError('Nonfinite long cache prefill')
    with OuroGuidance(model,GuidanceConfig('adaptive',omega_cap=1.)),observe_execution(model,cache,'adaptive') as dec,torch.inference_mode():
        output=model(input_ids=torch.tensor([[10]],device=args.device),attention_mask=torch.ones((1,target+1),dtype=torch.long,device=args.device),
                     past_key_values=cache,use_cache=True,logits_to_keep=1,cache_position=torch.tensor([target],device=args.device))
    check(report,'code/long_cache/finite',bool(torch.isfinite(output.logits).all()))
    check(report,'code/long_cache/all_slots',cache.get_seq_length()==target+1 and cache.max_cache_size==192 and
          all(len(storage)==192 and all(t is not None and t.shape[2]==target+1 for t in storage) for storage in (cache.key_cache,cache.value_cache)))
    report['resource_gate']=dict(target=target,prefill_execution=pre,decode_execution=dec,
                                measured_autoregressive_tokens=1,prefill_chunk_tokens=512,
                                limitation='Repeated fixture prefill plus one cached decode; not a full long code completion')
    check(report,'code/restoration',before==(model.forward,model.early_exit_step,model.early_exit_threshold))
    report['generation_config']=generation_config().to_dict()
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',choices=['0','1','2','3'],required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('Physical GPU binding mismatch')
    if a.output.exists():raise FileExistsError('Fresh GPU gate output required')
    pins=frozen_tree(ROOT,a.source_commit)
    report=dict(status='WAITING',pid=os.getpid(),checks=[],source_commit=a.source_commit,source_sha256=pins,
                started_at=datetime.now(timezone.utc).isoformat(),scope='Ouro base HumanEval correctness gate, no benchmark scores')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(report,stream)
    try:
        while True:
            state=gpu_status(a.gpu);report['device_state']=state;save(a.output,report)
            if state['ready']:break
            if not a.wait:raise RuntimeError('GPU occupied')
            time.sleep(30)
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed while waiting')
        report['status']='RUNNING';save(a.output,report);run(a,report)
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed during gate')
        report['status']='PASS'
    except BaseException:
        report.update(status='FAIL',error=traceback.format_exc());raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat();save(a.output,report)
        print(json.dumps(dict(status=report['status'],checks=len(report['checks']))),flush=True)

if __name__=='__main__':main()
