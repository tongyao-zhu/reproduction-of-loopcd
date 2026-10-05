"""Real R16/h1 Huginn checkpoint gate, including separate cached readouts."""
import argparse
from dataclasses import asdict
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
from gpu_smoke_huginn import observe, readout_oracle, check, exact, cache_equal, restored, save, stamp
from loopcd_repro.huginn import load_huginn
from loopcd_repro.huginn_logits import HuginnLogitsConfig as Config, HuginnLogitsGuidance as Guide
from loopcd_repro.runtime import provenance
import torch


def formula(final,early):
    # Independent arithmetic oracle, with a separate FP64 accuracy check.
    probabilities=final.float().softmax(-1).topk(2,dim=-1).values
    return final.float()+(.25*(1-(probabilities[...,:1]-probabilities[...,1:2])))*(final.float()-early.float())


@torch.inference_mode()
def run(args,report):
    torch.set_num_threads(2)
    model,tokenizer=load_huginn(args.model,'cuda:0')
    report['provenance']=provenance(args.model,model)
    report['configuration']=asdict(Config())
    methods={name:getattr(model,name) for name in ('forward','generate','initialize_state')}
    ids=tokenizer('def add(a, b):\n    return',return_tensors='pt')['input_ids'].to('cuda:0')
    report['fixture_token_ids']=ids.tolist()
    torch.manual_seed(991)
    with observe(model,16) as native_trace:
        native=model(ids,num_steps=16,use_cache=False)
    rng=torch.cuda.get_rng_state().clone()
    early=readout_oracle(model,native_trace['raw_states'][1],native_trace['raw_states'][1],0.)
    expected=formula(native.logits,early)
    for name,config in [('baseline',Config('baseline')),('zero_fixed',Config('fixed',omega=0)),
                        ('zero_adaptive',Config('adaptive',omega_cap=0)),('adaptive',Config())]:
        torch.manual_seed(991)
        with Guide(model,config) as adapter:
            with observe(model,16) as trace:output=model(ids,use_cache=False)
            exact(report,name+'/logits',output.logits,expected if config.enabled else native.logits)
            exact(report,name+'/initial_state',trace['initial_states'][0],native_trace['initial_states'][0])
            exact(report,name+'/final_recurrent_state',trace['raw_states'][16],native_trace['raw_states'][16])
            heads=2 if config.enabled else 1
            check(report,name+'/layers',trace['core_calls']==trace['adapter_calls']==16 and trace['head_calls']==heads
                  and trace['coda_calls']==[id(x) for x in model.transformer.coda]*heads)
            check(report,name+'/rng',torch.equal(torch.cuda.get_rng_state(),rng))
            if config.enabled:
                ps=native.logits.double().softmax(-1).topk(2,dim=-1).values
                high=native.logits.double()+.25*(1-(ps[...,:1]-ps[...,1:2]))*(native.logits.double()-early.double())
                torch.testing.assert_close(output.logits.double(),high,atol=2e-5,rtol=2e-6)
                check(report,name+'/fp64_oracle',True,max_abs_error=float((output.logits.double()-high).abs().max()))
                check(report,name+'/nonzero_effect',not torch.equal(output.logits,native.logits))
        restored(report,name+'/restored',model,methods)
    cache_class=type(Guide(model,Config()).new_cache())
    strong=cache_class(lookup_strategy='full');weak=cache_class(lookup_strategy='full')
    actual=Guide(model,Config()).new_cache()
    offset=0
    for step,block in enumerate((ids,ids[:,-2:-1],ids[:,-1:])):
        positions=torch.arange(offset,offset+block.shape[1],device='cuda:0')
        torch.manual_seed(710+step)
        with observe(model,16) as reference:
            normal=model(block,num_steps=16,use_cache=True,past_key_values=strong,cache_position=positions)
        weak._seen_tokens=strong.get_seq_length()
        early=readout_oracle(model,reference['raw_states'][1],reference['raw_states'][1],0.,cache=weak,positions=positions)
        expected=formula(normal.logits,early)
        torch.manual_seed(710+step)
        with Guide(model,Config()) as adapter:
            with observe(model,16) as trace:
                output=model(block,use_cache=True,past_key_values=actual,cache_position=positions)
            exact(report,f'cache/{step}/logits',output.logits,expected)
            exact(report,f'cache/{step}/initial',trace['initial_states'][0],reference['initial_states'][0])
            cache_equal(report,f'cache/{step}/strong',actual,strong)
            cache_equal(report,f'cache/{step}/reference',actual._loopcd_reference_coda_cache,weak)
            check(report,f'cache/{step}/separate',actual._loopcd_reference_coda_cache is not actual)
        offset+=block.shape[1]
    del strong,weak,actual,native,native_trace,reference,trace,early,expected,output,normal
    torch.cuda.empty_cache()
    # Resource/finite gate only. Short cached oracle above certifies arithmetic.
    long_ids=ids.repeat(1,(2048+ids.shape[1]-1)//ids.shape[1])[:,:2048]
    with Guide(model,Config()) as adapter:
        cache=adapter.new_cache()
        for step,block in enumerate((long_ids,ids[:,-1:])):
            position=torch.arange(0,2048,device='cuda:0') if step==0 else torch.tensor([2048],device='cuda:0')
            output=model(block,use_cache=True,past_key_values=cache,cache_position=position)
            check(report,f'long/{step}/finite',bool(torch.isfinite(output.logits).all()),length=block.shape[1])
            check(report,f'long/{step}/cache',cache.get_seq_length()==2048+step and
                  cache._loopcd_reference_coda_cache.get_seq_length()==2048+step)
            del output
    del cache,long_ids
    torch.cuda.empty_cache()
    with Guide(model,Config()) as adapter:
        output=model.generate(ids,max_new_tokens=2,min_new_tokens=2,do_sample=False,use_cache=True,
                              pad_token_id=model.config.pad_token_id,eos_token_id=model.config.eos_token_id)
        check(report,'native_generate/two_tokens',output.shape[1]==ids.shape[1]+2)
        check(report,'native_generate/guided',adapter.last_observation['lm_head_calls']==2)
    restored(report,'final/restored',model,methods)
    torch.cuda.synchronize()
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh gate output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:raise ValueError('Physical GPU binding mismatch')
    pins=frozen_tree(ROOT,args.source_commit)
    report={'status':'WAITING','pid':os.getpid(),'started_at':stamp(),'checks':[], 'source_commit':args.source_commit,
            'source_sha256':pins,'gpu':args.gpu,'scope':'Real checkpoint numerical/cache and 2048+1 resource gate, no benchmark score'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream)
    try:
        while True:
            state=gpu_status(args.gpu);report['device_state']=state;report['updated_at']=stamp();save(args.output,report)
            if state['ready']:break
            if not args.wait:raise RuntimeError('GPU is occupied')
            time.sleep(30)
        if frozen_tree(ROOT,args.source_commit)!=pins:raise ValueError('Source changed while waiting')
        report['status']='RUNNING';save(args.output,report)
        run(args,report)
        if frozen_tree(ROOT,args.source_commit)!=pins:raise ValueError('Source changed during gate')
        report.update(status='PASS',finished_at=stamp())
    except BaseException:
        report.update(status='FAIL',error=traceback.format_exc(),finished_at=stamp())
        raise
    finally:
        save(args.output,report)
        print(json.dumps({'status':report['status'],'checks':len(report['checks']),'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
