"""CPU audit of all164 prompts and narrow native Ouro HumanEval generation.

No checkpoint model is loaded; all prepared bytes are hashed. Narrow random
weights exercise the real native class, all48 layers/R4 and generation hooks.
"""
import argparse
from datetime import datetime,timezone
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from ouro_humaneval_protocol import model_identity,problems,build_prompt,generate_row,generation_config,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preflight must hide GPUs')
    a.output.mkdir(parents=True,exist_ok=False)
    pins=frozen_tree(ROOT,a.source_commit)
    import torch
    from transformers import AutoConfig,AutoTokenizer,LogitsProcessorList,LogitsProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    from generate_aime import set_seed
    torch.set_num_threads(2)
    identity=model_identity(a.model);data=problems(a.data)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    prompts=[build_prompt(problem,tokenizer) for problem in data]
    (a.output/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
    config=AutoConfig.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    config.hidden_size=32;config.intermediate_size=64;config.num_attention_heads=config.num_key_value_heads=2
    config.head_dim=16;config._attn_implementation='sdpa'
    native=get_class_from_dynamic_module('modeling_ouro.OuroForCausalLM',str(a.model),local_files_only=True)
    set_seed(1729);model=native(config).to(dtype=torch.bfloat16).eval()
    if sha(inspect.getfile(native))!=identity['model']['model_code_sha256']:raise ValueError('Loaded native code changed')
    checks=[];rows=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise ValueError(name)
    for mode in ('baseline','adaptive'):
        row=generate_row(model,tokenizer,data[0],mode,'cpu',cap=3);rows.append(row)
        check(mode+'/real_native_generate',row['generated_tokens']>=1)
        check(mode+'/observed_guidance',row['adapter_observation']['guidance_applied']==(mode=='adaptive'))
    class Force(LogitsProcessor):
        def __init__(self,tokens):self.tokens=tokens;self.i=0
        def __call__(self,ids,scores):
            token=self.tokens[min(self.i,len(self.tokens)-1)];self.i+=1
            scores.fill_(-float('inf'));scores[:,token]=0;return scores
    # Force only the output-selection stage: the full native loops/head/cache
    # still execute. Fixtures are never reported as model answers.
    fixtures={'eos':[0],'role_delimiter_not_eos':[2], 'cap':[10],
              'stop_string':tokenizer.encode('\nprint(',add_special_tokens=False)}
    for name,tokens in fixtures.items():
        for mode in ('baseline','adaptive'):
            row=generate_row(model,tokenizer,data[1],mode,'cpu',cap=8,processors=LogitsProcessorList([Force(tokens)]))
            rows.append(row)
            expected='eos_token' if name=='eos' else 'stop_string' if name=='stop_string' else 'token_cap'
            check(mode+'/'+name,row['stop_reason']==expected)
            if name=='eos':check(mode+'/eos_one_token',row['generated_token_ids']==[0])
            if name=='role_delimiter_not_eos':check(mode+'/im_end_keeps_generating',row['generated_token_ids']==[2]*8)
    if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Sources changed during CPU preflight')
    report=dict(status='PASS',created_at=datetime.now(timezone.utc).isoformat(),source_commit=a.source_commit,
                source_sha256=pins,model=identity,checks=checks,rows=rows,checkpoint_weights_loaded=False,gpu_execution=False,
                native_random_model=dict(hidden_size=32,num_hidden_layers=48,total_ut_steps=4,vocab_size=49152,seed=1729),
                prompt_audit=dict(n=164,sha256=sha(a.output/'prompts.json'),max_tokens=max(len(p['prompt_token_ids']) for p in prompts)),
                generation_config=generation_config().to_dict(),
                evalplus=dict(version=importlib.metadata.version('evalplus'),prompt_sha256=sha(inspect.getfile(make_raw_chat_prompt)),sanitize_sha256=sha(inspect.getfile(sanitize))),
                scope='CPU native interface and registered stopping only; no GPU gate or benchmark result')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks),prompts=164)))

if __name__=='__main__':main()
