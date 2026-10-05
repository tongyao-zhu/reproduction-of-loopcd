"""Registered Figure1b Ouro-2.6B HumanEval input and native generation contract."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate_humaneval import DATA_SHA256,INSTRUCTION,RESPONSE,STOPS,digest,task_seed,trim_stops
from generate_aime import new_cache,observe_execution,set_seed
MODEL_ID='ByteDance/Ouro-2.6B'
REVISION='1ed04250da1a9936042725d302e81c8fa2ab5abd'
MAX_NEW_TOKENS=2048
EOS=0
CONTEXT=65536


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):value.update(part)
    return value.hexdigest()


def model_identity(path):
    path=Path(path);identity=json.loads((path/'model_provenance.json').read_text())
    if identity['repo_id']!=MODEL_ID or identity['revision']!=REVISION:raise ValueError('Wrong Ouro base checkpoint')
    hashes={}
    for name,entry in identity['files'].items():
        expected=identity['model_code_sha256'] if name=='modeling_ouro.py' else entry.get('weight_blob',entry.get('sha256_original'))
        hashes[name]=sha(path/name)
        if hashes[name]!=expected:raise ValueError('Model file bytes changed: '+name)
    config=json.loads((path/'config.json').read_text())
    for key,value in dict(vocab_size=49152,total_ut_steps=4,num_hidden_layers=48,max_position_embeddings=65536,bos_token_id=0,eos_token_id=0).items():
        if config[key]!=value:raise ValueError('Native model configuration changed: '+key)
    if (path/'generation_config.json').exists():raise ValueError('Unexpected checkpoint generation overrides')
    return {'model':identity,'files_sha256':hashes,'native_config':config}


def problems(path):
    if sha(path)!=DATA_SHA256:raise ValueError('Changed HumanEvalPlus data')
    rows=sorted([json.loads(line) for line in Path(path).read_text().splitlines()],key=lambda r:int(r['task_id'].split('/')[-1]))
    if [r['task_id'] for r in rows]!=[f'HumanEval/{i}' for i in range(164)]:raise ValueError('Incomplete/duplicate task set')
    return rows


def build_prompt(problem,tokenizer):
    from evalplus.provider.utility import make_raw_chat_prompt
    if tokenizer.eos_token_id!=0 or tokenizer.bos_token_id!=0 or tokenizer.pad_token_id is not None:
        raise ValueError('Native tokenizer special token configuration changed')
    if tokenizer.convert_tokens_to_ids('<|im_start|>')!=1 or tokenizer.convert_tokens_to_ids('<|im_end|>')!=2:
        raise ValueError('Native chat role tokens changed')
    text=make_raw_chat_prompt(problem['prompt'].strip()+'\n',INSTRUCTION,RESPONSE,tokenizer)
    tokens=tokenizer.encode(text,add_special_tokens=False)
    if not tokens or any(type(t)is not int or not 0<=t<49152 for t in tokens):raise ValueError('Invalid prompt tokens')
    if tokens.count(0)!=0 or tokens.count(1)!=3 or tokens.count(2)!=2:
        raise ValueError('Unexpected native system/user/assistant template or extra BOS/EOS')
    if len(tokens)+MAX_NEW_TOKENS>CONTEXT:raise ValueError('Prompt/budget exceeds native context')
    return dict(task_id=problem['task_id'],problem_sha256=digest(problem),prompt=text,prompt_sha256=digest(text),
                prompt_token_ids=tokens,seed=task_seed(problem['task_id'],42),effective_max_new_tokens=MAX_NEW_TOKENS)


def generation_config(cap=MAX_NEW_TOKENS):
    from transformers import GenerationConfig
    if type(cap)is not int or not 1<=cap<=MAX_NEW_TOKENS:raise ValueError('Invalid token cap')
    # Match native EOS=0 and EvalPlus chat stop strings. im_end=2 is a role
    # delimiter, not an extra EOS under this explicitly registered contract.
    return GenerationConfig(do_sample=False,max_new_tokens=cap,num_beams=1,num_return_sequences=1,
                            eos_token_id=EOS,bos_token_id=0,pad_token_id=0,temperature=None,top_p=None,top_k=None,
                            use_cache=True,return_dict_in_generate=False,output_scores=False,output_logits=False,
                            output_attentions=False,output_hidden_states=False)


def stop_metadata(ids,decoded,cap):
    if not ids or any(type(t)is not int or not 0<=t<49152 for t in ids) or len(ids)>cap or EOS in ids[:-1]:
        raise ValueError('Invalid generated token sequence or generation after EOS')
    completion,stop=trim_stops(decoded)
    reason='stop_string' if stop else 'eos_token' if ids[-1]==EOS else 'token_cap'
    if reason=='token_cap' and len(ids)!=cap:raise ValueError('Unrecorded stopping condition')
    return dict(completion=completion,stop_string=stop,stop_reason=reason,cap_hit=reason=='token_cap')


def validate_execution(observation,mode,prompt_length,generated_length):
    heads=1 if mode in ('baseline','native','adaptive_zero') else 2
    expected=dict(forward_calls=generated_length,loop_calls=generated_length*4,head_calls=generated_length*heads,
                  observed_loop_pattern_valid=True,expected_head_calls_per_forward=heads,
                  cache_type='UniversalTransformerCache',cache_slots=192,fresh_cache_initial_length=0,
                  final_cache_length=prompt_length+generated_length-1)
    if observation!=expected:raise ValueError('Unexpected native generation/cache execution')


def stopping_criteria(tokenizer,prompt_length):
    from transformers import StoppingCriteria,StoppingCriteriaList
    class StopOnText(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=tokenizer.decode(input_ids[0,prompt_length:],skip_special_tokens=True)
            return any(stop in text for stop in STOPS)
    return StoppingCriteriaList([StopOnText()])


def generate_row(model,tokenizer,problem,mode,device,cap=MAX_NEW_TOKENS,processors=None):
    import torch
    from evalplus.sanitize import sanitize
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    if mode not in ('baseline','adaptive'):raise ValueError('Only Figure1b paired arms')
    prompt=build_prompt(problem,tokenizer);ids=prompt['prompt_token_ids']
    set_seed(prompt['seed'])
    settings=GuidanceConfig(mode=mode,omega=.5,omega_cap=1.,early_loop=1)
    cache=new_cache(model);inputs=torch.tensor([ids],device=device);mask=torch.ones_like(inputs)
    started=time.monotonic()
    with OuroGuidance(model,settings,total_loops=4) as adapter,observe_execution(model,cache,mode) as observation,torch.inference_mode():
        output=model.generate(input_ids=inputs,attention_mask=mask,past_key_values=cache,
                              generation_config=generation_config(cap),logits_to_keep=1,
                              stopping_criteria=stopping_criteria(tokenizer,len(ids)),logits_processor=processors)
        last=adapter.last_observation
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    if output.shape[0]!=1 or not torch.equal(output[0,:len(ids)],inputs[0]):raise ValueError('Generation changed prompt')
    tokens=output[0,len(ids):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=False)
    stopped=stop_metadata(tokens,tokenizer.decode(tokens,skip_special_tokens=True),cap)
    validate_execution(observation,mode,len(ids),len(tokens))
    row={**prompt,'arm':mode,'guidance':asdict(settings),'total_loops':4,
         'effective_max_new_tokens':cap,'generated_token_ids':tokens,'generated_tokens':len(tokens),
         'raw_generation':raw,**stopped,'solution':sanitize(stopped['completion'],entrypoint=problem['entry_point']),
         'execution_observation':observation,'adapter_observation':last,'elapsed_seconds':time.monotonic()-started}
    if not math.isfinite(row['elapsed_seconds']):raise ValueError('Invalid timing')
    return row
