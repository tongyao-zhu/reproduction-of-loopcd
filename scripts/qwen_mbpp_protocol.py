"""Preregistered independent Qwen MBPP contract; never executes generated code."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time
from compare_mbpp import digest, same, load_data, problem_digest, DATA_SHA256
from generate_mbpp import INSTRUCTION, RESPONSE, STOPS, task_seed, trim_stops
from gpu_smoke_qwen_loop import sha, REVISION
from continue_aime_figure1b import frozen_tree

GATE_COMMIT='f3b4a17b4bea9eb9b1d70c8d92153b0ec5e15204'
COMPUTE_COMMIT='11b9f6aff7451f1cfa22530103efa15c2af8b9de'
ARMS=('baseline','fixed')
EOS=(151645,151643)
CAP=2048
VOCAB=151936
GATE_NAMES=['R1/unit_damping/native','fixed_formula','incremental/5','KV_lengths/5','incremental/7','KV_lengths/7',
 'zero_guidance','strong_cache_unchanged','readout_counts','native_layer_indices',
 'long/baseline/prefill','long/baseline/decode','long/baseline/slots','long/fixed/prefill','long/fixed/decode','long/fixed/slots']

GATE_NAMES = (['bf16_oracle/'+fixture+'/'+str(end) for fixture,endpoints in [('original',(4,5,7)),('different',(4,5,9))] for end in endpoints]
              + ['fp32/'+name for name in GATE_NAMES]
              + ['bf16_long/'+arm+'/'+part for arm in ARMS for part in ('prefill','decode','slots')])

def gate_sources(project,pins):
    old=frozen_tree(project/'releases/qwen-gpu-v3-f3b4a17',GATE_COMMIT)
    compute=frozen_tree(project/'releases/qwen-gpu-11b9f6a',COMPUTE_COMMIT)
    compute.update({k:old[k] for k in ('scripts/qwen_stream_oracle.py','scripts/gpu_smoke_qwen_loop_v3.py')})
    if any(pins.get(k)!=v for k,v in compute.items()):raise ValueError('GPU-validated computation changed')
    return old

def build_prompt(problem,tokenizer):
    from evalplus.provider.utility import make_raw_chat_prompt
    text=make_raw_chat_prompt(problem['prompt'].strip()+'\n',INSTRUCTION,RESPONSE,tokenizer)
    ids=tokenizer.encode(text,add_special_tokens=False)
    same(tokenizer.encode(text),ids,'native tokenizer special-token default')
    same(tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),text,'prompt round trip')
    if not ids or len(ids)>229 or not text.endswith('```python\n'):raise ValueError('Unexpected audited prompt')
    return dict(task_id=problem['task_id'],problem_sha256=problem_digest(problem),prompt=text,prompt_sha256=digest(text),
                prompt_token_ids=ids,seed=task_seed(problem['task_id'],42),effective_max_new_tokens=CAP)


def runtime():
    import transformers.models.qwen3.modeling_qwen3 as native
    import transformers.cache_utils as cache
    import transformers.masking_utils as masking
    import transformers.integrations.sdpa_attention as sdpa
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    return dict(native={m.__name__:sha(inspect.getfile(m)) for m in (native,cache,masking,sdpa)},
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','evalplus')},
        prompt_sha256=sha(inspect.getfile(make_raw_chat_prompt)),sanitize_sha256=sha(inspect.getfile(sanitize)))


def environment(project,root,commit,model):
    pins=frozen_tree(root,commit)
    old=gate_sources(project,pins)
    gate_path=project/'results/runs/20261004-qwen-gpu-gate-v3/result.json'
    gate=json.loads(gate_path.read_text())
    same(gate['status'],'PASS','actual GPU gate')
    same(gate['source_commit'],GATE_COMMIT,'GPU source commit');same(gate['source_sha256'],old,'GPU sources')
    same([c['name'] for c in gate['checks']],GATE_NAMES,'all 28 GPU cases')
    if not all(c['passed'] is True for c in gate['checks']):raise ValueError('GPU check failed')
    audit_path=project/'results/runs/20261004-qwen-resource-audit/summary.json'
    audit=json.loads(audit_path.read_text())
    same(sha(audit_path),gate['resource_sha256'],'GPU resources');same(audit['revision'],REVISION,'checkpoint revision')
    same(audit['files'],gate['model_files'],'GPU model files');same(audit['prompt_count'],378,'all inputs')
    same(audit['status'],'PASS','resource audit');same(audit['model_id'],'Qwen/Qwen3-4B','model identity')
    for name,pin in audit['files'].items():
        same(sha(model/name),pin['sha256'],'model '+name);same((model/name).stat().st_size,pin['bytes'],'model size')
    current=runtime();same(current['native'],gate['runtime_sha256'],'native GPU runtime')
    same(current['versions'],audit['versions'],'audited versions');same(current['prompt_sha256'],audit['evalplus_prompt_source_sha256'],'prompt source')
    prompt_file=audit_path.parent/'prompts.jsonl';same(sha(prompt_file),audit['prompts_sha256'],'audit prompts bytes')
    return dict(git_commit=commit,source_sha256=pins,model_revision=REVISION,model_files=audit['files'],runtime=current,
                gpu_gate_sha256=sha(gate_path),resource_audit_sha256=sha(audit_path)),[json.loads(l) for l in prompt_file.read_text().splitlines()]


def config(arm,source,ids,limit):
    if arm not in ARMS or limit not in (None,2):raise ValueError('Unknown scope/arm')
    return dict(benchmark='MbppPlus-v0.2.0',data_sha256=DATA_SHA256,scope='Figure1b independent Qwen reconstruction; Apple wrapper identity unverified',
        arm=arm,loops=8,reference=1,damping=.125,omega=.3,guided=arm=='fixed',seed=42,do_sample=False,
        precision='BF16 weights/hidden, FP32 guidance',attention='sdpa',tensorfloat32=False,
        eos_token_id=list(EOS),pad_token_id=151643,max_new_tokens=CAP,stops=STOPS,
        cache='independent logical-depth KV and strong/weak tails; full-prefix oracle',
        source=source,task_ids=ids[:2] if limit else ids,limit=limit)


def stop_metadata(tokens,decoded,cap):
    if (type(cap)is not int or not 1<=cap<=CAP or not tokens or len(tokens)>cap
        or any(type(t)is not int or not 0<=t<VOCAB for t in tokens) or any(t in EOS for t in tokens[:-1])):
        raise ValueError('Invalid generation/EOS/budget')
    completion,stop=trim_stops(decoded)
    reason='stop_string' if stop else 'eos_token' if tokens[-1] in EOS else 'token_cap'
    if reason=='token_cap' and len(tokens)!=cap:raise ValueError('Unexplained early stop')
    return dict(completion=completion,stop_string=stop,stop_reason=reason,cap_hit=reason=='token_cap')


def decode_tokens(model,tokenizer,ids,arm,device,cap=CAP,processor=None):
    """Greedy batch-one decoding; processor is for controlled CPU tests only."""
    import torch
    from loopcd_repro.qwen_loop import QwenLoopConfig,forward_loop
    if arm not in ARMS or type(cap)is not int or not 1<=cap<=CAP:raise ValueError('Invalid decode contract')
    cfg=QwenLoopConfig(guided=arm=='fixed');cache=None;tokens=[];calls=0
    inp=torch.tensor([ids],device=device,dtype=torch.long)
    with torch.inference_mode():
        for _ in range(cap):
            out=forward_loop(model,inp,cfg,cache=cache);cache=out.cache;calls+=1
            expected=dict(prelude=15,core=32,strong_tail=17,reference_tail=17 if cfg.guided else 0)
            same(out.layer_calls,expected,'per-forward physical layers')
            logits=out.logits[0,-1]
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite guided logits')
            if processor is not None:logits=processor(tokens,logits)
            token=int(logits.argmax());tokens.append(token)
            text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if token in EOS or any(s in text for s in STOPS):break
            inp=torch.tensor([[token]],device=device)
        cache.validate(model,cfg)
        same(cache.length,len(ids)+len(tokens)-1,'final cache length')
    return tokens,dict(forward_calls=calls,cache_slots=cfg.slots,final_cache_length=cache.length,
                       layer_calls_per_forward=expected,fresh_cache_initial_length=0)


def generate_row(model,tokenizer,problem,arm,device):
    import torch
    from generate_aime import set_seed
    from evalplus.sanitize import sanitize
    prompt=build_prompt(problem,tokenizer);set_seed(prompt['seed']);start=time.monotonic()
    tokens,observation=decode_tokens(model,tokenizer,prompt['prompt_token_ids'],arm,device)
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    decoded=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    stopped=stop_metadata(tokens,decoded,CAP)
    return dict(**prompt,arm=arm,sample_id=0,generated_token_ids=tokens,generated_tokens=len(tokens),
        raw_generation=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),**stopped,
        solution=sanitize(stopped['completion'],entrypoint=problem['entry_point']),
        execution_observation=observation,elapsed_seconds=time.monotonic()-start)


def validate_rows(rows,cfg,data,tokenizer,sanitize,complete=True):
    same(cfg,config(cfg['arm'],cfg['source'],data['task_ids'],cfg['limit']),'exact protocol')
    ids=[r['task_id'] for r in rows]
    if len(set(ids))!=len(ids) or any(t not in cfg['task_ids'] for t in ids):raise ValueError('Unknown/duplicate task')
    if complete:same(ids,cfg['task_ids'],'ordered full task set')
    for row in rows:
        problem=data['problems'][row['task_id']]
        for k,v in build_prompt(problem,tokenizer).items():same(row[k],v,'input '+k)
        same(row['arm'],cfg['arm'],'arm');same(row['sample_id'],0,'greedy sample');same(row['config_hash'],digest(cfg),'config hash')
        tokens=row['generated_token_ids'];same(row['generated_tokens'],len(tokens),'length')
        decoded=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        for k,v in stop_metadata(tokens,decoded,CAP).items():same(row[k],v,'stop '+k)
        # No token may follow a string stop, even if the final decoded text looks valid.
        for i in range(1,len(tokens)):
            prefix=tokenizer.decode(tokens[:i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if any(s in prefix for s in STOPS):raise ValueError('Generation continued after string stop')
        same(row['raw_generation'],tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),'raw decode')
        same(row['solution'],sanitize(row['completion'],entrypoint=problem['entry_point']),'AST sanitation')
        guided=cfg['guided'];n=len(tokens)
        same(row['execution_observation'],dict(forward_calls=n,cache_slots=81 if guided else 64,
            final_cache_length=len(row['prompt_token_ids'])+n-1,layer_calls_per_forward=dict(prelude=15,core=32,strong_tail=17,reference_tail=17 if guided else 0),fresh_cache_initial_length=0),'execution')
        if type(row['elapsed_seconds']) not in (int,float) or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<=0:
            raise ValueError('Invalid elapsed time')


def validate_pair(folder,data,tokenizer,sanitize):
    pair={}
    for arm in ARMS:
        directory=folder/arm;m=json.loads((directory/'manifest.json').read_text());cfg=m['config']
        same(m['status'],'completed','manifest completed');same(cfg['arm'],arm,'directory arm')
        same(m['config_hash'],digest(cfg),'manifest config');n=len(cfg['task_ids'])
        same(m['expected_samples'],n,'expected');same(m['completed_samples'],n,'completed');same(m['is_full_split'],cfg['limit'] is None,'full split')
        raw=(directory/'samples.jsonl').read_bytes()
        if not raw.endswith(b'\n'):raise ValueError('Incomplete last row')
        rows=[json.loads(l) for l in raw.splitlines()];validate_rows(rows,cfg,data,tokenizer,sanitize)
        same(m['cap_hits'],sum(r['cap_hit'] for r in rows),'cap hits')
        if m.get('error'):raise ValueError('Manifest error')
        pair[arm]=dict(config=cfg,rows={r['task_id']:r for r in rows},samples_sha256=sha(directory/'samples.jsonl'))
    left,right=(pair[a] for a in ARMS)
    same({k:v for k,v in left['config'].items() if k not in ('arm','guided')},
         {k:v for k,v in right['config'].items() if k not in ('arm','guided')},'paired protocol')
    for task,row in left['rows'].items():
        for k in ('problem_sha256','prompt','prompt_token_ids','seed','effective_max_new_tokens'):
            same(row[k],right['rows'][task][k],'paired '+k)
    return pair
