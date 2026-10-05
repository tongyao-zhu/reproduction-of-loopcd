"""Generate the two preregistered Qwen MBPP arms; no code execution."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qwen_mbpp_protocol import *
from generate_humaneval import atomic_json
from launch_huginn_r16_suite import gpu_status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','model','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--limit',type=int,choices=[2]);p.add_argument('--smoke',type=Path)
    a=p.parse_args()
    if (a.limit is None)!=(a.smoke is not None):raise ValueError('Full378 requires same-release smoke')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Only allocated GPU2')
    source,audited=environment(a.project,ROOT,a.source_commit,a.model);data=load_data(a.data)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from evalplus.sanitize import sanitize
    import torch
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    audit=[build_prompt(data['problems'][t],tokenizer) for t in data['task_ids']]
    same([dict(task_id=r['task_id'],prompt=r['prompt'],input_ids=r['prompt_token_ids'],
               prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest()) for r in audit],audited,'all378 audited prompts')
    prior=validate_pair(a.smoke,data,tokenizer,sanitize) if a.smoke else None
    if not gpu_status('2')['ready']:raise ValueError('GPU occupied')
    a.output.mkdir(parents=True,exist_ok=False);atomic_json(a.output/'prompt_audit.json',audit)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,info=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
        device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict model load')
    model.eval();atomic_json(a.output/'loading.json',info);states={}
    for arm in ARMS:
        cfg=config(arm,source,data['task_ids'],a.limit)
        if prior:
            expected=dict(prior[arm]['config']);expected.update(limit=None,task_ids=data['task_ids'])
            same(cfg,expected,'smoke/full identical protocol')
        folder=a.output/arm;folder.mkdir()
        m=dict(status='running',config=cfg,config_hash=digest(cfg),is_full_split=a.limit is None,
               expected_samples=len(cfg['task_ids']),completed_samples=0)
        atomic_json(folder/'manifest.json',m);states[arm]=dict(folder=folder,manifest=m,rows=[])
    try:
        for task in (data['task_ids'][:2] if a.limit else data['task_ids']):
            for arm,state in states.items():
                m=state['manifest'];row=generate_row(model,tokenizer,data['problems'][task],arm,'cuda:0');row['config_hash']=m['config_hash']
                validate_rows([row],m['config'],data,tokenizer,sanitize,complete=False)
                with (state['folder']/'samples.jsonl').open('a') as f:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                state['rows'].append(row);m.update(completed_samples=len(state['rows']),updated_at=datetime.now(timezone.utc).isoformat())
                atomic_json(state['folder']/'manifest.json',m)
                print(json.dumps(dict(task_id=task,arm=arm,n=len(state['rows']),tokens=row['generated_tokens'],stop=row['stop_reason'])),flush=True)
        same(frozen_tree(ROOT,a.source_commit),source['source_sha256'],'final source')
        for state in states.values():
            m=state['manifest'];validate_rows(state['rows'],m['config'],data,tokenizer,sanitize)
            m.update(status='completed',cap_hits=sum(r['cap_hit'] for r in state['rows']));atomic_json(state['folder']/'manifest.json',m)
        validate_pair(a.output,data,tokenizer,sanitize)
        atomic_json(a.output/'pair_validation.json',dict(status='PASS',n=len(states['baseline']['rows']),full_378=a.limit is None,
            files={str(p.relative_to(a.output)):sha(p) for arm in ARMS for p in (a.output/arm).iterdir()}))
    except BaseException as e:
        for state in states.values():
            state['manifest'].update(status='failed',error=repr(e));atomic_json(state['folder']/'manifest.json',state['manifest'])
        raise
if __name__=='__main__':main()
