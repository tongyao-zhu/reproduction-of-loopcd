"""Frozen Figure1b Ouro-2.6B paired HumanEval generation, never code execution."""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from generate_humaneval import atomic_json
from ouro_humaneval_protocol import model_identity,problems,build_prompt,generate_row,digest,sha
from ouro_humaneval_checks import validate_gate,validate_pair,validate_rows,protocol_config,ARMS,same


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output','gate'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',choices=['0','1','2','3'],required=True)
    p.add_argument('--limit',type=int,choices=[2]);p.add_argument('--smoke',type=Path)
    a=p.parse_args()
    if (a.limit is None)!=(a.smoke is not None):raise ValueError('Full164 requires verified same-release smoke2')
    if os.getenv('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('GPU binding mismatch')
    pins=frozen_tree(ROOT,a.source_commit);gate=json.loads(a.gate.read_text());validate_gate(gate,pins)
    identity=model_identity(a.model);same(identity['files_sha256'],gate['model_files_sha256'],'GPU gate model bytes')
    data=problems(a.data)
    if not gpu_status(a.gpu)['ready']:raise RuntimeError('GPU occupied')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from loopcd_repro.runtime import load_ouro,provenance
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    torch.set_num_threads(2)
    model,tokenizer=load_ouro(a.model,'cuda:0');source=provenance(a.model,model)
    source.update(evalplus_version=importlib.metadata.version('evalplus'),evalplus_prompt_sha256=sha(inspect.getfile(make_raw_chat_prompt)),
                  evalplus_sanitize_sha256=sha(inspect.getfile(sanitize)))
    validate_gate(gate,pins,source)
    audit=[build_prompt(problem,tokenizer) for problem in data]
    same([dict(task_id=x['task_id'],prompt_sha256=x['prompt_sha256'],tokens=len(x['prompt_token_ids'])) for x in audit],
         gate['prompt_audit']['records'],'GPU task prompt audit')
    atomic_json(a.output/'prompt_audit.json',audit)
    binding=dict(gate_sha256=sha(a.gate),gate_source_commit=gate['source_commit'],generation_source_commit=a.source_commit,
                 unchanged_gate_source_sha256=gate['source_sha256'],model_files_sha256=identity['files_sha256'],
                 all_164_prompt_audit_sha256=digest(audit))
    atomic_json(a.output/'gate_binding.json',binding)
    previous=validate_pair(a.smoke,data,tokenizer,sanitize) if a.smoke else None
    states={};selected=data[:2] if a.limit else data
    for arm in ARMS:
        config=protocol_config(arm,source,binding,a.limit)
        if previous:
            expected=dict(previous[arm]['config']);expected.update(limit=None,task_ids=config['task_ids'])
            same(config,expected,'smoke/full configuration')
        folder=a.output/arm;folder.mkdir()
        m=dict(status='running',config=config,config_hash=digest(config),is_full_split=a.limit is None,
               expected_samples=len(selected),completed_samples=0,CUDA_VISIBLE_DEVICES=a.gpu,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(folder/'manifest.json',m);states[arm]=dict(manifest=m,folder=folder,rows=[])
    try:
        for problem in selected:
            for arm,state in states.items():
                row=generate_row(model,tokenizer,problem,arm,'cuda:0');m=state['manifest']
                row['config_hash']=m['config_hash']
                validate_rows([row],m['config'],data,tokenizer,sanitize,complete=False)
                with (state['folder']/'samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                state['rows'].append(row);m.update(completed_samples=len(state['rows']),updated_at=datetime.now(timezone.utc).isoformat())
                atomic_json(state['folder']/'manifest.json',m)
                print(json.dumps(dict(task_id=row['task_id'],arm=arm,completed=len(state['rows']),tokens=row['generated_tokens'],stop=row['stop_reason'])),flush=True)
        same(frozen_tree(ROOT,a.source_commit),pins,'final frozen source')
        for state in states.values():
            m=state['manifest'];validate_rows(state['rows'],m['config'],data,tokenizer,sanitize)
            m.update(status='completed',cap_hits=sum(r['cap_hit'] for r in state['rows']));atomic_json(state['folder']/'manifest.json',m)
        validate_pair(a.output,data,tokenizer,sanitize)
        atomic_json(a.output/'pair_validation.json',dict(status='PASS',n=len(selected),full_164=a.limit is None,
            arms={arm:{name:sha(a.output/arm/name) for name in ('manifest.json','samples.jsonl')} for arm in ARMS}))
    except BaseException as exc:
        for state in states.values():
            state['manifest'].update(status='failed',error=repr(exc));atomic_json(state['folder']/'manifest.json',state['manifest'])
        raise

if __name__=='__main__':main()
