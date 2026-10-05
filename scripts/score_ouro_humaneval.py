"""Score a fully validated Ouro R4 logits pair in the existing canonical164 sandbox.

This scorer has its own release. It imports the exact generator release's
input validators; no generated code is executed by this process.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('generation','generator-root','model','data','gate','sandbox-project','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--generator-commit',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args()
    # Resolve before importing anything; only modules from the frozen generation
    # release define its protocol, never the mutable project working directory.
    a.generator_root=a.generator_root.resolve()
    sys.path.insert(0,str(a.generator_root/'scripts'))
    from continue_aime_figure1b import frozen_tree
    from ouro_humaneval_checks import validate_gate,validate_pair,sha,same,digest,DATA_SHA256
    from ouro_humaneval_protocol import model_identity,problems as load_problems,build_prompt
    from compare_humaneval import read_scores,paired_metric,generation_stats
    generator_pins=frozen_tree(a.generator_root,a.generator_commit)
    scorer_pins=frozen_tree(ROOT,a.source_commit)
    gate=json.loads(a.gate.read_text());validate_gate(gate,generator_pins)
    model_pins=model_identity(a.model)['files_sha256']
    same(model_pins,gate['model_files_sha256'],'gate model bytes')
    same(sha(a.data),DATA_SHA256,'scoring data')
    problems=load_problems(a.data)
    from transformers import AutoTokenizer
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    pair=validate_pair(a.generation,problems,tokenizer,sanitize)
    inputs={};generations={}
    for arm,item in pair.items():
        config=item['config'];source=config['source']
        same(source['source_sha256'],generator_pins,'generation frozen source')
        same(source['git_commit'],a.generator_commit,'generation Git commit')
        validate_gate(gate,generator_pins,source)
        same(source['evalplus_version'],importlib.metadata.version('evalplus'),'EvalPlus version')
        same(source['evalplus_prompt_sha256'],sha(inspect.getfile(make_raw_chat_prompt)),'prompt builder')
        same(source['evalplus_sanitize_sha256'],sha(inspect.getfile(sanitize)),'sanitizer')
        binding=config['gate_binding']
        same(binding['gate_sha256'],sha(a.gate),'GPU certificate file')
        same(binding['model_files_sha256'],model_pins,'loaded model bytes')
        same(binding['generation_source_commit'],a.generator_commit,'binding commit')
        same(binding['unchanged_gate_source_sha256'],gate['source_sha256'],'computational files')
        audit=json.loads((a.generation/'prompt_audit.json').read_text())
        same(binding['all_164_prompt_audit_sha256'],digest(audit),'prompt audit')
        same(audit,[build_prompt(p,tokenizer) for p in problems],'actual all164 prompts')
        same(binding['gate_source_commit'],gate['source_commit'],'gate binding commit')
        same(config['limit'],2 if a.allow_smoke else None,'scoring full/smoke scope')
        for name in ('manifest.json','samples.jsonl'):
            path=a.generation/arm/name;inputs[str(path)]=sha(path)
        generations[arm]=dict(config=config,rows={r['task_id']:r for r in item['rows']},
                              samples_sha256=sha(a.generation/arm/'samples.jsonl'))
    scorer=a.sandbox_project.resolve()/'scripts/run_eval_sandbox.py'
    same(sha(scorer),generator_pins['scripts/run_eval_sandbox.py'],'canonical sandbox scorer')
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='scoring',source_commit=a.source_commit,source_sha256=scorer_pins,
               generator_commit=a.generator_commit,generation_sha256=inputs,
               started_at=datetime.now(timezone.utc).isoformat(),allow_smoke=a.allow_smoke,commands=[])
    def save():
        temporary=a.output/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(a.output/'status.json')
    save()
    def unchanged():
        for path,value in inputs.items():same(sha(path),value,'stable generation input')
        same(sha(scorer),generator_pins['scripts/run_eval_sandbox.py'],'unchanged sandbox scorer')
    try:
        scores={}
        for arm in pair:
            unchanged()
            command=['/usr/bin/python3',str(scorer),'--samples',str(a.generation/arm/'samples.jsonl'),
                     '--output',str(a.output/(arm+'.json')),'--timeout','1800']
            state['commands'].append(command);state['active_arm']=arm;save()
            with (a.output/(arm+'.log')).open('x') as log:
                done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1860)
            if done.returncode:raise RuntimeError('Isolated scoring failed for '+arm)
            scores[arm]=read_scores(a.output/(arm+'.json'),generations[arm])
        unchanged()
        base,guided=(scores[arm] for arm in ('baseline','adaptive'))
        for key in ('manifest_sha256','dataset_sha256','scorer_source_sha256','evaluator_patch','canonical_validation','scorer','numpy','psutil'):
            same(base['report'][key],guided['report'][key],'paired scorer '+key)
        for task,row in base['rows'].items():
            for suite in ('base','plus'):same(row[suite]['tests'],guided['rows'][task][suite]['tests'],'paired test count')
        metrics={}
        for suite in ('base','plus'):
            metric=paired_metric(base['rows'],guided['rows'],suite)
            metric['adaptive_pass_at_1_percent']=metric.pop('hidden_pass_at_1_percent')
            metrics[suite]=metric
        summary=dict(status='PASS',n=len(base['rows']),full_164=not a.allow_smoke,
                     interpretation='Two-task integration test only; not a benchmark result' if a.allow_smoke else
                                    'Full164 same-depth Figure1b Ouro R4 logits comparison; not an automatic replication verdict',
                     metrics=metrics,generation_sha256=inputs,source_commit=a.source_commit,
                     generator_commit=a.generator_commit,gate_sha256=sha(a.gate),
                     scores={arm:dict(sha256=scores[arm]['sha256'],generation=generation_stats(generations[arm]['rows'])) for arm in pair})
        (a.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
        same(frozen_tree(a.generator_root,a.generator_commit),generator_pins,'final generator source')
        same(frozen_tree(ROOT,a.source_commit),scorer_pins,'final scorer source')
        state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
        print(json.dumps(dict(status='PASS',n=summary['n'],full_164=summary['full_164'])))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
