"""Score validated Qwen generations in the canonical378 MBPP sandbox only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
from qwen_mbpp_protocol import *
from compare_mbpp import load_canonical,read_scores,paired_metric,generation_stats,RUNNER_SHA256,SAFETY_CHECKS,SCORER
from generate_humaneval import atomic_json


def read_subset(path,generation,context):
    """Two-task integration proof with full canonical378 evidence binding."""
    report=json.loads(path.read_text());ids=list(generation['rows']);n=len(ids)
    expected=dict(status='PASS',exit_code=0,evaluation_complete=True,expected_rows=n,completed_rows=n,
        samples_sha256=generation['samples_sha256'],manifest_sha256=context['manifest_sha256'],
        dataset_sha256=context['data']['sha256'],scorer_source_sha256=RUNNER_SHA256,scorer=SCORER,
        evaluator_patch=context['manifest']['evaluator_patch'],identity=context['manifest']['identity'],
        sandbox=str(Path(context['manifest']['rootfs']).parent),numpy=context['manifest']['packages']['numpy'],
        psutil=context['manifest']['packages']['psutil'],sample_validation=dict(samples=n,expected_full_tasks=378,is_full_task_set=False))
    for k,v in expected.items():same(report[k],v,'subset score '+k)
    if report.get('error'):raise ValueError('Scoring infrastructure error')
    same(report['safety']['checks'],{k:True for k in SAFETY_CHECKS},'all isolation checks');same(report['safety']['passed'],True,'isolation')
    cert=report['canonical_validation']
    for k,v in dict(status='PASS',passed_tasks=378,expected_rows=378,completed_rows=378,all_base_plus_passed=True,
        manifest_sha256=context['manifest_sha256'],dataset_sha256=context['data']['sha256'],scorer_source_sha256=RUNNER_SHA256,
        evaluator_patch=context['manifest']['evaluator_patch'],evidence_sha256=context['evidence_sha256'],
        result_sha256=context['evidence_sha256'],task_ids=context['data']['task_ids']).items():same(cert[k],v,'canonical '+k)
    if not Path(cert['result_path']).is_absolute():raise ValueError('Canonical path missing')
    rows={}
    for row in report['rows']:
        task=row['task_id']
        if task in rows or task not in ids:raise ValueError('Duplicate/foreign score task')
        same(row['sample_id'],0,'greedy score')
        for suite in ('base','plus'):
            s=row[suite];count=context['data']['counts'][task][suite];same(s['tests'],count,'actual test count')
            if s['status'] not in ('pass','fail','timeout'):raise ValueError('Scorer failure status')
            details=s['details']
            if not isinstance(details,list) or len(details)>count or any(type(v)is not bool for v in details):raise ValueError('Invalid test details')
            complete=len(details)==count and all(details)
            if s['status']=='pass' and not complete:raise ValueError('Incomplete passing proof')
            same(s['passed'],s['status']=='pass' and complete,'suite status')
        same(row['plus_passed'],row['base']['passed'] and row['plus']['passed'],'base+extended')
        rows[task]=row
    same(set(rows),set(ids),'full subset coverage')
    return dict(rows=rows,report=report,sha256=sha(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','model','data','generation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();source,_=environment(a.project,ROOT,a.source_commit,a.model);data=load_data(a.data)
    from transformers import AutoTokenizer
    from evalplus.sanitize import sanitize
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    pair=validate_pair(a.generation,data,tokenizer,sanitize)
    for item in pair.values():
        same(item['config']['source'],source,'frozen generation environment')
        same(item['config']['limit'],2 if a.allow_smoke else None,'score scope')
    sandbox=a.project/'.sandbox/mbpp-v2'
    certificate=json.loads((sandbox/'canonical_validation.json').read_text())
    context=load_canonical(Path(certificate['result_path']),sandbox/'manifest.json',data)
    runner=a.project/'releases/mbpp-scorer-v2-33fa680/scripts/run_mbpp_sandbox.py';same(sha(runner),RUNNER_SHA256,'canonical runner')
    inputs={str(p):sha(p) for arm in ARMS for p in (a.generation/arm/'manifest.json',a.generation/arm/'samples.jsonl')}
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='scoring',source_commit=a.source_commit,source_sha256=source['source_sha256'],inputs=inputs,commands=[])
    def save():atomic_json(a.output/'status.json',state)
    def unchanged():
        for path,value in inputs.items():same(sha(path),value,'stable score input')
        same(sha(runner),RUNNER_SHA256,'stable scorer')
    save()
    try:
        scores={}
        for arm in ARMS:
            unchanged();command=['/usr/bin/python3',str(runner),'--sandbox',str(sandbox),'--samples',str(a.generation/arm/'samples.jsonl'),
                               '--output',str(a.output/(arm+'.json')),'--timeout','7200']
            state.update(active_arm=arm);state['commands'].append(command);save()
            with (a.output/(arm+'.log')).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=7260)
            scores[arm]=(read_subset if a.allow_smoke else read_scores)(a.output/(arm+'.json'),pair[arm],context)
        unchanged();same(frozen_tree(ROOT,a.source_commit),source['source_sha256'],'scoring source')
        metrics={}
        for suite in ('base','plus'):
            m=paired_metric(scores['baseline']['rows'],scores['fixed']['rows'],suite)
            m['fixed_passed']=m.pop('hidden_passed');m['fixed_pass_at_1_percent']=m.pop('hidden_pass_at_1_percent');metrics[suite]=m
        atomic_json(a.output/'comparison.json',dict(status='PASS',n=len(pair['baseline']['rows']),full_378=not a.allow_smoke,
            interpretation='Independent preregistered reconstruction; Apple checkpoint/cache identity unverified',metrics=metrics,
            source_commit=a.source_commit,source_sha256=source['source_sha256'],inputs=inputs,
            canonical_sha256=context['evidence_sha256'],sandbox_manifest_sha256=context['manifest_sha256'],
            scores={arm:dict(sha256=scores[arm]['sha256'],generation=generation_stats(pair[arm]['rows'])) for arm in ARMS}))
        state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as e:state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
