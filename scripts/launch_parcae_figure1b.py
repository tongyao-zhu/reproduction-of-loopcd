"""Fresh Figure1b HellaSwag pair after the existing Huginn R16 queue on GPU2."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from compare_parcae import sha256 as sha,require,same
from gpu_smoke_parcae import validate_report
from compare_humaneval import read_scores,paired_metric

GEN='29ea1d89a557d5307afaa1586636e2ed3cb779c1'
SCORE='546bb4f3c8bd3422b8ae8e4d23f8d6229b88ee7e'
QUEUE='3c4875d999a1e521ce6a4eb2a274d867e8a5c3a0'
ARMS=('baseline8','adaptive8')


def validate_predecessor(root):
    """Validate completed input/score hashes and canonical164 isolation evidence."""
    base=root/'results/runs/20261004-huginn-he-logits-v2'
    state=json.loads((base/'queue/status.json').read_text())
    same(state['source_commit'],QUEUE,'predecessor queue commit')
    if state['status']=='failed':raise ValueError('Predecessor failed')
    require(state['status'] in ('waiting_smoke','score_smoke','waiting_gpu','generate_full','score_full','completed'),'Unknown predecessor state')
    if state['status']!='completed':return None
    pins=frozen_tree(root/'releases/he-logits-29ea1d8',GEN)
    frozen_tree(root/'releases/he-logits-score-546bb4f',SCORE)
    frozen_tree(root/'releases/he-logits-queue-3c4875d',QUEUE)
    summary=json.loads((base/'full_scores/comparison.json').read_text())
    same(summary['status'],'PASS','predecessor score status');same(summary['n'],164,'complete predecessor')
    require(summary['full_164'] is True,'Predecessor subset')
    same(summary['source_commit'],SCORE,'predecessor scorer');same(summary['generator_commit'],GEN,'predecessor generator')
    expected={str(base/'full'/arm/file) for arm in ('baseline16','adaptive16') for file in ('manifest.json','samples.jsonl')}
    same(set(summary['generation_sha256']),expected,'full predecessor input set')
    for path,value in summary['generation_sha256'].items():same(sha(path),value,'unchanged scored input')
    scores={};paired=None
    for arm in ('baseline16','adaptive16'):
        folder=base/'full'/arm;m=json.loads((folder/'manifest.json').read_text());config=m['config']
        same(m['status'],'completed','predecessor generation');same(m['completed_samples'],164,'predecessor count')
        same(config['source']['source_sha256'],pins,'actual predecessor source');same(config['source']['git_commit'],GEN,'actual generation commit')
        rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        same([r['task_id'] for r in rows],[f'HumanEval/{i}' for i in range(164)],'predecessor ordered tasks')
        fields=('task_id','problem_sha256','prompt','prompt_sha256','prompt_token_ids','seed','effective_max_new_tokens')
        inputs=[{k:r[k] for k in fields} for r in rows]
        if paired is not None:same(inputs,paired,'predecessor paired inputs')
        paired=inputs
        path=base/'full_scores'/(arm+'.json');same(sha(path),summary['scores'][arm]['sha256'],'predecessor score bytes')
        scores[arm]=read_scores(path,dict(config=config,rows={r['task_id']:r for r in rows},samples_sha256=sha(folder/'samples.jsonl')))
    for suite in ('base','plus'):
        metric=paired_metric(scores['baseline16']['rows'],scores['adaptive16']['rows'],suite)
        for key in ('wins','losses','ties','win_task_ids','loss_task_ids'):
            same(metric[key],summary['metrics'][suite][key],'predecessor paired scores '+key)
    return dict(status='PASS',comparison_sha256=sha(base/'full_scores/comparison.json'),n=164,
                generation_sha256=summary['generation_sha256'],source_commit=GEN,scorer_commit=SCORE)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();root=a.project.resolve();pins=frozen_tree(ROOT,a.source_commit)
    output=root/'results/runs/20261004-parcae13-figure1b'
    claims=root/'results/.parcae-suite-claims';claims.mkdir(exist_ok=True)
    claim=claims/'parcae13-figure1b-hellaswag-v1.json'
    require(not output.exists(),'Fresh output required')
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),output=str(output),source_commit=a.source_commit),f)
    output.mkdir();(output/'logs').mkdir()
    state=dict(status='waiting_predecessor',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,
               gpu='2',task='hellaswag',arms=list(ARMS),jobs=[],started_at=datetime.now(timezone.utc).isoformat())
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        tmp=output/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(output/'status.json')
    def verify():same(frozen_tree(ROOT,a.source_commit),pins,'frozen source')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',
             HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    def run(name,command,gpu=False):
        verify()
        if gpu:
            while True:
                device=gpu_status('2');state.update(status='waiting_gpu',gpu_status=device);save()
                if device['ready']:break
                time.sleep(30)
        verify();state['status']=name
        record=dict(name=name,command=list(map(str,command)),started_at=datetime.now(timezone.utc).isoformat())
        state['jobs'].append(record);save()
        with (output/'logs'/(name+'.log')).open('x') as log:
            child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save();verify()
        if code:raise RuntimeError(name+' failed; preserve output and inspect')
    save()
    try:
        while True:
            verify();proof=validate_predecessor(root)
            if proof is not None:break
            time.sleep(30)
        (output/'predecessor_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
        model=root/'models/Parcae-1.3B';gate=output/'gpu_gate/result.json'
        run('gpu_gate',[sys.executable,'-u',ROOT/'scripts/gpu_smoke_parcae.py','--model',model,'--output',gate.parent],gpu=True)
        validate_report(json.loads(gate.read_text()));gate_sha=sha(gate)
        for scope in ('smoke','full'):
            directories=[]
            for arm in ARMS:
                same(sha(gate),gate_sha,'GPU certificate');directory=output/(scope+'_'+arm);directories.append(directory)
                command=[sys.executable,'-u',ROOT/'scripts/evaluate_parcae.py','--model',model,'--task','hellaswag',
                         '--arm',arm,'--output',directory,'--smoke',gate,
                         '--prompt-audit',root/'results/parcae_prompt_audit_20261004/audit',
                         '--harness-reference','/path/to/your/workspace/lmharness_nov23/lm-evaluation-harness',
                         '--dataset-cache',root/'.cache/parcae-prompt-audit-20261004',
                         '--hub-cache','/path/to/your/workspace/.cache/huggingface/hub','--device','cuda:0']
                if scope=='smoke':command+=['--limit','2']
                run(scope+'_'+arm,command,gpu=True)
            comparison=output/(scope+'_comparison.json')
            run(scope+'_compare',[sys.executable,ROOT/'scripts/compare_parcae_figure1b.py','--runs',*directories,'--output',comparison])
            result=json.loads(comparison.read_text());same(result['status'],'PASS','comparison')
            same(result['n_documents'],2 if scope=='smoke' else 10042,'complete task scope')
            same(result['is_full_split'],scope=='full','full/smoke scope')
        verify();state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
