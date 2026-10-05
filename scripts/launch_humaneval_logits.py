"""Continue the existing two-task smoke into one full Figure 1b pair and isolated scores.

Fresh-only coordinator; immutable releases, exclusive claim, no retries and no
other benchmark/model. Existing smoke is observed, never submitted twice.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from humaneval_logits_checks import sha
GEN='29ea1d89a557d5307afaa1586636e2ed3cb779c1'
SCORE='546bb4f3c8bd3422b8ae8e4d23f8d6229b88ee7e'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--smoke-pid',type=int,required=True)
    a=p.parse_args();root=a.project.resolve()
    base=root/'results/runs/20261004-huginn-he-logits-v2'
    genroot=root/'releases/he-logits-29ea1d8';scoreroot=root/'releases/he-logits-score-546bb4f'
    tree=frozen_tree(ROOT,a.source_commit)
    genpins=frozen_tree(genroot,GEN);scorepins=frozen_tree(scoreroot,SCORE)
    gate=root/'results/runs/20261004-huginn-logits-gate/gate.json';gate_hash=sha(gate)
    output=base/'queue';output.mkdir(exist_ok=False)
    state=dict(status='waiting_smoke',pid=os.getpid(),smoke_pid=a.smoke_pid,source_commit=a.source_commit,
               generator_commit=GEN,scorer_commit=SCORE,started_at=datetime.now(timezone.utc).isoformat(),commands=[])
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        t=output/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(output/'status.json')
    def verify():
        if frozen_tree(ROOT,a.source_commit)!=tree or frozen_tree(genroot,GEN)!=genpins or frozen_tree(scoreroot,SCORE)!=scorepins or sha(gate)!=gate_hash:
            raise ValueError('Frozen source/certificate changed')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='2',
             HF_MODULES_CACHE=str(root/'.cache/he-logits-29ea1d8'))
    def run(command,label):
        verify();state['status']=label
        record=dict(command=[str(x) for x in command],started_at=datetime.now(timezone.utc).isoformat())
        state['commands'].append(record);save()
        with (output/(label+'.log')).open('x') as log:
            child=subprocess.Popen(record['command'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(returncode=code,finished_at=datetime.now(timezone.utc).isoformat());save()
        if code:raise RuntimeError(label+' failed; inspect preserved log/output')
    def scoring(where,destination,smoke=False):
        cmd=[a.python,'-u',scoreroot/'scripts/score_humaneval_logits.py','--generation',where,
             '--generator-root',genroot,'--generator-commit',GEN,'--source-commit',SCORE,
             '--model',root/'models/huginn-0125','--data','/path/to/your/workspace/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl',
             '--gate',gate,'--sandbox-project',root,'--output',destination]
        if smoke:cmd.append('--allow-smoke')
        return cmd
    save()
    try:
        while True:
            snapshots={}
            for arm in ('baseline16','adaptive16'):
                path=base/'smoke'/arm/'manifest.json'
                if not path.is_file():raise ValueError('Existing smoke manifest missing')
                m=json.loads(path.read_text());snapshots[arm]={k:m.get(k) for k in ('status','completed_samples','expected_samples')}
                if m['status'] not in ('running','completed'):raise ValueError('Existing smoke failed')
            state['smoke']=snapshots;save()
            if all(x['status']=='completed' and x['completed_samples']==x['expected_samples']==2 for x in snapshots.values()):
                proof=base/'smoke/pair_validation.json'
                if proof.exists() and json.loads(proof.read_text()).get('status')=='PASS':break
            proc=Path('/proc')/str(a.smoke_pid)
            if not proc.exists() or str(genroot/'scripts/generate_humaneval_logits.py').encode() not in (proc/'cmdline').read_bytes():
                raise RuntimeError('Smoke process exited without verified pair completion')
            time.sleep(30)
        run(scoring(base/'smoke',base/'smoke_scores',True),'score_smoke')
        summary=json.loads((base/'smoke_scores/comparison.json').read_text())
        if summary['status']!='PASS' or summary['n']!=2 or summary['full_164'] is not False:raise ValueError('Bad smoke scoring proof')
        while True:
            status=gpu_status('2');state.update(status='waiting_gpu',gpu=status);save()
            if status['ready']:break
            time.sleep(30)
        run([a.python,'-u',genroot/'scripts/generate_humaneval_logits.py','--model',root/'models/huginn-0125',
             '--data','/path/to/your/workspace/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl','--output',base/'full',
             '--smoke',base/'smoke','--gate',gate,'--source-commit',GEN,'--gpu','2'],'generate_full')
        run(scoring(base/'full',base/'full_scores'),'score_full')
        summary=json.loads((base/'full_scores/comparison.json').read_text())
        if summary['status']!='PASS' or summary['n']!=164 or summary['full_164'] is not True:raise ValueError('Bad full scoring proof')
        verify();state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
