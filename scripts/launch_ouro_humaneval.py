"""Observe the existing Ouro GPU gate, then smoke/score/full164/score once."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from ouro_humaneval_checks import validate_gate,sha,GATE_COMMIT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--python',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gate-pid',type=int,required=True)
    a=p.parse_args();root=a.project.resolve()
    base=root/'results/runs/20261004-ouro-he-suite'
    gate=root/'results/runs/20261004-ouro-he-gpu/gate.json'
    gate_root=root/'releases/ouro-he-49d0718'
    pins=frozen_tree(ROOT,a.source_commit);original=frozen_tree(gate_root,GATE_COMMIT)
    for name,value in original.items():
        if pins.get(name)!=value:raise ValueError('Changed computational gate source')
    base.mkdir(exist_ok=False);output=base/'queue';output.mkdir()
    state=dict(status='waiting_existing_gpu_gate',pid=os.getpid(),gate_pid=a.gate_pid,
               source_commit=a.source_commit,source_sha256=pins,started_at=datetime.now(timezone.utc).isoformat(),commands=[])
    gate_hash=None
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        t=output/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(output/'status.json')
    def verify():
        if frozen_tree(ROOT,a.source_commit)!=pins or frozen_tree(gate_root,GATE_COMMIT)!=original:
            raise ValueError('Changed frozen source')
        if gate_hash is not None and sha(gate)!=gate_hash:raise ValueError('Changed GPU certificate')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='3',
             HF_MODULES_CACHE=str(root/'.cache/ouro-he-suite'))
    def wait_gpu():
        while True:
            verify();status=gpu_status('3');state.update(status='waiting_gpu',gpu=status);save()
            if status['ready']:return
            time.sleep(30)
    def run(command,label):
        verify();state['status']=label
        record=dict(command=[str(x) for x in command],started_at=datetime.now(timezone.utc).isoformat())
        state['commands'].append(record);save()
        with (output/(label+'.log')).open('x') as log:
            child=subprocess.Popen(record['command'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(returncode=code,finished_at=datetime.now(timezone.utc).isoformat());save()
        if code:raise RuntimeError(label+' failed; preserve output and inspect')
    common=['--model',root/'models/Ouro-2.6B','--data','/path/to/your/workspace/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl',
            '--gate',gate,'--source-commit',a.source_commit]
    save()
    try:
        while True:
            verify();report=json.loads(gate.read_text());state['gate_status']=report['status'];save()
            if report['status']=='PASS':
                validate_gate(report,pins);gate_hash=sha(gate);break
            if report['status'] not in ('WAITING','RUNNING'):raise ValueError('Existing GPU gate failed')
            proc=Path('/proc')/str(a.gate_pid)
            if not proc.exists() or str(gate_root/'scripts/gpu_smoke_ouro_humaneval.py').encode() not in (proc/'cmdline').read_bytes():
                # Re-read to allow a just-completed gate to publish its final state.
                if json.loads(gate.read_text())['status']=='PASS':continue
                raise RuntimeError('Existing gate process exited without PASS')
            time.sleep(30)
        for scope in ('smoke','full'):
            wait_gpu()
            command=[a.python,'-u',ROOT/'scripts/generate_ouro_humaneval.py',*common,'--output',base/scope,'--gpu','3']
            command+=['--limit','2'] if scope=='smoke' else ['--smoke',base/'smoke']
            run(command,'generate_'+scope)
            command=[a.python,'-u',ROOT/'scripts/score_ouro_humaneval.py',*common,
                     '--generation',base/scope,'--generator-root',ROOT,'--generator-commit',a.source_commit,
                     '--sandbox-project',root,'--output',base/(scope+'_scores')]
            if scope=='smoke':command.append('--allow-smoke')
            run(command,'score_'+scope)
            summary=json.loads((base/(scope+'_scores')/'comparison.json').read_text())
            if summary['status']!='PASS' or summary['n']!=(2 if scope=='smoke' else 164) or summary['full_164']!=(scope=='full'):
                raise ValueError('Wrong scope/incomplete scoring proof')
        verify();state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
