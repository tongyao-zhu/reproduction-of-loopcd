"""One claimed Figure1b Qwen queue: existing gate, smoke+score, full378+score."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from qwen_mbpp_protocol import sha,same,frozen_tree,GATE_COMMIT,runtime,gate_sources
from launch_huginn_r16_suite import gpu_status
from generate_humaneval import atomic_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','model','preflight'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gate-pid',type=int,required=True);a=p.parse_args()
    pins=frozen_tree(ROOT,a.source_commit);gate_sources(a.project,pins)
    pre=json.loads((a.preflight/'result.json').read_text())
    for k,v in dict(status='PASS',source_commit=a.source_commit,source_sha256=pins,tests_run=6,errors=0,failures=0,skipped=0,
                    prompt_count=378,synthetic_rows_validated=756,cuda_initialized=False,checkpoint_loaded=False).items():same(pre[k],v,'CPU task gate '+k)
    same(pre['tests_sha256'],sha(ROOT/'tests/test_qwen_mbpp.py'),'task tests');same(pre['log_sha256'],sha(a.preflight/'tests.log'),'task log')
    original=subprocess.check_output(['git','show',a.source_commit+':tests/test_qwen_mbpp.py'],cwd=ROOT)
    if original!=(ROOT/'tests/test_qwen_mbpp.py').read_bytes():raise ValueError('CPU tests changed')
    same(pre['runtime'],runtime(),'task runtime')
    gate=a.project/'results/runs/20261004-qwen-gpu-gate-v3/result.json'
    base=a.project/'results/runs/20261004-qwen-mbpp-figure1b'
    claims=a.project/'results/.qwen-claims';claims.mkdir(exist_ok=True)
    with (claims/'figure1b-mbpp-task-v1.json').open('x') as f:json.dump(dict(pid=os.getpid(),output=str(base),source_commit=a.source_commit),f)
    base.mkdir(exist_ok=False);queue=base/'queue';queue.mkdir()
    state=dict(status='waiting_existing_gpu_gate',pid=os.getpid(),gate_pid=a.gate_pid,source_commit=a.source_commit,
               source_sha256=pins,preflight_sha256=sha(a.preflight/'result.json'),commands=[])
    def save():state.update(updated_at=datetime.now(timezone.utc).isoformat());atomic_json(queue/'status.json',state)
    def verify():
        same(frozen_tree(ROOT,a.source_commit),pins,'unchanged task source')
        same(sha(a.preflight/'result.json'),state['preflight_sha256'],'unchanged CPU task certificate')
    def run(command,label,gpu=False):
        verify()
        if gpu:
            while True:
                s=gpu_status('2');state.update(status='waiting_gpu',gpu=s);save()
                if s['ready']:break
                time.sleep(30);verify()
        state.update(status=label);record=dict(command=[str(x) for x in command],started_at=datetime.now(timezone.utc).isoformat())
        state['commands'].append(record);save()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2' if gpu else '',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        with (queue/(label+'.log')).open('x') as f:
            child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(returncode=code,finished_at=datetime.now(timezone.utc).isoformat());save()
        if code:raise ValueError(label+' failed; retain all output and inspect')
    save()
    try:
        while True:
            verify();g=json.loads(gate.read_text());state.update(gate_status=g['status']);save()
            if g['status']=='PASS':break
            if g['status']=='FAIL':raise ValueError('Existing GPU gate failed')
            proc=Path('/proc')/str(a.gate_pid)
            if not proc.exists() or b'gpu_smoke_qwen_loop_v3.py' not in (proc/'cmdline').read_bytes():
                if json.loads(gate.read_text())['status']=='PASS':continue
                raise ValueError('Existing gate exited without PASS')
            time.sleep(30)
        common=['--project',a.project,'--model',a.model,'--data',a.project/'data/mbpp/MbppPlus-v0.2.0.jsonl','--source-commit',a.source_commit]
        for scope in ('smoke','full'):
            command=[os.sys.executable,'-u',ROOT/'scripts/generate_qwen_mbpp.py',*common,'--output',base/scope]
            command+=['--limit','2'] if scope=='smoke' else ['--smoke',base/'smoke']
            run(command,'generate_'+scope,gpu=True)
            command=[os.sys.executable,'-u',ROOT/'scripts/score_qwen_mbpp.py',*common,'--generation',base/scope,'--output',base/(scope+'_scores')]
            if scope=='smoke':command.append('--allow-smoke')
            run(command,'score_'+scope)
            summary=json.loads((base/(scope+'_scores')/'comparison.json').read_text())
            same(summary['status'],'PASS','integration');same(summary['n'],2 if scope=='smoke' else 378,'scope');same(summary['full_378'],scope=='full','full flag')
        verify();state.update(status='completed');save()
    except BaseException as e:state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
