"""Run exactly two finite sampling-budget probes on one checked free GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','proof','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',default='2')
    a=p.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins=frozen_tree(ROOT,a.source_commit)
    proof=json.loads(a.proof.read_text())
    if proof['status']!='reference_and_forced_cache_verified' or len(proof['models'])!=2:raise ValueError('Independent reference proof missing')
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='checking',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,
        proof_sha256=hashlib.sha256(a.proof.read_bytes()).hexdigest(),jobs=[])
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat();tmp=a.output/'queue.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'queue.json')
    save()
    try:
        for model in ('Ouro-1.4B-Thinking','Ouro-2.6B-Thinking'):
            evidence=proof['models'][model];gate=a.project/evidence['status_path']
            if hashlib.sha256(gate.read_bytes()).hexdigest()!=evidence['status_sha256']:raise ValueError('Reference gate changed')
            state['status']='waiting_free_gpu_'+model;save()
            while True:
                state['gpu']=gpu_status(a.gpu);save()
                if state['gpu']['ready']:break
                time.sleep(15)
            out=a.output/model
            cmd=[sys.executable,'-u',str(ROOT/'scripts/probe_vllm_aime_budget.py'),'--model',str(a.project/'models'/model),
                 '--prompt-source',str(a.project/'results/runs/20261004-aime-figure1b'/model/'baseline/samples.jsonl'),
                 '--gate',str(gate),'--output',str(out),'--source-commit',a.source_commit,'--gpu',a.gpu]
            with (a.output/(model+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
                state['jobs'].append(dict(model=model,pid=child.pid,command=cmd));state['status']='running_'+model;save()
                code=child.wait();state['jobs'][-1]['returncode']=code;save()
            if code!=0:raise RuntimeError('Probe failed; do not retry in place')
            result=json.loads((out/'status.json').read_text())
            if result['status']!='completed':raise RuntimeError('Probe incomplete')
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed')
        state['status']='completed';save()
    except Exception:state['status']='failed';state['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
