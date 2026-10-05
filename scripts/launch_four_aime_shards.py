"""Fresh-only four-GPU disjoint continuation with per-model complete scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loopcd_repro.aime_protocol import atomic_json,stamp
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retired',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();pins=frozen_tree(ROOT,a.source_commit)
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='starting',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,jobs=[],scores={})
    def save():
        state['updated_at']=stamp();atomic_json(a.output/'queue.json',state)
    save();children=[]
    try:
        # Preserve existing model/GPU association for the prefix continuations.
        for size,name,gpu,start,end in [('14','Ouro-1.4B-Thinking','2',0,18),
                ('26','Ouro-2.6B-Thinking','1',0,18),('14','Ouro-1.4B-Thinking','0',18,30),
                ('26','Ouro-2.6B-Thinking','3',18,30)]:
            run=a.output/f'{size}-{start:02d}-{end:02d}'
            while True:
                state.update(status='waiting_gpu',waiting_gpu=gpu,gpu_check=gpu_status(gpu));save()
                if state['gpu_check']['ready']:break
                time.sleep(15)
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_vllm_aime_shard.py'),'--project',str(a.project),
                 '--model',str(a.project/'models'/name),'--proof',str(a.project/'results/vllm_budget_verified_20261004.json'),
                 '--output',str(run),'--source-commit',a.source_commit,'--gpu',gpu,
                 '--task-start',str(start),'--task-end',str(end)]
            if start==0:cmd+=['--predecessor',str(a.retired/size)]
            with (a.output/(run.name+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);state['jobs'].append(dict(size=size,model=name,gpu=gpu,run=str(run),pid=child.pid,command=cmd));save()
        state['status']='generating';save()
        while True:
            for child,job in zip(children,state['jobs']):
                job['returncode']=child.poll()
                if job['returncode'] is not None and job['returncode']!=0:
                    raise RuntimeError('Shard failed; other running shards retained, no automatic retry: '+job['run'])
            for size in ('14','26'):
                jobs=[j for j in state['jobs'] if j['size']==size]
                if size not in state['scores'] and all(j['returncode']==0 for j in jobs):
                    target=a.output/(size+'-scores')
                    cmd=[sys.executable,str(ROOT/'scripts/score_sharded_aime.py'),'--project',str(a.project),
                         '--model',str(a.project/'models'/jobs[0]['model']),'--runs',*[j['run'] for j in jobs],
                         '--output',str(target)]
                    with (a.output/(size+'-scores.log')).open('xb') as log:
                        subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,
                                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
                    state['scores'][size]=str(target/'comparison.json');save()
            if len(state['scores'])==2:break
            save();time.sleep(15)
        assert frozen_tree(ROOT,a.source_commit)==pins
        state['status']='completed';save()
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise

if __name__=='__main__':main()
