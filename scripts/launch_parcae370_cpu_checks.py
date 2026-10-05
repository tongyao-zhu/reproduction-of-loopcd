"""Fresh, frozen CPU-only 370M guidance and full ARC-C input checks."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from continue_aime_figure1b import frozen_tree
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,type=Path);p.add_argument('--source-commit',required=True)
    p.add_argument('--harness-reference',required=True,type=Path);p.add_argument('--hub-cache',required=True,type=Path)
    a=p.parse_args();root=a.project.resolve();pins=frozen_tree(ROOT,a.source_commit)
    out=root/'results/runs/20261004-parcae370-cpu-checks'
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,jobs=[],gpu_used=False)
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        t=out/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out/'status.json')
    model=root/'models/Parcae-370M'
    jobs=[('guidance',['parcae370_guidance_preflight.py','--model',model,'--output',out/'guidance','--source-commit',a.source_commit]),
          ('arc_prompts',['audit_parcae370_prompts.py','--model',model,'--output',out/'arc_prompts',
            '--harness-reference',a.harness_reference,'--hub-cache',a.hub_cache,
            '--dataset-cache',root/'.cache/parcae370-arc-prompts-20261004'])]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    save()
    try:
        for phase,args in jobs:
            if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Changed source')
            cmd=[sys.executable,'-u',str(ROOT/'scripts'/args[0]),*map(str,args[1:])]
            job=dict(phase=phase,command=cmd);state['jobs'].append(job);state['phase']=phase;save()
            with (out/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                job['pid']=child.pid;save();job['returncode']=child.wait();save()
            if job['returncode'] or json.loads((out/phase/'result.json').read_text())['status']!='PASS':
                raise RuntimeError(phase+' failed; retain evidence')
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Changed final source')
        state.update(status='PASS',phase='completed');save()
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));save();raise

if __name__=='__main__':main()
