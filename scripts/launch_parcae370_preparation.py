"""Fresh CPU-only pinned download, preparation and native loading checks."""
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
    p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();root=a.project.resolve();pins=frozen_tree(ROOT,a.source_commit)
    output=root/'results/runs/20261004-parcae370-preparation'
    cache=root/'.cache/parcae370-pinned-20261004';model=root/'models/Parcae-370M'
    if cache.exists() or model.exists():raise FileExistsError('Fresh cache/model required')
    source=Path(json.loads((root/'models/Parcae-1.3B/model_provenance.json').read_text())['source']['original_root'])
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,
               started_at=datetime.now(timezone.utc).isoformat(),jobs=[],gpu_used=False)
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        t=output/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(output/'status.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    jobs=[('download',['download_parcae370.py','--cache-dir',cache,'--tokenizer',root/'models/Parcae-1.3B/tokenizer.json']),
          ('prepare',['prepare_parcae370.py','--cache-dir',cache,'--source-root',source,'--output',model]),
          ('native_cpu',['parcae370_cpu_preflight.py','--model',model,'--output',output/'native_cpu'])]
    save()
    try:
        for phase,args in jobs:
            if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Changed source')
            command=[sys.executable,'-u',str(ROOT/'scripts'/args[0]),*map(str,args[1:])]
            job=dict(phase=phase,command=command,started_at=datetime.now(timezone.utc).isoformat())
            state['jobs'].append(job);state['phase']=phase;save()
            with (output/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['pid']=child.pid;save();code=child.wait()
            job.update(returncode=code,finished_at=datetime.now(timezone.utc).isoformat());save()
            if code:raise RuntimeError(phase+' failed; retain all artifacts')
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Changed final source')
        report=json.loads((output/'native_cpu/result.json').read_text())
        if report['status']!='PASS':raise ValueError('Native CPU checks incomplete')
        state.update(status='PASS',phase='completed',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));save();raise

if __name__=='__main__':main()
