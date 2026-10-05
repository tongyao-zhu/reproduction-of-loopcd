"""Figure1b ARC-C two-arm suite after the already queued 370M GPU gate."""
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
from loopcd_repro.parcae370_mc import validate_gpu_gate,sha256
from compare_parcae370 import require,same
GATE_COMMIT='f9e1304cb133ecc360db78c5158ef133dbc1ab1a'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--cpu-preflight',type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();pins=frozen_tree(ROOT,a.source_commit)
    cpu=json.loads(a.cpu_preflight.read_text());same(cpu['status'],'PASS','task CPU preflight')
    same(cpu['source_sha256'],pins,'task CPU source');same(cpu['source_commit'],a.source_commit,'task CPU commit')
    require(cpu['all_4687_audited_requests_loaded'] and cpu['paired_rng_initialization'] and not cpu['cuda_initialized'],'CPU checks incomplete')
    cpu_sha=sha256(a.cpu_preflight)
    output=root/'results/runs/20261004-parcae370-arc-figure1b'
    require(not output.exists(),'Fresh suite output required')
    claims=root/'results/.parcae-suite-claims';claims.mkdir(exist_ok=True)
    with (claims/'parcae370-figure1b-arc-v1.json').open('x') as f:json.dump(dict(pid=os.getpid(),source_commit=a.source_commit,output=str(output)),f)
    output.mkdir();(output/'logs').mkdir()
    state=dict(status='waiting_existing_gpu_gate',pid=os.getpid(),source_commit=a.source_commit,source_sha256=pins,jobs=[],gpu='3',cpu_preflight_sha256=cpu_sha)
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat();t=output/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(output/'status.json')
    def verify():
        same(frozen_tree(ROOT,a.source_commit),pins,'source');same(sha256(a.cpu_preflight),cpu_sha,'CPU evidence')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    def run(name,command,gpu=False):
        verify()
        if gpu:
            while True:
                device=gpu_status('3');state.update(status='waiting_gpu',gpu_status=device);save()
                if device['ready']:break
                time.sleep(30)
        verify();state['status']=name;job=dict(name=name,command=list(map(str,command)));state['jobs'].append(job);save()
        with (output/'logs'/(name+'.log')).open('x') as log:
            child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            job['pid']=child.pid;save();job['exit_code']=child.wait();save()
        verify()
        if job['exit_code']:raise RuntimeError(name+' failed; retain output')
    save()
    try:
        gate_parent=root/'results/runs/20261004-parcae370-gpu-gate'
        while True:
            d=json.loads((gate_parent/'status.json').read_text());same(d['source_commit'],GATE_COMMIT,'existing gate commit')
            if d['status']=='PASS':break
            require(d['status'] in ('waiting_gpu','running'),'Existing gate failed or unknown state')
            time.sleep(30)
        frozen_tree(root/'releases/parcae370-gpu-f9e1304',GATE_COMMIT)
        model=root/'models/Parcae-370M';gate=gate_parent/'gate/result.json'
        proof=validate_gpu_gate(gate,model,ROOT);same(proof['sha256'],d['gate_sha256'],'parent/certificate')
        (output/'gpu_verification.json').write_text(json.dumps(proof,indent=2)+'\n');gate_sha=sha256(gate)
        for scope in ('smoke','full'):
            directories=[]
            for arm in ('baseline8','adaptive8'):
                same(sha256(gate),gate_sha,'stable GPU certificate');dest=output/(scope+'_'+arm);directories.append(dest)
                command=[sys.executable,'-u',ROOT/'scripts/evaluate_parcae370.py','--model',model,'--task','arc_challenge','--arm',arm,'--output',dest,'--smoke',gate,'--prompt-audit',root/'results/runs/20261004-parcae370-cpu-checks/arc_prompts','--harness-reference','/path/to/your/workspace/lmharness_nov23/lm-evaluation-harness','--hub-cache','/path/to/your/workspace/.cache/huggingface/hub','--dataset-cache',root/'.cache/parcae370-arc-prompts-20261004','--device','cuda:0']
                if scope=='smoke':command+=['--limit','2']
                run(scope+'_'+arm,command,gpu=True)
            target=output/(scope+'_comparison.json')
            run(scope+'_compare',[sys.executable,ROOT/'scripts/compare_parcae370.py','--runs',*directories,'--output',target])
            result=json.loads(target.read_text());same(result['status'],'PASS','comparison');same(result['n_documents'],2 if scope=='smoke' else 1172,'scope');same(result['is_full_split'],scope=='full','full flag')
        state.update(status='completed');save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
