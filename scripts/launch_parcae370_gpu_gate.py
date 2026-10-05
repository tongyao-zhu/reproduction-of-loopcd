"""One claimed 370M GPU gate, after CPU checks and only on an idle GPU3."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from continue_aime_figure1b import frozen_tree
from launch_huginn_r16_suite import gpu_status
from parcae370_guidance_preflight import validate_records
from gpu_smoke_parcae370 import validate_report
ROOT=Path(__file__).resolve().parents[1]
CPU_COMMIT='f1480d89a913748f2bb68b064af7390bbae98093'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_cpu(root, source):
    cpu=root/'releases/parcae370-cpu-f1480d8'
    pins=frozen_tree(cpu,CPU_COMMIT)
    current=frozen_tree(ROOT,source)
    if any(current.get(name)!=value for name,value in pins.items()):
        raise ValueError('CPU-validated source changed')
    # New gate uses the exact numerical oracle already exercised on CPU.
    def oracle(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='numerical_cases'))
    if oracle(cpu/'scripts/gpu_smoke_parcae.py')!=oracle(ROOT/'scripts/gpu_smoke_parcae370.py'):
        raise ValueError('Numerical oracle differs from CPU check')
    out=root/'results/runs/20261004-parcae370-cpu-checks'
    paths=[out/'status.json',out/'guidance/result.json',out/'arc_prompts/result.json']
    parent,guidance,prompt=[json.loads(p.read_text()) for p in paths]
    if any(x.get('status')!='PASS' for x in (parent,guidance,prompt)):
        raise ValueError('Complete CPU guidance and input checks required')
    if any(x.get('source_commit')!=CPU_COMMIT or x.get('source_sha256')!=pins for x in (parent,guidance)):
        raise ValueError('CPU source binding changed')
    validate_records(guidance['records'])
    if guidance['model_provenance']['sha256']!=sha(root/'models/Parcae-370M/model_provenance.json'):
        raise ValueError('Model binding changed')
    if (prompt.get('profile')!='parcae370_arc_challenge_v1' or prompt.get('shots')!={'arc_challenge':25}
        or prompt.get('total_documents')!=1172 or prompt.get('total_requests')!=4687
        or prompt.get('registered_hf_policy_ready') is not True or prompt.get('cuda_initialized') is not False
        or prompt.get('model_manifest_sha256')!=guidance['model_provenance']['sha256']):
        raise ValueError('Incomplete 370M ARC-C input audit')
    record=prompt['tasks']['arc_challenge']['records']
    if record['file']!='arc_challenge.jsonl.gz':raise ValueError('Unexpected audit record file')
    path=out/'arc_prompts'/record['file']
    if sha(path)!=record['sha256']:raise ValueError('Prompt record hash changed')
    return {str(p):sha(p) for p in paths+[path]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,type=Path);p.add_argument('--source-commit',required=True)
    a=p.parse_args();root=a.project.resolve();pins=frozen_tree(ROOT,a.source_commit)
    evidence=validate_cpu(root,a.source_commit)
    out=root/'results/runs/20261004-parcae370-gpu-gate'
    if out.exists():raise FileExistsError('Fresh output required')
    claims=root/'results/.parcae-suite-claims';claims.mkdir(exist_ok=True)
    with (claims/'parcae370-figure1b-gpu-gate-v1.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),source_commit=a.source_commit,output=str(out)),f)
    out.mkdir()
    state=dict(status='waiting_gpu',pid=os.getpid(),gpu='3',source_commit=a.source_commit,
               source_sha256=pins,cpu_evidence_sha256=evidence,scope='GPU correctness gate only; no benchmark queue')
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        t=out/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out/'status.json')
    save()
    try:
        while True:
            device=gpu_status('3');state['gpu_status']=device;save()
            if device['ready']:break
            time.sleep(30)
        if frozen_tree(ROOT,a.source_commit)!=pins or validate_cpu(root,a.source_commit)!=evidence:
            raise ValueError('Source/CPU evidence changed before GPU launch')
        command=[sys.executable,'-u',str(ROOT/'scripts/gpu_smoke_parcae370.py'),'--model',str(root/'models/Parcae-370M'),'--output',str(out/'gate')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4')
        state.update(status='running',command=command);save()
        with (out/'gate.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();state['returncode']=child.wait();save()
        if state['returncode']:raise RuntimeError('GPU gate failed; retain output')
        validate_report(json.loads((out/'gate/result.json').read_text()))
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed during GPU gate')
        state.update(status='PASS',gate_sha256=sha(out/'gate/result.json'));save()
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));save();raise

if __name__=='__main__':main()
