"""One fresh AIME vLLM run after the chosen GPU is genuinely free."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('project', 'engine', 'output', 'queue'):
        p.add_argument('--'+n, required=True, type=Path)
    for n in ('engine-commit', 'source-commit', 'gpu', 'model-name'):
        p.add_argument('--'+n, required=True)
    a = p.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins = frozen_tree(ROOT, a.source_commit)
    engine_pins = frozen_tree(a.engine, a.engine_commit)
    assert not a.output.exists()
    a.queue.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), source_commit=a.source_commit, source_sha256=pins,
                 engine_commit=a.engine_commit, engine_source_sha256=engine_pins, status='waiting_gpu')
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        temp = a.queue/'status.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(a.queue/'status.json')
    save()
    try:
        while True:
            state['gpu'] = gpu_status(a.gpu)
            save()
            if state['gpu']['ready']:
                break
            time.sleep(15)
        assert frozen_tree(a.engine, a.engine_commit) == engine_pins
        cmd = [sys.executable, '-u', str(a.engine/'scripts/run_vllm_aime.py'), '--project', str(a.project),
               '--model', str(a.project/'models'/a.model_name), '--proof', str(a.project/'results/vllm_budget_verified_20261004.json'),
               '--output', str(a.output), '--source-commit', a.engine_commit, '--gpu', a.gpu]
        with (a.queue/'generation.log').open('xb') as log:
            child = subprocess.Popen(cmd, cwd=a.engine, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', child_pid=child.pid, command=cmd)
            save()
            code = child.wait()
        state['returncode'] = code
        if code != 0:
            raise RuntimeError('Generation failed; no automatic retry')
        assert json.loads((a.output/'status.json').read_text())['status'] == 'completed'
        assert frozen_tree(ROOT, a.source_commit) == pins
        state['status'] = 'completed'
        save()
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
