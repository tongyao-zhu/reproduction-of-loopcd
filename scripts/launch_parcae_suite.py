"""Fresh-only Parcae seven-arm queue, after completed MBPP generation on GPU3.

No code generation or generated-code execution occurs in this queue. The MBPP
CPU isolation scorer proceeds independently after releasing its generation GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from launch_aime_suite import source_tree, atomic_json, sha
from launch_huginn_r16_suite import gpu_status
from launch_mbpp_suite import verify_generation
from gpu_smoke_parcae import validate_report

TASKS = ('sciq', 'piqa', 'arc_challenge', 'arc_easy', 'winogrande', 'hellaswag', 'mmlu')
ARMS = ('baseline8', 'fixed8', 'adaptive8', 'hidden8', 'baseline4', 'fixed4', 'hidden4')
PREDECESSOR_COMMIT = 'd04ccabf6797a68bb9ca7f964a6f0acb27f2155a'


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessor_ready(matrix):
    if matrix.get('source_commit') != PREDECESSOR_COMMIT or str(matrix.get('gpu')) != '3':
        raise ValueError('Unexpected predecessor source or assigned GPU')
    if matrix.get('status') == 'failed' or any(r.get('status') == 'failed' for r in matrix.get('jobs', {}).values()):
        raise ValueError('Predecessor generation failed; retain outputs for inspection')
    if matrix.get('status') not in {'starting', 'running', 'completed_generation_unscored'}:
        raise ValueError('Unknown predecessor status')
    if matrix['status'] != 'completed_generation_unscored':
        return False
    if matrix.get('smoke_gate') != 'passed' or set(matrix.get('jobs', {})) != {'smoke', 'full'}:
        raise ValueError('Incomplete predecessor stages')
    if any(row.get('status') != 'completed' or row.get('exit_code') != 0 for row in matrix['jobs'].values()):
        raise ValueError('Incomplete predecessor jobs')
    return True


def claim(directory, output, commit):
    directory.mkdir(parents=True, exist_ok=True)
    # Permanent protocol-specific claim prevents duplicate queues even with a
    # different output path. It is never removed automatically after failure.
    path = directory / 'parcae-1.3b-seven-task-v1.json'
    with path.open('x') as stream:
        json.dump(dict(pid=os.getpid(), output=str(output), source_commit=commit, created_at=now()), stream, indent=2)
        stream.write('\n')
    return path


def stop_owned_child(child):
    """Only signal the process group created for this exact child."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def verify_generation_sources(directory, pins):
    """Bind the actual four generation manifests to the predecessor commit."""
    records = {}
    for arm in ('baseline32', 'hidden32', 'baseline16', 'hidden16'):
        path = Path(directory) / arm / 'manifest.json'
        source = json.loads(path.read_text())['config']['source']
        if source.get('git_commit') != PREDECESSOR_COMMIT or source.get('source_sha256') != pins:
            raise ValueError('Actual MBPP generation source differs from frozen predecessor: ' + arm)
        records[arm] = dict(manifest_sha256=sha(path), git_commit=source['git_commit'])
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--claim-dir', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--mbpp-data', type=Path, required=True)
    p.add_argument('--prompt-audit', type=Path, required=True)
    p.add_argument('--harness-reference', type=Path, required=True)
    p.add_argument('--dataset-cache', type=Path, required=True)
    p.add_argument('--hub-cache', type=Path, required=True)
    p.add_argument('--gpu', choices=['3'], default='3')
    p.add_argument('--poll-seconds', type=float, default=30)
    args = p.parse_args()
    if not 0 < args.poll_seconds <= 300:
        raise ValueError('Invalid polling interval')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Fresh output is required; no implicit resume')
    pins = source_tree(ROOT, args.source_commit, ['scripts/evaluate_parcae.py', 'scripts/compare_parcae.py',
                       'scripts/gpu_smoke_parcae.py', 'src/loopcd_repro/parcae_mc.py', 'configs/parcae_1_3b_mc.json'])
    predecessor = args.predecessor.resolve()
    predecessor_ready(json.loads((predecessor / 'matrix.json').read_text()))
    claim_path = claim(args.claim_dir.resolve(), args.output, args.source_commit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    stopped = threading.Event()
    def stop(*_): stopped.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = dict(schema_version=1, status='waiting', pid=os.getpid(), started_at=now(), source_commit=args.source_commit,
                 source_root=str(ROOT), source_sha256=pins, gpu=args.gpu, claim=str(claim_path),
                 predecessor=str(predecessor), tasks=list(TASKS), arms=list(ARMS), jobs={}, comparisons={},
                 smoke_gate='pending', full_suite_verified=False,
                 resource_policy='No GPU child until MBPP full generation is verified and actual GPU has no compute PID and <1000 MiB; no foreign signals.',
                 recovery_policy='Fresh-only, permanent claim, fail closed; never retry or overwrite automatically.')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': args.gpu,
           'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
    def save():
        state['updated_at'] = now(); atomic_json(args.output / 'matrix.json', state)
    def check_sources():
        if source_tree(ROOT, args.source_commit) != pins:
            raise ValueError('Frozen source changed')
    def check_stop():
        if stopped.is_set(): raise InterruptedError('Queue interrupted; outputs retained')
    def execute(name, command, gpu=False, comparison=False):
        check_stop(); check_sources()
        section = state['comparisons'] if comparison else state['jobs']
        section[name] = dict(status='waiting', command=command)
        if gpu:
            while True:
                check_stop()
                device = gpu_status(args.gpu)
                section[name].update(status='waiting_for_gpu', device=device)
                state['status'] = 'waiting'; save()
                if device['ready']: break
                stopped.wait(args.poll_seconds)
        check_stop(); check_sources()
        with (args.output / 'logs' / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                section[name].update(status='running', pid=child.pid, started_at=now())
                state['status'] = 'running'; save()
                while child.poll() is None:
                    stopped.wait(1)
                    check_stop()
                code = child.returncode
            except BaseException as error:
                # Includes failed state-file writes and unexpected poll errors.
                stop_owned_child(child)
                section[name].update(status='cancelled' if isinstance(error, InterruptedError) else 'failed',
                                     exit_code=child.returncode, error=repr(error), finished_at=now())
                raise
            finally:
                stop_owned_child(child)
        section[name].update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=now())
        save(); check_sources()
        if code: raise RuntimeError(f'{name} exited {code}; see retained log')
    save()
    try:
        while True:
            check_stop()
            before = json.loads((predecessor / 'matrix.json').read_text())
            state['predecessor_status'] = before['status']; save()
            if predecessor_ready(before): break
            stopped.wait(args.poll_seconds)
        old_pins = source_tree(Path(before['source_root']), PREDECESSOR_COMMIT)
        if before['source_sha256'] != old_pins:
            raise ValueError('Predecessor source differs from its original manifest')
        verification = verify_generation(predecessor / 'full', args.mbpp_data, 378, True)
        verification['frozen_generation_sources'] = verify_generation_sources(predecessor / 'full', old_pins)
        atomic_json(args.output / 'predecessor_verification.json', verification)
        state['predecessor_verified'] = True
        state['predecessor_matrix_sha256'] = sha(predecessor / 'matrix.json'); save()
        execute('gpu_gate', [sys.executable, '-u', str(ROOT/'scripts/gpu_smoke_parcae.py'), '--model', str(args.model.absolute()),
                            '--output', str(args.output/'gpu_gate')], gpu=True)
        gate = args.output/'gpu_gate/result.json'
        state['gpu_gate'] = dict(sha256=sha(gate), validation=validate_report(json.loads(gate.read_text())))
        save()
        for smoke, task in [(True, 'sciq')] + [(False, task) for task in TASKS]:
            directories = []
            for arm in ARMS:
                name = ('smoke_' if smoke else '') + task + '_' + arm
                directory = args.output/name; directories.append(directory)
                command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_parcae.py'), '--model', str(args.model.absolute()),
                           '--task', task, '--arm', arm, '--output', str(directory), '--smoke', str(gate),
                           '--prompt-audit', str(args.prompt_audit.absolute()), '--harness-reference', str(args.harness_reference.absolute()),
                           '--dataset-cache', str(args.dataset_cache.absolute()), '--hub-cache', str(args.hub_cache.absolute()), '--device', 'cuda:0']
                if smoke: command += ['--limit', '2']
                execute(name, command, gpu=True)
            name = ('smoke_' if smoke else '') + task + '_comparison'
            path = args.output / (name + '.json')
            execute(name, [sys.executable, '-u', str(ROOT/'scripts/compare_parcae.py'), '--runs', *map(str, directories),
                           '--output', str(path)], comparison=True)
            report = json.loads(path.read_text())
            # The comparer itself rejects an incomplete/unpaired seven-arm set.
            if report.get('task') != task or report.get('is_full_split') is not (not smoke):
                raise ValueError('Comparison returned the wrong task or split')
            if smoke:
                if report.get('n_documents') != 2: raise ValueError('Expected exactly two smoke documents')
                state['smoke_gate'] = 'passed'
            state['comparisons'][name]['sha256'] = sha(path); save()
        state.update(status='completed', full_suite_verified=True, finished_at=now()); save()
    except BaseException as error:
        state.update(status='cancelled' if isinstance(error, InterruptedError) else 'failed', error=repr(error), traceback=traceback.format_exc())
        save(); raise


if __name__ == '__main__':
    main()
