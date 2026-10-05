"""Wait for full MBPP generation, then score once in the pinned v2 sandbox.

This coordinator never imports or executes generated solutions. Only the
fixed Linux chroot launcher may execute them, after its full canonical gate.
A durable claim prevents a second coordinator from rescoring the same run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ARMS = ('baseline32', 'hidden32', 'baseline16', 'hidden16')
GENERATION_COMMIT = 'd04ccabf6797a68bb9ca7f964a6f0acb27f2155a'
GENERATION_RELEASE = 'releases/mbpp-d04ccab'
SCORER_RELEASE = 'releases/mbpp-scorer-v2-33fa680'
SCORER_SHA256 = '054e6996df65eaf1426da8f867086e75fc9e37d6ce9aeb56f296041301ff0e23'
DATA_SHA256 = 'b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad'
RELEASE = Path(__file__).resolve().parents[1]
FAILURE_STATUSES = {'failed', 'error', 'cancelled', 'canceled', 'interrupted'}


class GenerationFailed(RuntimeError):
    pass


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_hashes(pins):
    for name, expected in pins.items():
        if not Path(name).is_file() or sha(name) != expected:
            raise RuntimeError(f'Pinned source or evidence changed: {name}')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_generation_source(root):
    """Verify every generation-provenance source byte against the exact Git tree."""
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])
    if git('rev-parse', 'HEAD').decode().strip() != GENERATION_COMMIT:
        raise RuntimeError('Generation release HEAD is not frozen d04ccab')
    tracked = git('ls-tree', '-r', '--name-only', GENERATION_COMMIT).decode().splitlines()
    expected = {name for name in tracked if name.split('/')[0] in {'src', 'scripts', 'configs'}
                and Path(name).suffix in {'.py', '.json', '.yaml'}}
    found = {str(p.relative_to(root)) for folder in ('src', 'scripts', 'configs')
             for p in (root / folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.yaml'}}
    required = {'scripts/generate_mbpp.py', 'scripts/launch_mbpp_suite.py',
                'scripts/launch_huginn_r16_suite.py', 'src/loopcd_repro/huginn.py',
                'src/loopcd_repro/runtime.py'}
    if found != expected or not required <= found:
        raise RuntimeError('Generation frozen source file set differs from Git')
    hashes = {}
    for name in sorted(found):
        expected_hash = hashlib.sha256(git('show', f'{GENERATION_COMMIT}:{name}')).hexdigest()
        if sha(root / name) != expected_hash:
            raise RuntimeError(f'Generation release differs from Git: {name}')
        hashes[name] = expected_hash
    return hashes


def generation_snapshot(directory):
    """Read live manifests; sample files are opened only after all four finish."""
    summary, manifests, pins = {}, {}, {}
    for arm in ARMS:
        path = Path(directory) / arm / 'manifest.json'
        if not path.is_file():
            summary[arm] = {'status': 'manifest_missing'}
            continue
        content = path.read_bytes()
        record = json.loads(content)
        status = record.get('status')
        summary[arm] = {key: record.get(key) for key in
                        ('status', 'expected_samples', 'completed_samples', 'updated_at', 'is_full_split')}
        if status in FAILURE_STATUSES:
            raise GenerationFailed(f'{arm} generation ended with {status}: {record.get("error")}')
        if status not in {'running', 'completed', 'queued', 'waiting', 'initializing'}:
            raise ValueError(f'Unexpected generation status for {arm}: {status!r}')
        if record.get('is_full_split') is not True or record.get('config', {}).get('limit') is not None:
            raise ValueError(f'{arm} is a debug subset; this coordinator only accepts full MBPP')
        if type(record.get('expected_samples')) is not int or record['expected_samples'] != 378:
            raise ValueError(f'{arm} expected count must be 378')
        completed = record.get('completed_samples')
        if type(completed) is not int or not 0 <= completed <= 378:
            raise ValueError(f'{arm} has an invalid completed count')
        if status == 'completed' and completed != 378:
            raise ValueError(f'{arm} completed manifest is incomplete')
        manifests[arm] = record
        pins[str(path)] = hashlib.sha256(content).hexdigest()
    if len(manifests) != 4 or any(m['status'] != 'completed' for m in manifests.values()):
        return False, summary, {}
    shared_ids = None
    for arm, manifest in manifests.items():
        ids = manifest['config'].get('task_ids')
        if not isinstance(ids, list) or len(ids) != 378 or len(set(ids)) != 378:
            raise ValueError(f'{arm} requires 378 unique expected IDs')
        if shared_ids is not None and ids != shared_ids:
            raise ValueError('Four arms have different ordered task splits')
        shared_ids = ids
        path = Path(directory) / arm / 'samples.jsonl'
        content = path.read_bytes()
        if not content.endswith(b'\n'):
            raise ValueError(f'{arm} final sample line is not terminated')
        rows = [json.loads(line) for line in content.splitlines()]
        if [row.get('task_id') for row in rows] != ids:
            raise ValueError(f'{arm} has missing, duplicated, unknown, or unordered tasks')
        pins[str(path)] = hashlib.sha256(content).hexdigest()
        summary[arm]['verified_unique_samples'] = len(rows)
    verify_hashes(pins)
    return True, summary, pins


def verify_generation_provenance(generations, source_hashes):
    for arm in ARMS:
        source = generations[arm]['config']['source']
        if source.get('git_commit') != GENERATION_COMMIT or source.get('source_sha256') != source_hashes:
            raise RuntimeError(f'{arm} provenance differs from the frozen generation tree')


def acquire_claim(project, generation, output):
    """A permanent reservation also rejects retries after a failed coordinator."""
    key = hashlib.sha256(str(Path(generation).resolve()).encode()).hexdigest()
    folder = Path(project) / 'results' / '.mbpp-score-claims'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + '.json')
    record = {'generation': str(Path(generation).resolve()), 'output': str(Path(output).resolve()),
              'pid': os.getpid(), 'started_at': timestamp(), 'coordinator_sha256': sha(__file__),
              'policy': 'No automatic retries or claim removal; inspect recorded output before any manual recovery'}
    try:
        with path.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f'This generation already has a scoring reservation: {path}') from error
    return path


def process_is_active(pid, proc_root='/proc'):
    path=Path(proc_root,str(pid))
    if not path.exists():return False
    try:
        state=(path/'stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:
        if not path.exists():return False
        raise
    # An exited child may remain unreaped under container PID1. It cannot run.
    return state not in ('Z','X')


def acquire_recovery_claim(project, generation, output, previous):
    """Explicit one-time recovery of the observed pre-score Infinity hash bug."""
    previous=Path(previous).resolve();status_path=previous/'status.json'
    old=json.loads(status_path.read_text())
    if (old.get('status')!='failed' or old.get('failed_phase')!='validating_generation'
            or old.get('error')!="ValueError('Out of range float values are not JSON compliant')"
            or old.get('commands')!=[] or old.get('completed_arms')!=[] or old.get('active_child_pid') is not None):
        raise RuntimeError('Not the verified pre-score serialization failure')
    if process_is_active(old['pid']):raise RuntimeError('Previous coordinator is still active')
    if old['generation']!=str(Path(generation).resolve()) or old['output']!=str(previous):
        raise RuntimeError('Recovery generation/output mismatch')
    if {p.name for p in previous.iterdir()}!={'status.json'}:raise RuntimeError('Previous output contains unexpected evidence or scores')
    claim=Path(old['claim']);record=json.loads(claim.read_text())
    if (record['generation']!=old['generation'] or record['output']!=old['output'] or record['pid']!=old['pid']
            or record['coordinator_sha256']!='c4a9fc51606fed2384c7e398ebc46651d21eef5de8cc48a17749974a11216c61'):
        raise RuntimeError('Original claim does not bind the known failed coordinator')
    verify_hashes(old['generation_sha256']);verify_hashes(old['source_sha256']);verify_hashes(old['pinned_evidence'])
    destination=claim.with_name(claim.stem+'.recovery-'+sha(status_path)+'.json')
    with destination.open('x') as f:
        json.dump(dict(generation=str(Path(generation).resolve()),output=str(Path(output).resolve()),pid=os.getpid(),
                       previous_status=str(status_path),previous_status_sha256=sha(status_path),previous_claim_sha256=sha(claim),
                       coordinator_sha256=sha(__file__),started_at=timestamp()),f,indent=2)
    return destination


def verify_gate_acceptance(record, manifest_hash, gate, manifest, data):
    expected = {'status': 'PASS', 'actual_scoring_gate_accepted': True,
                'manifest_sha256': manifest_hash, 'canonical_evidence_sha256': gate['evidence_sha256'],
                'scorer_source_sha256': SCORER_SHA256, 'uid': manifest['identity']['uid'],
                'tasks': 378, 'base_tests': 1174, 'plus_tests': 39841, 'model_code_executed': False}
    if any(record.get(key) != value or type(record.get(key)) is not type(value) for key, value in expected.items()):
        raise RuntimeError('gate_acceptance.json does not certify this exact v2 runtime and canonical evidence')
    if data['sha256'] != DATA_SHA256 or len(data['task_ids']) != 378:
        raise RuntimeError('Wrong MBPP dataset')
    for suite, total in (('base', 1174), ('plus', 39841)):
        if sum(count[suite] for count in data['counts'].values()) != total:
            raise RuntimeError('Pinned dataset test counts changed')


def sandbox_gate(project, acceptance_path, data):
    """Invoke the exact frozen scoring gates, but never invoke inside/evaluation."""
    project = Path(project)
    scorer = project / SCORER_RELEASE / 'scripts/run_mbpp_sandbox.py'
    if sha(scorer) != SCORER_SHA256:
        raise RuntimeError('Scorer is not the pinned MBPP v2 runner')
    module = load_module('_mbpp_v2_sandbox_gate', scorer)
    box = project / '.sandbox/mbpp-v2'
    root, manifest_hash = module.verify_runtime(box)
    manifest_path = box / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('dataset_sha256') != DATA_SHA256 or manifest.get('expected_tasks') != 378:
        raise RuntimeError('Runtime does not contain complete pinned MBPP data')
    safety_path, gate_path = box / 'safety.json', box / 'canonical_validation.json'
    module.verify_safety_gate(json.loads(safety_path.read_text()), manifest_hash, manifest['evaluator_patch'])
    gate = json.loads(gate_path.read_text())
    module.verify_canonical_gate(gate, manifest_hash, data['task_ids'], data['counts'], manifest['evaluator_patch'])
    module.verify_identity(manifest['identity'])
    canonical_path = Path(gate['result_path']).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    if canonical_path != acceptance_path.parent / 'canonical378.json':
        raise RuntimeError('Acceptance and canonical evidence must belong to the same v2 validation run')
    verify_gate_acceptance(json.loads(acceptance_path.read_text()), manifest_hash, gate, manifest, data)
    paths = (scorer, manifest_path, safety_path, gate_path, canonical_path, acceptance_path,
             root / 'data/MbppPlus-v0.2.0.jsonl')
    pins = {str(path): sha(path) for path in paths}
    return {'scorer': scorer, 'sandbox': box, 'canonical': canonical_path, 'manifest': manifest_path,
            'pins': pins, 'manifest_sha256': manifest_hash, 'canonical_sha256': gate['evidence_sha256'],
            'identity': manifest['identity']}


def score_command(scorer, sandbox, samples, output, timeout):
    return ['/usr/bin/python3', str(scorer), '--sandbox', str(sandbox), '--samples', str(samples),
            '--output', str(output), '--timeout', str(timeout)]


def execute(command, output, label, state, save_state, poll_seconds=30, timeout=None):
    stdout_path, stderr_path = output / f'{label}.stdout.log', output / f'{label}.stderr.log'
    entry = {'label': label, 'command': [str(x) for x in command], 'started_at': timestamp(),
             'stdout': str(stdout_path), 'stderr': str(stderr_path), 'returncode': None}
    state['commands'].append(entry)
    save_state()
    started, child = time.monotonic(), None
    try:
        with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
            child = subprocess.Popen(entry['command'], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     close_fds=True)
            entry['pid'] = child.pid
            state['active_child_pid'] = child.pid
            save_state()
            while True:
                remaining = None if timeout is None else timeout - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(entry['command'], timeout)
                try:
                    entry['returncode'] = child.wait(timeout=poll_seconds if remaining is None else min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    save_state()
        if entry['returncode'] != 0:
            raise RuntimeError(f'{label} failed with exit code {entry["returncode"]}; inspect {stderr_path}')
    except BaseException:
        if child is not None and child.poll() is None:
            # The pinned scorer catches SIGTERM and cleans its own isolated process group.
            child.terminate()
            try:
                entry['returncode'] = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                entry['cleanup_pending'] = True
                entry['cleanup_note'] = 'Scorer did not finish SIGTERM cleanup; inspect its PID before any manual action'
        raise
    finally:
        state['active_child_pid'] = child.pid if child is not None and child.poll() is None else None
        entry.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        save_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True, help='The full/ directory, never smoke/')
    parser.add_argument('--output', type=Path, required=True, help='A new directory; existing evidence is never overwritten')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--gate-acceptance', type=Path)
    parser.add_argument('--recover-from', type=Path, help='Explicit verified pre-score Infinity failure; old output/claim remain unchanged')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--score-timeout', type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.score_timeout <= 0:
        parser.error('poll-seconds must be in [1,60]; score-timeout must be positive')
    project, generation, output = args.project.resolve(), args.generation.resolve(), args.output.resolve()
    data_path = (args.data or project / 'data/mbpp/MbppPlus-v0.2.0.jsonl').resolve()
    acceptance_path = (args.gate_acceptance or project / 'results/runs/20261004-mbpp-scorer-v2/gate_acceptance.json').resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / 'status.json'
    state = {'schema_version': 1, 'status': 'starting', 'phase': 'preflight', 'pid': os.getpid(),
             'started_at': timestamp(), 'project': str(project), 'generation': str(generation),
             'output': str(output), 'release': str(RELEASE), 'commands': [], 'completed_arms': [],
             'poll_seconds': args.poll_seconds, 'score_timeout': args.score_timeout,
             'active_child_pid': None, 'no_automatic_retry': True}
    def save_state():
        state['updated_at'] = timestamp()
        atomic_json(state_path, state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Coordinator received signal {signum}')
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    save_state()
    try:
        claim = acquire_recovery_claim(project, generation, output, args.recover_from) if args.recover_from else acquire_claim(project, generation, output)
        state['claim'] = str(claim)
        comparer_path = RELEASE / 'scripts/compare_mbpp.py'
        comparer = load_module('_frozen_mbpp_comparer', comparer_path)
        data = comparer.load_data(data_path)
        source_root = project / GENERATION_RELEASE
        source_hashes = frozen_generation_source(source_root)
        source_pins = {str(source_root / name): value for name, value in source_hashes.items()}
        source_pins.update({str(path): sha(path) for path in (Path(__file__).resolve(), comparer_path, data_path)})
        gates = sandbox_gate(project, acceptance_path, data)
        pins = {**source_pins, **gates['pins']}
        state.update(status='waiting', phase='waiting_for_full_generation', source_sha256=source_pins,
                     canonical_gate={'manifest_sha256': gates['manifest_sha256'],
                                     'canonical_sha256': gates['canonical_sha256'], 'identity': gates['identity'],
                                     'gate_acceptance': str(acceptance_path)},
                     pinned_evidence=gates['pins'])
        save_state()
        while True:
            verify_hashes(pins)
            ready, snapshot, input_pins = generation_snapshot(generation)
            state['generation_snapshot'] = snapshot
            save_state()
            if ready:
                break
            time.sleep(args.poll_seconds)
        state.update(status='validating', phase='validating_generation', generation_sha256=input_pins)
        save_state()
        pins.update(input_pins)
        generations = {arm: comparer.read_generation(generation / arm, data) for arm in ARMS}
        comparer.validate_generation_pairs(list(generations.values()))
        verify_generation_provenance(generations, source_hashes)
        # Retain the exact pre-scoring gate used by the immutable generation launcher.
        sys.path.insert(0, str(source_root / 'scripts'))
        try:
            launcher = load_module('_frozen_mbpp_generation_gate', source_root / 'scripts/launch_mbpp_suite.py')
            frozen_check = launcher.verify_generation(generation, data_path, 378, True)
        finally:
            sys.path.pop(0)
        atomic_json(output / 'generation_verification.json', frozen_check)
        verify_hashes(pins)
        state.update(status='scoring', phase='scoring')
        save_state()
        canonical_context = comparer.load_canonical(gates['canonical'], gates['manifest'], data)
        score_paths = []
        for arm in ARMS:
            verify_hashes(pins)
            current_gates = sandbox_gate(project, acceptance_path, data)
            if current_gates['pins'] != gates['pins']:
                raise RuntimeError('Canonical/runtime evidence changed between scoring arms')
            score = output / f'{arm}.json'
            if score.exists():
                raise FileExistsError(f'Refusing to overwrite score evidence: {score}')
            state['active_arm'] = arm
            execute(score_command(gates['scorer'], gates['sandbox'], generation / arm / 'samples.jsonl', score,
                                  args.score_timeout), output, 'score_' + arm, state, save_state, args.poll_seconds)
            verify_hashes(pins)
            # The comparer independently rejects incomplete scoring and checks its input hash.
            score_hash = sha(score)
            comparer.read_scores(score, generations[arm], canonical_context)
            if sha(score) != score_hash:
                raise RuntimeError(f'{arm} score changed during validation')
            pins[str(score)] = score_hash
            score_paths.append(score)
            state['completed_arms'].append(arm)
            state.setdefault('score_sha256', {})[str(score)] = sha(score)
            save_state()
        state.update(status='comparing', phase='comparing', active_arm=None)
        save_state()
        comparison = output / 'comparison.json'
        verify_hashes(pins)
        command = ['/usr/bin/python3', str(comparer_path), '--runs', *[str(generation / arm) for arm in ARMS],
                   '--scores', *[str(path) for path in score_paths], '--data', str(data_path),
                   '--canonical-evidence', str(gates['canonical']), '--sandbox-manifest', str(gates['manifest']),
                   '--output', str(comparison)]
        execute(command, output, 'compare', state, save_state, args.poll_seconds, timeout=120)
        verify_hashes(pins)
        result = json.loads(comparison.read_text())
        if result.get('full_378_verified') is not True or result.get('n_tasks') != 378:
            raise RuntimeError('Comparison did not certify the complete paired 378-task result')
        state.update(status='completed', phase='completed', finished_at=timestamp(),
                     comparison=str(comparison), comparison_sha256=sha(comparison),
                     n_tasks=378, full_378_verified=True)
        save_state()
        print(json.dumps({'status': 'completed', 'comparison': str(comparison), 'n_tasks': 378}), flush=True)
    except BaseException as error:
        state.update(status='generation_failed' if isinstance(error, GenerationFailed) else
                     'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     failed_phase=state['phase'], phase='stopped', finished_at=timestamp(),
                     error=repr(error), traceback=traceback.format_exc())
        save_state()
        print(json.dumps({'status': state['status'], 'error': str(error), 'status_file': str(state_path)}), flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
