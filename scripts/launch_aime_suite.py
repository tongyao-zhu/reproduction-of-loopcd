"""Queue the fixed two-model AIME reconstruction only after verified R16 completion.

No retries/resume, no foreign-process signals, and no GPU allocation until
all predecessor outputs and both model-specific smoke gates are verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('Ouro-1.4B-Thinking', 'Ouro-2.6B-Thinking')
YEARS = (2024, 2025)
ARMS = ('baseline', 'fixed', 'adaptive')
MC_TASKS = ('sciq', 'piqa', 'arc_challenge', 'arc_easy', 'winogrande', 'hellaswag', 'mmlu')
MC_ARMS = ('baseline16', 'hidden16')
MC_CANDIDATES = ('hidden16', 'half16')
MC_SOURCE_COMMIT = 'b88c8cc90c15765f88a11acebf76ed4137935a5c'
MC_LAUNCHER_COMMIT = '26ee9ac2b7d43abcf2c966c186e4bdb3eacaf0d5'
PROTOCOL_ID = 'ouro-thinking-aime-reconstruction-v1'
FAILED = {'failed', 'cancelled', 'canceled', 'interrupted', 'blocked_by_failed_worker',
          'blocked_by_failed_arm', 'blocked_by_gate_failure', 'skipped_smoke_only'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, record):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def inventory(path):
    path = Path(path)
    return {str(p.relative_to(path)): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(path.rglob('*')) if p.is_file()}


def source_tree(root, commit, required=()):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])
    if git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('Source HEAD differs from the declared frozen commit')
    tracked = git('ls-tree', '-r', '--name-only', commit).decode().splitlines()
    expected = {p for p in tracked if p.split('/')[0] in {'src', 'scripts', 'configs'}
                and Path(p).suffix in {'.py', '.json', '.yaml'}}
    found = {str(p.relative_to(root)) for folder in ('src', 'scripts', 'configs')
             for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.yaml'}}
    if found != expected or not set(required) <= found:
        raise ValueError('Frozen source file set differs from the required Git tree')
    hashes = {}
    for name in sorted(found):
        value = hashlib.sha256(git('show', f'{commit}:{name}')).hexdigest()
        if sha(root/name) != value:
            raise ValueError('Source differs from frozen Git bytes: '+name)
        hashes[name] = value
    return hashes


def verify_pins(pins):
    for name, checksum in pins.items():
        if not Path(name).is_file() or sha(name) != checksum:
            raise ValueError('Pinned source/input changed: '+name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def predecessor_snapshot(path):
    if not Path(path).is_file():
        return None, {'status': 'matrix_missing'}
    matrix = json.loads(Path(path).read_text())
    if matrix.get('status') in FAILED:
        raise ValueError('Predecessor R16 suite failed; AIME cannot use its GPUs')
    for section in ('jobs', 'comparisons'):
        if any(row.get('status') in FAILED for row in matrix.get(section, {}).values()):
            raise ValueError('Predecessor R16 '+section+' contains a failed/blocked job')
    summary = {'status': matrix.get('status'), 'full_suite_verified': matrix.get('full_suite_verified'),
               'completed_jobs': sum(x.get('status') == 'completed' for x in matrix.get('jobs', {}).values()),
               'completed_comparisons': sum(x.get('status') == 'completed' for x in matrix.get('comparisons', {}).values()),
               'updated_at': matrix.get('updated_at')}
    if matrix.get('status') != 'completed':
        if matrix.get('status') not in {'waiting', 'running'}:
            raise ValueError('Unknown predecessor R16 suite status')
        return None, summary
    if matrix.get('full_suite_verified') is not True or matrix.get('smoke_only') is not False:
        raise ValueError('Predecessor did not verify its full suite')
    expected_jobs = {f'{t}_{a}' for t in MC_TASKS for a in MC_ARMS} | {f'smoke_sciq_{a}' for a in MC_ARMS}
    expected_compares = {f'{t}_{a}_comparison' for t in MC_TASKS for a in MC_CANDIDATES} | {f'smoke_sciq_{a}_comparison' for a in MC_CANDIDATES}
    for section, expected in (('jobs', expected_jobs), ('comparisons', expected_compares)):
        records = matrix.get(section, {})
        if set(records) != expected or any(r.get('status') != 'completed' or r.get('exit_code') != 0 for r in records.values()):
            raise ValueError('Predecessor has missing/incomplete '+section)
    if matrix.get('smoke_gate', {}).get('status') != 'passed':
        raise ValueError('Predecessor smoke barrier has not passed')
    return matrix, summary


def verify_predecessor(project, matrix_path, matrix):
    """Recompute all 14 full comparisons from their immutable paired inputs."""
    project, matrix_path = Path(project).resolve(), Path(matrix_path).resolve()
    suite = project/'results/runs/20261004-huginn-r16-suite'
    old_suite = project/'results/runs/20261003-huginn-mc-suite'
    source = project/'releases/huginn-mc-final'
    launcher_root = project/'releases/phase2-26ee9ac'
    if matrix_path != suite/'matrix.json':
        raise ValueError('Expected the current R16 predecessor matrix')
    if (Path(matrix['source_root']).resolve() != source or matrix['source_commit'] != MC_SOURCE_COMMIT
            or Path(matrix['half_depth_suite']).resolve() != old_suite):
        raise ValueError('Predecessor source or reused-run path differs from the fixed suite')
    source_hashes = source_tree(source, MC_SOURCE_COMMIT, ('scripts/compare_huginn.py', 'scripts/evaluate_huginn.py'))
    launcher_hashes = source_tree(launcher_root, MC_LAUNCHER_COMMIT, ('scripts/launch_huginn_r16_suite.py',))
    launch = matrix['launcher']
    if (matrix['source_sha256'] != source_hashes or launch['commit'] != MC_LAUNCHER_COMMIT
            or Path(launch['release']).resolve() != launcher_root
            or Path(launch['path']).resolve() != launcher_root/'scripts/launch_huginn_r16_suite.py'
            or launch['sha256'] != launcher_hashes['scripts/launch_huginn_r16_suite.py']):
        raise ValueError('Predecessor frozen source provenance is inconsistent')
    pins = {str(matrix_path): sha(matrix_path)}
    pins.update({str(source/p): h for p, h in source_hashes.items()})
    pins.update({str(launcher_root/p): h for p, h in launcher_hashes.items()})
    prior_compare_module = sys.modules.pop('compare', None)
    sys.path.insert(0, str(source/'scripts'))
    try:
        comparer = load_module('_aime_predecessor_compare', source/'scripts/compare_huginn.py')
        validations = {}
        for task in MC_TASKS:
            for candidate in MC_CANDIDATES:
                name = f'{task}_{candidate}_comparison'
                entry = matrix['comparisons'][name]
                left = suite/f'{task}_baseline16'
                right = (suite/f'{task}_hidden16' if candidate == 'hidden16' else old_suite/f'{task}_hidden16')
                expected_paths = {str(left), str(right)}
                if set(entry.get('input_files', {})) != expected_paths:
                    raise ValueError('Predecessor comparison input paths changed: '+name)
                for directory in (left, right):
                    files = inventory(directory)
                    if files != entry['input_files'][str(directory)]:
                        raise ValueError('Predecessor comparison input bytes changed: '+name)
                    pins.update({str(directory/p): record['sha256'] for p, record in files.items()})
                    manifest = json.loads((directory/'manifest.json').read_text())
                    if (manifest.get('status') != 'completed' or manifest.get('is_full_split') is not True
                            or manifest.get('limit') is not None or manifest.get('task') != task
                            or manifest['provenance']['source_sha256'] != source_hashes):
                        raise ValueError('Predecessor full manifest/source invalid: '+str(directory))
                path = suite/(name+'.json')
                if Path(entry['output']).resolve() != path or sha(path) != entry['sha256']:
                    raise ValueError('Predecessor comparison evidence changed: '+name)
                saved = json.loads(path.read_text())
                actual = comparer.compare_huginn(left, right)
                if any(actual.get(k) != v for k, v in saved.items() if k != 'created_at') or set(actual) != set(saved):
                    raise ValueError('Recomputed predecessor comparison disagrees: '+name)
                if actual['full_split_count_verified'] is not True or actual['is_full_split'] is not True:
                    raise ValueError('Predecessor full document counts were not established')
                expected_ref = 7 if candidate == 'hidden16' else 6
                for side, mode, ref in (('baseline', 'baseline', 7), ('candidate', 'hidden', expected_ref)):
                    if actual[side]['guidance'] != {'mode': mode, 'total_loops': 16, 'reference_loop': ref, 'omega': .5}:
                        raise ValueError('Predecessor guidance mismatch')
                pins[str(path)] = sha(path)
                validations[name] = {'n_documents': actual['n_documents'], 'sha256': pins[str(path)],
                                     'native_initialization_stream': actual['validation']['native_initialization_stream']}
    finally:
        sys.path.pop(0)
        sys.modules.pop('compare', None)
        if prior_compare_module is not None:
            sys.modules['compare'] = prior_compare_module
    verify_pins(pins)
    return {'status': 'PASS', 'full_jobs': 14, 'full_comparisons': 14, 'pins': pins, 'comparisons': validations}


def gpu_status(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--id='+str(gpu), '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('GPU query did not identify exactly one device')
    uuid, memory = [x.strip() for x in rows[0].split(',')]
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    pids = [int(row.split(',')[1]) for row in output.splitlines() if row.strip() and row.split(',')[0].strip() == uuid]
    return {'uuid': uuid, 'memory_mib': int(memory), 'compute_pids': pids}


def device_idle(record):
    if (not isinstance(record.get('uuid'), str) or not record['uuid']
            or type(record.get('memory_mib')) is not int or record['memory_mib'] < 0
            or not isinstance(record.get('compute_pids'), list)
            or any(type(pid) is not int or pid <= 0 for pid in record['compute_pids'])):
        raise ValueError('GPU state is incomplete; refuse to infer availability')
    return record['memory_mib'] < 1000 and not record['compute_pids']


def claim_batch(project, output):
    key = hashlib.sha256(json.dumps({'protocol': PROTOCOL_ID, 'models': MODELS, 'years': YEARS}, sort_keys=True).encode()).hexdigest()
    folder = Path(project)/'results/.aime-suite-claims'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(key+'.json')
    with path.open('x') as stream:
        json.dump({'protocol': PROTOCOL_ID, 'models': MODELS, 'years': YEARS, 'pid': os.getpid(),
                   'output': str(Path(output).resolve()), 'created_at': now(), 'automatic_retry': False}, stream, indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return path


def current_sources(root):
    return {str(p.relative_to(root)): sha(p) for folder in ('src', 'scripts', 'configs')
            for p in sorted((root/folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}


class AIMEQueue:
    def __init__(self, args, gpu_probe=gpu_status):
        self.args = args
        self.project = Path(args.project).resolve()
        self.output = Path(args.output).resolve()
        self.source = ROOT
        self.protocol = Path(args.protocol or self.project/'data/aime/protocol-v1.json').resolve()
        self.data = Path(args.data_root or self.project/'data/aime/prepared-v1').resolve()
        self.models = Path(args.models_root or self.project/'models').resolve()
        # Preserve the virtualenv entry point: resolving its symlink can select
        # the base interpreter and silently change site-packages/sys.prefix.
        self.python = str(Path(args.python).absolute())
        self.gpu_probe = gpu_probe
        if not 1 <= args.poll_seconds <= 60:
            raise ValueError('Polling interval must be 1..60 seconds')
        if not Path(self.python).is_file():
            raise ValueError('Explicit Python executable is missing')
        required = ('scripts/launch_aime_suite.py', 'scripts/gpu_smoke_aime.py',
                    'scripts/generate_aime.py', 'scripts/score_aime.py', 'src/loopcd_repro/aime_protocol.py')
        self.sources = source_tree(self.source, args.source_commit, required)
        self.helper = load_module('_aime_queue_protocol', self.source/'src/loopcd_repro/aime_protocol.py')
        self.scorer = load_module('_aime_queue_scorer', self.source/'scripts/score_aime.py')
        protocol = self.helper.load_protocol(self.protocol)
        self.scoring_data = self.scorer.load_scoring_data(self.protocol, self.data)
        self.identities = {model: self.helper.read_model_identity(self.models/model, protocol) for model in MODELS}
        for model, identity in self.identities.items():
            if identity['repo_id'] != 'ByteDance/'+model:
                raise ValueError('Model directory has the wrong identity')
        self.pins = {str(self.protocol): sha(self.protocol)}
        self.pins.update({str(self.data/name): sha(self.data/name) for name in
                         ('manifest.json', 'aime2024.questions.jsonl', 'aime2024.answers.jsonl',
                          'aime2025.questions.jsonl', 'aime2025.answers.jsonl')})
        for model in MODELS:
            directory = self.models/model
            for name in ('model_provenance.json', 'modeling_ouro.py', 'configuration_ouro.py', 'config.json',
                         'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
                         'special_tokens_map.json', 'model.safetensors'):
                path = directory/name
                self.pins[str(path)] = sha(path)
            declared = json.loads((directory/'model_provenance.json').read_text())['files']['model.safetensors']['weight_blob']
            if declared != self.pins[str(directory/'model.safetensors')]:
                raise ValueError('Model weight SHA256 does not match the prepared HF blob')
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'logs').mkdir()
        self.lock = threading.RLock()
        self.abort = threading.Event()
        self.cancelled = threading.Event()
        self.children = {}
        self.state = {'schema_version': 1, 'kind': 'aime_suite', 'status': 'waiting', 'pid': os.getpid(),
                      'created_at': now(), 'protocol_id': PROTOCOL_ID, 'source_root': str(self.source),
                      'source_commit': args.source_commit, 'source_sha256': self.sources,
                      'input_sha256': dict(self.pins), 'models': self.identities,
                      'gpus': dict(zip(MODELS, (0, 1))), 'years': list(YEARS), 'samples_per_problem': 16,
                      'automatic_retry': False, 'automatic_resume': False,
                      'resource_policy': 'R16 full suite verified; no compute PID and <1000 MiB before each GPU child',
                      'smoke_gate': {'status': 'waiting'}, 'jobs': {}, 'workers': {}, 'full_suite_verified': False}
        self.state['jobs']['canonical'] = {'status': 'pending', 'phase': 'canonical'}
        for model in MODELS:
            self.state['workers'][model] = {'status': 'waiting'}
            self.state['jobs'][f'smoke_{model}'] = {'status': 'pending', 'phase': 'smoke', 'model': model}
            for year in YEARS:
                for phase in ('generate', 'score'):
                    self.state['jobs'][f'{phase}_{model}_{year}'] = {'status': 'pending', 'phase': phase, 'model': model, 'year': year}
        self._save()
        try:
            self.state['claim'] = str(claim_batch(self.project, self.output))
            self._save()
        except BaseException as error:
            self.state.update(status='failed', error=str(error), finished_at=now())
            self._save()
            raise

    def _save(self):
        with self.lock:
            self.state['updated_at'] = now()
            atomic_json(self.output/'matrix.json', self.state)

    def _check_inputs(self):
        if self.abort.is_set():
            raise RuntimeError('Batch cancelled after an earlier failure or signal')
        if current_sources(self.source) != self.sources:
            raise ValueError('Frozen batch source changed; no further child may run')
        with self.lock:
            pins = dict(self.pins)
        verify_pins(pins)

    def _pin_files(self, paths):
        with self.lock:
            for path in paths:
                path = Path(path).resolve()
                actual = sha(path)
                if str(path) in self.pins and self.pins[str(path)] != actual:
                    raise ValueError('Evidence changed before pinning: '+str(path))
                self.pins[str(path)] = actual
            self.state['input_sha256'] = dict(self.pins)
            self._save()

    def _wait_gpu(self, model):
        gpu = self.state['gpus'][model]
        while not self.abort.is_set():
            snapshot = self.gpu_probe(gpu)
            idle = device_idle(snapshot)
            with self.lock:
                self.state['workers'][model].update(status='gpu_available' if idle else 'waiting_for_gpu',
                                                    gpu=gpu, gpu_snapshot=snapshot, checked_at=now())
                self._save()
            if idle:
                return
            self.abort.wait(self.args.poll_seconds)
        raise RuntimeError('Batch stopped while waiting for its GPU')

    def _terminate_owned(self, process):
        # Only Popen objects registered by this instance may be signalled.
        with self.lock:
            owned = self.children.get(process.pid) is process
        if not owned or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=20)

    def cancel(self, signum=None, frame=None):
        self.cancelled.set()
        self.abort.set()
        with self.lock:
            self.state['signal'] = signum
            self._save()
        # The poll loop owns cleanup. The signal handler does not wait or kill
        # a process that it did not launch.

    def _command(self, script, *args):
        return [self.python, str(self.source/'scripts'/script), *map(str, args)]

    def _job(self, name, command, validator, model=None):
        process = None
        try:
            self._check_inputs()
            if model is not None:
                self._wait_gpu(model)
                # A long wait may outlive an input change. Recheck bytes and
                # then take a fresh resource snapshot immediately before spawn.
                self._check_inputs()
                self._wait_gpu(model)
            with self.lock:
                if self.abort.is_set():
                    raise RuntimeError('Batch stopped before launching child')
                record = self.state['jobs'][name]
                record.update(status='running', command=command, started_at=now(), log=str(self.output/'logs'/(name+'.log')))
                env = {**os.environ, 'PYTHONPATH': str(self.source/'src'), 'HF_HUB_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'HF_MODULES_CACHE': str(self.output/'.cache/modules'/str(model or 'cpu')),
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
                       'CUDA_VISIBLE_DEVICES': str(self.state['gpus'][model]) if model is not None else ''}
                with Path(record['log']).open('x') as log:
                    process = subprocess.Popen(command, cwd=self.source, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                self.children[process.pid] = process
                record['pid'] = process.pid
                self._save()
            while process.poll() is None:
                if self.abort.wait(1):
                    self._terminate_owned(process)
                    raise RuntimeError('Batch stopped; owned child terminated')
            with self.lock:
                record.update(exit_code=process.returncode, process_finished_at=now())
                self._save()
            self._check_inputs()
            if process.returncode != 0:
                raise RuntimeError(f'Child {name} exited {process.returncode}; inspect its retained log')
            evidence = validator()
            self._check_inputs()
            with self.lock:
                record.update(status='completed', finished_at=now(), evidence=evidence)
                self._save()
            return evidence
        except BaseException as error:
            self.abort.set()
            if process is not None:
                self._terminate_owned(process)
            with self.lock:
                self.state['jobs'][name].update(status='cancelled' if self.cancelled.is_set() else 'failed',
                                               error=str(error), finished_at=now())
                self._save()
            raise
        finally:
            if process is not None:
                with self.lock:
                    self.children.pop(process.pid, None)

    def _canonical(self):
        output = self.output/'canonical'
        def validate():
            path = output/'canonical_gate.json'
            actual = json.loads(path.read_text())
            expected = self.scorer.canonical_gate(self.scoring_data)
            if any(actual.get(k) != v for k, v in expected.items()):
                raise ValueError('Canonical integer parser/data gate is incomplete or inconsistent')
            self._pin_files([path])
            return {'status': 'PASS', 'canonical_tasks': 60, 'path': str(path), 'sha256': sha(path)}
        return self._job('canonical', self._command('score_aime.py', '--canonical-only', '--protocol', self.protocol,
                         '--data-root', self.data, '--output', output), validate)

    def _predecessor(self):
        path = self.project/'results/runs/20261004-huginn-r16-suite/matrix.json'
        while not self.abort.is_set():
            matrix, brief = predecessor_snapshot(path)
            with self.lock:
                self.state['predecessor'] = {'matrix': str(path), **brief}
                self._save()
            if matrix is not None:
                evidence = verify_predecessor(self.project, path, matrix)
                target = self.output/'predecessor_verification.json'
                atomic_json(target, evidence)
                self._pin_files([target])
                with self.lock:
                    self.state['predecessor'].update(verification=str(target), sha256=sha(target), verified_at=now())
                    self._save()
                return evidence
            self.abort.wait(self.args.poll_seconds)
        raise RuntimeError('Batch stopped before predecessor verification')

    def _smoke(self, model):
        output = self.output/(f'smoke_{model}.json')
        def validate():
            report = self.helper.validate_smoke(output)
            bindings = report['bindings']
            if (bindings['model_identity'] != self.identities[model] or bindings['source_sha256'] != self.sources
                    or bindings['loaded_model_code_sha256'] != self.identities[model]['model_code_sha256']
                    or report['source']['git_commit'] != self.args.source_commit
                    or report['source']['source_sha256'] != self.sources):
                raise ValueError('GPU smoke model/code/source differs from frozen batch inputs')
            slots = 96 if model == MODELS[0] else 192
            if report['resource_gate']['cache_slots'] != slots:
                raise ValueError('Wrong model-specific native-cache capacity gate')
            self._pin_files([output])
            return {'status': 'PASS', 'path': str(output), 'sha256': sha(output), 'resource_gate': report['resource_gate']}
        return self._job(f'smoke_{model}', self._command('gpu_smoke_aime.py', '--model', self.models/model,
                         '--data-root', self.data, '--protocol', self.protocol, '--output', output, '--device', 'cuda:0'), validate, model)

    def _generation(self, model, year):
        output = self.output/'full'/model/str(year)
        smoke = self.output/(f'smoke_{model}.json')
        def validate():
            evidence = {}
            paired = []
            for arm in ARMS:
                directory = output/arm
                manifest = json.loads((directory/'manifest.json').read_text())
                config = manifest['config']
                source = config['source']
                if (manifest.get('status') != 'completed' or manifest.get('is_full_split') is not True
                        or manifest.get('expected_samples') != 480 or manifest.get('completed_samples') != 480
                        or source['git_commit'] != self.args.source_commit or source['source_sha256'] != self.sources
                        or config['model_identity'] != self.identities[model]
                        or config['guidance'] != self.helper.guidance_dict(arm, self.identities[model], self.scoring_data['protocol'])
                        or config['dataset']['year'] != year
                        or manifest['source_git_observed'] != self.args.source_commit
                        or manifest['gpu_smoke_sha256'] != sha(smoke)
                        or source['loaded_model_code_sha256'] != self.identities[model]['model_code_sha256']):
                    raise ValueError('Generation is incomplete or belongs to another model/source/protocol')
                if manifest['samples_sha256'] != sha(directory/'samples.jsonl'):
                    raise ValueError('Generated sample bytes do not match their manifest')
                # Read-only validation establishes complete ordered rows and
                # pairing before the separate scoring child computes accuracy.
                paired.append(self.scorer.read_generation(directory, self.scoring_data))
                self._pin_files([p for p in directory.rglob('*') if p.is_file()])
                evidence[arm] = {'path': str(directory), 'manifest_sha256': sha(directory/'manifest.json'),
                                 'samples_sha256': sha(directory/'samples.jsonl'), 'rows': 480}
            self.scorer.verify_paired_generations(paired)
            return evidence
        command = self._command('generate_aime.py', '--model', self.models/model, '--year', year,
                                '--data-root', self.data, '--protocol', self.protocol, '--output', output,
                                '--device', 'cuda:0', '--smoke', smoke)
        return self._job(f'generate_{model}_{year}', command, validate, model)

    def _score(self, model, year):
        output = self.output/'scores'/model/str(year)
        runs = [self.output/'full'/model/str(year)/arm for arm in ARMS]
        def validate():
            summary = json.loads((output/'summary.json').read_text())
            if (summary.get('status') != 'PASS' or summary.get('full_480_verified') is not True
                    or summary.get('n_tasks') != 30 or summary.get('samples_per_problem') != 16
                    or summary.get('rows_per_arm') != 480 or set(summary.get('arms', {})) != set(ARMS)
                    or summary.get('year') != year or summary.get('model_identity') != self.identities[model]
                    or summary.get('protocol_sha256') != sha(self.protocol)
                    or summary.get('data_manifest_sha256') != sha(self.data/'manifest.json')
                    or summary.get('data_files') != self.scoring_data['files']
                    or summary.get('gpu_smoke_sha256') != sha(self.output/f'smoke_{model}.json')
                    or summary['common_config']['source']['git_commit'] != self.args.source_commit
                    or summary['common_config']['source']['source_sha256'] != self.sources
                    or set(summary.get('pairs', {})) != {'fixed', 'adaptive'}):
                raise ValueError('Scorer did not establish three complete 480-row arms')
            validation = summary.get('validation', {})
            for key in ('all60_gold_canonical_gate', 'complete_ordered_480_per_arm', 'source_runtime_config_paired',
                        'public_prompts_reconstructed', 'token_ids_seed_question_hashes_paired',
                        'native_r4_cache_observations_verified', 'gold_question_file_hashes_verified'):
                if validation.get(key) is not True:
                    raise ValueError('Missing full scorer validation: '+key)
            if validation.get('model_answers_executed') is not False:
                raise ValueError('Scoring must only parse integer text')
            if (summary.get('scorer_source_sha256') != self.sources['scripts/score_aime.py']
                    or summary.get('canonical_gate_file_sha256') != sha(output/'canonical_gate.json')
                    or sha(output/'canonical_gate.json') != sha(self.output/'canonical/canonical_gate.json')):
                raise ValueError('Scorer source or canonical evidence differs from the accepted gate')
            for arm, directory in zip(ARMS, runs):
                evidence = summary['arms'][arm]
                if (evidence['manifest_sha256'] != sha(directory/'manifest.json')
                        or evidence['samples_sha256'] != sha(directory/'samples.jsonl')
                        or Path(evidence['run_path']).resolve() != directory.resolve()
                        or evidence.get('score_file_sha256') != sha(output/(arm+'.json'))):
                    raise ValueError('Scorer used different generation inputs')
            self._pin_files([p for p in output.rglob('*') if p.is_file()])
            return {'status': 'PASS', 'summary': str(output/'summary.json'), 'sha256': sha(output/'summary.json'),
                    'full_480_verified': True}
        command = self._command('score_aime.py', '--runs', *runs, '--protocol', self.protocol,
                                '--data-root', self.data, '--output', output)
        return self._job(f'score_{model}_{year}', command, validate)

    def _worker(self, model):
        try:
            for year in YEARS:
                if self.abort.is_set():
                    raise RuntimeError('Earlier job failed; later model/year is blocked')
                self._generation(model, year)
                self._score(model, year)
            with self.lock:
                self.state['workers'][model]['status'] = 'completed'
                self._save()
        except BaseException:
            self.abort.set()
            raise

    def run(self):
        try:
            self._canonical()
            self._predecessor()
            with self.lock:
                self.state['status'] = 'running'
                self._save()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(self._smoke, model) for model in MODELS]
                for result in results:
                    result.result()
            if self.abort.is_set():
                raise RuntimeError('Both model smoke gates must pass before any full generation')
            with self.lock:
                self.state['smoke_gate'].update(status='passed', decided_at=now())
                self._save()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(self._worker, model) for model in MODELS]
                for result in results:
                    result.result()
            if not all(row['status'] == 'completed' for row in self.state['jobs'].values()):
                raise ValueError('The complete new batch was not verified')
            self._check_inputs()
            self.state.update(status='completed', full_suite_verified=True, total_generated_rows=5760)
            return 0
        except BaseException as error:
            self.abort.set()
            self.state.update(status='cancelled' if self.cancelled.is_set() else 'failed',
                              error=str(error), traceback=traceback.format_exc())
            return 1
        finally:
            with self.lock:
                owned = list(self.children.values())
            for process in owned:
                self._terminate_owned(process)
            with self.lock:
                for job in self.state['jobs'].values():
                    if job['status'] == 'pending':
                        job.update(status='blocked_by_failed_batch', reason='No automatic retries or resume')
                if self.state['smoke_gate']['status'] == 'waiting':
                    self.state['smoke_gate'].update(status='failed', decided_at=now())
                for worker in self.state['workers'].values():
                    if worker.get('status') != 'completed':
                        worker['status'] = 'cancelled' if self.cancelled.is_set() else 'failed'
                self.state['finished_at'] = now()
                self._save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--models-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    queue = AIMEQueue(args)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, queue.cancel)
    result = queue.run()
    print(json.dumps({'status': queue.state['status'], 'matrix': str(queue.output/'matrix.json')}), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
