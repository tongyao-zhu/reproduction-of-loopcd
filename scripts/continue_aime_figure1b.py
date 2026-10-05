"""Continue only AIME2024 baseline/adaptive with the unchanged frozen engine.

Computation provenance remains the engine's real Git tree. This separately
frozen scheduler is explicitly bound in continuation.json and every manifest.
Imported sample bytes and their original configuration hashes are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE_COMMIT = '773bb67d44312030389b1a59e02587e01ecd8e9b'
ARMS = ('baseline', 'adaptive')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_tree(root, commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if actual != commit:
        raise ValueError('Unexpected frozen Git commit')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=root, text=True).splitlines()
    records = {}
    for name in names:
        if not name.startswith(('scripts/', 'src/', 'configs/')) or not name.endswith(('.py', '.json', '.yaml')):
            continue
        original = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
        path = root / name
        if path.is_symlink() or path.read_bytes() != original:
            raise ValueError('Frozen source changed: ' + name)
        records[name] = sha(path)
    actual_names = {str(p.relative_to(root)) for folder in ('scripts', 'src', 'configs')
                    for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}
    if actual_names != set(records):
        raise ValueError('Untracked or missing engine source')
    return records


def pending(prompts, samples, done):
    for task in prompts:
        for sample in range(samples):
            for arm in ARMS:
                if (task, sample) not in done[arm]:
                    yield arm, task, sample


def validate_predecessor(predecessor, configs, prompts, engine, tokenizer, protocol):
    """Read only; fail closed on malformed tails, duplicates or changed input."""
    proof, done = {}, {}
    for arm in ('baseline', 'fixed', 'adaptive'):
        folder = predecessor / arm
        manifest = json.loads((folder / 'manifest.json').read_text())
        config = manifest['config']
        expected = dict(configs['baseline'])
        expected['guidance'] = engine.guidance_dict(arm, expected['model_identity'], protocol)
        if config != expected or manifest.get('config_hash') != engine.hash_json(config):
            raise ValueError('Predecessor configuration differs: ' + arm)
        if manifest.get('paired_config_hash') != engine.hash_json(engine.paired_config(config)):
            raise ValueError('Predecessor paired configuration differs')
        if manifest.get('kind') != 'aime_generation' or manifest.get('expected_samples') != 480 or manifest.get('is_full_split') is not True:
            raise ValueError('Unexpected predecessor scope')
        rows = engine.read_completed(folder / 'samples.jsonl', config, prompts)
        if manifest.get('completed_samples') != len(rows):
            raise ValueError('Predecessor manifest/count not at a stable checkpoint')
        for row in rows.values():
            for key, skip in (('raw_generation', False), ('completion', True)):
                if tokenizer.decode(row['generated_token_ids'], skip_special_tokens=skip, clean_up_tokenization_spaces=False) != row[key]:
                    raise ValueError('Predecessor token decoding differs')
        proof[arm] = {'n': len(rows), 'manifest_sha256': sha(folder / 'manifest.json'),
                      'samples_sha256': sha(folder / 'samples.jsonl'), 'config_hash': manifest['config_hash']}
        done[arm] = rows
    counts = [len(done[a]) for a in ('baseline', 'fixed', 'adaptive')]
    if not (counts[0] >= counts[1] >= counts[2] and counts[0] - counts[2] <= 1):
        raise ValueError('Not an interleaved three-arm prefix')
    return proof


def copy_inputs(predecessor, output, proof):
    for arm in ARMS:
        (output / arm).mkdir()
        for name, key in (('manifest.json', 'manifest_sha256'), ('samples.jsonl', 'samples_sha256')):
            original = predecessor / arm / name
            if sha(original) != proof[arm][key]:
                raise ValueError('Predecessor changed during migration')
            shutil.copyfile(original, output / arm / name)
            if sha(output / arm / name) != proof[arm][key]:
                raise ValueError('Migration copy is not byte-identical')


def run(args):
    driver_pins = frozen_tree(ROOT, args.driver_commit)
    engine_root = args.engine_root.resolve()
    engine_pins = frozen_tree(engine_root, ENGINE_COMMIT)
    sys.path[:0] = [str(engine_root / 'scripts'), str(engine_root / 'src')]
    engine = importlib.import_module('generate_aime')
    from loopcd_repro.runtime import load_ouro, provenance
    from launch_huginn_r16_suite import gpu_status
    if Path(engine.__file__).resolve() != engine_root / 'scripts/generate_aime.py':
        raise ValueError('Imported the wrong generation engine')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdigit() or not gpu_status(gpu)['ready']:
        raise ValueError('An actually idle physical GPU must be assigned')
    # The predecessor lock is held throughout, preventing accidental old resume.
    with engine.output_lock(args.predecessor):
        _run_locked(args, engine, load_ouro, provenance, engine_pins, driver_pins)


def _run_locked(args, engine, load_ouro, provenance, engine_pins, driver_pins):
    protocol = engine.load_protocol(args.protocol)
    identity = engine.read_model_identity(args.model, protocol)
    questions = engine.load_questions(args.data_root, 2024)
    model, tokenizer = load_ouro(args.model, 'cuda:0')
    source = provenance(args.model, model)
    source['precision'] = 'BF16 model; FP32 guidance before native sampling'
    if source['git_commit'] != ENGINE_COMMIT or source['source_sha256'] != engine_pins:
        raise ValueError('Actual computational source differs from the frozen engine')
    if source['loaded_model_code_sha256'] != identity['model_code_sha256']:
        raise ValueError('Loaded model code differs')
    engine.validate_smoke(args.smoke, engine.smoke_bindings(identity, source))
    prompts = {q['task_id']: engine.build_prompt(tokenizer, q, protocol) for q in questions}
    common = {'protocol_id': engine.PROTOCOL_ID, 'protocol_sha256': engine.PROTOCOL_SHA256,
              'dataset': {'year': 2024, 'questions_sha256': engine.QUESTIONS_SHA256[2024],
                          'manifest_sha256': engine.DATA_MANIFEST_SHA256, 'task_ids': list(prompts)},
              'samples_per_problem': 16, 'generation_config': engine.generation_config(protocol).to_dict(),
              'prompt_policy': protocol['prompt'], 'execution_policy': protocol['execution'],
              'model_identity': identity, 'source': source, 'is_full_split': True, 'debug': None}
    configs = {arm: {**common, 'guidance': engine.guidance_dict(arm, identity, protocol)} for arm in ARMS}
    proof = validate_predecessor(args.predecessor, configs, prompts, engine, tokenizer, protocol)
    binding = {'schema_version': 1, 'kind': 'figure1b_aime2024_continuation',
               'engine_commit': ENGINE_COMMIT, 'engine_source_sha256': engine_pins,
               'driver_commit': args.driver_commit, 'driver_source_sha256': driver_pins,
               'predecessor': str(args.predecessor.resolve()), 'predecessor_files': proof,
               'arms': list(ARMS), 'year': 2024, 'samples_per_arm': 480,
               'imported_sample_bytes_unchanged': True, 'generation_parameters_unchanged': True,
               'computation_provenance': 'config.source identifies immutable generation engine; execution_driver identifies this scheduler'}
    if args.resume:
        previous = json.loads((args.output / 'continuation.json').read_text())
        if previous['binding'] != binding:
            raise ValueError('Resume driver/engine/predecessor differs')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        copy_inputs(args.predecessor, args.output, proof)
        engine.atomic_json(args.output / 'continuation.json', {'binding': binding, 'created_at': engine.stamp()})
    binding_sha = engine.hash_json(binding)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stop.append(signum))
    with engine.output_lock(args.output):
        states = engine.initialize_states(args.output, configs, prompts, True, args.smoke, ENGINE_COMMIT)
        for state in states.values():
            for row in state['done'].values():
                for key, skip in (('raw_generation', False), ('completion', True)):
                    if tokenizer.decode(row['generated_token_ids'], skip_special_tokens=skip, clean_up_tokenization_spaces=False) != row[key]:
                        raise ValueError('Resume decoding mismatch')
            state['manifest']['execution_driver'] = {'commit': args.driver_commit, 'continuation_binding_sha256': binding_sha}
            engine.atomic_json(state['folder'] / 'manifest.json', state['manifest'])
        status = {'status': 'running', 'pid': os.getpid(), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                  'binding_sha256': binding_sha, 'started_at': engine.stamp(), 'completed': {a: len(s['done']) for a, s in states.items()}}
        def save():
            status['updated_at'] = engine.stamp()
            status['completed'] = {a: len(s['done']) for a, s in states.items()}
            engine.atomic_json(args.output / 'status.json', status)
        save()
        print(json.dumps({'migration': 'validated', 'counts': status['completed'], 'binding': binding_sha}), flush=True)
        try:
            for arm, task, sample in pending(prompts, 16, {a: s['done'] for a, s in states.items()}):
                if stop:
                    status.update(status='paused', signal=stop[-1]); save()
                    return
                state = states[arm]
                row = engine.generation_row(model, tokenizer, prompts[task], sample, configs[arm], 'cuda:0')
                with (state['folder'] / 'samples.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
                state['done'][(task, sample)] = row
                state['manifest'].update(completed_samples=len(state['done']), updated_at=engine.stamp())
                engine.atomic_json(state['folder'] / 'manifest.json', state['manifest'])
                save()
                print(json.dumps({'arm': arm, 'task': task, 'sample': sample, 'n': len(state['done']), 'seconds': row['elapsed_seconds']}), flush=True)
            if frozen_tree(args.engine_root, ENGINE_COMMIT) != engine_pins or frozen_tree(ROOT, args.driver_commit) != driver_pins:
                raise ValueError('Frozen source changed during generation')
            validate_predecessor(args.predecessor, configs, prompts, engine, tokenizer, protocol)
            for state in states.values():
                state['manifest'].update(status='completed', completed_samples=480,
                                         samples_sha256=sha(state['folder'] / 'samples.jsonl'),
                                         cap_hits=sum(r['cap_hit'] for r in state['done'].values()), updated_at=engine.stamp())
                engine.atomic_json(state['folder'] / 'manifest.json', state['manifest'])
            status.update(status='completed_generation_unscored'); save()
        except BaseException as error:
            status.update(status='failed', error=repr(error)); save()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine-root', 'model', 'data-root', 'protocol', 'smoke', 'predecessor', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--driver-commit', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.output.is_symlink() or args.predecessor.is_symlink():
        raise ValueError('Symlink output/predecessor forbidden')
    if args.output.exists() != args.resume:
        raise ValueError('Fresh output or explicit existing-output resume required')
    run(args)


if __name__ == '__main__':
    main()
