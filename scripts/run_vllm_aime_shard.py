"""Disjoint AIME shard with verified immutable vLLM predecessor batches.
Generation and per-batch validation remain unchanged from 78ac637.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from loopcd_repro.aime_protocol import (load_protocol, load_questions, read_model_identity,
    build_prompt, hash_json, hash_text, sha256_file, sample_seed, atomic_json, stamp)
from loopcd_repro.aime_vllm_protocol import (BACKEND_ID, ENGINE, SAMPLING, ARMS,
    require, validate_batch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'model', 'proof', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--task-start', type=int, required=True)
    p.add_argument('--task-end', type=int, required=True)
    p.add_argument('--predecessor', type=Path)
    a = p.parse_args()
    require(0 <= a.task_start < a.task_end <= 30, 'Invalid disjoint task range')
    from sharded_aime_common import load_predecessor
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins = frozen_tree(ROOT, a.source_commit)
    protocol_path = a.project / 'data/aime/protocol-v1.json'
    data_root = a.project / 'data/aime/prepared-v1'
    protocol = load_protocol(protocol_path)
    questions = load_questions(data_root, 2024)
    identity = read_model_identity(a.model, protocol)
    cap = protocol['models'][identity['repo_id']]['adaptive_cap']
    proof = json.loads(a.proof.read_text())
    require(proof['status'] == 'budget_sampling_verified', 'Independent budget proof missing')
    size = '14' if identity['repo_id'] == 'ByteDance/Ouro-1.4B-Thinking' else '26'
    evidence = proof['models'][size]
    probe_path = a.project / evidence['status_path']
    require(sha256_file(probe_path) == evidence['status_sha256'], 'Probe changed')
    probe = json.loads(probe_path.read_text())
    require(probe['status'] == 'completed', 'Probe incomplete')
    require(probe['model_provenance'] == json.loads((a.model/'model_provenance.json').read_text()), 'Probe model differs')
    for name in ('src/loopcd_repro/vllm_ouro.py', 'src/loopcd_repro/guidance.py',
                 'src/loopcd_repro/vllm_ouro_plugin.py', 'src/loopcd_repro/vllm_budget_runtime.py'):
        require(pins[name] == probe['source_sha256'][name], 'Validated engine changed')
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='checking', pid=os.getpid(), source_commit=a.source_commit,
                 source_sha256=pins, counts={arm: 0 for arm in ARMS})
    def save():
        state['updated_at'] = stamp()
        atomic_json(a.output / 'status.json', state)
    save()
    try:
        # Gold is opened by a separate CPU process only, never by this generator.
        subprocess.run([sys.executable, str(ROOT/'scripts/score_aime.py'), '--canonical-only',
            '--protocol', str(protocol_path), '--data-root', str(data_root),
            '--output', str(a.output/'canonical')], check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        state['gpu_before'] = gpu_status(a.gpu)
        require(state['gpu_before']['ready'], 'GPU not free')
        os.environ.pop('LOOPCD_VLLM_GATE', None)
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            TOKENIZERS_PARALLELISM='false', HF_MODULES_CACHE=str(a.output/'hf_modules'),
            VLLM_CACHE_ROOT=str(a.output/'vllm_cache'), TRITON_CACHE_DIR=str(a.output/'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'), VLLM_WORKER_MULTIPROC_METHOD='spawn',
            LOOPCD_VLLM_SETTINGS=json.dumps(dict(mode='baseline')))
        plugin = a.output/'plugins'
        dist = plugin/'loopcd_ouro-0.0.0.dist-info'
        dist.mkdir(parents=True)
        (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: loopcd-ouro\nVersion: 0.0.0\n')
        (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nloopcd_ouro = loopcd_repro.vllm_ouro_plugin:register\n')
        sys.path.insert(0, str(plugin))
        os.environ['PYTHONPATH'] = str(plugin) + os.pathsep + str(ROOT/'src')
        os.environ['VLLM_PLUGINS'] = 'loopcd_ouro'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
        prompts = [build_prompt(tokenizer, q, protocol) for q in questions]
        require(all(len(x['prompt_token_ids']) + 8192 <= 9216 for x in prompts), 'Context too short')
        require(tokenizer.eos_token_id == 2, 'EOS changed')
        from vllm import LLM, SamplingParams
        versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'vllm')}
        require(versions == probe['versions'], 'Runtime versions changed')
        manifest = dict(backend_id=BACKEND_ID, engine=ENGINE, sampling=SAMPLING,
            model_identity=identity, adaptive_cap=cap, versions=versions,
            source_commit=a.source_commit, source_sha256=pins,
            protocol_sha256=sha256_file(protocol_path), questions_sha256=sha256_file(data_root/'aime2024.questions.jsonl'),
            proof_sha256=sha256_file(a.proof), probe_sha256=evidence['status_sha256'],
            canonical_sha256=sha256_file(a.output/'canonical/canonical_gate.json'),
            prompts=prompts, samples_per_problem=16, expected_per_arm=480,
            schedule='question-major, sample pairs (0,1)..(14,15); baseline then adaptive',
            old_samples_imported=0, seconds_policy='batch wall time; score aggregate divides equally across two rows',
            limitation='Independent backend/batch reconstruction, not HF sample identity or author runtime recovery')
        manifest['source_root'] = str(ROOT)
        manifest['shard'] = dict(task_start=a.task_start, task_end=a.task_end, predecessor=None)
        previous, binding = load_predecessor(a.project, a.predecessor, manifest, tokenizer)
        manifest['shard']['predecessor'] = binding
        state['predecessor_samples'] = {arm: 2*sum(key[0] == arm for key in previous) for arm in ARMS}
        atomic_json(a.output/'manifest.json', manifest)
        for arm in ARMS:
            (a.output/arm).mkdir()
        state['status'] = 'loading'
        save()
        state['status'] = 'waiting_gpu_before_load'
        save()
        while not gpu_status(a.gpu)['ready']:
            time.sleep(15)
        state['status'] = 'loading'
        save()
        model = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=True, **ENGINE)
        for prompt in prompts[a.task_start:a.task_end]:
            for sample_start in range(0, 16, 2):
                for arm in ARMS:
                    if (arm, prompt['task_id'], sample_start) in previous:
                        continue
                    state.update(status='generating', task_id=prompt['task_id'], sample_start=sample_start, arm=arm)
                    save()
                    coefficient = cap if arm == 'adaptive' else 0
                    configured = model.collective_rpc('loopcd_budget_configure', args=(arm, coefficient))
                    seeds = [sample_seed(prompt['task_id'], x) for x in (sample_start, sample_start+1)]
                    params = [SamplingParams(**SAMPLING, seed=seed) for seed in seeds]
                    start = time.perf_counter()
                    generated = model.generate([dict(prompt_token_ids=prompt['prompt_token_ids']) for _ in seeds], params, use_tqdm=False)
                    seconds = time.perf_counter() - start
                    counters = model.collective_rpc('loopcd_budget_finish')
                    require(len(generated) == 2, 'Incomplete generation')
                    records = []
                    for i, output in enumerate(generated):
                        require(list(output.prompt_token_ids) == prompt['prompt_token_ids'] and len(output.outputs) == 1, 'Wrong input/output')
                        item = output.outputs[0]
                        tokens = list(item.token_ids)
                        raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        completion = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        records.append(dict(task_id=prompt['task_id'], sample_id=sample_start+i, seed=seeds[i],
                            generated_token_ids=tokens, generated_token_ids_sha256=hash_json(tokens),
                            raw_generation=raw, raw_generation_sha256=hash_text(raw), completion=completion,
                            completion_sha256=hash_text(completion), finish_reason=item.finish_reason,
                            stop_reason='eos' if tokens and tokens[-1] == 2 else 'max_new_tokens'))
                    batch = dict(manifest_sha256=hash_json(manifest), arm=arm, cap=coefficient, prompt=prompt,
                                 seconds=seconds, configured=configured, counters=counters, outputs=records)
                    validate_batch(batch, manifest, prompt, sample_start, arm, tokenizer)
                    path = a.output/arm/f"{prompt['task_id']}-{sample_start:02d}.json"
                    require(not path.exists(), 'Refusing overwrite')
                    atomic_json(path, batch)
                    state['counts'][arm] += 2
                    save()
        require(frozen_tree(ROOT, a.source_commit) == pins, 'Source changed during generation')
        state['status'] = 'completed_generation'
        save()
    except Exception:
        state['status'] = 'failed'
        state['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
