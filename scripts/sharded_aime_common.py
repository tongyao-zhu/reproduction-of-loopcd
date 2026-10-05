"""Validate immutable vLLM batch lineage and disjoint shard coverage."""
import json
from pathlib import Path
from loopcd_repro.aime_protocol import sha256_file
from loopcd_repro.aime_vllm_protocol import require, validate_batch
from continue_aime_figure1b import frozen_tree

OLD_COMMIT = '78ac637a58b5b0fb7c4dc77ac53a6c5fdefcaa2b'
COMPUTATION = ('src/loopcd_repro/vllm_ouro.py', 'src/loopcd_repro/guidance.py',
    'src/loopcd_repro/vllm_ouro_plugin.py', 'src/loopcd_repro/vllm_budget_runtime.py',
    'src/loopcd_repro/aime_protocol.py', 'src/loopcd_repro/aime_vllm_protocol.py',
    'scripts/score_aime.py')


def common(manifest):
    return {k:v for k,v in manifest.items() if k not in ('source_root','source_commit','source_sha256','shard')}


def verify_source(project, manifest):
    root = Path(manifest.get('source_root', project/'releases/aime-vllm-78ac637')).resolve()
    require(root.is_relative_to((project/'releases').resolve()), 'Source outside releases')
    require(frozen_tree(root, manifest['source_commit']) == manifest['source_sha256'], 'Source changed')
    original = frozen_tree(project/'releases/aime-vllm-78ac637', OLD_COMMIT)
    for name in COMPUTATION:
        require(original[name] == manifest['source_sha256'][name], 'Numerical/validation source changed')


def read_batches(folder, manifest, tokenizer):
    prompts = {p['task_id']:p for p in manifest['prompts']}
    result, hashes = {}, {}
    for arm in ('baseline','adaptive'):
        for path in sorted((folder/arm).iterdir()):
            require(path.suffix == '.json' and path.is_file(), 'Unexpected batch file')
            batch = json.loads(path.read_text())
            row = batch['outputs'][0]
            key = (arm,row['task_id'],row['sample_id'])
            require(path.name == f'{key[1]}-{key[2]:02d}.json', 'Batch filename/key mismatch')
            validate_batch(batch,manifest,prompts[key[1]],key[2],arm,tokenizer)
            require(key not in result, 'Duplicate batch')
            result[key] = batch
            hashes[str(path.relative_to(folder))] = sha256_file(path)
    return result, hashes


def expected_keys(prompts, start=0, end=30):
    return [(arm,p['task_id'],i) for p in prompts[start:end]
            for i in range(0,16,2) for arm in ('baseline','adaptive')]


def load_predecessor(project, folder, manifest, tokenizer):
    if folder is None:
        return {}, None
    require(manifest['shard']['task_start'] == 0, 'Only first shard may inherit prefix')
    old = json.loads((folder/'manifest.json').read_text())
    retirement = json.loads((folder/'retirement.json').read_text())
    require(retirement['status'] == 'stopped_immutable_snapshot', 'Predecessor not retired')
    for path,digest in retirement['files'].items():
        require(sha256_file(folder/path) == digest, 'Retired evidence changed')
    require(common(manifest) == common(old), 'Predecessor changes experiment configuration')
    verify_source(project,old)
    batches, hashes = read_batches(folder,old,tokenizer)
    require(set(hashes) <= set(retirement['files']), 'Unbound predecessor samples')
    order = expected_keys(old['prompts'])
    require(set(batches) == set(order[:len(batches)]), 'Predecessor must be strict interleaved prefix')
    allowed = set(expected_keys(old['prompts'],0,manifest['shard']['task_end']))
    require(set(batches) <= allowed, 'Predecessor overlaps second shard')
    return batches, dict(path=str(folder),manifest_sha256=sha256_file(folder/'manifest.json'),
                        retirement_sha256=sha256_file(folder/'retirement.json'), batch_sha256=hashes)


def combine_unique(parts, expected):
    merged = {}
    for part in parts:
        require(not (set(merged) & set(part)), 'Duplicate generation across sources')
        merged.update(part)
    require(set(merged) == set(expected), 'Missing or extra generation keys')
    return merged
