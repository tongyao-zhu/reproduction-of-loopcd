"""Read-only resource/input audit, NOT a Looped-Qwen implementation or GPU gate.

Never load model tensors or execute dataset solutions. Official file identities
come from a captured pinned Hub API response, including Git blob hashes for
small files and LFS SHA256 for weights. No shared-cache writes/downloads.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import struct
from datetime import datetime, timezone

REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
DATA_SHA = 'b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run this CPU-only audit with CUDA_VISIBLE_DEVICES empty')
    captured = json.loads(args.public_audit.read_text())
    url = 'https://huggingface.co/api/models/Qwen/Qwen3-4B/revision/' + REVISION + '?blobs=true'
    response, = [r for r in captured['requests'] if r['url'] == url]
    if response['status'] != 200 or response['json']['sha'] != REVISION:
        raise ValueError('Pinned official metadata missing')
    files = {}
    for record in response['json']['siblings']:
        name = record['rfilename']
        if Path(name).name != name:
            raise ValueError('Unexpected nested artifact')
        path = args.snapshot / name
        size = path.stat().st_size
        digest = sha(path)
        if size != record['size']:
            raise ValueError('Size differs: ' + name)
        if 'lfs' in record:
            if digest != record['lfs']['sha256']:
                raise ValueError('LFS SHA differs: ' + name)
        else:
            blob = b'blob ' + str(size).encode() + b'\0' + path.read_bytes()
            if hashlib.sha1(blob).hexdigest() != record['blobId']:
                raise ValueError('Git blob differs: ' + name)
        files[name] = {'bytes': size, 'sha256': digest, 'resolved_path': str(path.resolve())}
    index = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())
    shapes = {}
    total_bytes = 0
    for name in sorted(set(index['weight_map'].values())):
        with (args.snapshot / name).open('rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if not 0 < header_len < 10_000_000:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(f.read(header_len))
        intervals = []
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in shapes or index['weight_map'].get(key) != name or value['dtype'] != 'BF16':
                raise ValueError('Tensor index/dtype differs')
            begin, end = value['data_offsets']
            if end - begin != 2 * math.prod(value['shape']):
                raise ValueError('Tensor payload size differs')
            intervals.append((begin, end))
            shapes[key] = value['shape']
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError('Tensor payload gap or overlap')
            cursor = end
        if cursor + 8 + header_len != files[name]['bytes']:
            raise ValueError('Shard payload size differs')
        total_bytes += cursor
    if set(shapes) != set(index['weight_map']) or total_bytes != index['metadata']['total_size']:
        raise ValueError('Tensor inventory differs')
    config = json.loads((args.snapshot / 'config.json').read_text())
    if (config['model_type'], config['num_hidden_layers'], config['hidden_size']) != ('qwen3', 36, 2560):
        raise ValueError('Unexpected architecture')
    if sha(args.data) != DATA_SHA:
        raise ValueError('MBPP data differs')
    problems = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(problems) != 378 or len({p['task_id'] for p in problems}) != 378:
        raise ValueError('Incomplete/duplicate MBPP')
    from transformers import AutoTokenizer
    from evalplus.provider.utility import make_raw_chat_prompt
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    rows = []
    instruction = 'Please provide a self-contained Python script that solves the following problem in a markdown code block:'
    response_prefix = 'Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:'
    for problem in sorted(problems, key=lambda p: int(p['task_id'].split('/')[-1])):
        prompt = make_raw_chat_prompt(problem['prompt'].strip() + '\n', instruction, response_prefix, tokenizer)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.encode(prompt) != ids or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt:
            raise ValueError('Tokenizer roundtrip/default special-token difference')
        if not prompt.endswith('```python\n') or len(ids) + 2048 > config['max_position_embeddings']:
            raise ValueError('Unexpected assistant prefix or context overflow')
        rows.append({'task_id': problem['task_id'], 'prompt': prompt, 'input_ids': ids,
                     'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = args.output / 'prompts.jsonl'
    prompts.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'status': 'PASS', 'checked_at': datetime.now(timezone.utc).isoformat(),
               'kind': 'resource_and_prompt_audit_only', 'model_loaded': False, 'gpu_used': False,
               'model_id': 'Qwen/Qwen3-4B', 'revision': REVISION, 'files': files,
               'model_config': config, 'generation_config': json.loads((args.snapshot / 'generation_config.json').read_text()),
               'weight_tensors': len(shapes), 'weight_elements': total_bytes // 2,
               'weight_map_shapes': shapes, 'data_sha256': DATA_SHA, 'prompt_count': len(rows),
               'max_prompt_tokens': max(len(row['input_ids']) for row in rows),
               'min_prompt_tokens': min(len(row['input_ids']) for row in rows),
               'prompts_sha256': sha(prompts), 'prompt_policy': 'unmodified EvalPlus make_raw_chat_prompt; no explicit thinking override',
               'contains_think_tags': sum('<think>' in row['prompt'] for row in rows),
               'tokenizer_special_tokens_map': tokenizer.special_tokens_map,
               'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'evalplus')},
               'audit_source_sha256': sha(Path(__file__)), 'evalplus_prompt_source_sha256': sha(Path(inspect.getfile(make_raw_chat_prompt))),
               'public_audit_sha256': sha(args.public_audit),
               'limitation': 'No Apple checkpoint identity, wrapper/cache behavior, model forward, generation or score verified.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'weight_tensors', 'weight_elements', 'prompt_count', 'max_prompt_tokens', 'contains_think_tags')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('snapshot', 'public-audit', 'data', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    audit(parser.parse_args())
