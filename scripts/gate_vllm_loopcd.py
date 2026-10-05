"""Finite, diagnostic-only LoopCD vLLM integration gate; never AIME scores.

Checks native/zero identity, independent FP64 guidance oracle (absolute 1e-4),
native strong-state preservation at identical inputs, sampler consumption,
four distinct logical cache depths, prefill/decode, and long chunked input.
Does not certify HF/vLLM distribution equivalence or authorize migration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PROBE_SHA = 'bc47d064d8e69c1ec625bd980bccdc94a8e4f5fe3206928363491b8f01b11cdb'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'probe', 'prompt-source', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--long-after', type=Path, help='Completed short checks from v1; run only missing long fixture')
    a = parser.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins = frozen_tree(ROOT, a.source_commit)
    a.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='checking', pid=os.getpid(), source_commit=a.source_commit,
                  source_sha256=pins, purpose=__doc__, checks=[], groups=[])
    def save():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = a.output / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(a.output / 'status.json')
    def check(name, ok, **details):
        status['checks'].append(dict(name=name, passed=bool(ok), **details))
        save()
        if not ok:
            raise AssertionError(name)
    save()
    try:
        status['gpu_before'] = gpu_status(a.gpu)
        if not status['gpu_before']['ready']:
            raise RuntimeError('GPU is not free')
        if sha(a.probe) != PROBE_SHA:
            raise RuntimeError('Probe changed')
        probe = json.loads(a.probe.read_text())
        raw = a.prompt_source.open('rb').readline()
        if hashlib.sha256(raw).hexdigest() != probe['prompt_first_line_sha256']:
            raise RuntimeError('Prompt changed')
        prompt = json.loads(raw)['prompt_token_ids']
        trajectory = probe['batches'][1]['generated_token_ids'][0]
        fixtures = [('short', prompt), ('tie136', prompt + trajectory[:136]),
                    ('tie180', prompt + trajectory[:180]),
                    ('long', (prompt + trajectory * 33)[:8384])]
        if a.long_after:
            previous = json.loads(a.long_after.read_text())
            expected_labels = [f + '_' + m for f in ('short', 'tie136', 'tie180') for m in ('native', 'zero', 'adaptive', 'fixed')]
            if (previous['source_commit'] != '45190d270701bde0a8b0601188800aef0604cdf7' or
                previous['status'] != 'failed' or [g['label'] for g in previous['groups']] != expected_labels or
                len(previous['checks']) != 397 or not all(c['passed'] for c in previous['checks'][:-1]) or
                previous['checks'][-1] != {'name': 'long_native/steps', 'passed': False}):
                raise ValueError('Unexpected predecessor; cannot reuse partial gate')
            for name in ('src/loopcd_repro/vllm_ouro.py', 'src/loopcd_repro/vllm_ouro_plugin.py'):
                if previous['source_sha256'][name] != pins[name]:
                    raise ValueError('Adapter changed since short gate')
            for group in previous['groups']:
                for name, digest in group['files'].items():
                    if sha(a.long_after.parent / name) != digest:
                        raise ValueError('Predecessor trace changed')
            status['reused_short_checks'] = dict(path=str(a.long_after), sha256=sha(a.long_after), count=396)
            fixtures = fixtures[-1:]
        (a.output / 'fixtures.json').write_text(json.dumps(fixtures) + '\n')
        status['fixtures_sha256'] = sha(a.output / 'fixtures.json')
        status['probe_sha256'] = PROBE_SHA
        status['prompt_first_line_sha256'] = hashlib.sha256(raw).hexdigest()
        status['model_provenance'] = json.loads((a.model / 'model_provenance.json').read_text())
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                          TOKENIZERS_PARALLELISM='false', HF_MODULES_CACHE=str(a.output / 'hf_modules'),
                          VLLM_CACHE_ROOT=str(a.output / 'vllm_cache'), TRITON_CACHE_DIR=str(a.output / 'triton'),
                          TORCHINDUCTOR_CACHE_DIR=str(a.output / 'inductor'), VLLM_WORKER_MULTIPROC_METHOD='spawn',
                          LOOPCD_VLLM_SETTINGS=json.dumps(dict(mode='baseline')), LOOPCD_VLLM_GATE=str(a.output))
        plugin = a.output / 'plugins'
        dist = plugin / 'loopcd_ouro-0.0.0.dist-info'
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: loopcd-ouro\nVersion: 0.0.0\n')
        (dist / 'entry_points.txt').write_text('[vllm.general_plugins]\nloopcd_ouro = loopcd_repro.vllm_ouro_plugin:register\n')
        sys.path.insert(0, str(plugin))
        os.environ['PYTHONPATH'] = str(plugin) + os.pathsep + str(ROOT / 'src')
        os.environ['VLLM_PLUGINS'] = 'loopcd_ouro'
        import numpy as np
        from vllm import LLM, SamplingParams
        status['versions'] = {n: importlib.metadata.version(n) for n in ('torch', 'vllm', 'transformers')}
        if status['versions']['vllm'] != '0.13.0':
            raise RuntimeError('Unvalidated vLLM version')
        status['status'] = 'loading'
        save()
        model = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=True, dtype='bfloat16',
                    tensor_parallel_size=1, max_model_len=9216, gpu_memory_utilization=.85, max_num_seqs=2,
                    max_num_batched_tokens=4096, enable_prefix_caching=False, enforce_eager=True, seed=42)
        configs = [('native', dict(mode='baseline')), ('zero', dict(mode='adaptive', omega_cap=0.0)),
                   ('adaptive', dict(mode='adaptive', omega_cap=1.5)), ('fixed', dict(mode='fixed', omega=.5))]
        for fixture, ids in fixtures:
            references = {}
            for method, settings in configs:
                label = fixture + '_' + method
                status['status'] = label
                save()
                (a.output / 'control.json').write_text(json.dumps(dict(label=label, guidance=settings, native=method == 'native')))
                result = model.generate([dict(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, seed=42), use_tqdm=False)
                (a.output / 'control.json').unlink()
                tokens = list(result[0].outputs[0].token_ids)
                traces = sorted(a.output.glob(label + '__*.npz'))
                metadata = {path: json.loads(path.with_suffix('.json').read_text()) for path in traces}
                selected = [path for path in traces if metadata[path]['positions'][-1] >= len(ids) - 1]
                check(label + '/steps', len(selected) == len(tokens) == 8)
                # Intermediate chunk-prefill logits are discarded by vLLM.
                # Validate the entire input/position stream before selecting
                # the eight actual next-token readouts. Keep every raw trace.
                actual_positions = [pos for path in traces for pos in metadata[path]['positions']]
                actual_ids = [token for path in traces for token in metadata[path]['input_ids']]
                check(label + '/complete_input_stream', actual_positions == list(range(len(ids) + 7)) and actual_ids == ids + tokens[:-1])
                first = None
                for path in traces:
                    meta = metadata[path]
                    step = meta['positions'][-1] - (len(ids) - 1)
                    with np.load(path, allow_pickle=False) as arrays:
                        z = arrays['final'].copy()
                        g = arrays['guided'].copy()
                        e = arrays['early'].copy() if 'early' in arrays else None
                    check(label + '/shape/' + str(step), g.shape == (1, 49152) and np.isfinite(g).all())
                    if step >= 0:
                        check(label + '/sampler/' + str(step), int(g.argmax()) == tokens[step])
                    if e is not None:
                        p = np.exp(z.astype('float64') - z.max(axis=-1, keepdims=True))
                        p /= p.sum(axis=-1, keepdims=True)
                        top = np.sort(p, axis=-1)[:, -2:]
                        coefficient = 1.5 * (1.0 - (top[:, 1:] - top[:, :1])) if method == 'adaptive' else .5
                        expected = z.astype('float64') + coefficient * (z.astype('float64') - e.astype('float64'))
                        error = float(np.abs(expected - g).max())
                        check(label + '/formula/' + str(step), error <= 1e-4 and meta['norm_calls'] == 4, max_abs=error)
                    else:
                        check(label + '/identity/' + str(step), np.array_equal(z, g))
                    slots = meta['slots']
                    intervals = sorted((c['ptr'], c['ptr'] + c['numel'] * c['element_size']) for s in slots for c in s['cache'])
                    # Read the actual physical depth from the pinned checkpoint.
                    layers = model.llm_engine.model_config.hf_config.num_hidden_layers
                    check(label + '/cache/' + str(step), len(slots) == layers * 4 and
                          len({s['object'] for s in slots}) == len(slots) and
                          len({s['prefix'] for s in slots}) == len(slots) and len(intervals) == len(slots) and
                          all(lo > 0 and hi > lo for lo, hi in intervals) and
                          all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])))
                    if step == 0:
                        first = z
                    elif step > 0:
                        check(label + '/decode_position/' + str(step), meta['positions'] == [len(ids) + step - 1])
                references[method] = (first, tokens)
                if method != 'native':
                    check(label + '/strong_first', np.array_equal(first, references['native'][0]))
                if method == 'zero':
                    check(label + '/native_zero_tokens', tokens == references['native'][1])
                status['groups'].append(dict(label=label, prompt_length=len(ids), tokens=tokens,
                    internal_prefill_readouts=len(traces)-len(selected),
                    files={str(p.name): sha(p) for path in traces for p in (path, path.with_suffix('.json'))}))
                save()
        if frozen_tree(ROOT, a.source_commit) != pins:
            raise RuntimeError('Frozen source changed')
        status['status'] = 'completed'
        status['migration_accepted'] = False
        save()
    except Exception:
        status['status'] = 'failed'
        status['error'] = traceback.format_exc()
        save()
        raise

if __name__ == '__main__':
    main()
