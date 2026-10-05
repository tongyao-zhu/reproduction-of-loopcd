"""Read-only independent NumPy verification of two-stage vLLM gate evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(repo, short, long, config):
    first = json.loads((short / 'status.json').read_text())
    second = json.loads((long / 'status.json').read_text())
    assert first['status'] == 'failed' and first['checks'][-1] == {'name': 'long_native/steps', 'passed': False}
    assert len(first['checks']) == 397 and all(c['passed'] for c in first['checks'][:-1])
    assert second['status'] == 'completed' and all(c['passed'] for c in second['checks'])
    assert second['reused_short_checks']['sha256'] == sha(short / 'status.json')
    assert first['model_provenance'] == second['model_provenance']
    assert sha(config) == first['model_provenance']['files']['config.json']['sha256_original']
    model_config = json.loads(config.read_text())
    assert model_config['total_ut_steps'] == 4
    expected_slots = model_config['num_hidden_layers'] * 4
    records, refs, total_files, total_sources, max_error = [], {}, 0, 0, 0.0
    for root, status in ((short, first), (long, second)):
        fixtures_path = root / 'fixtures.json'
        assert sha(fixtures_path) == status['fixtures_sha256']
        fixtures = dict(json.loads(fixtures_path.read_text()))
        for name, digest in status['source_sha256'].items():
            blob = subprocess.check_output(['git', 'show', status['source_commit'] + ':' + name], cwd=repo)
            assert hashlib.sha256(blob).hexdigest() == digest, name
            total_sources += 1
        for group in status['groups']:
            label = group['label']; fixture, method = label.rsplit('_', 1)
            prompt = fixtures[fixture]; tokens = group['tokens']
            assert len(tokens) == 8 and group['prompt_length'] == len(prompt)
            for name, digest in group['files'].items():
                assert sha(root / name) == digest, name
                total_files += 1
            paths = sorted(root.glob(label + '__*.npz'))
            assert {p.name for p in paths} == {name for name in group['files'] if name.endswith('.npz')}
            positions, ids, selected, final_arrays, early_arrays = [], [], [], [], []
            for path in paths:
                meta = json.loads(path.with_suffix('.json').read_text())
                positions.extend(meta['positions']); ids.extend(meta['input_ids'])
                assert meta['control']['label'] == label
                with np.load(path, allow_pickle=False) as arrays:
                    z, g = arrays['final'].copy(), arrays['guided'].copy()
                    e = arrays['early'].copy() if 'early' in arrays else None
                assert z.shape == g.shape == (1, 49152) and np.isfinite(z).all() and np.isfinite(g).all()
                if method in ('native', 'zero'):
                    assert e is None and np.array_equal(z, g)
                else:
                    assert e is not None and e.shape == z.shape and np.isfinite(e).all()
                    assert meta['norm_calls'] == 4
                    strong = z.astype(np.float64); weak = e.astype(np.float64)
                    if method == 'adaptive':
                        exp = np.exp(strong - strong.max(-1, keepdims=True))
                        prob = exp / exp.sum(-1, keepdims=True)
                        top = np.sort(prob, axis=-1)[..., -2:]
                        omega = 1.5 * (1 - top[..., -1:] + top[..., :1])
                    else:
                        omega = .5
                    err = float(np.abs(g - (strong + omega * (strong - weak))).max())
                    assert err <= 1e-4
                    max_error = max(max_error, err)
                slots = meta['slots']
                assert len(slots) == expected_slots and len({s['object'] for s in slots}) == expected_slots and len({s['prefix'] for s in slots}) == expected_slots
                assert {s['prefix'] for s in slots} == {'model.layers.' + str(i) + '.self_attn.attn' for i in range(expected_slots)}
                spans = sorted((c['ptr'], c['ptr'] + c['numel'] * c['element_size']) for s in slots for c in s['cache'])
                assert len(spans) == expected_slots and all(lo > 0 and hi > lo for lo, hi in spans)
                assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
                if meta['positions'][-1] >= len(prompt) - 1:
                    step = len(selected)
                    assert meta['positions'][-1] == len(prompt) + step - 1
                    assert int(g.argmax()) == tokens[step]
                    selected.append(path.name); final_arrays.append(z); early_arrays.append(e)
            assert positions == list(range(len(prompt) + 7)) and ids == prompt + tokens[:-1] and len(selected) == 8
            if method == 'native':
                refs[fixture] = (tokens, final_arrays)
            else:
                assert np.array_equal(final_arrays[0], refs[fixture][1][0])
                if method == 'zero':
                    assert tokens == refs[fixture][0]
                    assert all(np.array_equal(a, b) for a, b in zip(final_arrays, refs[fixture][1]))
            records.append(dict(label=label, prompt_length=len(prompt), sampled_steps=len(selected),
                                internal_prefill_readouts=len(paths)-len(selected), native_strong_first_exact=True))
    assert len(records) == 16
    return dict(verified_at=datetime.now(timezone.utc).isoformat(), status='limited_integration_verified',
                benchmark_migration_accepted=False, model='Ouro-2.6B-Thinking', groups=records,
                raw_trace_files_verified=total_files, frozen_source_checks=total_sources,
                independent_formula_max_abs=max_error, formula_atol=1e-4,
                native_zero_all_decode_logits_exact=True,
                model_config_sha256=sha(config), physical_layers=model_config['num_hidden_layers'], logical_cache_slots=expected_slots,
                raw_status_sha256={str(root.relative_to(repo) / 'status.json'): sha(root / 'status.json') for root in (short, long)},
                failed_v1_preserved=True, reused_short_checks=396, new_long_checks=len(second['checks']),
                limitations=['One model; synthetic long prefill followed by 8 decode steps, not full 8192-token generation',
                    'Cache object/allocation uniqueness and observed native paths, not forced-token KV-content equality',
                    'No independent h1 reference or HF/vLLM acceptance, batching, graphs, or production sampling gate',
                    'No AIME benchmark outputs; original HF queues remain active'])

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--short', type=Path, required=True)
    p.add_argument('--long', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.repo, a.short, a.long, a.config)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'raw_trace_files_verified', 'independent_formula_max_abs', 'new_long_checks')}))
