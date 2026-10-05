"""Finite diagnostic only: independent native h1 and same-token actual KV gate.

No benchmark scores, production sampling, or speed claims. The adapter source
must remain byte-identical to the previously inspected 45190d2 computation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'prompt-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu', required=True)
    a = parser.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins = frozen_tree(ROOT, a.source_commit)
    original = subprocess.check_output(['git', 'show', '45190d2:src/loopcd_repro/vllm_ouro.py'], cwd=ROOT)
    if original != (ROOT / 'src/loopcd_repro/vllm_ouro.py').read_bytes():
        raise ValueError('Adapter computation changed')
    a.output.mkdir(parents=True, exist_ok=False)
    s = dict(status='checking', pid=os.getpid(), source_commit=a.source_commit,
             source_sha256=pins, purpose=__doc__, groups=[], checks=[])
    def save():
        s['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = a.output / 'status.tmp'; tmp.write_text(json.dumps(s, indent=2) + '\n'); tmp.replace(a.output / 'status.json')
    def check(name, ok, **details):
        s['checks'].append(dict(name=name, passed=bool(ok), **details)); save()
        if not ok:
            raise AssertionError(name)
    save()
    try:
        s['gpu_before'] = gpu_status(a.gpu)
        if not s['gpu_before']['ready']:
            raise RuntimeError('GPU not free')
        raw = a.prompt_source.open('rb').readline(); row = json.loads(raw)
        s['input_row_sha256'] = hashlib.sha256(raw).hexdigest()
        provenance = json.loads((a.model / 'model_provenance.json').read_text())
        config = json.loads((a.model / 'config.json').read_text())
        s['model_provenance'] = provenance; s['model_config'] = config
        if sha(a.model / 'config.json') != provenance['files']['config.json']['sha256_original']:
            raise ValueError('Model config differs from pinned source')
        cap = {'ByteDance/Ouro-1.4B-Thinking': 1.0, 'ByteDance/Ouro-2.6B-Thinking': 1.5}[provenance['repo_id']]
        s['adaptive_cap'] = cap
        prompt = row['prompt_token_ids']; forced = row['generated_token_ids'][:8]
        if len(forced) != 8:
            raise ValueError('Need eight existing tokens')
        fixtures = [('short', prompt), ('long', (prompt + row['generated_token_ids'] * 100)[:8384])]
        (a.output / 'fixtures.json').write_text(json.dumps(dict(fixtures=fixtures, forced_tokens=forced)) + '\n')
        s['fixtures_sha256'] = sha(a.output / 'fixtures.json')
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            TOKENIZERS_PARALLELISM='false', HF_MODULES_CACHE=str(a.output/'hf_modules'), VLLM_CACHE_ROOT=str(a.output/'vllm_cache'),
            TRITON_CACHE_DIR=str(a.output/'triton'), TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),
            VLLM_WORKER_MULTIPROC_METHOD='spawn', LOOPCD_VLLM_SETTINGS=json.dumps(dict(mode='baseline')), LOOPCD_VLLM_GATE=str(a.output))
        plugin = a.output / 'plugins'; dist = plugin / 'loopcd_reference-0.0.0.dist-info'; dist.mkdir(parents=True)
        (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: loopcd-reference\nVersion: 0.0.0\n')
        (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nloopcd_reference = loopcd_repro.vllm_reference_probe:register\n')
        sys.path.insert(0, str(plugin)); os.environ['PYTHONPATH'] = str(plugin) + os.pathsep + str(ROOT/'src'); os.environ['VLLM_PLUGINS'] = 'loopcd_reference'
        import numpy as np
        from vllm import LLM, SamplingParams
        s['versions'] = {n: importlib.metadata.version(n) for n in ('torch','transformers','vllm')}
        if s['versions']['vllm'] != '0.13.0':
            raise ValueError('Uninspected vLLM version')
        s['status'] = 'loading'; save()
        model = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=1,
                    max_model_len=9216, gpu_memory_utilization=.85, max_num_seqs=1, max_num_batched_tokens=4096,
                    enable_prefix_caching=False, enforce_eager=True, seed=42)
        for fixture, ids in fixtures:
            references = {}
            for method in ('native', 'zero', 'h1', 'adaptive'):
                label = fixture + '_' + method; s['status'] = label; save()
                settings = dict(mode='adaptive', omega_cap=cap if method=='adaptive' else 0) if method in ('zero','adaptive') else dict(mode='baseline')
                control = dict(label=label, guidance=settings, native=method in ('native','h1'), native_h1=method=='h1', prompt_length=len(ids), forced_tokens=forced)
                (a.output/'control.json').write_text(json.dumps(control))
                result = model.generate([dict(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, seed=42), use_tqdm=False)
                (a.output/'control.json').unlink()
                tokens = list(result[0].outputs[0].token_ids)
                check(label+'/forced_output', tokens == forced)
                paths = sorted(a.output.glob(label+'__*.npz'))
                positions, observed_ids, selected, records = [], [], [], []
                for path in paths:
                    meta = json.loads(path.with_suffix('.json').read_text()); positions.extend(meta['positions']); observed_ids.extend(meta['input_ids'])
                    with np.load(path, allow_pickle=False) as array:
                        z=array['final'].copy();g=array['guided'].copy();e=array['early'].copy() if 'early' in array else None
                    prefix = str(path.with_suffix('')); kv=np.load(prefix+'.kv.npy',allow_pickle=False); km=json.loads(Path(prefix+'.kv.json').read_text())
                    check(label+'/kv_layout/'+str(len(records)), kv.dtype == np.uint16 and kv.shape[0] == config['num_hidden_layers'] * (1 if method=='h1' else 4) and kv.shape[1] == 2 and len(km['names']) == kv.shape[0])
                    check(label+'/finite/'+str(len(records)), np.isfinite(z).all() and np.isfinite(g).all() and z.shape == g.shape == (1,config['vocab_size']))
                    if method=='adaptive':
                        strong=z.astype('float64');weak=e.astype('float64');p=np.exp(strong-strong.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);top=np.sort(p,axis=-1)[:,-2:]
                        oracle=strong+cap*(1-top[:,1:]+top[:,:1])*(strong-weak);err=float(np.abs(oracle-g).max())
                        check(label+'/formula/'+str(len(records)), err<=1e-4 and meta['norm_calls']==4, max_abs=err)
                    else:
                        check(label+'/identity/'+str(len(records)), np.array_equal(z,g))
                    rec=dict(z=z,e=e,kv=kv,names=km['names'],pos=meta['positions']); records.append(rec)
                    if meta['positions'][-1]>=len(ids)-1:
                        selected.append(rec)
                check(label+'/input_stream', positions==list(range(len(ids)+7)) and observed_ids==ids+forced[:-1] and len(selected)==8)
                references[method]=records
                if method in ('zero','adaptive'):
                    check(label+'/same_trace_shapes', [r['pos'] for r in records]==[r['pos'] for r in references['native']])
                    for i,(cur,native) in enumerate(zip(records,references['native'])):
                        check(label+'/strong_logits/'+str(i), np.array_equal(cur['z'],native['z']))
                        check(label+'/native_cache/'+str(i), cur['names']==native['names'] and np.array_equal(cur['kv'],native['kv']))
                if method=='adaptive':
                    check(label+'/h1_trace_shapes', [r['pos'] for r in records]==[r['pos'] for r in references['h1']])
                    for i,(cur,early) in enumerate(zip(records,references['h1'])):
                        check(label+'/h1_logits/'+str(i), np.array_equal(cur['e'],early['z']))
                        indices=[cur['names'].index(name) for name in early['names']]
                        check(label+'/h1_cache/'+str(i), np.array_equal(cur['kv'][indices],early['kv']))
                s['groups'].append(dict(label=label,prompt_length=len(ids),tokens=tokens,forward_calls=len(records),
                    files={p.name:sha(p) for p in a.output.glob(label+'__*') if p.is_file()}));save()
        if frozen_tree(ROOT,a.source_commit)!=pins:
            raise RuntimeError('Source changed')
        s['status']='completed';s['benchmark_migration_accepted']=False;save()
    except Exception:
        s['status']='failed';s['error']=traceback.format_exc();save();raise

if __name__=='__main__':main()
