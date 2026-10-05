"""Bounded native-vLLM throughput probe; never a Figure1b score or resume input.

Reads one already generated AIME prompt, tests greedy 256-token decode at
concurrency 1 and 4, without changing any production output or dependency.
LoopCD guidance and cross-backend numerical acceptance are NOT implemented here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompt-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu', default='3')
    a = p.parse_args()
    from continue_aime_figure1b import frozen_tree
    from launch_huginn_r16_suite import gpu_status
    pins = frozen_tree(ROOT, a.source_commit)
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='checking', purpose='native baseline throughput only; not numerical acceptance or benchmark score',
                 source_commit=a.source_commit, source_sha256=pins, pid=os.getpid(),
                 started_at=datetime.now(timezone.utc).isoformat(), batches=[])
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        t = a.output / 'status.tmp'
        t.write_text(json.dumps(state, indent=2)+'\n')
        t.replace(a.output / 'status.json')
    save()
    try:
        device = gpu_status(a.gpu)
        state['gpu_before'] = device
        if not device['ready']:
            raise RuntimeError('GPU is occupied; no probe submitted')
        # Keep all writable caches inside this fresh diagnostic output.
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONDONTWRITEBYTECODE='1',
                          HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                          VLLM_CACHE_ROOT=str(a.output/'vllm_cache'),
                          TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),
                          TRITON_CACHE_DIR=str(a.output/'triton'),
                          HF_MODULES_CACHE=str(a.output/'hf_modules'),
                          VLLM_WORKER_MULTIPROC_METHOD='spawn')
        import torch
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models import ouro
        state['versions'] = {n: importlib.metadata.version(n) for n in ('vllm','torch','transformers')}
        if state['versions']['vllm'] != '0.13.0':
            raise RuntimeError('Only inspected vLLM 0.13.0 is accepted')
        state['ouro_source_sha256'] = hashlib.sha256(Path(ouro.__file__).read_bytes()).hexdigest()
        if state['ouro_source_sha256'] != '93e1c32b50d31e12ac41236a327490b28747635b31322b93fa9f3eea4ba127ab':
            raise RuntimeError('Native Ouro implementation changed')
        raw = a.prompt_source.open('rb').readline()
        row = json.loads(raw)
        prompt = row['prompt_token_ids']
        state['prompt_first_line_sha256'] = hashlib.sha256(raw).hexdigest()
        state['prompt_task_id'] = row['task_id']
        state['prompt_tokens'] = len(prompt)
        config = json.loads((a.model/'config.json').read_text())
        if config.get('total_ut_steps',4) != 4:
            raise RuntimeError('Expected R4 checkpoint')
        state['model_config'] = config
        settings = dict(model=str(a.model), tokenizer=str(a.model), trust_remote_code=True,
                        dtype='bfloat16', tensor_parallel_size=1, max_model_len=9216,
                        gpu_memory_utilization=0.85, max_num_seqs=4,
                        max_num_batched_tokens=1024, enable_prefix_caching=False,
                        enforce_eager=True, seed=42)
        state['engine_settings'] = settings
        state['status']='loading';save()
        started=time.perf_counter();llm=LLM(**settings)
        state['load_seconds']=time.perf_counter()-started
        for n, budget, label in [(1,16,'warmup'),(1,256,'serial'),(4,256,'batch4')]:
            state['status']=label;save()
            params=SamplingParams(temperature=0.0, max_tokens=budget, ignore_eos=True, seed=42)
            torch.cuda.synchronize();start=time.perf_counter()
            outputs=llm.generate([dict(prompt_token_ids=prompt) for _ in range(n)],params,use_tqdm=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            ids=[list(x.outputs[0].token_ids) for x in outputs]
            if len(ids)!=n or any(len(x)!=budget for x in ids):
                raise RuntimeError('Incomplete throughput probe')
            state['batches'].append(dict(label=label,n=n,seconds=elapsed,
                output_tokens=sum(map(len,ids)),output_tokens_per_second=sum(map(len,ids))/elapsed,
                generated_token_ids=ids))
            save()
        serial=state['batches'][1];batch=state['batches'][2]
        state['batch4_over_serial_throughput']=batch['output_tokens_per_second']/serial['output_tokens_per_second']
        state['batch_greedy_matches_serial']=[x==serial['generated_token_ids'][0] for x in batch['generated_token_ids']]
        if frozen_tree(ROOT,a.source_commit)!=pins:raise RuntimeError('Probe source changed')
        state['status']='completed';save()
    except Exception:
        state['status']='failed';state['error']=traceback.format_exc();save();raise


if __name__ == '__main__':
    main()
