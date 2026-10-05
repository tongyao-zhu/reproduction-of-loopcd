"""Separate AIME backend protocol; never accepts or imports HF samples."""
import math
from .aime_protocol import hash_json, hash_text, sample_seed

BACKEND_ID = 'aime2024-vllm013-eager-batch2-v1'
ENGINE = dict(dtype='bfloat16', tensor_parallel_size=1, max_model_len=9216,
              max_num_seqs=2, max_num_batched_tokens=4096,
              gpu_memory_utilization=.85, enable_prefix_caching=False,
              enforce_eager=True, seed=42, generation_config='vllm',
              worker_extension_cls='loopcd_repro.vllm_budget_runtime.BudgetWorkerExtension')
SAMPLING = dict(n=1, temperature=1.0, top_p=.7, top_k=-1, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                max_tokens=8192, min_tokens=0, stop=None, stop_token_ids=[2],
                ignore_eos=False, skip_special_tokens=False)
ARMS = ('baseline', 'adaptive')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_batch(batch, manifest, prompt, sample_start, arm, tokenizer=None):
    """Reject altered identity, partial batches, stop rules and execution evidence."""
    require(manifest['backend_id'] == BACKEND_ID, 'Wrong backend')
    require(manifest['engine'] == ENGINE and manifest['sampling'] == SAMPLING, 'Settings changed')
    require(batch['manifest_sha256'] == hash_json(manifest), 'Manifest binding')
    require(batch['arm'] == arm and arm in ARMS, 'Arm mismatch')
    cap = manifest['adaptive_cap'] if arm == 'adaptive' else 0
    require(batch['cap'] == cap, 'Guidance mismatch')
    require(batch['prompt'] == prompt and len(prompt['prompt_token_ids']) + 8192 <= 9216, 'Prompt mismatch')
    require(len(batch['outputs']) == 2 and sample_start in range(0, 16, 2), 'Batch incomplete')
    require(math.isfinite(batch['seconds']) and batch['seconds'] > 0, 'Invalid batch duration')
    require(len(batch['configured']) == len(batch['counters']) == 1, 'Unexpected worker count')
    config = batch['configured'][0]
    require(config == dict(model_class='loopcd_repro.vllm_ouro.LoopCDOuroForCausalLM', total_loops=4,
                          guidance=dict(mode=arm, omega=1.0, omega_cap=cap, early_loop=1)), 'Actual model/guidance')
    counter = batch['counters'][0]
    counts = counter['counts']
    require(type(counts['forwards']) is int and counts['forwards'] > 0, 'Missing forwards')
    require(counts['loops'] == 4 * counts['forwards'] and
            counts['heads'] == (2 if cap else 1) * counts['forwards'], 'Loop/head count')
    require(counter['max_allocated_bytes'] > 0 and counter['max_reserved_bytes'] > 0, 'Missing memory evidence')
    for i, row in enumerate(batch['outputs']):
        sample = sample_start + i
        require((row['task_id'], row['sample_id'], row['seed']) ==
                (prompt['task_id'], sample, sample_seed(prompt['task_id'], sample)), 'Sample/seed pairing')
        tokens = row['generated_token_ids']
        require(tokens and len(tokens) <= 8192 and all(type(t) is int and t >= 0 for t in tokens), 'Invalid tokens')
        require(row['generated_token_ids_sha256'] == hash_json(tokens), 'Token hash')
        eos = tokens[-1] == 2
        require(2 not in tokens[:-1] and (eos or len(tokens) == 8192), 'Unexpected termination')
        require(row['finish_reason'] == ('stop' if eos else 'length'), 'Finish reason')
        require(row['stop_reason'] == ('eos' if eos else 'max_new_tokens'), 'Stop reason')
        require(row['raw_generation_sha256'] == hash_text(row['raw_generation']) and
                row['completion_sha256'] == hash_text(row['completion']), 'Text hash')
        if tokenizer is not None:
            for key, skip in [('raw_generation', False), ('completion', True)]:
                require(row[key] == tokenizer.decode(tokens, skip_special_tokens=skip,
                        clean_up_tokenization_spaces=False), 'Decoded text mismatch')


def scoring_rows(batches):
    """Elapsed time is allocated equally, not claimed as per-request latency."""
    rows = []
    for batch in batches:
        memory = batch['counters'][0]
        for row in batch['outputs']:
            rows.append(dict(row, generated_tokens=len(row['generated_token_ids']),
                cap_hit=len(row['generated_token_ids']) == 8192,
                elapsed_seconds=batch['seconds'] / 2,
                peak_allocated_bytes=memory['max_allocated_bytes'],
                peak_reserved_bytes=memory['max_reserved_bytes']))
    return rows
