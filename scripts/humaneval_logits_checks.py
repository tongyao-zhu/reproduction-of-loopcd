"""Fail-closed Figure 1b HumanEval input, certificate and output validation.

No model-generated program is executed here. Sanitize only parses Python AST.
The old generation module supplies unchanged data/prompt/stop/seed constants.
"""
import hashlib
import json
import math
from pathlib import Path
from generate_humaneval import DATA_SHA256, INSTRUCTION, RESPONSE, STOPS, digest, task_seed, trim_stops

GATE_COMMIT = '79563ac171b931639c279e9af6747304df52340c'
CONFIG = dict(mode='adaptive', total_loops=16, reference_loop=1, omega=.2, omega_cap=.25)
ARMS = ('baseline16', 'adaptive16')


def same(actual, expected, label):
    if actual != expected:
        raise ValueError('Changed or incomplete ' + label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_names():
    names = {f'{arm}/{key}' for arm in ('baseline','zero_fixed','zero_adaptive','adaptive')
             for key in ('logits','initial_state','final_recurrent_state','layers','rng','restored')}
    names |= {'adaptive/fp64_oracle', 'adaptive/nonzero_effect'}
    names |= {f'cache/{step}/{key}' for step in range(3) for key in
              ('logits','initial','strong/metadata','strong/all_tensors','reference/metadata','reference/all_tensors','separate')}
    names |= {f'long/{step}/{key}' for step in range(2) for key in ('finite','cache')}
    return names | {'native_generate/two_tokens','native_generate/guided','final/restored'}


def validate_gate(gate, current_hashes, source=None):
    same(gate.get('status'), 'PASS', 'GPU gate status')
    same(gate.get('source_commit'), GATE_COMMIT, 'GPU gate commit')
    same(gate.get('configuration'), CONFIG, 'R16 h1 cap .25 gate configuration')
    checks = gate['checks']
    names = [check['name'] for check in checks]
    same(set(names), check_names(), 'GPU gate coverage')
    same(len(names), len(set(names)), 'GPU gate uniqueness')
    if any(check.get('passed') is not True for check in checks) or gate.get('error') or not gate.get('finished_at'):
        raise ValueError('Incomplete or failed GPU gate')
    for check in checks:
        if 'finite' in check and check['finite'] is not True:
            raise ValueError('Nonfinite GPU gate')
        if 'max_abs_diff' in check and check['max_abs_diff'] != 0.:
            raise ValueError('Exact GPU oracle differs')
        if 'max_abs_error' in check and not (0 <= check['max_abs_error'] < 2e-5):
            raise ValueError('FP64 GPU oracle differs')
    original = gate['source_sha256']
    same(len(original), 57, 'original frozen source inventory')
    same(original, gate['provenance']['source_sha256'], 'gate/provenance source hashes')
    same(gate['provenance']['git_commit'], GATE_COMMIT, 'gate provenance commit')
    for name, value in original.items():
        same(current_hashes.get(name), value, 'unchanged computational source ' + name)
    if source is not None:
        for key in ('python','packages','cuda','gpu','model','arxiv','precision','attention','loaded_model_code_sha256'):
            same(source[key], gate['provenance'][key], 'GPU gate runtime ' + key)


def validate_model_files(path, record):
    path = Path(path)
    same(json.loads((path/'model_provenance.json').read_text()), record, 'model provenance')
    pins = {}
    for name, entry in record['files'].items():
        value = sha(path/name)
        expected = record['model_code_sha256'] if name == 'raven_modeling_minimal.py' else entry.get('weight_blob',entry.get('sha256_original'))
        same(value, expected, 'prepared model bytes ' + name)
        pins[name] = value
    return pins


def prompt_record(problem, tokenizer, builder):
    prompt = builder(problem['prompt'].strip()+'\n', INSTRUCTION, RESPONSE, tokenizer)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or any(type(x) is not int or not 0 <= x < 65536 for x in ids) or ids.count(65504) != 1 or 65509 in ids:
        raise ValueError('Invalid HumanEval unpadded single-BOS prompt')
    cap = min(2048, 4096-len(ids))
    if cap < 1:
        raise ValueError('HumanEval prompt exceeds context')
    return dict(task_id=problem['task_id'], problem_sha256=digest(problem), prompt=prompt,
                prompt_sha256=digest(prompt), prompt_token_ids=ids,
                seed=task_seed(problem['task_id'],42), effective_max_new_tokens=cap)


def audit_prompts(problems, tokenizer, builder):
    same([p['task_id'] for p in problems], [f'HumanEval/{i}' for i in range(164)], 'all164 prompt task IDs')
    rows = [prompt_record(p,tokenizer,builder) for p in problems]
    return {'status':'PASS','n':164,'records':rows,
            'max_prompt_tokens':max(len(r['prompt_token_ids']) for r in rows),
            'min_generation_cap':min(r['effective_max_new_tokens'] for r in rows)}


def validate_rows(rows, config, problems, tokenizer, builder, sanitize, complete=True):
    same(config['benchmark'],'HumanEvalPlus-v0.1.10','benchmark')
    same(config['data_sha256'],DATA_SHA256,'dataset')
    guidance=config['guidance']; mode=guidance['mode']
    if mode not in ('baseline','adaptive'):
        raise ValueError('Expected baseline/adaptive logits')
    same(guidance,dict(CONFIG,mode=mode),'guidance')
    for key,value in dict(seed=42,max_new_tokens=2048,native_context_length=4096,do_sample=False,
                          eos_token_id=[65505,65508],pad_token_id=65509,stops=STOPS,
                          instruction=INSTRUCTION,response_prefix=RESPONSE).items():
        same(config[key],value,'protocol '+key)
    limit=config['limit']
    if limit not in (None,2):raise ValueError('Unknown subset')
    ids=[f'HumanEval/{i}' for i in range(164 if limit is None else 2)]
    same(config['task_ids'],ids,'configured ordered task IDs')
    actual=[r['task_id'] for r in rows]
    if len(set(actual))!=len(actual) or not set(actual).issubset(ids):raise ValueError('Duplicate/unknown task')
    if complete:same(actual,ids,'full ordered sample set')
    tasks={p['task_id']:p for p in problems}
    for row in rows:
        problem=tasks[row['task_id']]
        for key,value in prompt_record(problem,tokenizer,builder).items():same(row[key],value,'sample '+key)
        same(row['config_hash'],digest(config),'sample config hash')
        generated=row['generated_token_ids']
        if not generated or any(type(x)is not int or not 0<=x<65536 for x in generated):raise ValueError('Invalid output tokens')
        same(len(generated),row['generated_tokens'],'generated length')
        if len(generated)>row['effective_max_new_tokens']:raise ValueError('Token cap exceeded')
        same(tokenizer.decode(generated,skip_special_tokens=False),row['raw_generation'],'raw decoded tokens')
        decoded=tokenizer.decode(generated,skip_special_tokens=True)
        completion,stop=trim_stops(decoded)
        same(row['completion'],completion,'decoded trimmed completion')
        same(row['solution'],sanitize(completion,entrypoint=problem['entry_point']),'AST sanitation')
        same(row['stop_string'],stop,'stop string')
        reason='stop_string' if stop else 'eos_token' if generated[-1] in config['eos_token_id'] else 'token_cap'
        same(row['stop_reason'],reason,'stop reason')
        if type(row['cap_hit']) is not bool:raise ValueError('Nonboolean cap claim')
        same(row['cap_hit'],reason=='token_cap','cap flag')
        if reason=='token_cap':same(len(generated),row['effective_max_new_tokens'],'cap termination')
        if not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0:raise ValueError('Invalid time')
        obs=row['adapter_observation']; active=mode=='adaptive'
        for key,value in dict(mode=mode,total_loops=16,guidance_applied=active,
                              extra_coda_passes=int(active),extra_lm_head_calls=int(active)).items():same(obs[key],value,'observed '+key)
        if active:
            for key,value in dict(reference_loop=1,executed_physical_loops=list(range(1,17)),lm_head_calls=2,
                                  coda_layer_calls=4,reference_cache_separate=True,reference_rng_unchanged=True,
                                  reference_cache_length=len(row['prompt_token_ids'])+len(generated)-1).items():same(obs[key],value,'observed '+key)


def validate_pair(directory, problems, tokenizer, builder, sanitize):
    directory=Path(directory); result={}; common=None; previous=None
    for arm in ARMS:
        manifest=json.loads((directory/arm/'manifest.json').read_text())
        same(manifest['status'],'completed','generation completion')
        config=manifest['config'];same(manifest['config_hash'],digest(config),'manifest config hash')
        same(config['guidance']['mode'],'baseline' if arm=='baseline16' else 'adaptive','arm mode')
        blob=(directory/arm/'samples.jsonl').read_bytes()
        if not blob.endswith(b'\n'):raise ValueError('Unterminated generation record')
        rows=[json.loads(line) for line in blob.splitlines()]
        validate_rows(rows,config,problems,tokenizer,builder,sanitize)
        for key in ('completed_samples','expected_samples'):same(manifest[key],len(rows),'manifest '+key)
        same(manifest['is_full_split'],config['limit'] is None,'full claim')
        same(manifest['cap_hits'],sum(r['cap_hit'] for r in rows),'manifest caps')
        shared={k:v for k,v in config.items() if k!='guidance'}
        if common is not None:
            same(shared,common,'paired config/source')
            for a,b in zip(previous,rows):
                for key in ('task_id','problem_sha256','prompt','prompt_sha256','prompt_token_ids','seed','effective_max_new_tokens'):
                    same(a[key],b[key],'paired '+key)
        common=shared;previous=rows;result[arm]={'config':config,'rows':rows}
    return result
