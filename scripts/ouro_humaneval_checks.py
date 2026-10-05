"""Fail-closed validation for the registered Figure1b Ouro HumanEval pair."""
import json
import math
from pathlib import Path
from ouro_humaneval_protocol import (DATA_SHA256,INSTRUCTION,RESPONSE,STOPS,MODEL_ID,REVISION,
    build_prompt,digest,sha,stop_metadata,validate_execution,MAX_NEW_TOKENS,generation_config)

GATE_COMMIT='49d07181a7f8d04e80fa4ec854ac866618394603'
ARMS=('baseline','adaptive')


def same(a,b,label):
    if a!=b:raise ValueError('Changed/incomplete '+label)


def check_names():
    names={'native/full/execution','native/full/normalized_states','native/full_sequence_shape',
           'native/last/execution','native/last/normalized_states','exception/restore'}
    names|={f'{mode}/{suffix}' for mode in ('baseline','fixed_zero','adaptive_zero','fixed','adaptive')
            for suffix in ('full/execution','full/normalized_states','manual_all_positions','restore')}
    names|={f'{mode}/{suffix}' for mode in ('baseline','fixed','adaptive') for suffix in ('last_same_shape','last_restore')}
    names|={f'{mode}/cache{step}/{suffix}' for mode in ('fixed','adaptive') for step in range(3)
            for suffix in ('native/execution','native/normalized_states','guided/execution','guided/normalized_states',
                           'logits','metadata','all_kv_tensors','restore')}
    names|={'code/baseline/execution','code/adaptive/execution','code/native_baseline_tokens',
            'code/fixed_zero/tokens','code/adaptive_zero/tokens','code/long_cache/finite','code/long_cache/all_slots','code/restoration'}
    names|={f'code/controlled/{stop}/{mode}' for stop in ('eos','role_delimiter_not_eos','cap','stop_string') for mode in ARMS}
    return names


def validate_gate(gate,pins,source=None):
    same(gate['generation_config'],generation_config().to_dict(),'GPU generation settings')
    same(gate['status'],'PASS','GPU gate status');same(gate['source_commit'],GATE_COMMIT,'GPU gate commit')
    checks=gate['checks'];names=[c['name'] for c in checks]
    same(set(names),check_names(),'GPU gate coverage');same(len(names),len(set(names)),'unique GPU checks')
    if gate.get('error') or not gate.get('finished_at') or any(c.get('passed') is not True for c in checks):
        raise ValueError('Incomplete/failed GPU certificate')
    for c in checks:
        if 'finite' in c and c['finite'] is not True:raise ValueError('Nonfinite oracle')
        if 'max_abs_diff' in c and c['max_abs_diff']!=0.:raise ValueError('Inexact oracle')
    hashes=gate['source_sha256'];same(len(hashes),64,'original source inventory')
    for name,value in hashes.items():same(pins.get(name),value,'frozen computational source '+name)
    original=gate['provenance'];same(original,gate['generation_provenance'],'two gate model loads')
    same(original['source_sha256'],hashes,'gate runtime source');same(original['git_commit'],GATE_COMMIT,'gate runtime commit')
    same(original['model']['repo_id'],MODEL_ID,'model family');same(original['model']['revision'],REVISION,'model revision')
    same(original['loaded_model_code_sha256'],original['model']['model_code_sha256'],'loaded native code')
    if source is not None:
        for key in ('python','packages','cuda','gpu','model','arxiv','precision','attention','loaded_model_code_sha256'):
            same(source[key],original[key],'gate runtime '+key)
    audit=gate['prompt_audit'];same(audit['n'],164,'gate all164')
    same([r['task_id'] for r in audit['records']],[f'HumanEval/{i}' for i in range(164)],'gate task IDs')
    same(audit['max_tokens'],463,'registered max prompt')
    resource=gate['resource_gate'];same(resource['target'],2511,'long cache target')
    same(resource['measured_autoregressive_tokens'],1,'long decode scope');same(resource['prefill_chunk_tokens'],512,'prefill chunks')
    for key,calls,heads,initial,final in [('prefill_execution',5,1,0,2511),('decode_execution',1,2,2511,2512)]:
        expected=dict(forward_calls=calls,loop_calls=4*calls,head_calls=heads*calls,observed_loop_pattern_valid=True,
                      expected_head_calls_per_forward=heads,cache_type='UniversalTransformerCache',cache_slots=192,
                      fresh_cache_initial_length=initial,final_cache_length=final)
        same(resource[key],expected,'long cache execution '+key)
    for mode,row in gate['generation_rows'].items():
        if mode not in ARMS:raise ValueError('Unknown fixture arm')
        validate_execution(row['execution_observation'],mode,len(row['prompt_token_ids']),row['generated_tokens'])
        validate_observation(row['adapter_observation'],mode)
    same(set(gate['generation_rows']),set(ARMS),'fixture arms')
    same(gate['native_generation_tokens'],gate['generation_rows']['baseline']['generated_token_ids'],'native token parity')
    for stop in ('eos','role_delimiter_not_eos','cap','stop_string'):
        for mode in ARMS:
            row=gate['controlled_generation'][stop][mode]
            expected='eos_token' if stop=='eos' else 'stop_string' if stop=='stop_string' else 'token_cap'
            same(row['stop_reason'],expected,'controlled stop')
            validate_execution(row['execution_observation'],mode,len(row['prompt_token_ids']),row['generated_tokens'])
            validate_observation(row['adapter_observation'],mode)
            if stop=='eos':same(row['generated_token_ids'],[0],'controlled EOS')
            if stop=='role_delimiter_not_eos':same(row['generated_token_ids'],[2]*8,'role2 continuation')
            if stop=='cap':same(row['generated_token_ids'],[10]*8,'controlled cap')


def validate_observation(obs,mode):
    active=mode=='adaptive'
    for key,value in dict(mode=mode,guidance_applied=active,extra_lm_head_calls=int(active),native_exit_at_step=3).items():
        same(obs[key],value,'adapter '+key)
    if active:
        for key,value in dict(executed_source_indices=[0,1,2,3],early_loop=1,already_normalized_identity=True,
                              score_dtype='torch.float32',guided_logit_shape=[1,1,49152]).items():same(obs[key],value,'adapter '+key)


def protocol_config(mode,source,binding,limit):
    if mode not in ARMS or limit not in (None,2):raise ValueError('Unregistered arm/scope')
    return dict(benchmark='HumanEvalPlus-v0.1.10',data_sha256=DATA_SHA256,
                guidance=dict(mode=mode,omega=.5,omega_cap=1.,early_loop=1),total_loops=4,seed=42,
                max_new_tokens=2048,native_context_length=65536,limit=limit,
                task_ids=[f'HumanEval/{i}' for i in range(2 if limit==2 else 164)],do_sample=False,
                stops=STOPS,instruction=INSTRUCTION,response_prefix=RESPONSE,eos_token_id=0,pad_token_id=0,
                generation_config=generation_config().to_dict(),source=source,gate_binding=binding)


def validate_rows(rows,config,data,tokenizer,sanitize,complete=True):
    expected=protocol_config(config['guidance']['mode'],config['source'],config['gate_binding'],config['limit'])
    same(config,expected,'registered full configuration')
    tasks={p['task_id']:p for p in data};seen=[r['task_id'] for r in rows]
    if len(set(seen))!=len(seen) or not set(seen).issubset(config['task_ids']):raise ValueError('Duplicate/unknown row')
    if complete:same(seen,config['task_ids'],'full ordered task set')
    for row in rows:
        problem=tasks[row['task_id']];prompt=build_prompt(problem,tokenizer)
        for key,value in prompt.items():same(row[key],value,'input '+key)
        same(row['config_hash'],digest(config),'row config hash');same(row['guidance'],config['guidance'],'row guidance')
        same(row['arm'],config['guidance']['mode'],'row arm');same(row['total_loops'],4,'row depth')
        ids=row['generated_token_ids'];same(row['generated_tokens'],len(ids),'token count')
        same(row['raw_generation'],tokenizer.decode(ids,skip_special_tokens=False),'raw token decoding')
        expected_stop=stop_metadata(ids,tokenizer.decode(ids,skip_special_tokens=True),MAX_NEW_TOKENS)
        for key,value in expected_stop.items():same(row[key],value,'stop '+key)
        same(row['solution'],sanitize(row['completion'],entrypoint=problem['entry_point']),'AST sanitation')
        validate_execution(row['execution_observation'],row['arm'],len(row['prompt_token_ids']),len(ids))
        validate_observation(row['adapter_observation'],row['arm'])
        if not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0:raise ValueError('Invalid time')


def validate_pair(path,data,tokenizer,sanitize):
    path=Path(path);result={};common=None
    for arm in ARMS:
        m=json.loads((path/arm/'manifest.json').read_text());config=m['config']
        same(m['status'],'completed','manifest status');same(m['config_hash'],digest(config),'manifest config')
        same(config['guidance']['mode'],arm,'folder arm')
        content=(path/arm/'samples.jsonl').read_bytes()
        if not content.endswith(b'\n'):raise ValueError('Unterminated output')
        rows=[json.loads(line) for line in content.splitlines()]
        validate_rows(rows,config,data,tokenizer,sanitize)
        for key in ('completed_samples','expected_samples'):same(m[key],len(rows),'manifest '+key)
        same(m['is_full_split'],config['limit'] is None,'manifest full split')
        same(m['cap_hits'],sum(r['cap_hit'] for r in rows),'manifest cap count')
        shared={k:v for k,v in config.items() if k!='guidance'}
        if common is not None:
            same(shared,common,'paired configuration')
            for a,b in zip(result['baseline']['rows'],rows):
                for key in ('task_id','problem_sha256','prompt','prompt_sha256','prompt_token_ids','seed','effective_max_new_tokens'):
                    same(a[key],b[key],'paired '+key)
        common=shared;result[arm]=dict(config=config,rows=rows)
    return result
