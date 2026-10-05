import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import humaneval_logits_checks as c

class Tokens:
    def encode(self,text,add_special_tokens=False):return [65504,12,13]
    def decode(self,ids,skip_special_tokens=False):
        return 'pass\n' if skip_special_tokens else 'pass\n<eos>'


def builder(problem,instruction,response,tokenizer):return instruction+'\n'+problem+response

def sanitize(text,entrypoint):return text.strip()


def fixture(mode='baseline'):
    problems=[dict(task_id=f'HumanEval/{i}',prompt=f'def f{i}():',entry_point=f'f{i}') for i in range(164)]
    config=dict(benchmark='HumanEvalPlus-v0.1.10',data_sha256=c.DATA_SHA256,
                guidance=dict(c.CONFIG,mode=mode),seed=42,max_new_tokens=2048,native_context_length=4096,
                do_sample=False,eos_token_id=[65505,65508],pad_token_id=65509,stops=c.STOPS,
                instruction=c.INSTRUCTION,response_prefix=c.RESPONSE,limit=2,task_ids=['HumanEval/0','HumanEval/1'])
    rows=[]
    for p in problems[:2]:
        row=c.prompt_record(p,Tokens(),builder)
        row.update(config_hash=c.digest(config),generated_token_ids=[123,65505],generated_tokens=2,
                   raw_generation='pass\n<eos>',completion='pass\n',solution='pass',stop_string=None,
                   stop_reason='eos_token',cap_hit=False,elapsed_seconds=1.,
                   adapter_observation=dict(mode=mode,total_loops=16,guidance_applied=mode=='adaptive',
                                            extra_coda_passes=int(mode=='adaptive'),extra_lm_head_calls=int(mode=='adaptive')))
        if mode=='adaptive':row['adapter_observation'].update(reference_loop=1,executed_physical_loops=list(range(1,17)),
                    lm_head_calls=2,coda_layer_calls=4,reference_cache_separate=True,reference_rng_unchanged=True,reference_cache_length=4)
        rows.append(row)
    return config,problems,rows


class Checks(unittest.TestCase):
    def test_actual_gate_passes(self):
        gate=json.loads((ROOT/'results/huginn_logits_gpu_gate_20261004.json').read_text())
        c.validate_gate(gate,gate['source_sha256'],gate['provenance'])
        self.assertEqual(len(c.check_names()),54)

    def test_gate_rejects_corruption(self):
        original=json.loads((ROOT/'results/huginn_logits_gpu_gate_20261004.json').read_text())
        mutations=[lambda g:g['checks'].pop(),lambda g:g['checks'].append(g['checks'][0]),
                   lambda g:g['checks'][0].update(passed=False),lambda g:g['checks'][0].update(finite=False),
                   lambda g:g['checks'][0].update(max_abs_diff=1e-8),lambda g:g.update(status='RUNNING'),
                   lambda g:g['configuration'].update(omega_cap=.5),lambda g:g.update(source_commit='bad')]
        for mutate in mutations:
            g=copy.deepcopy(original);mutate(g)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):c.validate_gate(g,original['source_sha256'])
        hashes=dict(original['source_sha256']);hashes['src/loopcd_repro/huginn_logits.py']='bad'
        with self.assertRaises(ValueError):c.validate_gate(original,hashes)
        source=copy.deepcopy(original['provenance']);source['packages']['torch']='bad'
        with self.assertRaises(ValueError):c.validate_gate(original,original['source_sha256'],source)

    def test_all164_prompts(self):
        _,problems,_=fixture()
        self.assertEqual(c.audit_prompts(problems,Tokens(),builder)['n'],164)
        with self.assertRaises(ValueError):c.audit_prompts(problems[:-1],Tokens(),builder)
        class DoubleBos(Tokens):
            def encode(self,*args,**kwargs):return [65504,65504,5]
        with self.assertRaises(ValueError):c.audit_prompts(problems,DoubleBos(),builder)

    def test_both_row_modes(self):
        for mode in ('baseline','adaptive'):
            config,problems,rows=fixture(mode)
            c.validate_rows(rows,config,problems,Tokens(),builder,sanitize)

    def test_reject_changed_sample_evidence(self):
        config,problems,rows=fixture('adaptive')
        for key,value in dict(seed=7,prompt='other',generated_tokens=9,raw_generation='changed',solution='unsafe',
                              stop_reason='token_cap',effective_max_new_tokens=2,config_hash='bad').items():
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_rows(changed,config,problems,Tokens(),builder,sanitize)
        for key,value in dict(reference_cache_length=5,reference_cache_separate=False,reference_rng_unchanged=False,lm_head_calls=1).items():
            changed=copy.deepcopy(rows);changed[0]['adapter_observation'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_rows(changed,config,problems,Tokens(),builder,sanitize)

    def test_reject_missing_duplicate_and_reordered_rows(self):
        config,problems,rows=fixture()
        for bad in (rows[:1],rows+rows[:1],rows[::-1]):
            with self.assertRaises(ValueError):c.validate_rows(bad,config,problems,Tokens(),builder,sanitize)

    def test_pair_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for arm,mode in zip(c.ARMS,('baseline','adaptive')):
                config,problems,rows=fixture(mode);(root/arm).mkdir()
                manifest=dict(status='completed',config=config,config_hash=c.digest(config),
                              completed_samples=2,expected_samples=2,is_full_split=False,cap_hits=0)
                (root/arm/'manifest.json').write_text(json.dumps(manifest))
                (root/arm/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            c.validate_pair(root,problems,Tokens(),builder,sanitize)
            target=root/c.ARMS[1]/'manifest.json';manifest=json.loads(target.read_text())
            manifest['config']['extra_source']='changed';manifest['config_hash']=c.digest(manifest['config'])
            target.write_text(json.dumps(manifest))
            rows_path=root/c.ARMS[1]/'samples.jsonl'
            rows=[json.loads(r) for r in rows_path.read_text().splitlines()]
            for r in rows:r['config_hash']=manifest['config_hash']
            rows_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):c.validate_pair(root,problems,Tokens(),builder,sanitize)

if __name__=='__main__':unittest.main()
