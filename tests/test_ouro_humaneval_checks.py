"""Corruption rejection using recorded native CPU rows; no GPU PASS claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ouro_humaneval_checks as c


class Checks(unittest.TestCase):
    def setUp(self):
        self.record=json.loads((ROOT/'results/ouro_he_cpu_summary_20261004.json').read_text())
        self.setting=self.record['generation_config']
        self.patch=patch.object(c,'generation_config',return_value=SimpleNamespace(to_dict=lambda:copy.deepcopy(self.setting)))
        self.patch.start();self.addCleanup(self.patch.stop)
        self.row=next(copy.deepcopy(r) for r in self.record['rows'] if r['arm']=='baseline' and r['stop_reason']=='eos_token')
        self.row['effective_max_new_tokens']=2048
        keys=('task_id','problem_sha256','prompt','prompt_sha256','prompt_token_ids','seed','effective_max_new_tokens')
        self.prompt={k:self.row[k] for k in keys}
        self.prompt_patch=patch.object(c,'build_prompt',return_value=self.prompt)
        self.prompt_patch.start();self.addCleanup(self.prompt_patch.stop)
        self.config=c.protocol_config('baseline',{}, {},2);self.row['config_hash']=c.digest(self.config)
        self.data=[dict(task_id=self.row['task_id'],entry_point='unused')]
        self.tokenizer=SimpleNamespace(decode=lambda ids,skip_special_tokens: '' if skip_special_tokens else self.row['raw_generation'])
        self.sanitize=lambda text,entrypoint:self.row['solution']
    def validate(self,row=None,config=None):
        c.validate_rows([row or self.row],config or self.config,self.data,self.tokenizer,self.sanitize,complete=False)
    def test_recorded_native_eos_row_accepted(self):self.validate()
    def test_generation_configuration_not_outer_arm(self):
        wrong=c.protocol_config('adaptive',{}, {},2)
        with self.assertRaises(ValueError):self.validate(config=wrong)
    def test_row_corruption_rejected(self):
        changes=dict(seed=0,effective_max_new_tokens=8,prompt_sha256='bad',prompt_token_ids=[1],config_hash='bad',
                     arm='adaptive',total_loops=3,generated_tokens=2,raw_generation='changed',stop_reason='token_cap',
                     cap_hit=True,solution='changed',elapsed_seconds=float('nan'))
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(dict(self.row,**{key:value}))
    def test_cached_execution_corruption_rejected(self):
        for key,value in dict(forward_calls=2,head_calls=2,loop_calls=3,cache_slots=48,fresh_cache_initial_length=1,final_cache_length=1).items():
            row=copy.deepcopy(self.row);row['execution_observation'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(row)
    def test_adaptive_dtype_and_reference_rejected(self):
        obs=copy.deepcopy(next(r['adapter_observation'] for r in self.record['rows'] if r['arm']=='adaptive'))
        c.validate_observation(obs,'adaptive')
        for key,value in dict(early_loop=2,extra_lm_head_calls=0,score_dtype='torch.bfloat16',guided_logit_shape=[1,2,49152],already_normalized_identity=False).items():
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_observation(dict(obs,**{key:value}),'adaptive')
    def test_missing_and_duplicate_tasks(self):
        for rows in ([self.row],[self.row,self.row],[]):
            with self.subTest(n=len(rows)),self.assertRaises(ValueError):
                c.validate_rows(rows,self.config,self.data,self.tokenizer,self.sanitize)
    def test_changed_registered_settings(self):
        for key,value in dict(max_new_tokens=8192,total_loops=8,seed=43,eos_token_id=2,do_sample=True,stops=[]).items():
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(config=dict(self.config,**{key:value}))
    def certificate(self):
        # Synthetic integration fixture: combines actual recorded CPU rows and
        # source inventory, never exported as an executed GPU certificate.
        rows={m:next(copy.deepcopy(r) for r in self.record['rows'] if r['arm']==m) for m in c.ARMS}
        controlled={}
        for label,reason in [('eos','eos_token'),('role_delimiter_not_eos','token_cap'),('cap','token_cap'),('stop_string','stop_string')]:
            controlled[label]={}
            for mode in c.ARMS:
                candidates=[r for r in self.record['rows'] if r['arm']==mode and r['stop_reason']==reason]
                if label=='role_delimiter_not_eos':candidates=[r for r in candidates if r['generated_token_ids']==[2]*8]
                if label=='cap':candidates=[r for r in candidates if r['generated_token_ids']==[10]*8]
                controlled[label][mode]=copy.deepcopy(candidates[0])
        provenance=dict(model=dict(repo_id=c.MODEL_ID,revision=c.REVISION,model_code_sha256='code'),
                        loaded_model_code_sha256='code',source_sha256=self.record['source_sha256'],git_commit=c.GATE_COMMIT)
        resource=dict(target=2511,measured_autoregressive_tokens=1,prefill_chunk_tokens=512)
        for key,calls,heads,initial,final in [('prefill_execution',5,1,0,2511),('decode_execution',1,2,2511,2512)]:
            resource[key]=dict(forward_calls=calls,loop_calls=4*calls,head_calls=heads*calls,observed_loop_pattern_valid=True,
                              expected_head_calls_per_forward=heads,cache_type='UniversalTransformerCache',cache_slots=192,
                              fresh_cache_initial_length=initial,final_cache_length=final)
        return dict(status='PASS',source_commit=c.GATE_COMMIT,source_sha256=self.record['source_sha256'],
                    generation_config=self.setting,checks=[dict(name=n,passed=True) for n in sorted(c.check_names())],finished_at='fixture',
                    provenance=provenance,generation_provenance=copy.deepcopy(provenance),
                    prompt_audit=dict(n=164,max_tokens=463,records=[dict(task_id=f'HumanEval/{i}') for i in range(164)]),
                    resource_gate=resource,generation_rows=rows,native_generation_tokens=rows['baseline']['generated_token_ids'],controlled_generation=controlled)
    def test_certificate_coverage_and_sources(self):
        self.assertEqual(len(c.check_names()),96)
        gate=self.certificate();c.validate_gate(gate,self.record['source_sha256'])
        changes=[('status','WAITING'),('source_commit','different'),('checks',gate['checks'][:-1]),
                 ('error','failure'),('finished_at',None),('generation_config',dict(self.setting,eos_token_id=2))]
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_gate(dict(gate,**{key:value}),self.record['source_sha256'])
        with self.assertRaises(ValueError):c.validate_gate(gate,{})
        altered=copy.deepcopy(gate);altered['checks'][0]['passed']=False
        with self.assertRaises(ValueError):c.validate_gate(altered,self.record['source_sha256'])

if __name__=='__main__':unittest.main()
