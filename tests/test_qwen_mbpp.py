"""Actual native CPU generation controls and strict output corruption tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qwen_mbpp_protocol import *

class Tokenizer:
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ''.join('' if skip_special_tokens and x in EOS else {10:'x',11:'\n```\n'}.get(x,'a') for x in ids)

class QwenGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=VOCAB,hidden_size=8,intermediate_size=12,num_hidden_layers=36,
            num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=256,
            attention_dropout=0.,tie_word_embeddings=True,use_sliding_window=False)
        cfg._attn_implementation='sdpa';torch.manual_seed(53)
        cls.model=Qwen3ForCausalLM(cfg).eval();cls.tok=Tokenizer()

    def controlled(self,sequence,arm='fixed',cap=4):
        def processor(tokens,logits):
            v=torch.full_like(logits,float('-inf'));v[sequence[len(tokens)]]=0;return v
        return decode_tokens(self.model,self.tok,[1,2,3],arm,'cpu',cap,processor)

    def test_both_eos(self):
        for arm in ARMS:
            for eos in EOS:
                tokens,obs=self.controlled([10,eos,10],arm)
                self.assertEqual(tokens,[10,eos]);self.assertEqual(obs['forward_calls'],2)
                self.assertEqual(stop_metadata(tokens,self.tok.decode(tokens,True),4)['stop_reason'],'eos_token')

    def test_string_and_cap(self):
        tokens,_=self.controlled([10,11,10]);self.assertEqual(tokens,[10,11])
        self.assertEqual(stop_metadata(tokens,self.tok.decode(tokens,True),4)['completion'],'x')
        tokens,_=self.controlled([10]*4);self.assertEqual(len(tokens),4)
        self.assertTrue(stop_metadata(tokens,self.tok.decode(tokens,True),4)['cap_hit'])

    def test_uncontrolled_greedy_matches_prefix_oracle(self):
        from test_qwen_loop import full_prefix_oracle
        from loopcd_repro.qwen_loop import QwenLoopConfig
        for arm in ARMS:
            actual,_=decode_tokens(self.model,self.tok,[1,2,3],arm,'cpu',3)
            ids=[1,2,3];expected=[]
            for _ in range(3):
                strong,weak=full_prefix_oracle(self.model,torch.tensor([ids]),QwenLoopConfig())
                logits=strong if arm=='baseline' else strong+.3*(strong-weak)
                token=int(logits[0,-1].argmax());expected.append(token);ids.append(token)
                if token in EOS or any(s in self.tok.decode(expected,True) for s in STOPS):break
            self.assertEqual(actual,expected)

    def test_invalid_stopping(self):
        for ids,cap in [([],4),([151645,10],4),([10],4),([-1],1),([10]*5,4),([True],1)]:
            with self.assertRaises(ValueError):stop_metadata(ids,self.tok.decode(ids,True),cap)

    def fixture(self):
        tasks=['Mbpp/2','Mbpp/3'];data=dict(task_ids=tasks,problems={t:dict(task_id=t,entry_point='f') for t in tasks})
        def prompt(p,tok):return dict(task_id=p['task_id'],problem_sha256='a'*64,prompt='abc',prompt_sha256=digest('abc'),prompt_token_ids=[1,2,3],seed=task_seed(p['task_id'],42),effective_max_new_tokens=CAP)
        def sanitize(s,entrypoint):return s
        tmp=tempfile.TemporaryDirectory();folder=Path(tmp.name)
        for arm in ARMS:
            cfg=config(arm,{'fixture':True},tasks,2);rows=[]
            for task in tasks:
                ids=[10,151645];text=self.tok.decode(ids,True)
                rows.append(dict(**prompt(data['problems'][task],self.tok),arm=arm,sample_id=0,generated_token_ids=ids,generated_tokens=2,
                    raw_generation=self.tok.decode(ids),**stop_metadata(ids,text,CAP),solution=text,elapsed_seconds=1.,config_hash=digest(cfg),
                    execution_observation=dict(forward_calls=2,cache_slots=81 if arm=='fixed' else 64,final_cache_length=4,
                        layer_calls_per_forward=dict(prelude=15,core=32,strong_tail=17,reference_tail=17 if arm=='fixed' else 0),fresh_cache_initial_length=0)))
            d=folder/arm;d.mkdir();(d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (d/'manifest.json').write_text(json.dumps(dict(status='completed',config=cfg,config_hash=digest(cfg),expected_samples=2,completed_samples=2,is_full_split=False,cap_hits=0)))
        return tmp,folder,data,prompt,sanitize

    def test_strict_pair_and_corrupt_rows(self):
        tmp,folder,data,prompt,sanitize=self.fixture()
        with tmp,patch('qwen_mbpp_protocol.build_prompt',prompt):
            self.assertEqual(len(validate_pair(folder,data,self.tok,sanitize)['baseline']['rows']),2)
            path=folder/'fixed/samples.jsonl';original=path.read_text();rows=[json.loads(l) for l in original.splitlines()]
            for key,value in [('seed',99),('solution','changed'),('generated_tokens',1),('elapsed_seconds',float('nan')),('config_hash','0'*64),('arm','baseline')]:
                bad=copy.deepcopy(rows);bad[0][key]=value;path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):validate_pair(folder,data,self.tok,sanitize)
            path.write_text(original.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):validate_pair(folder,data,self.tok,sanitize)

    def test_config_identity(self):
        cfg=config('fixed',{},['Mbpp/2','Mbpp/3'],2)
        self.assertEqual(cfg['max_new_tokens'],2048);self.assertEqual(cfg['omega'],.3);self.assertFalse(cfg['do_sample'])
        self.assertEqual(cfg['eos_token_id'],list(EOS));self.assertEqual(cfg['damping'],.125)

if __name__=='__main__':unittest.main()
