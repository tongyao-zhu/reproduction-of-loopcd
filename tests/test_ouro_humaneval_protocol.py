import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ouro_humaneval_protocol import stop_metadata,validate_execution

class Protocol(unittest.TestCase):
    def test_eos_role_and_budget(self):
        self.assertEqual(stop_metadata([0],'',8)['stop_reason'],'eos_token')
        self.assertEqual(stop_metadata([2]*8,'',8)['stop_reason'],'token_cap')
        with self.assertRaises(ValueError):stop_metadata([2],'',8)
        with self.assertRaises(ValueError):stop_metadata([0,10],'x',8)
        with self.assertRaises(ValueError):stop_metadata([10]*9,'x',8)
        with self.assertRaises(ValueError):stop_metadata([],'',8)
        with self.assertRaises(ValueError):stop_metadata([49152],'x',1)
    def test_stop_trim_and_precedence(self):
        r=stop_metadata([10,11],'def f(): pass\nprint(1)',8)
        self.assertEqual(r['completion'],'def f(): pass')
        self.assertEqual(r['stop_reason'],'stop_string')
        self.assertFalse(r['cap_hit'])
        r=stop_metadata([10,0],'x\n```\n',2)
        self.assertEqual(r['stop_reason'],'stop_string')
    def test_no_direct_completion_stops(self):
        text='def f(): pass\ndef g(): pass\nimport math'
        self.assertEqual(stop_metadata([10],text,1)['completion'],text)
    def test_exact_execution_and_wrong_cache(self):
        obs=dict(forward_calls=3,loop_calls=12,head_calls=6,observed_loop_pattern_valid=True,
                 expected_head_calls_per_forward=2,cache_type='UniversalTransformerCache',cache_slots=192,
                 fresh_cache_initial_length=0,final_cache_length=9)
        validate_execution(obs,'adaptive',7,3)
        for key,value in dict(head_calls=3,loop_calls=9,cache_slots=48,fresh_cache_initial_length=1,
                              final_cache_length=10,observed_loop_pattern_valid=False).items():
            with self.subTest(key=key),self.assertRaises(ValueError):validate_execution(dict(obs,**{key:value}),'adaptive',7,3)

if __name__=='__main__':unittest.main()
