import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_mbpp as c
import score_mbpp_when_ready as q

class Recovery(unittest.TestCase):
    def test_running_and_exited_process_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(q.process_is_active(123,tmp))
            p=Path(tmp)/'123';p.mkdir()
            for state,active in [('Z',False),('X',False),('R',True),('S',True),('D',True)]:
                (p/'stat').write_text('123 (name with ) character) '+state+' 1 2 3')
                self.assertEqual(q.process_is_active(123,tmp),active)
    def test_raw_official_problem_hash_matches_generator(self):
        problem={'task_id':'Mbpp/404','plus_input':[[-float('inf'),float('inf')]]}
        raw='{"plus_input": [[-Infinity, Infinity]], "task_id": "Mbpp/404"}'
        self.assertEqual(c.problem_digest(problem),hashlib.sha256(raw.encode()).hexdigest())
        with self.assertRaises(ValueError):c.digest(problem)
        with self.assertRaises(ValueError):c.digest({'score':float('nan')})
    def test_finite_hashes_unchanged(self):
        value={'task_id':'Mbpp/3','plus_input':[[1,2]]}
        self.assertEqual(c.problem_digest(value),c.digest(value))
    def test_wrong_failure_or_any_started_scoring_rejected(self):
        good=dict(status='failed',failed_phase='validating_generation',error="ValueError('Out of range float values are not JSON compliant')",
                  commands=[],completed_arms=[],active_child_pid=None)
        for key,value in [('status','completed'),('failed_phase','scoring'),('error','other'),('commands',[{}]),('completed_arms',['baseline32']),('active_child_pid',123)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'status.json').write_text(json.dumps(dict(good,**{key:value})))
                with self.assertRaises(RuntimeError):q.acquire_recovery_claim(root,root/'generation',root/'new',root)

if __name__=='__main__':unittest.main()
