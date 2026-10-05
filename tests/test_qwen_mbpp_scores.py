"""Scoring certificate corruption checks using retained canonical evidence.

Reads JSON only. Does not import or run any canonical or generated solution.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from score_qwen_mbpp import read_subset
from compare_mbpp import load_data,load_canonical,read_scores

class ScoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=load_data(ROOT/'data/mbpp/MbppPlus-v0.2.0.jsonl')
        base=ROOT/'results/mbpp_scorer_v2_final_20261004'
        cls.context=load_canonical(base/'canonical378.json',base/'runtime_manifest.json',data)
        scores=ROOT/'results/runs/20261004-mbpp-full-scores-v2/baseline32.json'
        cls.report=json.loads(scores.read_text())
        cls.generation=dict(samples_sha256=cls.report['samples_sha256'],rows={r['task_id']:{} for r in cls.report['rows']})
        read_scores(scores,cls.generation,cls.context)

    def fixture(self):
        report=copy.deepcopy(self.report);report['rows']=report['rows'][:2];report['expected_rows']=report['completed_rows']=2
        report['sample_validation']=dict(samples=2,expected_full_tasks=378,is_full_task_set=False)
        generation=dict(samples_sha256=report['samples_sha256'],rows={r['task_id']:{} for r in report['rows']})
        return report,generation

    def test_subset_binding(self):
        report,generation=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic_subset.json';p.write_text(json.dumps(report));self.assertEqual(len(read_subset(p,generation,self.context)['rows']),2)
            for mutate in [lambda x:x.update(samples_sha256='0'*64),lambda x:x['canonical_validation'].update(evidence_sha256='0'*64),
                           lambda x:x['safety']['checks'].update(network_blocked=False),lambda x:x['rows'].append(x['rows'][0]),
                           lambda x:x['rows'][0]['base'].update(tests=999),lambda x:x['rows'][0]['base'].update(status='error')]:
                bad=copy.deepcopy(report);mutate(bad);p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):read_subset(p,generation,self.context)

if __name__=='__main__':unittest.main()
