import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import launch_parcae_figure1b as q

class Predecessor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/'results/runs/20261004-huginn-he-logits-v2'
        (self.base/'queue').mkdir(parents=True);(self.base/'full_scores').mkdir()
    def state(self,status,commit=q.QUEUE):
        (self.base/'queue/status.json').write_text(json.dumps(dict(status=status,source_commit=commit)))
    def test_running_not_resubmitted(self):
        for status in ('waiting_smoke','score_smoke','waiting_gpu','generate_full','score_full'):
            self.state(status);self.assertIsNone(q.validate_predecessor(self.root))
    def test_failed_cancelled_and_wrong_source_rejected(self):
        for status,commit in [('failed',q.QUEUE),('cancelled',q.QUEUE),('completed_generation_unscored',q.QUEUE),('generate_full','wrong')]:
            self.state(status,commit)
            with self.subTest(status=status),self.assertRaises(ValueError):q.validate_predecessor(self.root)
    def test_completed_without_evidence_rejected(self):
        self.state('completed')
        with patch.object(q,'frozen_tree',return_value={}),self.assertRaises(FileNotFoundError):q.validate_predecessor(self.root)
    def test_smoke_and_wrong_scorer_not_full(self):
        self.state('completed')
        good=dict(status='PASS',n=164,full_164=True,source_commit=q.SCORE,generator_commit=q.GEN)
        for key,value in [('status','FAIL'),('n',2),('full_164',False),('source_commit','wrong'),('generator_commit','wrong')]:
            (self.base/'full_scores/comparison.json').write_text(json.dumps(dict(good,**{key:value})))
            with self.subTest(key=key),patch.object(q,'frozen_tree',return_value={}),self.assertRaises(ValueError):q.validate_predecessor(self.root)

if __name__=='__main__':unittest.main()
