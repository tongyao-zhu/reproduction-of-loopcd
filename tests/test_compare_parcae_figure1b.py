"""Real evidence-reader integration over synthetic HellaSwag pair fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_parcae_figure1b import compare_pair
from test_compare_parcae import Fixture,write_json

class Pair(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.fixture=Fixture(self.tmp.name,limit=2,task='hellaswag',arms=('baseline8','adaptive8'))
    def compare(self):
        return compare_pair(self.fixture.paths,self.fixture.audit,self.fixture.source_root)
    def test_smoke_pair(self):
        r=self.compare();self.assertEqual(r['n_documents'],2);self.assertFalse(r['is_full_split'])
        self.assertEqual(r['metrics']['acc']['correct_counts'],{'baseline8':1,'adaptive8':0})
        self.assertEqual(r['metrics']['acc']['paired_vs_baseline8']['adaptive8']['delta_percentage_points'],-50)
    def test_duplicate_and_missing_arm(self):
        for dirs in (self.fixture.paths[:1],[self.fixture.paths[0]]*2,self.fixture.paths*2):
            with self.subTest(n=len(dirs)),self.assertRaises(ValueError):compare_pair(dirs,self.fixture.audit,self.fixture.source_root)
    def test_unpaired_source(self):
        p=self.fixture.paths[1]/'manifest.json';x=json.loads(p.read_text());x['provenance']['git_commit']='b'*40;write_json(p,x)
        with self.assertRaises(ValueError):self.compare()
    def test_unpaired_initialization(self):
        p=self.fixture.paths[1]/'request_trace.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        rows[0]['pairing']['initialization']['first_16_values_sha256']='b'*64
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.fixture.rehash_trace(p.parent)
        with self.assertRaises(ValueError):self.compare()
    def test_full_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp,limit=None,task='hellaswag',arms=('baseline8','adaptive8'))
            r=compare_pair(f.paths,f.audit,f.source_root)
            self.assertTrue(r['full_split_count_verified']);self.assertEqual(r['n_documents'],10042)
            self.assertEqual(r['n_requests'],40168)
    def test_other_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp,limit=2,arms=('baseline8','adaptive8'))
            with self.assertRaises(ValueError):compare_pair(f.paths,f.audit,f.source_root)

if __name__=='__main__':unittest.main()
