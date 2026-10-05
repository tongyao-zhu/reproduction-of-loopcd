import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import launch_parcae370_gpu_gate as gate
from test_parcae370_guidance_preflight import records

class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.out=self.root/'results/runs/20261004-parcae370-cpu-checks'
        for name in ('guidance','arc_prompts'):(self.out/name).mkdir(parents=True)
        model=self.root/'models/Parcae-370M';model.mkdir(parents=True)
        (model/'model_provenance.json').write_text('{}')
        cpu=self.root/'releases/parcae370-cpu-f1480d8/scripts';cpu.mkdir(parents=True)
        (cpu/'gpu_smoke_parcae.py').write_text((gate.ROOT/'scripts/gpu_smoke_parcae.py').read_text())
        (self.out/'arc_prompts/arc_challenge.jsonl.gz').write_bytes(b'fixed audit')
        self.parent=dict(status='PASS',source_commit=gate.CPU_COMMIT,source_sha256={'one':'a'})
        self.guidance=dict(self.parent,records=records(),model_provenance=dict(sha256=gate.sha(model/'model_provenance.json')))
        self.prompt=dict(status='PASS',profile='parcae370_arc_challenge_v1',shots={'arc_challenge':25},
            total_documents=1172,total_requests=4687,registered_hf_policy_ready=True,cuda_initialized=False,
            model_manifest_sha256=self.guidance['model_provenance']['sha256'],
            tasks=dict(arc_challenge=dict(records=dict(file='arc_challenge.jsonl.gz',sha256=gate.sha(self.out/'arc_prompts/arc_challenge.jsonl.gz')))))
        self.save()
    def save(self):
        for name,value in [('status.json',self.parent),('guidance/result.json',self.guidance),('arc_prompts/result.json',self.prompt)]:
            (self.out/name).write_text(json.dumps(value))
    def validate(self):
        with patch.object(gate,'frozen_tree',return_value={'one':'a'}):
            return gate.validate_cpu(self.root,'new')
    def test_valid_cpu_binding(self):self.assertEqual(len(self.validate()),4)
    def test_incomplete_audit_rejected(self):
        for name in ('status','registered_hf_policy_ready','total_documents','shots'):
            original=self.prompt[name];self.prompt[name]=None;self.save()
            with self.assertRaises(ValueError):self.validate()
            self.prompt[name]=original
    def test_changed_records_rejected(self):
        (self.out/'arc_prompts/arc_challenge.jsonl.gz').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.validate()
    def test_wrong_cpu_source_rejected(self):
        self.guidance['source_commit']='wrong';self.save()
        with self.assertRaises(ValueError):self.validate()
    def test_changed_numerical_oracle_rejected(self):
        p=self.root/'releases/parcae370-cpu-f1480d8/scripts/gpu_smoke_parcae.py'
        p.write_text('def numerical_cases(model,tokens):\n    return []\n')
        with self.assertRaisesRegex(ValueError,'Numerical oracle'):self.validate()

if __name__=='__main__':unittest.main()
