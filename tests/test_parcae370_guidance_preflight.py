import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from parcae370_guidance_preflight import validate_records, CASES
from audit_parcae_prompts import audit

def records():
    rows=[]
    for depth in (4,8):
        for mode,omega,cap in CASES:
            enabled=mode!='baseline' and (cap!=0 if mode=='adaptive' else omega!=0)
            n=2 if enabled and mode in ('fixed','adaptive') else 1
            rows.append(dict(length=3,depth=depth,mode=mode,omega=omega,cap=cap,
                finite=True,oracle_passed=True,rng_identical=True,initial_state_identical=True,
                core_states_identical=True,restored=True,cache_is_none=True,dtype='torch.float32',
                max_memory_allocated_bytes=0,max_memory_reserved_bytes=0,max_abs_error=0.,maximum_tolerance_ratio=0.,
                calls=dict(initialization=1,prelude=4,core=4*depth,C=n,coda=4*n,norm=n,head=n)))
    return rows

class GateTests(unittest.TestCase):
    def test_complete(self):
        validate_records(records())
    def test_wrong_model_layer_count_rejected(self):
        rows=records();rows[0]['calls']['prelude']=8
        with self.assertRaises(ValueError):validate_records(rows)
    def test_wrong_trajectory_rejected(self):
        for key in ('rng_identical','initial_state_identical','core_states_identical','restored'):
            rows=records();rows[0][key]=False
            with self.assertRaises(ValueError):validate_records(rows)
    def test_missing_and_repeated_cases_rejected(self):
        for rows in (records()[:-1],records()+records()[:1],list(reversed(records()))):
            with self.assertRaises(ValueError):validate_records(rows)
    def test_nonfinite_and_inexact_errors_rejected(self):
        for key,value in [('max_abs_error',float('nan')),('max_abs_error',float('inf')),
                          ('max_abs_error',.01),('maximum_tolerance_ratio',1.01)]:
            rows=records();rows[0][key]=value
            with self.assertRaises(ValueError):validate_records(rows)
    def test_cpu_cannot_be_gpu_certificate(self):
        from gpu_smoke_parcae import validate_report
        with self.assertRaises(ValueError):validate_report(dict(status='PASS',records=records(),device_type='cpu'))
        rows=records();rows[0]['max_memory_allocated_bytes']=100
        with self.assertRaises(ValueError):validate_records(rows)
    def test_unknown_audit_profile_rejected_before_side_effects(self):
        with self.assertRaisesRegex(ValueError,'Unknown pinned'):
            audit(None,profile='arbitrary')

if __name__=='__main__':unittest.main()
