"""Reject incomplete or numerically invalid real-GPU certificates."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gpu_smoke_parcae import CASES, validate_report


def certificate():
    rows = []
    for length in (3, 2048):
        for depth in (4, 8):
            for mode, omega, cap in CASES:
                enabled = mode != 'baseline' and (cap != 0 if mode == 'adaptive' else omega != 0)
                n = 2 if enabled and mode in {'fixed', 'adaptive'} else 1
                rows.append(dict(length=length, depth=depth, mode=mode, omega=omega, cap=cap,
                                 finite=True, oracle_passed=True, rng_identical=True, initial_state_identical=True,
                                 core_states_identical=True, restored=True, cache_is_none=True, dtype='torch.float32',
                                 max_abs_error=0., maximum_tolerance_ratio=0., max_memory_allocated_bytes=100,
                                 calls=dict(initialization=1, prelude=8, core=8*depth, C=n, coda=8*n, norm=n, head=n)))
    return dict(schema_version=1, status='PASS', device_type='cuda', model_repo='SandyResearch/parcae-1.3b',
                lengths=[3, 2048], depths=[4, 8], loading=[dict(keys=225, missing=[], unexpected=[], strict=True)],
                records=rows, checks=dict(model_prepared_hashes_before_after=True, sources_before_after=True,
                strict_loading=True, actual_gpu=True, native_rng_and_trajectories=True, short_and_context_limit=True,
                all_32_cases=True))


class CertificateTests(unittest.TestCase):
    def test_complete_shape(self):
        self.assertEqual(validate_report(certificate())['cases'], 32)

    def test_missing_cpu_and_duplicates_rejected(self):
        for field, value in [('status', 'running'), ('device_type', 'cpu'), ('lengths', [3]),
                             ('loading', []), ('checks', {})]:
            r = certificate(); r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(r)
        for rows in (certificate()['records'][:-1], certificate()['records'] + certificate()['records'][:1]):
            r = certificate(); r['records'] = rows
            with self.assertRaises(ValueError): validate_report(r)

    def test_corrupt_actual_observations_rejected(self):
        for field, value in [('rng_identical', False), ('initial_state_identical', False), ('restored', False),
                             ('cache_is_none', False), ('calls', {}), ('max_abs_error', float('nan')),
                             ('max_abs_error', 1e-8), ('maximum_tolerance_ratio', 1.1),
                             ('maximum_tolerance_ratio', None), ('max_memory_allocated_bytes', 0)]:
            r = certificate(); r['records'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_report(r)

    def test_adaptive_tolerance_is_pointwise_bounded(self):
        r = certificate(); r['records'][5].update(max_abs_error=1e-6, maximum_tolerance_ratio=.1)
        validate_report(r)
        r['records'][5]['maximum_tolerance_ratio'] = float('inf')
        with self.assertRaises(ValueError): validate_report(r)


if __name__ == '__main__':
    unittest.main()
