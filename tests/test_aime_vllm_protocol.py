"""Corruption rejection for the new independent backend record format."""
import copy
import unittest
from loopcd_repro.aime_protocol import hash_json, hash_text, sample_seed
from loopcd_repro.aime_vllm_protocol import BACKEND_ID, ENGINE, SAMPLING, validate_batch, scoring_rows


class RecordTests(unittest.TestCase):
    def fixture(self):
        p = dict(task_id='AIME2024-I-01', prompt_token_ids=[1, 12, 13])
        m = dict(backend_id=BACKEND_ID, engine=ENGINE, sampling=SAMPLING, adaptive_cap=1.5)
        rows = [dict(task_id=p['task_id'], sample_id=i, seed=sample_seed(p['task_id'], i),
            generated_token_ids=[8, 2], generated_token_ids_sha256=hash_json([8, 2]),
            raw_generation='answer', raw_generation_sha256=hash_text('answer'),
            completion='answer', completion_sha256=hash_text('answer'),
            finish_reason='stop', stop_reason='eos') for i in range(2)]
        b = dict(manifest_sha256=hash_json(m), arm='adaptive', cap=1.5, prompt=p, seconds=10.,
            configured=[dict(model_class='loopcd_repro.vllm_ouro.LoopCDOuroForCausalLM', total_loops=4,
                guidance=dict(mode='adaptive', omega=1., omega_cap=1.5, early_loop=1))],
            counters=[dict(counts=dict(forwards=2, loops=8, heads=4), max_allocated_bytes=100, max_reserved_bytes=200)], outputs=rows)
        return b, m, p

    def test_good_and_time_accounting(self):
        b, m, p = self.fixture()
        validate_batch(b, m, p, 0, 'adaptive')
        rows = scoring_rows([b])
        self.assertEqual(sum(r['elapsed_seconds'] for r in rows), 10.)
        self.assertEqual([r['cap_hit'] for r in rows], [False, False])

    def test_reject_corruptions(self):
        mutations = [
            lambda b: b.update(manifest_sha256='x'),
            lambda b: b.update(arm='baseline'),
            lambda b: b.update(cap=1.),
            lambda b: b.update(prompt={}),
            lambda b: b.update(seconds=float('nan')),
            lambda b: b['outputs'].pop(),
            lambda b: b['outputs'][1].update(sample_id=0),
            lambda b: b['outputs'][0].update(seed=1),
            lambda b: b['outputs'][0].update(generated_token_ids=[2, 2]),
            lambda b: b['outputs'][0].update(generated_token_ids=[8], generated_token_ids_sha256=hash_json([8])),
            lambda b: b['outputs'][0].update(finish_reason='length'),
            lambda b: b['outputs'][0].update(completion='tampered'),
            lambda b: b['counters'][0]['counts'].update(heads=2),
            lambda b: b['configured'][0]['guidance'].update(early_loop=2),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                b, m, p = self.fixture()
                b = copy.deepcopy(b)
                mutate(b)
                with self.assertRaises(ValueError):
                    validate_batch(b, m, p, 0, 'adaptive')

    def test_decode_validation(self):
        class WrongTokenizer:
            def decode(self, *args, **kwargs): return 'wrong text'
        b, m, p = self.fixture()
        with self.assertRaises(ValueError):
            validate_batch(b, m, p, 0, 'adaptive', WrongTokenizer())

    def test_length_stop(self):
        b, m, p = self.fixture()
        for row in b['outputs']:
            row.update(generated_token_ids=[8]*8192, generated_token_ids_sha256=hash_json([8]*8192),
                       finish_reason='length', stop_reason='max_new_tokens')
        validate_batch(b, m, p, 0, 'adaptive')
        self.assertTrue(all(r['cap_hit'] for r in scoring_rows([b])))


if __name__ == '__main__':
    unittest.main()
