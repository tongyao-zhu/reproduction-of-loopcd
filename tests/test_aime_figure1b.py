"""Migration retains original row bytes; rejects unpaired/corrupt prefixes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import continue_aime_figure1b as driver
from test_aime_protocol import ap, config_fixture, prompt_fixture, sample_fixture


class Migration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.predecessor = self.root / 'old'
        self.predecessor.mkdir()
        self.prompts = {'AIME2024-I-01': prompt_fixture()}
        self.configs = {a: config_fixture(a) for a in ('baseline', 'fixed', 'adaptive')}
        self.engine = SimpleNamespace(read_completed=ap.read_completed, hash_json=ap.hash_json,
                                      paired_config=ap.paired_config,
                                      guidance_dict=lambda arm, identity, protocol: self.configs[arm]['guidance'])
        self.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens, **kw: 'hello' if skip_special_tokens else 'hello<|im_end|>')
        for arm in self.configs:
            folder = self.predecessor / arm
            folder.mkdir()
            config = self.configs[arm]
            manifest = dict(config=config, config_hash=ap.hash_json(config), paired_config_hash=ap.hash_json(ap.paired_config(config)),
                            kind='aime_generation', expected_samples=480, is_full_split=True, completed_samples=1)
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            (folder / 'samples.jsonl').write_text(json.dumps(sample_fixture(config)) + '\n')

    def validate(self):
        return driver.validate_predecessor(self.predecessor, self.configs, self.prompts, self.engine, self.tokenizer, {})

    def corrupt(self, arm, field, value):
        p = self.predecessor / arm / 'samples.jsonl'
        row = json.loads(p.read_text()); row[field] = value
        p.write_text(json.dumps(row) + '\n')

    def test_byte_identical_copy_and_no_fixed(self):
        proof = self.validate()
        output = self.root / 'new'; output.mkdir()
        before = {str(p): p.read_bytes() for p in self.predecessor.rglob('*') if p.is_file()}
        driver.copy_inputs(self.predecessor, output, proof)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['adaptive', 'baseline'])
        for arm in driver.ARMS:
            for filename in ('manifest.json', 'samples.jsonl'):
                self.assertEqual((output / arm / filename).read_bytes(), (self.predecessor / arm / filename).read_bytes())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.predecessor.rglob('*') if p.is_file()})

    def test_invalid_seed_rejected(self):
        self.corrupt('adaptive', 'seed', 123)
        with self.assertRaisesRegex(ValueError, 'seed'): self.validate()

    def test_invalid_text_rejected(self):
        self.corrupt('baseline', 'completion', 'wrong')
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_rejected(self):
        p = self.predecessor / 'baseline/samples.jsonl'
        p.write_text(p.read_text() * 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.validate()

    def test_config_mutation_rejected(self):
        p = self.predecessor / 'adaptive/manifest.json'
        m = json.loads(p.read_text()); m['config']['generation_config']['max_new_tokens'] = 5
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'configuration'): self.validate()

    def test_copy_race_rejected(self):
        proof = self.validate()
        output = self.root / 'new'; output.mkdir()
        self.corrupt('baseline', 'seed', 123)
        with self.assertRaisesRegex(ValueError, 'changed during'): driver.copy_inputs(self.predecessor, output, proof)

    def test_partial_tail_rejected_without_repair(self):
        p = self.predecessor / 'baseline/samples.jsonl'
        p.write_text(p.read_text() + '{"partial":')
        before = p.read_bytes()
        with self.assertRaises(ValueError): self.validate()
        self.assertEqual(p.read_bytes(), before)

    def test_counts_checkpoint_rejected(self):
        p = self.predecessor / 'baseline/manifest.json'
        m = json.loads(p.read_text()); m['completed_samples'] = 9; p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'stable checkpoint'): self.validate()

    def test_pending_preserves_seed_key_order_skips_imports(self):
        done = {'baseline': {('a', 0): {}}, 'adaptive': {}}
        self.assertEqual(list(driver.pending(['a', 'b'], 2, done)),
                         [('adaptive', 'a', 0), ('baseline', 'a', 1), ('adaptive', 'a', 1),
                          ('baseline', 'b', 0), ('adaptive', 'b', 0), ('baseline', 'b', 1), ('adaptive', 'b', 1)])


if __name__ == '__main__': unittest.main()
