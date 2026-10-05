"""MBPP protocol checks; these never load a model or execute benchmark solutions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_mbpp', ROOT / 'scripts/generate_mbpp.py')
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class MbppGenerationTests(unittest.TestCase):
    def test_prompt_uses_only_public_task_prompt(self):
        problem = {'prompt': '\n"""task; assert f(1)==2"""\n',
                   'canonical_solution': 'SECRET', 'base_input': ['HIDDEN'],
                   'plus_input': ['HIDDEN'], 'assertion': 'OTHER', 'contract': 'OTHER'}
        self.assertEqual(generation.prompt_text(problem), '"""task; assert f(1)==2"""\n')

    def test_one_bos_policy_records_and_checks_official_difference(self):
        class Tokenizer:
            bos_token_id = 65504
            def encode(self, prompt, add_special_tokens=True):
                tokens = [65504, 65506, 42]
                return [65504] + tokens if add_special_tokens else tokens
        self.assertEqual(generation.tokenize_prompt(Tokenizer(), "prompt"), [65504, 65506, 42])
        class ChangedTokenizer(Tokenizer):
            def encode(self, prompt, add_special_tokens=True):
                return [65504, 65506, 42]
        with self.assertRaisesRegex(ValueError, 'double-BOS'):
            generation.tokenize_prompt(ChangedTokenizer(), "prompt")

    def test_pin_rejects_wrong_artifact_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('not json')
            with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                generation.load_problems(path)

    def test_official_artifact_if_available(self):
        path = ROOT / 'data/mbpp/MbppPlus-v0.2.0.jsonl'
        if not path.exists():
            self.skipTest('Official artifact is not distributed in git')
        problems = generation.load_problems(path)
        summary = generation.data_summary(problems)
        self.assertEqual((summary['tasks'], summary['base_tests'], summary['plus_tests']), (378, 1174, 39841))
        self.assertEqual((summary['task_ids'][0], summary['task_ids'][-1]), ('Mbpp/2', 'Mbpp/809'))
        self.assertFalse(summary['code_executed'])

    def test_resume_requires_unique_ordered_prefix(self):
        tasks = {'Mbpp/2': {'prompt': 'first'}, 'Mbpp/7': {'prompt': 'second'}}
        def row(task):
            return {'task_id': task, 'config_hash': 'c', 'problem_sha256': generation.digest(tasks[task])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.jsonl'
            path.write_text(json.dumps(row('Mbpp/7')) + '\n')
            with self.assertRaisesRegex(ValueError, 'ordered task prefix'):
                generation.read_completed(path, 'c', tasks)
            path.write_text(json.dumps(row('Mbpp/2')) + '\n')
            self.assertEqual(list(generation.read_completed(path, 'c', tasks)), ['Mbpp/2'])
            with self.assertRaisesRegex(ValueError, 'changed configuration'):
                generation.read_completed(path, 'changed', tasks)
            path.write_text((json.dumps(row('Mbpp/2')) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                generation.read_completed(path, 'c', tasks)

    def test_hidden_table_a4_parameters_and_task_seed(self):
        self.assertEqual(generation.ARMS['hidden32'], ('hidden', 32, 7))
        self.assertEqual(generation.ARMS['hidden16'], ('hidden', 16, 6))
        self.assertEqual(generation.task_seed('Mbpp/2', 42), generation.task_seed('Mbpp/2', 42))
        self.assertNotEqual(generation.task_seed('Mbpp/2', 42), generation.task_seed('Mbpp/7', 42))

    def test_chat_stops_preserve_multifunction_and_assertions(self):
        text = 'def helper():\n    return 1\n\ndef f():\n    return helper()\nassert f()==1\n```\nignored'
        result, stop = generation.trim_stops(text)
        self.assertIn('def f()', result)
        self.assertIn('assert f()==1', result)
        self.assertEqual(stop, '\n```\n')

    def test_model_pin_rejects_changed_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_provenance.json'
            path.write_text(json.dumps({'repo_id': 'tomg-group-umd/huginn-0125', 'revision': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'pinned Huginn'):
                generation.validate_model_directory(Path(directory))


if __name__ == '__main__':
    unittest.main()
