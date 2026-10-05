"""CPU queue tests: never load a model, score a solution, or modify a sandbox."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('score_mbpp_ready', ROOT / 'scripts/score_mbpp_when_ready.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def write_run(root, arm, status='completed', full=True, count=378, rows=None):
    folder = root / arm
    folder.mkdir(parents=True)
    ids = [f'Mbpp/{i+2}' for i in range(378)]
    manifest = {'status': status, 'is_full_split': full, 'expected_samples': 378,
                'completed_samples': count, 'config': {'task_ids': ids, 'limit': None if full else count}}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    if rows is None:
        rows = ids
    (folder / 'samples.jsonl').write_text(''.join(json.dumps({'task_id': task})+'\n' for task in rows))
    return folder


class MbppScoringQueueTests(unittest.TestCase):
    def test_missing_manifest_and_running_partial_samples_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, snapshot, pins = queue.generation_snapshot(root)
            self.assertFalse(ready)
            self.assertEqual(len(snapshot), 4)
            self.assertFalse(pins)
            folder = write_run(root, 'baseline32', status='running', count=1)
            (folder / 'samples.jsonl').write_text('{unfinished')
            self.assertFalse(queue.generation_snapshot(root)[0])

    def test_failed_generation_stops_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, 'hidden16', status='failed', count=123)
            with self.assertRaises(queue.GenerationFailed):
                queue.generation_snapshot(root)

    def test_debug_and_false_completed_counts_reject(self):
        for full, count in ((False, 2), (True, 377)):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_run(root, 'baseline32', full=full, count=count)
                with self.assertRaises(ValueError):
                    queue.generation_snapshot(root)

    def test_completed_samples_require_order_uniqueness_and_newline(self):
        for corruption in ('duplicate', 'reverse', 'partial_tail'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for arm in queue.ARMS:
                    write_run(root, arm)
                path = root / 'hidden16/samples.jsonl'
                lines = path.read_text().splitlines()
                if corruption == 'duplicate':
                    lines[-1] = lines[0]
                elif corruption == 'reverse':
                    lines.reverse()
                path.write_text('\n'.join(lines)+('' if corruption == 'partial_tail' else '\n'))
                with self.assertRaises(ValueError):
                    queue.generation_snapshot(root)

    def test_all_four_complete_inputs_are_pinned_and_mutation_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in queue.ARMS:
                write_run(root, arm)
            ready, snapshot, pins = queue.generation_snapshot(root)
            self.assertTrue(ready)
            self.assertEqual(len(pins), 8)
            self.assertTrue(all(x['verified_unique_samples'] == 378 for x in snapshot.values()))
            (root / 'baseline16/samples.jsonl').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                queue.verify_hashes(pins)

    def test_provenance_rejects_source_addition_or_wrong_commit(self):
        hashes = {'scripts/generate_mbpp.py': 'a'*64}
        good = {'config': {'source': {'git_commit': queue.GENERATION_COMMIT, 'source_sha256': hashes}}}
        generations = {arm: good for arm in queue.ARMS}
        queue.verify_generation_provenance(generations, hashes)
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            queue.verify_generation_provenance(generations, {**hashes, 'extra.py': 'b'*64})
        good['config']['source']['git_commit'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            queue.verify_generation_provenance(generations, hashes)

    def test_generation_tree_rejects_changed_bytes_and_untracked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            names = {'scripts/generate_mbpp.py', 'scripts/launch_mbpp_suite.py',
                     'scripts/launch_huginn_r16_suite.py', 'src/loopcd_repro/huginn.py',
                     'src/loopcd_repro/runtime.py'}
            originals = {name: ('frozen ' + name).encode() for name in names}
            for name, value in originals.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            def git(command):
                if 'rev-parse' in command:
                    return queue.GENERATION_COMMIT.encode()
                if 'ls-tree' in command:
                    return ('\n'.join(sorted(names))+'\n').encode()
                return originals[command[-1].split(':', 1)[1]]
            with mock.patch.object(queue.subprocess, 'check_output', side_effect=git):
                self.assertEqual(set(queue.frozen_generation_source(root)), names)
                changed = root/'scripts/generate_mbpp.py'
                changed.write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'differs from Git'):
                    queue.frozen_generation_source(root)
                changed.write_bytes(originals['scripts/generate_mbpp.py'])
                (root/'scripts/untracked.py').write_text('unexpected')
                with self.assertRaisesRegex(RuntimeError, 'file set differs'):
                    queue.frozen_generation_source(root)

    def test_wrong_scorer_hash_is_rejected_before_module_import(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            scorer = project/queue.SCORER_RELEASE/'scripts/run_mbpp_sandbox.py'
            scorer.parent.mkdir(parents=True)
            scorer.write_text('must not be imported')
            with mock.patch.object(queue, 'load_module') as loader:
                with self.assertRaisesRegex(RuntimeError, 'pinned MBPP v2'):
                    queue.sandbox_gate(project, project/'gate_acceptance.json', {})
                loader.assert_not_called()

    def test_permanent_claim_prevents_duplicate_scoring_in_another_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = queue.acquire_claim(root, root/'generation', root/'scores1')
            self.assertEqual(json.loads(path.read_text())['output'], str((root/'scores1').resolve()))
            with self.assertRaisesRegex(RuntimeError, 'already has a scoring reservation'):
                queue.acquire_claim(root, root/'generation', root/'scores2')
            self.assertTrue(path.is_file())

    def test_gate_acceptance_exact_counts_runtime_and_nonexecution(self):
        gate = {'evidence_sha256': 'canonical'}
        manifest = {'identity': {'uid': 60002}}
        data = {'sha256': queue.DATA_SHA256, 'task_ids': [str(i) for i in range(378)],
                'counts': {'one': {'base': 1174, 'plus': 39841}}}
        record = {'status': 'PASS', 'actual_scoring_gate_accepted': True,
                  'manifest_sha256': 'runtime', 'canonical_evidence_sha256': 'canonical',
                  'scorer_source_sha256': queue.SCORER_SHA256, 'uid': 60002,
                  'tasks': 378, 'base_tests': 1174, 'plus_tests': 39841, 'model_code_executed': False}
        queue.verify_gate_acceptance(record, 'runtime', gate, manifest, data)
        for key, value in (('tasks', 377), ('actual_scoring_gate_accepted', False),
                           ('model_code_executed', True), ('scorer_source_sha256', 'wrong'),
                           ('manifest_sha256', 'wrong'), ('canonical_evidence_sha256', 'wrong')):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                queue.verify_gate_acceptance({**record, key: value}, 'runtime', gate, manifest, data)

    def test_score_command_has_only_isolated_fixed_runner_entry(self):
        command = queue.score_command(Path('/release/run_mbpp_sandbox.py'), Path('/project/.sandbox/mbpp-v2'),
                                      Path('/generation/samples.jsonl'), Path('/new/score.json'), 7200)
        self.assertEqual(command[0], '/usr/bin/python3')
        self.assertEqual(command[2:4], ['--sandbox', '/project/.sandbox/mbpp-v2'])
        self.assertNotIn('--inside', command)
        self.assertNotIn('--canonical-all', command)
        self.assertEqual(command[-2:], ['--timeout', '7200'])

    def test_nonzero_child_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            child = mock.Mock(pid=123)
            child.wait.return_value = 7
            child.poll.return_value = 7
            state = {'commands': []}
            with mock.patch.object(queue.subprocess, 'Popen', return_value=child) as launch:
                with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                    queue.execute(['/safe/scorer'], Path(directory), 'score', state, lambda: None, 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(state['commands'][0]['returncode'], 7)
            self.assertIsNone(state['active_child_pid'])

    def test_interrupt_forwards_sigterm_and_records_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            child = mock.Mock(pid=123)
            child.wait.side_effect = [KeyboardInterrupt(), 1]
            child.poll.side_effect = [None, 1]
            state = {'commands': []}
            with mock.patch.object(queue.subprocess, 'Popen', return_value=child):
                with self.assertRaises(KeyboardInterrupt):
                    queue.execute(['/safe/scorer'], Path(directory), 'score', state, lambda: None, 1)
            child.terminate.assert_called_once()
            self.assertEqual(state['commands'][0]['returncode'], 1)
            self.assertIsNone(state['active_child_pid'])

    def test_first_score_failure_prevents_later_arms_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            release = project/'queue-release'
            (release/'scripts').mkdir(parents=True)
            (release/'scripts/compare_mbpp.py').write_text('unused')
            generation, output = project/'generation', project/'new-scores'
            data_path = project/'data.jsonl'
            data_path.write_text('unused')
            source_root = project/queue.GENERATION_RELEASE
            source_root.mkdir(parents=True)
            data = {'sha256': queue.DATA_SHA256, 'task_ids': [], 'counts': {}}
            comparer = SimpleNamespace(load_data=lambda _: data,
                                       read_generation=lambda *a: {'config': {'source': {}}},
                                       validate_generation_pairs=lambda _: None,
                                       load_canonical=lambda *a: {})
            launcher = SimpleNamespace(verify_generation=lambda *a: {'passed': True})
            gates = {'pins': {}, 'manifest_sha256': 'm', 'canonical_sha256': 'c', 'identity': {},
                     'canonical': project/'canonical.json', 'manifest': project/'manifest.json',
                     'scorer': project/'run_mbpp_sandbox.py', 'sandbox': project/'.sandbox/mbpp-v2'}
            argv = ['score_mbpp_when_ready.py', '--project', str(project), '--generation', str(generation),
                    '--output', str(output), '--data', str(data_path)]
            def modules(name, path):
                return launcher if 'generation_gate' in name else comparer
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(queue, 'RELEASE', release), \
                 mock.patch.object(queue, 'load_module', side_effect=modules), \
                 mock.patch.object(queue, 'frozen_generation_source', return_value={}), \
                 mock.patch.object(queue, 'sandbox_gate', return_value=gates), \
                 mock.patch.object(queue, 'generation_snapshot', return_value=(True, {}, {})), \
                 mock.patch.object(queue, 'verify_generation_provenance'), \
                 mock.patch.object(queue, 'execute', side_effect=RuntimeError('scorer failed')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
                    queue.main()
            self.assertEqual(execute.call_count, 1)
            status = json.loads((output/'status.json').read_text())
            self.assertEqual((status['status'], status['failed_phase']), ('failed', 'scoring'))
            self.assertEqual(status['completed_arms'], [])
            self.assertTrue(Path(status['claim']).is_file())


if __name__ == '__main__':
    unittest.main()
