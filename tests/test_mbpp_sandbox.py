"""No solution execution: fail-closed MBPP sandbox record and policy tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module('run_mbpp_sandbox')
prepare = module('prepare_mbpp_sandbox')


class MbppSandboxTests(unittest.TestCase):
    def test_isolated_runtime_and_same_dataset_pin(self):
        self.assertEqual(runner.SANDBOX, ROOT / '.sandbox/mbpp')
        self.assertEqual(prepare.SANDBOX, runner.SANDBOX)
        self.assertEqual(prepare.DATA_SHA256, runner.DATA_SHA256)
        self.assertNotEqual(runner.SANDBOX, ROOT / '.sandbox/evalplus')

    def test_explicit_sandbox_path_cannot_target_humaneval_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / '.sandbox/mbpp'
            for mod in (runner, prepare):
                self.assertEqual(mod.sandbox_location(target), target)
                self.assertEqual(mod.sandbox_location(target.with_name('mbpp-v2')), target.with_name('mbpp-v2'))
                for invalid in ('evalplus', 'mbpp-v0', 'mbpp-v3', 'mbpp-v02', 'mbpp-v2-copy'):
                    with self.assertRaisesRegex(ValueError, 'non-symlink'):
                        mod.sandbox_location(target.with_name(invalid))
            target.parent.mkdir()
            other = target.parent / 'other'
            other.mkdir()
            target.symlink_to(other, target_is_directory=True)
            for mod in (runner, prepare):
                with self.assertRaisesRegex(ValueError, 'non-symlink'):
                    mod.sandbox_location(target)

    def test_worker_forbids_network_fork_exec_and_signals(self):
        common = set(runner.COMMON_DENIED)
        strict = common | set(runner.WORKER_DENIED)
        self.assertTrue({'socket', 'connect', 'execve', 'execveat', 'chroot', 'unshare', 'setns',
                         'ptrace', 'process_vm_readv', 'io_uring_setup', 'setuid', 'capset'} <= common)
        self.assertTrue({'fork', 'vfork', 'clone', 'clone3', 'kill', 'tgkill', 'pidfd_send_signal'} <= strict)

    def test_partial_preparation_or_omitted_files_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'PREPARATION_FAILED').touch()
            with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
                runner.verify_runtime(root)
            (root / 'PREPARATION_FAILED').unlink()
            (root / 'manifest.json').write_text(json.dumps({'immutable_sha256': {}}))
            with self.assertRaisesRegex(RuntimeError, 'omits required'):
                runner.verify_runtime(root)

    def test_sample_parser_rejects_unknown_duplicate_and_non_greedy(self):
        ids = ['Mbpp/2', 'Mbpp/7']
        row = {'task_id': ids[0], 'solution': 'never executed'}
        self.assertFalse(runner.validate_samples([row], ids)['is_full_task_set'])
        for rows in ([row, row], [{'task_id': 'HumanEval/2', 'solution': ''}],
                     [{**row, 'sample_id': 1}], [{**row, 'solution': None}], []):
            with self.assertRaises(ValueError):
                runner.validate_samples(rows, ids)

    def test_host_reader_rejects_sandbox_created_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target = base / 'outside.json'
            target.write_text('{"sensitive": "not read"}')
            link = base / 'result.json'
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, 'direct regular file'):
                runner.read_child_report(link)
            link.unlink()
            link.write_text('{"status": "PASS"}')
            self.assertEqual(runner.read_child_report(link), {'status': 'PASS'})

    def test_suite_requires_every_detail_and_success_status(self):
        self.assertTrue(runner.suite_record('pass', [True, True], 2)['passed'])
        for status, details in [('pass', [True]), ('pass', [True, False]), ('fail', [True, True])]:
            self.assertFalse(runner.suite_record(status, details, 2)['passed'])

    def canonical_fixture(self, directory):
        ids = [f'Mbpp/{i}' for i in range(378)]
        counts = {task: {'base': 2, 'plus': 3} for task in ids}
        rows = [{'task_id': task, 'plus_passed': True,
                 'base': runner.suite_record('pass', [True] * 2, 2),
                 'plus': runner.suite_record('pass', [True] * 3, 3)} for task in ids]
        result = {'status': 'PASS', 'manifest_sha256': 'runtime', 'dataset_sha256': runner.DATA_SHA256,
                  'scorer_source_sha256': runner.sha(Path(runner.__file__)),
                  'evaluation_complete': True, 'expected_rows': 378, 'completed_rows': 378, 'rows': rows,
                  'exit_code': 0, 'evaluator_patch': {'applied': False},
                  'safety': {'passed': True, 'checks': {k: True for k in runner.SAFETY_CHECKS}}}
        path = Path(directory) / 'canonical.json'
        path.write_text(json.dumps(result))
        gate = {k: v for k, v in result.items() if k not in ('rows', 'evaluation_complete')}
        gate.update(passed_tasks=378, all_base_plus_passed=True, task_ids=ids,
                    result_path=str(path), evidence_sha256=runner.sha(path), result_sha256=runner.sha(path),
                    evaluator_patch={'applied': False})
        return gate, ids, counts, path, result

    def test_complete_canonical_evidence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, ids, counts, _, _ = self.canonical_fixture(directory)
            runner.verify_canonical_gate(gate, 'runtime', ids, counts, {'applied': False})

    def test_real_mbpp793_empty_plus_passes_but_missing_or_truncated_details_fail(self):
        # Official MbppPlus-v0.2.0 Mbpp/793: 3 base inputs, 0 plus inputs.
        # This checks evidence shape only; no canonical/model code is executed.
        with tempfile.TemporaryDirectory() as directory:
            gate, ids, counts, path, result = self.canonical_fixture(directory)
            old_id = ids[-1]
            ids[-1] = 'Mbpp/793'
            counts.pop(old_id)
            counts['Mbpp/793'] = {'base': 3, 'plus': 0}
            result['rows'][-1] = {'task_id': 'Mbpp/793', 'plus_passed': True,
                                  'base': runner.suite_record('pass', [True] * 3, 3),
                                  'plus': runner.suite_record('pass', [], 0)}
            gate['task_ids'] = ids
            def check(value):
                path.write_text(json.dumps(value))
                changed = {**gate, 'evidence_sha256': runner.sha(path), 'result_sha256': runner.sha(path)}
                runner.verify_canonical_gate(changed, 'runtime', ids, counts, {'applied': False})
            check(result)
            for case in ('missing_details', 'missing_tests', 'null_details', 'mapping_details',
                         'bool_tests', 'truncate_base', 'truncate_other_plus'):
                broken = copy.deepcopy(result)
                plus = broken['rows'][-1]['plus']
                if case == 'missing_details':
                    plus.pop('details')
                elif case == 'missing_tests':
                    plus.pop('tests')
                elif case == 'null_details':
                    plus['details'] = None
                elif case == 'mapping_details':
                    plus['details'] = {}
                elif case == 'bool_tests':
                    plus['tests'] = False
                elif case == 'truncate_base':
                    broken['rows'][-1]['base'] = runner.suite_record('pass', [], 0)
                else:
                    broken['rows'][0]['plus'] = runner.suite_record('pass', [], 0)
                with self.subTest(case=case), self.assertRaisesRegex(RuntimeError, 'details are incomplete'):
                    check(broken)

    def test_official_artifact_has_only_mbpp793_empty_plus_if_available(self):
        path = ROOT / 'data/mbpp/MbppPlus-v0.2.0.jsonl'
        if not path.exists():
            self.skipTest('Official artifact is private and not distributed in git')
        self.assertEqual(runner.sha(path), runner.DATA_SHA256)
        problems = [json.loads(line) for line in path.read_text().splitlines()]
        empty = [(row['task_id'], suite) for row in problems for suite in ('base', 'plus')
                 if len(row[suite + '_input']) == 0]
        self.assertEqual(empty, [('Mbpp/793', 'plus')])
        problem = next(row for row in problems if row['task_id'] == 'Mbpp/793')
        self.assertEqual(len(problem['base_input']), 3)

    def test_canonical_gate_rejects_changed_source_or_partial_count(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, ids, counts, _, _ = self.canonical_fixture(directory)
            for key, value in [('passed_tasks', 377), ('scorer_source_sha256', 'changed'),
                               ('manifest_sha256', 'changed'), ('all_base_plus_passed', False),
                               ('task_ids', ids[:-1])]:
                with self.assertRaises(RuntimeError):
                    runner.verify_canonical_gate({**gate, key: value}, 'runtime', ids, counts, {'applied': False})

    def test_canonical_evidence_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, ids, counts, path, _ = self.canonical_fixture(directory)
            path.write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 changed'):
                runner.verify_canonical_gate(gate, 'runtime', ids, counts, {'applied': False})

    def test_rehashed_duplicate_or_missing_tests_still_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, ids, counts, path, original = self.canonical_fixture(directory)
            for kind in ('duplicate', 'skipped_test', 'failed_test', 'partial'):
                result = copy.deepcopy(original)
                if kind == 'duplicate':
                    result['rows'][-1]['task_id'] = result['rows'][0]['task_id']
                elif kind == 'skipped_test':
                    result['rows'][0]['plus']['tests'] = 2
                    result['rows'][0]['plus']['details'].pop()
                elif kind == 'failed_test':
                    result['rows'][0]['plus']['details'][0] = False
                else:
                    result['evaluation_complete'] = False
                path.write_text(json.dumps(result))
                changed = {**gate, 'evidence_sha256': runner.sha(path), 'result_sha256': runner.sha(path)}
                with self.assertRaises(RuntimeError):
                    runner.verify_canonical_gate(changed, 'runtime', ids, counts, {'applied': False})

    def test_safety_requires_all_checks_exit_and_benign(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, result = self.canonical_fixture(directory)
            result['benign_fixture'] = runner.suite_record('pass', [True, True], 2)
            runner.verify_safety_gate(result, 'runtime', {'applied': False})
            for change in ('exit', 'missing', 'false', 'benign'):
                broken = copy.deepcopy(result)
                if change == 'exit':
                    broken['exit_code'] = 1
                elif change == 'missing':
                    broken['safety']['checks'].pop('network_blocked')
                elif change == 'false':
                    broken['safety']['checks']['network_blocked'] = False
                else:
                    broken['benign_fixture']['details'].pop()
                with self.assertRaises(RuntimeError):
                    runner.verify_safety_gate(broken, 'runtime', {'applied': False})

    def test_inside_entry_cannot_execute_on_host(self):
        with self.assertRaisesRegex(RuntimeError, 'only run after chroot'):
            runner.inside('/nonexistent.json')

    def test_identity_selection_reserves_other_project_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for name, uid in (('evalplus', 60000), ('mbpp', 60001)):
                target = project / '.sandbox' / name / 'manifest.json'
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps({'identity': {'uid': uid, 'gid': uid}}))
            with mock.patch.object(prepare, 'PROJECT', project), mock.patch.object(prepare.pwd, 'getpwall', return_value=[]), mock.patch.object(prepare.grp, 'getgrall', return_value=[]):
                self.assertNotIn(prepare.select_identity()['uid'], (60000, 60001))


if __name__ == '__main__':
    unittest.main()
