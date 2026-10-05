"""CPU-only scheduling and fail-closed AIME queue gates; no model loading."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
import venv
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import launch_aime_suite as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def predecessor():
    result = {'status': 'completed', 'full_suite_verified': True, 'smoke_only': False,
              'smoke_gate': {'status': 'passed'}, 'jobs': {}, 'comparisons': {}}
    for name in {f'{t}_{a}' for t in q.MC_TASKS for a in q.MC_ARMS} | {f'smoke_sciq_{a}' for a in q.MC_ARMS}:
        result['jobs'][name] = {'status': 'completed', 'exit_code': 0}
    for name in {f'{t}_{a}_comparison' for t in q.MC_TASKS for a in q.MC_CANDIDATES} | {f'smoke_sciq_{a}_comparison' for a in q.MC_CANDIDATES}:
        result['comparisons'][name] = {'status': 'completed', 'exit_code': 0}
    return result


class BasicGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_missing_and_running_predecessor_wait(self):
        path = self.root/'matrix.json'
        self.assertIsNone(q.predecessor_snapshot(path)[0])
        write(path, {'status': 'running', 'jobs': {}, 'comparisons': {}})
        self.assertIsNone(q.predecessor_snapshot(path)[0])

    def test_all_fourteen_jobs_and_comparisons_required(self):
        path = self.root/'matrix.json'
        complete = predecessor()
        write(path, complete)
        self.assertIsNotNone(q.predecessor_snapshot(path)[0])
        for section in ('jobs', 'comparisons'):
            damaged = copy.deepcopy(complete)
            damaged[section].pop(next(iter(damaged[section])))
            write(path, damaged)
            with self.assertRaisesRegex(ValueError, 'missing/incomplete'):
                q.predecessor_snapshot(path)

    def test_failed_running_predecessor_is_not_waited_forever(self):
        path = self.root/'matrix.json'
        write(path, {'status': 'running', 'jobs': {'x': {'status': 'failed'}}})
        with self.assertRaisesRegex(ValueError, 'failed/blocked'):
            q.predecessor_snapshot(path)

    def test_completed_smoke_only_cannot_release_devices(self):
        value = predecessor()
        value['smoke_only'] = True
        path = self.root/'matrix.json'
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'full suite'):
            q.predecessor_snapshot(path)

    def test_gpu_needs_both_low_memory_and_no_process(self):
        base = {'uuid': 'GPU-0', 'memory_mib': 999, 'compute_pids': []}
        self.assertTrue(q.device_idle(base))
        self.assertFalse(q.device_idle({**base, 'memory_mib': 1000}))
        self.assertFalse(q.device_idle({**base, 'compute_pids': [123]}))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            q.device_idle({**base, 'memory_mib': '0'})

    def test_gpu_query_errors_fail_closed(self):
        with patch.object(q.subprocess, 'check_output', side_effect=RuntimeError('query failed')):
            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                q.gpu_status(0)

    def test_durable_claim_blocks_different_output(self):
        path = q.claim_batch(self.root, self.root/'first')
        with self.assertRaises(FileExistsError):
            q.claim_batch(self.root, self.root/'second')
        self.assertEqual(json.loads(path.read_text())['output'], str(self.root/'first'))
        self.assertTrue(path.exists())

    def test_source_pin_detects_mutated_input(self):
        path = self.root/'data'
        path.write_text('before')
        pins = {str(path): q.sha(path)}
        path.write_text('after')
        with self.assertRaisesRegex(ValueError, 'changed'):
            q.verify_pins(pins)

    def test_virtualenv_symlink_entry_is_preserved_and_used_by_child(self):
        environment = self.root/'private-venv'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        entry = environment/'bin/python'
        self.assertTrue(entry.is_symlink())
        self.assertNotEqual(entry.resolve(), entry.absolute())
        args = SimpleNamespace(project=self.root, output=self.root/'output',
                               protocol=None, data_root=None, models_root=None,
                               python=str(entry), poll_seconds=30, source_commit='unused')
        instance = q.AIMEQueue.__new__(q.AIMEQueue)
        # Stop before repository/model verification, after the real initializer
        # has validated and stored its interpreter path.
        with patch.object(q, 'source_tree', side_effect=RuntimeError('source gate reached')):
            with self.assertRaisesRegex(RuntimeError, 'source gate reached'):
                instance.__init__(args)
        self.assertEqual(instance.python, str(entry.absolute()))
        instance.source = self.root
        probe = self.root/'scripts/probe.py'
        probe.parent.mkdir()
        probe.write_text('import json,sys; print(json.dumps({"prefix":sys.prefix,"executable":sys.executable}))\n')
        command = instance._command('probe.py')
        self.assertEqual(command[0], str(entry.absolute()))
        result = json.loads(q.subprocess.check_output(command, text=True))
        self.assertEqual(Path(result['prefix']).resolve(), environment.resolve())
        self.assertEqual(Path(result['executable']).absolute(), entry.absolute())

    def test_predecessor_inputs_verified_before_recomputation(self):
        suite = self.root/'results/runs/20261004-huginn-r16-suite'
        old = self.root/'results/runs/20261003-huginn-mc-suite'
        source = self.root/'releases/huginn-mc-final'
        launcher = self.root/'releases/phase2-26ee9ac'
        matrix = predecessor()
        source_hashes = {'scripts/compare_huginn.py': 'hash'}
        launcher_hashes = {'scripts/launch_huginn_r16_suite.py': 'launcher'}
        matrix.update(source_root=str(source), source_commit=q.MC_SOURCE_COMMIT, half_depth_suite=str(old),
                      source_sha256=source_hashes, launcher={'commit': q.MC_LAUNCHER_COMMIT, 'release': str(launcher),
                      'path': str(launcher/'scripts/launch_huginn_r16_suite.py'), 'sha256': 'launcher'})
        left, right = suite/'sciq_baseline16', suite/'sciq_hidden16'
        write(left/'manifest.json', {'status': 'completed'})
        write(right/'manifest.json', {'status': 'completed'})
        entry = matrix['comparisons']['sciq_hidden16_comparison']
        entry['input_files'] = {str(left): q.inventory(left), str(right): q.inventory(right)}
        write(left/'manifest.json', {'status': 'changed'})
        write(suite/'matrix.json', matrix)
        comparer = Mock()
        with patch.object(q, 'source_tree', side_effect=[source_hashes, launcher_hashes]), patch.object(q, 'load_module', return_value=comparer):
            with self.assertRaisesRegex(ValueError, 'input bytes changed'):
                q.verify_predecessor(self.root, suite/'matrix.json', matrix)
        comparer.compare_huginn.assert_not_called()


class FakeQueue(q.AIMEQueue):
    def __init__(self, root, failed_smoke=None, failed_full=None):
        self.output = root
        self.output.mkdir()
        self.lock = threading.RLock()
        self.abort = threading.Event()
        self.cancelled = threading.Event()
        self.children = {}
        self.events = []
        self.failed_smoke, self.failed_full = failed_smoke, failed_full
        self.state = {'status': 'waiting', 'smoke_gate': {'status': 'waiting'},
                      'workers': {m: {} for m in q.MODELS}, 'jobs': {}, 'full_suite_verified': False}
        self.state['jobs']['canonical'] = {'status': 'pending'}
        for model in q.MODELS:
            self.state['jobs']['smoke_'+model] = {'status': 'pending'}
            for year in q.YEARS:
                for phase in ('generate', 'score'):
                    self.state['jobs'][f'{phase}_{model}_{year}'] = {'status': 'pending'}

    def _check_inputs(self):
        if self.abort.is_set():
            raise RuntimeError('abort')

    def _done(self, name):
        with self.lock:
            self.events.append(name)
            self.state['jobs'][name]['status'] = 'completed'

    def _canonical(self):
        self._done('canonical')

    def _predecessor(self):
        self.events.append('predecessor_verified')

    def _smoke(self, model):
        if self.failed_smoke == model:
            self.abort.set()
            self.state['jobs']['smoke_'+model]['status'] = 'failed'
            raise RuntimeError('smoke failed')
        self._done('smoke_'+model)

    def _generation(self, model, year):
        self.assert_barrier()
        if self.failed_full == (model, year):
            self.state['jobs'][f'generate_{model}_{year}']['status'] = 'failed'
            raise RuntimeError('full failed')
        self._done(f'generate_{model}_{year}')

    def assert_barrier(self):
        assert self.state['smoke_gate']['status'] == 'passed'
        assert all('smoke_'+model in self.events for model in q.MODELS)
        assert 'predecessor_verified' in self.events

    def _score(self, model, year):
        self._done(f'score_{model}_{year}')


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_both_smokes_precede_every_formal_job(self):
        queue = FakeQueue(self.root/'success')
        self.assertEqual(queue.run(), 0)
        first_generate = min(i for i, name in enumerate(queue.events) if name.startswith('generate'))
        self.assertTrue(all(queue.events.index('smoke_'+m) < first_generate for m in q.MODELS))
        self.assertEqual(queue.state['total_generated_rows'], 5760)
        self.assertTrue(queue.state['full_suite_verified'])
        for model in q.MODELS:
            self.assertLess(queue.events.index(f'score_{model}_2024'), queue.events.index(f'generate_{model}_2025'))

    def test_failed_smoke_blocks_all_formal_generations(self):
        queue = FakeQueue(self.root/'failed', failed_smoke=q.MODELS[1])
        self.assertEqual(queue.run(), 1)
        self.assertFalse(any(x.startswith('generate') for x in queue.events))
        self.assertFalse(queue.state['full_suite_verified'])
        self.assertEqual(queue.state['smoke_gate']['status'], 'failed')

    def test_failed_full_does_not_retry_or_advance_its_year(self):
        model = q.MODELS[0]
        queue = FakeQueue(self.root/'failed', failed_full=(model, 2024))
        self.assertEqual(queue.run(), 1)
        self.assertNotIn(f'generate_{model}_2025', queue.events)
        self.assertNotIn(f'score_{model}_2024', queue.events)
        self.assertEqual(queue.state['jobs'][f'generate_{model}_2025']['status'], 'blocked_by_failed_batch')

    def test_signals_only_mark_cancellation_until_owned_cleanup(self):
        queue = FakeQueue(self.root/'signal')
        with patch.object(q.os, 'killpg') as kill:
            queue.cancel(signal.SIGTERM)
        kill.assert_not_called()
        self.assertTrue(queue.abort.is_set())
        self.assertTrue(queue.cancelled.is_set())

    def test_cleanup_never_signals_unregistered_process(self):
        queue = FakeQueue(self.root/'cleanup')
        stranger = Mock(pid=800, poll=Mock(return_value=None))
        owned = Mock(pid=900, poll=Mock(return_value=None))
        queue.children[900] = owned
        with patch.object(q.os, 'killpg') as kill:
            queue._terminate_owned(stranger)
            kill.assert_not_called()
            queue._terminate_owned(owned)
            kill.assert_called_once_with(900, signal.SIGTERM)
        owned.wait.assert_called_once_with(timeout=20)

    def test_changed_source_stops_before_gpu_probe_or_spawn(self):
        queue = FakeQueue(self.root/'changed')
        queue.source = self.root
        queue.sources = {'expected': 'sha'}
        queue.pins = {}
        # Restore production gate for this test.
        queue._check_inputs = q.AIMEQueue._check_inputs.__get__(queue)
        queue._wait_gpu = Mock()
        with patch.object(q, 'current_sources', return_value={'different': 'sha'}), patch.object(q.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'source changed'):
                queue._job('canonical', ['unused'], lambda: {}, model=q.MODELS[0])
        spawn.assert_not_called()
        queue._wait_gpu.assert_not_called()

    def test_cancel_reaps_a_real_owned_cpu_child(self):
        queue = FakeQueue(self.root/'child')
        queue.source = self.root
        (queue.output/'logs').mkdir()
        timer = threading.Timer(.1, queue.cancel, args=(signal.SIGTERM,))
        timer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'owned child terminated'):
                q.AIMEQueue._job(queue, 'canonical', [sys.executable, '-c', 'import time; time.sleep(30)'], lambda: {})
        finally:
            timer.cancel()
            timer.join()
        self.assertFalse(queue.children)
        self.assertEqual(queue.state['jobs']['canonical']['status'], 'cancelled')


class GenerationBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.queue = FakeQueue(self.root/'batch')
        queue = self.queue
        queue.models, queue.data, queue.protocol = self.root/'models', self.root/'data', self.root/'protocol'
        queue.source, queue.python = self.root/'release', 'python'
        queue.args = SimpleNamespace(source_commit='pinned-commit')
        queue.sources = {'scripts/generate_aime.py': 'pinned-sha', 'scripts/score_aime.py': 'scorer-sha'}
        queue.identities = {q.MODELS[0]: {'repo_id': 'ByteDance/'+q.MODELS[0], 'model_code_sha256': 'code'}}
        queue.scoring_data = {'protocol': {}}
        queue.scorer = SimpleNamespace(read_generation=Mock(return_value={}), verify_paired_generations=Mock())
        queue.helper = SimpleNamespace(guidance_dict=lambda arm, identity, protocol: {'mode': arm, 'omega': .5})
        queue.pins = {}
        self.model, self.year = q.MODELS[0], 2024
        self.commands = []
        queue._job = lambda name, command, validate, model=None: self.commands.append(command) or validate()
        self.smoke = queue.output/f'smoke_{self.model}.json'
        self.smoke.write_text('smoke evidence')
        for arm in q.ARMS:
            directory = queue.output/'full'/self.model/str(self.year)/arm
            directory.mkdir(parents=True)
            (directory/'samples.jsonl').write_text('retained rows\n')
            manifest = {'status': 'completed', 'is_full_split': True, 'expected_samples': 480, 'completed_samples': 480,
                        'source_git_observed': 'pinned-commit', 'gpu_smoke_sha256': q.sha(self.smoke),
                        'samples_sha256': q.sha(directory/'samples.jsonl'),
                        'config': {'source': {'git_commit': 'pinned-commit', 'source_sha256': queue.sources,
                                              'loaded_model_code_sha256': 'code'}, 'model_identity': queue.identities[self.model],
                                   'dataset': {'year': 2024}, 'guidance': {'mode': arm, 'omega': .5}}}
            write(directory/'manifest.json', manifest)

    def manifest(self):
        return self.queue.output/'full'/self.model/str(self.year)/'adaptive/manifest.json'

    def test_generation_command_has_fixed_protocol_and_no_resume_or_debug(self):
        result = q.AIMEQueue._generation(self.queue, self.model, self.year)
        self.assertEqual(set(result), set(q.ARMS))
        command = self.commands[0]
        self.assertIn('--smoke', command)
        self.assertNotIn('--resume', command)
        self.assertFalse(any(x.startswith('--debug') for x in command))
        self.assertEqual(command[command.index('--year')+1], '2024')

    def test_self_consistent_other_source_commit_is_rejected(self):
        path = self.manifest()
        manifest = json.loads(path.read_text())
        manifest['config']['source']['git_commit'] = 'different'
        write(path, manifest)
        with self.assertRaisesRegex(ValueError, 'another model/source/protocol'):
            q.AIMEQueue._generation(self.queue, self.model, self.year)

    def test_changed_smoke_evidence_is_rejected(self):
        self.smoke.write_text('different evidence')
        with self.assertRaisesRegex(ValueError, 'another model/source/protocol'):
            q.AIMEQueue._generation(self.queue, self.model, self.year)

    def test_truncated_sample_file_is_rejected(self):
        (self.manifest().parent/'samples.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'sample bytes'):
            q.AIMEQueue._generation(self.queue, self.model, self.year)

    def test_full_row_validator_cannot_be_replaced_by_manifest_counts(self):
        self.queue.scorer.read_generation.side_effect = ValueError('full sample count')
        with self.assertRaisesRegex(ValueError, 'full sample count'):
            q.AIMEQueue._generation(self.queue, self.model, self.year)

    def test_pairing_failure_stops_before_scoring(self):
        self.queue.scorer.verify_paired_generations.side_effect = ValueError('paired sample seed')
        with self.assertRaisesRegex(ValueError, 'paired sample seed'):
            q.AIMEQueue._generation(self.queue, self.model, self.year)

    def make_summary(self):
        queue = self.queue
        queue.protocol.write_text('fixed protocol')
        write(queue.data/'manifest.json', {'registered': True})
        queue.scoring_data['files'] = {'questions': 'same'}
        arms = {}
        for arm in q.ARMS:
            directory = queue.output/'full'/self.model/str(self.year)/arm
            arms[arm] = {'manifest_sha256': q.sha(directory/'manifest.json'),
                         'samples_sha256': q.sha(directory/'samples.jsonl'), 'run_path': str(directory)}
        result = {'status': 'PASS', 'full_480_verified': True, 'n_tasks': 30, 'samples_per_problem': 16,
                  'rows_per_arm': 480, 'year': self.year, 'model_identity': queue.identities[self.model],
                  'protocol_sha256': q.sha(queue.protocol), 'data_manifest_sha256': q.sha(queue.data/'manifest.json'),
                  'data_files': queue.scoring_data['files'], 'gpu_smoke_sha256': q.sha(self.smoke),
                  'common_config': {'source': {'git_commit': queue.args.source_commit, 'source_sha256': queue.sources}},
                  'arms': arms, 'pairs': {'fixed': {}, 'adaptive': {}},
                  'validation': {key: True for key in ('all60_gold_canonical_gate', 'complete_ordered_480_per_arm',
                    'source_runtime_config_paired', 'public_prompts_reconstructed', 'token_ids_seed_question_hashes_paired',
                    'native_r4_cache_observations_verified', 'gold_question_file_hashes_verified')}}
        result['validation']['model_answers_executed'] = False
        path = queue.output/'scores'/self.model/str(self.year)/'summary.json'
        write(queue.output/'canonical/canonical_gate.json', {'gate': 'same'})
        write(path.parent/'canonical_gate.json', {'gate': 'same'})
        result['scorer_source_sha256'] = queue.sources['scripts/score_aime.py']
        result['canonical_gate_file_sha256'] = q.sha(path.parent/'canonical_gate.json')
        for arm in q.ARMS:
            write(path.parent/(arm+'.json'), {'scores': 'fixture'})
            result['arms'][arm]['score_file_sha256'] = q.sha(path.parent/(arm+'.json'))
        write(path, result)
        return path, result

    def test_scorer_output_must_bind_actual_generation_bytes(self):
        path, result = self.make_summary()
        result['arms']['adaptive']['manifest_sha256'] = 'different'
        write(path, result)
        with self.assertRaisesRegex(ValueError, 'different generation inputs'):
            q.AIMEQueue._score(self.queue, self.model, self.year)

    def test_score_cannot_claim_other_frozen_source_or_year(self):
        path, original = self.make_summary()
        for change in ('source', 'year'):
            result = copy.deepcopy(original)
            if change == 'source':
                result['common_config']['source']['git_commit'] = 'another'
            else:
                result['year'] = 2025
            write(path, result)
            with self.assertRaisesRegex(ValueError, 'three complete'):
                q.AIMEQueue._score(self.queue, self.model, self.year)

    def test_bound_score_is_accepted(self):
        path, result = self.make_summary()
        evidence = q.AIMEQueue._score(self.queue, self.model, self.year)
        self.assertEqual(evidence['sha256'], q.sha(path))
        self.assertTrue(evidence['full_480_verified'])


if __name__ == '__main__':
    unittest.main()
