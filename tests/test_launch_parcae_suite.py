"""Queue integration with fake children; these tests never certify a GPU run."""
import contextlib
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import launch_parcae_suite as queue
from test_gpu_smoke_parcae import certificate


def predecessor():
    return dict(source_commit=queue.PREDECESSOR_COMMIT, source_root='/fake/frozen', source_sha256={'source': 'hash'},
                gpu='3', status='completed_generation_unscored', smoke_gate='passed',
                jobs={name: dict(status='completed', exit_code=0) for name in ('smoke', 'full')})


class QueueTests(unittest.TestCase):
    def test_cleanup_only_signals_owned_group_and_escalates_after_timeout(self):
        child = MagicMock(pid=6789, returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = [queue.subprocess.TimeoutExpired('owned', 30), -9]
        with patch.object(queue.os, 'killpg') as kill:
            queue.stop_owned_child(child)
        self.assertEqual([tuple(call.args) for call in kill.call_args_list],
                         [(6789, queue.signal.SIGTERM), (6789, queue.signal.SIGKILL)])
        child.poll.return_value = 0
        with patch.object(queue.os, 'killpg') as kill:
            queue.stop_owned_child(child)
            kill.assert_not_called()

    def test_predecessor_missing_failed_or_partial_cannot_unlock(self):
        self.assertTrue(queue.predecessor_ready(predecessor()))
        r = predecessor(); r['status'] = 'running'
        self.assertFalse(queue.predecessor_ready(r))
        for key, val in [('status', 'failed'), ('gpu', '2'), ('source_commit', 'other'), ('smoke_gate', 'pending'), ('jobs', {})]:
            r = predecessor(); r[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError): queue.predecessor_ready(r)

    def test_claim_is_permanent_across_output_directories(self):
        with tempfile.TemporaryDirectory() as d:
            queue.claim(Path(d), Path('/fake/output1'), 'a')
            with self.assertRaises(FileExistsError): queue.claim(Path(d), Path('/fake/output2'), 'b')

    def test_actual_generation_sources_must_match_predecessor(self):
        with tempfile.TemporaryDirectory() as d:
            for arm in ('baseline32', 'hidden32', 'baseline16', 'hidden16'):
                path = Path(d)/arm; path.mkdir()
                (path/'manifest.json').write_text(json.dumps(dict(config=dict(source=dict(
                    git_commit=queue.PREDECESSOR_COMMIT, source_sha256={'source': 'hash'})))))
            self.assertEqual(len(queue.verify_generation_sources(d, {'source': 'hash'})), 4)
            path = Path(d)/'hidden16/manifest.json'
            r = json.loads(path.read_text()); r['config']['source']['git_commit'] = 'wrong'
            path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'Actual MBPP generation source'):
                queue.verify_generation_sources(d, {'source': 'hash'})

    def simulate(self, *, cpu_gate=False, failed_smoke=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); before = root/'predecessor'; before.mkdir()
            (before/'matrix.json').write_text(json.dumps(predecessor()))
            for arm in ('baseline32', 'hidden32', 'baseline16', 'hidden16'):
                directory = before/'full'/arm; directory.mkdir(parents=True)
                (directory/'manifest.json').write_text(json.dumps(dict(config=dict(source=dict(
                    git_commit=queue.PREDECESSOR_COMMIT, source_sha256={'source': 'hash'})))))
            output = root/'suite'; commands = []; probes = []
            class Child:
                pid = 12345
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    script = Path(command[2]).name
                    target = Path(command[command.index('--output')+1])
                    self.returncode = 0
                    if script == 'gpu_smoke_parcae.py':
                        target.mkdir()
                        r = certificate()
                        if cpu_gate: r['device_type'] = 'cpu'
                        (target/'result.json').write_text(json.dumps(r))
                    elif script == 'evaluate_parcae.py': target.mkdir()
                    else:
                        smoke = target.name.startswith('smoke_')
                        task = target.stem.removeprefix('smoke_').removesuffix('_comparison')
                        if failed_smoke and smoke: self.returncode = 1
                        else: target.write_text(json.dumps(dict(task=task, is_full_split=not smoke, n_documents=2 if smoke else 1000)))
                def poll(self): return self.returncode
            def probe(gpu):
                probes.append(gpu)
                return dict(ready=True, memory_mib=0, compute_pids=[])
            argv = ['queue', '--source-commit', 'test', '--model', str(root/'model'), '--output', str(output),
                    '--claim-dir', str(root/'claims'), '--predecessor', str(before), '--mbpp-data', str(root/'mbpp'),
                    '--prompt-audit', str(root/'audit'), '--harness-reference', str(root/'harness'),
                    '--dataset-cache', str(root/'data'), '--hub-cache', str(root/'hub')]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(queue, 'source_tree', return_value={'source': 'hash'}))
                stack.enter_context(patch.object(queue, 'verify_generation', return_value={'paired': True}))
                stack.enter_context(patch.object(queue, 'gpu_status', side_effect=probe))
                stack.enter_context(patch.object(queue.subprocess, 'Popen', Child))
                stack.enter_context(patch.object(queue.signal, 'signal'))
                if cpu_gate or failed_smoke:
                    with self.assertRaises((ValueError, RuntimeError)): queue.main()
                else: queue.main()
            return json.loads((output/'matrix.json').read_text()), commands, probes

    def test_full_queue_gates_and_every_gpu_child_probe(self):
        state, commands, probes = self.simulate()
        self.assertEqual(state['status'], 'completed')
        self.assertTrue(state['full_suite_verified'])
        scripts = [Path(c[2]).name for c in commands]
        self.assertEqual(scripts[0], 'gpu_smoke_parcae.py')
        self.assertEqual(scripts[8], 'compare_parcae.py')
        self.assertEqual(scripts.count('evaluate_parcae.py'), 56)
        self.assertEqual(scripts.count('compare_parcae.py'), 8)
        self.assertEqual(len(probes), 57)

    def test_cpu_gate_and_failed_pairing_block_all_full_jobs(self):
        for options, expected_count in [({'cpu_gate': True}, 1), ({'failed_smoke': True}, 9)]:
            state, commands, _ = self.simulate(**options)
            self.assertEqual(state['status'], 'failed')
            self.assertFalse(state['full_suite_verified'])
            self.assertEqual(len(commands), expected_count)


if __name__ == '__main__': unittest.main()
