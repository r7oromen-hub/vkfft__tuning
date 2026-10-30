import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_rotation_runner_test', HERE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Ledger:
    def __init__(self, limit=None): self.start_calls = []; self.finish_calls = []; self.events = []; self.limit = limit
    def start(self, work, expected):
        if self.limit is not None and len(self.start_calls) >= self.limit: return False
        self.start_calls.append((work, expected)); return True
    def finish(self, status): self.finish_calls.append(status)
    def event(self, kind, **details): self.events.append((kind, details))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.args = types.SimpleNamespace(gpu='GPU-test', rotation_original=Path('original'), rotation_expanded=Path('expanded'),
                                          timeout=3, output=Path('unused'))
        self.case = {'n': 256, 'batch': 16, 'candidate_id': 'candidate',
                     'request': {'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4},
                     'hot_original_signature': {'original': 1}, 'hot_candidate_signature': {'candidate': 1},
                     'rotation': {'slots': 2, 'total_bytes': 131072, 'graph_repeats': 64, 'l2_bytes': 16384},
                     'status': 'eligible'}
        self.proof = {'plan_sha256': 'plan', 'cases': [self.case]}
        self.contract = {'rotation_build': {'binaries': {'original': 'orig', 'expanded': 'exp'}}}
        self.events, self.calls = [], []

    def fake_run(self, args, workload, candidate, phase, round_no, order):
        self.calls.append((candidate, workload.copy(), phase, round_no, order))
        result = {'status': 'ok', 'raw_interval_ms_samples': [2.5],
                  'exec_ms_samples': [0.9 if candidate[0] != 'original' else 1.0]}
        if getattr(self, 'numerical_fail', False) and phase == 'rotation_timing':
            result['status'] = 'correctness_failed'
        if getattr(self, 'calibration_uncovered', False) and phase == 'rotation_calibration':
            result = {'status': 'error', 'error': 'rotation_unsupported_device_memory', 'failure_phase': 'initialize',
                      'gpu': {'uuid': 'GPU-test', 'cuda_visible_devices': 'GPU-test'},
                      'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': workload['graph_repeats'],
                      'header_instrumentation': False, 'requested_config': {'makeForwardPlanOnly': 1}}
            return {'record_type': 'execution', 'status': 'failed', 'returncode': 1, 'guard_clean': True, 'guard_gpu_uuid': 'GPU-test',
                    'guard_run_id': 'run-1', 'result': result, 'candidate_id': candidate[0],
                    'direct_environment': candidate[1], 'workload': workload}
        return {'record_type': 'execution', 'status': 'ok', 'guard_run_id': 'run-' + str(len(self.calls)), 'result': result}

    def session(self, ledger):
        with patch.dict(os.environ, {'VKFFT_STUDY_GUARD_ID': 'guard', 'VKFFT_STUDY_GUARD_UUID': 'GPU-test'}), \
             patch.object(runner.frontier, 'append', side_effect=lambda path, row: self.events.append(row)), \
             patch.object(runner.suite, 'run_one', side_effect=self.fake_run), \
             patch.object(runner.rotation, 'check', side_effect=lambda row, *a, **kw: row['result']), \
             patch.object(runner.search, 'source_signature', side_effect=lambda _r, original: {'original': 1} if original else {'candidate': 1}):
            return runner.run_session(self.args, ledger, self.contract, self.proof)

    def test_ten_pairs_after_calibration_preserve_working_set(self):
        ledger = Ledger(); self.session(ledger)
        self.assertEqual(len(self.calls), 21)
        self.assertEqual(self.calls[0][2], 'rotation_calibration')
        self.assertEqual(len([row for row in self.events if row['record_type'] == 'rotation_pair']), 10)
        disposition = next(row for row in self.events if row['record_type'] == 'rotation_disposition')
        self.assertEqual(disposition['status'], 'complete')
        self.assertEqual(disposition['rotation']['l2_bytes'], 16384)
        self.assertEqual(disposition['rotation']['total_bytes'], 131072)
        self.assertEqual((self.events[-1]['status'], self.events[-1]['measured_count'], self.events[-1]['uncovered_count']),
                         ('complete', 1, 0))
        self.assertEqual(len(ledger.start_calls), len(ledger.finish_calls))
        self.assertTrue(all(call[1]['graph_repeats'] == 64 for call in self.calls))

    def test_unsupported_pool_and_device_report_without_claim(self):
        self.case['rotation'] = None; self.session(Ledger())
        self.assertFalse(self.calls)
        self.assertEqual(next(row for row in self.events if row['record_type'] == 'rotation_disposition')['status'],
                         'uncovered_pool_or_graph_limit')
        self.assertEqual((self.events[-1]['status'], self.events[-1]['measured_count'], self.events[-1]['uncovered_count']),
                         ('complete_with_uncovered', 0, 1))
        self.events.clear(); self.case['rotation'] = {'slots': 2, 'total_bytes': 131072, 'graph_repeats': 64, 'l2_bytes': 16384}
        self.calibration_uncovered = True; self.session(Ledger())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(next(row for row in self.events if row['record_type'] == 'rotation_disposition')['status'], 'uncovered')

    def test_uncovered_requires_matching_device_environment_and_phase(self):
        self.calibration_uncovered = True
        workload = {'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': 64}
        row = self.fake_run(self.args, workload, ('original', {}), 'rotation_calibration', 0, 0)
        with patch.object(runner.search, 'source_signature', return_value={'original': 1}):
            self.assertEqual(runner.uncovered(row, 'GPU-test', self.case, self.case['rotation']), 'rotation_unsupported_device_memory')
            for field in ('environment', 'device', 'phase', 'signature'):
                from copy import deepcopy
                bad = deepcopy(row)
                if field == 'environment': bad['direct_environment'] = {'VKFFT_EXPANDED_REGS': '8'}
                if field == 'device': bad['result']['gpu']['cuda_visible_devices'] = 'wrong'
                if field == 'phase': bad['result']['failure_phase'] = 'graph_build'
                if field == 'signature': bad['candidate_id'] = 'wrong'
                self.assertIsNone(runner.uncovered(bad, 'GPU-test', self.case, self.case['rotation']))

    def test_numerical_failure_stops_before_other_side(self):
        self.numerical_fail = True
        with self.assertRaises(runner.frontier.NumericalFailure): self.session(Ledger())
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[-1][2], 'rotation_timing')
        self.assertEqual(self.events[-1]['record_type'], 'rotation_numerical_stop')
        self.assertFalse(any(row['record_type'] == 'rotation_pair' for row in self.events))

    def test_budget_denial_prevents_pair(self):
        with self.assertRaises(runner.frontier.BudgetStop): self.session(Ledger(limit=1))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2], 'rotation_calibration')

    def test_absolute_deadline_prevents_work_even_if_ledger_allows_it(self):
        ledger = Ledger()
        bound = runner.DeadlineLedger(ledger, deadline=200, clock=lambda: 170)
        self.assertFalse(bound.start({'phase': 'timing'}, 30))
        self.assertFalse(ledger.start_calls)
        self.assertEqual(ledger.events[0][1]['reason'], 'absolute_campaign_or_stage_deadline')
        self.assertTrue(bound.start({'phase': 'timing'}, 29))
        self.assertEqual(len(ledger.start_calls), 1)

    def test_original_tree_must_be_committed_checkout(self):
        root = HERE.parents[2]
        with self.assertRaisesRegex(ValueError, 'committed submodule'):
            runner.verify_rotation_build(root, HERE, HERE, HERE, HERE, {}, {}, HERE, HERE, HERE, 'commit')

    def test_gpu_child_retains_lock_without_affecting_monitor_commands(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            binary = Path(tmp) / 'binary'; binary.write_text('x')
            calls = []
            launch = runner.lock_inheriting_popen(lambda *args, **kw: calls.append((args, kw)), 10, [binary])
            launch([str(binary)], pass_fds=(8,))
            launch(['nvidia-smi'])
            self.assertEqual(calls[0][1]['pass_fds'], (8, 10))
            self.assertNotIn('pass_fds', calls[1][1])

    def test_campaign_deadline_and_artifact_binding(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp); protocol = root / 'protocol.json'; record = root / 'record.json'
            start = root / 'start.json'; budget = root / 'budget.jsonl'
            protocol.write_text(json.dumps({'schema': 'frontier-campaign/v1', 'gpu_uuid': 'GPU-test',
                                            'total_reserved_wall_seconds': 200}), encoding='utf8')
            start.write_text(json.dumps({'start_monotonic': 100, 'total_reserved_wall_seconds': 200,
                                         'gpu': 'GPU-test', 'execution_commit': 'commit'}), encoding='utf8')
            state = {'contract': {'gpu': 'GPU-test', 'execution_commit': 'commit'}, 'boot': 'boot',
                     'start': 110, 'last': 120, 'inflight': None}
            budget.write_text(json.dumps({'event': 'start', 'state': state}) + '\n', encoding='utf8')
            items = [{'path': p.relative_to(root).as_posix(), 'sha256': runner.frontier.sha(p), 'bytes': p.stat().st_size}
                     for p in (start, budget)]
            record.write_text(json.dumps({'schema': 'git-experiment/v1', 'status': 'completed', 'execution_commit': 'commit',
                                          'protocol': {'path': 'protocol.json', 'sha256': runner.frontier.sha(protocol)},
                                          'artifacts': items}), encoding='utf8')
            with patch.object(runner.frontier, 'git', return_value=''):
                result = runner.campaign(root, record, start, budget, 'GPU-test', now=150, boot='boot')
                self.assertEqual(result['deadline'], 300)
                with self.assertRaisesRegex(ValueError, 'deadline'):
                    runner.campaign(root, record, start, budget, 'GPU-test', now=300, boot='boot')
                with self.assertRaisesRegex(ValueError, 'boot'):
                    runner.campaign(root, record, start, budget, 'GPU-test', now=150, boot='different')
                start.write_text('{}', encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    runner.campaign(root, record, start, budget, 'GPU-test', now=150, boot='boot')


if __name__ == '__main__':
    unittest.main()
