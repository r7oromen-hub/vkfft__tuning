import copy
import importlib.util
import json
import os
from pathlib import Path
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_knobs_audit_tests', HERE / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
runner = audit.runner


class Ledger:
    def __init__(self, contract, maximum=None):
        self.state = {'contract': contract, 'seconds': 1000, 'boot': 'boot', 'start': 100,
                      'last': 100, 'inflight': None}
        self.events = []
        self.maximum = maximum
        self.count = 0
        self.event('start')

    def event(self, event, **extra):
        self.events.append(dict(event=event, state=self.state.copy(), **extra))

    def start(self, identity, expected):
        if self.maximum is not None and self.count >= self.maximum:
            self.state['last'] = 1068
            self.event('denied', identity=identity)
            return False
        self.count += 1
        self.state['inflight'] = identity
        self.event('work_start')
        return True

    def finish(self, status):
        identity = self.state['inflight']
        self.state['inflight'] = None
        self.event('work_finish', identity=identity, status=status)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.gpu = 'GPU-test'
        self.n, self.batch = 128, 16
        self.req = {'radix': [8, 8, 2], 'logical_registers': 8, 'grouped_batch': 8,
                    'use_lut': -1, 'coalesced_memory': 64}
        self.cid = runner.requests.candidate_id(self.n, self.batch, self.req)
        self.manifest = (self.n, self.batch, [{'candidate_id': self.cid, 'request': self.req}])
        self.binaries = {'original': 'original_hash', 'expanded': 'expanded_hash'}
        self.contract = {'gpu': self.gpu, 'protocol': 'frontier-knobs-execution/v1',
                         'timeout': 2, 'binaries': self.binaries, 'execution_commit': 'a' * 40}
        self.rows = []
        self.args = types.SimpleNamespace(gpu=self.gpu, original=Path('/original'), expanded=Path('/expanded'),
                                          timeout=2, output=Path('/unused'))
        patches = [
            mock.patch.dict(os.environ, {'VKFFT_STUDY_GUARD_UUID': self.gpu, 'VKFFT_STUDY_GUARD_ID': 'guard'}),
            mock.patch.object(runner, 'run_one', side_effect=self.fake_one),
            mock.patch.object(runner.v2, 'append', side_effect=lambda path, row: self.rows.append(json.loads(json.dumps(row)))),
            mock.patch.object(runner.v2, 'representable', return_value=({'radix': [8, 8, 2],
                'logical_registers': 8, 'grouped_batch': 8}, None)),
            mock.patch.object(runner.validation, 'check_execute', side_effect=lambda row, *a, **kw: row['result']),
            mock.patch.object(runner.validation, 'check_describe', side_effect=lambda row, *a, **kw: row['result']),
            mock.patch.object(runner.validation, 'check_matched_identity'),
            mock.patch.object(runner.validation.comparison, 'compare'),
            mock.patch.object(runner.search, 'source_signature', side_effect=lambda result, **kw: result['role']),
            mock.patch.object(audit.guard_audit, 'audit', return_value={'status': 'pass', 'errors': []}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def fake_one(self, args, workload, candidate, phase, round_no, order):
        name, env = candidate
        original = name == 'original'
        result = {'role': 'original' if original else 'candidate', 'raw_interval_ms_samples': [2.5],
                  'exec_ms_samples': [1.0 if original else .8], 'warmup_actual': 5, 'warmup_wall_ms': 1000,
                  'precompile_kernels': [{'canonical_descriptor': {}, 'source_sha256': 'source',
                                          'source_bytes': 10, 'compiler_input': {}}]}
        command = [str(args.exe), '--n', str(workload['n']), '--batch', str(workload['batch']),
                   '--study-mode', workload['study_mode'], '--placement', 'outofplace']
        if workload['study_mode'] == 'execute':
            command += ['--seed', str(workload['seed']), '--timing-mode', 'graph',
                        '--graph-repeats', str(workload['graph_repeats']), '--warmup', str(workload['warmup']),
                        '--warmup-ms', str(workload['warmup_ms']), '--repeats', str(workload['repeats']),
                        '--pattern', workload['pattern']]
        command += workload['knob_args']
        return {'record_type': 'execution', 'candidate_id': name, 'phase': phase, 'round': round_no, 'order': order,
                'workload_id': f"n{workload['n']}_b{workload['batch']}", 'workload': workload,
                'command': command, 'direct_environment': env, 'executable_sha256': args.exe_sha256,
                'status': 'ok', 'returncode': 0, 'guard_clean': True, 'guard_gpu_uuid': self.gpu,
                'result': result, 'stdout': json.dumps(result), 'stderr': ''}

    def trace(self, maximum=None):
        ledger = Ledger(self.contract, maximum)
        try:
            runner.screen(self.args, ledger, self.contract, self.manifest)
        except runner.v2.BudgetStop:
            self.rows.append({'record_type': 'budget_exhausted', 'contract': self.contract, 'guard_id': 'guard'})
        return ledger.events

    def verify(self, ledger):
        return audit.audit(self.rows, [{'guard_id': 'guard'}], [], ledger,
                           self.manifest, self.gpu, self.binaries)

    def test_complete_trace_recomputes_pairs(self):
        report = self.verify(self.trace())
        self.assertFalse(report['budget_boundary'])
        self.assertEqual(report['candidates'][0]['status'], 'screen_complete')
        self.assertEqual(report['candidates'][0]['pair_ratios'], [.8, .8, .8])

    def test_denial_during_validation_leaves_candidate_partial(self):
        report = self.verify(self.trace(maximum=6))
        self.assertTrue(report['budget_boundary'])
        self.assertEqual(report['candidates'][0]['status'], 'partial')

    def test_denial_before_baseline_leaves_candidate_unstarted(self):
        report = self.verify(self.trace(maximum=0))
        self.assertTrue(report['budget_boundary'])
        self.assertEqual(report['candidates'][0]['status'], 'not_started')

    def test_changed_pair_and_stdout_are_rejected(self):
        ledger = self.trace()
        pair = next(r for r in self.rows if r['record_type'] == 'screen_pair')
        pair['ratio'] = .7
        with self.assertRaisesRegex(ValueError, 'ratio mismatch'):
            self.verify(ledger)
        pair['ratio'] = .8
        row = next(r for r in self.rows if r['record_type'] == 'execution')
        row['stdout'] = '{}'
        with self.assertRaisesRegex(ValueError, 'stdout mismatch'):
            self.verify(ledger)

    def test_wrong_environment_and_binary_are_rejected(self):
        ledger = self.trace()
        row = next(r for r in self.rows if r.get('phase') == 'describe')
        old = copy.deepcopy(row)
        row['direct_environment'] = {}
        with self.assertRaisesRegex(ValueError, 'identity, workload or binary'):
            self.verify(ledger)
        row.update(old)
        row['executable_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity, workload or binary'):
            self.verify(ledger)

    def test_denial_identity_and_budget_order_are_rejected(self):
        ledger = self.trace(maximum=6)
        saved = copy.deepcopy(ledger)
        ledger[-1]['identity']['candidate_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'denied work identity'):
            self.verify(ledger)
        ledger = saved
        ledger[1]['event'] = 'work_finish'
        with self.assertRaisesRegex(ValueError, 'budget execution sequence'):
            self.verify(ledger)


if __name__ == '__main__':
    unittest.main()
