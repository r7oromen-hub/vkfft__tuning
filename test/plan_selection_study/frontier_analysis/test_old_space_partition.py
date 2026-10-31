import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import analyze
import old_space_partition as subject
import test_analyze


def fixture():
    coverage, rows, digest = test_analyze.fixture()
    table = [16 if i in (2, 4, 8, 16) else 0 for i in range(33)]
    baseline = {'n': 128, 'batch': 16,
                'structural_features': {'numAxisUploads': 1, 'kernels': [
                    {'stageRadix': [8, 16], 'localSize': [8, 1, 1], 'groupedBatch': 1}]},
                'precompile_kernels': [{'raw_descriptor': {
                    'registerBoost': 1, 'fftDim': 128, 'complexSize': 8, 'axisSwapped': 0,
                    'registers_per_radix': table, 'min_logical_registers_per_thread': 16,
                    'logical_registers_per_thread': 16}}],
                'resolved_config': {'sharedMemorySize': 49152, 'maxThreadsNum': 1024}}
    rows.insert(1, {'record_type': 'execution', 'phase': 'identity', 'candidate_id': 'expanded_default',
                    'workload': {'n': 128, 'batch': 16}, 'result': baseline,
                    'contract': rows[0]['contract']})
    requests = {c['candidate_id']: c['request'] for c in coverage['cases']}
    for row in rows:
        if row.get('record_type') != 'execution' or row.get('phase') != 'timing':
            continue
        req = requests.get(row['candidate_id'])
        if req is None:
            continue
        group, reg = req['grouped_batch'], req['logical_registers']
        layout = [group, 128 // reg, 1] if group > 1 else [128 // reg, 1, 1]
        if reg == 128:
            layout = None
        row['result']['structural_features']['kernels'] = [{'localSize': layout}]
    return coverage, rows, digest


class OldSpacePartitionTests(unittest.TestCase):
    def test_partition_identifies_endpoint_register_and_missing_evidence(self):
        coverage, rows, digest = fixture()
        report = subject.partition(coverage, rows, digest)
        first = next(w for w in report['workloads'] if (w['n'], w['batch']) == (128, 16))
        groups = first['groups']
        self.assertEqual([groups[name]['count'] for name in subject.GROUPS], [2, 2, 1])
        self.assertEqual(groups['old_structural_subset']['identity_count'], 1)
        self.assertEqual(groups['old_structural_subset']['best_screened_ratio'], .9)
        self.assertEqual(groups['old_structural_subset']['best_nonidentity_screened_ratio'], 1.03)
        reasons = {x['old_space']['reason'] for x in groups['outside_old_space']['candidates']}
        self.assertEqual(reasons, {'fixed_endpoints_or_two_stage_rule', 'register_table_or_envelope_changed'})
        self.assertEqual(groups['not_comparable']['candidates'][0]['old_space']['reason'],
                         'missing_or_invalid_actual_layout')
        second = next(w for w in report['workloads'] if (w['n'], w['batch']) == (256, 1))
        self.assertIsNone(second['groups']['old_structural_subset']['best_screened_ratio'])
        self.assertEqual(second['groups']['not_comparable']['candidates'][0]['old_space']['reason'],
                         'missing_expanded_default_baseline')

    def test_changed_layout_is_not_old_structural(self):
        coverage, rows, digest = fixture()
        target = next(c for c in coverage['cases'] if c['request'] ==
                      {'radix': [8, 16], 'logical_registers': 16, 'grouped_batch': 2})
        for row in rows:
            if row.get('phase') == 'timing' and row.get('candidate_id') == target['candidate_id']:
                row['result']['structural_features']['kernels'][0]['localSize'] = [8, 2, 1]
        groups = subject.partition(coverage, rows, digest)['workloads'][0]['groups']
        self.assertIn('actual_block_differs_from_old_request',
                      {x['old_space']['reason'] for x in groups['outside_old_space']['candidates']})

    def test_binding_and_contract_mismatch_fail(self):
        coverage, rows, digest = fixture()
        with self.assertRaises(ValueError):
            subject.partition(coverage, rows, 'f' * 64)
        wrong = copy.deepcopy(coverage); wrong['gpu'] = 'GPU-other'
        with self.assertRaises(ValueError):
            subject.partition(wrong, rows, digest)
        wrong = copy.deepcopy(coverage); wrong['cases'][0]['request']['grouped_batch'] = 4
        with self.assertRaises(ValueError):
            subject.partition(wrong, rows, digest)

    def test_cli_binds_files_and_fails_after_raw_change(self):
        coverage, rows, _ = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); raw = root / 'rows.jsonl'; audit = root / 'coverage.json'
            output = root / 'partition.json'; bad_output = root / 'bad.json'
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            coverage['evidence_sha256'] = {'/remote/rows.jsonl': analyze.sha(raw)}
            audit.write_text(json.dumps(coverage), encoding='utf8')
            cmd = [sys.executable, str(Path(subject.__file__)), '--coverage', str(audit),
                   '--rows', str(raw), '--output']
            run = subprocess.run(cmd + [str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(report['input_sha256']['rows'], analyze.sha(raw))
            self.assertEqual(report['input_sha256']['coverage'], analyze.sha(audit))
            with raw.open('a', encoding='utf8') as out:
                out.write('\n')
            run = subprocess.run(cmd + [str(bad_output)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse(bad_output.exists())

    def test_source_change_aborts_before_output(self):
        coverage, rows, _ = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); raw = root / 'rows.jsonl'; audit = root / 'coverage.json'; out = root / 'out.json'
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            coverage['evidence_sha256'] = {'/remote/rows.jsonl': analyze.sha(raw)}
            audit.write_text(json.dumps(coverage), encoding='utf8')
            argv = ['old_space_partition.py', '--coverage', str(audit), '--rows', str(raw), '--output', str(out)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(subject, 'source_hashes',
                    side_effect=[{'source': 'a'}, {'source': 'b'}]):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    subject.main()
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
