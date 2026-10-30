import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.default = {'radix': [4, 2, 4], 'logical_registers': 4, 'grouped_batch': 2}
        self.winner = {'radix': [2, 4, 4], 'logical_registers': 8, 'grouped_batch': 4}

    def test_complete_enumeration_and_rejection_accounting(self):
        header, rows = manifest.build(32, 4, None, [], 10000)
        original = list(manifest.prototype.candidates(32, 4))
        old_subset = [r for r in rows if r['request']['logical_registers'] <= 64]
        self.assertEqual({r['candidate_id'] for r in old_subset}, {r['candidate_id'] for r in original})
        self.assertEqual(header['enumerated_count'], len(original) * 8 // 6)
        self.assertEqual(header['static_rejected_count'], sum(r['status'] == 'rejected_static' for r in rows))
        self.assertEqual(sum(header['static_rejection_reasons'].values()), header['static_rejected_count'])
        self.assertEqual(header['selected_count'], header['pending_count'])
        self.assertTrue(all(r['selection'] == 'rejected_static' for r in rows if r['reason']))
        self.assertEqual(header['anchors'][0]['status'], 'not_provided')

    def test_anchors_neighbors_and_reproducible_selection(self):
        first, rows = manifest.build(32, 4, self.default, [self.winner], 20, 77)
        second, rows2 = manifest.build(32, 4, self.default, [self.winner], 20, 77)
        self.assertEqual((first, rows), (second, rows2))
        selected = {r['candidate_id']: r for r in rows if r['selection'] == 'selected'}
        self.assertIn(manifest.candidate_id(self.default), selected)
        self.assertIn(manifest.candidate_id(self.winner), selected)
        by_id = {r['candidate_id']: r for r in rows}
        neighbors = manifest.neighbor_ids(self.default, by_id)
        self.assertTrue(neighbors)
        for cid in neighbors:
            if not by_id[cid]['reason']:
                self.assertIn(cid, selected)
                self.assertIn('identity_neighbor', selected[cid]['selection_reasons'])
        self.assertEqual(sorted(r['selection_index'] for r in selected.values()), list(range(20)))

    def test_unsupported_anchors_and_limit_are_explicit(self):
        outside = {'radix': [3, 4], 'logical_registers': 4, 'grouped_batch': 1}
        rejected = {'radix': [32], 'logical_registers': 2, 'grouped_batch': 1}
        header, _ = manifest.build(32, 4, outside, [rejected], 5)
        self.assertEqual(header['anchors'][0]['status'], 'outside_prototype_domain')
        self.assertEqual(header['anchors'][1]['status'], 'radix_exceeds_registers')
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            manifest.build(32, 4, self.default, [self.winner], 1)

    def test_wide_register_requests_and_rejections(self):
        rows = list(manifest.candidates(128, 1))
        wide = {'radix': [8, 16], 'logical_registers': 128, 'grouped_batch': 1}
        row = next(r for r in rows if r['request'] == wide)
        self.assertEqual(row['planned_fft_threads'], 1)
        self.assertEqual(row['status'], 'pending_scheduler_and_device_validation')
        self.assertEqual(next(r for r in rows if r['request'] == dict(wide, logical_registers=256))['reason'],
                         'registers_exceed_fft_length')

    def test_cli_jsonl_is_exclusive_and_self_counted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            spec = Path(tmp) / 'spec.json'
            output = Path(tmp) / 'manifest.jsonl'
            spec.write_text(json.dumps(dict(n=32, batch=4, identity=self.default,
                                            historical=[self.winner], limit=20, seed=77)))
            cmd = [sys.executable, str(Path(manifest.__file__)), '--spec', str(spec), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            records = [json.loads(line) for line in output.read_text(encoding='utf8').splitlines()]
            self.assertEqual(records[0]['enumerated_count'], len(records) - 1)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
