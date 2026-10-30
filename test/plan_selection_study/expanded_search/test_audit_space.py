import copy
from collections import Counter
import hashlib
import json
import unittest

import audit_space
import enumerate_candidates
import selection


class ManifestAuditTest(unittest.TestCase):
    def fixture(self):
        identity = {'radix': [4, 4], 'logical_registers': 4, 'grouped_batch': 1}
        rows, _ = selection.select(list(enumerate_candidates.candidates(16, 4)), identity, seed=20261030, limit=12)
        return rows, identity

    def test_full_space_and_deterministic_selection(self):
        rows, identity = self.fixture()
        result = audit_space.audit(rows, 16, 4, identity, limit=12)
        self.assertEqual(result['status'], 'pass', result['errors'])
        self.assertEqual(result['counts']['selected'], 12)
        self.assertEqual(result['counts']['identity_control'], 1)
        self.assertEqual(result['counts']['mathematical_requests'], 8 * 6 * 3)

    def test_missing_or_reordered_records_fail(self):
        rows, identity = self.fixture()
        self.assertEqual(audit_space.audit(rows[:-1], 16, 4, identity, limit=12)['status'], 'fail')
        rows[0], rows[1] = rows[1], rows[0]
        self.assertEqual(audit_space.audit(rows, 16, 4, identity, limit=12)['status'], 'fail')

    def test_tampered_hash_or_sampling_fails(self):
        rows, identity = self.fixture()
        bad = copy.deepcopy(rows)
        bad[0]['candidate_id'] = '0' * 64
        self.assertEqual(audit_space.audit(bad, 16, 4, identity, limit=12)['status'], 'fail')
        selected = [x for x in rows if x['selection'] == 'selected']
        selected[0]['selection_index'], selected[1]['selection_index'] = selected[1]['selection_index'], selected[0]['selection_index']
        self.assertEqual(audit_space.audit(rows, 16, 4, identity, limit=12)['status'], 'fail')

    def test_rejected_or_identity_cannot_be_selected(self):
        rows, identity = self.fixture()
        next(x for x in rows if x['selection'] == 'rejected_static')['selection'] = 'selected'
        result = audit_space.audit(rows, 16, 4, identity, limit=12)
        self.assertEqual(result['status'], 'fail')

    def test_manifest_envelope_and_digest(self):
        rows, identity = self.fixture()
        for row in rows:
            row['record_type'] = 'candidate_manifest'
        digest = hashlib.sha256()
        for row in rows:
            digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode())
        header = {'record_type': 'manifest_protocol', 'n': 16, 'batch': 4, 'identity_request': identity,
                  'selection_seed': 20261030, 'limit': 12}
        footer = {'record_type': 'manifest_complete', 'total': len(rows),
                  'selected': 12, 'counts': dict(Counter(x['selection'] for x in rows)),
                  'candidate_lines_sha256': digest.hexdigest()}
        doc = [header, *rows, footer]
        self.assertEqual(audit_space.audit_document(doc, 16, 4, identity, limit=12)['status'], 'pass')
        self.assertEqual(audit_space.audit_document(doc[:-1], 16, 4, identity, limit=12)['status'], 'fail')
        doc[1]['scope'] = 'tampered'
        self.assertEqual(audit_space.audit_document(doc, 16, 4, identity, limit=12)['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
