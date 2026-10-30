import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import runner


class LedgerTests(unittest.TestCase):
    def test_resume_and_inflight_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            now = [100.0]; path = Path(tmp) / 'ledger'; clock = lambda: now[0]
            ledger = runner.Ledger(path, {'gpu':'a'}, 100, clock=clock, boot='boot')
            self.assertTrue(ledger.start({'work':'a'}, 5))
            with self.assertRaisesRegex(ValueError, 'in-flight'):
                runner.Ledger(path, {'gpu':'a'}, 100, clock=clock, boot='boot')
            ledger.finish('ok'); now[0] += 10
            resumed = runner.Ledger(path, {'gpu':'a'}, 100, clock=clock, boot='boot')
            self.assertFalse(resumed.start({'work':'b'}, 90))
            with self.assertRaises(ValueError): runner.Ledger(path, {'gpu':'other'}, 100, clock=clock, boot='boot')
            with self.assertRaises(ValueError): runner.Ledger(path, {'gpu':'a'}, 100, clock=clock, boot='other')

    def test_only_clean_guard_sessions_resume(self):
        rows = [dict(record_type='candidate_disposition', guard_id='ok', n=128, batch=1, candidate_id='a', status='screen_complete'),
                dict(record_type='candidate_disposition', guard_id='bad', n=128, batch=1, candidate_id='b', status='screen_complete')]
        guards = [dict(event='guard_complete', guard_id='ok', success=True, clean=True),
                  dict(event='guard_abort', guard_id='bad', success=False, clean=False)]
        self.assertEqual(runner.completed_candidates(rows, guards), {(128,1,'a')})

    def test_alternating_randomized_order(self):
        order = [runner.seeded_order(128,16,'x',i) for i in range(3)]
        self.assertNotEqual(order[0], order[1]); self.assertEqual(order[0], order[2])

    def test_generated_provenance_detects_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            base = Path(tmp); (base/'include').mkdir(); (base/'include'/'rejection.hpp').write_text('r',encoding='utf8')
            (base/'include'/'candidate.hpp').write_text('c',encoding='utf8')
            root = Path(__file__).resolve().parents[3]
            source = root/'test'/'plan_selection_study'/'frontier_search'
            doc = {'upstream_commit':'commit', 'expanded':{'include/candidate.hpp':runner.sha(base/'include'/'candidate.hpp')},
                   'frontier_v2':{'generator_sha256':runner.sha(source/'prepare_headers.py'),
                                  'candidate_sha256':runner.sha(source/'candidate.hpp'),
                                  'rejection_instrumentation_sha256':runner.sha(source/'rejection_instrumentation.py'),
                                  'rejection_header_sha256':runner.sha(source/'rejection.hpp'),
                                  'generated_rejection_header_sha256':runner.sha(base/'include'/'rejection.hpp')}}
            p=base/'expanded_provenance.json'; p.write_text(json.dumps(doc),encoding='utf8')
            runner.verify_generated(root,p,'commit')
            (base/'include'/'candidate.hpp').write_text('changed',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'generated source hash mismatch'):
                runner.verify_generated(root,p,'commit')


class ScreeningTests(unittest.TestCase):
    def run_case(self, *, fail_validation=False):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent); self.addCleanup(tmp.cleanup)
        base = Path(tmp.name); request = {'radix':[8,16], 'logical_registers':16, 'grouped_batch':1}
        cid = hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        manifest = base/'manifest.jsonl'
        manifest.write_text('\n'.join(json.dumps(x) for x in [
            {'schema':'frontier-manifest/v1','workload':{'n':128,'batch':16}},
            {'candidate_id':cid, 'request':request, 'selection':'selected', 'selection_index':0,
             'status':'pending_scheduler_and_device_validation'}])+'\n', encoding='utf8')
        args = SimpleNamespace(gpu='GPU-test', manifest=[manifest], output=base/'out', log=base/'guard',
                               original=base/'original', expanded=base/'expanded', timeout=10)
        contract = {'binaries':{'original':'a','expanded':'b'}}
        class Budget:
            def start(self, *a): return True
            def finish(self, *a): pass
        calls = []
        descriptor = {'canonical_descriptor':{}, 'source_sha256':'hash', 'source_bytes':1, 'compiler_input':{}}
        def one(a, workload, candidate, phase, round_no, order):
            calls.append((phase, workload['pattern'], workload['seed'], candidate[0], round_no, order))
            result = {'precompile_kernels':[descriptor], 'raw_interval_ms_samples':[2.0], 'warmup_actual':5,
                      'warmup_wall_ms':1000, 'exec_ms_median':1.0, 'exec_ms_samples':[1.0]}
            return {'record_type':'execution','status':'ok','workload':workload,'result':result}
        with mock.patch.dict(os.environ, {'VKFFT_STUDY_GUARD_UUID':'GPU-test','VKFFT_STUDY_GUARD_ID':'guard'}), \
             mock.patch.object(runner.suite,'run_one',side_effect=one), \
             mock.patch.object(runner.validation,'check',side_effect=lambda row,*a,**kw: row['result']), \
             mock.patch.object(runner.validation,'check_describe',side_effect=lambda row,*a: row['result']), \
             mock.patch.object(runner.validation,'known_rejection',return_value=False), \
             mock.patch.object(runner.validation,'signature',return_value='same'), \
             mock.patch.object(runner.comparison,'compare'), \
             mock.patch.object(runner,'representable',return_value=(request,None)), \
             mock.patch.object(runner.search_controller,'source_signature',return_value='same'), \
             mock.patch.object(runner.search_controller,'numerical_failure',return_value=fail_validation):
            runner.screen(args, Budget(), contract, [])
        return calls, runner.read_log(args.output)

    def test_complete_validation_precedes_timing(self):
        calls, rows = self.run_case()
        validation = [x for x in calls if x[0] == 'validation']
        self.assertEqual({(x[1],x[2]) for x in validation}, {(p,s) for p in runner.PATTERNS for s in runner.SEEDS})
        self.assertEqual(len(validation),15)
        self.assertLess(max(i for i,x in enumerate(calls) if x[0]=='validation'), min(i for i,x in enumerate(calls) if x[0]=='timing'))
        timing = [x for x in calls if x[0]=='timing']; self.assertEqual(len(timing),6)
        self.assertEqual(timing[0][3]=='original', timing[3][3]=='original')
        self.assertEqual([x['status'] for x in rows if x['record_type']=='candidate_disposition'], ['screen_complete'])

    def test_numerical_failure_never_times_candidate(self):
        calls, rows = self.run_case(fail_validation=True)
        self.assertFalse(any(x[0]=='timing' for x in calls))
        self.assertTrue(any(x[0]=='post_rejection' for x in calls))
        self.assertEqual([x['status'] for x in rows if x['record_type']=='candidate_disposition'], ['numerically_invalid'])


if __name__ == '__main__': unittest.main()
