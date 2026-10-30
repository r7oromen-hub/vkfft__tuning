"""GPU-free regression checks of missing evidence, fixed families and inference."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_study import h1_analysis, headline_analysis, h3_analysis
from report_helpers import headline_stat, holm, paired, read_evidence, selection_eligible
from study_common import canonical, sha


class StatisticalTests(unittest.TestCase):
    def test_zero_variance_is_noninferential_and_stays_in_holm_family(self):
        zero = headline_stat([1.1]*10)
        self.assertFalse(zero['inferential'])
        self.assertIsNone(zero['interval95'])
        self.assertNotIn('t_statistic',zero)
        tests = [dict(zero,quality_valid=True),dict(p=.02,speedup=1.08,inferential=True,quality_valid=True)]
        holm(tests)
        self.assertEqual(tests[1]['holm_p'],.04)
        self.assertEqual(tests[0]['holm_p'],1)
        self.assertFalse(tests[0]['clear_gain'])

    def test_test_uses_two_percent_null_not_zero_effect(self):
        ratios = [math.exp(math.log(1.02)+v) for v in [-.005,-.004,-.003,-.002,-.001,.001,.002,.003,.004,.005]]
        result = headline_stat(ratios)
        self.assertAlmostEqual(result['p'],.5,places=10)
        self.assertEqual(result['df'],9)
        self.assertTrue(result['interval95'][0] < 1.02 < result['interval95'][1])

    def test_missing_round_does_not_shrink_replication_count(self):
        self.assertEqual(headline_stat([1.2]*9)['p'],1)
        rows = [dict(round=k,candidate_id=c,status='ok',input_seed=k,record_sha256=str(k)+c,
                     result=dict(exec_ms_median=2 if c == 'default' else 1)) for k in range(2) for c in ['default','winner']]
        self.assertIsNone(paired(rows,'default','winner',3))
        rows += [copy.deepcopy(rows[0])]
        self.assertIsNone(paired(rows,'default','winner',2))

    def test_later_success_cannot_override_preflight_ineligible_cell(self):
        rows = [dict(round=k,candidate_id=c,status='ok',input_seed=k,record_sha256=str(k)+c,
                     selection_eligible=c != 'winner',result=dict(exec_ms_median=1.)) for k in range(3) for c in ['default','winner']]
        self.assertFalse(selection_eligible(rows[1]))
        self.assertIsNone(paired(rows,'default','winner',3))

    def test_holm_monotonic_and_quality_gate(self):
        tests = [dict(p=p,speedup=1.2,inferential=True,quality_valid=quality) for p,quality in [(.04,True),(.001,False),(.01,True)]]
        holm(tests)
        self.assertEqual([x['holm_p'] for x in tests],[.04,.003,.02])
        self.assertFalse(tests[1]['clear_gain'])

    def test_missing_frozen_direction_is_retained(self):
        manifest = dict(workloads=[dict(workload_id='n1_b1')],directional_tests=[dict(workload_id='n1_b1',reference='default',target='winner')])
        result = headline_analysis([],manifest,dict(passed=True))
        self.assertEqual(result['family_size'],1)
        self.assertEqual(result['tests'][0]['p'],1)
        self.assertFalse(result['tests'][0]['clear_gain'])
        self.assertEqual(result['status'],'incomplete')

    def test_missing_data_does_not_become_a_null_effect(self):
        design = dict(workloads=[dict(id='n1_b1',n=1,batch=1)],cost_profile=dict(workloads=['n1_b1']))
        result = h1_analysis(design,dict(screen=[],confirmation=[]),None)
        self.assertEqual(result['confirmed_workloads'],0)
        self.assertIsNone(result['equal_workload_geometric_speedup'])
        self.assertNotIn('speedup',result['effects'][0])
        cost = h3_analysis(design,[],None,None)
        self.assertEqual(cost['passed_pairs'],0)
        self.assertEqual(cost['status'],'unresolved_or_incomplete')

    def test_headline_does_not_apply_health_two_percent_gate(self):
        rows = [dict(round=k,candidate_id=c,status='ok',input_seed=k,record_sha256=str(k)+c,
                     result=dict(exec_ms_median=1.03 if c == 'default' else 1.)) for k in range(10) for c in ['default','default_replica']]
        manifest = dict(workloads=[dict(workload_id='n1_b1',candidates=['default','default_replica'])],directional_tests=[])
        for r in rows:
            r['workload_id'] = 'n1_b1'
        result = headline_analysis(rows,manifest,dict(passed=True))
        self.assertTrue(result['negative_controls'][0]['valid'])
        self.assertEqual(result['status'],'complete')
        manifest['workloads'][0]['candidates'].append('missing_candidate')
        self.assertEqual(headline_analysis(rows,manifest,dict(passed=True))['status'],'incomplete')

    def test_crossover_uses_frozen_exact_directions_not_reason(self):
        rows = []
        for wid,reverse in [('n8_b1',False),('n8_b2',True)]:
            for k in range(10):
                base = 1.+k*.0001
                times = dict(default=base,default_replica=base,winner=base/(1.1+k*.0001))
                if reverse:
                    times['winner'] = base*(1.1+k*.0001)
                for cid,value in times.items():
                    rows.append(dict(workload_id=wid,round=k,candidate_id=cid,status='ok',input_seed=k,
                                     record_sha256=wid+str(k)+cid,result=dict(exec_ms_median=value)))
        directions = [dict(workload_id='n8_b1',reference='default',target='winner',reason='frozen_winner_effect'),
                      dict(workload_id='n8_b2',reference='winner',target='default',reason='exact_ID_crossover_reverse')]
        manifest = dict(workloads=[dict(workload_id=w,candidates=['default','default_replica','winner']) for w in ['n8_b1','n8_b2']],
                        directional_tests=directions,crossovers=[dict(directions=directions)])
        result = headline_analysis(rows,manifest,dict(passed=True),dict(workloads=[]))
        self.assertTrue(result['confirmed_crossover'])
        self.assertEqual(result['tests'][0]['selection_reason'],'frozen_winner_effect')


class EvidenceTests(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value)+'\n',encoding='utf-8')

    def fixture(self,root,cids=('default','winner','default_replica')):
        design = dict(study_id='new-study',_sha256='design-hash',scope=dict(device_uuid='gpu'),candidates=[dict(id=c) for c in cids])
        self.write(root/'manifest.json',dict(study_id='new-study',design_sha256='design-hash'))
        rows = []
        for c in cids:
            row = dict(study_id='new-study',phase='screen',workload_id='n1_b1',round=0,block_attempt=0,candidate_id=c,status='ok')
            row['record_sha256'] = sha(canonical(row))
            rows.append(row)
        path = root/'screen'/'measurements.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        block = dict(quality=dict(ok=True),candidate_ids=list(cids),record_hashes=[r['record_sha256'] for r in rows],block_attempt=0)
        self.write(root/'screen'/'blocks'/'one.json',block)
        return design,rows,block

    def cost_fixture(self,root):
        design,_,_ = self.fixture(root)
        rows = []
        for cid,attempt in [('describe-select-cost',0),('describe-select-cost',1),('full-search-cost',1)]:
            row = dict(study_id=design['study_id'],phase='cost',workload_id='n1_b1',round=1,
                       candidate_id=cid,block_attempt=attempt,status='ok')
            row['record_sha256'] = sha(canonical(row))
            rows.append(row)
        base = root/'cost';base.mkdir()
        (base/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        selected = dict(pairs=[dict(record_hashes=[r['record_sha256'] for r in rows[1:]])],physical_processes_estimate=4)
        admission = dict(engineering_only_record_hashes=[rows[0]['record_sha256']],preserved_record_hashes=[],quarantined_guard_run_id='quarantined')
        self.write(base/'accepted_pairs.json',selected)
        self.write(root/'cost_recovery_admission.json',admission)
        return design,rows,selected,admission

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_metadata_cannot_hide_missing_candidate(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,rows,block = self.fixture(root)
            block['record_hashes'].pop()
            block['candidate_ids'].pop()
            self.write(root/'screen'/'blocks'/'one.json',block)
            e = read_evidence(root,d)
            self.assertEqual(e['rows']['screen'],[])
            self.assertTrue(any(x.get('error') == 'incomplete_or_mismatched_frozen_block' for x in e['errors']))

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=False,errors=['drift']))
    def test_quality_is_recomputed_not_trusted(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,_,_ = self.fixture(root)
            e = read_evidence(root,d)
            self.assertEqual(e['rows']['screen'],[])
            self.assertTrue(any(x.get('error') == 'recomputed_block_quality_failed' for x in e['errors']))

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_tampered_record_and_foreign_study_are_not_used(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,rows,_ = self.fixture(root)
            rows[0]['status'] = 'changed_after_hash'
            (root/'screen'/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(read_evidence(root,d)['rows']['screen'],[])
            self.write(root/'manifest.json',dict(study_id='pilot',design_sha256='design-hash'))
            self.assertFalse(read_evidence(root,d)['identity_ok'])

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_clean_partial_block_survives_missing_final_stage_gate(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,rows,_ = self.fixture(root)
            e = read_evidence(root,d)
            self.assertEqual(len(e['rows']['screen']),len(rows))
            self.assertFalse(e['coverage']['screen']['gate_passed'])

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_complete_attempt_matrix_can_contain_invalid_candidates(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,rows,block = self.fixture(root)
            rows[1]['status'] = 'execution_invalid'
            rows[1]['selection_eligible'] = False
            rows[1]['record_sha256'] = sha(canonical({k:v for k,v in rows[1].items() if k != 'record_sha256'}))
            (root/'screen'/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            block['record_hashes'] = [r['record_sha256'] for r in rows]
            self.write(root/'screen'/'blocks'/'one.json',block)
            self.write(root/'screen'/'gate.json',dict(passed=True,protocol_valid=True,all_candidates_passed=False))
            self.write(root/'screen'/'status.json',dict(state='completed_with_candidate_invalid',expected_processes=3))
            e = read_evidence(root,d)
            c = e['coverage']['screen']
            self.assertEqual(c['unique_attempted_cells'],3)
            self.assertEqual(c['accepted_records'],3)
            self.assertEqual(c['eligible_records'],2)
            self.assertTrue(c['protocol_valid'])
            self.assertFalse(c['all_candidates_passed'])

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_cost_uses_explicit_whole_pair_and_keeps_successful_orphan_engineering_only(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,raw,selection,admission = self.cost_fixture(root)
            with patch('cost_recovery.selected_cost_rows',return_value=(raw[1:],selection,admission)) as choose:
                e = read_evidence(root,d)
            choose.assert_called_once_with(root,require_complete=False,require_sessions=True)
            self.assertEqual(e['rows']['cost'],raw[1:])
            c = e['coverage']['cost']
            self.assertEqual((c['raw_records'],c['accepted_records'],c['accepted_pairs']),(3,2,1))
            self.assertEqual(c['physical_processes_estimate'],4)
            old = next(r for r in e['ledger'] if r['record_sha256'] == raw[0]['record_sha256'])
            self.assertFalse(old['included_in_estimators'])
            self.assertEqual(old['engineering_only_reason'],'A6_original_unpaired_orphan')

    @patch('report_helpers.guard_check',return_value=[])
    @patch('report_helpers.block_quality',return_value=dict(ok=True))
    def test_invalid_cost_selection_has_no_success_or_latest_attempt_fallback(self,*mocks):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            d,raw,_,_ = self.cost_fixture(root)
            with patch('cost_recovery.selected_cost_rows',side_effect=ValueError('A6 source hash changed')):
                e = read_evidence(root,d)
            self.assertEqual(e['rows']['cost'],[])
            self.assertEqual(e['coverage']['cost']['accepted_pairs'],0)
            self.assertEqual(e['coverage']['cost']['raw_records'],3)
            self.assertTrue(any(x.get('error') == 'A6 source hash changed' for x in e['errors']))
            (root/'cost/accepted_pairs.json').unlink()
            with patch('cost_recovery.selected_cost_rows') as choose:
                missing = read_evidence(root,d)
            choose.assert_not_called()
            self.assertEqual(missing['rows']['cost'],[])
            self.assertEqual(missing['coverage']['cost']['selection_state'],'missing_explicit_accepted_pairs')


if __name__ == '__main__':
    unittest.main()

