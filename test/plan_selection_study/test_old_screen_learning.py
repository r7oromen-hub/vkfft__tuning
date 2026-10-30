"""CPU-only tests; no GPU and no model fitting.

Synthetic evidence is not a scientific result. The actual H4 run is separately
gated by the frozen admission and must occur only after the cloud sequence.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from learning_check import ABLATIONS, MODEL, prepare_rows, run_learning
from study_common import block_candidates, candidate_config, canonical, input_seed, load_design, sha, write_json


HERE = Path(__file__).parent


def fake_screen(design):
    rows = []
    for workload in design['workloads']:
        for candidate in design['candidates']:
            rows.append(dict(study_id=design['study_id'],phase='screen',round=0,
                             workload_id=workload['id'],candidate_id=candidate['id'],block_attempt=0,
                             status='ok',guard_clean=True,guard_gpu_uuid=design['scope']['device_uuid'],
                             record_sha256=workload['id']+'/'+candidate['id'],
                             result=dict(exec_ms_median=1.0,precompile_kernels=[dict(axis=0,upload=0,
                                  canonical_descriptor=dict(fftDim=workload['n'],numAxisUploads=1))])))
    return rows


class LearningProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = load_design(HERE/'research_design.json')

    def test_fixed_config_and_four_grouped_folds_are_not_tuned(self):
        self.assertEqual(MODEL,dict(n_estimators=256,max_depth=8,min_samples_leaf=2,
                                   max_features=1.0,random_state=20261027))
        self.assertEqual(ABLATIONS,('workload_knobs','workload_structure','workload_knobs_structure'))
        lengths = sorted({w['n'] for w in self.design['workloads']})
        self.assertEqual(len(lengths),16)
        self.assertEqual([sum(i%4==fold for i in range(len(lengths))) for fold in range(4)],[4,4,4,4])

    def test_unverified_evidence_never_invokes_model(self):
        result = run_learning(self.design, [], evidence_verified=False)
        self.assertEqual(result['status'],'not_run_unverified_evidence')
        self.assertTrue(result['heldout_validity_used_for_candidate_filter'])
        self.assertFalse(result['deployment_end_to_end_evaluated'])

    def test_invalid_and_semantic_excluded_cells_are_not_labels(self):
        rows = fake_screen(self.design)
        excluded = rows[2]
        failed = rows[3]
        excluded['selection_eligible'] = False
        failed['status'] = 'failed'
        records,coverage,errors = prepare_rows(self.design,rows)
        self.assertEqual(len(coverage),36)
        self.assertEqual(errors,[])
        hashes = {r['record_sha256'] for r in records}
        self.assertNotIn(excluded['record_sha256'],hashes)
        self.assertNotIn(failed['record_sha256'],hashes)
        self.assertFalse(any(r['candidate_id']=='default_replica' for r in records))
        self.assertIn(excluded['candidate_id'],coverage[0]['excluded_candidates'])

    def test_missing_request_quarantines_all_batches_for_exact_n(self):
        rows = fake_screen(self.design)
        n = self.design['workloads'][0]['n']
        records,coverage,errors = prepare_rows(self.design,rows[1:])
        self.assertFalse(any(x['n']==n for x in records))
        self.assertFalse(any(x['n']==n for x in coverage))
        self.assertTrue(any('incomplete' in x['reason'] for x in errors))

    def test_old_identity_or_guard_cannot_be_forged_by_matching_candidate_ids(self):
        rows = fake_screen(self.design)
        broken = copy.deepcopy(rows)
        broken[0]['study_id'] = 'cloud-study'
        records,coverage,_ = prepare_rows(self.design,broken)
        bad_n = self.design['workloads'][0]['n']
        self.assertFalse(any(x['n']==bad_n for x in records))
        self.assertFalse(any(x['n']==bad_n for x in coverage))
        broken = copy.deepcopy(rows)
        broken[0]['guard_gpu_uuid'] = 'GPU-new-host'
        records,coverage,_ = prepare_rows(self.design,broken)
        self.assertFalse(any(x['n']==bad_n for x in records))
        self.assertFalse(any(x['n']==bad_n for x in coverage))


class IndependentScreenAuditTests(unittest.TestCase):
    """Exercise evidence checks with fake guard/trace witnesses, without fitting."""
    def setUp(self):
        import old_screen_learning as wrapper
        self.wrapper = wrapper
        self.design = load_design(HERE/'research_design.json')
        self.design['_sha256'] = sha((HERE/'research_design.json').read_bytes())
        self.temp = tempfile.TemporaryDirectory(dir=HERE,prefix='old_learning_audit_cpu_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidates = {x['id']:x for x in self.design['candidates']}
        self.selection = dict(study_id=self.design['study_id'],design_sha256=self.design['_sha256'],
                              global_candidate='default',global_scores={},workloads={})
        write_json(self.root/'screen_selection.json',self.selection)
        write_json(self.root/'manifest.json',dict(study_id=self.design['study_id'],design_sha256=self.design['_sha256'],
                                                   gpu_uuid=self.design['scope']['device_uuid']))
        write_json(self.root/'screen'/'status.json',dict(state='complete'))
        write_json(self.root/'screen'/'gate.json',dict(passed=True,selection_sha256=wrapper.digest(self.root/'screen_selection.json'),
            guard_sessions=[dict(guard_id='guard',guard_log='fake-guard')]))
        write_json(self.root/'semantics'/'gate.json',dict(protocol_valid=True,semantic_invalid_cells=[]))
        repeats = {w['id']:dict(graph_repeats=1) for w in self.design['workloads']}
        write_json(self.root/'graph_repeats.json',repeats)
        self.rows = []
        for w in self.design['workloads']:
            order = block_candidates(self.design,w['id'],'screen',0)
            group = []
            for pos,cid in enumerate(order):
                config = candidate_config(self.candidates[cid])
                names = dict(aim_threads='aimThreads',register_boost='registerBoost',use_lut='useLUT',
                             register_boost_nonpow2='registerBoostNonPow2',coalesced_memory='coalescedMemory',
                             use_lut_4step='useLUT_4step',swap_to_2stage='swapTo2Stage4Step',swap_to_3stage='swapTo3Stage4Step')
                request = {'makeForwardPlanOnly':1}
                request.update({names[k]:v for k,v in config.items()})
                seed = input_seed(self.design,'screen',w['id'],0)
                row = dict(study_id=self.design['study_id'],phase='screen',round=0,workload_id=w['id'],
                    candidate_id=cid,block_attempt=0,position_in_block=pos,config=config,
                    input_seed=seed,guard_id='guard',guard_log='fake-guard',
                    guard_clean=True,guard_gpu_uuid=self.design['scope']['device_uuid'],status='ok',
                    workload=dict(w,graph_repeats=1,samples=self.design['measurement']['screen_samples']),
                    result=dict(exec_ms_median=1.0,requested_config=request,seed=seed,gpu=dict(uuid=self.design['scope']['device_uuid']),
                                precompile_kernels=[dict(axis=0,upload=0,canonical_descriptor=dict(fftDim=w['n'],numAxisUploads=1))]))
                row['record_sha256'] = sha(canonical(row))
                self.rows.append(row)
                group.append(row['record_sha256'])
            write_json(self.root/'screen'/'blocks'/f"screen__{w['id']}__r0.json",dict(candidate_ids=order,record_hashes=group))
        self.rows += [dict(phase='calibration') for _ in range(71)]
        self.patches = [patch.object(wrapper,'read_jsonl',side_effect=lambda p:self.rows),
            patch.object(wrapper,'guard_check',return_value=[]),patch.object(wrapper,'resolve_log',return_value=self.root/'fake'),
            patch.object(wrapper,'completed_session',return_value=True),patch.object(wrapper,'execution_errors',return_value=[]),
            patch.object(wrapper,'block_quality',return_value=dict(ok=True)),
            patch.object(wrapper,'freeze_screen',return_value={k:self.selection[k] for k in ('global_candidate','global_scores','workloads')})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def audit(self):
        return self.wrapper.audit_old_screen(self.root,self.design,{})

    def test_exact_936_rows_are_returned_without_calibration_labels(self):
        result = self.audit()
        self.assertEqual(len(result['rows']),936)
        self.assertTrue(all(r['phase']=='screen' for r in result['rows']))
        self.assertEqual(result['global_candidate'],'default')
        self.assertEqual(result['audit']['calibration_records'],71)

    def test_missing_or_duplicate_candidate_cannot_be_replaced(self):
        self.rows.pop(0)
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'coverage'):
            self.audit()

    def test_seed_or_old_identity_tampering_is_rejected(self):
        for key,value in [('input_seed',-1),('study_id','cloud-study')]:
            original = self.rows[0][key]
            self.rows[0][key] = value
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'identity_or_guard'):
                self.audit()
            self.rows[0][key] = original

    def test_result_request_seed_and_uuid_are_independently_checked(self):
        row = self.rows[0]
        original = copy.deepcopy(row)
        path = self.root/'screen'/'blocks'/f"screen__{row['workload_id']}__r0.json"
        metadata = json.loads(path.read_text(encoding='utf-8'))
        for key,value in [('requested_config',{}),('seed',-1),('gpu',dict(uuid='GPU-cloud'))]:
            row['result'][key] = value
            row.pop('record_sha256')
            row['record_sha256'] = sha(canonical(row))
            metadata['record_hashes'][0] = row['record_sha256']
            write_json(path,metadata)
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'result_request_seed_or_uuid_mismatch'):
                self.audit()
            row.clear()
            row.update(copy.deepcopy(original))
            metadata['record_hashes'][0] = row['record_sha256']
            write_json(path,metadata)

    def test_guard_boundary_and_completed_session_are_required(self):
        with patch.object(self.wrapper,'guard_check',return_value=['missing_before_after_samples']):
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'identity_or_guard'):
                self.audit()
        with patch.object(self.wrapper,'completed_session',return_value=False):
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'guard_session_incomplete'):
                self.audit()

    def test_semantic_exclusion_cannot_be_silently_removed(self):
        row = self.rows[1]
        write_json(self.root/'semantics'/'gate.json',dict(protocol_valid=True,semantic_invalid_cells=[
            dict(workload_id=row['workload_id'],candidate_id=row['candidate_id'])]))
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'semantic_eligibility_mismatch'):
            self.audit()

    def test_failed_negative_control_and_reselected_global_are_rejected(self):
        with patch.object(self.wrapper,'block_quality',return_value=dict(ok=False,errors=['replica_mismatch'])):
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'block_quality_failed'):
                self.audit()
        changed = dict(global_candidate='changed_candidate',global_scores={},workloads={})
        with patch.object(self.wrapper,'freeze_screen',return_value=changed):
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'frozen_selection_mismatch'):
                self.audit()


class AdmissionAndSingleRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import old_screen_learning as wrapper
        cls.wrapper = wrapper
        cls.admission_path = HERE/'old_screen_h4_admission.json'
        cls.freeze_path = HERE/'old_screen_h4_admission_freeze.json'
        cls.admission = json.loads(cls.admission_path.read_text(encoding='utf-8'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE,prefix='old_learning_gate_cpu_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_paths_and_admission_hash_are_required(self):
        spec = self.admission
        actual = self.wrapper.check_admission(self.admission_path,self.freeze_path,
            HERE/spec['source']['root'],HERE/spec['activation']['primary_root'],HERE/spec['output']['root'])
        self.assertEqual(actual['analysis_id'],spec['analysis_id'])
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'frozen_input_or_output_path_mismatch'):
            self.wrapper.check_admission(self.admission_path,self.freeze_path,
                HERE/spec['source']['root'],HERE/spec['activation']['primary_root'],self.root)
        altered = self.root/'changed.json'
        altered.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'admission_freeze_mismatch'):
            self.wrapper.check_admission(altered,self.freeze_path,
                HERE/spec['source']['root'],HERE/spec['activation']['primary_root'],HERE/spec['output']['root'])

    def primary_fixture(self):
        spec = copy.deepcopy(self.admission)
        ident = spec['activation']['primary_study_id']
        write_json(self.root/'design.json',dict(study_id=ident))
        spec['activation']['primary_design_sha256'] = self.wrapper.digest(self.root/'design.json')
        return spec

    def status(self,spec,stage,state):
        write_json(self.root/stage/'status.json',dict(study_id=spec['activation']['primary_study_id'],stage=stage,state=state))

    def test_running_or_missing_primary_stage_never_authorizes_fitting(self):
        spec = self.primary_fixture()
        for stage in spec['activation']['primary_stages']:
            self.status(spec,stage,'complete')
        self.status(spec,'cost','running')
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'still_running'):
            self.wrapper.check_primary(spec,self.root)
        (self.root/'cost'/'status.json').unlink()
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'not_complete'):
            self.wrapper.check_primary(spec,self.root)

    def test_technical_stop_requires_bound_status_and_explicit_disposition(self):
        spec = self.primary_fixture()
        for stage in spec['activation']['primary_stages'][:3]: self.status(spec,stage,'complete')
        self.status(spec,'health','stopped')
        statuses = {s:self.wrapper.digest(self.root/s/'status.json') for s in spec['activation']['primary_stages'] if (self.root/s/'status.json').exists()}
        value = dict(schema_version='test-terminal/v1',study_id=spec['activation']['primary_study_id'],
            design_sha256=spec['activation']['primary_design_sha256'],sequence_ended=True,disposition='technical_stop',
            status_file_bindings=statuses,reason='mock technical stop',missing_primary_evidence=['H1','H2','H3'])
        target = self.root/'disposition.json'
        write_json(target,value)
        self.assertEqual(self.wrapper.check_primary(spec,self.root,target)['disposition'],'technical_stop')
        value['status_file_bindings']['health'] = 'wrong'
        write_json(target,value)
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'technical_stop_not_established'):
            self.wrapper.check_primary(spec,self.root,target)
        self.status(spec,'health','running')
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'still_running'):
            self.wrapper.check_primary(spec,self.root,target)

    def run_fixture(self):
        args = argparse.Namespace(admission=self.admission_path,freeze=self.freeze_path,old_root=self.root/'old',
             cloud_root=self.root/'cloud',output=self.root/'out',design=HERE/'research_design.json',
             primary_disposition=None,audit_only=False)
        write_json(args.old_root/'semantics'/'gate.json',dict(protocol_valid=True))
        verified = dict(rows=[dict(phase='screen')],global_candidate='default',audit=dict(screen_records=936))
        return args,verified

    def run_with(self,args,verified,model,primary=None):
        primary = {'sequence_ended':True} if primary is None else primary
        with (patch.object(self.wrapper,'check_admission',return_value=self.admission),
              patch.object(self.wrapper,'audit_old_screen',return_value=verified),
              patch.object(self.wrapper,'check_primary',return_value=primary),
              patch.object(self.wrapper,'run_learning',model),
              patch.object(self.wrapper,'validate_result',return_value=None)):
            return self.wrapper.run(args)

    def test_primary_gate_blocks_model_and_output(self):
        args,verified = self.run_fixture()
        with (patch.object(self.wrapper,'check_admission',return_value=self.admission),
              patch.object(self.wrapper,'audit_old_screen',return_value=verified),
              patch.object(self.wrapper,'check_primary',side_effect=self.wrapper.AdmissionError('not_complete')),
              patch.object(self.wrapper,'run_learning') as model):
            with self.assertRaisesRegex(self.wrapper.AdmissionError,'not_complete'):
                self.wrapper.run(args)
            model.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_success_calls_original_once_and_reservation_blocks_repeat(self):
        args,verified = self.run_fixture()
        from unittest.mock import Mock
        model = Mock(return_value=dict(status='completed_auxiliary'))
        result = self.run_with(args,verified,model)
        self.assertEqual(result['status'],'completed_auxiliary')
        self.assertEqual(model.call_count,1)
        self.assertEqual(model.call_args.kwargs,dict(evidence_verified=True,frozen_global_candidate='default'))
        self.assertEqual(model.call_args.args[1],verified['rows'])
        with self.assertRaises(self.wrapper.AdmissionError):
            self.run_with(args,verified,model)
        self.assertEqual(model.call_count,1)

    def test_fit_exception_is_recorded_and_never_retried(self):
        args,verified = self.run_fixture()
        from unittest.mock import Mock
        model = Mock(side_effect=RuntimeError('synthetic fit interruption'))
        with self.assertRaisesRegex(RuntimeError,'synthetic fit interruption'):
            self.run_with(args,verified,model)
        self.assertTrue((args.output/'analysis_failed.json').exists())
        with self.assertRaises(self.wrapper.AdmissionError):
            self.run_with(args,verified,model)
        self.assertEqual(model.call_count,1)

    def test_audit_only_never_creates_reservation_or_calls_model(self):
        args,verified = self.run_fixture()
        args.audit_only = True
        from unittest.mock import Mock
        model = Mock()
        result = self.run_with(args,verified,model)
        self.assertEqual(result['status'],'audit_passed_no_fitting')
        self.assertFalse(args.output.exists())
        model.assert_not_called()

    def test_existing_result_provenance_or_failure_blocks_even_without_marker(self):
        from unittest.mock import Mock
        for filename in (self.admission['output']['learning_result'],self.admission['output']['provenance'],'analysis_failed.json'):
            with self.subTest(filename=filename):
                args,verified = self.run_fixture()
                args.output = self.root/('out-'+filename.replace('.','-'))
                args.output.mkdir()
                (args.output/filename).write_text('{}',encoding='utf-8')
                model = Mock()
                with self.assertRaises(self.wrapper.AdmissionError):
                    self.run_with(args,verified,model)
                model.assert_not_called()

    def test_result_validation_rejects_changed_fold_and_duplicate_workload(self):
        spec = self.admission
        fold_by_n = {int(k):v for k,v in spec['analysis']['fold_assignment_by_exact_N'].items()}
        coverage = spec['source']['eligible_candidate_sets']
        folds = [dict(fold=k,test_N=sorted(n for n,v in fold_by_n.items() if v==k),
                      train_N=sorted(n for n,v in fold_by_n.items() if v!=k)) for k in range(4)]
        predictions = [dict(workload_id=x['workload_id'],n=x['n'],batch=x['batch'],fold=fold_by_n[x['n']],
                            selected_candidate='default') for x in coverage]
        outcomes = [dict(target=t,ablation=a,fold_details=copy.deepcopy(folds),workloads=copy.deepcopy(predictions))
                    for t in (spec['analysis']['primary_target'],spec['analysis']['secondary_target']) for a in spec['analysis']['ablations']]
        baseline_rows = [dict(workload_id=x['workload_id'],fold=fold_by_n[x['n']],selected_candidate='default',available=True) for x in coverage]
        baselines = {name:dict(workloads=copy.deepcopy(baseline_rows)) for name in
                     ('default','training_only_global','frozen_S1_global_diagnostic','screen_oracle_diagnostic')}
        result = dict(status='completed_auxiliary',study_id=spec['source']['study_id'],model=spec['analysis']['model'],
                      distinct_N=16,source_record_hashes=spec['source']['model_source_record_hashes'],coverage=coverage,
                      excluded_workloads=[],outcomes=outcomes,baselines=baselines,folds=spec['analysis']['fold_assignment_by_exact_N'])
        self.wrapper.validate_result(result,spec)
        changed = copy.deepcopy(result)
        changed['outcomes'][0]['fold_details'][0]['test_N'][0] = 123
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'fold_assignment_mismatch'):
            self.wrapper.validate_result(changed,spec)
        changed = copy.deepcopy(result)
        changed['outcomes'][0]['workloads'][1] = copy.deepcopy(changed['outcomes'][0]['workloads'][0])
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'workload_coverage_mismatch'):
            self.wrapper.validate_result(changed,spec)
        changed = copy.deepcopy(result)
        changed['outcomes'][0]['workloads'][0]['fold'] = (changed['outcomes'][0]['workloads'][0]['fold']+1)%4
        with self.assertRaisesRegex(self.wrapper.AdmissionError,'prediction_identity_mismatch'):
            self.wrapper.validate_result(changed,spec)


if __name__ == '__main__':
    unittest.main()
