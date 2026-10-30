"""CPU-only C1 completion tests; no real model fit or GPU call."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloud_recovery_report
import old_screen_h4_recovery_activation as activation
import old_screen_learning as old
from cloud_reproduction import build_schedule
from study_common import sha, write_json


HERE = Path(__file__).parent


class AttachmentTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(attachment=HERE/'old_screen_h4_recovery_activation_C1.json',
            attachment_freeze=HERE/'old_screen_h4_recovery_activation_C1_freeze.json',
            admission=HERE/'old_screen_h4_admission.json',freeze=HERE/'old_screen_h4_admission_freeze.json',
            recovery_admission=HERE/'cloud_reproduction_recovery_E1.json',recovery_freeze=HERE/'cloud_reproduction_recovery_E1_freeze.json',
            cloud_root=HERE/'results/cloud_reproduction_20261028_r1',recovery_root=HERE/'results/cloud_reproduction_20261028_recovery1',
            primary_disposition=None,verify_only=False)

    def test_real_attachment_and_all_frozen_source_hashes_match(self):
        attachment = activation.check_attachment(self.args())
        self.assertEqual(attachment['attachment_id'],'old_screen_h4_recovery_activation_C1')
        self.assertEqual(attachment['recovery_id'],'plan_selection_cloud_reproduction_20261028_recovery1')

    def test_changed_root_or_unauthorized_disposition_rejected(self):
        args = self.args()
        args.cloud_root = HERE/'results/resume_20261028_attempt2'
        with self.assertRaises(old.AdmissionError): activation.check_attachment(args)
        args = self.args()
        args.primary_disposition = HERE/'not-an-activation'
        with self.assertRaises(old.AdmissionError): activation.check_attachment(args)

    def test_scoped_patch_calls_original_wrapper_and_restores_check(self):
        args = self.args()
        args.verify_only = False
        original = old.check_primary
        with patch.object(activation,'verify_completion',return_value={'sequence_ended':True,'disposition':'complete'}) as verify,\
             patch.object(old,'run') as run:
            def fake_original(a):
                return old.check_primary({'analysis_id':'plan_selection_old_screen_h4_analysis_20261028_r1'},a.cloud_root)
            run.side_effect = fake_original
            self.assertEqual(activation.run(args)['disposition'],'complete')
            verify.assert_called_once()
        self.assertIs(old.check_primary,original)

    def test_failed_completion_never_reaches_model(self):
        args = self.args()
        with patch.object(activation,'verify_completion',side_effect=old.AdmissionError('incomplete')) as verify,\
             patch.object(old,'run') as run,patch.object(old,'run_learning') as model:
            run.side_effect=lambda a:old.check_primary({'analysis_id':'plan_selection_old_screen_h4_analysis_20261028_r1'},a.cloud_root)
            with self.assertRaisesRegex(old.AdmissionError,'incomplete'): activation.run(args)
            verify.assert_called_once()
            model.assert_not_called()

    def test_original_one_shot_marker_remains_effective_with_adapter(self):
        from unittest.mock import Mock
        args = self.args()
        base_admission = json.loads(args.admission.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=HERE,prefix='recovery_activation_fit_cpu_') as tmp:
            root = Path(tmp)
            args.old_root,args.output,args.design = root/'old',root/'out',HERE/'research_design.json'
            args.audit_only = False
            write_json(args.old_root/'semantics'/'gate.json',{'protocol_valid':True})
            verified = dict(rows=[dict(phase='screen')],global_candidate='default',audit={'screen_records':936})
            model = Mock(return_value={'status':'completed_auxiliary'})
            with patch.object(old,'check_admission',return_value=base_admission),\
                 patch.object(old,'audit_old_screen',return_value=verified),\
                 patch.object(activation,'verify_completion',return_value={'sequence_ended':True,'disposition':'complete'}),\
                 patch.object(old,'run_learning',model),patch.object(old,'validate_result'):
                self.assertEqual(activation.run(args)['status'],'completed_auxiliary')
                with self.assertRaises(old.AdmissionError): activation.run(args)
            self.assertEqual(model.call_count,1)
            self.assertEqual(model.call_args.args[1],verified['rows'])
            self.assertTrue((args.output/'analysis_started.json').exists())


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE,prefix='recovery_activation_cpu_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original,self.recovery = self.root/'original',self.root/'recovery'
        self.original.mkdir(); self.recovery.mkdir()
        self.study_id = 'synthetic-cloud'
        design_path = self.original/'design.json'
        design = json.loads((HERE/'cloud_reproduction_design.json').read_text())
        design['study_id'] = self.study_id
        write_json(design_path,design)
        self.admission = self.root/'e1.json'
        self.freeze = self.root/'e1_freeze.json'
        self.attachment = self.root/'c1.json'
        self.attach_freeze = self.root/'c1_freeze.json'
        for path in (self.admission,self.freeze,self.attachment,self.attach_freeze): write_json(path,{'synthetic':True})
        self.c1 = dict(recovery_id='synthetic-e1',primary_study_id=self.study_id,primary_design_sha256=sha(design_path.read_bytes()),
                       base_admission_sha256='a',recovery_admission_sha256=sha(self.admission.read_bytes()),recovery_freeze_sha256=sha(self.freeze.read_bytes()))
        schedule = build_schedule(design)
        blocks = []
        for index,block in enumerate(schedule):
            hashes = [f"{index:03d}-{j}" for j in range(len(block['candidate_ids']))]
            blocks.append(dict(**block,record_hashes=hashes,source_root='original' if index<34 else 'recovery'))
        write_json(self.recovery/'execution_selection.json',dict(recovery_id='synthetic-e1',blocks=blocks,
                   selected_old_records=99,selected_new_records=100,excluded_orphan_record_hashes=['orphan1','orphan2']))
        write_json(self.recovery/'recovery_manifest.json',{'synthetic':True})
        for stage in ('equivalence','semantics','calibration','health','execution'):
            write_json(self.original/stage/'status.json',dict(study_id=self.study_id,stage=stage,state='stopped' if stage=='execution' else 'complete'))
        for stage,count in (('health',30),('execution',100),('cost',48)):
            write_json(self.recovery/stage/'status.json',dict(study_id=self.study_id,stage=stage,state='complete'))
            write_json(self.recovery/stage/'gate.json',dict(passed=True,record_count=count))
        self.report = dict(study_id=self.study_id,design_sha256=self.c1['primary_design_sha256'],gpu_uuid=design['scope']['device_uuid'],
                           recovery_id='synthetic-e1',stage_errors={},admission_sha256=self.c1['recovery_admission_sha256'],
                           stage_record_counts=dict(health=30,execution=100,cost=48),old_preserved_count=99,
                           selected_record_count=199,selected_block_count=65,cost={'feasibility':'negative'},directional_tests=design['execution']['directional_tests'],
                           execution_selection_sha256=sha((self.recovery/'execution_selection.json').read_bytes()),
                           recovery_manifest_sha256=sha((self.recovery/'recovery_manifest.json').read_bytes()))
        self.args = argparse.Namespace(cloud_root=self.original,recovery_root=self.recovery,recovery_admission=self.admission,
                                      recovery_freeze=self.freeze,attachment=self.attachment,attachment_freeze=self.attach_freeze)

    def verify(self):
        with patch.object(activation,'check_attachment',return_value=self.c1),\
             patch.object(cloud_recovery_report,'analyze_recovery',return_value=self.report):
            return activation.verify_completion(self.args)

    def test_complete_with_scientific_negative_is_accepted_and_statuses_bound(self):
        result = self.verify()
        self.assertEqual(result['disposition'],'complete')
        self.assertTrue(result['sequence_ended'])
        self.assertEqual(len(result['status_file_bindings']),8)
        self.assertEqual(result['missing_primary_evidence'],[])

    def test_running_missing_guard_or_incomplete_cost_rejected(self):
        for change in ('running','errors','cost','count','orphan','direction','device'):
            original_report = copy.deepcopy(self.report)
            path = self.recovery/'health'/'status.json'
            original_status = path.read_bytes()
            selection_path = self.recovery/'execution_selection.json'
            original_selection = selection_path.read_bytes()
            if change=='running': write_json(path,dict(study_id=self.study_id,stage='health',state='running'))
            elif change=='errors': self.report['stage_errors']={'health':['guard_incomplete']}
            elif change=='cost': self.report['cost']=None
            elif change=='count': self.report['stage_record_counts']['execution']=99
            elif change=='direction': self.report['directional_tests'][0]['target']='not-the-frozen-candidate'
            elif change=='device': self.report['gpu_uuid']='GPU-foreign'
            else:
                value = json.loads(original_selection)
                value['excluded_orphan_record_hashes'] = [value['blocks'][0]['record_hashes'][0]]
                write_json(selection_path,value)
                self.report['execution_selection_sha256'] = sha(selection_path.read_bytes())
            with self.subTest(change=change),self.assertRaises(old.AdmissionError): self.verify()
            self.report = original_report
            path.write_bytes(original_status)
            selection_path.write_bytes(original_selection)


if __name__=='__main__':
    unittest.main()
