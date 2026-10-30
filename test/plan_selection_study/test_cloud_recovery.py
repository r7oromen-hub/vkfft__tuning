"""CPU-only tests for explicit recovery after the captured monitor timeout.

These tests inspect retained evidence and use mocks for all recovery processes.
They do not launch a harness, query a GPU, or perform statistical selection.
"""
import collections
import copy
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cloud_reproduction as cloud
import cloud_recovery as recovery
import cloud_recovery_report as recovered_report
from cloud_reproduction import build_schedule, validate_design
from study_common import append_json, block_quality, canonical, input_seed, read_jsonl, sha, write_json
from test_cloud_reproduction import fake_execution


HERE = Path(__file__).parent
ROOT = HERE/'results'/'cloud_reproduction_20261028_r1'


def incident_admission():
    design = json.loads((HERE/'cloud_reproduction_design.json').read_text(encoding='utf-8'))
    schedule = build_schedule(design)
    rows = read_jsonl(ROOT/'execution'/'measurements.jsonl')
    events = read_jsonl(ROOT/'guards'/'execution.jsonl')
    blocks = []
    for b in schedule[:34]:
        path = ROOT/'execution'/'blocks'/f"execution__{b['workload_id']}__r{b['round']}.json"
        meta = json.loads(path.read_text(encoding='utf-8'))
        blocks.append(dict(**b,block_sha256=sha(path.read_bytes()),block_path=str(path.relative_to(HERE)),record_hashes=meta['record_hashes']))
    quarantine = next(e for e in events if e['event']=='quarantined_run')
    sample_index,sample = next((i,e) for i,e in enumerate(events) if e['event']=='sample' and e.get('ok') is not True)
    new_blocks = [dict(**b,block_attempt=1 if (b['workload_id'],b['round'])==('n16384_b2048',3) else 0) for b in schedule[34:]]
    return dict(execution=dict(new_blocks=new_blocks),source=dict(preserved_blocks=blocks,guard_id=quarantine['guard_id'],
                            root='results/cloud_reproduction_20261028_r1',
                            record_hashes=[h for b in blocks for h in b['record_hashes']],
                            manifest_sha256=sha((ROOT/'manifest.json').read_bytes()),
                            quarantined_guard_run_id=quarantine['active_run']['guard_run_id'],
                            incident_sample_event_index=sample_index,incident_sample_canonical_sha256=sha(canonical(sample)),
                            quarantined_row_canonical_sha256=sha(canonical(quarantine['quarantined_row'])),
                            quarantined_returncode=quarantine['quarantined_row']['returncode'],
                            orphan_record_hashes=[r['record_sha256'] for r in rows[-2:]]))


class CapturedIncidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((HERE/'cloud_reproduction_design.json').read_text(encoding='utf-8'))
        cls.schedule = build_schedule(cls.design)
        cls.rows = read_jsonl(ROOT/'execution'/'measurements.jsonl')
        cls.events = read_jsonl(ROOT/'guards'/'execution.jsonl')
        cls.blocks = {p.name:json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'execution'/'blocks').glob('*.json')}

    def test_original_closed_prefix_is_34_blocks_and_99_rows(self):
        self.assertTrue(validate_design(self.design))
        self.assertEqual(len(self.schedule),65)
        self.assertEqual(sum(len(b['candidate_ids']) for b in self.schedule),199)
        self.assertEqual(len(self.rows),101)
        self.assertEqual(len(self.blocks),34)
        byhash = {r['record_sha256']:r for r in self.rows}
        self.assertEqual(len(byhash),101)
        chosen = []
        for b in self.schedule[:34]:
            name = f"execution__{b['workload_id']}__r{b['round']}.json"
            self.assertIn(name,self.blocks)
            meta = self.blocks[name]
            group = [byhash[h] for h in meta['record_hashes']]
            self.assertEqual(meta['candidate_ids'],b['candidate_ids'])
            self.assertEqual([r['candidate_id'] for r in group],b['candidate_ids'])
            self.assertTrue(block_quality(group)['ok'])
            self.assertEqual(len({r['input_seed'] for r in group}),1)
            for r in group:
                self.assertEqual(r['phase'],'execution')
                self.assertEqual(r['workload_id'],b['workload_id'])
                self.assertEqual(r['round'],b['round'])
                self.assertEqual(r['study_id'],self.design['study_id'])
                self.assertEqual(r['guard_gpu_uuid'],self.design['scope']['device_uuid'])
                self.assertEqual(r['input_seed'],input_seed(self.design,'execution',b['workload_id'],b['round']))
                self.assertEqual(r['record_sha256'],sha(canonical({k:v for k,v in r.items() if k!='record_sha256'})))
            chosen.extend(group)
        self.assertEqual(len(chosen),99)
        self.assertEqual(len({r['record_sha256'] for r in chosen}),99)
        self.assertEqual({r['record_sha256'] for r in chosen},{r['record_sha256'] for r in self.rows[:99]})

    def test_incomplete_block_is_two_orphans_and_one_quarantine(self):
        partial = self.schedule[34]
        self.assertEqual((partial['workload_id'],partial['round']),('n16384_b2048',3))
        self.assertEqual(partial['candidate_ids'],['default_replica','a128_r4_l1_np1','a128_r2_l1_np1','default'])
        self.assertEqual([(r['workload_id'],r['round'],r['candidate_id']) for r in self.rows[99:]],
                         [(partial['workload_id'],3,x) for x in partial['candidate_ids'][:2]])
        self.assertNotIn('execution__n16384_b2048__r3.json',self.blocks)
        quarantine = [e for e in self.events if e['event']=='quarantined_run']
        self.assertEqual(len(quarantine),1)
        self.assertEqual(quarantine[0]['active_run']['candidate_id'],partial['candidate_ids'][2])
        self.assertFalse(quarantine[0]['clean'])
        self.assertNotIn(quarantine[0]['active_run']['guard_run_id'],{r['guard_run_id'] for r in self.rows})
        status = json.loads((ROOT/'execution'/'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['state'],'stopped')
        self.assertEqual(status['recorded_processes'],101)

    def test_retained_rows_close_before_first_incident(self):
        incident = [e for e in self.events if e['event']=='incident']
        self.assertEqual(len(incident),1)
        self.assertEqual(incident[0]['reasons'],['monitor_query_failed'])
        self.assertIsNone(incident[0]['sample']['gpu'])
        self.assertIn('TimeoutExpired',incident[0]['sample']['monitor_error'])
        self.assertEqual(len([e for e in self.events if e['event']=='guard_abort']),1)
        self.assertFalse(any(e['event']=='guard_complete' for e in self.events))
        run_end = {e['guard_run_id']:e for e in self.events if e['event']=='run_one_end'}
        for row in self.rows[:99]:
            ending = run_end[row['guard_run_id']]
            self.assertTrue(ending['clean'])
            self.assertLess(ending['utc'],incident[0]['utc'])
            samples = [e for e in self.events if e['event']=='sample' and (e.get('active_run') or {}).get('guard_run_id')==row['guard_run_id']]
            self.assertTrue({'before','after'} <= {e.get('boundary') for e in samples})
            self.assertTrue(all(e.get('ok') is True and not e.get('foreign_compute_pids') and not e.get('violation_reasons') for e in samples))
            self.assertTrue(all(e['gpu_uuid']==self.design['scope']['device_uuid'] and e.get('monitor_error') is None and
                                e['gpu']['memory_used_mib']<=2048 and e['query_wall_seconds']<=2 for e in samples))

    def test_remaining_fixed_schedule_and_budget(self):
        pending = self.schedule[34:]
        self.assertEqual(len(pending),31)
        self.assertEqual(sum(len(x['candidate_ids']) for x in pending),100)
        self.assertEqual(len({(b['workload_id'],b['round']) for b in self.schedule}),65)
        # 102 launched attempts include the quarantined process.  Remaining
        # schedules are complete whole blocks, plus fresh health and cost.
        total = 8+28+23+30+102+30+100+48
        self.assertEqual(total,369)

    def test_recovery_guard_changes_only_script_allowlist(self):
        original = (HERE/'gpu_guard.py').read_text(encoding='utf-8')
        recovery = (HERE/'gpu_guard_cloud_recovery.py').read_text(encoding='utf-8')
        original = original.replace('("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py")',
                                    '("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py", "cloud_recovery.py")')
        original = original.replace('--script must be run_suite.py, controlled_recheck.py or cloud_reproduction.py',
                                    '--script must be run_suite.py, controlled_recheck.py, cloud_reproduction.py or cloud_recovery.py')
        self.assertEqual(original,recovery)

    def test_both_frozen_root_layouts_and_unsafe_layouts(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='recovery_layout_') as tmp:
            base = Path(tmp).resolve()
            here = base/'test'/'plan_selection_study'
            admission = {'source':{'root':'results/cloud_original'},'output':{'root':'results/cloud_recovery'}}
            for rootbase in (base,here):
                self.assertEqual(recovery.validate_root_layout(here,rootbase/'results/cloud_original',rootbase/'results/cloud_recovery',admission),rootbase)
            invalid = [(base/'results/cloud_original',here/'results/cloud_recovery'),
                       (base/'results/cloud_original',base/'results/cloud_original'),
                       (base/'results/cloud_original',base/'results/cloud_original/inner'),
                       (base/'other/cloud_original',base/'results/cloud_recovery')]
            for old,new in invalid:
                with self.subTest(old=old,new=new),self.assertRaises(recovery.Stop):
                    recovery.validate_root_layout(here,old,new,admission)

    def test_clean_process_import_uses_guard_wrapped_run_one(self):
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import run_suite;"
                "sentinel=lambda *args,**kwargs:None;run_suite.run_one=sentinel;"
                "import cloud_recovery;"
                "assert cloud_recovery.RecoveryController.attempt.__globals__['run_one'] is sentinel")
        completed = subprocess.run([sys.executable,'-c',code,str(HERE)],capture_output=True,text=True,timeout=15)
        self.assertEqual(completed.returncode,0,completed.stderr)


class InterruptedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((HERE/'cloud_reproduction_design.json').read_text(encoding='utf-8'))
        cls.admission = incident_admission()

    def test_only_whole_blocks_enter_frozen_remaining_schedule(self):
        blocks = recovery.remaining_schedule(self.design,self.admission)
        self.assertEqual(len(blocks),31)
        self.assertEqual(sum(len(x['candidate_ids']) for x in blocks),100)
        self.assertEqual(blocks[0]['candidate_ids'],['default_replica','a128_r4_l1_np1','a128_r2_l1_np1','default'])
        changed = copy.deepcopy(self.admission)
        changed['source']['preserved_blocks'].pop()
        with self.assertRaises(recovery.Stop): recovery.remaining_schedule(self.design,changed)

    def test_captured_exception_admits_only_99_closed_rows(self):
        result = recovery.audit_interrupted(ROOT,self.design,self.admission)
        self.assertEqual(len(result['rows']),99)
        self.assertEqual(len(result['blocks']),34)
        self.assertEqual(result['orphan_record_hashes'],self.admission['source']['orphan_record_hashes'])
        self.assertFalse(set(result['orphan_record_hashes']) & {r['record_sha256'] for r in result['rows']})
        self.assertNotIn(result['quarantined_guard_run_id'],{r['guard_run_id'] for r in result['rows']})

    def test_quarantine_and_prior_guard_abort_are_required(self):
        events = read_jsonl(ROOT/'guards'/'execution.jsonl')
        for mutant in ([e for e in events if e['event']!='quarantined_run'],
                       [e for e in events if e['event']!='guard_abort'],
                       events+[dict(event='guard_complete',guard_id=self.admission['source']['guard_id'])]):
            with self.subTest(events=len(mutant)),patch.object(recovery,'read_guard',return_value=mutant),self.assertRaises(recovery.Stop):
                recovery.audit_interrupted(ROOT,self.design,self.admission)

    def test_any_monitor_error_in_selected_run_is_fatal(self):
        events = read_jsonl(ROOT/'guards'/'execution.jsonl')
        mut = copy.deepcopy(events)
        sample = next(e for e in mut if e['event']=='sample' and e.get('boundary')=='before' and e.get('active_run'))
        sample['monitor_error'] = 'synthetic fault'
        with patch.object(recovery,'read_guard',return_value=mut),self.assertRaises(recovery.Stop):
            recovery.audit_interrupted(ROOT,self.design,self.admission)

    def test_boundary_samples_must_be_between_start_and_end(self):
        events = read_jsonl(ROOT/'guards'/'execution.jsonl')
        mut = copy.deepcopy(events)
        idx = next(i for i,e in enumerate(mut) if e['event']=='sample' and e.get('boundary')=='after' and e.get('active_run'))
        sample = mut.pop(idx)
        run_id = sample['active_run']['guard_run_id']
        end = next(i for i,e in enumerate(mut) if e.get('event')=='run_one_end' and e.get('guard_run_id')==run_id)
        mut.insert(end+1,sample)
        with patch.object(recovery,'read_guard',return_value=mut),self.assertRaises(recovery.Stop):
            recovery.audit_interrupted(ROOT,self.design,self.admission)

    def test_late_during_sample_must_still_be_clean(self):
        events = read_jsonl(ROOT/'guards'/'execution.jsonl')
        # Real logs include a clean during sample emitted after the run end due
        # to asynchronous monitoring; it may not be discarded merely for that.
        ends = {e['guard_run_id']:i for i,e in enumerate(events) if e.get('event')=='run_one_end'}
        late = next(i for i,e in enumerate(events) if e.get('event')=='sample' and e.get('boundary')=='during' and
                    (e.get('active_run') or {}).get('guard_run_id') in ends and
                    i > ends[(e.get('active_run') or {})['guard_run_id']])
        for kind in ('violation','foreign','slow','memory'):
            mut = copy.deepcopy(events)
            if kind=='violation': mut[late]['violation_reasons']=['synthetic']
            elif kind=='foreign': mut[late]['foreign_compute_pids']=[424242]
            elif kind=='slow': mut[late]['query_wall_seconds']=2.001
            else: mut[late]['gpu']['memory_used_mib']=2048.1
            with self.subTest(kind=kind),patch.object(recovery,'read_guard',return_value=mut),self.assertRaises(recovery.Stop):
                recovery.audit_interrupted(ROOT,self.design,self.admission)

    def test_reject_unfrozen_orphan_or_changed_block(self):
        for kind in ('orphan','block'):
            changed = copy.deepcopy(self.admission)
            if kind=='orphan': changed['source']['orphan_record_hashes'].reverse()
            else: changed['source']['preserved_blocks'][0]['block_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(recovery.Stop):
                recovery.audit_interrupted(ROOT,self.design,changed)


class RecoveryStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE,prefix='cloud_recovery_cpu_')
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.design = json.loads((HERE/'cloud_reproduction_design.json').read_text(encoding='utf-8'))
        self.admission = incident_admission()
        self.initial = recovery.audit_interrupted(ROOT,self.design,self.admission)
        self.original_ledger_hash = sha((ROOT/'execution'/'measurements.jsonl').read_bytes())
        self.original_status_hash = sha((ROOT/'execution'/'status.json').read_bytes())
        (self.output/'graph_repeats.json').write_bytes((ROOT/'graph_repeats.json').read_bytes())

    def controller(self,stage):
        study = object.__new__(recovery.RecoveryController)
        study.output,study.stage,study.stage_path = self.output,stage,self.output/stage
        study.stage_path.mkdir(parents=True)
        study.design,study.admission,study.preserved = self.design,self.admission,self.initial
        study.original = ROOT
        study.admission['recovery_id'] = 'synthetic-recovery'
        study.workloads = {w['id']:w for w in self.design['calibration']['workloads']}
        study.workloads.update({w['id']:w for w in self.design['workloads']})
        study.candidates = {c['id']:c for c in self.design['candidates']}
        admission_path = self.output/'synthetic_admission.json'
        write_json(admission_path,study.admission)
        study.args = argparse.Namespace(design_data=self.design,admission=admission_path)
        study.rows = []
        study.guard_id = 'synthetic-recovery-guard'
        return study

    def fake_attempt(self,study):
        def fake(w,cid,phase,k):
            from study_common import candidate_config
            row = fake_execution(study.args,w,(cid,candidate_config(study.candidates[cid]) if cid in study.candidates else {}),phase,k,len(study.rows))
            row['guard_id'] = study.guard_id
            row['guard_log'] = str(self.output/'guards'/'mock.jsonl')
            row['record_sha256'] = sha(canonical(row))
            study.rows.append(row)
            append_json(study.stage_path/'measurements.jsonl',row)
            return row
        return fake

    def test_fresh_health_uses_original_repeats_and_exactly_30_calls(self):
        study = self.controller('health')
        with patch.object(study,'attempt',side_effect=self.fake_attempt(study)) as calls:
            study.health()
        self.assertEqual(calls.call_count,30)
        self.assertTrue(all(call.args[2]=='health' for call in calls.call_args_list))
        original = json.loads((ROOT/'graph_repeats.json').read_text())
        self.assertTrue(all(call.args[0]['graph_repeats']==original[call.args[0]['id']]['graph_repeats'] for call in calls.call_args_list))
        self.assertEqual(len(list((study.stage_path/'blocks').glob('*.json'))),15)
        self.assertTrue(json.loads((study.stage_path/'gate.json').read_text())['passed'])

    def test_execution_repeats_the_entire_interrupted_block_then_96_remaining(self):
        study = self.controller('execution')
        with patch.object(study,'attempt',side_effect=self.fake_attempt(study)) as calls:
            study.execution()
        self.assertEqual(calls.call_count,100)
        self.assertEqual(len(list((study.stage_path/'blocks').glob('*.json'))),31)
        first = calls.call_args_list[:4]
        self.assertEqual([call.args[1] for call in first],['default_replica','a128_r4_l1_np1','a128_r2_l1_np1','default'])
        self.assertTrue(all((call.args[0]['id'],call.args[3],call.args[0]['block_attempt'])==('n16384_b2048',3,1) for call in first))
        self.assertTrue(all(call.args[0]['block_attempt']==0 for call in calls.call_args_list[4:]))
        selection = json.loads((self.output/'execution_selection.json').read_text())
        self.assertEqual(len(selection['blocks']),65)
        self.assertEqual((selection['selected_old_records'],selection['selected_new_records']),(99,100))
        hashes = [h for b in selection['blocks'] for h in b['record_hashes']]
        self.assertEqual(len(hashes),199)
        self.assertEqual(len(set(hashes)),199)
        self.assertFalse(set(hashes)&set(self.initial['orphan_record_hashes']))
        self.assertEqual(sha((ROOT/'execution'/'measurements.jsonl').read_bytes()),self.original_ledger_hash)
        self.assertEqual(sha((ROOT/'execution'/'status.json').read_bytes()),self.original_status_hash)
        with (patch.object(recovery,'completed_session',return_value=False),
              patch.object(recovery,'resolve_log',return_value=self.output/'mock')):
            with self.assertRaisesRegex(recovery.Stop,'guard_incomplete'):
                study.require_recovery_execution()

    def test_main_requires_completed_health_before_execution_or_cost(self):
        required = ['--admission','a','--recovery-freeze','f','--original','o','--design','d',
                    '--freeze','z','--output','x','--exe','e','--plain-exe','p','--build-provenance','b']
        for stage in ('execution','cost'):
            with self.subTest(stage=stage),patch.object(sys,'argv',['cloud_recovery.py',*required,'--stage',stage]),\
                 patch.object(recovery,'RecoveryController') as cls,patch.object(recovery.cloud,'run_one'):
                instance = cls.return_value
                instance.output = self.output
                write_json(self.output/'recovery_manifest.json',{'synthetic':True})
                instance.require.side_effect = recovery.Stop('health_guard_incomplete')
                with self.assertRaisesRegex(recovery.Stop,'health_guard_incomplete'):
                    recovery.main()
                instance.require.assert_called_once_with('health')
                self.assertFalse(instance.execution.called)
                self.assertFalse(instance.cost.called)
                instance.status.assert_called_once()
                self.assertEqual(instance.status.call_args.args,('stopped',))

    def test_main_never_runs_cost_without_completed_execution_guard(self):
        required = ['--admission','a','--recovery-freeze','f','--original','o','--design','d',
                    '--freeze','z','--output','x','--exe','e','--plain-exe','p','--build-provenance','b']
        with patch.object(sys,'argv',['cloud_recovery.py',*required,'--stage','cost']),\
             patch.object(recovery,'RecoveryController') as cls,patch.object(recovery.cloud,'run_one'):
            instance = cls.return_value
            instance.output = self.output
            write_json(self.output/'recovery_manifest.json',{'synthetic':True})
            instance.require_recovery_execution.side_effect = recovery.Stop('recovery_execution_guard_incomplete')
            with self.assertRaisesRegex(recovery.Stop,'recovery_execution_guard_incomplete'):
                recovery.main()
            instance.require.assert_called_once_with('health')
            self.assertFalse(instance.cost.called)

    def test_preflight_only_does_not_construct_controller_or_touch_output(self):
        required = ['--admission','a','--recovery-freeze','f','--original','o','--design','d',
                    '--freeze','z','--output','x','--exe','e','--plain-exe','p','--build-provenance','b']
        with patch.object(sys,'argv',['cloud_recovery.py',*required,'--preflight-only']),\
             patch.object(recovery,'preflight_validate',return_value={'status':'passed'}) as validate,\
             patch.object(recovery,'RecoveryController') as cls,contextlib.redirect_stdout(io.StringIO()):
            recovery.main()
            validate.assert_called_once()
            cls.assert_not_called()

    def test_real_e1_preflight_audits_capture_without_creating_output(self):
        admission = json.loads((HERE/'cloud_reproduction_recovery_E1.json').read_text(encoding='utf-8'))
        output = HERE/admission['output']['root']
        self.assertFalse(output.exists())
        args = argparse.Namespace(admission=HERE/'cloud_reproduction_recovery_E1.json',
            recovery_freeze=HERE/'cloud_reproduction_recovery_E1_freeze.json',original=ROOT,output=output,
            design=HERE/'cloud_reproduction_design.json',freeze=HERE/'cloud_reproduction_freeze.json',
            exe='not-executed',plain_exe='not-executed',build_provenance=ROOT/'build_provenance.json')
        manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        # The real Linux binaries are remote; only their runtime-manifest return
        # is mocked. All E1 source hashes, guard logs and records are real.
        with patch.object(cloud,'runtime_manifest',return_value=manifest):
            result = recovery.preflight_validate(args)
        self.assertEqual(result['status'],'preflight_passed_no_guard_no_gpu')
        self.assertEqual((result['preserved_records'],result['remaining_execution_processes']),(99,100))
        self.assertEqual(result['total_physical_max'],369)
        self.assertFalse(output.exists())

    def test_guard_wrapped_result_is_bound_to_recovery_manifest_before_recording(self):
        required = ['--admission','a','--recovery-freeze','f','--original','o','--design','d',
                    '--freeze','z','--output','x','--exe','e','--plain-exe','p','--build-provenance','b']
        manifest = self.output/'recovery_manifest.json'
        write_json(manifest,{'synthetic':True})
        observed = []
        with patch.object(sys,'argv',['cloud_recovery.py',*required,'--stage','health']),\
             patch.object(recovery,'RecoveryController') as cls,patch.object(recovery,'run_one',return_value={'guard_clean':True}),\
             patch.object(recovery.cloud,'run_one'):
            instance = cls.return_value
            instance.output = self.output
            instance.health.side_effect = lambda: observed.append(recovery.cloud.run_one(None,None,None,None,None,None))
            recovery.main()
        self.assertEqual(observed,[{'guard_clean':True,'recovery_manifest_sha256':sha(manifest.read_bytes())}])

    def test_cost_runs_exact_24_frozen_pairs_without_reselection(self):
        study = self.controller('cost')
        audit = {'ok':False,'errors':['describe_full_inequivalent:default'],'comparisons':[]}
        with patch.object(study,'attempt',side_effect=self.fake_attempt(study)) as calls,patch.object(cloud,'audit_pair',return_value=audit):
            study.cost()
        self.assertEqual(calls.call_count,48)
        pairs = [json.loads(p.read_text()) for p in (study.stage_path/'pairs').glob('*.json')]
        self.assertEqual(len(pairs),24)
        self.assertTrue(all(p['classification']=='negative' for p in pairs))
        for pair in pairs:
            self.assertEqual(pair['selected_id'],self.design['cost_profile']['selected_candidates'][pair['workload_id']])
            modes = ['full-search-cost','describe-select-cost']
            self.assertEqual(pair['candidate_ids'],modes if pair['round']%2==0 else list(reversed(modes)))
        gate = json.loads((study.stage_path/'gate.json').read_text())
        self.assertTrue(gate['protocol_valid'])
        self.assertFalse(gate['all_pairs_equivalent'])

    def test_explicit_stage_caps_stop_before_harness_call(self):
        for stage,cap in [('health',30),('execution',100),('cost',48)]:
            study = object.__new__(recovery.RecoveryController)
            study.stage,study.rows = stage,[{}]*cap
            with self.subTest(stage=stage),patch.object(cloud.CloudController,'attempt') as attempt:
                with self.assertRaisesRegex(recovery.Stop,'recovery_stage_process_budget_exhausted'):
                    study.attempt({},'default',stage,0)
                attempt.assert_not_called()

    def test_failed_process_is_retained_and_whole_block_is_not_selected(self):
        study = self.controller('execution')
        study.args.exe = 'synthetic'
        study.args.plain_exe = 'synthetic-plain'
        study.args.expected_executable_sha256 = 'synthetic'
        study.measurements = study.stage_path/'measurements.jsonl'
        study.active = None
        write_json(study.output/'manifest.json',{'synthetic':True})
        def failed(*args):
            row = fake_execution(*args)
            row['status'] = 'failed'
            row['validation_errors'] = ['numerical_failure:correctness']
            return row
        first = recovery.remaining_schedule(self.design,self.admission)[0]
        w = study.workloads[first['workload_id']]
        with patch.object(cloud,'run_one',side_effect=failed) as runner,patch.object(cloud,'audit_row',return_value={'ok':True}),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(recovery.Stop,'process_failed'):
                study.block(w,'execution',first['round'],first['candidate_ids'],1,101)
        self.assertEqual(runner.call_count,1)
        retained = read_jsonl(study.measurements)
        self.assertEqual(len(retained),1)
        self.assertEqual(retained[0]['status'],'failed')
        self.assertEqual(retained[0]['validation_errors'],['numerical_failure:correctness'])
        self.assertFalse((study.output/'execution_selection.json').exists())


class RecoveryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admission_path = HERE/'cloud_reproduction_recovery_E1.json'
        cls.freeze_path = HERE/'cloud_reproduction_recovery_E1_freeze.json'
        cls.admission = json.loads(cls.admission_path.read_text(encoding='utf-8'))
        cls.original = ROOT
        cls.unfinished = HERE/cls.admission['output']['root']

    def test_actual_partial_e1_reports_99_preserved_but_no_accepted_selection(self):
        self.assertFalse(self.unfinished.exists())
        result = recovered_report.analyze_recovery(self.original,self.unfinished,self.admission_path,self.freeze_path)
        self.assertEqual(result['old_preserved_count'],99)
        self.assertEqual(result['selected_record_count'],0)
        self.assertEqual(result['stage_record_counts'],{'health':0,'execution':0,'cost':0})
        self.assertTrue({'health','execution','cost','recovery'} <= set(result['stage_errors']))
        self.assertIsNone(result['cost'])
        self.assertTrue(all(not x['inferential'] and x['p']==1 for x in result['directional_tests']))
        self.assertFalse(self.unfinished.exists())

    def test_missing_frozen_source_hash_blocks_all_results(self):
        original_json_file = recovered_report.json_file
        altered = copy.deepcopy(self.admission)
        altered['source']['files'][0]['path'] = altered['source']['root']+'/does-not-exist'
        def read_modified(path):
            return altered if Path(path).resolve()==self.admission_path.resolve() else original_json_file(path)
        with patch.object(recovered_report,'json_file',side_effect=read_modified):
            result = recovered_report.analyze_recovery(self.original,self.unfinished,self.admission_path,self.freeze_path)
        self.assertIn('original',result['stage_errors'])
        self.assertTrue(any('source_hash_mismatch' in e for e in result['stage_errors']['original']))
        self.assertEqual(result['selected_record_count'],0)

    def test_synthetic_full_selection_and_guard_rejection(self):
        design = json.loads((HERE/'cloud_reproduction_design.json').read_text())
        old = recovery.audit_interrupted(self.original,design,self.admission)['rows']
        raw = read_jsonl(self.original/'execution'/'measurements.jsonl')
        new = []
        for i in range(100):
            row = copy.deepcopy(raw[i])
            row['record_sha256'] = 'synthetic-new-'+str(i)
            new.append(row)
        schedule = build_schedule(design)
        blocks = []
        offsets = {'original':0,'recovery':0}
        for i,b in enumerate(schedule):
            src = 'original' if i<34 else 'recovery'
            source = old if src=='original' else new
            start = offsets[src]
            hashes = [r['record_sha256'] for r in source[start:start+len(b['candidate_ids'])]]
            offsets[src] += len(hashes)
            blocks.append(dict(**b,source_root=src,record_hashes=hashes))
        self.assertEqual(offsets,{'original':99,'recovery':100})
        with tempfile.TemporaryDirectory(dir=HERE,prefix='recovery_report_cpu_') as tmp:
            out = Path(tmp)
            (out/'manifest.json').write_bytes((self.original/'manifest.json').read_bytes())
            (out/'graph_repeats.json').write_bytes((self.original/'graph_repeats.json').read_bytes())
            write_json(out/'execution_selection.json',{'blocks':blocks})
            manifest = dict(recovery_id=self.admission['recovery_id'],admission_sha256=sha(self.admission_path.read_bytes()),
                freeze_sha256=sha(self.freeze_path.read_bytes()),original_manifest_sha256=sha((self.original/'manifest.json').read_bytes()),
                graph_repeats_sha256=self.admission['source']['graph_repeats_sha256'],
                recovery_guard_sha256=self.admission['guards']['recovery_guard_sha256'],device_uuid=design['scope']['device_uuid'],
                recovery_controller_sha256=sha((HERE/'cloud_recovery.py').read_bytes()))
            write_json(out/'recovery_manifest.json',manifest)
            for stage in ('health','execution','cost'): write_json(out/stage/'status.json',{'state':'complete'})
            rmsha = sha((out/'recovery_manifest.json').read_bytes())
            mock_rows = {'health':[{'recovery_manifest_sha256':rmsha} for _ in range(30)],
                         'execution':new,'cost':[{'recovery_manifest_sha256':rmsha} for _ in range(48)]}
            for row in new: row['recovery_manifest_sha256']=rmsha
            def stage_rows(_root,stage,_design,_manifest): return mock_rows[stage],[]
            negative = {'feasibility':'negative','pairs':[]}
            with patch.object(recovered_report,'validate_root_layout',return_value=HERE),\
                 patch.object(recovered_report.report,'stage_rows',side_effect=stage_rows),\
                 patch.object(recovered_report.CloudController,'require'),\
                 patch.object(recovered_report.RecoveryController,'require_recovery_execution'),\
                 patch.object(recovered_report.report,'analyze_cost',return_value=negative):
                result = recovered_report.analyze_recovery(self.original,out,self.admission_path,self.freeze_path)
                self.assertEqual(result['stage_errors'],{})
                self.assertEqual((result['old_preserved_count'],result['selected_record_count'],result['selected_block_count']),(99,199,65))
                self.assertEqual(result['cost']['feasibility'],'negative')
                def unclean(_root,stage,_design,_manifest):
                    return mock_rows[stage],['guard_session_incomplete'] if stage=='execution' else []
                with patch.object(recovered_report.report,'stage_rows',side_effect=unclean):
                    failed = recovered_report.analyze_recovery(self.original,out,self.admission_path,self.freeze_path)
                self.assertIn('execution',failed['stage_errors'])
                self.assertEqual(failed['selected_record_count'],0)
                self.assertIsNone(failed['cost'])


if __name__=='__main__':
    unittest.main()
