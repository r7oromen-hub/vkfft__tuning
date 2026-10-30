import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import search_controller as search


class FakeBudget:
    def __init__(self, limit=10000):
        self.state={'phase':'search','stopped':False,'inflight':None}
        self.limit=limit; self.started=[]; self.finished=[]
    def may_start(self, kind, **kw):
        if len(self.started)>=self.limit: return False
        self.started.append((kind,kw));self.state['inflight']=kw;return True
    def finish_work(self,status,**kw):
        if not self.state['inflight']: raise AssertionError('not in flight')
        self.finished.append((status,kw));self.state['inflight']=None


class SearchTest(unittest.TestCase):
    def test_only_complete_guarded_numerical_failures_are_recoverable(self):
        req={'radix':[8,4,8],'logical_registers':8,'grouped_batch':4}
        corr={'passed':False,'criterion':'criterion_v2','reference':'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity',
              'checked_elements':256*16,'l2_scaled_error_threshold':1e-5,'linf_input_scaled_error_threshold':1e-4,
              'nonfinite_count':1,'failing_batches':1}
        r={'gpu':{'uuid':'GPU-test','cuda_visible_devices':'GPU-test'},'n':256,'batch':16,'seed':42,'pattern':'random',
           'header_instrumentation':True,'requested_config':{'makeForwardPlanOnly':1},
           'validation_output_initialization':'0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
           'single_forward_input_unchanged':True,'descriptor_validation':{'all_match':True},
           'structural_features':{'numAxisUploads':1,'kernels':[{'localSize':[32,4,1],'stageRadix':req['radix'],'groupedBatch':4}]},
           'precompile_kernels':[{'compiler_options_match':True,'raw_descriptor':{'axisSwapped':0,'logical_registers_per_thread':8,'registerBoost':1}}],
           'expanded_resource_validation':{'dynamic_shared_bytes':100,'function_dynamic_limit_bytes':200,'static_shared_bytes':0,
                                           'device_shared_limit_bytes':200,'threads':128,'function_thread_limit':1024},
           'status':'error','failure_phase':'single_forward_validation',
           'error':'Single out-of-place forward failed immutable-input or numerical validation','single_forward_correctness':corr}
        row={'guard_clean':True,'guard_gpu_uuid':'GPU-test','returncode':1,'status':'failed','result':r,
             'workload':{'n':256,'batch':16,'seed':42,'pattern':'random'},'direct_environment':search.validation.request_env(req)}
        self.assertTrue(search.numerical_failure(row,'GPU-test',req))
        for field in ('guard_clean','returncode'):
            bad=copy.deepcopy(row);bad[field]=False if field=='guard_clean' else 2
            self.assertFalse(search.numerical_failure(bad,'GPU-test',req))
        bad=copy.deepcopy(row);bad['result']['expanded_resource_validation']['threads']=1
        self.assertFalse(search.numerical_failure(bad,'GPU-test',req))
        bad=copy.deepcopy(row);bad['result']['single_forward_correctness'].pop('reference')
        self.assertFalse(search.numerical_failure(bad,'GPU-test',req))

    def test_admission_gate_fails_closed(self):
        evidence={'rows':'r','guard':'g','incidents':'i','protocol':'p'}; binaries={'expanded':'e','original':'o'}
        gate={'schema':'expanded-admission-gate/v1','status':'pass','integrity_status':'pass',
              'required_dimensions':dict.fromkeys(search.REQUIRED_ADMISSION,True),
              'evidence_sha256':evidence,'binaries_sha256':binaries}
        search.check_admission_gate(gate,evidence,binaries)
        for change in ('coverage','approval','binary','evidence'):
            bad=copy.deepcopy(gate)
            if change=='coverage': bad['required_dimensions']['first_radix_changed']=False
            if change=='approval': bad['status']='ready_for_review'
            if change=='binary': bad['binaries_sha256']['original']='other'
            if change=='evidence': bad['evidence_sha256']['rows']='other'
            with self.assertRaises(ValueError): search.check_admission_gate(bad,evidence,binaries)

    def test_workload_and_candidate_round_order(self):
        p={'lengths':[128,256,512], 'batches':[1,16]}
        order=search.workload_order(p)
        self.assertEqual(order,[(128,1),(256,1),(512,1),(128,16),(256,16),(512,16)])
        contexts={(128,1):{'selected':[0,1]},(512,1):{'selected':[2,3]}}
        self.assertEqual(list(search.candidate_rounds(contexts,order)),[((128,1),0),((512,1),2),((128,1),1),((512,1),3)])
        self.assertEqual(search.seeded_order(3,1,2),search.seeded_order(3,1,2))

    def test_budgeted_pairs_and_complete_raw_records(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);original=root/'original';expanded=root/'expanded'
            original.write_bytes(b'original');expanded.write_bytes(b'expanded')
            protocol=root/'study.json';protocol.write_bytes((Path(__file__).parent/'study_protocol.json').read_bytes())
            admission_protocol=root/'admission.json';admission_protocol.write_text('{}')
            rows=root/'admission_rows.jsonl';guard=root/'guard.jsonl';incidents=root/'incidents.jsonl'
            hashes={'original':search.sha(original),'expanded':search.sha(expanded)}
            rows.write_text(json.dumps({'record_type':'admission_protocol','sha256':hashes})+'\n')
            guard.write_text('');incidents.write_text('')
            gate=root/'gate.json'
            gate.write_text(json.dumps({'schema':'expanded-admission-gate/v1','status':'pass','integrity_status':'pass',
                'required_dimensions':dict.fromkeys(search.REQUIRED_ADMISSION,True),
                'evidence_sha256':{'rows':search.sha(rows),'guard':search.sha(guard),'incidents':search.sha(incidents),'protocol':search.sha(admission_protocol)},
                'binaries_sha256':hashes}))
            a=SimpleNamespace(gpu='GPU-test',expanded_exe=expanded,original_exe=original,budget_dir=root/'budget',
                              output=root/'search.jsonl',manifest_dir=root/'manifests',study_protocol=protocol,
                              admission_protocol=admission_protocol,admission_rows=rows,admission_guard=guard,
                              admission_incidents=incidents,admission_gate=gate,timeout=10)
            p=json.loads(protocol.read_text());p['gpu_uuid']=a.gpu;protocol.write_text(json.dumps(p))
            budget=FakeBudget()
            def make_manifest(path,n,batch,req,seed,limit):
                path.write_text('fixture\n')
                selected=[]
                if (n,batch) in ((128,1),(256,1)):
                    selected=[{'request':req,'candidate_id':f'c{n}', 'selection_index':0,'planned_fft_threads':n//8}]
                return selected,{'selected':len(selected)}
            calls=[]
            def fake_run(args,w,candidate,phase,round_no,order):
                name,env=candidate;calls.append((phase,w['n'],w['batch'],name,w['pattern'],w['seed']))
                source={'canonical_descriptor':{'x':1},'source_sha256':'src','source_bytes':1,'compiler_input':{'x':1},
                        'expected_compiler_input':{'x':1},'actual_compiler_options':['opt'], 'raw_descriptor':{'axisSwapped':0}}
                r={'n':w['n'],'batch':w['batch'],'raw_interval_ms_samples':[3.0]*w['repeats'],
                   'exec_ms_samples':[3.0/w['graph_repeats']]*w['repeats'],
                   'warmup_actual':5,'warmup_wall_ms':1000,
                   'structural_features':{'kernels':[{'localSize':[16,1,1]}]},'compiled_diagnostics':[],
                   'resolved_config':{},'precompile_kernels':[copy.deepcopy(source)],'postinit_source_kernels':[copy.deepcopy(source)]}
                return {'record_type':'execution','status':'ok','result':r,'workload':w,'guard_run_id':f'run-{len(calls)}','direct_environment':env}
            def passthrough(row,*_args,**_kwargs): return row['result']
            with patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'guard','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(search.audit_admission,'audit',return_value={'status':'pass'}), \
                 patch.object(search.original_comparison,'compare'), \
                 patch.object(search.identity,'representable',return_value=({'radix':[8,8,2],'logical_registers':8,'grouped_batch':1},None)), \
                 patch.object(search.manifest,'make',side_effect=make_manifest), \
                 patch.object(search.suite,'run_one',side_effect=fake_run), \
                 patch.object(search.validation,'check',side_effect=passthrough), \
                 patch.object(search.validation,'check_describe',side_effect=passthrough), \
                 patch.object(search.validation,'known_rejection',return_value=False):
                self.assertEqual(search.run(a,budget_obj=budget),'completed_selected')
                a.output=root/'interrupted.jsonl';a.manifest_dir=root/'interrupted_manifests'
                limited=FakeBudget(limit=143)
                self.assertEqual(search.run(a,budget_obj=limited),'budget_exhausted')
                partial=[json.loads(line) for line in a.output.read_text().splitlines()]
                self.assertEqual(partial[-1]['status'],'budget_exhausted')
                self.assertEqual(len([x for x in partial if x['record_type']=='candidate_start']),1)
                self.assertFalse(any(x.get('status')=='screen_complete' for x in partial))
                self.assertEqual(len(limited.started),len(limited.finished))
            output=[json.loads(line) for line in (root/'search.jsonl').read_text().splitlines()]
            screens=[x for x in output if x['record_type']=='screen_pair']
            self.assertEqual(len(screens),6)
            self.assertEqual(len([x for x in calls if x[0]=='candidate_validation']),2*15*2+2*2)
            self.assertEqual(len([x for x in calls if x[0]=='screen_timing']),2*3*2)
            completed=[x for x in output if x.get('status')=='screen_complete']
            self.assertEqual(len(completed),2)
            self.assertTrue(all(len(x['pair_ratios'])==3 and x['graph_repeats']==64 for x in completed))
            self.assertTrue(all(x['status']=='completed_selected' for x in output if x['record_type']=='search_complete'))
            self.assertEqual(len(budget.started),len(budget.finished))
            self.assertTrue(all(kw['expected_upper_seconds']==40 for kind,kw in budget.started if kind=='search_process'))
            self.assertTrue(all(kw['expected_upper_seconds']==80 for kind,kw in budget.started if kind=='search_pair'))
            self.assertEqual(len([x for x in budget.started if x[0]=='manifest_write']),28)


if __name__=='__main__': unittest.main()
