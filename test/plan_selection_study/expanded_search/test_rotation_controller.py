import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

import rotation_controller as ctl
import rotation_evidence as evidence
# Windows can run the CPU mock; actual device locking remains Linux-only.
if os.name == 'nt' and 'fcntl' not in sys.modules: sys.modules['fcntl']=types.ModuleType('fcntl')
import rotation_runner


class FakeBudget:
    def __init__(self,limit=100):
        self.state={'phase':'extension','stopped':False,'inflight':None};self.limit=limit;self.starts=[];self.ends=[];self.checkpoints=[]
    def may_start(self,kind,**kwargs):
        if len(self.starts)>=self.limit:return False
        self.starts.append((kind,kwargs));self.state['inflight']=kwargs;return True
    def finish_work(self,status,**kwargs):
        if self.state['inflight'] is None:raise AssertionError('missing in-flight work')
        self.ends.append((status,kwargs));self.state['inflight']=None
    def checkpoint(self,kind,**kwargs):self.checkpoints.append((kind,kwargs))


def case():
    return {'n':256,'batch':16,'candidate_id':'candidate','request':{'radix':[8,4,8],'logical_registers':8,'grouped_batch':4},
            'graph_repeats':64,'original_signature':{'original':True},'candidate_signature':{'candidate':True}}


def valid_row(original):
    c=case();gpu='GPU-test';slots=2;R=64;bytes_=c['n']*c['batch']*8
    correctness={'passed':True,'criterion':'criterion_v2','checked_elements':c['n']*c['batch'],
                 'reference':'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity',
                 'nonfinite_count':0,'failing_batches':0,'l2_scaled_error_threshold':1e-5,'linf_input_scaled_error_threshold':1e-4}
    entries=[{'slot':i,'input_unchanged':True,'correctness':copy.deepcopy(correctness)} for i in range(slots)]
    summary={'passed':True,'slots_checked':slots,'elements_per_slot':c['n']*c['batch'],'slots':entries}
    d={'axisSwapped':0,'logical_registers_per_thread':8,'registerBoost':1}
    r={'status':'ok','gpu':{'uuid':gpu,'cuda_visible_devices':gpu},'n':c['n'],'batch':c['batch'],'seed':42,'pattern':'random',
       'header_instrumentation':not original,'requested_config':{'makeForwardPlanOnly':1},
       'rotation_mode':'multi_address_graph_random_unique_scaled_inputs','rotation_pool_slots':slots,
       'rotation_pool_total_bytes':2*slots*bytes_,'rotation_input_pool_bytes':slots*bytes_,'rotation_output_pool_bytes':slots*bytes_,
       'rotation_device_l2_bytes':bytes_,'rotation_graph_kernel_arguments_and_order_verified':True,
       'rotation_distinct_input_representatives_verified':True,
       'rotation_output_initialization':'0xff_bytes_fp32_nan_before_initial_and_separate_post_timing_validation_graph',
       'graph_repeats':R,'graph_kernel_count':R,'graph_event_record_count':2,'graph_node_count':R+2,
       'timing_scope':'multi_address_graph_forward_same_stream_no_transfers_or_validation_inside_event_interval',
       'rotation_initial_validation':copy.deepcopy(summary),'rotation_final_validation':copy.deepcopy(summary),
       'rotation_first_graph_interval_ms':2.0,'rotation_final_validation_graph_interval_ms':2.0,
       'warmup_actual':5,'warmup_wall_ms':1000,'exec_ms_samples':[2.0/R]*31,'raw_interval_ms_samples':[2.0]*31,
       'structural_features':{'numAxisUploads':1,'kernels':[{'stageRadix':c['request']['radix'],'groupedBatch':4,'localSize':[32,4,1]}]},
       'precompile_kernels':[{'compiler_options_match':True,'raw_descriptor':d}],
       'postinit_source_kernels':[{}],'init_trace':{'available':False},'descriptor_validation':{'all_match':True},
       'expanded_resource_validation':{'dynamic_shared_bytes':100,'function_dynamic_limit_bytes':200,'static_shared_bytes':0,
                                       'device_shared_limit_bytes':200,'threads':128,'function_thread_limit':1024}}
    return {'result':r,'workload':{'n':c['n'],'batch':c['batch']},'status':'ok','returncode':0,'guard_clean':True,
            'guard_gpu_uuid':gpu,'direct_environment':{} if original else ctl.validation.request_env(c['request'])}


class RotationTest(unittest.TestCase):
    def test_runner_never_builds_without_confirmed_winners(self):
        class Lock:
            fd=0
            def __enter__(self):return self
            def __exit__(self,*_):return False
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);args=[]
            for name in ('state','results','expanded-generated-root','plan','gate','protocol','first-gate','first-audit','second-audit','combined-audit',
                         'first-rows','first-guard','first-incidents','first-budget','second-rows','second-guard','second-incidents','second-budget'):
                args.extend(['--'+name,str(root/name)])
            (root/'gate').write_text('{}');budget=FakeBudget()
            with patch.object(rotation_runner.device_lock,'DeviceLock',return_value=Lock()), \
                 patch.object(rotation_runner.budget,'Budget',return_value=budget), \
                 patch.object(rotation_runner.rotation_evidence,'verify',return_value=({}, {}, [], {})), \
                 patch.object(rotation_runner.subprocess,'run') as spawn:
                self.assertEqual(rotation_runner.main(args),0)
                spawn.assert_not_called()
            self.assertEqual(budget.checkpoints[-1][1]['reason'],'no_two_session_confirmed_winners')

    def test_two_session_gate_bound_to_exact_evidence_and_confirmed_plan(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);a=SimpleNamespace()
            names=('plan','gate','protocol','first_gate','first_audit','second_audit','combined_audit',
                   'first_rows','first_guard','first_incidents','first_budget','second_rows','second_guard','second_incidents','second_budget')
            for name in names:
                path=root/(name+'.json');path.write_text('{}');setattr(a,name,path)
            c=case();plan={'sha256':{'original':'o','expanded':'e'},'workloads':[{'n':c['n'],'batch':c['batch'],'candidate':c}]}
            a.plan.write_text(json.dumps(plan));a.protocol.write_text('{"protocol":1}')
            filemap={side:{field:evidence.sha(getattr(a,side+'_'+field)) for field in ('rows','guard','incidents','budget')} for side in ('first','second')}
            firstgate={'evidence_sha256':filemap['first'],'binaries_sha256':plan['sha256']};a.first_gate.write_text(json.dumps(firstgate))
            first={'status':'pass','session_kind':'first','session_id':'f'};second={'status':'pass','session_kind':'second','session_id':'s'}
            budget={'status':'pass','phase':'confirmation','inflight':None}
            a.first_audit.write_text(json.dumps({'status':'pass','first_session':first,'budget':budget}))
            a.second_audit.write_text(json.dumps({'status':'pass','second_session':second}))
            combined={'status':'pass','candidates':[{'n':256,'batch':16,'candidate_id':'candidate','status':'confirmed_under_preregistered_rule'}]}
            a.combined_audit.write_text(json.dumps({'status':'pass','combined':combined,'first_session':first,'second_session':second,
                                                    'first_budget':budget,'second_budget':budget}))
            gate={'schema':'expanded-two-session-gate/v1','status':'pass','plan_sha256':evidence.sha(a.plan),
                  'protocol_sha256':evidence.sha(a.protocol),'first_gate_sha256':evidence.sha(a.first_gate),
                  'first_audit_sha256':evidence.sha(a.first_audit),'second_audit_sha256':evidence.sha(a.second_audit),
                  'combined_audit_sha256':evidence.sha(a.combined_audit),'session_evidence_sha256':filemap,
                  'binaries_sha256':plan['sha256'],'confirmed':[c]}
            a.gate.write_text(json.dumps(gate))
            def fake_read(path):
                if Path(path)==a.first_rows:
                    return [{'side':'first'},{'record_type':'execution','candidate_id':'original','workload':{'n':256,'batch':16},
                                             'result':{'gpu':{'uuid':'GPU-test','l2_cache_bytes':32768}}}]
                if Path(path)==a.second_rows:
                    return [{'side':'second','record_type':'confirmation_protocol'},{'record_type':'execution','candidate_id':'original',
                            'workload':{'n':256,'batch':16},'result':{'gpu':{'uuid':'GPU-test','l2_cache_bytes':32768}}}]
                return []
            with patch.object(evidence.audit_confirmation,'read',side_effect=fake_read), \
                 patch.object(evidence.audit_confirmation,'audit',side_effect=lambda rows,*_a:first if rows[0]['side']=='first' else second), \
                 patch.object(evidence.audit_two_sessions,'combine',return_value=combined):
                _p,_g,confirmed,l2=evidence.verify(a)
                self.assertEqual(confirmed,[c]);self.assertEqual(l2[(256,16,'candidate')],('GPU-test',32768))
                bad=copy.deepcopy(gate);bad['confirmed'][0]['request']['logical_registers']=16
                a.gate.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):evidence.verify(a)
                a.gate.write_text(json.dumps(gate));a.protocol.write_text('{"changed":true}')
                with self.assertRaises(ValueError):evidence.verify(a)

    def test_pool_threshold_and_graph_capacity(self):
        self.assertEqual(evidence.pool(1024,4096,16),{'slots':8,'total_bytes':16384,'graph_repeats':64})
        self.assertIsNone(evidence.pool(1,100000,64))
        self.assertIsNone(evidence.pool(800*1024*1024,1,64))

    def test_full_pool_graph_and_input_checks_fail_closed(self):
        c=case();rotation={'slots':2,'graph_repeats':64,'l2_bytes':c['n']*c['batch']*8}
        with patch.object(ctl.search_controller,'source_signature',side_effect=lambda _r,original:{'original':True} if original else {'candidate':True}):
            for original in (True,False):
                row=valid_row(original);ctl.check(row,'GPU-test',c,rotation,original=original,timing=True)
                for alteration in ('missing_slot','input_changed','node_unverified','too_small','nonfinite'):
                    bad=copy.deepcopy(row);r=bad['result']
                    if alteration=='missing_slot':r['rotation_final_validation']['slots'].pop()
                    if alteration=='input_changed':r['rotation_initial_validation']['slots'][0]['input_unchanged']=False
                    if alteration=='node_unverified':r['rotation_graph_kernel_arguments_and_order_verified']=False
                    if alteration=='too_small':r['rotation_device_l2_bytes']*=2
                    if alteration=='nonfinite':r['rotation_final_validation']['slots'][1]['correctness']['nonfinite_count']=1
                    with self.assertRaises(ValueError):ctl.check(bad,'GPU-test',c,rotation,original=original,timing=True)

    def test_ten_pairs_and_budget_exhaustion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);orig=root/'original';expanded=root/'expanded';orig.write_bytes(b'o');expanded.write_bytes(b'e')
            evidence_names=('plan','gate','protocol','first-gate','first-audit','second-audit','combined-audit',
                            'first-rows','first-guard','first-incidents','first-budget','second-rows','second-guard','second-incidents','second-budget')
            args=['--gpu','GPU-test','--budget-dir',str(root/'budget'),'--output',str(root/'rotation.jsonl'),
                  '--rotation-original-exe',str(orig),'--rotation-expanded-exe',str(expanded),
                  '--rotation-original-sha256',evidence.sha(orig),'--rotation-expanded-sha256',evidence.sha(expanded),'--timeout','10']
            for name in evidence_names:
                path=root/(name+'.json');path.write_text('{}');args.extend(['--'+name,str(path)])
            c=case();plan={'sha256':{'original':'hot-o','expanded':'hot-e'},'selection_seed':42}
            l2={(256,16,'candidate'):('GPU-test',32768)};calls=[]
            def fake_run(a,w,candidate,phase,round_no,order):
                calls.append((candidate[0],phase,round_no,order))
                return {'record_type':'execution','status':'ok','returncode':0,'guard_clean':True,'guard_gpu_uuid':'GPU-test','guard_run_id':f'run-{len(calls)}',
                        'workload':w,'result':{'exec_ms_samples':[2.0/w['graph_repeats']]*w['repeats'],
                                               'raw_interval_ms_samples':[2.0]*w['repeats']}}
            budget=FakeBudget()
            with patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'guard','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(ctl.budget,'Budget',return_value=budget), \
                 patch.object(ctl.rotation_evidence,'verify',return_value=(plan,{},[c],l2)), \
                 patch.object(ctl.suite,'run_one',side_effect=fake_run), \
                 patch.object(ctl,'check',side_effect=lambda row,*a,**kw:row['result']):
                ctl.main(args)
            result=[json.loads(line) for line in (root/'rotation.jsonl').read_text().splitlines()]
            self.assertEqual(len([x for x in result if x['record_type']=='rotation_pair']),10)
            self.assertEqual(len(calls),21);self.assertEqual(result[-1]['status'],'completed')
            self.assertEqual(len(budget.starts),len(budget.ends))
            self.assertEqual(sum(x[0]=='rotation_pair' for x in budget.starts),10)
            args[args.index('--output')+1]=str(root/'partial.jsonl');limited=FakeBudget(limit=2)
            with patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'guard','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(ctl.budget,'Budget',return_value=limited), \
                 patch.object(ctl.rotation_evidence,'verify',return_value=(plan,{},[c],l2)), \
                 patch.object(ctl.suite,'run_one',side_effect=fake_run), \
                 patch.object(ctl,'check',side_effect=lambda row,*a,**kw:row['result']):
                ctl.main(args)
            partial=[json.loads(line) for line in (root/'partial.jsonl').read_text().splitlines()]
            self.assertEqual(partial[-1]['status'],'budget_exhausted')
            self.assertEqual(len(partial[-1]['incomplete_or_not_started']),1)
            self.assertFalse(any(x.get('status')=='complete' for x in partial))


if __name__=='__main__':unittest.main()
