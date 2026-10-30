import copy, json, unittest
import audit_joint as audit

def fixture():
    gpu='GPU-example'; gid='guard'; sha='a'*64
    req={'radix':[2,2],'grouped_batch':1,'block':[2,1,1]}; cid=audit.candidate_id(req)
    desc={'axisSwapped':0,'fftDim':4,'complexSize':8,'registerBoost':1,'registers_per_radix':[0,0,2]+[0]*30,'logical_registers_per_thread':2,'min_logical_registers_per_thread':2}
    check={'checked_elements':4,'passed':True,'nonfinite_count':0,'failing_batches':0,'reference':'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity','criterion':'criterion_v2','l2_scaled_error_threshold':1e-5,'linf_input_scaled_error_threshold':1e-4}
    result={'status':'ok','n':4,'batch':1,'seed':42,'gpu':{'uuid':gpu,'cuda_visible_devices':gpu,'name':'NVIDIA GeForce RTX 4090'},'requested_config':{'makeForwardPlanOnly':1},'vkfft_result_code':0,
            'backend':'CUDA_NVRTC','precision':'complex_fp32','forward_direction':-1,'placement':'outofplace','study_mode':'execute','timing_mode':'graph','pattern':'random', 'single_forward_input_unchanged':True,'input_unchanged':True,'single_forward_correctness':check,'correctness':check,
            'validation_output_initialization':'0xff_bytes_fp32_nan_before_single_and_final_timed_forward','descriptor_validation':{'all_match':True},
            'precompile_kernels':[{'canonical_descriptor':'stable','source_sha256':'s','compiler_input':'x','actual_compiler_options':'x','compiler_options_match':True,'raw_descriptor':desc}],
            'structural_features':{'numAxisUploads':1,'kernels':[{'stageRadix':[2,2],'groupedBatch':1,'localSize':[2,1,1]}]},
            'resolved_config':{'sharedMemorySize':49152,'maxThreadsNum':1024},'graph_repeats':1,'graph_kernel_count':1,'graph_event_record_count':2,
            'direct_resource_validation':{'dynamic_shared_bytes':0,'static_shared_bytes':0,'function_dynamic_limit_bytes':49152,'device_shared_limit_bytes':49152,'threads':2,'function_thread_limit':1024}}
    protocol={'record_type':'protocol','n':4,'batch':1,'gpu':gpu,'seed':42,'exe_sha256':sha,'limit':0,'screen_rounds':3,'confirm_rounds':10}
    records=[protocol];guard=[{'event':'guard_start','gpu_uuid':gpu,'guard_id':gid,'own_signal_method':'pidfd_with_rechecked_PPID_exe_starttime','parent_pid':100,'idle_seconds':0,'script':'/study/joint_controller.py'},{'event':'idle_complete','gpu_uuid':gpu,'guard_id':gid,'clean':True,'observed_idle_seconds':0}]
    def run(name,phase,num,order=0,round=0):
        rid=f'{gid}:{num}'; timing=phase=='null_timing'; count=31 if timing else 1
        w={'n':4,'batch':1,'seed':42,'config':{},'pattern':'random','graph_repeats':1,'repeats':count,'warmup':5 if timing else 0,'warmup_ms':1000 if timing else 0}
        r=copy.deepcopy(result);r['exec_ms_samples']=[2.0]*count;r['raw_interval_ms_samples']=[2.0]*count
        env={'VKFFT_DIRECT_RADIX':'2,2','VKFFT_DIRECT_BATCH':'1','VKFFT_DIRECT_BLOCK':'2,1,1'} if name=='identity' else {}
        row={'candidate_id':name,'phase':phase,'round':round,'order':order,'workload_id':'n4_b1','workload':w,'command':['/binary','--n','4','--batch','1','--seed','42','--timing-mode','graph','--placement','outofplace','--graph-repeats','1','--warmup',str(w['warmup']),'--warmup-ms',str(w['warmup_ms']),'--repeats',str(count),'--pattern','random','--study-mode','execute'],'guard_id':gid,'guard_run_id':rid,'guard_gpu_uuid':gpu,'guard_clean':True,'direct_environment':env,'status':'ok','returncode':0,'result':r}
        row['stdout']=json.dumps(r)+'\n';records.append(row)
        active={'guard_run_id':rid,'candidate_id':name,'phase':phase,'round':round,'order':order,'expected_exe':'/binary'}
        common={'guard_id':gid,'gpu_uuid':gpu}
        guard.append(dict(event='run_one_start',active_run=active,**common))
        for event in ('before','registered','after'):
            if event=='registered': guard.append(dict(event='own_harness_registered',active_run=active,pidfd_retained=True,identity={'pid':num+101,'ppid':100,'start_ticks':1000+num,'exe':'/binary'},**common))
            else: guard.append(dict(event='sample',boundary=event,active_run=active,ok=True,violation_reasons=[],foreign_compute_pids=[],monitor_error=None,gpu={'uuid':gpu,'memory_used_mib':1},**common))
        guard.append(dict(event='run_one_end',guard_run_id=rid,status='ok',clean=True,candidate_id=name,workload_id='n4_b1',**common))
    run('default','calibration',1);records.append({'record_type':'calibration_complete','graph_repeats':1})
    run('default','baseline',2);run('identity','identity',3)
    for i in range(3): run('default_a','null_timing',4+2*i,0,i);run('default_b','null_timing',5+2*i,1,i)
    records.append({'record_type':'null_control','paired_ratios':[1,1,1],'median_ratio':1})
    records.append({'record_type':'candidate_manifest','candidate_id':cid,'request':req,'status':'identity_control','reason':'same_as_default_request'})
    records.append({'record_type':'manifest_complete','total':1,'eligible':0,'selected':0})
    records.append({'record_type':'controller_complete','status':'no_unique_candidates'})
    guard.append({'event':'sample','guard_id':gid,'gpu_uuid':gpu,'boundary':'final','ok':True,'violation_reasons':[],'foreign_compute_pids':[],'monitor_error':None,'gpu':{'uuid':gpu,'memory_used_mib':1}})
    guard.append({'event':'guard_complete','guard_id':gid,'gpu_uuid':gpu,'success':True,'clean':True,'script_exit_code':0,'sample_count':19})
    return records,guard,sha

def screened_fixture():
    records,guard,sha=fixture();protocol=records[0];protocol['batch']=2
    for row in records:
        if 'workload' not in row: continue
        row['workload']['batch']=2;row['workload_id']='n4_b2';row['command'][4]='2';row['result']['batch']=2
        row['result']['correctness']['checked_elements']=8;row['result']['single_forward_correctness']['checked_elements']=8
        row['stdout']=json.dumps(row['result'])+'\n'
    for e in guard:
        if e.get('event')=='run_one_end': e['workload_id']='n4_b2'
    manifest=next(x for x in records if x.get('record_type')=='manifest_complete');manifest.update(total=2,eligible=1,selected=1)
    req={'radix':[2,2],'grouped_batch':2,'block':[2,2,1]};cid=audit.candidate_id(req)
    records.insert(records.index(manifest),{'record_type':'candidate_manifest','candidate_id':cid,'request':req,'status':'pending_scheduler_resource_and_numerical_validation','reason':None})
    complete=records.pop();guard_complete=guard.pop();final_sample=guard.pop()
    template=next(x for x in records if x.get('phase')=='baseline');counter=20
    def add(name,phase,pattern='random',rnd=0,order=0,timing=False):
        nonlocal counter
        counter+=1;row=copy.deepcopy(template);rid=f'guard:{counter}';row.update(candidate_id=name,phase=phase,round=rnd,order=order,guard_run_id=rid)
        w=row['workload'];w.update(pattern=pattern,repeats=31 if timing else 1,warmup=5 if timing else 0,warmup_ms=1000 if timing else 0)
        cmd=row['command'];cmd[cmd.index('--pattern')+1]=pattern;cmd[cmd.index('--repeats')+1]=str(w['repeats']);cmd[cmd.index('--warmup')+1]=str(w['warmup']);cmd[cmd.index('--warmup-ms')+1]=str(w['warmup_ms'])
        r=row['result'];r['pattern']=pattern
        if pattern!='random':
            r['correctness']['reference']='analytic_pattern_reference_with_independent_CPU_FFT_crosscheck';r['single_forward_correctness']['reference']='analytic_pattern_reference_with_independent_CPU_FFT_crosscheck'
        r['exec_ms_samples']=[2.0]*w['repeats'];r['raw_interval_ms_samples']=[2.0]*w['repeats']
        if name==cid:
            row['direct_environment']={'VKFFT_DIRECT_RADIX':'2,2','VKFFT_DIRECT_BATCH':'2','VKFFT_DIRECT_BLOCK':'2,2,1'}
            r['structural_features']['kernels'][0].update(groupedBatch=2,localSize=[2,2,1]);r['direct_resource_validation']['threads']=4;r['precompile_kernels'][0]['canonical_descriptor']='candidate'
        row['stdout']=json.dumps(r)+'\n';records.append(row)
        active={'guard_run_id':rid,'candidate_id':name,'phase':phase,'round':rnd,'order':order,'expected_exe':'/binary'}; common={'guard_id':'guard','gpu_uuid':'GPU-example'}
        guard.append(dict(event='run_one_start',active_run=active,**common))
        guard.append(dict(event='sample',boundary='before',active_run=active,ok=True,violation_reasons=[],foreign_compute_pids=[],monitor_error=None,gpu={'uuid':'GPU-example','memory_used_mib':1},**common))
        guard.append(dict(event='own_harness_registered',active_run=active,pidfd_retained=True,identity={'pid':counter+101,'ppid':100,'start_ticks':1000+counter,'exe':'/binary'},**common))
        guard.append(dict(event='sample',boundary='after',active_run=active,ok=True,violation_reasons=[],foreign_compute_pids=[],monitor_error=None,gpu={'uuid':'GPU-example','memory_used_mib':1},**common))
        guard.append(dict(event='run_one_end',guard_run_id=rid,status='ok',clean=True,candidate_id=name,workload_id='n4_b2',**common))
    add(cid,'numerical')
    for pat in ('zero','impulse','constant','quarter-tone','random'):
        add('default','numerical_control',pat);add(cid,'numerical',pat)
    for i in range(3): add('default','screen_timing',rnd=i,order=0,timing=True);add(cid,'screen_timing',rnd=i,order=1,timing=True)
    records.append({'record_type':'candidate_disposition','candidate_id':cid,'status':'screened','paired_ratios':[1,1,1],'median_ratio':1})
    for i in range(10): add(cid,'confirmation_timing',rnd=i,order=0,timing=True);add('default','confirmation_timing',rnd=i,order=1,timing=True)
    complete.update(status='confirmed_measurements',winner=cid,confirmation_paired_ratios=[1]*10,confirmation_median_ratio=1);records.append(complete)
    guard.append(final_sample);guard_complete['sample_count']=sum(x['event']=='sample' for x in guard);guard.append(guard_complete)
    return records,guard,sha

class AuditTests(unittest.TestCase):
    def test_clean_complete(self):
        r,g,sha=fixture();out=audit.audit(r,g,[],sha)
        self.assertEqual(out['status'],'complete',out['errors'])
    def test_screen_confirmation_and_pair_mutations(self):
        r,g,sha=screened_fixture();out=audit.audit(r,g,[],sha)
        self.assertEqual(out['status'],'complete',out['errors']);self.assertEqual(len(out['confirmation']),10)
        bad=copy.deepcopy(r);next(x for x in bad if x.get('phase')=='confirmation_timing')['round']=99
        self.assertEqual(audit.audit(bad,g,[],sha)['status'],'invalid')
        bad=copy.deepcopy(r);next(x for x in bad if x.get('phase')=='numerical_control')['workload']['pattern']='wrong'
        self.assertEqual(audit.audit(bad,g,[],sha)['status'],'invalid')
        bad=copy.deepcopy(r);next(x for x in bad if x.get('record_type')=='candidate_disposition')['median_ratio']=0.5
        self.assertEqual(audit.audit(bad,g,[],sha)['status'],'invalid')

    def test_partial_never_concludes(self):
        r,g,sha=fixture();r.pop();g.pop();out=audit.audit(r,g,[],sha)
        self.assertEqual(out['status'],'incomplete');self.assertNotIn('速度比',audit.report(out))
    def test_candidate_replay_and_resource_guards(self):
        r,g,sha=fixture();protocol=r[0];row=next(x for x in r if x.get('phase')=='identity');req={'radix':[2,2],'grouped_batch':1,'block':[2,1,1]}
        audit.check_result(row,protocol,req=req)
        for mutate in ('shape','resource','nan','wrong_input'):
            bad=copy.deepcopy(row)
            if mutate=='shape': bad['result']['structural_features']['kernels'][0]['localSize']=[1,2,1]
            if mutate=='resource': bad['result']['direct_resource_validation']['function_thread_limit']=1
            if mutate=='nan': bad['result']['exec_ms_samples']=[float('nan')]
            if mutate=='wrong_input': bad['result']['input_unchanged']=False
            bad['stdout']=json.dumps(bad['result'])+'\n'
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError): audit.check_result(bad,protocol,req=req)

    def test_fails_closed_for_evidence_mutations(self):
        r,g,sha=fixture()
        mutations=[]
        bad=copy.deepcopy(g);next(x for x in bad if x['event']=='own_harness_registered')['pidfd_retained']=False;mutations.append((r,bad,[],sha))
        bad=copy.deepcopy(g);next(x for x in bad if x['event']=='sample')['foreign_compute_pids']=[55];mutations.append((r,bad,[],sha))
        bad=copy.deepcopy(r);next(x for x in bad if x.get('phase')=='baseline')['result']['correctness']['checked_elements']=3;mutations.append((bad,g,[],sha))
        bad=copy.deepcopy(r);next(x for x in bad if x.get('record_type')=='candidate_manifest')['request']['radix']=[4];mutations.append((bad,g,[],sha))
        mutations.append((r,g,[{'event':'incident'}],sha));mutations.append((r,g,[],'wrong'))
        for args in mutations:
            with self.subTest(args=args[2:]): self.assertEqual(audit.audit(*args)['status'],'invalid')
if __name__=='__main__':unittest.main()
