#!/usr/bin/env python3
"""Independent offline audit of a guarded joint-search JSONL and Chinese report."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path

PATTERNS={'zero','impulse','constant','quarter-tone','random'}
RADICES=(2,4,8,16,32)
TIMING={'null_timing','screen_timing','confirmation_timing'}

def load(path):
    rows=[]; errors=[]
    try:
        with path.open(encoding='utf8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                try: rows.append(json.loads(line))
                except (ValueError,TypeError): errors.append(f'{path.name}:{line_no} JSON不完整或无效')
    except OSError as e: errors.append(f'{path}: {e}')
    return rows,errors

def sig(r):
    k=r['precompile_kernels'][0]
    return json.dumps([k['canonical_descriptor'],k['source_sha256'],k['compiler_input'],k['actual_compiler_options']],sort_keys=True)

def candidate_id(req):
    return hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def rebuild(base):
    """Reconstruct the stated restricted space independently of the controller."""
    k=base['structural_features']['kernels'][0]; d=base['precompile_kernels'][0]['raw_descriptor']
    n,batch=base['n'],base['batch']; shared=base['resolved_config']['sharedMemorySize']
    stages=k['stageRadix']; local=k['localSize']; swap=d['axisSwapped']; threads=local[1] if swap else local[0]
    if not 2<=n<=8192 or n&(n-1) or math.prod(stages)!=n or any(r not in RADICES for r in stages): raise ValueError('基线不在受限空间')
    if d['fftDim']!=n or d['complexSize']!=8 or swap not in (0,1) or local[2]!=1 or (local[0] if swap else local[1])!=k['groupedBatch']:
        raise ValueError('基线布局不满足枚举前提')
    def tails(rem):
        if rem==0:
            yield ()
            return
        for r in RADICES:
            exp=r.bit_length()-1
            if exp<=rem:
                for t in tails(rem-exp): yield (r,)+t
    sequences=[tuple(stages)] if len(stages)<=2 else [(stages[0],)+t+(stages[-1],) for t in tails(sum(r.bit_length()-1 for r in stages[1:-1])) if len(t)<=30]
    result={}
    for seq in sequences:
        for group in range(1,min(batch,1024)+1):
            if batch%group or group&(group-1): continue
            block=[group,threads,1] if group>1 and group*n<shared//8 else [threads,group,1]
            req=dict(radix=list(seq),grouped_batch=group,block=block)
            active=seq[:-1] if d['registerBoost']>1 else seq; regs=d['registers_per_radix']
            if group*threads>base['resolved_config']['maxThreadsNum'] or group*(n//d['registerBoost'])>shared//8:
                reason='thread_or_shared_memory_precheck'
            elif not active or any(regs[r]<=0 or regs[r]%r for r in active): reason='radix_register_table_missing_or_not_divisible'
            elif min(regs[r] for r in active)!=d['min_logical_registers_per_thread'] or max(regs[r] for r in active)!=d['logical_registers_per_thread']: reason='register_envelope_changed'
            else: reason=None
            result[candidate_id(req)]=(req,reason)
    return result

def check_result(row,protocol,base_sig=None,req=None,allow_invalid=False):
    r=row['result']; w=row['workload']; n=protocol['n']; batch=protocol['batch']; gpu=protocol['gpu']
    output=[json.loads(x) for x in row['stdout'].splitlines() if x.strip()]; assert output==[r]
    assert row.get('guard_clean') is True and row.get('guard_gpu_uuid')==gpu
    assert (w['n'],w['batch'],w['seed'],w['config'])==(n,batch,protocol['seed'],{})
    assert (r['n'],r['batch'],r['seed'])==(n,batch,protocol['seed']) and r['gpu']['uuid']==gpu and r['gpu']['cuda_visible_devices']==gpu
    cmd=row['command']; assert len(cmd)==23 and len(cmd[1:])%2==0
    opts=dict(zip(cmd[1::2],cmd[2::2])); assert len(opts)==11
    expected_opts={'--n':str(n),'--batch':str(batch),'--seed':str(protocol['seed']),'--timing-mode':'graph','--placement':'outofplace',
                   '--graph-repeats':str(w['graph_repeats']),'--warmup':str(w['warmup']),'--warmup-ms':str(w['warmup_ms']),
                   '--repeats':str(w['repeats']),'--pattern':w['pattern'],'--study-mode':'execute'}
    assert opts==expected_opts and r['requested_config']=={'makeForwardPlanOnly':1}
    assert (r['backend'],r['precision'],r['forward_direction'],r['placement'],r['study_mode'],r['timing_mode'],r['pattern'])==('CUDA_NVRTC','complex_fp32',-1,'outofplace','execute','graph',w['pattern'])
    assert r['vkfft_result_code']==0 and r['descriptor_validation']['all_match'] is True
    assert r['validation_output_initialization']=='0xff_bytes_fp32_nan_before_single_and_final_timed_forward'
    assert r['single_forward_input_unchanged'] is True and r['structural_features']['numAxisUploads']==1 and len(r['structural_features']['kernels'])==1
    pc=r['precompile_kernels']; assert len(pc)==1 and pc[0]['compiler_options_match'] is True
    k=r['structural_features']['kernels'][0]
    if req:
        assert row['direct_environment']=={'VKFFT_DIRECT_RADIX':','.join(map(str,req['radix'])),'VKFFT_DIRECT_BATCH':str(req['grouped_batch']),'VKFFT_DIRECT_BLOCK':','.join(map(str,req['block']))}
        assert (k['stageRadix'],k['groupedBatch'],k['localSize'])==(req['radix'],req['grouped_batch'],req['block'])
        lim=r['direct_resource_validation']; assert 0<=lim['dynamic_shared_bytes']<=lim['function_dynamic_limit_bytes'] and 0<=lim['static_shared_bytes'] and lim['dynamic_shared_bytes']+lim['static_shared_bytes']<=lim['device_shared_limit_bytes'] and lim['threads']<=lim['function_thread_limit'] and lim['threads']==math.prod(req['block'])
    else: assert not row['direct_environment']
    if base_sig is not None: assert sig(r)==base_sig
    def numerical(x):
        assert x['checked_elements']==n*batch and isinstance(x['passed'],bool) and isinstance(x['nonfinite_count'],int) and isinstance(x['failing_batches'],int)
        assert x['reference']==('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' if w['pattern']=='random' else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck') and x['criterion']=='criterion_v2' and math.isclose(x['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and math.isclose(x['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12)
        return x['passed'] and x['nonfinite_count']==0 and x['failing_batches']==0
    single=numerical(r['single_forward_correctness'])
    if allow_invalid:
        assert row['returncode']==1 and row['phase']=='numerical'
        if r['status']=='error':
            assert r['failure_phase']=='single_forward_validation' and r['error']=='Single out-of-place forward failed immutable-input or numerical validation' and not single
        else:
            assert r['status']=='correctness_failed' and r['input_unchanged'] is True and single and not numerical(r['correctness'])
        return
    assert row['status']=='ok' and row['returncode']==0 and r['status']=='ok' and r['input_unchanged'] is True and single and numerical(r['correctness'])
    assert r['graph_repeats']==w['graph_repeats'] and r['graph_kernel_count']==w['graph_repeats'] and r['graph_event_record_count']==2
    expected=w['repeats']; samples=r['exec_ms_samples']; raw=r['raw_interval_ms_samples']
    assert len(samples)==expected and len(raw)==expected and all(math.isfinite(x) and x>0 for x in samples+raw)
    assert all(math.isclose(x,y/w['graph_repeats'],rel_tol=2e-6,abs_tol=1e-9) for x,y in zip(samples,raw))
    if row['phase'] in TIMING: assert expected==31 and min(raw)>=0.5 and w['warmup']>=5 and w['warmup_ms']>=1000

def audit(records,guard,incidents,exe_sha256=None):
    errors=[]; warnings=[]
    def need(condition,msg):
        if not condition: errors.append(msg)
    protocol_rows=[x for x in records if x.get('record_type')=='protocol']; completed=[x for x in records if x.get('record_type')=='controller_complete']
    starts=[x for x in guard if x.get('event')=='guard_start']; ends=[x for x in guard if x.get('event')=='guard_complete']
    finished=len(completed)==1 and len(ends)==1
    if not finished: warnings.append('控制器或守卫尚未完整结束；不计算性能结论。')
    if not protocol_rows or not starts: return dict(status='incomplete',errors=errors,warnings=warnings,finished=False)
    need(len(protocol_rows)==1 and len(starts)==1,'协议或guard_start数量错误')
    p=protocol_rows[0]; start=starts[0]; gid=start.get('guard_id'); gpu=p.get('gpu')
    need(start.get('gpu_uuid')==gpu and start.get('own_signal_method')=='pidfd_with_rechecked_PPID_exe_starttime' and start.get('script','').replace('\\','/').endswith('/joint_controller.py'),'守卫GPU、目标脚本或pidfd方法不符')
    need(not incidents,'事故日志非空')
    need(not any(x.get('incident') or x.get('event') in ('incident','guard_abort','quarantined_run','own_harness_sigterm','termination_error') for x in guard),'守卫事故或中止')
    need(all(x.get('guard_id')==gid and x.get('gpu_uuid')==gpu for x in guard),'守卫ID/GPU不一致')
    samples=[x for x in guard if x.get('event')=='sample']
    need(all(x.get('ok') is True and not x.get('violation_reasons') and not x.get('foreign_compute_pids') and x.get('monitor_error') is None and x.get('gpu',{}).get('uuid')==gpu and x.get('gpu',{}).get('memory_used_mib',999999)<=2048 for x in samples),'守卫采样异常')
    if finished:
        need(records[-1] is completed[0], '控制器完成记录不是末条记录')
        need(ends[0].get('success') is True and ends[0].get('clean') is True and ends[0].get('script_exit_code')==0 and guard[-1] is ends[0],'守卫完成记录无效')
        need(ends[0].get('sample_count')==len(samples),'守卫采样计数不匹配')
        idle=[x for x in guard if x.get('event')=='idle_complete']; final_samples=[x for x in samples if x.get('boundary')=='final']
        need(len(idle)==1 and idle[0].get('clean') is True and idle[0].get('observed_idle_seconds',-1)>=start.get('idle_seconds',0) and len(final_samples)==1,'守卫空闲或最终采样不完整')
    if not finished:
        counts=collections.Counter(x.get('status','unknown') for x in records if x.get('record_type')=='candidate_manifest')
        return dict(status='invalid' if errors else 'incomplete',errors=errors,warnings=warnings,finished=False,counts=dict(counts))
    need(exe_sha256 is not None and exe_sha256==p.get('exe_sha256'),'缺少实际运行二进制或SHA-256不一致')
    executions=[x for x in records if 'phase' in x and 'workload' in x]
    run_ids=[x.get('guard_run_id') for x in executions]
    need(None not in run_ids and len(set(run_ids))==len(run_ids),'执行guard_run_id缺失或重复')
    events=collections.defaultdict(list)
    for e in guard:
        rid=e.get('guard_run_id') or (e.get('active_run') or {}).get('guard_run_id')
        if rid: events[rid].append(e)
    for row in executions:
        rid=row.get('guard_run_id'); ev=events.get(rid,[])
        start_ev=[x for x in ev if x.get('event')=='run_one_start']; end_ev=[x for x in ev if x.get('event')=='run_one_end']; reg=[x for x in ev if x.get('event')=='own_harness_registered']
        before=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='before']; after=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='after']
        ok=(row.get('guard_id')==gid and len(start_ev)==len(end_ev)==len(reg)==len(before)==len(after)==1)
        if ok:
            ident=reg[0].get('identity',{}); active=start_ev[0].get('active_run',{})
            ok=(reg[0].get('pidfd_retained') is True and ident.get('ppid')==start.get('parent_pid') and ident.get('pid',0)>0 and ident.get('start_ticks',0)>0 and ident.get('exe')==active.get('expected_exe')==row.get('command',[None])[0] and end_ev[0].get('clean') is True and end_ev[0].get('status')==row.get('status') and end_ev[0].get('candidate_id')==row.get('candidate_id') and end_ev[0].get('workload_id')==row.get('workload_id') and active.get('candidate_id')==row.get('candidate_id') and active.get('phase')==row.get('phase') and active.get('round')==row.get('round') and active.get('order')==row.get('order') and ev.index(start_ev[0])<ev.index(before[0])<ev.index(reg[0])<ev.index(after[0])<ev.index(end_ev[0]))
        need(ok,f'执行{rid}的pidfd/守卫边界不完整')
    if finished: need(set(events)==set(run_ids),'守卫与结果的执行集合不一致')
    manifests=[x for x in records if x.get('record_type')=='candidate_manifest']; dispos=[x for x in records if x.get('record_type')=='candidate_disposition']
    by_id={x.get('candidate_id'):x for x in manifests}; disposition=collections.defaultdict(list)
    for d in dispos: disposition[d.get('candidate_id')].append(d)
    need(len(by_id)==len(manifests),'候选ID重复')
    baselines=[x for x in executions if x['phase']=='baseline']; identities=[x for x in executions if x['phase']=='identity']
    if len(baselines)!=1 or len(identities)!=1:
        if finished: errors.append('缺少唯一默认基线/身份重放')
        return dict(status='invalid' if errors else 'incomplete',errors=errors,warnings=warnings,finished=False)
    base=baselines[0]['result']; base_sig=sig(base);
    need(all(x.get('result',{}).get('gpu',{}).get('name')==base.get('gpu',{}).get('name') for x in executions),'结果GPU型号不一致'); bk=base['structural_features']['kernels'][0]
    identity_req={'radix':bk['stageRadix'],'grouped_batch':bk['groupedBatch'],'block':bk['localSize']}
    try:
        expected=rebuild(base)
        if finished: need(set(by_id)==set(expected),'manifest与独立重建的完整范围不一致')
        complete_manifest=[x for x in records if x.get('record_type')=='manifest_complete']
        eligible=sum(1 for req,reason in expected.values() if reason is None and req!=identity_req)
        selected=sum(m['status']=='pending_scheduler_resource_and_numerical_validation' for m in manifests)
        need(len(complete_manifest)==1 and complete_manifest[0]['total']==len(expected) and complete_manifest[0]['eligible']==eligible and complete_manifest[0]['selected']==selected, 'manifest计数不匹配')
        calibrations=[x for x in executions if x['phase']=='calibration']
        cal_done=[x for x in records if x.get('record_type')=='calibration_complete']
        need(bool(calibrations) and len(cal_done)==1 and min(calibrations[-1]['result']['raw_interval_ms_samples'])>=2.0 and cal_done[0]['graph_repeats']==calibrations[-1]['workload']['graph_repeats'] and all(x['workload']['graph_repeats']==cal_done[0]['graph_repeats'] for x in executions if x['phase']!='calibration'),'计时校准不完整')
        for ident,m in by_id.items():
            req,reason=expected[ident]; need(m['request']==req and candidate_id(req)==ident,'候选请求或哈希不匹配')
            if req==identity_req: need(m['status']=='identity_control','默认身份项标记错误')
            elif reason: need(m['status']=='rejected_precheck' and m['reason']==reason,'预筛拒绝原因错误')
            elif m['status']=='not_selected': need(p['limit']>0,'全量协议出现未选候选')
            else: need(m['status']=='pending_scheduler_resource_and_numerical_validation','候选初始状态错误')
        check_result(baselines[0],p,base_sig);check_result(identities[0],p,base_sig,identity_req)
    except (KeyError,TypeError,ValueError,AssertionError,IndexError) as e: errors.append('基线/身份或manifest审计失败: '+str(e))
    for ident,m in by_id.items():
        ds=disposition.get(ident,[]); active=m.get('status')=='pending_scheduler_resource_and_numerical_validation'
        if finished: need(len(ds)==(1 if active else 0),f'候选{ident}最终状态缺失或重复')
        if ds: need(ds[0].get('status') in ('candidate_rejected','numeric_invalid','duplicate_actual_structure','screened'),f'候选{ident}状态无效')
        rows=[x for x in executions if x['candidate_id']==ident]
        if not active: need(not rows,f'未入选或预筛拒绝候选{ident}存在执行')
        if ds and ds[0]['status'] in ('numeric_invalid','candidate_rejected'):
            failed=[x for x in rows if x.get('result',{}).get('status') in ('error','correctness_failed')]
            need(len(failed)==1,f'候选{ident}缺少唯一失败证据')
    need(not(set(disposition)-set(by_id)),'存在manifest以外的候选结果')
    # Check every saved benchmark row. Explicit pre-launch rejections have no descriptor.
    for row in executions:
        ident=row['candidate_id']; phase=row['phase']; m=by_id.get(ident); ds=disposition.get(ident,[])
        state=ds[0]['status'] if ds else None
        try:
            if ident in ('default','default_a','default_b'): check_result(row,p,base_sig)
            elif ident=='identity': check_result(row,p,base_sig,identity_req)
            elif m:
                if state=='candidate_rejected' and row['result'].get('status')=='error' and row['result'].get('failure_phase')=='initialize':
                    rr=row['result']; assert [json.loads(x) for x in row['stdout'].splitlines() if x.strip()]==[rr] and row['phase']=='numerical'; req=m['request']; env={'VKFFT_DIRECT_RADIX':','.join(map(str,req['radix'])),'VKFFT_DIRECT_BATCH':str(req['grouped_batch']),'VKFFT_DIRECT_BLOCK':','.join(map(str,req['block']))}
                    assert row['guard_clean'] is True and row['guard_gpu_uuid']==gpu and row['returncode']==1 and row['direct_environment']==env
                    assert rr['gpu']['uuid']==gpu and (rr['n'],rr['batch'],rr['seed'])==(p['n'],p['batch'],p['seed']) and rr['requested_config']=={'makeForwardPlanOnly':1}
                    assert rr['error'] in ('initializeVkFFT failed with code 3002','direct candidate exceeds compiled function or device resource limit') and ds[0]['reason']==rr['error']
                elif state=='numeric_invalid' and row['result'].get('status') in ('error','correctness_failed'): check_result(row,p,req=m['request'],allow_invalid=True)
                else: check_result(row,p,req=m['request'])
            else: raise ValueError('执行候选不在manifest')
        except (KeyError,TypeError,ValueError,AssertionError,IndexError) as e: errors.append(f'执行{ident}/{phase}校验失败: {e}')
    for i,row in enumerate(executions):
        d=disposition.get(row['candidate_id'],[])
        if d and d[0]['status'] in ('numeric_invalid','candidate_rejected') and row.get('result',{}).get('status') in ('error','correctness_failed'):
            need(i+1<len(executions) and executions[i+1]['phase']=='health_check' and executions[i+1]['candidate_id']=='default',f'候选{row["candidate_id"]}失败后缺少默认健康检查')
    # Successful replays of a request must keep the same actual structure.
    candidate_sigs={}
    for ident in by_id:
        good=[x for x in executions if x['candidate_id']==ident and (x.get('result',{}).get('status')=='ok' or (disposition.get(ident) and disposition[ident][0]['status']=='numeric_invalid' and x.get('result',{}).get('precompile_kernels')))]
        if good:
            signatures={sig(x['result']) for x in good}; need(len(signatures)==1,f'候选{ident}的实际结构漂移')
            candidate_sigs[ident]=next(iter(signatures))
    for d in dispos:
        if d['status']=='duplicate_actual_structure':
            target=d['duplicate_of']; expected_sig=base_sig if target=='default' else candidate_sigs.get(target)
            need(d['candidate_id'] in candidate_sigs and expected_sig is not None and candidate_sigs[d['candidate_id']]==expected_sig,f'候选{d["candidate_id"]}去重证据不匹配')
    if finished: need(len([x for x in executions if x['phase']=='screen_timing'])==2*p['screen_rounds']*len([d for d in dispos if d['status']=='screened']),'筛选计时记录总数不一致')
    def pairs(phase,expected_rounds,candidate=None):
        rows=[r for r in executions if r['phase']==phase]; vals=[]
        if len(rows)!=2*expected_rounds: errors.append(f'{phase}配对数量错误');return vals
        for i in range(expected_rounds):
            pair=rows[2*i:2*i+2]; a,b=pair
            try:
                assert {a['order'],b['order']}=={0,1} and a['round']==b['round']==i
                lookup={x['candidate_id']:x for x in pair}
                if phase=='null_timing': num,den=lookup['default_a'],lookup['default_b']
                else: num,den=lookup[candidate],lookup['default']
                vals.append(statistics.median(num['result']['exec_ms_samples'])/statistics.median(den['result']['exec_ms_samples']))
            except (KeyError,AssertionError,TypeError,ZeroDivisionError): errors.append(f'{phase}第{i}轮配对错误')
        return vals
    screened=[d for d in dispos if d['status']=='screened']; summary=[]
    # Screen rows from different candidates are consecutive blocks, not grouped globally.
    for d in screened:
        ident=d['candidate_id']; candrows=[x for x in executions if x['candidate_id']==ident]
        patterns={x['workload']['pattern'] for x in candrows if x['phase']=='numerical' and x['result'].get('status')=='ok'}
        numeric=[x for x in candrows if x['phase']=='numerical' and x['result'].get('status')=='ok']
        need(patterns==PATTERNS and len(numeric)==6 and sum(x['workload']['pattern']=='random' for x in numeric)==2,f'候选{ident}缺少完整数值模式')
        for row in [x for x in candrows if x['phase']=='numerical' and x['workload']['pattern'] in PATTERNS][1:]:
            index=executions.index(row); prev=executions[index-1] if index else {}
            need(prev.get('phase')=='numerical_control' and prev.get('candidate_id')=='default' and prev.get('workload',{}).get('pattern')==row['workload']['pattern'],f'候选{ident}缺少数值默认对照')
        rows=[x for x in executions if x['phase']=='screen_timing']; idx=[i for i,x in enumerate(rows) if x['candidate_id']==ident]
        ratios=[]
        for i in idx:
            j=i+1 if rows[i]['order']==0 else i-1
            try:
                c,control=rows[i],rows[j]; assert control['candidate_id']=='default' and control['round']==c['round'] and {control['order'],c['order']}=={0,1}
                ratios.append((c['round'],statistics.median(c['result']['exec_ms_samples'])/statistics.median(control['result']['exec_ms_samples'])))
            except (IndexError,AssertionError,KeyError,ZeroDivisionError): errors.append(f'候选{ident}筛选配对错误')
        ratios=[v for _,v in sorted(ratios)]
        need(len(ratios)==p['screen_rounds'] and all(math.isclose(x,y,rel_tol=1e-9) for x,y in zip(ratios,d['paired_ratios'])) and len(d['paired_ratios'])==len(ratios) and bool(ratios) and math.isclose(statistics.median(ratios),d['median_ratio'],rel_tol=1e-9),f'候选{ident}筛选统计不一致')
        summary.append(dict(candidate_id=ident,request=by_id[ident]['request'],screen_ratio=statistics.median(ratios) if ratios else None))
    if finished:
        null=pairs('null_timing',3)
        nullrec=[x for x in records if x.get('record_type')=='null_control']
        need(len(nullrec)==1 and len(null)==3 and all(math.isclose(x,y,rel_tol=1e-9) for x,y in zip(null,nullrec[0]['paired_ratios'])),'默认/默认负对照不一致')
        final=completed[0]; winner=final.get('winner')
        confirm=[]
        if screened:
            need(final.get('status')=='confirmed_measurements' and winner==min(screened,key=lambda d:(d['median_ratio'],d['candidate_id']))['candidate_id'],'筛选胜者不一致')
            confirm=pairs('confirmation_timing',p['confirm_rounds'],winner)
            need(len(confirm)==p['confirm_rounds'] and len(final.get('confirmation_paired_ratios',[]))==len(confirm) and all(math.isclose(x,y,rel_tol=1e-9) for x,y in zip(confirm,final['confirmation_paired_ratios'])) and bool(confirm) and math.isclose(statistics.median(confirm),final['confirmation_median_ratio'],rel_tol=1e-9),'独立确认统计不一致')
        else: need(final.get('status')=='no_unique_candidates','无候选完成状态错误')
    else: winner=None;confirm=[];null=[]
    counts=collections.Counter(m['status'] if m['status']!='pending_scheduler_resource_and_numerical_validation' else (disposition[m['candidate_id']][0]['status'] if disposition.get(m['candidate_id']) else 'pending') for m in manifests)
    outcomes=[dict(candidate_id=m['candidate_id'],request=m['request'],status=(m['status'] if m['status']!='pending_scheduler_resource_and_numerical_validation' else (disposition[m['candidate_id']][0]['status'] if disposition.get(m['candidate_id']) else 'pending')), reason=(m.get('reason') or (disposition[m['candidate_id']][0].get('reason') if disposition.get(m['candidate_id']) else None)), duplicate_of=(disposition[m['candidate_id']][0].get('duplicate_of') if disposition.get(m['candidate_id']) else None)) for m in manifests]
    return dict(status='invalid' if errors else ('complete' if finished else 'incomplete'),finished=finished,errors=errors,warnings=warnings,
                protocol=p,gpu_name=base.get('gpu',{}).get('name'),default_structure=identity_req,counts=dict(counts),outcomes=outcomes,screened=summary,winner=winner,confirmation=confirm,null_control=null)

def report(a):
    lines=['# 联合候选离线审计',f"状态：**{a['status']}**"]
    for e in a.get('errors',[]): lines.append('- 审计错误：'+e)
    for w in a.get('warnings',[]): lines.append('- '+w)
    if 'counts' in a:
        lines+=['','候选状态：'+json.dumps(a['counts'],ensure_ascii=False,sort_keys=True)]
    if a['status']!='complete':
        lines+=['','记录未完整通过审计，不报告性能结论。'];return '\n'.join(lines)+'\n'
    p=a['protocol'];default=a['default_structure'];lines+=['',f"工作负载：N={p['n']}，batch={p['batch']}，GPU {a['gpu_name']} ({p['gpu']})。",f"默认结构：radix {default['radix']}，batch分组 {default['grouped_batch']}，block {default['block']}。",f"运行二进制 SHA-256：{p['exe_sha256']}。",'',
        '| 候选ID（前12位） | radix | batch分组 | block | 筛选耗时比 |', '|---|---|---:|---|---:|']
    for x in sorted(a['screened'],key=lambda v:v['screen_ratio']):
        r=x['request'];lines.append(f"| {x['candidate_id'][:12]} | {r['radix']} | {r['grouped_batch']} | {r['block']} | {x['screen_ratio']:.6f} |")
    if a['winner']:
        ratio=statistics.median(a['confirmation']);lines+=['',f"筛选胜者 {a['winner']} 的独立确认：候选/默认耗时中位比 {ratio:.6f}，默认/候选速度比 {1/ratio:.6f}；{len(a['confirmation'])} 轮配对。",f"独立确认各轮候选/默认耗时比：{[round(x,6) for x in a['confirmation']]}。",f"默认/默认负对照的配对耗时比：{[round(x,6) for x in a['null_control']]}。"]
    lines+=['','完整候选状态：','', '| 候选ID | radix | batch分组 | block | 状态 | 原因或重复目标 |', '|---|---|---:|---|---|---|']
    for x in a['outcomes']:
        r=x['request']; detail=x['reason'] or x['duplicate_of'] or '—'
        lines.append(f"| {x['candidate_id']} | {r['radix']} | {r['grouped_batch']} | {r['block']} | {x['status']} | {detail} |")
    lines+=['','筛选阶段挑选最小值会产生赢家偏差；独立确认与筛选数据分开。配对中位数并不单独证明统计显著性。',
            '范围仅为默认计划固定首尾 radix 和 FFT 线程维度、受寄存器约束的内部 power-of-two 分解及 batch 的 power-of-two 分组，不是所有合法 VkFFT 计划或 CUDA block；预筛及设备拒绝的请求不视为有效计划。']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True,type=Path);p.add_argument('--guard',required=True,type=Path);p.add_argument('--incidents',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--exe',required=True,type=Path,help='实际运行的二进制副本，用于核对SHA-256')
    args=p.parse_args();records,e1=load(args.results);guard,e2=load(args.guard);incidents,e3=load(args.incidents)
    try: exe_hash=hashlib.sha256(args.exe.read_bytes()).hexdigest()
    except OSError as e: exe_hash=None; e3.append(f'二进制: {e}')
    try: result=audit(records,guard,incidents,exe_hash)
    except Exception as e: result=dict(status='invalid',errors=[f'审计异常: {type(e).__name__}: {e}'],warnings=[],finished=False)
    if e1+e2+e3: result['errors'].extend(e1+e2+e3);result['status']='invalid'
    args.output.write_text(report(result),encoding='utf8');print(result['status'])
    return 0 if result['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
