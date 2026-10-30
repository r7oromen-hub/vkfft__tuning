#!/usr/bin/env python3
"""Independent offline audit of guarded multi-address rotation evidence.

This program does not call CUDA. Graph node address/order checks are assertions
performed by the frozen harness on the device; only their recorded result and
the harness/build provenance can be checked offline, not the original graph.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import audit_budget
import audit_confirmation
import audit_guard
import audit_pairs
import audit_search
import audit_two_sessions


def read(path):
    with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return audit_search.sha(Path(path))
def near(a,b):return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12)


def verify_gate(plan, gate, first_gate, first_report, second_report, combined_report, evidence, hashes):
    errors=[]
    expected={'schema':'expanded-two-session-gate/v1','status':'pass','plan_sha256':hashes['plan'],
              'protocol_sha256':hashes['protocol'],'first_gate_sha256':hashes['first_gate'],
              'first_audit_sha256':hashes['first_audit'],'second_audit_sha256':hashes['second_audit'],
              'combined_audit_sha256':hashes['combined_audit'],'session_evidence_sha256':hashes['sessions'],
              'binaries_sha256':plan['sha256']}
    if any(gate.get(k)!=v for k,v in expected.items()):errors.append('two-session gate evidence, protocol, plan or binary hash mismatch')
    if first_gate.get('evidence_sha256')!=hashes['sessions']['first'] or first_gate.get('binaries_sha256')!=plan['sha256']:
        errors.append('first gate evidence or binary mismatch')
    sessions={}
    for name,report in (('first',first_report),('second',second_report)):
        files=evidence[name]
        sessions[name]=audit_confirmation.audit(read(files['rows']),read(files['guard']),read(files['incidents']),plan,hashes['plan'])
        if report.get('status')!='pass' or sessions[name].get('status')!='pass' or sessions[name]!=report.get(name+'_session',report):
            errors.append(name+' session audit differs from raw guarded evidence')
    header=next((x for x in read(evidence['second']['rows']) if x.get('record_type')=='confirmation_protocol'),{})
    combined=audit_two_sessions.combine(sessions['first'],sessions['second'],first_gate,header,
                                       hashes['first_gate'],hashes['plan'],hashes['first_audit'])
    if (combined.get('status')!='pass' or combined_report.get('status')!='pass' or
            combined_report.get('combined')!=combined or combined_report.get('first_session')!=sessions['first'] or
            combined_report.get('second_session')!=sessions['second']):
        errors.append('combined two-session audit differs from raw evidence')
    for name in ('first_budget','second_budget'):
        b=combined_report.get(name,{})
        if b.get('status')!='pass' or b.get('phase')!='confirmation' or b.get('inflight') is not None:
            errors.append(name+' audit incomplete')
    lookup={(x['n'],x['batch'],x['candidate']['candidate_id']):x['candidate'] for x in plan['workloads'] if x.get('candidate')}
    confirmed=[]
    for result in combined.get('candidates',[]):
        if result.get('status')!='confirmed_under_preregistered_rule':continue
        key=(result['n'],result['batch'],result['candidate_id']);c=lookup.get(key)
        if c is None:
            errors.append(f'{key}: confirmed candidate absent from plan');continue
        confirmed.append({'n':key[0],'batch':key[1],'candidate_id':key[2],
                          'request':c['request'],'graph_repeats':c['graph_repeats'],
                          'original_signature':c['original_signature'],'candidate_signature':c['candidate_signature']})
    confirmed.sort(key=lambda x:(x['n'],x['batch'],x['candidate_id']))
    if gate.get('confirmed')!=confirmed:errors.append('gate candidates differ from independently confirmed plan')
    return {'status':'fail' if errors else 'pass','errors':errors,'confirmed':confirmed}


def slot_report(r,n,batch,slots):
    try:
        for stage in ('rotation_initial_validation','rotation_final_validation'):
            value=r[stage];entries=value['slots']
            if (value['passed'] is not True or value['slots_checked']!=slots or value['elements_per_slot']!=n*batch or
                    len(entries)!=slots or [x['slot'] for x in entries]!=list(range(slots))):return False
            for entry in entries:
                c=entry['correctness']
                if not (entry['input_unchanged'] is True and c['passed'] is True and c['criterion']=='criterion_v2' and
                        c['checked_elements']==n*batch and c['reference']=='independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' and
                        c['nonfinite_count']==0 and c['failing_batches']==0 and
                        math.isclose(c['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and
                        math.isclose(c['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12)):return False
        return True
    except (KeyError,TypeError,ValueError,IndexError):return False


def check_execution(row, case, rotation, gpu, *, original, timing):
    errors=[]
    try:
        r,w=row['result'],row['workload'];n,batch=case['n'],case['batch'];q=case['request']
        env={} if original else {'VKFFT_EXPANDED_RADIX':','.join(map(str,q['radix'])),
                                 'VKFFT_EXPANDED_REGS':str(q['logical_registers']),'VKFFT_EXPANDED_BATCH':str(q['grouped_batch'])}
        if (row.get('returncode')!=0 or row.get('status')!='ok' or r['status']!='ok' or row.get('direct_environment')!=env or
                r['header_instrumentation'] is not (not original) or r['requested_config']!={'makeForwardPlanOnly':1} or
                r['gpu']['uuid']!=gpu or r['gpu']['cuda_visible_devices']!=gpu or
                (r['n'],r['batch'],r['seed'],r['pattern'])!=(n,batch,42,'random')):raise ValueError('process, binary, request, GPU or workload mismatch')
        if audit_search.source_signature(r,original)!=case['original_signature' if original else 'candidate_signature']:
            raise ValueError('plan, source or compiled resource signature differs from confirmed plan')
        slots,R,l2=rotation['slots'],rotation['graph_repeats'],rotation['l2_bytes'];per=n*batch*8
        minimal=2
        while minimal*per<=l2:minimal*=2
        if (not isinstance(slots,int) or slots!=minimal or rotation['total_bytes']!=2*slots*per or 2*slots*per>1280*1024*1024 or
                R<max(64,slots,case['graph_repeats']) or R>2048 or R&(R-1) or R%slots or r['rotation_pool_slots']!=slots or
                r['rotation_input_pool_bytes']!=slots*per or r['rotation_output_pool_bytes']!=slots*per or
                r['rotation_pool_total_bytes']!=2*slots*per or r['rotation_device_l2_bytes']!=l2 or
                r['rotation_mode']!='multi_address_graph_random_unique_scaled_inputs'):
            raise ValueError('pool does not meet recorded >L2/capacity requirements')
        if (r.get('rotation_graph_kernel_arguments_and_order_verified') is not True or
                r.get('rotation_distinct_input_representatives_verified') is not True or
                r['rotation_output_initialization']!='0xff_bytes_fp32_nan_before_initial_and_separate_post_timing_validation_graph' or
                r['graph_repeats']!=R or r['graph_kernel_count']!=R or r['graph_event_record_count']!=2 or r['graph_node_count']!=R+2 or
                r['timing_scope']!='multi_address_graph_forward_same_stream_no_transfers_or_validation_inside_event_interval'):
            raise ValueError('harness graph address/order or event assertion missing')
        if not slot_report(r,n,batch,slots):raise ValueError('initial/final complete slot validation failed')
        if any(not isinstance(r.get(k),(float,int)) or not math.isfinite(r[k]) or r[k]<=0
               for k in ('rotation_first_graph_interval_ms','rotation_final_validation_graph_interval_ms')):
            raise ValueError('validation graph interval invalid')
        if original:
            if r['init_trace']['available'] is not False or len(r['postinit_source_kernels'])!=1:raise ValueError('original source evidence missing')
        else:
            k=r['structural_features']['kernels'][0];p=r['precompile_kernels'][0];d=p['raw_descriptor'];swap=d['axisSwapped'];local=k['localSize'];lim=r['expanded_resource_validation']
            if (r['descriptor_validation']['all_match'] is not True or p['compiler_options_match'] is not True or
                    k['stageRadix']!=q['radix'] or k['groupedBatch']!=q['grouped_batch'] or d['logical_registers_per_thread']!=q['logical_registers'] or
                    d['registerBoost']!=1 or swap not in (0,1) or len(local)!=3 or local[2]!=1 or local[1 if swap else 0]!=n//q['logical_registers'] or
                    local[0 if swap else 1]!=q['grouped_batch'] or lim['threads']!=math.prod(local) or lim['threads']>lim['function_thread_limit'] or
                    not 0<=lim['dynamic_shared_bytes']<=lim['function_dynamic_limit_bytes'] or lim['static_shared_bytes']<0 or
                    lim['dynamic_shared_bytes']+lim['static_shared_bytes']>lim['device_shared_limit_bytes']):
                raise ValueError('candidate structure or compiled resource validation failed')
        values,raw=r['exec_ms_samples'],r['raw_interval_ms_samples'];count=31 if timing else 1
        if (w['graph_repeats']!=R or w['repeats']!=count or w['warmup']!=5 or w['warmup_ms']!=1000 or
                r['warmup_actual']<5 or r['warmup_wall_ms']<1000 or
                len(values)!=count or len(raw)!=count or not all(math.isfinite(x) and x>0 for x in values+raw) or
                (timing and min(raw)<0.5) or not all(math.isclose(x,y/R,rel_tol=2e-6,abs_tol=1e-9) for x,y in zip(values,raw))):
            raise ValueError('timing samples or actual warmup invalid')
    except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError) as exc:errors.append(str(exc))
    return errors


def audit_build(ledger, header, original_binary, expanded_binary, rotation_source, generated_benchmark,
                compile_commands, original_dependencies, original_include_prefix, generated_include_prefix,
                frozen_source_root, expanded_generated_root, original_header_root, instrumentation_root):
    errors=[]
    builds=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_build_complete']
    if len(builds)!=1:return {'status':'fail','errors':['missing or duplicate rotation build checkpoint']}
    build=builds[0];actual={'original':sha(original_binary),'expanded':sha(expanded_binary)}
    if build.get('sha256')!=actual or header.get('sha256')!=actual:errors.append('built rotation binary hashes mismatch')
    freeze=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_source_freeze']
    if len(freeze)!=1 or any(freeze[0].get(k)!=build.get(k) for k in ('source_hashes','input_tree_sha256')):
        errors.append('source freeze and build checkpoint differ')
    expected_sources={'prepare_rotation.py','rotation_pool.hpp','CMakeLists_rotation.txt','test_rotation_pool.cpp',
        'test_rotation_generation.py','test_rotation_controller.py','rotation_cpu/CMakeLists.txt','rotation_evidence.py','rotation_controller.py',
        'rotation_gpu_guard.py','rotation_runner.py','run_suite.py','search_controller.py','validation.py','prepare_headers.py',
        'enumerate_candidates.py','selection.py','manifest.py','identity.py','audit_confirmation.py','audit_two_sessions.py','audit_search.py',
        'audit_guard.py','audit_admission.py','audit_screen.py','audit_space.py','audit_pairs.py','audit_budget.py','budget.py','device_lock.py',
        'runner.py','../original_compare/original_compare_guard.py','../original_compare/original_compare_controller.py'}
    if set(build.get('source_hashes',{}))!=expected_sources:errors.append('frozen source inventory differs from required complete list')
    for name,digest in build.get('source_hashes',{}).items():
        if sha(Path(frozen_source_root)/name)!=digest:errors.append('frozen source mismatch: '+name)
    if not build.get('source_hashes'):errors.append('frozen source inventory missing')
    def tree_sha(path):
        root=Path(path);digest=hashlib.sha256()
        for item in sorted((x for x in root.rglob('*') if x.is_file()),key=lambda x:x.relative_to(root).as_posix()):
            rel=item.relative_to(root).as_posix().encode();digest.update(len(rel).to_bytes(4,'big'));digest.update(rel);digest.update(bytes.fromhex(sha(item)))
        return digest.hexdigest()
    inputs={'expanded_generated_tree':tree_sha(expanded_generated_root),'original_header_tree':tree_sha(original_header_root)}
    for suffix,name in (('reference','cpu_reference.hpp'),('source','vkfft_instrumented_benchmark.cpp'),('trace','trace.hpp'),('prepare','prepare_headers.py')):
        inputs['instrumentation_'+suffix]=sha(Path(instrumentation_root)/name)
    if inputs!=build.get('input_tree_sha256'):errors.append('frozen source input tree mismatch')
    root=Path(rotation_source);prov=load(root/'rotation_provenance.json')
    if (prov.get('expanded_benchmark_sha256')!=sha(generated_benchmark) or
            sha(generated_benchmark)!=sha(Path(expanded_generated_root)/'expanded_benchmark.cpp')):
        errors.append('rotation source does not match frozen expanded benchmark')
    for name in ('rotating_benchmark.cpp','rotation_pool.hpp','CMakeLists.txt'):
        if prov.get(name)!=sha(root/name):errors.append('rotation source provenance hash mismatch: '+name)
    commands=load(compile_commands)
    orig=[x['command'] for x in commands if 'vkfft_rotation_original.dir' in x.get('command','')]
    expanded=[x['command'] for x in commands if 'vkfft_rotation_expanded.dir' in x.get('command','')]
    if (len(orig)!=1 or len(expanded)!=1 or original_include_prefix not in orig[0] or generated_include_prefix not in orig[0] or
            orig[0].index(original_include_prefix)>orig[0].index(generated_include_prefix)):
        errors.append('original compiler include precedence not established')
    deps=Path(original_dependencies).read_text(encoding='utf8')
    if (original_include_prefix.rstrip('/')+'/vkFFT.h' not in deps or
            generated_include_prefix.rstrip('/')+'/vkFFT/' in deps or generated_include_prefix.rstrip('/')+'/vkFFT.h' in deps):
        errors.append('original compiler dependency list does not prove original header isolation')
    build_steps=[x for x in ledger if x.get('event')=='work_finished' and x.get('kind')=='rotation_build']
    if len(build_steps)!=7 or not all(x.get('status')=='ok' and x.get('exit_code')==0 for x in build_steps):
        errors.append('one or more rotation build or CPU test steps failed')
    return {'status':'fail' if errors else 'pass','errors':errors,'binary_sha256':actual,
            'rotation_provenance_sha256':sha(root/'rotation_provenance.json'),
            'compile_commands_sha256':sha(compile_commands),'original_dependencies_sha256':sha(original_dependencies),
            'scope':'hashes, generated-source provenance and compiler dependency isolation; no independent binary reproducible build'}


def audit_records(rows,guard,incidents,gate,plan,ledger,protocol,previous_rows,hot_rows):
    errors=[];counts=Counter();heads=[x for x in rows if x.get('record_type')=='rotation_protocol']
    ends=[x for x in rows if x.get('record_type')=='rotation_complete']
    if len(heads)!=1 or len(ends)!=1:
        return {'status':'uncovered','errors':['rotation protocol or terminal missing/duplicated'],
                'scope':'no claim about speed, cache hits or device validity'}
    h,t=heads[0],ends[0];session=h.get('session_id');gpu=h.get('gpu')
    if (not session or h.get('confirmed')!=gate.get('confirmed') or h.get('hot_binary_sha256')!=plan.get('sha256') or
            h.get('rounds')!=10 or t.get('session_id')!=session):errors.append('rotation protocol or terminal identity mismatch')
    clean=t.get('status') in ('completed','budget_exhausted')
    if not clean:errors.append('rotation did not end normally or at budget boundary')
    ga=audit_guard.audit(rows,guard,incidents,gpu);errors.extend('guard: '+x for x in ga['errors'])
    ba=audit_budget.audit(ledger,protocol,previous_rows+rows)
    errors.extend('budget: '+x for x in ba['errors'])
    if ba.get('phase')!='extension' or ba.get('inflight') is not None:errors.append('extension budget snapshot is not closed')
    cases={(x['n'],x['batch'],x['candidate_id']):x for x in gate.get('confirmed',[])}
    if len(cases)!=len(gate.get('confirmed',[])):errors.append('duplicate gate candidate')
    hot_l2={}
    for key in cases:
        observed={(x['result']['gpu']['uuid'],x['result']['gpu']['l2_cache_bytes']) for x in hot_rows
                  if x.get('record_type')=='execution' and x.get('candidate_id')=='original' and
                  (x.get('workload',{}).get('n'),x.get('workload',{}).get('batch'))==key[:2]}
        if len(observed)!=1:errors.append(f'{key}: confirmed hot GPU/L2 metadata missing or inconsistent')
        else:hot_l2[key]=next(iter(observed))
    executions={}
    for row in (x for x in rows if x.get('record_type')=='execution'):
        rid=row.get('guard_run_id');binary='original' if row.get('candidate_id')=='original' else 'expanded'
        try:
            if not rid or rid in executions or row['session_id']!=session or row['executable_sha256']!=h['sha256'][binary]:
                raise ValueError('run ID, session or executable mismatch')
            if [json.loads(x) for x in row['stdout'].splitlines() if x.strip()]!=[row['result']]:raise ValueError('stdout/result mismatch')
            executions[rid]=row
        except (KeyError,TypeError,ValueError) as exc:errors.append(f'{rid}: {exc}')
    calibrations={}
    for record in (x for x in rows if x.get('record_type')=='rotation_calibration_complete'):
        key=(record.get('n'),record.get('batch'),record.get('candidate_id'))
        try:
            if key not in cases or key in calibrations:raise ValueError('unknown or duplicate calibration')
            row=executions[record['guard_run_id']];rot=record['rotation']
            if (row.get('phase')!='rotation_calibration' or row.get('candidate_id')!='original' or
                    hot_l2.get(key)!=(gpu,rot['l2_bytes'])):raise ValueError('calibration identity or GPU/L2 mismatch')
            failure=check_execution(row,cases[key],rot,gpu,original=True,timing=False)
            if failure:raise ValueError('; '.join(failure))
            measured=min(row['result']['raw_interval_ms_samples'])
            if measured<2.0 or not near(record['raw_interval_ms'],measured):raise ValueError('calibration interval not met')
            calibrations[key]=rot
        except (KeyError,TypeError,ValueError) as exc:errors.append(f'{key}: {exc}')
    pairs=defaultdict(dict);rotation_by_case={}
    for record in (x for x in rows if x.get('record_type')=='rotation_pair'):
        key=(record.get('n'),record.get('batch'),record.get('candidate_id'));round_no=record.get('round')
        try:
            if key not in cases or record['session_id']!=session or round_no not in range(10) or round_no in pairs[key]:
                raise ValueError('unregistered candidate/session or duplicate round')
            rot=record['rotation'];old=rotation_by_case.setdefault(key,rot)
            if old!=rot:raise ValueError('rotation pool parameters changed across rounds')
            if key not in hot_l2 or hot_l2[key]!=(gpu,rot['l2_bytes']):raise ValueError('rotation GPU/L2 differs from confirmed sessions')
            original=executions[record['original_guard_run_id']];candidate=executions[record['candidate_guard_run_id']]
            seed=json.dumps([plan['selection_seed'],*key,'rotation',round_no,42],separators=(',',':')).encode()
            order=['original','candidate'];random.Random(int.from_bytes(hashlib.sha256(seed).digest(),'big')).shuffle(order)
            if (record['order']!=order or original.get('candidate_id')!='original' or candidate.get('candidate_id')!=key[2] or
                    original.get('order')!=order.index('original') or candidate.get('order')!=order.index('candidate') or
                    any(x.get('phase')!='rotation_timing' or x.get('round')!=round_no for x in (original,candidate))):
                raise ValueError('pair identity, phase or seeded order mismatch')
            for row,is_original in ((original,True),(candidate,False)):
                failures=check_execution(row,cases[key],rot,gpu,original=is_original,timing=True)
                if failures:raise ValueError('; '.join(failures))
            om=statistics.median(original['result']['exec_ms_samples']);cm=statistics.median(candidate['result']['exec_ms_samples'])
            if not (near(record['original_median_ms'],om) and near(record['candidate_median_ms'],cm) and near(record['candidate_over_original'],cm/om)):
                raise ValueError('pair summary differs from raw samples')
            pairs[key][round_no]=(cm,om);counts['verified_pairs']+=1
        except (KeyError,TypeError,ValueError,ZeroDivisionError,IndexError) as exc:errors.append(f'{key}/{round_no}: {exc}')
    completed=[];disposed=set();uncovered=[]
    for d in (x for x in rows if x.get('record_type')=='rotation_disposition'):
        key=(d.get('n'),d.get('batch'),d.get('candidate_id'))
        if key not in cases or key in disposed:errors.append(f'{key}: unregistered or duplicate disposition');continue
        disposed.add(key);status=d.get('status')
        if status=='complete':
            try:
                if set(pairs[key])!=set(range(10)):raise ValueError('ten unique rotation rounds missing')
                if calibrations.get(key)!=rotation_by_case[key]:raise ValueError('matching completed calibration missing')
                values=[pairs[key][i] for i in range(10)];ratios=[x/y for x,y in values]
                if (len(d['paired_ratios'])!=10 or not all(near(x,y) for x,y in zip(ratios,d['paired_ratios'])) or
                        not near(statistics.median(ratios),d['median_ratio']) or d['rotation']!=rotation_by_case[key]):
                    raise ValueError('rotation disposition differs from raw pairs')
                completed.append({'n':key[0],'batch':key[1],'candidate_id':key[2],
                                  'rotation':rotation_by_case[key],**audit_pairs.ten_round_session(values)})
                counts['complete']+=1
            except (KeyError,TypeError,ValueError) as exc:errors.append(f'{key}: {exc}')
        elif isinstance(status,str) and status.startswith('uncovered'):
            uncovered.append({'n':key[0],'batch':key[1],'candidate_id':key[2],'status':status,'reason':d.get('reason')});counts[status]+=1
        else:errors.append(f'{key}: unknown rotation disposition')
    missing=set(cases)-disposed
    reported={(x['n'],x['batch'],x['candidate_id']) for x in t.get('incomplete_or_not_started',[])}
    if reported!=missing:errors.append('terminal incomplete/not-started candidate list mismatch')
    return {'status':'fail' if errors else 'pass','errors':errors,'counts':dict(counts),'complete':completed,'uncovered':uncovered,
            'incomplete_or_not_started':[{'n':x[0],'batch':x[1],'candidate_id':x[2]} for x in sorted(missing)],'budget':ba,
            'scope':'L2-exceeding cycling only; not a cache-miss measurement, proof of cold data, or end-to-end performance',
            'graph_evidence':'node address/order verification was performed and logged by the frozen harness; offline audit cannot reread the CUDA graph',
            'statistics':'within-session paired t intervals do not account for search selection or establish cross-device generalization'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','protocol','gate','first-gate','first-audit','second-audit','combined-audit',
                 'first-rows','first-guard','first-incidents','first-budget',
                 'second-rows','second-guard','second-incidents','second-budget','budget'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--prior-rows',nargs='+',required=True,type=Path,help='all admissions and search rows preceding both confirmation sessions')
    for name in ('rows','guard','incidents','original-binary','expanded-binary','rotation-source','generated-benchmark',
                 'compile-commands','original-dependencies','frozen-source-root','expanded-generated-root','original-header-root','instrumentation-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--original-include-prefix');p.add_argument('--generated-include-prefix')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    plan=load(a.plan);protocol=load(a.protocol);gate=load(a.gate);first_gate=load(a.first_gate)
    evidence={name:{key:getattr(a,name+'_'+key) for key in ('rows','guard','incidents','budget')} for name in ('first','second')}
    hashes={'plan':sha(a.plan),'protocol':sha(a.protocol),'first_gate':sha(a.first_gate),'first_audit':sha(a.first_audit),
            'second_audit':sha(a.second_audit),'combined_audit':sha(a.combined_audit),
            'sessions':{name:{key:sha(value) for key,value in files.items()} for name,files in evidence.items()}}
    combined=load(a.combined_audit);prior=[row for path in a.prior_rows for row in read(path)]
    frows=read(a.first_rows);srows=read(a.second_rows)
    gate_audit=verify_gate(plan,gate,first_gate,load(a.first_audit),load(a.second_audit),combined,evidence,hashes)
    for name,files in evidence.items():
        replay=audit_budget.audit(read(files['budget']),protocol,prior+frows+(srows if name=='second' else []))
        if replay!=combined.get(name+'_budget'):
            gate_audit['errors'].append(name+' budget differs from independently replayed ledger');gate_audit['status']='fail'
    ledger=read(a.budget);summary={'gate':gate_audit,'extension_budget_sha256':sha(a.budget)}
    required=('rows','guard','incidents','original_binary','expanded_binary','rotation_source','generated_benchmark',
              'compile_commands','original_dependencies','frozen_source_root','expanded_generated_root','original_header_root',
              'instrumentation_root','original_include_prefix','generated_include_prefix')
    if gate_audit['status']!='pass':summary.update(status='fail',reason='confirmation gate invalid')
    elif not gate_audit['confirmed']:
        summary.update(status='uncovered',reason='no candidates passed the preregistered two-session rule')
    elif not all(getattr(a,k) is not None for k in required):
        summary.update(status='uncovered',reason='rotation data or build evidence not supplied; no performance conclusion')
    else:
        rows=read(a.rows);headers=[x for x in rows if x.get('record_type')=='rotation_protocol']
        if len(headers)!=1:
            summary.update(status='uncovered',reason='rotation session did not establish a unique protocol')
        else:
            header=headers[0]
            result=audit_records(rows,read(a.guard),read(a.incidents),gate,plan,ledger,protocol,prior+frows+srows,frows+srows)
            build=audit_build(ledger,header,a.original_binary,a.expanded_binary,a.rotation_source,a.generated_benchmark,
                              a.compile_commands,a.original_dependencies,a.original_include_prefix,a.generated_include_prefix,
                              a.frozen_source_root,a.expanded_generated_root,a.original_header_root,a.instrumentation_root)
            if header.get('plan_sha256')!=hashes['plan'] or header.get('two_session_gate_sha256')!=sha(a.gate):
                result['errors'].append('rotation header is not bound to supplied plan and gate');result['status']='fail'
            summary.update(status='pass' if result['status']=='pass' and build['status']=='pass' else 'fail',records=result,build=build)
    encoded=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf8') as out:out.write(encoded)
    print(encoded,end='')
    return 0 if summary['status'] in ('pass','uncovered') else 1


if __name__=='__main__':raise SystemExit(main())
