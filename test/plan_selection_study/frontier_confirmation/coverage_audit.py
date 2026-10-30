#!/usr/bin/env python3
"""Audit all requested frontier screen coverage, including a clean budget boundary."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evidence
base, validation = evidence.frontier, evidence.validation
frontier = base
search_controller, comparison, audit_guard = evidence.search_controller, evidence.comparison, evidence.audit_guard


def need(value, message):
    if not value: raise ValueError(message)


def power_two(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0 and not value & (value - 1)


def domain_cardinality(n, batch):
    """Independent integer-composition count; does not import the generator."""
    need(power_two(n) and 2 <= n <= 8192 and type(batch) is int and 1 <= batch <= 65536,
         'invalid counting domain')
    exponent=n.bit_length()-1
    def compositions(max_step):
        counts=[1]+[0]*exponent
        for total in range(1,exponent+1):
            counts[total]=sum(counts[total-step] for step in range(1,min(total,max_step)+1))
        return counts[exponent]
    groups=1; value=batch
    while value%2==0:
        groups+=1;value//=2
    total=compositions(5)*8*groups
    supported=sum(compositions(min(k,5)) for k in range(1,min(exponent,8)+1))*groups
    return {'enumerated':total,'static_supported':supported,'static_rejected':total-supported}


def read_manifests(paths, contract):
    need(sorted(base.sha(p) for p in paths) == sorted(contract['manifests'].values()), 'manifest hash mismatch')
    totals = Counter(); selected = {}; workloads = set()
    for path in paths:
        rows = base.read_log(path)
        need(bool(rows) and rows[0].get('schema') == 'frontier-manifest/v1', 'manifest schema mismatch')
        header = rows[0]; n,batch = header['workload']['n'],header['workload']['batch']
        need(power_two(n) and 2 <= n <= 8192 and isinstance(batch,int) and not isinstance(batch,bool) and 1 <= batch <= 65536 and
             (n,batch) not in workloads, 'manifest workload invalid or duplicated')
        workloads.add((n,batch)); seen = set(); indices = []; local = Counter()
        for row in rows[1:]:
            req = row['request']
            need(isinstance(req,dict) and set(req)=={'radix','logical_registers','grouped_batch'}, 'request schema mismatch')
            radix,r,group = req['radix'],req['logical_registers'],req['grouped_batch']
            payload=json.dumps(req,sort_keys=True,separators=(',',':')).encode()
            cid=hashlib.sha256(payload).hexdigest()
            need(cid == row['candidate_id'] and cid not in seen, 'manifest candidate ID mismatch or duplicate')
            seen.add(cid)
            need(isinstance(radix,list) and bool(radix) and all(isinstance(x,int) and not isinstance(x,bool) and
                 x in (2,4,8,16,32) for x in radix) and math.prod(radix) == n and
                 isinstance(r,int) and not isinstance(r,bool) and r in (2,4,8,16,32,64,128,256) and
                 power_two(group) and batch % group == 0, 'manifest request outside prototype')
            reason = ('registers_exceed_fft_length' if r > n else 'radix_exceeds_registers' if max(radix) > r else None)
            choice = row['selection']; status = row['status']
            need(row.get('reason') == reason, 'incorrect static rejection reason')
            if reason:
                need(choice == status == 'rejected_static', 'static rejection mislabeled')
                local['static_rejected'] += 1
            else:
                need(status == 'pending_scheduler_and_device_validation' and choice in ('selected','not_selected'),
                     'static-valid request mislabeled')
                local[choice] += 1
                if choice == 'selected':
                    idx=row['selection_index']; indices.append(idx)
                    selected[(n,batch,cid)] = {'n':n,'batch':batch,'candidate_id':cid,'request':req,
                                               'selection_index':idx}
        need(sorted(indices) == list(range(len(indices))), 'manifest selected indices are not contiguous')
        need(header.get('enumerated_count') == len(seen) and header.get('selected_count') == local['selected'] and
             header.get('static_rejected_count') == local['static_rejected'] and
             header.get('pending_count') == local['selected']+local['not_selected'], 'manifest count mismatch')
        mathematical=domain_cardinality(n,batch)
        totals['independent_domain_total']+=mathematical['enumerated']
        totals['independent_static_supported']+=mathematical['static_supported']
        if (len(seen)==mathematical['enumerated'] and
            local['selected']+local['not_selected']==mathematical['static_supported']):
            totals['complete_inventory_workloads']+=1
        totals.update(local); totals['manifest_total'] += len(seen)
    return workloads, selected, totals


def audit_coverage(rows, guards, incidents, ledger, manifests, gpu, binaries):
    """Return a fail-closed partition of one clean guarded screen session.

    All paths in manifests and binaries are read and hashed. A budget-denied
    partial final candidate is unmeasured, not numerically invalid.
    """
    try:
        need(rows and guards and ledger, 'missing rows, guard or ledger')
        contract=rows[0]['contract']
        need(all(r.get('contract') == contract for r in rows) and contract['gpu'] == gpu, 'contract or GPU mismatch')
        need(all(r.get('record_type') in ('session_start','execution','workload_status','candidate_start',
                                        'screen_pair','candidate_disposition','budget_exhausted') for r in rows),
             'unknown screen record type')
        commit=contract['execution_commit']
        need(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),
             'invalid execution commit identity')
        need(set(binaries) == {'original','expanded'} and
             {k:base.sha(v) for k,v in binaries.items()} == contract['binaries'], 'binary hash mismatch')
        ga=audit_guard.audit(rows,guards,incidents,gpu)
        need(ga['status'] == 'pass', 'guard audit failed: ' + '; '.join(ga.get('errors',[])[:5]))
        session=[r for r in rows if r.get('record_type')=='session_start']
        need(len(session) == 1, 'coverage audit requires one clean session')
        sid,gid=session[0]['session_id'],session[0]['guard_id']
        need(guards[0].get('guard_id') == gid and guards[0].get('gpu_uuid') == gpu, 'session guard identity mismatch')
        need(all(r.get('session_id') == sid and r.get('guard_id') == gid for r in rows
                 if r.get('record_type') != 'budget_exhausted'), 'mixed sessions or guards')
        need(ledger[0]['event'] == 'start' and ledger[-1]['state']['inflight'] is None and
             all(x['state']['contract'] == contract for x in ledger), 'budget state invalid')
        initial=ledger[0]['state']; start,seconds,boot=initial['start'],initial['seconds'],initial['boot']
        need(all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,seconds)) and seconds>0 and
             isinstance(boot,str) and bool(boot), 'invalid budget clock origin')
        previous=start
        for event in ledger:
            state=event['state']; last=state['last']
            need(state['start']==start and state['seconds']==seconds and state['boot']==boot and
                 isinstance(last,(int,float)) and math.isfinite(last) and last>=previous, 'budget clock or policy changed')
            previous=last
        exhausted=[r for r in rows if r.get('record_type')=='budget_exhausted']
        denied=[r for r in ledger if r.get('event')=='denied']
        need((not exhausted and not denied) or (len(exhausted)==len(denied)==1 and
             ledger[-1]['event']=='denied' and rows[-1]['record_type']=='budget_exhausted'),
             'budget boundary lacks matching terminal evidence')
        budget_boundary=bool(exhausted)
        if budget_boundary:
            need(denied[0]['state']['last'] + contract['timeout'] + 30 >= start + seconds,
                 'budget denial happened before the deadline margin')
        executions=[r for r in rows if r.get('record_type')=='execution']
        begins=[r for r in ledger if r.get('event')=='work_start']
        ends=[r for r in ledger if r.get('event')=='work_finish']
        need(len(executions)==len(begins)==len(ends), 'budget execution count mismatch')
        need([r['event'] for r in ledger] == ['start']+['work_start','work_finish']*len(executions)+
             (['denied'] if budget_boundary else []), 'budget event order mismatch')
        for st,end,row in zip(begins,ends,executions):
            item=end['identity']; w=row['workload']
            need(st['state']['inflight']==item and end['status']==row['status'] and
                 (item['n'],item['batch'],item['candidate_id'],item['phase'],item['round']) ==
                 (w['n'],w['batch'],row['candidate_id'],row['phase'],row['round']), 'budget work mismatch')
            decoded=[json.loads(x) for x in row['stdout'].splitlines() if x.strip()]
            need(len(decoded)==1 and decoded[0]==row['result'], 'raw stdout mismatch')
            binary='original' if row['candidate_id']=='original' else 'expanded'
            need(row['executable_sha256']==contract['binaries'][binary], 'execution binary mismatch')
            need(row['phase'] in ('identity','calibration','describe','validation','timing','post_rejection'),
                 'unknown execution phase')
        base.reject_prior_numerical_failure(rows)
        workloads,selected,counts=read_manifests(manifests,contract)
        need(all((r['workload']['n'],r['workload']['batch']) in workloads for r in executions), 'unknown workload execution')
        segments={}; active=None
        for row in rows:
            if row.get('record_type')=='candidate_start':
                key=(row['n'],row['batch'],row['candidate_id'])
                need(key in selected and key not in segments and row['request']==selected[key]['request'],
                     'unselected, duplicated or altered candidate start')
                segments[key]=[row]; active=key
            elif row.get('record_type')=='execution' and row['phase'] in ('identity','calibration'):
                active=None
            elif active is not None:
                segments[active].append(row)
            elif row.get('record_type') in ('screen_pair','candidate_disposition') or (
                row.get('record_type')=='execution' and row['phase'] in ('describe','validation','timing','post_rejection')):
                raise ValueError('candidate evidence without candidate start')
        status_rows=[r for r in rows if r.get('record_type')=='workload_status']
        need(len({(r['n'],r['batch']) for r in status_rows})==len(status_rows) and
             all((r['n'],r['batch']) in workloads for r in status_rows), 'invalid or duplicate workload status')
        orig_signatures={}; graphs={}
        for n,batch in workloads:
            controls=[r for r in executions if r['phase']=='identity' and (r['workload']['n'],r['workload']['batch'])==(n,batch)]
            statuses=[r for r in status_rows if (r['n'],r['batch'])==(n,batch)]
            names=[r['candidate_id'] for r in controls]
            need(names in ([],['original'],['original','expanded_default'],['original','expanded_default','identity']),
                 'identity sequence invalid')
            for r in controls:
                need((r['workload']['pattern'],r['workload']['seed'],r['workload']['graph_repeats']) == ('random',42,64),
                     'identity input mismatch')
            original=validation.check(controls[0],gpu,original=True) if controls else None
            default=validation.check(controls[1],gpu) if len(controls)>=2 else None
            if default is not None: comparison.compare(default,original)
            identity,why=base.representable(default) if default is not None else (None,None)
            if len(controls)==3:
                need(identity is not None, 'identity unexpectedly unsupported')
                replay=validation.check(controls[2],gpu,request=identity)
                need(validation.signature(default)==validation.signature(replay), 'identity replay differs')
            if statuses:
                status=statuses[0]
                if status['status']=='unsupported_identity':
                    need(len(controls)==2 and identity is None and status['reason']==why, 'invalid unsupported identity')
                else:
                    need(status['status']=='identity_passed' and len(controls)==3 and status['request']==identity,
                         'identity status mismatch')
                    orig_signatures[(n,batch)]=search_controller.source_signature(original,original=True)
            else:
                need(budget_boundary and len(controls)<3 and not (len(controls)==2 and identity is None),
                     'missing workload status without a valid interrupted identity')
            cal=[r for r in executions if r['phase']=='calibration' and (r['workload']['n'],r['workload']['batch'])==(n,batch)]
            if cal: need((n,batch) in orig_signatures, 'calibration before identity passed')
            for index,row in enumerate(cal):
                result=validation.check(row,gpu,original=True)
                need(row['candidate_id']=='original' and row['workload']['graph_repeats']==64*2**index<=2048 and
                     (row['workload']['pattern'],row['workload']['seed'])==('random',42) and
                     search_controller.source_signature(result,original=True)==orig_signatures[(n,batch)], 'calibration mismatch')
                if index<len(cal)-1: need(min(result['raw_interval_ms_samples'])<2, 'calibration overshot')
            if cal and min(result['raw_interval_ms_samples'])>=2: graphs[(n,batch)]=cal[-1]['workload']['graph_repeats']
            elif cal: need(budget_boundary, 'calibration incomplete without budget boundary')
        unsupported={(r['n'],r['batch']) for r in status_rows if r['status']=='unsupported_identity'}
        if not budget_boundary:
            need(len(status_rows)==len(workloads) and all(key in segments or key[:2] in unsupported for key in selected),
                 'unstarted selected work without budget boundary or unsupported identity')
        cases=[]; partials=[]
        for key,request_row in selected.items():
            n,batch,cid=key; req=request_row['request']
            if key not in segments:
                reason='unsupported_identity' if (n,batch) in unsupported else 'budget_boundary'
                cases.append(dict(request_row,status='selected_not_started',not_started_reason=reason)); continue
            need((n,batch) in graphs, 'candidate started before successful workload calibration')
            graph=graphs[(n,batch)]; seg=segments[key]
            ex=[r for r in seg if r.get('record_type')=='execution']
            desc=[r for r in ex if r['phase']=='describe']; vals=[r for r in ex if r['phase']=='validation']
            timings=[r for r in ex if r['phase']=='timing']; post=[r for r in ex if r['phase']=='post_rejection']
            need(len(ex)==len(desc)+len(vals)+len(timings)+len(post), 'unexpected candidate execution')
            need([r['phase'] for r in ex] == ['describe']*len(desc)+['validation']*len(vals)+
                 ['timing']*len(timings)+['post_rejection']*len(post), 'candidate phase order mismatch')
            need(len(desc)<=1 and len(post)<=1 and len(vals)<=15 and len(timings)<=6, 'duplicate candidate execution')
            disposition=[r for r in seg if r.get('record_type')=='candidate_disposition']
            need(len(disposition)<=1, 'duplicate candidate disposition')
            for row in ex:
                need((row['workload']['n'],row['workload']['batch'])==(n,batch) and
                     row['workload']['graph_repeats']==graph, 'candidate workload mismatch')
            failed=None; described=None; sig=None
            if desc:
                d=desc[0]
                need(d['candidate_id']==cid and d['workload']['study_mode']=='describe', 'describe identity mismatch')
                if validation.known_rejection(d,gpu): failed=d
                else: described=validation.check_describe(d,gpu,req)
            need(not vals or described is not None, 'validation before successful describe')
            expected_inputs=[(pattern,seed) for pattern in base.PATTERNS for seed in base.SEEDS]
            need([(r['workload']['pattern'],r['workload']['seed']) for r in vals] == expected_inputs[:len(vals)],
                 'validation input sequence mismatch')
            for index,row in enumerate(vals):
                need(row['candidate_id']==cid and row['workload']['study_mode']=='execute', 'validation identity mismatch')
                if validation.known_rejection(row,gpu):
                    need(index==len(vals)-1 and failed is None, 'rejection is not terminal'); failed=row; break
                result=validation.check(row,gpu,request=req); dp=described['precompile_kernels'][0]; cp=result['precompile_kernels'][0]
                need(all(cp[k]==dp[k] for k in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')),
                     'describe and validation code differ')
                actual=search_controller.source_signature(result,original=False)
                need(sig is None or sig==actual, 'validation signature drift'); sig=actual
            if failed:
                need(failed.get('status')=='failed' and failed['direct_environment']==validation.request_env(req) and
                     failed['candidate_id']==cid and not timings and not any(r.get('record_type')=='screen_pair' for r in seg),
                     'rejection environment or phase mismatch')
                if failed is desc[0]: need(not vals, 'validation after describe rejection')
                kind='scheduler_rejected' if failed['result']['error']=='initializeVkFFT failed with code 3002' else 'resource_rejected'
                if kind=='scheduler_rejected':
                    need(failed['result'].get('vkfft_result_code')==3002, '3002 error text and VkFFT code differ')
                if kind=='resource_rejected': need(failed['phase']=='validation', 'resource rejection before compile')
                if post:
                    need(post[0]['candidate_id']=='original', 'invalid post-rejection control')
                    pr=validation.check(post[0],gpu,original=True)
                    need(search_controller.source_signature(pr,original=True)==orig_signatures[(n,batch)], 'post-control drift')
                if disposition:
                    d=disposition[0]
                    need(len(post)==1 and d['status']==kind and d['request']==req and
                         (d['n'],d['batch'],d['candidate_id'])==key and d.get('diagnostic')==
                         failed['result'].get('expanded_rejection_reason','unattributed'), 'rejection disposition mismatch')
                    cases.append(dict(request_row,status=kind,diagnostic=d.get('diagnostic'),
                                      rejection_phase=failed['phase'],error=failed['result']['error'],
                                      guard_run_id=failed.get('guard_run_id'))); continue
                need(not post, 'post-control completed without rejection disposition')
            else:
                need(not post, 'post-rejection control without a rejection')
                if timings: need(len(vals)==15, 'timing before 5x3 validation')
                ratios=[]
                for index,row in enumerate(timings):
                    round_no=index//2; order=index%2
                    expected_name=base.seeded_order(n,batch,cid,round_no)[order]
                    name='original' if expected_name=='original' else cid
                    need(row['round']==round_no and row['order']==order and row['candidate_id']==name and
                         (row['workload']['pattern'],row['workload']['seed'])==('random',42), 'timing order or input mismatch')
                    original=name=='original'
                    result=validation.check(row,gpu,original=original,request=None if original else req,timing=True)
                    need(result['warmup_actual']>=5 and result['warmup_wall_ms']>=1000 and
                         search_controller.source_signature(result,original=original)==
                         (orig_signatures[(n,batch)] if original else sig), 'timing source, resource or warmup mismatch')
                pairs=[r for r in seg if r.get('record_type')=='screen_pair']
                need(len(pairs)==len(timings)//2, 'pair record count mismatch')
                for round_no in range(len(pairs)):
                    pair=timings[round_no*2:round_no*2+2]
                    values={('original' if r['candidate_id']=='original' else 'candidate'):
                            statistics.median(r['result']['exec_ms_samples']) for r in pair}
                    ratio=values['candidate']/values['original']; ratios.append(ratio)
                    pr=pairs[round_no]
                    need((pr['n'],pr['batch'],pr['candidate_id'],pr['round'])==(n,batch,cid,round_no) and
                         math.isclose(pr['ratio'],ratio,rel_tol=1e-12), 'screen pair mismatch')
                if disposition:
                    d=disposition[0]
                    need(len(vals)==15 and len(timings)==6 and len(pairs)==3 and d['status']=='screen_complete' and
                         (d['n'],d['batch'],d['candidate_id'])==key and d['request']==req and
                         d['pair_ratios']==ratios and math.isclose(d['median_ratio'],statistics.median(ratios),rel_tol=1e-12),
                         'screen disposition mismatch')
                    cases.append(dict(request_row,status='screen_complete',median_ratio=statistics.median(ratios))); continue
                need(len(timings)<6, 'three complete pairs without screen disposition')
            need(not disposition and budget_boundary, 'unexplained incomplete candidate')
            expected_phase='describe' if not desc else 'post_rejection' if failed else 'validation' if len(vals)<15 else 'timing'
            expected_name='original' if expected_phase=='post_rejection' else cid
            expected_round=0
            if expected_phase=='timing':
                expected_round=len(timings)//2
                expected_name=('original' if base.seeded_order(n,batch,cid,expected_round)[len(timings)%2]=='original' else cid)
            expected_denied={'n':n,'batch':batch,'candidate_id':expected_name,'phase':expected_phase,'round':expected_round}
            need(denied[-1].get('identity')==expected_denied, 'partial candidate does not match denied work')
            cases.append(dict(request_row,status='partial')); partials.append(key)
        need(len(partials)<=1 and (not partials or list(segments)[-1]==partials[0]), 'multiple or nonterminal partial candidates')
        if budget_boundary and not partials:
            item=denied[0].get('identity') or {}; work=(item.get('n'),item.get('batch'))
            need(work in workloads and item.get('round')==0, 'denied workload identity is invalid')
            if item.get('phase')=='identity':
                controls=[r for r in executions if r['phase']=='identity' and
                          (r['workload']['n'],r['workload']['batch'])==work]
                expected_names=['original','expanded_default','identity']
                need(len(controls)<3 and item.get('candidate_id')==expected_names[len(controls)] and
                     not any((r['n'],r['batch'])==work for r in status_rows), 'denied identity is not the next control')
            elif item.get('phase')=='calibration':
                need(work in orig_signatures and work not in graphs and item.get('candidate_id')=='original',
                     'denied calibration is not the next calibration')
            else:
                raise ValueError('denied work has no matching partial candidate or incomplete identity')
        counts.update(Counter(c['status'] for c in cases))
        result_counts={name:counts[name] for name in ('manifest_total','static_rejected','not_selected','selected',
                       'selected_not_started','partial','scheduler_rejected','resource_rejected','screen_complete')}
        need(result_counts['selected'] == sum(result_counts[k] for k in ('selected_not_started','partial',
             'scheduler_rejected','resource_rejected','screen_complete')), 'selected partition mismatch')
        return {'schema':'frontier-screen-coverage/v1','status':'pass','gpu':gpu,'execution_commit':contract['execution_commit'],
                'budget_exhausted':budget_boundary,'counts':result_counts,'cases':cases,'guard_audit':ga,
                'unmeasured_note':'not_selected, selected_not_started and partial are unmeasured, not invalid',
                'independent_domain_total':counts['independent_domain_total'],
                'independent_static_supported':counts['independent_static_supported'],
                'all_workload_inventories_complete':counts['complete_inventory_workloads']==len(workloads),
                'scope':'one clean guarded session; request validity and uniqueness checked and inventory completeness compared with independent integer-composition counts; no global VkFFT optimum claim'}
    except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OSError,base.NumericalFailure) as error:
        return {'schema':'frontier-screen-coverage/v1','status':'fail','errors':[f'{type(error).__name__}: {error}']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','guard','incidents','budget','original','expanded','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest',type=Path,action='append',required=True); p.add_argument('--gpu',required=True)
    args=p.parse_args()
    result=audit_coverage(base.read_log(args.rows),base.read_log(args.guard),base.read_log(args.incidents),
                          base.read_log(args.budget),args.manifest,args.gpu,{'original':args.original,'expanded':args.expanded})
    result['evidence_sha256']={str(x):base.sha(x) for x in
        [args.rows,args.guard,args.incidents,args.budget,args.original,args.expanded,*args.manifest]}
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,sort_keys=True,indent=2,allow_nan=False); out.write('\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__': sys.exit(main())
