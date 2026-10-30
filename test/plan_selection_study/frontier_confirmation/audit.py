#!/usr/bin/env python3
"""Independent raw-evidence audit for one or two confirmation sessions."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location('frontier_execution_audit_base', STUDY / 'frontier_execution' / 'runner.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(STUDY / 'expanded_search'))
import audit_guard
import audit_pairs
import search_controller
import validation
import original_compare_controller as comparison


def need(condition, message):
    if not condition: raise ValueError(message)


def signature_sha(result, original):
    signature = search_controller.source_signature(result, original=original)
    data = json.dumps(signature, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def audit_session(rows, guard, incidents, ledger, plan, plan_sha256, gpu, execution_commit,
                  binaries, screen_proofs, pair_order):
    """Recompute every case from raw guarded executions; never trust recorded summaries."""
    try:
        need(rows and guard and ledger, 'missing confirmation evidence')
        contract = rows[0]['contract']
        need(all(r.get('contract') == contract for r in rows), 'mixed contracts')
        need(contract.get('schema') == 'frontier-confirmation/v1' and contract.get('gpu') == gpu and
             contract.get('execution_commit') == execution_commit and contract.get('plan_sha256') == plan_sha256 and
             contract.get('source_sha256') == plan['source_sha256'], 'contract, GPU, plan or source mismatch')
        need(len(execution_commit) == 40 and all(c in '0123456789abcdef' for c in execution_commit), 'invalid execution commit')
        need(contract.get('session_kind') in ('first', 'second'), 'invalid session kind')
        need(set(binaries) == {'original', 'expanded'} and
             {k:base.sha(v) for k,v in binaries.items()} == contract['binaries'], 'binary hash mismatch')
        need(plan.get('schema') == 'frontier-confirmation-plan/v1' and plan.get('gpu') == gpu, 'plan mismatch')
        guard_result = audit_guard.audit(rows, guard, incidents, gpu)
        need(guard_result['status'] == 'pass', 'guard audit failed: ' + '; '.join(guard_result.get('errors', [])[:5]))
        start = [r for r in rows if r.get('record_type') == 'session_start']
        end = [r for r in rows if r.get('record_type') == 'session_complete']
        need(len(start) == len(end) == 1 and end[0].get('status') == 'complete', 'missing or partial session')
        session, gid = start[0]['session_id'], start[0]['guard_id']
        need(bool(session) and bool(gid) and all(r.get('session_id') == session and r.get('guard_id') == gid for r in rows),
             'session or guard ID mismatch')
        need(guard[0].get('guard_id') == gid, 'output and guard ID differ')
        executions = [r for r in rows if r.get('record_type') == 'execution']
        for row in executions:
            need(row.get('status') == 'ok' and row.get('returncode') == 0 and
                 row.get('phase') in ('identity', 'calibration', 'confirmation_timing'), 'failed or unexpected execution')
            parsed = [json.loads(line) for line in row['stdout'].splitlines() if line.strip()]
            need(len(parsed) == 1 and parsed[0] == row['result'], 'execution result differs from stdout')
            binary = 'original' if row['candidate_id'] == 'original' else 'expanded'
            need(row['executable_sha256'] == contract['binaries'][binary], 'execution binary mismatch')
        need(ledger[0]['event'] == 'start' and ledger[-1]['state']['inflight'] is None and
             all(item['state']['contract'] == contract for item in ledger), 'budget state incomplete or mismatched')
        starts = [r for r in ledger if r.get('event') == 'work_start']
        finishes = [r for r in ledger if r.get('event') == 'work_finish']
        need(len(starts) == len(finishes) == len(executions), 'budget count differs from executions')
        for begun, ended, row in zip(starts, finishes, executions):
            item = ended['identity']
            need(begun['state']['inflight'] == item and ended['status'] == row['status'] and
                 (item['n'], item['batch'], item['candidate_id'], item['phase'], item['round']) ==
                 (row['workload']['n'], row['workload']['batch'], row['candidate_id'], row['phase'], row['round']),
                 'budget work identity differs')
        cases = plan['cases']
        keys = [(c['n'], c['batch'], c['candidate_id']) for c in cases]
        need(cases and len(set(keys)) == len(keys), 'empty or duplicate confirmation plan')
        need(all(c.get('role','winner') in ('winner','negative_control') for c in cases), 'invalid candidate role')
        need(end[0].get('candidate_count') == len(cases), 'session completion candidate count mismatch')
        need(set(screen_proofs) == set(keys), 'screen proof does not cover exact plan')
        starts_by_key = {}
        for row in (r for r in rows if r.get('record_type') == 'candidate_start'):
            key = (row['n'], row['batch'], row['candidate_id'])
            need(key not in starts_by_key, 'duplicate candidate start'); starts_by_key[key] = row
        need(set(starts_by_key) == set(keys), 'candidate starts differ from plan')
        guard_ids = {r['guard_run_id']:r for r in executions}
        need(len(guard_ids) == len(executions), 'missing or duplicate execution ID')
        used_timing_ids = set()
        results = []
        for n,batch in sorted({(c['n'],c['batch']) for c in cases}):
            identities = [r for r in executions if r['phase'] == 'identity' and
                          (r['workload']['n'],r['workload']['batch']) == (n,batch)]
            need(Counter(r['candidate_id'] for r in identities) == Counter({'original':1,'expanded_default':1,'identity':1}),
                 'identity controls missing or duplicated')
            for row in identities:
                need((row['workload']['pattern'],row['workload']['seed']) == ('random',42), 'identity input mismatch')
            controls = {r['candidate_id']:r for r in identities}
            orig = validation.check(controls['original'],gpu,original=True)
            default = validation.check(controls['expanded_default'],gpu)
            comparison.compare(default,orig)
            identity, reason = base.representable(default)
            need(identity is not None, 'identity not representable: ' + str(reason))
            replay = validation.check(controls['identity'],gpu,request=identity)
            need(validation.signature(default) == validation.signature(replay), 'identity replay differs')
            status = [r for r in rows if r.get('record_type') == 'workload_status' and (r['n'],r['batch']) == (n,batch)]
            need(len(status) == 1 and status[0]['status'] == 'identity_passed' and status[0]['request'] == identity,
                 'identity disposition mismatch')
            original_hash = signature_sha(orig,True)
            calibrations = [r for r in executions if r['phase'] == 'calibration' and
                            (r['workload']['n'],r['workload']['batch']) == (n,batch)]
            need(bool(calibrations), 'calibration missing')
            for index,row in enumerate(calibrations):
                result = validation.check(row,gpu,original=True)
                need(row['candidate_id'] == 'original' and row['workload']['graph_repeats'] == 64*2**index <= 2048 and
                     (row['workload']['pattern'],row['workload']['seed']) == ('random',42) and
                     signature_sha(result,True) == original_hash, 'calibration identity mismatch')
                if index < len(calibrations)-1: need(min(result['raw_interval_ms_samples']) < 2, 'calibration overshot')
            need(min(result['raw_interval_ms_samples']) >= 2, 'calibration too short')
            graph = calibrations[-1]['workload']['graph_repeats']
            for case in (c for c in cases if (c['n'],c['batch']) == (n,batch)):
                cid, request = case['candidate_id'], case['request']; role=case.get('role','winner'); key = (n,batch,cid)
                candidate = starts_by_key[key]; proof = screen_proofs[key]
                need(candidate['request'] == request and proof['request'] == request and
                     candidate.get('role','winner') == role == proof.get('role','winner') and
                     candidate['screen_candidate_signature_sha256'] == proof['screen_candidate_signature_sha256'] and
                     candidate['screen_original_signature_sha256'] == proof['screen_original_signature_sha256'] == original_hash,
                     'screen proof or original source signature mismatch')
                pairs = [r for r in rows if r.get('record_type') == 'confirmation_pair' and
                         (r['n'],r['batch'],r['candidate_id']) == key]
                need(len(pairs) == 10 and {p['round'] for p in pairs} == set(range(10)), 'missing or duplicate paired round')
                times = []
                for pair in sorted(pairs,key=lambda p:p['round']):
                    round_no = pair['round']
                    o = guard_ids[pair['original_guard_run_id']]; c = guard_ids[pair['candidate_guard_run_id']]
                    for row, name in ((o,'original'),(c,cid)):
                        rid = row['guard_run_id']; need(rid not in used_timing_ids, 'timing execution reused')
                        used_timing_ids.add(rid)
                        need(row['candidate_id'] == name and row['phase'] == 'confirmation_timing' and
                             row['round'] == round_no and (row['workload']['n'],row['workload']['batch'],
                             row['workload']['pattern'],row['workload']['seed'],row['workload']['graph_repeats']) ==
                             (n,batch,'random',42,graph), 'timing execution identity mismatch')
                    order = pair_order(plan_sha256,contract['session_kind'],n,batch,cid,round_no)
                    need({o['order'],c['order']} == {0,1} and
                         [('original' if row['candidate_id'] == 'original' else 'candidate')
                          for row in sorted((o,c),key=lambda r:r['order'])] == list(order), 'timing order mismatch')
                    original = validation.check(o,gpu,original=True,timing=True)
                    selected = validation.check(c,gpu,request=request,timing=True)
                    for result in (original,selected):
                        need(result['warmup_actual'] >= 5 and result['warmup_wall_ms'] >= 1000, 'short warmup')
                    need(signature_sha(original,True) == original_hash and
                         signature_sha(selected,False) == proof['screen_candidate_signature_sha256'], 'timing signature drift')
                    candidate_ms, original_ms = statistics.median(selected['exec_ms_samples']), statistics.median(original['exec_ms_samples'])
                    need(math.isclose(pair['candidate_ms'],candidate_ms,rel_tol=1e-12) and
                         math.isclose(pair['original_ms'],original_ms,rel_tol=1e-12), 'recorded timing mismatch')
                    times.append((candidate_ms,original_ms))
                dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition' and
                                (r['n'],r['batch'],r['candidate_id']) == key]
                need(len(dispositions) == 1 and dispositions[0].get('status') == 'confirmation_complete' and
                     dispositions[0].get('request') == request and dispositions[0].get('role','winner') == role,
                     'candidate incomplete or failed')
                stats = audit_pairs.ten_round_session(times)
                meets_threshold=audit_pairs.passes_session_criterion(stats)
                results.append({'n':n,'batch':batch,'candidate_id':cid,'request':request,'role':role,
                                'statistics':stats,'meets_practical_threshold':meets_threshold,
                                'practical':role=='winner' and meets_threshold,
                                'default_faster':role=='negative_control' and stats['geometric_mean_ratio']>1 and stats['ci95_low']>1})
        timing_rows = {r['guard_run_id'] for r in executions if r['phase'] == 'confirmation_timing'}
        need(timing_rows == used_timing_ids, 'orphan timing execution')
        need(len([r for r in executions if r['phase'] == 'identity']) == 3*len({(c['n'],c['batch']) for c in cases}),
             'unexpected identity execution')
        need(len([r for r in rows if r.get('record_type') == 'candidate_disposition']) == len(cases), 'unexpected disposition')
        need(len([r for r in rows if r.get('record_type') == 'confirmation_pair']) == 10*len(cases), 'unexpected pair record')
        workloads = {(c['n'],c['batch']) for c in cases}
        need(all((r['workload']['n'],r['workload']['batch']) in workloads for r in executions), 'unexpected workload execution')
        return {'schema':'frontier-confirmation-session-audit/v1','status':'pass', 'gpu':gpu,
                'execution_commit':execution_commit,'vkfft_commit':contract['vkfft_commit'],
                'binaries':contract['binaries'],'plan_sha256':plan_sha256,'source_sha256':plan['source_sha256'],
                'session_kind':contract['session_kind'],'session_id':session,'guard_id':gid,
                'first_session':contract.get('first_session'),
                'guard_started_utc':guard[0]['utc'],'guard_completed_utc':guard[-1]['utc'],
                'guard_run_ids':sorted(guard_ids),'results':results,
                'guard_audit':guard_result,'scope':'single session, raw validated pairs; within-session log-t intervals',
                'role_interpretation':'only winners can be practical; a negative control is default_faster when its candidate/default ratio and 95% CI lower bound exceed 1'}
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        return {'schema':'frontier-confirmation-session-audit/v1','status':'fail','errors':[f'{type(error).__name__}: {error}']}


def audit_two(first, second, first_evidence_sha256):
    try:
        need(first['status'] == second['status'] == 'pass', 'both session audits must pass')
        need(first['session_kind'] == 'first' and second['session_kind'] == 'second', 'session order mismatch')
        for key in ('gpu','execution_commit','vkfft_commit','binaries','plan_sha256','source_sha256'):
            need(first[key] == second[key], 'two-session ' + key + ' mismatch')
        need(first['session_id'] != second['session_id'] and first['guard_id'] != second['guard_id'] and
             not set(first['guard_run_ids']) & set(second['guard_run_ids']), 'sessions or process evidence reused')
        need(second['first_session'] == {'session_id':first['session_id'],'guard_id':first['guard_id'],
                                         'evidence_sha256':first_evidence_sha256},
             'second session is not bound to these first-session evidence files')
        one = datetime.fromisoformat(first['guard_completed_utc'].replace('Z','+00:00'))
        two = datetime.fromisoformat(second['guard_started_utc'].replace('Z','+00:00'))
        need(one <= two, 'second session predates first completion')
        def key(r): return (r['n'],r['batch'],r['candidate_id'])
        a = {key(r):r for r in first['results']}; b = {key(r):r for r in second['results']}
        need(len(a) == len(first['results']) and len(b) == len(second['results']) and set(a) == set(b), 'case sets differ')
        results = []
        for k in sorted(a):
            need(a[k]['request'] == b[k]['request'], 'request changed between sessions')
            role=a[k].get('role','winner')
            need(role in ('winner','negative_control') and b[k].get('role','winner') == role, 'role changed between sessions')
            first_meets=audit_pairs.passes_session_criterion(a[k]['statistics'])
            second_meets=audit_pairs.passes_session_criterion(b[k]['statistics'])
            both_practical=role=='winner' and a[k]['practical'] and b[k]['practical'] and first_meets and second_meets
            results.append({'n':k[0],'batch':k[1],'candidate_id':k[2],'request':a[k]['request'],'role':role,
                            'first':a[k]['statistics'],'second':b[k]['statistics'],
                            'first_meets_practical_threshold':first_meets,'second_meets_practical_threshold':second_meets,
                            'first_practical':role=='winner' and a[k]['practical'] and first_meets,
                            'second_practical':role=='winner' and b[k]['practical'] and second_meets,
                            'same_improvement_direction':a[k]['statistics']['geometric_mean_ratio'] < 1 and
                                                         b[k]['statistics']['geometric_mean_ratio'] < 1,
                            'both_sessions_practical':both_practical,'positive_confirmation':both_practical,
                            'both_sessions_default_faster':role=='negative_control' and
                                a[k]['statistics']['geometric_mean_ratio']>1 and a[k]['statistics']['ci95_low']>1 and
                                b[k]['statistics']['geometric_mean_ratio']>1 and b[k]['statistics']['ci95_low']>1,
                            'negative_control_unexpected_improvement':role=='negative_control' and first_meets and second_meets})
        return {'schema':'frontier-confirmation-two-session-audit/v1','status':'pass',
                'gpu':first['gpu'],'plan_sha256':first['plan_sha256'],'results':results,
                'scope':'two ordered independent guarded sessions on one GPU; no multiple-testing adjustment',
                'role_interpretation':'positive_confirmation is reserved for winners; negative controls report default-faster and unexpected-improvement outcomes separately'}
    except (KeyError, ValueError, TypeError) as error:
        return {'schema':'frontier-confirmation-two-session-audit/v1','status':'fail','errors':[f'{type(error).__name__}: {error}']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','screen-rows','screen-guard','screen-incidents','screen-budget','first-rows','first-guard',
                 'first-incidents','first-budget','original','expanded','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('second-rows','second-guard','second-incidents','second-budget'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--gpu',required=True); p.add_argument('--execution-commit',required=True)
    args=p.parse_args()
    secondary = (args.second_rows,args.second_guard,args.second_incidents,args.second_budget)
    if any(secondary) and not all(secondary): p.error('second session requires all four evidence files')
    try:
        import evidence
        plan = json.loads(args.plan.read_text(encoding='utf8'))
        binaries={'original':args.original,'expanded':args.expanded}
        paths={'rows':args.screen_rows,'guard':args.screen_guard,'incidents':args.screen_incidents,'budget':args.screen_budget}
        screen = evidence.audit_screen(plan,paths,binaries)
        need(screen['gpu'] == args.gpu and screen['screen_contract']['binaries'] == {k:base.sha(v) for k,v in binaries.items()},
             'screen GPU or binaries mismatch')
        proofs={(x['n'],x['batch'],x['candidate_id']):x for x in screen['cases']}
        def session(rows,guard,incidents,budget):
            outcome=audit_session(base.read_log(rows),base.read_log(guard),base.read_log(incidents),base.read_log(budget),
                                  plan,base.sha(args.plan),args.gpu,args.execution_commit,binaries,proofs,evidence.pair_order)
            if outcome['status'] == 'pass' and outcome['vkfft_commit'] != screen['screen_contract']['vkfft_commit']:
                return {'schema':'frontier-confirmation-session-audit/v1','status':'fail','errors':['screen VkFFT commit mismatch']}
            return outcome
        first=session(args.first_rows,args.first_guard,args.first_incidents,args.first_budget)
        need(first.get('session_kind') == 'first' and first.get('status') == 'pass',
             'first session failed: ' + '; '.join(first.get('errors', [])))
        report={'schema':'frontier-confirmation-audit-report/v1','status':'pass','screen':screen,
                'first':first,'second':None,'combined':None}
        if all(secondary):
            second=session(*secondary)
            first_hashes={'rows':base.sha(args.first_rows),'guard':base.sha(args.first_guard),
                          'incidents':base.sha(args.first_incidents),'budget':base.sha(args.first_budget)}
            combined=audit_two(first,second,first_hashes)
            need(combined['status'] == 'pass', 'two-session audit failed: ' + '; '.join(combined.get('errors', [])))
            report['second']=second; report['combined']=combined
        all_paths=[args.plan,*paths.values(),args.first_rows,args.first_guard,args.first_incidents,args.first_budget,
                   args.original,args.expanded,*[x for x in secondary if x is not None]]
        report['evidence_sha256']={str(x):base.sha(x) for x in all_paths}
    except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OSError) as error:
        report={'schema':'frontier-confirmation-audit-report/v1','status':'fail','errors':[f'{type(error).__name__}: {error}']}
    with args.output.open('x',encoding='utf8') as out:
        json.dump(report,out,sort_keys=True,indent=2,allow_nan=False); out.write('\n')
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__': sys.exit(main())
