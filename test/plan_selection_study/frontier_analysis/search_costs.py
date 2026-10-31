#!/usr/bin/env python3
"""Wall-clock accounting for one coverage-audited frontier screen ledger."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analyze

PHASES=('identity','calibration','describe','validation','timing','post_rejection')
FIELDS=('n','batch','candidate_id','phase','round')


def need(ok, message):
    if not ok: raise ValueError(message)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def identity(value):
    need(isinstance(value,dict) and set(value)==set(FIELDS), 'malformed ledger work identity')
    need(type(value['n']) is int and value['n']>0 and type(value['batch']) is int and value['batch']>0 and
         isinstance(value['candidate_id'],str) and bool(value['candidate_id']) and
         value['phase'] in PHASES and type(value['round']) is int and value['round']>=0,
         'invalid ledger work identity')
    return value


def analyze_costs(coverage, rows, ledger, rows_sha256, budget_sha256):
    need(coverage.get('schema')=='frontier-screen-coverage/v1' and coverage.get('status')=='pass' and
         (coverage.get('guard_audit') or {}).get('status')=='pass', 'coverage or guard audit did not pass')
    hashes=coverage.get('evidence_sha256') or {}
    need(digest(rows_sha256) and digest(budget_sha256) and rows_sha256!=budget_sha256 and
         list(hashes.values()).count(rows_sha256)==1 and list(hashes.values()).count(budget_sha256)==1,
         'raw rows or budget are not uniquely bound to coverage')
    need(rows and rows[0].get('record_type')=='session_start' and isinstance(rows[0].get('contract'),dict),
         'missing screen session contract')
    contract=rows[0]['contract']; gpu=contract.get('gpu'); commit=contract.get('execution_commit')
    need(isinstance(gpu,str) and bool(gpu) and coverage.get('gpu')==gpu and
         isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit) and
         coverage.get('execution_commit')==commit and all(r.get('contract')==contract for r in rows) and
         finite(contract.get('timeout')) and contract['timeout']>0,
         'GPU, execution commit or screen contract mismatch')
    need(ledger and ledger[0].get('event')=='start', 'missing ledger start')
    initial=ledger[0].get('state') or {}; start=initial.get('start'); seconds=initial.get('seconds'); boot=initial.get('boot')
    need(finite(start) and finite(seconds) and seconds>0 and isinstance(boot,str) and bool(boot) and
         initial.get('contract')==contract and initial.get('inflight') is None, 'invalid ledger origin')
    executions=[(i,r) for i,r in enumerate(rows) if r.get('record_type')=='execution']
    pairs=[]; previous=start; pending=None; denied=None
    for position,event in enumerate(ledger):
        state=event.get('state') or {}; last=state.get('last')
        need(state.get('boot')==boot and finite(state.get('start')) and state.get('start')==start and
             finite(state.get('seconds')) and state.get('seconds')==seconds and
             state.get('contract')==contract and finite(last) and last>=previous,
             'ledger boot, contract, origin or monotonic clock mismatch')
        previous=last; kind=event.get('event')
        if position==0:
            need(kind=='start' and state.get('inflight') is None, 'invalid start event')
        elif kind=='work_start':
            need(pending is None and denied is None, 'overlapping work or work after denial')
            pending=(dict(identity(state.get('inflight'))),last)
        elif kind=='work_finish':
            need(pending is not None and state.get('inflight') is None and
                 identity(event.get('identity'))==pending[0], 'unmatched finish or unresolved inflight')
            need(len(pairs)<len(executions), 'ledger has an unrecorded execution')
            row_index,row=executions[len(pairs)]; work=row.get('workload') or {}
            actual={'n':work.get('n'),'batch':work.get('batch'),'candidate_id':row.get('candidate_id'),
                    'phase':row.get('phase'),'round':row.get('round')}
            need(identity(actual)==pending[0] and row.get('status')==event.get('status'), 'ledger and execution identity/status mismatch')
            duration=last-pending[1]; wall=row.get('wall_seconds')
            need(finite(wall) and wall<=duration+1e-6, 'execution wall time exceeds its ledger interval')
            pairs.append({'row_index':row_index,'identity':pending[0],'start':pending[1],
                          'end':last,'seconds':duration,'status':event.get('status')})
            pending=None
        elif kind=='denied':
            need(pending is None and denied is None and position==len(ledger)-1 and
                 state.get('inflight') is None, 'invalid or nonterminal denial')
            denied=dict(identity(event.get('identity')))
        else:
            raise ValueError('unexpected ledger event')
    need(pending is None and len(pairs)==len(executions), 'unfinished or missing ledger work')
    exhausted=[i for i,r in enumerate(rows) if r.get('record_type')=='budget_exhausted']
    need((denied is None and not exhausted) or
         (denied is not None and exhausted==[len(rows)-1]), 'budget denial and screen boundary differ')
    if denied is not None:
        # The reserve decision precedes the recorded event clock read; this is
        # a necessary condition, not proof of the exact decision instant.
        need(previous+contract['timeout']+30 >= start+seconds, 'budget denial before reserve boundary')
    phase_seconds={p:sum(x['seconds'] for x in pairs if x['identity']['phase']==p) for p in PHASES}
    work_seconds=sum(x['seconds'] for x in pairs); elapsed=previous-start
    need(work_seconds<=elapsed+1e-6, 'overlapping ledger intervals')
    by_row={item['row_index']:item for item in pairs}
    selected={}; segments={}; active=None
    for case in coverage.get('cases',[]):
        key=(case['n'],case['batch'],case['candidate_id'])
        need(key not in selected, 'duplicate coverage candidate')
        selected[key]=case
    for index,row in enumerate(rows):
        kind=row.get('record_type')
        if kind=='candidate_start':
            key=(row['n'],row['batch'],row['candidate_id']); case=selected.get(key)
            need(active is None and case is not None and key not in segments and row.get('request')==case.get('request'),
                 'unknown, overlapping or duplicate candidate start')
            segments[key]={'executions':[],'pairs':[],'disposition':None}; active=key
        elif kind=='execution':
            phase=row.get('phase')
            if phase in ('identity','calibration'):
                need(active is None, 'workload preparation overlaps a candidate segment')
            else:
                need(active is not None, 'candidate execution has no candidate segment')
                work=row['workload']; need((work['n'],work['batch'])==active[:2] and
                    row['candidate_id'] in (active[2],'original'), 'candidate segment execution mismatch')
                segments[active]['executions'].append(index)
        elif kind=='screen_pair':
            need(active is not None and (row.get('n'),row.get('batch'),row.get('candidate_id'))==active,
                 'pair has no matching candidate segment')
            segments[active]['pairs'].append((index,row))
        elif kind=='candidate_disposition':
            need(active is not None and (row.get('n'),row.get('batch'),row.get('candidate_id'))==active and
                 row.get('request')==selected[active].get('request') and row.get('status')==selected[active].get('status'),
                 'candidate disposition mismatch')
            segments[active]['disposition']=row.get('status');active=None
    for key,case in selected.items():
        seg=segments.get(key); status=case.get('status')
        need(status in ('screen_complete','scheduler_rejected','resource_rejected','partial','selected_not_started'),
             'unknown coverage candidate status')
        if status=='selected_not_started': need(seg is None, 'unstarted candidate has a segment')
        elif status=='partial': need(seg is not None and seg['disposition'] is None, 'partial candidate segment mismatch')
        else: need(seg is not None and seg['disposition']==status, 'complete or rejected candidate segment mismatch')
    if active is not None:
        need(selected[active]['status']=='partial' and denied is not None and
             (denied['n'],denied['batch'])==active[:2] and denied['candidate_id'] in (active[2],'original') and
             denied['phase'] in ('describe','validation','timing','post_rejection'), 'unfinished candidate segment')
    elif denied is not None:
        need(denied['phase'] in ('identity','calibration'), 'denial without matching candidate or workload preparation')
    assigned=set(); candidate_costs=[]
    for key,case in sorted(selected.items()):
        seg=segments.get(key)
        if seg is None:
            candidate_costs.append({'n':key[0],'batch':key[1],'candidate_id':key[2],
                'status':case['status'],'started':False,'attributed_work_seconds':None,'observed_span':None})
            continue
        own=[i for i in seg['executions'] if rows[i]['candidate_id']==key[2]]
        baseline=[]
        for pair_index,pair in seg['pairs']:
            choices=[i for i in seg['executions'] if i<pair_index and rows[i]['phase']=='timing' and
                     rows[i]['round']==pair.get('round')]
            orig=[i for i in choices if rows[i]['candidate_id']=='original']
            cand=[i for i in choices if rows[i]['candidate_id']==key[2]]
            if (len(orig)==len(cand)==1 and {rows[orig[0]].get('order'),rows[cand[0]].get('order')}=={0,1} and
                    orig[0] not in baseline): baseline.append(orig[0])
        controls=[i for i in seg['executions'] if rows[i]['candidate_id']=='original' and
                  rows[i]['phase']=='post_rejection']
        if len(controls)!=1 or seg['disposition'] not in ('scheduler_rejected','resource_rejected'):
            controls=[]
        linked=own+baseline+controls
        need(len(set(linked))==len(linked) and not assigned.intersection(linked), 'execution assigned twice')
        assigned.update(linked)
        costs={p:sum(by_row[i]['seconds'] for i in own if rows[i]['phase']==p) for p in PHASES}
        own_seconds=sum(by_row[i]['seconds'] for i in own)
        base_seconds=sum(by_row[i]['seconds'] for i in baseline)
        control_seconds=sum(by_row[i]['seconds'] for i in controls)
        span=None
        if linked:
            begin=min(by_row[i]['start'] for i in linked); end=max(by_row[i]['end'] for i in linked)
            inside=[x for x in pairs if x['start']>=begin and x['end']<=end]
            other=sum(x['seconds'] for x in inside if x['row_index'] not in linked)
            between=end-begin-sum(x['seconds'] for x in inside)
            need(between>=-1e-6, 'candidate work interval overlap')
            span={'start_monotonic':begin,'end_monotonic':end,'seconds':end-begin,
                  'unattributed_work_seconds':other,'between_work_seconds':max(0,between)}
        candidate_costs.append({'n':key[0],'batch':key[1],'candidate_id':key[2],'status':case['status'],
            'started':True,'own_phase_seconds':costs,'own_work_seconds':own_seconds,
            'paired_original_timing_seconds':base_seconds,'post_rejection_original_seconds':control_seconds,
            'attributed_work_seconds':own_seconds+base_seconds+control_seconds,'observed_span':span})
    unassigned=[x for x in pairs if x['row_index'] not in assigned]
    unattrib_by_phase={p:sum(x['seconds'] for x in unassigned if x['identity']['phase']==p) for p in PHASES}
    return {'schema':'frontier-search-costs/v1','gpu':gpu,'execution_commit':commit,'boot':boot,
            'evidence_sha256':{'rows':rows_sha256,'budget':budget_sha256},
            'interval':{'start_monotonic':start,'end_monotonic':previous,'elapsed_seconds':elapsed,
                        'work_seconds':work_seconds,'gap_seconds':max(0,elapsed-work_seconds),'work_count':len(pairs),
                        'terminal_event':ledger[-1]['event'],'denied_identity':denied},
            'phase_seconds':phase_seconds,'candidate_costs':candidate_costs,
            'unattributed':{'work_seconds':sum(x['seconds'] for x in unassigned),'phase_seconds':unattrib_by_phase,
                            'unpaired_original_timing_seconds':sum(x['seconds'] for x in unassigned if
                                x['identity']['phase']=='timing' and x['identity']['candidate_id']=='original')},
            'scope':{'interval':'ledger start through last recorded finish or denial; not whole campaign wall time',
                     'work':'work_start to work_finish includes subprocess, JIT, guard checks, logging and scheduling; not pure compile or kernel time',
                     'gaps':'all remaining observed interval; may include CPU work, logging, guard monitoring, waiting or scheduler delay',
                     'candidate':'first through last attributable finished work is a lower-bound observation, not exact candidate-start-to-disposition elapsed; post-end work and pre-start setup are unknown',
                     'original':'original timing assigned only when a unique candidate/round screen_pair proves the pair; ambiguous or partial pairs remain unattributed',
                     'external':'build, manifest generation, device lock and initial guard may precede ledger; final guard, cleanup, tail logging and confirmation may follow it',
                     'trust':'passing coverage report is a trusted input rather than an authenticated signature; a different boot/session must not be combined'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('coverage','rows','budget','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); coverage_bytes=a.coverage.read_bytes(); coverage=json.loads(coverage_bytes)
    row_hash=analyze.sha(a.rows);budget_hash=analyze.sha(a.budget)
    rows=[json.loads(x) for x in a.rows.read_text(encoding='utf8').splitlines() if x.strip()]
    ledger=[json.loads(x) for x in a.budget.read_text(encoding='utf8').splitlines() if x.strip()]
    need(row_hash==analyze.sha(a.rows) and budget_hash==analyze.sha(a.budget), 'input changed while reading')
    report=analyze_costs(coverage,rows,ledger,row_hash,budget_hash)
    import hashlib
    report['evidence_sha256']['coverage']=hashlib.sha256(coverage_bytes).hexdigest()
    need(report['evidence_sha256']['coverage']==analyze.sha(a.coverage),'coverage changed while reading')
    with a.output.open('x',encoding='utf8') as out: json.dump(report,out,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
