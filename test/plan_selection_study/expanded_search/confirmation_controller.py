"""Independent randomized confirmation against the original VkFFT default."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

import budget
import run_suite as suite
import search_controller
import validation


class BudgetStop(Exception): pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--budget-dir',required=True,type=Path)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--session-kind',required=True,choices=('first','second'));p.add_argument('--first-results',type=Path)
    p.add_argument('--first-gate',type=Path);p.add_argument('--first-audit',type=Path);p.add_argument('--first-guard',type=Path);p.add_argument('--first-incidents',type=Path)
    p.add_argument('--first-budget-snapshot',type=Path)
    p.add_argument('--timeout',type=int,default=300)
    a=p.parse_args(argv)
    if not 1<=a.timeout<=3600:p.error('invalid timeout')
    if os.environ.get('VKFFT_STUDY_GUARD_UUID')!=a.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):p.error('guard required')
    b=budget.Budget(a.budget_dir)
    if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):p.error('confirmation budget unavailable')
    plan=json.loads(a.plan.read_text(encoding='utf8'))
    if plan.get('schema')!='expanded-confirmation-plan/v1' or not plan.get('screen_gate_sha256'):
        p.error('invalid or ungated confirmation plan')
    original=str(Path(plan['binaries']['original']).resolve(strict=True));expanded=str(Path(plan['binaries']['expanded']).resolve(strict=True))
    hashes={'original':sha(original),'expanded':sha(expanded)}
    if hashes!=plan['sha256'] or original==expanded:p.error('confirmation binary mismatch')
    positives=set();ranked_positive=[];first_sha=None;first_gate_sha=None
    if a.session_kind=='second':
        if any(x is None for x in (a.first_results,a.first_gate,a.first_audit,a.first_guard,a.first_incidents,a.first_budget_snapshot)):
            p.error('second session requires independently audited first-session evidence and gate')
        first=[json.loads(x) for x in a.first_results.read_text(encoding='utf8').splitlines() if x.strip()]
        heads=[r for r in first if r.get('record_type')=='confirmation_protocol']
        ends=[r for r in first if r.get('record_type')=='confirmation_complete']
        if (len(heads)!=1 or heads[0].get('session_kind')!='first' or heads[0].get('plan_sha256')!=sha(a.plan) or
            len(ends)!=1 or ends[0].get('status') not in ('completed','budget_exhausted','reserved_for_second')):p.error('invalid first confirmation session')
        gate=json.loads(a.first_gate.read_text(encoding='utf8'))
        evidence={'rows':sha(a.first_results),'guard':sha(a.first_guard),'incidents':sha(a.first_incidents),'budget':sha(a.first_budget_snapshot)}
        if (gate.get('schema')!='expanded-first-session-gate/v1' or gate.get('status')!='pass' or
            gate.get('session_id')!=heads[0]['session_id'] or gate.get('plan_sha256')!=sha(a.plan) or
            gate.get('evidence_sha256')!=evidence or gate.get('binaries_sha256')!=hashes or gate.get('audit_sha256')!=sha(a.first_audit)):
            p.error('first-session independent gate or evidence mismatch')
        eligible=gate['eligible']
        if any(not isinstance(r.get('passes_practical_and_ci'),bool) for r in eligible):p.error('first gate missing criterion flags')
        ranked_positive=sorted((r for r in eligible if r['geometric_mean_ratio']<1),
                               key=lambda r:(not r['passes_practical_and_ci'],r['geometric_mean_ratio'],r['n'],r['batch'],r['candidate_id']))
        positives={(r['n'],r['batch'],r['candidate_id']) for r in ranked_positive}
        if len(positives)!=len(ranked_positive):p.error('duplicate first-session eligible candidate')
        first_sha=sha(a.first_results);first_gate_sha=sha(a.first_gate)
    elif any(x is not None for x in (a.first_results,a.first_gate,a.first_audit,a.first_guard,a.first_incidents,a.first_budget_snapshot)):
        p.error('first session cannot depend on prior results')
    session=uuid.uuid4().hex;a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='confirmation_protocol',session_id=session,session_kind=a.session_kind,plan_sha256=sha(a.plan),
                  first_results_sha256=first_sha,first_gate_sha256=first_gate_sha,binaries=plan['binaries'],sha256=hashes,gpu=a.gpu,rounds=10,
                  selection_seed=plan['selection_seed'],interpretation='paired device execution, not end-to-end'))
        terminal='completed'
        try:
            by_workload={(w['n'],w['batch']):w for w in plan['workloads']}
            plan_entries={(e['n'],e['batch'],e['candidate_id']):e for e in plan['confirmation_order']}
            entries=plan['confirmation_order'] if a.session_kind=='first' else [plan_entries[(e['n'],e['batch'],e['candidate_id'])] for e in ranked_positive]
            for entry in entries:
                item=by_workload[(entry['n'],entry['batch'])]
                n,batch=item['n'],item['batch'];candidate=item.get('candidate')
                if not candidate:continue
                cid=candidate['candidate_id'];req=candidate['request'];R=candidate['graph_repeats']
                if entry['candidate_id']!=cid or entry['role']!=item['role']:raise ValueError('confirmation order differs from plan')
                if a.session_kind=='second' and (item['role']!='screen_faster' or (n,batch,cid) not in positives):continue
                ratios=[]
                for round_no in range(10):
                    work_id=uuid.uuid4().hex
                    # Preserve the second half of the fixed confirmation stage
                    # for an independently guarded follow-up session.
                    if a.session_kind=='first' and b.state['phase_start_monotonic']+3600-time.monotonic() <= 2*(a.timeout+30):
                        b.checkpoint('confirmation_second_session_reserve',session_id=session,n=n,batch=batch,candidate_id=cid,round=round_no)
                        raise BudgetStop('reserved_for_second')
                    if not b.may_start('confirmation_pair',expected_upper_seconds=2*(a.timeout+30),work_id=work_id,
                                       session_id=session,session_kind=a.session_kind,n=n,batch=batch,candidate_id=cid,round=round_no):
                        raise BudgetStop()
                    order=search_controller.seeded_order(plan['selection_seed'],a.session_kind,n,batch,cid,'confirmation',round_no,42)
                    pair={}
                    try:
                        for ix,name in enumerate(order):
                            a.exe=original if name=='original' else expanded;a.exe_sha256=hashes[name if name=='original' else 'expanded'];a.session_id=session
                            w=dict(n=n,batch=batch,seed=42,config={},pattern='random',graph_repeats=R,
                                   warmup=5,warmup_ms=1000,repeats=31,study_mode='execute')
                            row=suite.run_one(a,w,('original' if name=='original' else cid,
                                                   {} if name=='original' else validation.request_env(req)),
                                              'confirmation_timing',round_no,ix)
                            emit(row);pair[name]=row
                        orig=validation.check(pair['original'],a.gpu,original=True,timing=True)
                        cand=validation.check(pair['candidate'],a.gpu,request=req,timing=True)
                        for result in (orig,cand):
                            if result.get('warmup_actual',0)<5 or result.get('warmup_wall_ms',0)<1000:raise ValueError('actual warmup insufficient')
                        if (search_controller.source_signature(orig,original=True)!=candidate['original_signature'] or
                            search_controller.source_signature(cand,original=False)!=candidate['candidate_signature']):
                            raise ValueError('confirmation source, plan or resource drift')
                        om=statistics.median(orig['exec_ms_samples']);cm=statistics.median(cand['exec_ms_samples']);ratio=cm/om
                        emit(dict(record_type='confirmation_pair',session_id=session,session_kind=a.session_kind,n=n,batch=batch,
                                  candidate_id=cid,round=round_no,order=order,original_guard_run_id=pair['original']['guard_run_id'],
                                  candidate_guard_run_id=pair['candidate']['guard_run_id'],original_median_ms=om,candidate_median_ms=cm,
                                  candidate_over_original=ratio))
                        ratios.append(ratio)
                        b.finish_work('completed',work_id=work_id,guard_run_ids=[pair[x]['guard_run_id'] for x in order])
                    except BaseException as error:
                        b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}')
                        raise
                emit(dict(record_type='confirmation_disposition',session_id=session,session_kind=a.session_kind,n=n,batch=batch,
                          candidate_id=cid,status='complete',paired_ratios=ratios,median_ratio=statistics.median(ratios)))
        except BudgetStop as error:
            terminal=str(error) if str(error)=='reserved_for_second' else 'budget_exhausted'
        except BaseException as error:
            emit(dict(record_type='confirmation_complete',session_id=session,status='error',error=f'{type(error).__name__}: {error}'))
            raise
        emit(dict(record_type='confirmation_complete',session_id=session,status=terminal))


if __name__=='__main__':raise SystemExit('use phase_gpu_guard; unguarded GPU execution is disabled')
