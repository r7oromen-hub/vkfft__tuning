#!/usr/bin/env python3
"""Offline audit of raw, single-session nonuniform correctness evidence.

Passing evidence is scoped to this session. The caller-supplied campaign
deadline is not proof of an earlier or migrated total-budget ledger.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import admission
import candidate
import identity
import validation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('nonuniform_guard_evidence_audit',
    HERE.parent / 'expanded_search/audit_guard.py')
guard_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_audit)


def need(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def unique_object(pairs):
    result={}
    for key,value in pairs:
        need(key not in result, 'duplicate JSON object key')
        result[key]=value
    return result


def read_jsonl(path):
    with Path(path).open(encoding='utf8') as src:
        return [json.loads(line, object_pairs_hook=unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                for line in src if line.strip()]


def run_id(event):
    active = event.get('active_run')
    return event.get('guard_run_id') or (active.get('guard_run_id') if type(active) is dict else None)


def audit_guard_rows(rows, guard, incidents, gpu):
    """Audit every guard interval, including one terminal no-launch interval."""
    need(guard and type(guard[0].get('parent_pid')) is int, 'missing guard parent')
    parent, gid = guard[0]['parent_pid'], guard[0].get('guard_id')
    need(all(e.get('guard_id')==gid and e.get('gpu_uuid')==gpu for e in guard),
         'guard contains mixed device or guard identities')
    need(not incidents and not any(e.get('incident') or e.get('event') in
             ('incident','guard_abort','quarantined_run','termination_error') for e in guard),
         'guard incident, abort or quarantine')
    for event in guard:
        if event.get('event') != 'sample':
            continue
        device=event.get('gpu') or {}; apps=event.get('compute_apps')
        need(type(device) is dict and device.get('uuid') == gpu and number(device.get('memory_used_mib')) and
             device['memory_used_mib'] <= 2048 and type(apps) is list, 'invalid guard device sample')
        active=event.get('active_run')
        begin, finish, duration = (event.get(k) for k in ('sample_started_monotonic',
                                                           'sample_completed_monotonic','query_wall_seconds'))
        need(number(begin) and number(finish) and number(duration) and begin <= finish and
             abs(finish-begin-duration) <= 1e-6 and
             (duration <= 2 or (active is None and event.get('idle_monitor_delay') is True)),
             'invalid guard sample interval')
        if active is None:
            need(apps == [], 'GPU process outside guarded run')
        else:
            need(type(active) is dict, 'invalid active sample')
            rid=active.get('guard_run_id')
            registrations=[e for e in guard if e.get('event')=='own_harness_registered' and run_id(e)==rid]
            for app in apps:
                need(type(app) is dict and app.get('gpu_uuid')==gpu and app.get('own_harness') is True and
                     len(registrations)==1 and all(app.get(k)==registrations[0].get('identity',{}).get(k)
                         for k in ('pid','ppid','start_ticks','exe')), 'unregistered process in guard sample')
    actions = [x for x in rows if x.get('record_type') in ('execution', 'launch_denied')]
    denied = [x for x in actions if x.get('record_type') == 'launch_denied']
    need(len(denied) <= 1, 'multiple launch denials')
    ids = [x.get('guard_run_id') for x in actions]
    need(all(type(x) is str and x for x in ids) and len(ids) == len(set(ids)), 'duplicate or missing guard run ID')
    active_id=None; seen=set()
    for event in guard:
        kind=event.get('event'); rid=run_id(event)
        if kind=='run_one_start':
            need(active_id is None and rid in ids and rid not in seen, 'overlapping or unexpected guard run')
            active_id=rid; seen.add(rid)
        elif kind=='run_one_end':
            need(active_id is not None and rid==active_id, 'unmatched guard run end')
            active_id=None
        elif kind in ('sample','own_harness_registered','own_exit_identity_recheck'):
            need(rid==active_id, 'guard event is outside its active interval')
            if kind=='own_exit_identity_recheck':
                need(active_id is not None and event.get('proof') is True and event.get('pidfd_retained') is True,
                     'unproven exited child identity')
    need(active_id is None and seen==set(ids), 'guard runs do not partition transcript')
    launches={}
    for row in actions:
        rid = row['guard_run_id']
        indexed = [(i, e) for i, e in enumerate(guard) if run_id(e) == rid]
        def one(kind, boundary=None):
            found = [(i, e) for i, e in indexed if e.get('event') == kind and
                     (boundary is None or e.get('boundary') == boundary)]
            need(len(found) == 1, rid + ': missing or duplicate ' + kind)
            return found[0]
        si, start = one('run_one_start'); bi, before = one('sample', 'before')
        ai, after = one('sample', 'after'); ei, end = one('run_one_end')
        need(si < bi < ai < ei and end.get('clean') is True and row.get('guard_id') == gid and
             row.get('guard_gpu_uuid') == gpu and row.get('guard_clean') is True,
             rid + ': guard sequence or identity mismatch')
        active = start.get('active_run') or {}
        need(all(active.get(k) == row.get(k) for k in ('phase','workload_id','candidate_id','round','order')) and
             active.get('expected_exe') == row['command'][0] and end.get('status') == row.get('status'),
             rid + ': guard work differs')
        times=[start.get('run_transition_monotonic'), before.get('sample_started_monotonic'),
               before.get('sample_completed_monotonic'), row.get('started_monotonic'), row.get('ended_monotonic'),
               after.get('sample_started_monotonic'), after.get('sample_completed_monotonic'),
               end.get('run_transition_monotonic')]
        need(all(number(x) for x in times) and all(a <= b for a,b in zip(times,times[1:])),
             rid + ': guard and execution time spans differ')
        for sample in (before, after):
            need(sample.get('ok') is True and sample.get('gpu', {}).get('uuid') == gpu and
                 not sample.get('violation_reasons') and not sample.get('foreign_compute_pids') and
                 sample.get('monitor_error') is None, rid + ': unclean boundary sample')
        regs = [(i, e) for i, e in indexed if e.get('event') == 'own_harness_registered']
        if row['record_type'] == 'execution':
            need(len(regs) == 1 and bi < regs[0][0] < ai, rid + ': missing ordered pidfd registration')
            reg = regs[0][1]; proc = reg.get('identity') or {}
            need(reg.get('pidfd_retained') is True and type(proc.get('pid')) is int and proc['pid'] > 0 and
                 proc.get('ppid') == parent and type(proc.get('start_ticks')) is int and proc['start_ticks'] > 0 and
                 proc.get('exe') == row['command'][0], rid + ': invalid pidfd process identity')
            sample_end, launch = before.get('sample_completed_monotonic'), reg.get('harness_launch_check_monotonic')
            age = reg.get('before_sample_age_seconds')
            need(number(sample_end) and number(launch) and number(age) and 0 <= launch-sample_end <= 2 and
                 abs(launch-sample_end-age) <= 1e-6 and row['started_monotonic'] <= launch <= row['ended_monotonic'],
                 rid + ': stale before sample or launch outside execution')
            launches[rid]=launch
        else:
            need(not regs and row.get('status') == 'budget_denied_before_launch' and
                 row.get('reason') == 'absolute_stage_or_campaign_deadline' and
                 not any(k in row for k in ('stdout','stderr','result','returncode')),
                 rid + ': invalid no-launch evidence')
            need(before.get('compute_apps') == [] and after.get('compute_apps') == [],
                 rid + ': denied interval contains a GPU process')
            need(not any(e.get('event') in ('own_harness_registered','own_harness_sigterm') for _, e in indexed),
                 rid + ': denied run had a child')
    # Validate all samples and incidents before projecting the single no-launch
    # interval out of the legacy guard auditor, which requires a child for every run.
    for event in guard:
        if event.get('event') == 'sample' and run_id(event) in {x['guard_run_id'] for x in denied}:
            need(event.get('ok') is True and event.get('gpu', {}).get('uuid') == gpu and
                 not event.get('violation_reasons') and not event.get('foreign_compute_pids') and
                 event.get('compute_apps') == [] and event.get('monitor_error') is None, 'unclean denied-run sample')
    denied_ids = {x['guard_run_id'] for x in denied}
    for event in guard:
        if event.get('event')=='sample' and event.get('idle_monitor_delay') is True:
            begin, finish = event.get('sample_started_monotonic'), event.get('sample_completed_monotonic')
            need(number(begin) and number(finish), 'invalid idle delay interval')
            for row in denied:
                events=[e for e in guard if run_id(e)==row['guard_run_id']]
                start=next(e for e in events if e.get('event')=='run_one_start')['run_transition_monotonic']
                end=next(e for e in events if e.get('event')=='run_one_end')['run_transition_monotonic']
                need(finish < start or begin > end, 'idle monitor delay overlaps denied interval')
    projected = [e for e in guard if run_id(e) not in denied_ids]
    result = guard_audit.audit(rows, projected, incidents, gpu)
    need(result['status'] == 'pass', 'guard audit: ' + '; '.join(result.get('errors', [])))
    return {'status': 'pass', 'execution_count': len(actions)-len(denied), 'launch_denied_count': len(denied),
            'launch_monotonic':launches}


def audit_ledger(rows, ledger, contract, protocol):
    need(ledger and ledger[0].get('event') == 'start', 'missing ledger start')
    initial = ledger[0]['state']; start = initial['start']; seconds = initial['seconds']; boot = initial['boot']
    need(number(start) and type(seconds) in (int,float) and math.isfinite(seconds) and seconds == protocol['seconds'] and
         boot == protocol['boot_id'] and initial.get('inflight') is None, 'ledger origin mismatch')
    deadline = min(start + seconds, contract['campaign_deadline_monotonic'])
    need(number(deadline) and deadline > start, 'invalid absolute deadline')
    actions = [r for r in rows if r.get('record_type') in ('execution','launch_denied')]
    index = 1; previous = start
    for action in actions:
        need(index + 1 < len(ledger), 'missing ledger work events')
        begun, ended = ledger[index:index+2]; index += 2
        need(begun.get('event') == 'work_start' and ended.get('event') == 'work_finish', 'ledger work order')
        work = ended['identity']; w = action['workload']
        need(work == begun['state']['inflight'] and ended['state']['inflight'] is None and
             ended['status'] == action['status'] and
             (work['n'],work['batch'],work['candidate_id'],work['phase'],work['round']) ==
             (w['n'],w['batch'],action['candidate_id'],action['phase'],action['round']), 'ledger work mismatch')
        need(begun['state']['last'] + protocol['timeout'] + 30 < deadline, 'work started without deadline reserve')
        need(number(action.get('started_monotonic')) and number(action.get('ended_monotonic')) and
             begun['state']['last'] <= action['started_monotonic'] <= action['ended_monotonic'] <= ended['state']['last'],
             'work monotonic interval mismatch')
        if action['record_type'] == 'launch_denied':
            need(action['ended_monotonic'] + protocol['timeout'] + 3 >= deadline, 'premature launch denial')
    remaining = ledger[index:]
    need(not remaining or (len(remaining) == 1 and remaining[0].get('event') == 'denied'), 'unexpected ledger tail')
    if remaining:
        need(remaining[0]['state']['inflight'] is None and
             remaining[0]['state']['last'] + protocol['timeout'] + 30 >= deadline, 'premature budget denial')
    for event in ledger:
        state = event['state']; last = state['last']
        need(state.get('contract') == contract and state['start'] == start and state['seconds'] == seconds and
             state['boot'] == boot and number(last) and last >= previous, 'ledger contract or clock mismatch')
        previous = last
    need(ledger[-1]['state']['inflight'] is None, 'unresolved in-flight work')
    late = [r for r in actions if r['record_type'] == 'launch_denied']
    need(not (late and remaining) and (not late or actions[-1] is late[0]), 'invalid terminal denial')
    budget = [r for r in rows if r.get('record_type') == 'budget_exhausted']
    need(bool(budget) == bool(remaining or late) and len(budget) <= 1 and
         (not budget or rows[-1] is budget[0]), 'budget terminal evidence mismatch')
    return {'status':'pass', 'deadline_monotonic':deadline, 'budget_exhausted':bool(budget)}


def audit_transcript(rows, items, gpu, protocol):
    """Recompute identities and every completed 5-by-3 candidate matrix."""
    need(rows and rows[0].get('record_type') == 'session_start' and
         rows[0].get('candidate_count') == len(items) and rows[0].get('gpu') == gpu, 'session start mismatch')
    budget = rows[-1].get('record_type') == 'budget_exhausted'
    end = len(rows) - 1
    if not budget:
        need(rows[-1].get('record_type') == 'session_checks_complete' and
             rows[-1].get('status') == 'pending_independent_audit', 'missing session completion')
    pos = 1; reports = []; prior = {}; stopped = False; denied_work = None
    for item in items:
        req, cid = item['request'], item['candidate_id']; n, batch = req['workload']['n'], req['workload']['batch']
        key = (n,batch,cid)
        started = pos < end and rows[pos].get('record_type') == 'candidate_start'
        if started:
            start = rows[pos]; pos += 1
            need((start.get('n'),start.get('batch'),start.get('candidate_id')) == key and start.get('request') == req and
                 not stopped, 'candidate start/order mismatch')
        actions=[]; proofs=[]; proof_positions=[]
        while pos < end and rows[pos].get('record_type') in ('execution','launch_denied','identity_passed'):
            r=rows[pos]; pos += 1
            if r['record_type']=='identity_passed':
                proof_positions.append(len(actions)); proofs.append(r)
            else:
                actions.append(r)
        need(pos < end and rows[pos].get('record_type') == 'candidate_disposition', 'missing candidate disposition')
        disp=rows[pos]; pos+=1
        need((disp.get('n'),disp.get('batch'),disp.get('candidate_id')) == key and disp.get('request') == req,
             'candidate disposition mismatch')
        status=disp.get('status'); need(status in ('checks_passed_pending_audit','unsupported_identity','partial','not_started'),
                                      'failed or unknown candidate state')
        need(started or (status == 'not_started' and not actions and not proofs), 'unstarted candidate has work')
        if stopped: need(status == 'not_started' and not actions and not proofs, 'work after budget stop')
        first_stopped = not stopped and status in ('partial','not_started')
        if status in ('partial','not_started'):
            need(budget and disp.get('reason') == 'budget_denied', 'partial without budget boundary')
            stopped=True
        steps = [('describe',cid,'describe','random',42),('identity','original','execute','random',42),
                 ('identity','expanded_default','execute','random',42),('identity','identity','execute','random',42)]
        steps += [('validation',cid,'execute',pattern,seed) for pattern in validation.PATTERNS for seed in admission.SEEDS]
        need(len(actions) <= len(steps), 'extra candidate executions')
        described=original=default=replay=None; signature=None
        for i, row in enumerate(actions):
            phase, name, mode, pattern, seed = steps[i]; w=row['workload']
            need((row.get('phase'),row.get('candidate_id'),w.get('study_mode'),w.get('pattern'),w.get('seed')) ==
                 (phase,name,mode,pattern,seed) and (w.get('n'),w.get('batch')) == (n,batch) and
                 w.get('graph_repeats') == protocol['graph_repeats'] and w.get('repeats') == 1 and
                 w.get('warmup') == 0 and w.get('warmup_ms') == 0 and w.get('config') == {}, 'execution sequence mismatch')
            if row['record_type'] == 'launch_denied':
                need(i == len(actions)-1 and status in ('partial','not_started'), 'nonterminal launch denial')
                expected_req = (identity.representable(default)[0] if i == 3 and default is not None else
                                req if i == 0 or i >= 4 else None)
                need(i != 3 or expected_req is not None, 'denied identity lacks representable default')
                expected_env = candidate.request_environment(expected_req) if expected_req is not None else {}
                need(row.get('direct_environment') == expected_env, 'denied launch environment mismatch')
                continue
            if i == 0: described=validation.check_describe(row,gpu,req)
            elif i == 1: original=validation.check(row,gpu,original=True)
            elif i == 2:
                default=validation.check(row,gpu); admission.v2.comparison.compare(default,original)
            elif i == 3:
                base, why=identity.representable(default); need(base is not None, 'identity not representable: '+str(why))
                replay=validation.check(row,gpu,request=base)
            else:
                need(replay is not None, 'candidate validation without identity')
                result=validation.check(row,gpu,request=req); admission.compare_plan(described,result)
                current=admission.plan_signature(result)
                need(signature is None or signature == current, 'candidate signature drift'); signature=current
        executed = sum(r['record_type']=='execution' for r in actions)
        if status == 'not_started': need(executed == 0, 'not-started candidate executed')
        if status == 'partial': need(executed > 0, 'partial candidate never executed')
        if status == 'unsupported_identity':
            need(started and default is not None and executed == len(actions) == 3 and not proofs,
                 'unsupported identity lacks complete default evidence')
        if first_stopped:
            need(started and executed < len(steps), 'invalid first budget-stopped candidate')
            phase, name, _, _, _ = steps[executed]
            denied_work={'n':n,'batch':batch,'candidate_id':name,'phase':phase,'round':0}
        if default is not None:
            base, reason = identity.representable(default)
            if status == 'unsupported_identity':
                need(base is None and disp.get('reason') == reason and executed == 3 and not proofs,
                     'unsupported identity evidence mismatch')
            elif replay is not None:
                need(base is not None and len(proofs)==1, 'identity proof missing')
                proof=identity.compare_three(original,default,replay)
                need(proofs[0].get('proof') == proof and
                     (proofs[0].get('n'),proofs[0].get('batch'),proofs[0].get('candidate_id')) == key,
                     'identity proof mismatch')
                bases=(admission.plan_signature(original,original=True),admission.plan_signature(default))
                need((n,batch) not in prior or prior[(n,batch)] == bases, 'default plan drift')
                prior[(n,batch)]=bases
            else:
                need(base is not None or status == 'unsupported_identity',
                     'unrepresentable default was not recorded as unsupported')
        if status == 'checks_passed_pending_audit':
            need(executed == 19 and len(actions)==19 and len(proofs)==1 and disp.get('validation_count')==15,
                 'incomplete validation matrix')
        elif status != 'unsupported_identity':
            need(not proofs or replay is not None, 'unexpected identity proof')
        need(not proofs or proof_positions == [4], 'identity proof is not between replay and validation')
        reports.append({'n':n,'batch':batch,'candidate_id':cid,'status':status})
    need(pos == end and (not budget or stopped), 'unexpected transcript tail')
    return {'status':'pass','cases':reports,'counts':dict(Counter(r['status'] for r in reports)),
            'complete_session':not budget, 'denied_work':denied_work}


def audit_raw_rows(rows, contract, protocol, binaries):
    need(rows and rows[0].get('record_type')=='session_start', 'missing raw session')
    sid, gid = rows[0].get('session_id'), rows[0].get('guard_id')
    need(type(sid) is str and bool(sid) and type(gid) is str and bool(gid), 'missing session identity')
    need(all(r.get('contract') == contract and r.get('session_id') == sid and r.get('guard_id') == gid
             for r in rows), 'mixed sessions or execution contracts')
    allowed={'session_start','candidate_start','execution','launch_denied','identity_passed',
             'candidate_disposition','session_checks_complete','budget_exhausted'}
    need(all(r.get('record_type') in allowed for r in rows), 'unknown transcript record')
    actions=[r for r in rows if r['record_type'] in ('execution','launch_denied')]
    invocations=[r.get('invocation_id') for r in actions]
    need(all(type(v) is str and v for v in invocations) and len(set(invocations))==len(invocations),
         'missing or reused invocation identity')
    for row in actions:
        name=row['candidate_id']; w=row['workload']; original=name=='original'
        exe=Path(binaries['original' if original else 'expanded']).resolve(strict=True)
        expected=[str(exe),'--n',str(w['n']),'--batch',str(w['batch']),
                  '--seed',str(w['seed']),'--timing-mode','graph','--placement','outofplace',
                  '--graph-repeats',str(w['graph_repeats']),'--warmup','0','--warmup-ms','0',
                  '--repeats','1','--pattern',w['pattern'],'--study-mode',w['study_mode'],'--candidate-id',name]
        need(row.get('command')==expected and row.get('executable_sha256') ==
             contract['binaries']['original' if original else 'expanded'] and
             row.get('cuda_visible_devices')==protocol['gpu_uuid'] and row.get('round')==0 and row.get('order')==0 and
             row.get('candidate_environment_policy')=='nonuniform, expanded and direct prefixes cleared before explicit override',
             'execution command or binary mismatch')
        need(number(row.get('wall_seconds')) and math.isclose(row['wall_seconds'],
             row['ended_monotonic']-row['started_monotonic'],abs_tol=1e-6), 'execution duration mismatch')
        if row['record_type']=='execution':
            decoded=[json.loads(line,object_pairs_hook=unique_object,
                                parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                     for line in row['stdout'].splitlines() if line.strip()]
            need(len(decoded)==1 and type(decoded[0]) is dict and decoded[0]==row.get('result') and
                 row.get('returncode')==0 and row.get('status')=='ok' and
                 row.get('validation_errors')==[] and row['result'].get('candidate_id')==name,
                 'raw stdout, process status or result mismatch')
    return {'status':'pass','session_id':sid,'guard_id':gid}


def _record_artifacts(root, record, inputs):
    entries=record.get('artifacts')
    need(type(entries) is list, 'record artifact inventory missing')
    index={}
    for item in entries:
        name=item.get('path'); need(type(name) is str and name not in index, 'duplicate artifact')
        index[name]=item
    base=(root/'experiments/artifacts'/record['id']).resolve(strict=True)
    for raw in inputs:
        path=Path(raw).resolve(strict=True)
        need(path.is_relative_to(base), 'evidence is outside experiment artifacts')
        entry=index.get(path.relative_to(root).as_posix())
        need(entry and entry.get('bytes')==path.stat().st_size and entry.get('sha256')==admission.sha(path),
             'record artifact mismatch: '+str(path))


def audit_declared_command(command, args, contract, root):
    def from_root(value):
        path=Path(value)
        return (path if path.is_absolute() else root/path).resolve()
    need(type(command) is list and len(command) >= 2 and all(type(x) is str for x in command) and
         re.fullmatch(r'python(?:3(?:\.[0-9]+)?)?',Path(command[0]).name) is not None and
         from_root(command[1])==(HERE/'admission.py').resolve(), 'not a direct Python admission invocation')
    paths={'--protocol':args.protocol,'--manifest':args.manifest,'--original':args.original,
           '--expanded':args.expanded,'--generated-provenance':args.generated_provenance,
           '--build-attestation':args.build_attestation,'--compile-commands':args.compile_commands,
           '--original-dependencies':args.original_dependencies,'--output':args.rows,
           '--budget-ledger':args.budget,'--log':args.guard,'--incidents':args.incidents}
    flags=set(paths) | {'--campaign-deadline-monotonic','--lock'}
    need(len(command)==2+2*len(flags) and set(command[2::2])==flags and len(set(command[2::2]))==len(flags),
         'declared command has extra, missing or duplicate options')
    for flag, path in paths.items():
        need(command.count(flag)==1 and command.index(flag)+1 < len(command) and
             from_root(command[command.index(flag)+1])==Path(path).resolve(),
             'declared command path mismatch: '+flag)
    flag='--campaign-deadline-monotonic'
    need(command.count(flag)==1 and command.index(flag)+1<len(command) and
         float(command[command.index(flag)+1])==contract['campaign_deadline_monotonic'],
         'declared deadline mismatch')
    flag='--lock'
    need(command.count(flag)==1 and command.index(flag)+1<len(command) and
         from_root(command[command.index(flag)+1])==
         (Path('/run/lock')/('vkfft-study-'+contract['gpu']+'.lock')).resolve(), 'declared lock mismatch')


def audit_frozen_sources(root, parent, args, contract, generated):
    sources=[HERE/name for name in ('admission.py','candidate.py','candidate.hpp','identity.py',
                                    'validation.py','prepare_headers.py','CMakeLists.txt')]
    sources += [admission.FRONTIER, HERE.parent/'expanded_search/prepare_headers.py',
                HERE.parent/'expanded_search/device_lock.py',
                HERE.parent/'original_compare/original_compare_guard.py',
                HERE.parent/'original_compare/original_compare_controller.py',
                HERE.parent/'instrumentation/prepare_headers.py', HERE.parent/'instrumentation/trace.hpp',
                HERE.parent/'instrumentation/vkfft_instrumented_benchmark.cpp',
                HERE.parent/'instrumentation/cpu_reference.hpp',
                root/'experiments/research/nonuniform_register_admission_contract.json',
                Path(args.protocol),Path(args.manifest)]
    inputs=[args.protocol,args.manifest,args.original,args.expanded,args.generated_provenance,
            args.build_attestation,args.compile_commands,args.original_dependencies]
    expected={str(Path(p).resolve(strict=True)):admission.sha(p) for p in [*sources,*inputs]}
    expected.update(generated['generated_files'])
    need(contract.get('frozen_files')==expected, 'frozen source/input inventory differs')
    need(contract.get('tree_roots')=={'original':str((root/'VkFFT/vkFFT').resolve()),
         'generated':str((Path(args.generated_provenance).resolve().parent/'include').resolve())},
         'frozen tree roots differ')
    # Recheck all loaded legacy helpers against the execution parent, including
    # imports not individually retained in the controller's frozen_files map.
    legacy=[HERE.parent/'expanded_search'/name for name in
            ('run_suite.py','validation.py','search_controller.py','audit_admission.py','budget.py',
             'identity.py','manifest.py','enumerate_candidates.py','selection.py')]
    for path in [*sources,*legacy]:
        resolved=Path(path).resolve(strict=True)
        need(resolved.is_relative_to(root), 'source outside checkout')
        admission.v2.git(root,'ls-files','--error-unmatch',str(resolved))
        admission.v2.git(root,'diff','--quiet',parent,'--',str(resolved))


def audit_files(args):
    root=ROOT.resolve(strict=True)
    record_path=Path(args.record).resolve(strict=True)
    record=json.loads(record_path.read_text(encoding='utf8'))
    run=record.get('id')
    need(type(run) is str and record_path == (root/'experiments/records'/(run+'.json')).resolve(strict=True) and
         record.get('schema')=='git-experiment/v1' and record.get('status')=='completed' and record.get('returncode')==0,
         'experiment record not completed successfully')
    admission.v2.git(root,'ls-files','--error-unmatch',str(record_path))
    admission.v2.git(root,'diff','--quiet','HEAD','--',str(record_path))
    admission.v2.git(root,'merge-base','--is-ancestor',record['execution_commit'],'HEAD')
    for name in ('rows','guard','incidents','budget','execution'):
        need(Path(getattr(args,name)).is_file(), 'missing raw evidence: '+name)
    _record_artifacts(root,record,[args.rows,args.guard,args.incidents,args.budget,args.execution])
    declaration=json.loads(Path(args.execution).read_text(encoding='utf8'))
    need(declaration.get('id')==run and declaration.get('execution_commit')==record['execution_commit'] and
         declaration.get('command')==record.get('command') and declaration.get('protocol')==record.get('protocol'),
         'execution declaration/record mismatch')
    committed=json.loads(admission.v2.git(root,'show',record['execution_commit']+':'+
                                          record_path.relative_to(root).as_posix()))
    need(committed.get('command')==record.get('command') and committed.get('protocol')==record.get('protocol') and
         committed.get('id')==run and committed.get('status')=='declared', 'committed invocation differs')
    protocol=admission.load_protocol(args.protocol); items=admission.load_manifest(args.manifest)
    parent=admission.verify_declaration(root,record['execution_commit'],run,args.protocol)
    rows=read_jsonl(args.rows); guard=read_jsonl(args.guard); incidents=read_jsonl(args.incidents); ledger=read_jsonl(args.budget)
    need(rows, 'missing raw rows'); contract=rows[0]['contract']
    need(guard and guard[0].get('event') == 'guard_start' and guard[-1].get('event') == 'guard_complete' and
         guard[0].get('idle_seconds') == protocol['idle_seconds'] and
         guard[0].get('memory_limit_mib') == 2048 and guard[0].get('monitor_interval_seconds') == 1,
         'guard policy or terminal order mismatch')
    idle=[r for r in guard if r.get('event')=='idle_complete']
    need(len(idle)==1 and idle[0].get('clean') is True and number(idle[0].get('observed_idle_seconds')) and
         idle[0]['observed_idle_seconds'] >= protocol['idle_seconds'] and
         all(guard.index(idle[0]) < i for i,e in enumerate(guard) if e.get('event')=='run_one_start'),
         'missing or insufficient initial idle evidence')
    need(contract.get('schema')=='nonuniform-admission-execution/v1' and
         contract.get('execution_commit')==record['execution_commit'] and contract.get('code_commit')==parent and
         contract.get('gpu')==protocol['gpu_uuid'] and contract.get('boot_id')==protocol['boot_id'] and
         contract.get('hostname')==protocol['hostname'] and contract.get('machine_id')==protocol['machine_id'] and
         contract.get('protocol',{}).get('sha256')==admission.sha(args.protocol) and
         contract.get('manifest',{}).get('sha256')==admission.sha(args.manifest), 'contract or host identity mismatch')
    audit_declared_command(record['command'],args,contract,root)
    for name in ('protocol','manifest'):
        need(Path(contract[name]['path']).resolve(strict=True)==Path(getattr(args,name)).resolve(strict=True),
             name+' path differs from recorded evidence')
    vkfft=admission.v2.git(root/'VkFFT/vkFFT','rev-parse','HEAD')
    need(vkfft==contract.get('vkfft_commit') and admission.v2.git(root,'rev-parse',parent+':VkFFT')==vkfft,
         'VkFFT revision mismatch')
    binaries={'original':Path(args.original).resolve(strict=True),'expanded':Path(args.expanded).resolve(strict=True)}
    need(binaries['original']!=binaries['expanded'] and
         {k:admission.sha(v) for k,v in binaries.items()}==contract['binaries'], 'binary hashes differ')
    generated=admission.verify_generated(root,args.generated_provenance,vkfft)
    need(generated==contract['generated_provenance'], 'generated source or tree evidence mismatch')
    attest=admission.verify_attestation(args.build_attestation,parent,vkfft,generated,contract['binaries'],
                 args.compile_commands,args.original_dependencies,root/'VkFFT/vkFFT',root/'test/plan_selection_study/instrumentation')
    need(attest==contract['build_attestation'], 'build attestation mismatch')
    audit_frozen_sources(root,parent,args,contract,generated)
    admission.verify_frozen(contract)
    raw=audit_raw_rows(rows,contract,protocol,binaries)
    ga=audit_guard_rows(rows,guard,incidents,protocol['gpu_uuid'])
    la=audit_ledger(rows,ledger,contract,protocol)
    need(all(t + protocol['timeout'] + 3 < la['deadline_monotonic']
             for t in ga['launch_monotonic'].values()), 'child launch lacked deadline reserve')
    ta=audit_transcript(rows,items,protocol['gpu_uuid'],protocol)
    if la['budget_exhausted']:
        if ledger[-1]['event']=='denied':
            need(ledger[-1].get('identity')==ta['denied_work'], 'denied ledger work differs from transcript')
        else:
            need(ledger[-1].get('identity')==ta['denied_work'] and
                 ledger[-1].get('status')=='budget_denied_before_launch', 'late denial differs from transcript')
    paths=[args.record,args.execution,args.protocol,args.manifest,args.original,args.expanded,args.generated_provenance,
           args.build_attestation,args.compile_commands,args.original_dependencies,args.rows,args.guard,args.incidents,args.budget]
    hashes={str(Path(p).resolve(strict=True)):admission.sha(p) for p in paths}
    admission.verify_frozen(contract)
    need(hashes=={str(Path(p).resolve(strict=True)):admission.sha(p) for p in paths}, 'audit input changed during reading')
    return {'schema':'nonuniform-admission-audit/v1','status':'pass','gpu':protocol['gpu_uuid'],
            'execution_commit':record['execution_commit'],'raw':raw,'guard':ga,'ledger':la,'transcript':ta,
            'input_sha256':hashes,'scope':'raw single-session correctness evidence; not performance confirmation',
            'limitations':['build attestation is not independent binary reproduction',
                           'periodic monitoring cannot rule out between-sample interference',
                           'child lock-descriptor inheritance is checked from frozen controller code, not independently observed in raw logs',
                           'legacy imported helpers not listed by the controller cannot be proven continuously frozen',
                           'caller-supplied deadline origin and any prior campaign budget are not independently authenticated']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('record','execution','protocol','manifest','original','expanded','generated-provenance',
                 'build-attestation','compile-commands','original-dependencies','rows','guard','incidents','budget','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:
        report=audit_files(args)
    except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError,OSError,AttributeError,
            StopIteration,admission.subprocess.SubprocessError) as error:
        report={'schema':'nonuniform-admission-audit/v1','status':'fail','errors':[f'{type(error).__name__}: {error}']}
        try:
            observed=read_jsonl(args.rows)
            dispositions=[{'candidate_id':r.get('candidate_id'),'status':r.get('status')}
                          for r in observed if r.get('record_type')=='candidate_disposition']
            report['unverified_observed_dispositions']=dispositions
            report['disposition_note']='Observed only; failed audit has not authenticated these states.'
        except (OSError,ValueError,TypeError,KeyError):
            pass
    with args.output.open('x',encoding='utf8') as dest:
        json.dump(report,dest,indent=2,sort_keys=True,allow_nan=False);dest.write('\n')
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
