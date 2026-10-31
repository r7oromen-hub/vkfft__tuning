#!/usr/bin/env python3
"""Audit guard evidence shared by admission, search and confirmation logs."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def audit(executions, guard, incidents, gpu):
    errors = []
    if any(not isinstance(row, dict) for row in guard) or any(not isinstance(row, dict) for row in executions):
        return {'status':'fail', 'errors':['malformed guard or execution record'], 'counts':{},
                'scope':'guard/event linkage only; periodic sampling cannot rule out between-sample interference'}
    starts = [x for x in guard if x.get('event') == 'guard_start']
    ends = [x for x in guard if x.get('event') == 'guard_complete']
    if len(starts) != 1 or len(ends) != 1:
        errors.append('expected exactly one guard start and clean completion')
    gid = starts[0].get('guard_id') if starts else None
    if starts and (starts[0].get('gpu_uuid') != gpu or starts[0].get('own_signal_method') != 'pidfd_with_rechecked_PPID_exe_starttime'):
        errors.append('guard start device or process-ownership method mismatch')
    if ends and (ends[0].get('guard_id') != gid or ends[0].get('clean') is not True or ends[0].get('success') is not True):
        errors.append('guard completion is not clean')
    if incidents or any(x.get('incident') or x.get('event') in ('guard_abort', 'quarantined_run', 'termination_error') for x in guard):
        errors.append('guard incident or quarantine')
    events = defaultdict(list)
    indexed_events = defaultdict(list)
    delay_indices = []
    previous_sample_completed = None
    for index, event in enumerate(guard):
        if event.get('guard_id') != gid or event.get('gpu_uuid') != gpu:
            errors.append('guard event device or ID mismatch')
        if event.get('event') == 'sample':
            flag = event.get('idle_monitor_delay', False)
            if flag is not True and flag is not False:
                errors.append('invalid idle monitor delay flag')
            duration = event.get('query_wall_seconds')
            if 'query_wall_seconds' in event and not finite_nonnegative(duration):
                errors.append('invalid monitor query duration')
            completed = event.get('sample_completed_monotonic')
            if 'sample_completed_monotonic' in event:
                if not finite_nonnegative(completed):
                    errors.append('invalid sample completion time')
                elif previous_sample_completed is not None and completed < previous_sample_completed:
                    errors.append('sample completion clock regressed')
                else:
                    previous_sample_completed = completed
            if flag is True:
                delay_indices.append(index)
                device = event.get('gpu')
                memory = device.get('memory_used_mib') if isinstance(device, dict) else None
                sample_started = event.get('sample_started_monotonic')
                if (event.get('boundary') not in ('idle', 'between') or 'active_run' not in event or
                    event['active_run'] is not None or event.get('guard_run_id') is not None or
                    event.get('ok') is not False or event.get('violation_reasons') != [] or
                    'monitor_error' not in event or event['monitor_error'] is not None or
                    event.get('foreign_compute_pids') != [] or event.get('compute_apps') != [] or
                    not isinstance(device, dict) or device.get('uuid') != gpu or
                    not finite_nonnegative(memory) or memory > 2048 or
                    not finite_nonnegative(duration) or duration <= 2 or not finite_nonnegative(completed) or
                    not finite_nonnegative(sample_started) or sample_started > completed or
                    abs(completed - sample_started - duration) > 1e-6):
                    errors.append('invalid idle monitor delay evidence')
            elif (event.get('ok') is not True or event.get('violation_reasons') or event.get('foreign_compute_pids') or
                  event.get('monitor_error') is not None or not isinstance(event.get('gpu'), dict) or
                  event['gpu'].get('uuid') != gpu or (duration is not None and
                  (not finite_nonnegative(duration) or
                   ('idle_monitor_delay' in event and duration > 2)))):
                errors.append('unclean sample')
        active = event.get('active_run')
        if active is not None and not isinstance(active, dict):
            errors.append('malformed active run')
        rid = event.get('guard_run_id') or (active.get('guard_run_id') if isinstance(active,dict) else None)
        if rid:
            events[rid].append(event)
            indexed_events[rid].append((index,event))
    if delay_indices:
        guard_start = next((i for i,e in enumerate(guard) if e.get('event') == 'guard_start'), -1)
        guard_end = next((i for i,e in enumerate(guard) if e.get('event') == 'guard_complete'), -1)
        active_run = None
        active_start = None
        previous_transition = None
        intervals = []
        for index, event in enumerate(guard):
            kind = event.get('event')
            if kind in ('run_one_start', 'run_one_end'):
                transition = event.get('run_transition_monotonic')
                if not finite_nonnegative(transition) or (previous_transition is not None and transition < previous_transition):
                    errors.append('invalid run transition time')
                else:
                    previous_transition = transition
            if kind == 'run_one_start':
                active = event.get('active_run')
                rid = active.get('guard_run_id') if isinstance(active,dict) else None
                if not rid or active_run is not None: errors.append('overlapping or malformed run start')
                active_run = rid
                active_start = event.get('run_transition_monotonic')
            elif kind == 'run_one_end':
                rid = event.get('guard_run_id')
                if not rid or rid != active_run: errors.append('unmatched run end')
                end_time = event.get('run_transition_monotonic')
                if finite_nonnegative(active_start) and finite_nonnegative(end_time):
                    intervals.append((active_start, end_time))
                active_run = None
                active_start = None
            elif kind == 'sample' and event.get('idle_monitor_delay') is True:
                if active_run is not None or not guard_start < index < guard_end:
                    errors.append('idle monitor delay overlaps a run or lies outside guard')
        if active_run is not None: errors.append('run remains active at guard end')
        for index in delay_indices:
            sample = guard[index]
            started, completed = sample.get('sample_started_monotonic'), sample.get('sample_completed_monotonic')
            if finite_nonnegative(started) and finite_nonnegative(completed):
                if any(not (completed < start or started > end) for start,end in intervals):
                    errors.append('idle monitor delay interval overlaps a run')
    ids, counts = set(), Counter()
    for r in executions:
        if r.get('record_type') != 'execution':
            continue
        rid = r.get('guard_run_id')
        if not rid or rid in ids:
            errors.append('missing or duplicate execution run ID')
            continue
        ids.add(rid)
        if r.get('guard_id') != gid or r.get('guard_gpu_uuid') != gpu or r.get('guard_clean') is not True:
            errors.append(f'{rid}: guard provenance mismatch')
        ev = events.get(rid, [])
        for kind in ('run_one_start', 'run_one_end', 'own_harness_registered'):
            if sum(x.get('event') == kind for x in ev) != 1:
                errors.append(f'{rid}: missing or duplicate {kind}')
        for boundary in ('before', 'after'):
            if sum(x.get('event') == 'sample' and x.get('boundary') == boundary for x in ev) != 1:
                errors.append(f'{rid}: missing or duplicate {boundary} sample')
        reg = next((x for x in ev if x.get('event') == 'own_harness_registered'), None)
        identity = reg.get('identity') if reg else None
        command = r.get('command')
        if reg and (reg.get('pidfd_retained') is not True or not isinstance(identity,dict) or
                    not isinstance(command,list) or not command or identity.get('exe') != command[0]):
            errors.append(f'{rid}: process identity mismatch')
        # Old clean logs can lack monitor timestamps. A run started after a
        # tolerated idle delay must instead prove a fresh pre-launch sample.
        indexed = indexed_events.get(rid, [])
        positions = {kind: [i for i,e in indexed if e.get('event') == kind]
                     for kind in ('run_one_start', 'own_harness_registered', 'run_one_end')}
        before = [(i,e) for i,e in indexed if e.get('event') == 'sample' and e.get('boundary') == 'before']
        after = [(i,e) for i,e in indexed if e.get('event') == 'sample' and e.get('boundary') == 'after']
        if positions['run_one_start'] and any(i < positions['run_one_start'][0] for i in delay_indices):
            if (any(len(positions[k]) != 1 for k in positions) or len(before) != 1 or len(after) != 1 or
                not (positions['run_one_start'][0] < before[0][0] < positions['own_harness_registered'][0] <
                     after[0][0] < positions['run_one_end'][0])):
                errors.append(f'{rid}: delayed-idle run event order mismatch')
            if (len(before) != 1 or not finite_nonnegative(before[0][1].get('query_wall_seconds')) or
                before[0][1]['query_wall_seconds'] > 2 or
                not finite_nonnegative(before[0][1].get('sample_completed_monotonic')) or
                reg is None or not finite_nonnegative(reg.get('before_sample_age_seconds')) or
                reg['before_sample_age_seconds'] > 2 or
                not finite_nonnegative(reg.get('harness_launch_check_monotonic')) or
                not 0 <= reg['harness_launch_check_monotonic'] - before[0][1]['sample_completed_monotonic'] <= 2 or
                abs(reg['harness_launch_check_monotonic'] - before[0][1]['sample_completed_monotonic'] -
                    reg['before_sample_age_seconds']) > 1e-6):
                errors.append(f'{rid}: missing or stale successful before sample')
            start_events = [e for _,e in indexed if e.get('event') == 'run_one_start']
            end_events = [e for _,e in indexed if e.get('event') == 'run_one_end']
            if len(start_events) == len(end_events) == len(before) == 1 and reg is not None:
                start_time = start_events[0].get('run_transition_monotonic')
                end_time = end_events[0].get('run_transition_monotonic')
                sample_start = before[0][1].get('sample_started_monotonic')
                sample_end = before[0][1].get('sample_completed_monotonic')
                duration = before[0][1].get('query_wall_seconds')
                launch_check = reg.get('harness_launch_check_monotonic')
                if (not all(finite_nonnegative(v) for v in
                            (start_time, sample_start, sample_end, duration, launch_check, end_time)) or
                    not start_time <= sample_start <= sample_end <= launch_check <= end_time or
                    abs(sample_end - sample_start - duration) > 1e-6):
                    errors.append(f'{rid}: before sample time interval mismatch')
        start = next((x for x in ev if x.get('event') == 'run_one_start'), {}).get('active_run')
        if not isinstance(start,dict): start = {}
        if any(start.get(field) != r.get(field) for field in ('phase', 'workload_id', 'candidate_id', 'round', 'order')):
            errors.append(f'{rid}: start metadata mismatch')
        counts['executions'] += 1
    unmatched = set(events) - ids
    if unmatched:
        errors.append(f'{len(unmatched)} guarded run(s) missing execution records')
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'counts': dict(counts),
            'scope': 'guard/event linkage only; periodic sampling cannot rule out between-sample interference'}


def read(path):
    with path.open(encoding='utf8') as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True); p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--incidents', type=Path, required=True); p.add_argument('--gpu', required=True)
    a = p.parse_args(); result = audit(read(a.rows), read(a.guard), read(a.incidents), a.gpu)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
