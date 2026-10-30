#!/usr/bin/env python3
"""Audit guard evidence shared by admission, search and confirmation logs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def audit(executions, guard, incidents, gpu):
    errors = []
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
    for event in guard:
        if event.get('guard_id') != gid or event.get('gpu_uuid') != gpu:
            errors.append('guard event device or ID mismatch')
        if event.get('event') == 'sample':
            if (event.get('ok') is not True or event.get('violation_reasons') or event.get('foreign_compute_pids') or
                    event.get('monitor_error') is not None or event.get('gpu', {}).get('uuid') != gpu):
                errors.append('unclean sample')
        rid = event.get('guard_run_id') or (event.get('active_run') or {}).get('guard_run_id')
        if rid:
            events[rid].append(event)
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
        if reg and (reg.get('pidfd_retained') is not True or reg.get('identity', {}).get('exe') != r.get('command', [None])[0]):
            errors.append(f'{rid}: process identity mismatch')
        start = next((x for x in ev if x.get('event') == 'run_one_start'), {}).get('active_run', {})
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
