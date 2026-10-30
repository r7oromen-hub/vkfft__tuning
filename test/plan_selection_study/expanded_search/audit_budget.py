#!/usr/bin/env python3
"""Offline replay of the budget ledger; does not touch the device."""
import argparse
import json
import math
from pathlib import Path


def audit(ledger, protocol, execution_rows=None):
    errors = []
    phases = list(protocol['stage_budget_seconds'].items())
    total = sum(x[1] for x in phases)
    if not ledger or ledger[0].get('event') != 'budget_start':
        return {'status': 'fail', 'errors': ['missing budget_start'], 'overrun_seconds': None}
    first = ledger[0]['state']
    start = first.get('start_monotonic')
    boot = first.get('boot_id')
    expected_phases = [[name, duration] for name, duration in phases]
    prior = None
    for i, row in enumerate(ledger):
        try:
            state = row['state']; now = row['monotonic']; event = row['event']
            if not math.isfinite(now) or state['last_monotonic'] != now or (prior and now < prior['monotonic']):
                raise ValueError('monotonic time inconsistent')
            if (state['start_monotonic'] != start or state['boot_id'] != boot or state['phases'] != expected_phases or
                    state['total_seconds'] != total or state['global_deadline_monotonic'] != start + total):
                raise ValueError('budget policy, start or boot changed')
            if state['phase'] not in dict(phases):
                raise ValueError('unknown phase')
            phase_end = state['phase_start_monotonic'] + dict(phases)[state['phase']]
            if state['phase_deadline_monotonic'] != phase_end:
                raise ValueError('phase deadline mismatch')
            if prior:
                prev = prior['state']; phase_index = [x[0] for x in phases].index(prev['phase'])
                if prev['stopped'] and not state['stopped']:
                    raise ValueError('stopped state resumed')
                if event == 'phase_advance':
                    if prev.get('inflight') is not None or phase_index + 1 >= len(phases) or state['phase'] != phases[phase_index + 1][0]:
                        raise ValueError('invalid phase transition')
                elif state['phase'] != prev['phase']:
                    raise ValueError('phase changed without transition')
                if event == 'work_start_allowed':
                    upper = row['expected_upper_seconds']; remaining = row['remaining_seconds']
                    if prev.get('inflight') is not None or prev['stopped'] or not 0 <= upper < remaining:
                        raise ValueError('work started without reserve or with pending work')
                    if state.get('inflight') is None or state['inflight']['kind'] != row['kind']:
                        raise ValueError('in-flight checkpoint missing')
                    # The decision snapshot precedes the fsynced event by a small amount.
                    actual = min(start + total - now, prev['phase_deadline_monotonic'] - now)
                    if abs(actual - remaining) > 1 or actual <= 0:
                        raise ValueError('work started outside budget')
                elif event == 'work_finished':
                    if prev.get('inflight') is None or state.get('inflight') is not None:
                        raise ValueError('finished work without matching in-flight state')
                    if row.get('kind') != prev['inflight']['kind'] or row.get('identity') != prev['inflight']['identity']:
                        raise ValueError('finished identity mismatch')
                elif event not in ('work_start_denied', 'checkpoint', 'phase_advance', 'budget_stop'):
                    raise ValueError('unknown ledger event')
                if event not in ('work_start_allowed', 'work_finished') and state.get('inflight') != prev.get('inflight'):
                    raise ValueError('in-flight state changed without start/finish')
            prior = row
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'ledger line {i+1}: {exc}')
    if execution_rows is not None:
        recorded = [r.get('guard_run_id') for r in execution_rows if r.get('record_type') == 'execution']
        budgeted = []
        for row in ledger:
            if row.get('event') != 'work_finished':
                continue
            if row.get('guard_run_ids') is not None:
                budgeted.extend(row['guard_run_ids'])
            elif row.get('guard_run_id') is not None:
                budgeted.append(row['guard_run_id'])
        if (None in recorded or None in budgeted or len(set(recorded)) != len(recorded) or
                len(set(budgeted)) != len(budgeted) or set(recorded) != set(budgeted)):
            errors.append('budget work_finished and guarded execution IDs do not match one-to-one')
    last = ledger[-1]
    elapsed = last['monotonic'] - start
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'elapsed_seconds': elapsed,
            'global_remaining_seconds_at_last_event': max(0, total-elapsed),
            'overrun_seconds': max(0, elapsed-total), 'phase': last['state'].get('phase'),
            'inflight': last['state'].get('inflight'),
            'scope': 'recorded ledger consistency only; verify ledger start predates guard, builds and device reservation separately; monotonic ledger is not tamper-proof'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--rows', type=Path, nargs='+', help='all admission execution JSONL files for run-by-run budget reconciliation')
    args = p.parse_args()
    with args.ledger.open(encoding='utf8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    execution = None
    if args.rows:
        execution = []
        for path in args.rows:
            with path.open(encoding='utf8') as source:
                execution.extend(json.loads(line) for line in source if line.strip())
    result = audit(rows, json.loads(args.protocol.read_text(encoding='utf8')), execution)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
