"""Persistent fail-closed wall-clock budget for an exclusive GPU study.

The first start is the beginning of device occupancy, including builds and
guarded idle waits. Time continues while the process is down; recovery requires
the same Linux boot and nondecreasing monotonic time. A phase boundary prevents
starting new work, but the caller must drain the item already in flight.
"""
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

PHASES = (('admission', 5400), ('search', 12600), ('confirmation', 7200), ('extension', 3600))
TOTAL = sum(duration for _, duration in PHASES)


def utc():
    return datetime.now(timezone.utc).isoformat()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()


class Budget:
    def __init__(self, directory, *, clock=time.monotonic, get_boot=boot_id):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'budget_state.json'
        self.ledger = self.root / 'budget_ledger.jsonl'
        self.clock = clock
        self.get_boot = get_boot
        if self.path.exists() or self.ledger.exists():
            if not self.ledger.exists():
                raise RuntimeError('budget ledger missing; cannot safely resume')
            lines = self.ledger.read_text(encoding='utf8').splitlines()
            if not lines:
                raise RuntimeError('budget ledger empty; cannot safely resume')
            try:
                self.state = json.loads(lines[-1])['state']
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError('budget ledger tail invalid; manual audit required') from error
            self._check()
        else:
            self.state = None

    def _check(self):
        s = self.state
        if s['boot_id'] != self.get_boot() or self.clock() < s['last_monotonic']:
            raise RuntimeError('boot changed or monotonic clock regressed; manual audit required')
        if s['phases'] != [list(x) for x in PHASES] or s['total_seconds'] != TOTAL:
            raise RuntimeError('budget policy changed')

    def _commit(self, event, **details):
        now = self.clock()
        self.state['last_monotonic'] = now
        record = dict(event=event, utc=utc(), monotonic=now, **details,
                      state=self.state.copy())
        with self.ledger.open('a', encoding='utf8') as out:
            out.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            out.flush(); os.fsync(out.fileno())
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf8') as out:
            json.dump(self.state, out, sort_keys=True, allow_nan=False)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(tmp, self.path)

    def start(self):
        if self.state is not None:
            raise RuntimeError('budget already started; resume existing state')
        now = self.clock()
        self.state = dict(version=1, boot_id=self.get_boot(), phases=[list(x) for x in PHASES],
                          total_seconds=TOTAL, start_monotonic=now, last_monotonic=now,
                          global_deadline_monotonic=now+TOTAL, phase='admission',
                          phase_start_monotonic=now, phase_deadline_monotonic=now+PHASES[0][1],
                          completed=[], stopped=False, inflight=None)
        self._commit('budget_start')

    def remaining(self):
        self._check()
        s = self.state; now = self.clock()
        phase_limit = dict(PHASES)[s['phase']]
        return min(s['start_monotonic'] + TOTAL - now,
                   s['phase_start_monotonic'] + phase_limit - now)

    def may_start(self, kind, *, expected_upper_seconds, **identity):
        """Persist every decision before starting a logical item or GPU build."""
        remaining = self.remaining()
        if not isinstance(expected_upper_seconds, (int, float)) or not 0 <= expected_upper_seconds <= TOTAL:
            raise ValueError('invalid conservative work duration')
        if self.state.get('inflight') is not None:
            raise RuntimeError('unresolved in-flight item: audit before any new work')
        allowed = not self.state['stopped'] and remaining > expected_upper_seconds
        if allowed:
            self.state['inflight'] = dict(kind=kind, identity=identity, started_monotonic=self.clock(),
                                          expected_upper_seconds=expected_upper_seconds)
        self._commit('work_start_allowed' if allowed else 'work_start_denied',
                     kind=kind, remaining_seconds=remaining, expected_upper_seconds=expected_upper_seconds, **identity)
        return allowed

    def finish_work(self, status, **details):
        self._check()
        item = self.state.get('inflight')
        if item is None:
            raise RuntimeError('no in-flight item')
        elapsed = self.clock() - item['started_monotonic']
        self.state['inflight'] = None
        self._commit('work_finished', kind=item['kind'], identity=item['identity'], status=status,
                     elapsed_seconds=elapsed, overrun_seconds=max(0.0, -self.remaining()), **details)

    def checkpoint(self, kind, **details):
        self._check()
        self._commit('checkpoint', kind=kind, remaining_seconds=self.remaining(), **details)

    def advance(self, reason):
        self._check()
        if self.state.get('inflight') is not None:
            raise RuntimeError('cannot advance with work in flight')
        old = self.state['phase']; index = [p[0] for p in PHASES].index(old)
        self.state['completed'].append(old)
        if index == len(PHASES)-1 or self.clock() >= self.state['start_monotonic'] + TOTAL:
            self.state['stopped'] = True
            self._commit('budget_stop', reason=reason, previous_phase=old)
            return False
        self.state['phase'] = PHASES[index+1][0]
        self.state['phase_start_monotonic'] = self.clock()
        self.state['phase_deadline_monotonic'] = self.state['phase_start_monotonic'] + PHASES[index+1][1]
        self._commit('phase_advance', reason=reason, previous_phase=old)
        return True
