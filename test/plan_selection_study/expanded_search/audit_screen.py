#!/usr/bin/env python3
"""Recompute screen pairs from raw executions; does not certify guard or validity."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def close(a, b):
    return isinstance(a, (float, int)) and isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


def audit(rows):
    errors, counts = [], Counter()
    runs = {}
    for r in rows:
        if r.get('record_type') == 'execution':
            rid = r.get('guard_run_id')
            if not rid or rid in runs:
                errors.append('missing or duplicate execution guard_run_id')
            else:
                runs[rid] = r
    pairs = defaultdict(dict)
    for p in (r for r in rows if r.get('record_type') == 'screen_pair'):
        key = (p.get('n'), p.get('batch'), p.get('candidate_id'))
        round_no = p.get('round')
        if round_no not in (0, 1, 2) or round_no in pairs[key]:
            errors.append(f'{key}: invalid or duplicate screen round {round_no}')
            continue
        try:
            a = runs[p['original_guard_run_id']]; b = runs[p['candidate_guard_run_id']]
            if a is b or any(r.get('phase') != 'screen_timing' or r.get('round') != round_no for r in (a, b)):
                raise ValueError('run or phase mismatch')
            if a.get('candidate_id') != 'original' or b.get('candidate_id') != key[2]:
                raise ValueError('candidate identity mismatch')
            if p.get('order') not in (['original', 'candidate'], ['candidate', 'original']):
                raise ValueError('invalid pair order')
            seed_payload = json.dumps([20261030, key[0], key[1], key[2], 'screen_timing', round_no, 'random', 42], separators=(',', ':')).encode()
            expected_order = ['original', 'candidate']
            random.Random(int.from_bytes(hashlib.sha256(seed_payload).digest(), 'big')).shuffle(expected_order)
            if p['order'] != expected_order:
                raise ValueError('pair order differs from seeded protocol')
            if a.get('order') != p['order'].index('original') or b.get('order') != p['order'].index('candidate'):
                raise ValueError('recorded execution order mismatch')
            for r in (a, b):
                w, result = r['workload'], r['result']
                if ((w['n'], w['batch']) != key[:2] or w['seed'] != 42 or w['pattern'] != 'random' or
                        r.get('returncode') != 0 or r.get('status') != 'ok' or result.get('status') != 'ok' or
                        w['repeats'] != 31 or w['warmup'] < 5 or w['warmup_ms'] < 1000 or
                        result['warmup_actual'] < 5 or result['warmup_wall_ms'] < 1000 or
                        result['graph_repeats'] != w['graph_repeats'] or result['graph_kernel_count'] != w['graph_repeats'] or
                        result['graph_event_record_count'] != 2):
                    raise ValueError('workload, process or timing protocol mismatch')
                x, raw = result['exec_ms_samples'], result['raw_interval_ms_samples']
                if (len(x) != 31 or len(raw) != 31 or min(raw) < 0.5 or
                        not all(math.isfinite(v) and v > 0 for v in x + raw) or
                        not all(math.isclose(u, v/w['graph_repeats'], rel_tol=2e-6, abs_tol=1e-9) for u,v in zip(x, raw))):
                    raise ValueError('invalid timing samples')
            if a['workload']['graph_repeats'] != b['workload']['graph_repeats']:
                raise ValueError('graph repeats differ within pair')
            om = statistics.median(a['result']['exec_ms_samples']); cm = statistics.median(b['result']['exec_ms_samples'])
            ratio = cm / om
            if not (close(p['original_median_ms'], om) and close(p['candidate_median_ms'], cm) and close(p['candidate_over_original'], ratio)):
                raise ValueError('pair summary does not match raw samples')
            pairs[key][round_no] = ratio
            counts['verified_raw_pairs'] += 1
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f'{key}/{round_no}: {exc}')
    complete, seen = {}, set()
    for d in (r for r in rows if r.get('record_type') == 'candidate_disposition' and r.get('status') == 'screen_complete'):
        key = (d.get('n'), d.get('batch'), d.get('candidate_id'))
        if key in seen:
            errors.append(f'{key}: duplicate screen disposition')
            continue
        seen.add(key)
        rounds = pairs.get(key, {})
        if set(rounds) != {0, 1, 2} or len(d.get('pair_ratios', [])) != 3:
            errors.append(f'{key}: screen disposition has incomplete rounds')
            continue
        values = [rounds[i] for i in range(3)]
        if not all(close(x, y) for x, y in zip(values, d['pair_ratios'])) or not close(statistics.median(values), d.get('median_ratio')):
            errors.append(f'{key}: screen disposition summary mismatch')
            continue
        complete[key] = statistics.median(values)
    counts['screen_complete'] = len(complete)
    counts['screen_pairs_without_complete_disposition'] = sum(len(v) for k, v in pairs.items() if k not in complete)
    by_workload = defaultdict(list)
    for (n, batch, cid), value in complete.items():
        by_workload[(n, batch)].append((value, cid))
    best = []
    for (n, batch), values in sorted(by_workload.items()):
        value, cid = min(values)
        best.append({'n': n, 'batch': batch, 'best_nondefault_candidate': cid,
                     'best_nondefault_screen_ratio': value,
                     'screen_fallback_ratio': min(1.0, value),
                     'screen_selected': cid if value < 1 else 'default'})
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'counts': dict(counts),
            'workload_best_screen_only': best,
            'scope': 'raw timing arithmetic only; requires separate guard, manifest, budget, correctness, selection and independent confirmation audits; no validated speedup claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('rows', type=Path); args = p.parse_args()
    with args.rows.open(encoding='utf8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    result = audit(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
