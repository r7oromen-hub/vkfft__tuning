#!/usr/bin/env python3
"""Independently audit one complete expanded-search candidate manifest.

This checks mathematical coverage and seeded *requests*, not device legality,
guard events, numerical correctness, timing, or budget. The caller must audit
those separately. Input may contain protocol and manifest_complete records.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random

RADICES = (2, 4, 8, 16, 32)
REGS = (2, 4, 8, 16, 32, 64)


def all_sequences(n):
    if n == 1:
        yield ()
        return
    for r in RADICES:
        if n % r == 0:
            for rest in all_sequences(n // r):
                yield (r,) + rest


def expected_requests(n, batch):
    groups = []
    g = 1
    while g <= batch:
        if batch % g == 0:
            groups.append(g)
        g *= 2
    for seq in all_sequences(n):
        for regs in REGS:
            for group in groups:
                yield {'radix': list(seq), 'logical_registers': regs, 'grouped_batch': group}


def identifier(req):
    data = json.dumps(req, sort_keys=True, separators=(',', ':')).encode('utf8')
    return hashlib.sha256(data).hexdigest()


def selected_order(eligible, seed, limit):
    """Replay the specified stage/structure/planned-layout sampling rule."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        r = row['request']
        key = (tuple(sorted(r['radix'])), row['planned_fft_threads'], r['grouped_batch'])
        grouped[len(r['radix'])][key].append(row['candidate_id'])
    rng = random.Random(seed)
    stages = sorted(grouped)
    rng.shuffle(stages)
    queues = {}
    for stage in stages:
        keys = sorted(grouped[stage])
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(grouped[stage][key])
        queues[stage] = deque(deque(grouped[stage][key]) for key in keys)
    result = []
    while len(result) < limit and any(queues.values()):
        for stage in stages:
            if len(result) >= limit:
                break
            if queues[stage]:
                bucket = queues[stage].popleft()
                result.append(bucket.popleft())
                if bucket:
                    queues[stage].append(bucket)
    return result


def audit(rows, n, batch, identity, seed=20261030, limit=64):
    errors, counts, eligible, selected = [], Counter(), [], []
    if n < 2 or n > 8192 or n & (n-1) or batch < 1 or batch > 65536:
        raise ValueError('n or batch outside the declared prototype domain')
    if not 0 <= limit <= 64:
        raise ValueError('selection limit outside 0..64')
    expected = iter(expected_requests(n, batch))
    seen = set()
    for index, row in enumerate(rows):
        want = next(expected, None)
        req = row.get('request')
        if req != want:
            errors.append(f'row {index}: missing, extra, reordered or changed request')
            # Subsequent rows remain checked, but do not claim full coverage.
        if not isinstance(req, dict):
            continue
        try:
            seq, regs, group = req['radix'], req['logical_registers'], req['grouped_batch']
            if set(req) != {'radix', 'logical_registers', 'grouped_batch'}:
                raise ValueError('request fields differ')
            if not seq or any(r not in RADICES for r in seq) or math.prod(seq) != n:
                raise ValueError('invalid ordered factorization')
            if regs not in REGS or group <= 0 or group & (group-1) or batch % group:
                raise ValueError('invalid register or group value')
            cid = identifier(req)
            if row.get('candidate_id') != cid:
                raise ValueError('candidate hash mismatch')
            if cid in seen:
                raise ValueError('duplicate request in mathematical manifest')
            seen.add(cid)
            reason = ('registers_exceed_fft_length' if regs > n else
                      'radix_exceeds_registers' if max(seq) > regs else None)
            status = 'rejected_static' if reason else 'pending_scheduler_and_device_validation'
            if row.get('status') != status or row.get('reason') != reason:
                raise ValueError('static status or reason mismatch')
            if row.get('planned_fft_threads') != (n // regs if reason is None else None):
                raise ValueError('planned FFT threads mismatch')
            sel = row.get('selection')
            if reason:
                if sel != 'rejected_static':
                    raise ValueError('statically rejected candidate marked selectable')
                counts['rejected_static'] += 1
            else:
                counts['static_eligible'] += 1
                if req == identity:
                    if sel != 'identity_control':
                        raise ValueError('identity not reserved for control')
                    counts['identity_control'] += 1
                elif sel in ('selected', 'not_selected'):
                    eligible.append(row)
                    counts[sel] += 1
                    if sel == 'selected':
                        selected.append((row.get('selection_index'), cid))
                else:
                    raise ValueError('invalid selection state')
            counts['mathematical_requests'] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'row {index}: {exc}')
    if next(expected, None) is not None:
        errors.append('manifest truncated: expected more mathematical requests')
    if identity is not None and counts['identity_control'] != 1:
        errors.append('expected exactly one representable identity request')
    expected_order = selected_order(eligible, seed, limit)
    indices = [x[0] for x in selected]
    if any(not isinstance(i, int) or isinstance(i, bool) for i in indices) or sorted(indices) != list(range(len(selected))):
        errors.append('selection indices must be unique and contiguous')
    else:
        if [cid for _, cid in sorted(selected)] != expected_order:
            errors.append('selected IDs do not reproduce pre-registered seeded sampling')
    if len(selected) > limit:
        errors.append('non-default selected count exceeds limit')
    return {'status': 'pass' if not errors else 'fail', 'n': n, 'batch': batch,
            'seed': seed, 'limit': limit, 'counts': dict(counts), 'errors': errors,
            'scope': 'manifest only; no claim of device validity, execution or performance'}


def audit_document(records, n, batch, identity, seed=20261030, limit=64):
    """Check explicit protocol and canonical content digest around candidate rows."""
    errors = []
    if len(records) < 2 or records[0].get('record_type') != 'manifest_protocol' or records[-1].get('record_type') != 'manifest_complete':
        errors.append('missing ordered manifest_protocol or manifest_complete')
        rows = [r for r in records if r.get('record_type') == 'candidate_manifest']
        header, footer = {}, {}
    else:
        header, footer = records[0], records[-1]
        rows = records[1:-1]
    if any(r.get('record_type') != 'candidate_manifest' for r in rows):
        errors.append('unexpected record in candidate section')
        rows = [r for r in rows if r.get('record_type') == 'candidate_manifest']
    if (header.get('n'), header.get('batch'), header.get('identity_request'),
            header.get('selection_seed'), header.get('limit')) != (n, batch, identity, seed, limit):
        errors.append('manifest protocol differs from audit parameters')
    h = hashlib.sha256()
    for row in rows:
        line = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n'
        h.update(line.encode('utf8'))
    selection_counts = Counter(row.get('selection') for row in rows)
    if (footer.get('total') != len(rows) or footer.get('selected') != selection_counts.get('selected', 0)
            or footer.get('counts') != dict(selection_counts) or footer.get('candidate_lines_sha256') != h.hexdigest()):
        errors.append('manifest footer counts or canonical candidate digest mismatch')
    result = audit(rows, n, batch, identity, seed, limit)
    result['errors'] = errors + result['errors']
    result['status'] = 'fail' if result['errors'] else 'pass'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--identity-json', type=Path)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--limit', type=int, default=64)
    args = p.parse_args()
    with args.manifest.open(encoding='utf8') as source:
        records = [json.loads(line) for line in source if line.strip()]
    identity = (json.loads(args.identity_json.read_text(encoding='utf8')) if args.identity_json else
                records[0].get('identity_request') if records else None)
    result = audit_document(records, args.n, args.batch, identity, args.seed, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
