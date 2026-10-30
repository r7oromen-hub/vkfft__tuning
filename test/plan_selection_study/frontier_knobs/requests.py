"""Versioned, offline joint request identity and invocation construction."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_v2_manifest', HERE.parent / 'frontier_search' / 'manifest.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

SCHEMA = 'frontier-knobs-request/v1'
LUTS = (-1, 0, 1)  # off, automatic, on
COALESCED = (0, 16, 32, 64, 128)  # automatic or bytes
FIELDS = {'radix', 'logical_registers', 'grouped_batch', 'use_lut', 'coalesced_memory'}


def integer(value):
    return type(value) is int


def validate_knobs(lut, coalesced):
    if not integer(lut) or lut not in LUTS or not integer(coalesced) or coalesced not in COALESCED:
        raise ValueError('unsupported LUT or coalesced-memory request')


def validate_shape(n, batch, request):
    if not integer(n) or not 2 <= n <= 8192 or n & (n-1):
        raise ValueError('n must be a power of two in 2..8192')
    if not integer(batch) or not 1 <= batch <= 65536:
        raise ValueError('batch outside 1..65536')
    if not isinstance(request, dict) or set(request) != FIELDS:
        raise ValueError('joint request has unexpected fields')
    validate_knobs(request['use_lut'], request['coalesced_memory'])
    radix, r, group = request['radix'], request['logical_registers'], request['grouped_batch']
    if (not isinstance(radix, list) or not radix or len(radix) > 32 or
        any(not integer(x) or x not in (2, 4, 8, 16, 32) for x in radix) or
        not integer(r) or r not in v2.REGISTERS or
        not integer(group) or group < 1 or group & (group-1) or batch % group):
        raise ValueError('outside the joint prototype request domain')
    product = 1
    for factor in radix:
        product *= factor
    if product != n:
        raise ValueError('radix product does not match n')


def static_reason(n, batch, request):
    validate_shape(n, batch, request)
    r = request['logical_registers']
    return ('registers_exceed_fft_length' if r > n else
            'radix_exceeds_registers' if max(request['radix']) > r else None)


def candidate_id(n, batch, request):
    validate_shape(n, batch, request)
    data = {'schema': SCHEMA, 'workload': {'n': n, 'batch': batch}, 'request': request}
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def enumerate_requests(n, batch):
    """Stream the full Cartesian inventory, including inherited static rejects."""
    for base in v2.candidates(n, batch):
        for lut, coalesced in itertools.product(LUTS, COALESCED):
            request = dict(base['request'], use_lut=lut, coalesced_memory=coalesced)
            yield dict(base, candidate_id=candidate_id(n, batch, request), request=request,
                       scope='power-of-two 1D FP32 C2C forward out-of-place, one upload, boost 1, joint LUT/coalesced')


def inventory_summary(n, batch):
    """Count the complete Cartesian domain without materializing its rows."""
    counts = Counter(row['reason'] or 'pending' for row in v2.candidates(n, batch))
    multiplier = len(LUTS) * len(COALESCED)
    totals = {reason: count * multiplier for reason, count in sorted(counts.items())}
    return {'schema': 'frontier-knobs-inventory/v1', 'workload': {'n': n, 'batch': batch},
            'enumerated_count': sum(totals.values()), 'pending_count': totals.pop('pending', 0),
            'static_rejection_reasons': totals, 'static_rejected_count': sum(totals.values()),
            'validity': 'static screening only; scheduler and device validation required'}


def invocation(n, batch, request, *, inject=True):
    if static_reason(n, batch, request):
        raise ValueError('static rejection cannot be invoked')
    env = ({'VKFFT_EXPANDED_RADIX': ','.join(map(str, request['radix'])),
            'VKFFT_EXPANDED_REGS': str(request['logical_registers']),
            'VKFFT_EXPANDED_BATCH': str(request['grouped_batch'])} if inject else {})
    return {'env': env, 'args': ['--use-lut', str(request['use_lut']),
                                 '--coalesced-memory', str(request['coalesced_memory'])]}


def expected_config(lut, coalesced):
    validate_knobs(lut, coalesced)
    out = {'makeForwardPlanOnly': 1}
    if lut:
        out['useLUT'] = lut
    if coalesced:
        out['coalescedMemory'] = coalesced
    return out


def comparison_contract(n, batch, identity):
    """Describe, but do not execute, matched controls and the performance baseline."""
    matched = invocation(n, batch, identity, inject=False)
    replay = invocation(n, batch, identity, inject=True)
    return {
        'original_no_knob_performance_denominator': {'binary': 'original', 'env': {}, 'args': []},
        'matched_original_default': dict(binary='original', **matched),
        'matched_experimental_default': dict(binary='experimental', **matched),
        'matched_identity_replay': dict(binary='experimental', **replay),
    }
