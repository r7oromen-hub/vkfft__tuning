"""Pure CPU request validation and lazy enumeration; not GPU plan admission."""
import hashlib
import itertools
import json


SCHEMA = 'nonuniform-register-request/v1'
RADICES = (2, 4, 8, 16, 32)
REGISTERS = (2, 4, 8, 16, 32, 64, 128, 256)
PROFILES = ('nonuniform_active_only', 'uniform_compatibility')
STATUS = 'pending_scheduler_resource_and_numerical_validation'


def _integer(value):
    return type(value) is int


def _workload(n, batch):
    if not _integer(n) or n < 2 or n > 8192 or n & (n - 1):
        raise ValueError('n must be a power of two in 2..8192')
    if not _integer(batch) or not 1 <= batch <= 65536:
        raise ValueError('batch must be an integer in 1..65536')


def validate_request(request):
    """Validate the complete v1 request; return derived values, not device validity."""
    if type(request) is not dict or set(request) != {
            'schema', 'workload', 'radix', 'register_table', 'profile', 'grouped_batch'}:
        raise ValueError('request must contain exactly the v1 fields')
    if request['schema'] != SCHEMA:
        raise ValueError('unsupported request schema')
    workload = request['workload']
    if type(workload) is not dict or set(workload) != {'n', 'batch'}:
        raise ValueError('workload must contain exactly n and batch')
    n, batch = workload['n'], workload['batch']
    _workload(n, batch)
    radix = request['radix']
    if type(radix) is not list or not 1 <= len(radix) <= 32:
        raise ValueError('radix must be a nonempty list of at most 32 stages')
    product = 1
    for stage in radix:
        if not _integer(stage) or stage not in RADICES or product > n // stage:
            raise ValueError('unsupported radix or stage product exceeds n')
        product *= stage
    if product != n:
        raise ValueError('stage product must equal n')
    table = request['register_table']
    if type(table) is not list or len(table) != 33 or any(not _integer(v) for v in table):
        raise ValueError('register_table must be exactly 33 integers')
    active = set(radix)
    for stage in active:
        value = table[stage]
        if value not in REGISTERS or value < stage or value > n or value % stage or n % value:
            raise ValueError('invalid active radix register value')
    min_r = min(table[stage] for stage in active)
    max_r = max(table[stage] for stage in active)
    if n % min_r or any(max_r % table[stage] for stage in active):
        raise ValueError('incompatible active register envelope')
    profile = request['profile']
    if profile == 'nonuniform_active_only':
        if any(table[i] != 0 for i in range(33) if i not in active):
            raise ValueError('active-only profile requires all inactive slots to be zero')
    elif profile == 'uniform_compatibility':
        if min_r != max_r:
            raise ValueError('uniform compatibility requires one active register value')
        expected = [max_r if i in RADICES and max_r % i == 0 else 0 for i in range(33)]
        if table != expected:
            raise ValueError('uniform compatibility requires the complete frontier table')
    else:
        raise ValueError('unsupported table profile')
    group = request['grouped_batch']
    if not _integer(group) or group < 1 or group & (group - 1) or batch % group:
        raise ValueError('grouped_batch must be a power-of-two divisor of batch')
    return {'min_registers': min_r, 'max_registers': max_r,
            'planned_fft_threads': n // min_r}


def canonical_request(request):
    """Return the stable, versioned JSON identity after strict validation."""
    validate_request(request)
    return json.dumps(request, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def candidate_id(request):
    return hashlib.sha256(canonical_request(request).encode('utf8')).hexdigest()


def candidate_record(request):
    """Freeze the request and identify it; a successful result is only prospective."""
    derived = validate_request(request)
    encoded = json.dumps(request, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False)
    return {'candidate_id': hashlib.sha256(encoded.encode('utf8')).hexdigest(),
            'request': json.loads(encoded), **derived, 'status': STATUS,
            'validity': 'static contract only; not scheduler, resource or numerical admission'}


def _decompositions(remaining, prefix=()):
    if remaining == 1:
        yield prefix
        return
    for stage in RADICES:
        if remaining % stage == 0:
            yield from _decompositions(remaining // stage, prefix + (stage,))


def _groups(batch):
    value = 1
    while value <= batch:
        if batch % value == 0:
            yield value
        value *= 2


def enumerate_candidates(n, batch, profiles=PROFILES):
    """Lazily yield all static v1 candidates for the requested workload/profiles.

    The potentially large domain is never materialized. Distinct profiles remain
    distinct requests even when the resulting table contents happen to match.
    """
    _workload(n, batch)
    if (type(profiles) not in (tuple, list) or not profiles or
            any(type(p) is not str or p not in PROFILES for p in profiles) or
            len(set(profiles)) != len(profiles)):
        raise ValueError('profiles must be a nonempty unique list of supported profiles')
    profiles = tuple(profiles)

    def generate():
        for radix in _decompositions(n):
            active = sorted(set(radix))
            for profile in profiles:
                if profile == 'nonuniform_active_only':
                    values = [[r for r in REGISTERS if r >= stage and r <= n and
                               r % stage == 0 and n % r == 0] for stage in active]
                    assignments = itertools.product(*values)
                else:
                    assignments = ((r,) * len(active) for r in REGISTERS
                                   if r <= n and n % r == 0 and
                                   all(r >= stage and r % stage == 0 for stage in active))
                for assignment in assignments:
                    table = [0] * 33
                    if profile == 'nonuniform_active_only':
                        for stage, value in zip(active, assignment):
                            table[stage] = value
                    else:
                        value = assignment[0]
                        for stage in RADICES:
                            if value % stage == 0:
                                table[stage] = value
                    for group in _groups(batch):
                        request = {'schema': SCHEMA, 'workload': {'n': n, 'batch': batch},
                                   'radix': list(radix), 'register_table': list(table),
                                   'profile': profile, 'grouped_batch': group}
                        yield candidate_record(request)
    return generate()
