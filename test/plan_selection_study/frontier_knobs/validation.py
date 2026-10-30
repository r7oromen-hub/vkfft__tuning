"""Fail-closed joint-knob audit layered on the existing execution checks."""
import copy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requests = load('frontier_knobs_requests', HERE / 'requests.py')
old = load('frontier_expanded_validation', HERE.parent / 'expanded_search' / 'validation.py')
# This module's compare() is the existing strict original-versus-instrumented
# source/config/resource check. Its run_suite import is used only by its CLI.
sys.path.insert(0, str(HERE.parent / 'original_compare'))
try:
    comparison = load('frontier_original_compare', HERE.parent / 'original_compare' / 'original_compare_controller.py')
finally:
    sys.path.pop(0)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check_knobs(result, lut, coalesced, *, original):
    """Check requested and resolved knobs and the actual per-kernel LUT state."""
    expected = requests.expected_config(lut, coalesced)
    need(result.get('requested_config') == expected, 'requested configuration mismatch')
    config = result['resolved_config']
    actual_lut, actual_coalesced = config['useLUT'], config['coalescedMemory']
    need(type(actual_lut) is int and actual_lut in (-1, 1), 'unexpected resolved LUT')
    need(type(actual_coalesced) is int and actual_coalesced > 0, 'unexpected resolved coalesced memory')
    if lut:
        need(actual_lut == lut, 'resolved LUT differs from explicit request')
    if coalesced:
        need(actual_coalesced == coalesced, 'resolved coalesced memory differs from explicit request')
    key = 'postinit_source_kernels' if original else 'precompile_kernels'
    kernels = result[key]
    need(len(kernels) == 1, 'expected one kernel descriptor')
    descriptor = kernels[0]['canonical_descriptor' if original else 'raw_descriptor']
    need(type(descriptor['LUT']) is int and descriptor['LUT'] == (1 if actual_lut == 1 else 0),
         'actual descriptor LUT differs from resolved config')
    size = descriptor['bufferLUTSize']
    # A single stage can legitimately have no LUT entries even with LUT enabled.
    need(type(size) is int and size >= 0, 'invalid LUT buffer size')
    return result


def check_execute(row, gpu, *, lut, coalesced, original=False, request=None, timing=False):
    """Keep every old guard, resource, descriptor, and numerical requirement."""
    if request is not None:
        n, batch = row['workload']['n'], row['workload']['batch']
        if requests.static_reason(n, batch, request):
            raise ValueError('statically rejected request')
        need((request['use_lut'], request['coalesced_memory']) == (lut, coalesced), 'request knobs mismatch')
    result = row['result']
    check_knobs(result, lut, coalesced, original=original)
    # The old check hard-codes the default requested_config; validate the real
    # field above, then normalize only a private copy for its other checks.
    normalized = copy.deepcopy(row)
    normalized['result']['requested_config'] = {'makeForwardPlanOnly': 1}
    old.check(normalized, gpu, original=original, request=request, timing=timing)
    return result


def check_describe(row, gpu, request):
    n, batch = row['workload']['n'], row['workload']['batch']
    if requests.static_reason(n, batch, request):
        raise ValueError('statically rejected request')
    check_knobs(row['result'], request['use_lut'], request['coalesced_memory'], original=False)
    old.check_describe(row, gpu, request)
    return row['result']


def check_matched_identity(original_row, experimental_row, replay_row, gpu, identity):
    """Audit same-knob original/default/replay; caller runs all three through guard."""
    lut, coalesced = identity['use_lut'], identity['coalesced_memory']
    workload_fields = ('n', 'batch', 'seed', 'pattern', 'graph_repeats', 'repeats')
    workloads = [tuple(row['workload'][key] for key in workload_fields)
                 for row in (original_row, experimental_row, replay_row)]
    need(workloads[0] == workloads[1] == workloads[2], 'identity workloads differ')
    original = check_execute(original_row, gpu, lut=lut, coalesced=coalesced, original=True)
    default = check_execute(experimental_row, gpu, lut=lut, coalesced=coalesced)
    replay = check_execute(replay_row, gpu, lut=lut, coalesced=coalesced, request=identity)
    comparison.compare(default, original)
    need(old.signature(default) == old.signature(replay), 'identity replay differs from experimental default')
    return {'status': 'pass', 'knobs': {'use_lut': lut, 'coalesced_memory': coalesced},
            'identity': identity, 'performance_denominator': 'original_no_knob_default'}


def check_performance_baseline(row, gpu, *, timing=False):
    return check_execute(row, gpu, lut=0, coalesced=0, original=True, timing=timing)
