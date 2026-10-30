"""Determine whether the uniform-register request can express a default."""
import math


def representable(default):
    f = default['structural_features']
    if f['numAxisUploads'] != 1 or len(f['kernels']) != 1:
        return None, 'requires_one_upload_and_kernel'
    k = f['kernels'][0]
    d = default['precompile_kernels'][0]['raw_descriptor']
    radix = k['stageRadix']; r = d['logical_registers_per_thread']; group = k['groupedBatch']
    if d['registerBoost'] != 1:
        return None, 'register_boost_not_one'
    if r not in (2,4,8,16,32,64) or d['min_logical_registers_per_thread'] != r:
        return None, 'nonuniform_or_unsupported_register_envelope'
    regs = d['registers_per_radix']
    expected = [r if i in (2,4,8,16,32) and r % i == 0 else 0 for i in range(33)]
    if regs != expected:
        return None, 'nonuniform_register_table'
    if math.prod(radix) != default['n'] or group < 1 or group & (group-1) or default['batch'] % group:
        return None, 'unsupported_radix_or_batch'
    if default['n'] % r or any(x not in (2,4,8,16,32) or x > r or r % x for x in radix):
        return None, 'unsupported_radix_register_relation'
    swap=d['axisSwapped']; local=k['localSize']
    if (swap not in (0,1) or len(local)!=3 or local[2]!=1 or
        local[1 if swap else 0] != default['n']//r or local[0 if swap else 1] != group):
        return None, 'unsupported_local_layout'
    return {'radix':radix, 'logical_registers':r, 'grouped_batch':group}, None
