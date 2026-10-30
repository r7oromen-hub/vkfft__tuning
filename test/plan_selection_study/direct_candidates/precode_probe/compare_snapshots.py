#!/usr/bin/env python3
"""Compare pre-codegen and precompile snapshots in probe JSONL output."""
import argparse
import json
from pathlib import Path


def compare(records):
    changes = {}
    unchanged = set()
    pairs = 0
    cases = []
    for record in records:
        if record.get('status') != 'ok':
            raise ValueError(f"non-success record: {record.get('status')!r}")
        if record.get('study_mode') != 'describe' or record.get('describe_only') is not True:
            raise ValueError('only describe records are accepted')
        counters = record['init_trace']['counters']
        if any(counters[name] != 0 for name in ('nvrtc_compile', 'module_load', 'driver_kernel_launch')):
            raise ValueError('describe record has compiler, module, or launch activity')
        if record.get('descriptor_validation', {}).get('all_match') is not True:
            raise ValueError('precompile/postinitialize canonical descriptor validation failed')
        kernels = record['precompile_kernels']
        if not kernels:
            raise ValueError('no kernel snapshots')
        if record['init_trace']['stage_totals']['shader_codegen']['calls'] != len(kernels):
            raise ValueError('shader codegen/snapshot count mismatch')
        seen = set()
        for kernel in kernels:
            key = (kernel['axis'], kernel['upload'])
            if key in seen:
                raise ValueError(f'duplicate axis/upload: {key}')
            seen.add(key)
            pre, post = kernel['precodegen_raw_descriptor'], kernel['raw_descriptor']
            if pre.keys() != post.keys():
                raise ValueError(f'descriptor schema mismatch: {key}')
            if (pre['axis'], pre['upload']) != key or (post['axis'], post['upload']) != key:
                raise ValueError(f'descriptor key mismatch: {key}')
            case = {'n': record.get('n'), 'batch': record.get('batch'), 'axis': key[0], 'upload': key[1]}
            for field in pre:
                if pre[field] != post[field]:
                    changes.setdefault(field, []).append({'case': case, 'precodegen': pre[field], 'precompile': post[field]})
                else:
                    unchanged.add(field)
            pairs += 1
        cases.append({'n': record.get('n'), 'batch': record.get('batch'), 'kernels': len(kernels)})
    if not pairs:
        raise ValueError('no records')
    return {'record_count': len(cases), 'kernel_pairs': pairs, 'cases': cases,
            'changed_fields': changes, 'unchanged_in_all_observed_pairs': sorted(unchanged - changes.keys()),
            'interpretation': 'Observed equality is sample-specific; this does not establish general pre-codegen invariants.'}


def compare_control(probe, control):
    """Require matching requests and generated source/structure for each paired record."""
    if len(probe) != len(control) or not probe:
        raise ValueError('probe/control record counts differ or are empty')
    compare(probe)
    compared = []
    for p, c in zip(probe, control):
        for field in ('status', 'study_mode', 'n', 'batch', 'requested_config', 'placement', 'backend', 'precision',
                      'cuda_runtime_version', 'cuda_driver_version', 'nvrtc_version'):
            if field not in p or field not in c or p[field] != c[field]:
                raise ValueError(f'probe/control {field} mismatch')
        for field in ('uuid', 'compute_capability'):
            if p['gpu'][field] != c['gpu'][field]:
                raise ValueError(f'probe/control GPU {field} mismatch')
        if c['status'] != 'ok' or c['study_mode'] != 'describe' or c.get('describe_only') is not True:
            raise ValueError('control must be successful describe')
        if c.get('descriptor_validation', {}).get('all_match') is not True:
            raise ValueError('control descriptor validation failed')
        counters = c['init_trace']['counters']
        if any(counters[name] != 0 for name in ('nvrtc_compile', 'module_load', 'driver_kernel_launch')):
            raise ValueError('control describe has compiler, module, or launch activity')
        def indexed(record):
            values = record['precompile_kernels']
            result = {(k['axis'], k['upload']): k for k in values}
            if len(result) != len(values):
                raise ValueError('duplicate control kernel key')
            return result
        pk, ck = indexed(p), indexed(c)
        if pk.keys() != ck.keys():
            raise ValueError('probe/control kernel keys differ')
        if c['init_trace']['stage_totals']['shader_codegen']['calls'] != len(ck):
            raise ValueError('control shader codegen/snapshot count mismatch')
        for key in pk:
            for field in ('raw_descriptor', 'canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input'):
                if pk[key][field] != ck[key][field]:
                    raise ValueError(f'probe/control mismatch in {key}: {field}')
        compared.append({'n': p['n'], 'batch': p['batch'], 'kernels': len(pk)})
    return {'record_count': len(compared), 'cases': compared,
            'matched': ['requested_config', 'raw_descriptor', 'canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input'],
            'interpretation': 'Paired describe outputs agree; compiler and execution equivalence are not tested here.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='+', help='probe JSONL files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, nargs='+', help='ordered, request-matched baseline describe JSONL files')
    args = parser.parse_args()
    def read(paths):
        records = []
        for path in paths:
            with path.open(encoding='utf8') as source:
                records.extend(json.loads(line) for line in source if line.strip())
        return records
    records = read(args.input)
    result = compare(records)
    if args.control:
        result['baseline_comparison'] = compare_control(records, read(args.control))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
