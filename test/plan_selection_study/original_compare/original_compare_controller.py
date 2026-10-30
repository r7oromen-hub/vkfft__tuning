#!/usr/bin/env python3
"""Paired original VkFFT header vs joint default, run only through the guard."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import run_suite as suite

PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')

def freeze(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(row, args, workload, variant, count):
    expected_env = args.winner_env if variant == 'winner' else {}
    if row.get('status') != 'ok' or row.get('guard_clean') is not True or row.get('direct_environment') != expected_env:
        raise ValueError('process failed, unguarded, or incorrect injection')
    r = row.get('result', {})
    if r.get('status') != 'ok' or r.get('gpu', {}).get('uuid') != args.gpu:
        raise ValueError('benchmark status or GPU UUID mismatch')
    if r.get('header_instrumentation') is not (variant != 'original'):
        raise ValueError('incorrect header instrumentation for binary')
    if (r.get('n'), r.get('batch'), r.get('seed'), r.get('pattern')) != (workload['n'], workload['batch'], workload['seed'], workload['pattern']):
        raise ValueError('workload identity mismatch')
    if r.get('requested_config') != {'makeForwardPlanOnly': 1}:
        raise ValueError('not default request')
    if r.get('validation_output_initialization') != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward':
        raise ValueError('poison protocol mismatch')
    if r.get('input_unchanged') is not True or r.get('single_forward_input_unchanged') is not True:
        raise ValueError('input changed')
    for key in ('correctness', 'single_forward_correctness'):
        v = r.get(key, {})
        if v.get('passed') is not True or v.get('nonfinite_count') != 0 or v.get('failing_batches') != 0 or v.get('checked_elements') != workload['n'] * workload['batch']:
            raise ValueError(key + ' incomplete or failed')
    shape = r.get('structural_features', {})
    if shape.get('numAxisUploads') != 1 or len(shape.get('kernels', [])) != 1:
        raise ValueError('one upload required')
    if r.get('graph_repeats') != args.graph_repeats or r.get('graph_kernel_count') != args.graph_repeats or r.get('graph_event_record_count') != 2:
        raise ValueError('graph shape mismatch')
    for field in ('exec_ms_samples', 'raw_interval_ms_samples'):
        values = r.get(field, [])
        if len(values) != count or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values):
            raise ValueError('invalid ' + field)
    if count == 31 and min(r['raw_interval_ms_samples']) < .5:
        raise ValueError('raw interval below 0.5 ms')
    if variant != 'original':
        if r.get('descriptor_validation', {}).get('all_match') is not True or len(r.get('precompile_kernels', [])) != 1 or r['precompile_kernels'][0].get('compiler_options_match') is not True:
            raise ValueError('joint precompile validation missing')
        if variant == 'winner':
            k=shape['kernels'][0]; limits=r.get('direct_resource_validation', {})
            if (k.get('stageRadix') != args.winner_radix or k.get('groupedBatch') != args.winner_group or k.get('localSize') != args.winner_block):
                raise ValueError('winner actual plan does not match request')
            if (not limits or limits['dynamic_shared_bytes'] > limits['function_dynamic_limit_bytes'] or limits['dynamic_shared_bytes'] + limits['static_shared_bytes'] > limits['device_shared_limit_bytes'] or limits['threads'] > limits['function_thread_limit']):
                raise ValueError('winner compiled resource limit failed')
    else:
        if len(r.get('postinit_source_kernels', [])) != 1 or r.get('init_trace', {}).get('available') is not False:
            raise ValueError('original retained-source/trace metadata missing')
    return r

def signature(result, variant):
    p=(result['postinit_source_kernels'] if variant == 'original' else result['precompile_kernels'])[0]
    return (result['structural_features'],result['compiled_diagnostics'],p['canonical_descriptor'],p['source_sha256'],p['source_bytes'])

def compare(a, b):
    """a is joint, b original: compare actual structures and source fingerprint."""
    if a.get('structural_features') != b.get('structural_features'):
        raise ValueError('actual structural_features differ')
    if a.get('compiled_diagnostics') != b.get('compiled_diagnostics'):
        raise ValueError('compiled kernel resource attributes differ')
    x, y = dict(a['resolved_config']), dict(b['resolved_config'])
    # Original retains source only until after this resolved snapshot.
    if x.pop('keepShaderCode', None) != 0 or y.pop('keepShaderCode', None) != 1 or x != y:
        raise ValueError('resolved configuration differs beyond source retention')
    p, q = a['precompile_kernels'][0], b['postinit_source_kernels'][0]
    if p.get('source_sha256') != q.get('source_sha256') or p.get('source_bytes') != q.get('source_bytes') or not p.get('source_sha256'):
        raise ValueError('kernel source fingerprint differs')
    if p.get('canonical_descriptor') != q.get('canonical_descriptor'):
        raise ValueError('precompile and original postinit descriptors differ')
    expected = q.get('expected_compiler_input', {})
    if expected.get('options') != p.get('actual_compiler_options') or expected.get('original_compile_header_sha256') != p.get('compiler_input', {}).get('original_compile_header_sha256'):
        raise ValueError('expected original compiler options/header differ')
    if q.get('actual_compiler_options', 'missing') is not None:
        raise ValueError('original actual NVRTC options are unavailable, expected null')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-exe', required=True); p.add_argument('--joint-sha256', required=True); p.add_argument('--original-exe', required=True)
    p.add_argument('--gpu', required=True); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--n', required=True, type=int); p.add_argument('--batch', required=True, type=int)
    p.add_argument('--graph-repeats', required=True, type=int)
    p.add_argument('--winner-radix', required=True, help='comma-separated actual winning stage sequence')
    p.add_argument('--winner-group', required=True, type=int)
    p.add_argument('--winner-block', required=True, help='comma-separated exact CUDA localSize')
    p.add_argument('--rounds', type=int, default=10); p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu:
        p.error('requires original_compare_guard for this GPU UUID')
    if args.n < 4 or args.n > 8192 or args.n & (args.n - 1) or not 1 <= args.batch <= 65536 or not 1 <= args.graph_repeats <= 2048 or not 2 <= args.rounds <= 100 or not 1 <= args.timeout <= 3600 or not 0 <= args.seed <= 0xffffffff:
        p.error('invalid workload, repetitions, rounds, timeout or seed')
    try:
        args.winner_radix=[int(v) for v in args.winner_radix.split(',')]
        args.winner_block=[int(v) for v in args.winner_block.split(',')]
        if (not args.winner_radix or any(v not in (2,4,8,16,32) for v in args.winner_radix) or math.prod(args.winner_radix) != args.n or len(args.winner_block) != 3 or any(v <= 0 for v in args.winner_block) or math.prod(args.winner_block) > 1024 or args.winner_group < 1 or args.batch % args.winner_group):
            raise ValueError()
    except ValueError:
        p.error('invalid winner radix, group or block')
    args.winner_env={'VKFFT_DIRECT_RADIX':','.join(map(str,args.winner_radix)), 'VKFFT_DIRECT_BATCH':str(args.winner_group), 'VKFFT_DIRECT_BLOCK':','.join(map(str,args.winner_block))}
    binaries = {'joint': str(Path(args.joint_exe).resolve(strict=True)), 'original': str(Path(args.original_exe).resolve(strict=True))}
    if binaries['joint'] == binaries['original']:
        p.error('two distinct binaries required')
    hashes = {k:freeze(v) for k,v in binaries.items()}
    if hashes['joint'] != args.joint_sha256.lower():
        p.error('joint executable does not match the recorded winner binary SHA-256')
    binaries['winner']=binaries['joint'];hashes['winner']=hashes['joint']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.output.open('x', encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); out.flush(); os.fsync(out.fileno())
        emit(dict(record_type='protocol', n=args.n, batch=args.batch, gpu=args.gpu, seed=args.seed,
                  graph_repeats=args.graph_repeats, rounds=args.rounds, binaries=binaries, sha256=hashes, winner_environment=args.winner_env,
                  caveat='original NVRTC options are expected from source, not captured; initialization times are not comparable'))
        def one(variant, phase, pattern, round_no, order, timing=False):
            args.exe=binaries[variant]; args.exe_sha256=hashes[variant]
            w=dict(n=args.n,batch=args.batch,seed=args.seed,config={},pattern=pattern,graph_repeats=args.graph_repeats,
                   warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1)
            env=args.winner_env if variant == 'winner' else {}
            row=suite.run_one(args,w,(variant,env),phase,round_no,order)
            try: result=validate(row,args,w,variant,w['repeats'])
            except (ValueError,KeyError,TypeError,IndexError) as error:
                row['status']='failed';row.setdefault('validation_errors',[]).append(str(error));emit(row);raise
            emit(row); return result
        stable_signatures={}
        def pair(phase,pattern,round_no,timing):
            order=['joint','original','winner'];rng.shuffle(order);results={}
            for i, variant in enumerate(order): results[variant]=one(variant,phase,pattern,round_no,i,timing)
            try:
                compare(results['joint'],results['original'])
                for variant in order:
                    current=signature(results[variant],variant)
                    if variant in stable_signatures and stable_signatures[variant] != current:
                        raise ValueError(variant+' plan changed across runs')
                    stable_signatures[variant]=current
            except (ValueError,KeyError,TypeError,IndexError) as error:
                emit(dict(record_type='comparison_failure',phase=phase,pattern=pattern,round=round_no,error=str(error)));raise
            original=results['original']; winner=results['winner']
            emit(dict(record_type='original_winner_comparison',phase=phase,pattern=pattern,round=round_no,
                      original_structure=original['structural_features'],winner_structure=winner['structural_features'],
                      original_compiled_resources=original['compiled_diagnostics'],winner_compiled_resources=winner['compiled_diagnostics'],
                      original_source_sha256=original['postinit_source_kernels'][0]['source_sha256'],
                      winner_source_sha256=winner['precompile_kernels'][0]['source_sha256']))
            return results
        for pattern in PATTERNS: pair('numerical',pattern,0,False)
        ratios=[];winner_ratios=[]
        for i in range(args.rounds):
            results=pair('timing','random',i,True)
            ratios.append(statistics.median(results['joint']['exec_ms_samples'])/statistics.median(results['original']['exec_ms_samples']))
            winner_ratios.append(statistics.median(results['winner']['exec_ms_samples'])/statistics.median(results['original']['exec_ms_samples']))
        emit(dict(record_type='controller_complete', status='ok', joint_over_original_median_ratios=ratios,
                  joint_over_original_median=statistics.median(ratios),winner_over_original_median_ratios=winner_ratios,
                  winner_over_original_median=statistics.median(winner_ratios),note='execution only; no initialization-cost comparison'))

if __name__ == '__main__': main()
