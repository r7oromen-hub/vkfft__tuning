#!/usr/bin/env python3
"""Paired direct-radix acceptance, invoked only through direct_gpu_guard.py."""
import argparse, hashlib, json, math, os, random
from pathlib import Path
import run_suite as suite

CASES = {
 'n16384': dict(n=16384, batch=1, old=[32,32,8,2], new=[32,8,32,2]),
 'n32768': dict(n=32768, batch=1, old=[32,32,8,4], new=[32,8,32,4]),
}
CONFIG = dict(aim_threads=128,register_boost=4,use_lut=1,register_boost_nonpow2=1)
PATTERNS=('zero','impulse','constant','quarter-tone','random')

def valid(row, args):
    if row.get('status')!='ok' or row.get('guard_clean') is not True: raise ValueError('failed or unguarded process')
    r=row.get('result',{})
    if r.get('status')!='ok' or r.get('gpu',{}).get('uuid') != args.gpu: raise ValueError('benchmark status or GPU UUID mismatch')
    w=row['workload']
    if (r.get('n'),r.get('batch'),r.get('seed')) != (w['n'],w['batch'],w['seed']): raise ValueError('workload or seed mismatch')
    expected=dict(makeForwardPlanOnly=1,aimThreads=128,registerBoost=4,useLUT=1,registerBoostNonPow2=1)
    if r.get('requested_config')!=expected: raise ValueError('requested configuration mismatch')
    if r.get('validation_output_initialization') != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward': raise ValueError('output poison missing')
    if r.get('input_unchanged') is not True or r.get('single_forward_input_unchanged') is not True: raise ValueError('input changed')
    for key in ('correctness','single_forward_correctness'):
        c=r.get(key,{})
        if c.get('passed') is not True or c.get('nonfinite_count') != 0 or c.get('failing_batches') != 0: raise ValueError(key+' failed')
    if r.get('descriptor_validation',{}).get('all_match') is not True: raise ValueError('descriptor validation failed')
    if r.get('graph_repeats') != args.graph_repeats or r.get('graph_kernel_count') != args.graph_repeats or r.get('graph_event_record_count') != 2:
        raise ValueError('graph execution shape mismatch')
    if row.get('direct_environment'):
        limits=r.get('direct_resource_validation',{})
        if not limits or limits['dynamic_shared_bytes'] > limits['function_dynamic_limit_bytes'] or limits['dynamic_shared_bytes'] + limits['static_shared_bytes'] > limits['device_shared_limit_bytes'] or limits['threads'] > limits['function_thread_limit']:
            raise ValueError('direct resource validation missing or failed')
    k=r.get('precompile_kernels',[])
    if len(k)!=1 or k[0].get('compiler_options_match') is not True: raise ValueError('precompile count/options mismatch')
    f=r.get('structural_features',{})
    if f.get('numAxisUploads')!=1 or len(f.get('kernels',[]))!=1: raise ValueError('requires one upload')
    return r

def signature(r):
    k=r['precompile_kernels'][0]
    return (k['canonical_descriptor'],k['source_sha256'],k['compiler_input'],k['actual_compiler_options'])

def override(radix,batch,block):
    return dict(VKFFT_DIRECT_RADIX=','.join(map(str,radix)), VKFFT_DIRECT_BATCH=str(batch), VKFFT_DIRECT_BLOCK=','.join(map(str,block)))

def validate_timing(result):
    samples=result.get('exec_ms_samples',[])
    if len(samples)!=31 or not all(math.isfinite(x) and x>0 for x in samples):
        raise ValueError('invalid timing samples')
    raw=result.get('raw_interval_ms_samples',[])
    if len(raw)!=31 or not all(math.isfinite(x) and x>=0.5 for x in raw):
        raise ValueError('raw event interval below 0.5 ms or invalid')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exe',required=True); p.add_argument('--gpu',required=True); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case',required=True,choices=sorted(CASES)); p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--timeout',type=int,default=300); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--graph-repeats',type=int,default=256)
    args=p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID')!=args.gpu: p.error('must be invoked by direct_gpu_guard.py for the same GPU UUID')
    if not 1<=args.rounds<=100 or not 1<=args.timeout<=3600 or not 1<=args.graph_repeats<=2048: p.error('invalid rounds, timeout or graph repetitions')
    args.exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest()
    case=CASES[args.case]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as out:
      def emit(row):
        out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); out.flush(); os.fsync(out.fileno())
      def one(name,env,phase,pattern,round_no=0,order=0,timing=False):
        workload=dict(n=case['n'],batch=case['batch'],seed=args.seed,config=CONFIG,pattern=pattern,graph_repeats=args.graph_repeats,
                      warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1)
        row=suite.run_one(args,workload,(name,env),phase,round_no,order)
        try:
          result=valid(row,args)
        except (ValueError,KeyError,TypeError) as error:
          row['status']='failed'; row.setdefault('validation_errors',[]).append(str(error)); emit(row)
          raise
        emit(row)
        return result
      baseline=one('baseline',{},'baseline','random')
      k=baseline['structural_features']['kernels'][0]
      if k['stageRadix']!=case['old']: raise ValueError('target baseline does not match historical structure')
      block=k['localSize']; group=k['groupedBatch']
      same=override(case['old'],group,block); new=override(case['new'],group,block)
      identity=one('identity',same,'identity','random')
      if signature(baseline)!=signature(identity): raise ValueError('identity replay not source/descriptor/options equivalent')
      for pattern in PATTERNS:
        for name,env in (('baseline',{}),('candidate',new)):
          result=one(name,env,'numerical',pattern)
          actual=result['structural_features']['kernels'][0]
          if actual['stageRadix']!=(case['old'] if name=='baseline' else case['new']) or actual['localSize']!=block:
            raise ValueError('stage or block mismatch')
      rng=random.Random(args.seed)
      for i in range(args.rounds):
        order=[('baseline',{}),('candidate',new)]
        if rng.randrange(2): order.reverse()
        for j,(name,env) in enumerate(order):
          result=one(name,env,'timing','random',i,j,True)
          actual=result['structural_features']['kernels'][0]
          if actual['stageRadix']!=(case['old'] if name=='baseline' else case['new']) or actual['localSize']!=block:
            raise ValueError('timing stage or block mismatch')
          validate_timing(result)
      emit(dict(record_type='controller_complete',case=args.case,rounds=args.rounds,exe_sha256=args.exe_sha256))

if __name__=='__main__': main()
