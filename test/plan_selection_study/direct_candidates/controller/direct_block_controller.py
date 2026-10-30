#!/usr/bin/env python3
"""One fixed-radix block-group comparison, run only by direct_block_gpu_guard.py."""
import argparse, hashlib, json, os, random
from pathlib import Path
import run_suite as suite
from direct_controller import CONFIG, PATTERNS, valid, signature, override, validate_timing

CASES={'n256':256,'n1024':1024}

def predict_block(baseline, group):
    f=baseline['structural_features']; k=f['kernels'][0]
    d=baseline['precompile_kernels'][0]['raw_descriptor']
    size=baseline['resolved_config']['sharedMemorySize']
    n=baseline['n']; local=k['localSize']; swapped=d['axisSwapped']; baseline_group=k['groupedBatch']
    if n not in CASES.values() or f['numAxisUploads']!=1 or d['fftDim']!=n or d['complexSize']!=8 or swapped not in (0,1):
        raise ValueError('baseline layout outside predicted 1D power-of-two scope')
    if group not in (1,2,4) or baseline['batch']%group or size<=0 or local[2]!=1:
        raise ValueError('invalid group, batch or baseline layout')
    threads=local[1] if swapped else local[0]
    if (local[0] if swapped else local[1])!=baseline_group or threads<=0:
        raise ValueError('baseline groupedBatch/localSize inconsistent')
    # Splitter's post-injection swap predicate for this restricted path.
    block=[group,threads,1] if group>1 and group*n < size//8 else [threads,group,1]
    return block,threads

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exe',required=True); p.add_argument('--gpu',required=True); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case',required=True,choices=sorted(CASES)); p.add_argument('--group',required=True,type=int,choices=(1,2,4))
    p.add_argument('--batch',type=int,default=16); p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--timeout',type=int,default=300); p.add_argument('--seed',type=int,default=42); p.add_argument('--graph-repeats',type=int,default=256)
    args=p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID')!=args.gpu: p.error('must be invoked by direct_block_gpu_guard.py for the same GPU')
    if not 1<=args.batch<=65536 or args.batch%args.group or not 1<=args.rounds<=100 or not 1<=args.timeout<=3600 or not 1<=args.graph_repeats<=2048: p.error('invalid parameters')
    args.exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest(); n=CASES[args.case]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as out:
      def emit(row):
        out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); out.flush(); os.fsync(out.fileno())
      def one(name,env,phase,pattern,round_no=0,order=0,timing=False):
        w=dict(n=n,batch=args.batch,seed=args.seed,config=CONFIG,pattern=pattern,graph_repeats=args.graph_repeats,
               warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1)
        row=suite.run_one(args,w,(name,env),phase,round_no,order)
        try: result=valid(row,args)
        except (ValueError,KeyError,TypeError) as e:
          row['status']='failed'; row.setdefault('validation_errors',[]).append(str(e)); emit(row); raise
        emit(row); return result
      baseline=one('baseline',{},'baseline','random')
      k=baseline['structural_features']['kernels'][0]; radix=k['stageRadix']; baseline_block=k['localSize']
      identity=one('identity',override(radix,k['groupedBatch'],baseline_block),'identity','random')
      if signature(baseline)!=signature(identity): raise ValueError('identity replay differs in source/descriptor/options')
      block,threads=predict_block(baseline,args.group)
      env=override(radix,args.group,block)
      def assert_shape(r,expected_group,expected_block):
        kernel=r['structural_features']['kernels'][0]
        if kernel['stageRadix']!=radix or kernel['groupedBatch']!=expected_group or kernel['localSize']!=expected_block:
          raise ValueError('radix, groupedBatch or launch block mismatch')
        raw=r['precompile_kernels'][0]['raw_descriptor']
        fft_threads=kernel['localSize'][1] if raw['axisSwapped'] else kernel['localSize'][0]
        if fft_threads!=threads: raise ValueError('FFT thread dimension changed')
      for pattern in PATTERNS:
        for name,request,expected_group,expected_block in [('baseline',{},k['groupedBatch'],baseline_block),('candidate',env,args.group,block)]:
          assert_shape(one(name,request,'numerical',pattern),expected_group,expected_block)
      rng=random.Random(args.seed)
      for i in range(args.rounds):
        order=[('baseline',{},k['groupedBatch'],baseline_block),('candidate',env,args.group,block)]
        if rng.randrange(2): order.reverse()
        for j,(name,request,expected_group,expected_block) in enumerate(order):
          result=one(name,request,'timing','random',i,j,True)
          assert_shape(result,expected_group,expected_block); validate_timing(result)
      emit(dict(record_type='controller_complete',case=args.case,group=args.group,rounds=args.rounds,baseline_block=baseline_block,
                candidate_block=block,fft_threads=threads,radix=radix,exe_sha256=args.exe_sha256))
if __name__=='__main__': main()
