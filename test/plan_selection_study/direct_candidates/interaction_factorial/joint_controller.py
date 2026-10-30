#!/usr/bin/env python3
"""Guarded, paired 2x2 radix/group confirmation, plus default and identity."""
import argparse, hashlib, json, math, os, random, statistics
from pathlib import Path
import core_validation as core
import run_suite as suite

RADICES = {'A': [8,4,4,4,2], 'B': [8,8,2,4,2]}
SEEDS = (42, 53, 71)
PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exe',required=True); p.add_argument('--gpu',required=True)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--timeout',type=int,default=300); p.add_argument('--order-seed',type=int,default=20261030)
    args=p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu:
        p.error('requires the joint GPU guard and the matching device')
    if not 10<=args.rounds<=100 or not 1<=args.timeout<=3600: p.error('invalid rounds/timeout')
    args.exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest()
    args.graph_repeats=64
    args.output.parent.mkdir(parents=True,exist_ok=True)
    requests={name+str(g): {'radix':radix, 'grouped_batch':g, 'block':[g,128,1]}
              for name,radix in RADICES.items() for g in (4,8)}
    with args.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='protocol',n=1024,batch=16,gpu=args.gpu,exe_sha256=args.exe_sha256,
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seeds=SEEDS,rounds=args.rounds,order_seed=args.order_seed,requests=requests,
                  input_policy='paired immutable input and seed within each round',
                  scope='fixed radix endpoints, register envelope and FFT thread dimension'))
        def one(name,req,phase,pattern,seed,round_no=0,order=0,timing=False):
            env={} if req is None else core.override(req['radix'],req['grouped_batch'],req['block'])
            w=dict(n=1024,batch=16,seed=seed,config={},pattern=pattern,graph_repeats=args.graph_repeats,
                   warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1)
            row=suite.run_one(args,w,(name,env),phase,round_no,order)
            try:
                result=core.validate(row,args,w['repeats'])
                if req is not None: core.assert_shape(result,req)
                if timing and min(result['raw_interval_ms_samples']) < 0.5:
                    raise ValueError('timing interval below 0.5 ms')
            except (ValueError,KeyError,TypeError) as e:
                row['status']='failed';row.setdefault('validation_errors',[]).append(str(e));emit(row);raise
            emit(row);return result
        while True:
            base=one('default',None,'calibration','random',42)
            if min(base['raw_interval_ms_samples'])>=2.0: break
            if args.graph_repeats>=2048: raise ValueError('baseline calibration failed')
            args.graph_repeats=min(2048,args.graph_repeats*2)
        emit(dict(record_type='calibration_complete',graph_repeats=args.graph_repeats))
        base=one('default',None,'baseline','random',42)
        k=base['structural_features']['kernels'][0]
        if (k['stageRadix'],k['groupedBatch'],k['localSize'])!=([8,8,8,2],1,[128,1,1]):
            raise ValueError('target default structure differs from expected')
        identity={'radix':k['stageRadix'],'grouped_batch':1,'block':k['localSize']}
        same=one('identity',identity,'identity','random',42)
        if core.signature(base)!=core.signature(same): raise ValueError('identity differs from baseline')
        allowed={json.dumps(x['request'],sort_keys=True):x for x in core.enum.candidates(base)}
        for name,req in requests.items():
            key=json.dumps(req,sort_keys=True)
            if key not in allowed or allowed[key]['reason'] is not None: raise ValueError('requested candidate rejected by precheck: '+name)
        signatures={'default':core.signature(base)}
        for name,req in requests.items():
            for pattern,seed in [(p,42) for p in PATTERNS] + [('random',s) for s in SEEDS]:
                r=one(name,req,'numerical',pattern,seed)
                sig=core.signature(r)
                if name in signatures and sig!=signatures[name]: raise ValueError('candidate structure changed across inputs')
                signatures[name]=sig
        for pattern,seed in [(p,42) for p in PATTERNS] + [('random',s) for s in SEEDS]:
            r=one('default',None,'numerical_control',pattern,seed)
            if core.signature(r)!=signatures['default']: raise ValueError('default structure changed')
        rng=random.Random(args.order_seed); effects=[]
        for i in range(args.rounds):
            names=['default','A4','B4','A8','B8'];rng.shuffle(names);times={}
            for j,name in enumerate(names):
                r=one(name,requests.get(name),'paired_timing','random',SEEDS[i%len(SEEDS)],i,j,True)
                if core.signature(r)!=signatures[name]: raise ValueError('timing structure changed')
                times[name]=statistics.median(r['exec_ms_samples'])
            effect=math.log(times['A8']/times['B8'])-math.log(times['A4']/times['B4'])
            effects.append(effect)
            emit(dict(record_type='paired_round',round=i,seed=SEEDS[i%len(SEEDS)],order=names,
                      median_ms=times,log_interaction=effect,ratio_of_ratios=math.exp(effect)))
        emit(dict(record_type='controller_complete',rounds=args.rounds,mean_log_interaction=statistics.mean(effects),
                  geometric_interaction=math.exp(statistics.mean(effects)),scope='2x2 confirmatory comparison, not global search'))


if __name__=='__main__': main()
