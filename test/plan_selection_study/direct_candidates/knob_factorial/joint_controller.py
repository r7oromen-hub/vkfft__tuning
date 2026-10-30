#!/usr/bin/env python3
"""Guarded 2x2x2 public-knob factorial with actual-plan fingerprinting."""
import argparse, hashlib, itertools, json, math, os, random, statistics
from pathlib import Path
import core_validation as core
import run_suite as suite

SEEDS=(42,53,71)
PATTERNS=('zero','impulse','constant','quarter-tone')
CONFIGS={f'a{a}_r{r}_l{l}':{'aim_threads':a,'register_boost':r,'use_lut':l}
         for a,r,l in itertools.product((64,256),(1,4),(-1,1))}


def checked(row,args,repeats,config):
    expected={'makeForwardPlanOnly':1}
    expected.update({'aimThreads':config['aim_threads'],'registerBoost':config['register_boost'],'useLUT':config['use_lut']} if config else {})
    if row.get('direct_environment') or row['result'].get('requested_config')!=expected:
        raise ValueError('requested public knobs differ from design')
    clone=dict(row); clone['result']=dict(row['result']); clone['result']['requested_config']={'makeForwardPlanOnly':1}
    return core.validate(clone,args,repeats)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--exe',required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('small','large'),required=True);p.add_argument('--rounds',type=int,default=10);p.add_argument('--timeout',type=int,default=300);p.add_argument('--order-seed',type=int,default=20261030)
    args=p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID')!=args.gpu:p.error('matching GPU guard required')
    if not 10<=args.rounds<=100 or not 1<=args.timeout<=3600:p.error('invalid rounds/timeout')
    n,batch={'small':(1024,16),'large':(256,4096)}[args.case]
    args.exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest();args.graph_repeats=64
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as out:
        def emit(row):out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='protocol',n=n,batch=batch,gpu=args.gpu,exe_sha256=args.exe_sha256,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  configs=CONFIGS,seeds=SEEDS,rounds=args.rounds,order_seed=args.order_seed,
                  meaning='2x2x2 requested knobs; actual scheduler structures may fold or couple; no direct radix/block injection'))
        def one(name,phase,pattern,seed,round_no=0,order=0,timing=False):
            config=CONFIGS.get(name,{})
            w=dict(n=n,batch=batch,seed=seed,config=config,pattern=pattern,graph_repeats=args.graph_repeats,
                   warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1)
            row=suite.run_one(args,w,(name,{}),phase,round_no,order)
            try:
                result=checked(row,args,w['repeats'],config)
                if timing and min(result['raw_interval_ms_samples'])<0.5:raise ValueError('timing interval below 0.5 ms')
            except (ValueError,KeyError,TypeError) as e:
                row['status']='failed';row.setdefault('validation_errors',[]).append(str(e));emit(row);raise
            emit(row);return result
        while True:
            base=one('default','calibration','random',42)
            if min(base['raw_interval_ms_samples'])>=2:break
            if args.graph_repeats>=2048:raise ValueError('calibration cannot reach 2 ms')
            args.graph_repeats=min(2048,args.graph_repeats*2)
        emit(dict(record_type='calibration_complete',graph_repeats=args.graph_repeats))
        names=['default',*CONFIGS]; signatures={}
        for name in names:
            for pattern,seed in [(p,42) for p in PATTERNS]+[('random',s) for s in SEEDS]:
                r=one(name,'numerical',pattern,seed)
                sig=core.signature(r)
                if name in signatures and signatures[name]!=sig:raise ValueError('actual plan changed with input/seed')
                signatures[name]=sig
        rng=random.Random(args.order_seed)
        for i in range(args.rounds):
            order=list(names);rng.shuffle(order);times={}
            for j,name in enumerate(order):
                r=one(name,'paired_timing','random',SEEDS[i%len(SEEDS)],i,j,True)
                if core.signature(r)!=signatures[name]:raise ValueError('plan signature changed in timing')
                times[name]=statistics.median(r['exec_ms_samples'])
            emit(dict(record_type='paired_round',round=i,seed=SEEDS[i%len(SEEDS)],order=order,median_ms=times))
        emit(dict(record_type='controller_complete',rounds=args.rounds,scope='requested knobs only; effective plan and code can fold or couple'))


if __name__=='__main__':main()
