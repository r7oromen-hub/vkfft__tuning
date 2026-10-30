#!/usr/bin/env python3
"""Guarded restricted joint search, same-binary default control, fail closed."""
import argparse, hashlib, json, math, os, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import enumerate_joint as enum
import run_suite as suite
from direct_controller import signature, override, PATTERNS

def validate(row,args,repeats):
    if row.get('status') != 'ok' or row.get('guard_clean') is not True:
        raise ValueError('process failed or guard did not attest clean')
    r=row['result']
    if r.get('status') != 'ok' or r.get('gpu',{}).get('uuid') != args.gpu:
        raise ValueError('benchmark or GPU identity failure')
    if r.get('requested_config') != {'makeForwardPlanOnly':1}:
        raise ValueError('not the unmodified requested default configuration')
    w=row['workload']
    if (r.get('n'),r.get('batch'),r.get('seed')) != (w['n'],w['batch'],w['seed']):
        raise ValueError('workload identity mismatch')
    if r.get('validation_output_initialization') != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward':
        raise ValueError('output poisoning missing')
    if r.get('input_unchanged') is not True or r.get('single_forward_input_unchanged') is not True:
        raise ValueError('input changed')
    for key in ('correctness','single_forward_correctness'):
        c=r.get(key,{})
        if c.get('passed') is not True or c.get('nonfinite_count') != 0 or c.get('failing_batches') != 0 or c.get('checked_elements') != w['n'] * w['batch']:
            raise ValueError(key+' failed')
    if r.get('descriptor_validation',{}).get('all_match') is not True:
        raise ValueError('descriptor validation failed')
    if r.get('graph_repeats') != args.graph_repeats or r.get('graph_kernel_count') != args.graph_repeats or r.get('graph_event_record_count') != 2:
        raise ValueError('graph execution mismatch')
    if len(r.get('precompile_kernels',[])) != 1 or r['precompile_kernels'][0].get('compiler_options_match') is not True:
        raise ValueError('compiler descriptor/options mismatch')
    if r.get('structural_features',{}).get('numAxisUploads') != 1 or len(r['structural_features'].get('kernels',[])) != 1:
        raise ValueError('requires one upload and one kernel')
    if row.get('direct_environment'):
        limits=r.get('direct_resource_validation',{})
        if (not limits or limits['dynamic_shared_bytes'] > limits['function_dynamic_limit_bytes'] or
            limits['dynamic_shared_bytes']+limits['static_shared_bytes'] > limits['device_shared_limit_bytes'] or
            limits['threads'] > limits['function_thread_limit']):
            raise ValueError('compiled resource validation missing or failed')
    samples=r.get('exec_ms_samples',[]); raw=r.get('raw_interval_ms_samples',[])
    if len(samples)!=repeats or len(raw)!=repeats or not all(math.isfinite(x) and x>0 for x in samples) or not all(math.isfinite(x) and x>0 for x in raw):
        raise ValueError('invalid timing samples')
    return r

def numeric_invalid(row,args,w):
    """Recognize only completed GPU work with unchanged input and failed reference."""
    r=row.get('result',{}); single=r.get('single_forward_correctness',{}); final=r.get('correctness',{})
    common=(row.get('guard_clean') is True and r.get('gpu',{}).get('uuid') == args.gpu and
            r.get('requested_config') == {'makeForwardPlanOnly':1} and
            (r.get('n'),r.get('batch'),r.get('seed')) == (w['n'],w['batch'],w['seed']) and
            r.get('single_forward_input_unchanged') is True and
            r.get('validation_output_initialization') == '0xff_bytes_fp32_nan_before_single_and_final_timed_forward' and
            r.get('descriptor_validation',{}).get('all_match') is True and
            bool(r.get('direct_resource_validation')) and len(r.get('precompile_kernels',[])) == 1)
    if not common: return False
    if (r.get('status')=='error' and r.get('failure_phase')=='single_forward_validation' and
        r.get('error')=='Single out-of-place forward failed immutable-input or numerical validation'):
        return single.get('checked_elements') == w['n']*w['batch'] and single.get('passed') is False
    if r.get('status')=='correctness_failed' and r.get('input_unchanged') is True:
        return (single.get('passed') is True and single.get('checked_elements') == w['n']*w['batch'] and
                final.get('checked_elements') == w['n']*w['batch'] and final.get('passed') is False)
    return False

def median_timing(r): return statistics.median(r['exec_ms_samples'])
def assert_shape(r,req):
    k=r['structural_features']['kernels'][0]
    if k['stageRadix'] != req['radix'] or k['localSize'] != req['block'] or k['groupedBatch'] != req['grouped_batch']:
        raise ValueError('actual structure differs from request')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--n',type=int,required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--screen-rounds',type=int,default=3);p.add_argument('--confirm-rounds',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--timeout',type=int,default=300)
    args=p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu: p.error('must run under joint_gpu_guard.py with the same GPU UUID')
    if args.n < 2 or args.n > 8192 or args.n & (args.n-1) or not 1<=args.batch<=65536 or not 0<=args.limit<=100000 or not 1<=args.screen_rounds<=100 or not 2<=args.confirm_rounds<=100 or not 1<=args.timeout<=3600:
        p.error('invalid n/batch/limit/rounds/timeout')
    args.exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest(); args.graph_repeats=64
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed)
    with args.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='protocol',n=args.n,batch=args.batch,exe_sha256=args.exe_sha256,gpu=args.gpu,limit=args.limit,
                  screen_rounds=args.screen_rounds,confirm_rounds=args.confirm_rounds,seed=args.seed,
                  selection='all prechecked requests unless a nonzero seeded subset cap is specified; select minimum median paired screen ratio; independent paired confirmation',
                  limitations='restricted endpoint-preserving power-of-two decompositions and grouped batch; prospective candidates require runtime validation; no global optimum claim'))
        def one(name,env,phase,pattern,round_no=0,order=0,timing=False,allow_reject=False):
            repeats=31 if timing else 1
            w=dict(n=args.n,batch=args.batch,seed=args.seed,config={},pattern=pattern,graph_repeats=args.graph_repeats,
                   warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=repeats)
            row=suite.run_one(args,w,(name,env),phase,round_no,order)
            rejected = row.get('result',{})
            reason = rejected.get('error','')
            if (allow_reject and row.get('guard_clean') is True and rejected.get('status')=='error' and
                rejected.get('failure_phase')=='initialize' and reason in
                ('initializeVkFFT failed with code 3002', 'direct candidate exceeds compiled function or device resource limit')):
                emit(row)
                emit(dict(record_type='candidate_disposition',candidate_id=name,status='candidate_rejected',reason=reason))
                return None
            if allow_reject and numeric_invalid(row,args,w):
                emit(row)
                emit(dict(record_type='candidate_disposition',candidate_id=name,status='numeric_invalid',reason='independent_reference_failed_after_clean_execution'))
                return None
            try: r=validate(row,args,repeats)
            except (ValueError,KeyError,TypeError) as error:
                row.setdefault('validation_errors',[]).append(str(error));row['status']='failed';emit(row);raise
            emit(row);return r
        # Calibrate on the same baseline, without changing plan settings. Probe
        # uses one event sample, then all paired timings enforce 0.5 ms each.
        while True:
            base=one('default',{},'calibration','random')
            if min(base['raw_interval_ms_samples']) >= 2.0: break
            if args.graph_repeats >= 2048: raise ValueError('calibration cannot reach 2.0 ms within graph limit')
            args.graph_repeats=min(2048,args.graph_repeats*2)
        emit(dict(record_type='calibration_complete',graph_repeats=args.graph_repeats))
        base=one('default',{},'baseline','random')
        k=base['structural_features']['kernels'][0]
        ident_req={'radix':k['stageRadix'],'grouped_batch':k['groupedBatch'],'block':k['localSize']}
        identity=one('identity',override(ident_req['radix'],ident_req['grouped_batch'],ident_req['block']),'identity','random')
        assert_shape(identity,ident_req)
        if signature(base)!=signature(identity): raise ValueError('identity replay source/descriptor/options mismatch')
        null_ratios=[]
        for i in range(3):
            null_order=['default_a','default_b'];rng.shuffle(null_order);pair={}
            for j,name in enumerate(null_order):
                r=one(name,{},'null_timing','random',i,j,True)
                if min(r['raw_interval_ms_samples']) < 0.5 or signature(r)!=signature(base):
                    raise ValueError('null timing interval or baseline structure mismatch')
                pair[name]=median_timing(r)
            null_ratios.append(pair['default_a']/pair['default_b'])
        emit(dict(record_type='null_control',paired_ratios=null_ratios,median_ratio=statistics.median(null_ratios)))
        manifest=list(enum.candidates(base))
        eligible=[x for x in manifest if x['reason'] is None and x['request'] != ident_req]
        rng.shuffle(eligible); selected_items=eligible if args.limit == 0 else eligible[:args.limit]; selected={x['candidate_id'] for x in selected_items}
        for x in manifest:
            if x['request']==ident_req: x['status']='identity_control';x['reason']='same_as_default_request'
            elif x['reason'] is None and x['candidate_id'] not in selected: x['status']='not_selected';x['reason']='seeded_subset_limit'
            emit(dict(record_type='candidate_manifest',**x))
        emit(dict(record_type='manifest_complete',total=len(manifest),eligible=len(eligible),selected=len(selected)))
        seen={json.dumps(signature(base),sort_keys=True):'default'}; scores=[]
        for item in selected_items:
            ident=item['candidate_id']; req=item['request']; env=override(req['radix'],req['grouped_batch'],req['block'])
            # The first execution both checks numerical correctness and obtains
            # the actual generated structure for source/descriptor deduplication.
            r=one(ident,env,'numerical','random',allow_reject=True)
            if r is None: continue
            assert_shape(r,req)
            sig=json.dumps(signature(r),sort_keys=True)
            if sig in seen:
                emit(dict(record_type='candidate_disposition',candidate_id=ident,status='duplicate_actual_structure',duplicate_of=seen[sig]));continue
            seen[sig]=ident
            candidate_invalid=False
            for pattern in PATTERNS:
                # Default is an explicit negative control for each candidate.
                b=one('default',{},'numerical_control',pattern)
                if signature(b)!=signature(base): raise ValueError('default structure drift')
                r=one(ident,env,'numerical',pattern,allow_reject=True)
                if r is None:
                    candidate_invalid=True;break
                assert_shape(r,req)
                if json.dumps(signature(r),sort_keys=True)!=sig: raise ValueError('candidate structure drift')
            if candidate_invalid: continue
            ratios=[]
            for i in range(args.screen_rounds):
                order=[('default',{}),(ident,env)];rng.shuffle(order); pair={}
                for j,(name,setting) in enumerate(order):
                    r=one(name,setting,'screen_timing','random',i,j,True)
                    if min(r['raw_interval_ms_samples'])<0.5: raise ValueError('screen timing below calibrated interval')
                    if name==ident: assert_shape(r,req)
                    if json.dumps(signature(r),sort_keys=True)!=(sig if name==ident else json.dumps(signature(base),sort_keys=True)): raise ValueError('timing structure drift')
                    pair[name]=median_timing(r)
                ratios.append(pair[ident]/pair['default'])
            score=statistics.median(ratios);scores.append((score,ident,req,env,sig))
            emit(dict(record_type='candidate_disposition',candidate_id=ident,status='screened',paired_ratios=ratios,median_ratio=score))
        if not scores:
            emit(dict(record_type='controller_complete',status='no_unique_candidates'));return
        _,winner,req,env,sig=min(scores,key=lambda x:(x[0],x[1])); ratios=[]
        for i in range(args.confirm_rounds):
            order=[('default',{}),(winner,env)];rng.shuffle(order);pair={}
            for j,(name,setting) in enumerate(order):
                r=one(name,setting,'confirmation_timing','random',i,j,True)
                if min(r['raw_interval_ms_samples'])<0.5: raise ValueError('confirmation timing below calibrated interval')
                if name==winner: assert_shape(r,req)
                if json.dumps(signature(r),sort_keys=True)!=(sig if name==winner else json.dumps(signature(base),sort_keys=True)): raise ValueError('confirmation structure drift')
                pair[name]=median_timing(r)
            ratios.append(pair[winner]/pair['default'])
        emit(dict(record_type='controller_complete',status='confirmed_measurements',winner=winner,confirmation_paired_ratios=ratios,
                  confirmation_median_ratio=statistics.median(ratios),interpretation='selection on screen data; independent confirmation only; no global optimum or significance claim'))
if __name__=='__main__':main()
