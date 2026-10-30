#!/usr/bin/env python3
"""Independent audit and paired effects for a complete public-knob factorial."""
import argparse, collections, hashlib, itertools, json, math, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import audit_joint


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(rows,guard,incidents,exe_sha,controller_sha):
    assert not incidents and rows[0]['record_type']=='protocol' and rows[-1]['record_type']=='controller_complete'
    p=rows[0];end=rows[-1];configs=p['configs'];starts=[x for x in guard if x.get('event')=='guard_start'];ends=[x for x in guard if x.get('event')=='guard_complete']
    assert len(starts)==len(ends)==1 and guard[-1] is ends[0] and ends[0]['clean'] and ends[0]['success'] and ends[0]['script_exit_code']==0
    assert p['exe_sha256']==exe_sha and p['controller_sha256']==controller_sha
    gid=starts[0]['guard_id'];gpu=p['gpu'];assert starts[0]['gpu_uuid']==gpu and starts[0]['own_signal_method']=='pidfd_with_rechecked_PPID_exe_starttime'
    assert all(x.get('guard_id')==gid and x.get('gpu_uuid')==gpu for x in guard)
    assert not any(x.get('incident') or x.get('event') in ('guard_abort','quarantined_run','own_harness_sigterm','termination_error') for x in guard)
    samples=[x for x in guard if x.get('event')=='sample']
    assert ends[0]['sample_count']==len(samples) and all(x.get('ok') is True and not x.get('violation_reasons') and not x.get('foreign_compute_pids') and x.get('monitor_error') is None and x.get('gpu',{}).get('uuid')==gpu and x.get('gpu',{}).get('memory_used_mib',999999)<=2048 for x in samples)
    idle=[x for x in guard if x.get('event')=='idle_complete'];assert len(idle)==1 and idle[0]['clean'] and idle[0]['observed_idle_seconds']>=starts[0]['idle_seconds']
    assert len([x for x in samples if x.get('boundary')=='final'])==1
    runs=[x for x in rows if 'phase' in x and 'workload' in x];ids=[x.get('guard_run_id') for x in runs]
    assert None not in ids and len(set(ids))==len(ids)
    events=collections.defaultdict(list)
    for x in guard:
        rid=x.get('guard_run_id') or (x.get('active_run') or {}).get('guard_run_id')
        if rid:events[rid].append(x)
    assert set(events)==set(ids)
    expected={f'a{a}_r{r}_l{l}':{'aim_threads':a,'register_boost':r,'use_lut':l} for a,r,l in itertools.product((64,256),(1,4),(-1,1))}
    assert configs==expected and len(configs)==8
    cal=[x for x in runs if x['phase']=='calibration'];cd=[x for x in rows if x.get('record_type')=='calibration_complete']
    assert cal and len(cd)==1 and min(cal[-1]['result']['raw_interval_ms_samples'])>=2.0
    assert all(x['workload']['graph_repeats']==cd[0]['graph_repeats'] for x in runs if x['phase']!='calibration')
    signatures={}; sources={}; structures={}
    for row in runs:
        ev=events[row['guard_run_id']];st=[x for x in ev if x.get('event')=='run_one_start'];en=[x for x in ev if x.get('event')=='run_one_end'];reg=[x for x in ev if x.get('event')=='own_harness_registered'];before=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='before'];after=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='after']
        assert len(st)==len(en)==len(reg)==len(before)==len(after)==1
        ident=reg[0]['identity'];active=st[0]['active_run'];assert reg[0]['pidfd_retained'] and ident['ppid']==starts[0]['parent_pid'] and ident['start_ticks']>0 and ident['exe']==active['expected_exe']==row['command'][0]
        assert en[0]['clean'] and en[0]['status']==row['status'] and active['candidate_id']==row['candidate_id'] and active['phase']==row['phase'] and active['round']==row['round'] and active['order']==row['order']
        assert ev.index(st[0])<ev.index(before[0])<ev.index(reg[0])<ev.index(after[0])<ev.index(en[0])
        name=row['candidate_id'];config=configs.get(name,{})
        assert row['workload']['config']==config and not row['direct_environment']
        requested={'makeForwardPlanOnly':1}
        if config:requested.update(aimThreads=config['aim_threads'],registerBoost=config['register_boost'],useLUT=config['use_lut'])
        result=row['result'];assert result['requested_config']==requested and [json.loads(x) for x in row['stdout'].splitlines() if x.strip()]==[result]
        # Validate the original request first, then normalize only a local copy
        # so the existing no-knob GPU/precision/reference checks can be reused.
        clone=dict(row);clone['result']=dict(result);clone['result']['requested_config']={'makeForwardPlanOnly':1};clone['stdout']=json.dumps(clone['result'])
        clone['workload']=dict(row['workload']);clone['workload']['config']={}
        command=list(row['command']);knob_names={'--aim-threads','--register-boost','--use-lut'}
        pairs=list(zip(command[1::2],command[2::2]));assert sum(k in knob_names for k,v in pairs)==(3 if config else 0)
        clone['command']=[command[0]]+[x for k,v in pairs if k not in knob_names for x in (k,v)]
        protocol={'n':p['n'],'batch':p['batch'],'gpu':gpu,'seed':row['workload']['seed']}
        audit_joint.check_result(clone,protocol)
        sig=audit_joint.sig(result);assert name not in signatures or signatures[name]==sig;signatures[name]=sig
        pc=result['precompile_kernels'][0];sources[name]=(pc['source_sha256'],json.dumps(pc['actual_compiler_options'],sort_keys=True));structures[name]=result['structural_features']
        if row['phase']=='paired_timing':assert row['workload']['repeats']==31 and row['workload']['warmup_ms']>=1000 and min(result['raw_interval_ms_samples'])>=0.5
    patterns={('zero',42),('impulse',42),('constant',42),('quarter-tone',42),('random',42),('random',53),('random',71)}
    for name in ['default',*configs]:
        items=[x for x in runs if x['phase']=='numerical' and x['candidate_id']==name]
        assert len(items)==7 and {(x['workload']['pattern'],x['workload']['seed']) for x in items}==patterns
    records=[x for x in rows if x.get('record_type')=='paired_round'];assert len(records)==p['rounds']==end['rounds']
    logs=[];ratios=collections.defaultdict(list)
    for i,record in enumerate(records):
        block=[x for x in runs if x['phase']=='paired_timing' and x['round']==i];assert len(block)==9 and {x['candidate_id'] for x in block}=={'default',*configs}
        assert sorted(x['order'] for x in block)==list(range(9)) and all(x['workload']['seed']==p['seeds'][i%len(p['seeds'])] for x in block)
        assert record['round']==i and record['order']==[x['candidate_id'] for x in sorted(block,key=lambda x:x['order'])]
        ts={x['candidate_id']:statistics.median(x['result']['exec_ms_samples']) for x in block};assert all(math.isclose(ts[n],record['median_ms'][n],rel_tol=1e-9) for n in ts)
        logs.append({n:math.log(ts[n]) for n in configs})
        for n in configs:ratios[n].append(ts[n]/ts['default'])
    dimensions=['aim_threads','register_boost','use_lut'];levels={'aim_threads':(64,256),'register_boost':(1,4),'use_lut':(-1,1)}
    def name_for(cfg):return f"a{cfg['aim_threads']}_r{cfg['register_boost']}_l{cfg['use_lut']}"
    contrasts={}
    for dim in dimensions:
        rest=[d for d in dimensions if d!=dim];per=[]
        for block in logs:
            d=[]
            for vals in itertools.product(*(levels[x] for x in rest)):
                q=dict(zip(rest,vals));q[dim]=levels[dim][0];lo=block[name_for(q)];q[dim]=levels[dim][1];d.append(block[name_for(q)]-lo)
            per.append(statistics.mean(d))
        contrasts[dim]=per
    for x,y in itertools.combinations(dimensions,2):
        other=next(d for d in dimensions if d not in (x,y));per=[]
        for block in logs:
            d=[]
            for z in levels[other]:
                q={other:z};q[x]=levels[x][0];q[y]=levels[y][0];ll=block[name_for(q)];q[x]=levels[x][1];hl=block[name_for(q)];q[y]=levels[y][1];hh=block[name_for(q)];q[x]=levels[x][0];lh=block[name_for(q)];d.append(hh-hl-lh+ll)
            per.append(statistics.mean(d))
        contrasts[x+'*'+y]=per
    from scipy.stats import t
    def summary(v):
        mean=statistics.mean(v);half=t.ppf(.975,len(v)-1)*statistics.stdev(v)/math.sqrt(len(v))
        return {'ratio':math.exp(mean),'ci95':[math.exp(mean-half),math.exp(mean+half)],'log_per_round':v}
    classes=collections.defaultdict(list)
    for name,fp in sources.items():classes[fp].append(name)
    return {'gpu':gpu,'n':p['n'],'batch':p['batch'],'rounds':len(records),'processes':len(runs),'guard_samples':len(samples),
            'source_classes':list(classes.values()),'plans':structures,'candidate_over_default_medians':{n:statistics.median(v) for n,v in ratios.items()},
            'effects':{n:summary(v) for n,v in contrasts.items()},'graph_repeats':cd[0]['graph_repeats'],
            'interpretation':'Requested-knob factorial; interactions may arise through changed radix, block, register or LUT and are not independent structure interventions. CI describes within-session block variation only.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True,type=Path);p.add_argument('--guard',required=True,type=Path);p.add_argument('--incidents',required=True,type=Path);p.add_argument('--exe',required=True,type=Path);p.add_argument('--controller',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    values=[]
    for path in (a.rows,a.guard,a.incidents):
        data,errors=audit_joint.load(path)
        if errors:raise ValueError(errors)
        values.append(data)
    result=audit(*values,sha(a.exe),sha(a.controller))
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
