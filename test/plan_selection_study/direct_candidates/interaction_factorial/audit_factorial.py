#!/usr/bin/env python3
"""Audit a complete paired factorial run; refuse all partial/contaminated runs."""
import argparse, collections, hashlib, json, math, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import audit_joint


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(rows, guard, incidents, exe_sha, controller_sha):
    assert not incidents and rows[0]['record_type']=='protocol' and rows[-1]['record_type']=='controller_complete'
    p=rows[0]; end=rows[-1]; starts=[x for x in guard if x.get('event')=='guard_start']; ends=[x for x in guard if x.get('event')=='guard_complete']
    assert len(starts)==len(ends)==1 and guard[-1] is ends[0] and ends[0]['clean'] and ends[0]['success'] and ends[0]['script_exit_code']==0
    assert p['exe_sha256']==exe_sha and p['controller_sha256']==controller_sha
    gid=starts[0]['guard_id']; gpu=p['gpu']; assert starts[0]['gpu_uuid']==gpu and starts[0]['own_signal_method']=='pidfd_with_rechecked_PPID_exe_starttime'
    assert all(x.get('guard_id')==gid and x.get('gpu_uuid')==gpu for x in guard)
    assert not any(x.get('incident') or x.get('event') in ('guard_abort','quarantined_run','own_harness_sigterm','termination_error') for x in guard)
    samples=[x for x in guard if x.get('event')=='sample']
    assert ends[0]['sample_count']==len(samples) and all(x.get('ok') is True and not x.get('violation_reasons') and not x.get('foreign_compute_pids') and x.get('monitor_error') is None and x.get('gpu',{}).get('uuid')==gpu and x.get('gpu',{}).get('memory_used_mib',999999)<=2048 for x in samples)
    idle=[x for x in guard if x.get('event')=='idle_complete']; assert len(idle)==1 and idle[0]['clean'] and idle[0]['observed_idle_seconds']>=starts[0]['idle_seconds']
    assert len([x for x in samples if x.get('boundary')=='final'])==1
    runs=[x for x in rows if 'phase' in x and 'workload' in x]; ids=[x.get('guard_run_id') for x in runs]
    assert None not in ids and len(set(ids))==len(ids)
    events=collections.defaultdict(list)
    for x in guard:
        rid=x.get('guard_run_id') or (x.get('active_run') or {}).get('guard_run_id')
        if rid: events[rid].append(x)
    assert set(events)==set(ids)
    req=p['requests']; assert set(req)=={'A4','B4','A8','B8'}
    base=next(x for x in runs if x['phase']=='baseline'); base_sig=audit_joint.sig(base['result'])
    ident=next(x for x in runs if x['phase']=='identity');k=base['result']['structural_features']['kernels'][0]
    identity={'radix':k['stageRadix'],'grouped_batch':k['groupedBatch'],'block':k['localSize']}
    assert identity=={'radix':[8,8,8,2],'grouped_batch':1,'block':[128,1,1]} and audit_joint.sig(ident['result'])==base_sig
    shapes={'A4':([8,4,4,4,2],4),'B4':([8,8,2,4,2],4),'A8':([8,4,4,4,2],8),'B8':([8,8,2,4,2],8)}
    for name,(r,g) in shapes.items(): assert req[name]=={'radix':r,'grouped_batch':g,'block':[g,128,1]}
    cal=[x for x in runs if x['phase']=='calibration']; cd=[x for x in rows if x.get('record_type')=='calibration_complete']
    assert cal and len(cd)==1 and min(cal[-1]['result']['raw_interval_ms_samples'])>=2.0
    assert all(x['workload']['graph_repeats']==cd[0]['graph_repeats'] for x in runs if x['phase']!='calibration')
    signatures={}
    for row in runs:
        ev=events[row['guard_run_id']]; starts1=[x for x in ev if x.get('event')=='run_one_start'];ends1=[x for x in ev if x.get('event')=='run_one_end'];reg=[x for x in ev if x.get('event')=='own_harness_registered'];before=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='before'];after=[x for x in ev if x.get('event')=='sample' and x.get('boundary')=='after']
        assert len(starts1)==len(ends1)==len(reg)==len(before)==len(after)==1
        ident1=reg[0]['identity']; active=starts1[0]['active_run']
        assert reg[0]['pidfd_retained'] and ident1['ppid']==starts[0]['parent_pid'] and ident1['start_ticks']>0 and ident1['exe']==active['expected_exe']==row['command'][0]
        assert ends1[0]['clean'] and ends1[0]['status']==row['status'] and active['candidate_id']==row['candidate_id'] and active['phase']==row['phase'] and active['round']==row['round'] and active['order']==row['order']
        assert ev.index(starts1[0])<ev.index(before[0])<ev.index(reg[0])<ev.index(after[0])<ev.index(ends1[0])
        name=row['candidate_id']; desired=identity if name=='identity' else req.get(name)
        protocol={'n':p['n'],'batch':p['batch'],'gpu':gpu,'seed':row['workload']['seed']}
        audit_joint.check_result(row,protocol,base_sig if name in ('default','identity') else None,desired)
        sig=audit_joint.sig(row['result']);assert name not in signatures or signatures[name]==sig;signatures[name]=sig
        if row['phase']=='paired_timing':
            assert row['workload']['repeats']==31 and row['workload']['warmup_ms']>=1000 and min(row['result']['raw_interval_ms_samples'])>=0.5
    expected_patterns={('zero',42),('impulse',42),('constant',42),('quarter-tone',42),('random',42),('random',53),('random',71)}
    for name in ['default',*req]:
        phase='numerical_control' if name=='default' else 'numerical'
        selected=[x for x in runs if x['phase']==phase and x['candidate_id']==name]
        assert len(selected)==7 and {(x['workload']['pattern'],x['workload']['seed']) for x in selected}==expected_patterns
    records=[x for x in rows if x.get('record_type')=='paired_round'];assert len(records)==p['rounds']==end['rounds']
    effects=[];ratio_groups=collections.defaultdict(list)
    for i,record in enumerate(records):
        block=[x for x in runs if x['phase']=='paired_timing' and x['round']==i]; assert len(block)==5 and {x['candidate_id'] for x in block}=={'default',*req}
        assert sorted(x['order'] for x in block)==list(range(5)) and all(x['workload']['seed']==p['seeds'][i%len(p['seeds'])] for x in block)
        ordered=sorted(block,key=lambda x:x['order']);assert record['round']==i and record['order']==[x['candidate_id'] for x in ordered]
        ts={x['candidate_id']:statistics.median(x['result']['exec_ms_samples']) for x in block}
        assert all(math.isclose(ts[n],record['median_ms'][n],rel_tol=1e-9) for n in ts)
        for n in req:ratio_groups[n].append(ts[n]/ts['default'])
        value=math.log(ts['A8']/ts['B8'])-math.log(ts['A4']/ts['B4']); assert math.isclose(value,record['log_interaction'],rel_tol=1e-9,abs_tol=1e-12)
        effects.append(value)
    assert math.isclose(math.exp(statistics.mean(effects)),end['geometric_interaction'],rel_tol=1e-9)
    from scipy.stats import t
    mean=statistics.mean(effects); se=statistics.stdev(effects)/math.sqrt(len(effects)); critical=t.ppf(.975,len(effects)-1)
    return {'gpu':gpu,'samples':len(samples),'processes':len(runs),'rounds':len(records),'seed_values':p['seeds'],
            'geometric_interaction':math.exp(mean),'ci95': [math.exp(mean-critical*se),math.exp(mean+critical*se)],
            'candidate_over_default_median':{k:statistics.median(v) for k,v in ratio_groups.items()},
            'round_log_interactions':effects,'calibrated_graph_repeats':cd[0]['graph_repeats']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True,type=Path);p.add_argument('--guard',required=True,type=Path);p.add_argument('--incidents',required=True,type=Path);p.add_argument('--exe',required=True,type=Path);p.add_argument('--controller',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    values=[]
    for path in (a.rows,a.guard,a.incidents):
        data,errors=audit_joint.load(path)
        if errors: raise ValueError(errors)
        values.append(data)
    result=audit(*values,digest(a.exe),digest(a.controller))
    with a.output.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
