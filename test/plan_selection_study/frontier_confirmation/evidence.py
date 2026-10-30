"""Re-audit the exact screen evidence underlying a frozen confirmation plan."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'frontier_execution'))
spec = importlib.util.spec_from_file_location('frontier_execution_runner', HERE.parent / 'frontier_execution' / 'runner.py')
frontier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontier)
import audit_guard
import search_controller
import validation
import original_compare_controller as comparison


def signature_hash(signature):
    payload = json.dumps(signature,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def pair_order(plan_sha256, session_kind, n, batch, cid, round_no):
    key = json.dumps([plan_sha256,session_kind,n,batch,cid,round_no],separators=(',',':')).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(),'big'))
    order = ['original','candidate']; rng.shuffle(order)
    return order


def require(condition, message):
    if not condition: raise ValueError(message)


def audit_screen(plan, evidence_paths, binaries):
    require(plan.get('schema') == 'frontier-confirmation-plan/v1' and
            set(evidence_paths) == {'rows','guard','incidents','budget'}, 'plan or evidence schema mismatch')
    actual = {k:frontier.sha(v) for k,v in evidence_paths.items()}
    require(plan.get('source_sha256') == actual, 'frozen screening evidence hash mismatch')
    rows, guards, incidents, budget = (frontier.read_log(evidence_paths[k]) for k in ('rows','guard','incidents','budget'))
    require(rows and guards and budget, 'missing screening evidence')
    contract = rows[0]['contract']; gpu = plan['gpu']
    require(all(r.get('contract') == contract for r in rows), 'screen contracts differ')
    require(contract['gpu'] == gpu and {k:frontier.sha(v) for k,v in binaries.items()} == contract['binaries'],
            'screen device or binaries differ')
    guard_audit = audit_guard.audit(rows,guards,incidents,gpu)
    require(guard_audit['status'] == 'pass', 'screen guard failed: ' + '; '.join(guard_audit.get('errors',[])[:5]))
    require(budget[0]['event'] == 'start' and budget[-1]['state']['inflight'] is None and
            all(x['state']['contract'] == contract for x in budget), 'screen budget evidence invalid')
    starts = [r for r in budget if r['event'] == 'work_start']
    ends = [r for r in budget if r['event'] == 'work_finish']
    executions = [r for r in rows if r.get('record_type') == 'execution']
    require(len(starts) == len(ends) == len(executions), 'screen budget count mismatch')
    for st,end,row in zip(starts,ends,executions):
        i=end['identity']; w=row['workload']
        require(st['state']['inflight'] == i and end['status'] == row['status'] and
                (i['n'],i['batch'],i['candidate_id'],i['phase'],i['round']) ==
                (w['n'],w['batch'],row['candidate_id'],row['phase'],row['round']), 'screen budget linkage mismatch')
        decoded=[json.loads(line) for line in row['stdout'].splitlines() if line.strip()]
        require(len(decoded)==1 and decoded[0]==row['result'], 'screen stdout mismatch')
        expected_binary=contract['binaries']['original' if row['candidate_id']=='original' else 'expanded']
        require(row['executable_sha256']==expected_binary, 'screen execution binary mismatch')
    frontier.reject_prior_numerical_failure(rows)
    cases=plan.get('cases'); require(isinstance(cases,list) and cases, 'empty confirmation plan')
    groups={}; active=None
    for row in rows:
        if row.get('record_type')=='candidate_start':
            key=(row['n'],row['batch'],row['candidate_id'])
            require(key not in groups, 'duplicate screen candidate start')
            groups[key]=[row]; active=key
        elif row.get('record_type')=='execution' and row.get('phase')=='identity': active=None
        elif active is not None: groups[active].append(row)
    seen=set(); verified=[]
    for case in cases:
        n,batch,cid,req=(case['n'],case['batch'],case['candidate_id'],case['request'])
        key=(n,batch,cid); require(key not in seen and key in groups, 'candidate missing or duplicate in screen')
        seen.add(key)
        payload=json.dumps(req,sort_keys=True,separators=(',',':')).encode()
        require(hashlib.sha256(payload).hexdigest()==cid, 'candidate request hash mismatch')
        segment=groups[key]; require(segment[0]['request']==req, 'screen request mismatch')
        controls=[r for r in executions if r['phase']=='identity' and (r['workload']['n'],r['workload']['batch'])==(n,batch)]
        require(len(controls)==3 and {r['candidate_id'] for r in controls}=={'original','expanded_default','identity'}, 'screen identity controls missing')
        require(all((r['workload']['pattern'],r['workload']['seed'],r['workload']['graph_repeats']) == ('random',42,64)
                    for r in controls), 'screen identity input differs')
        c={r['candidate_id']:r for r in controls}
        original=validation.check(c['original'],gpu,original=True); default=validation.check(c['expanded_default'],gpu)
        comparison.compare(default,original); identity,why=frontier.representable(default)
        require(identity is not None, 'screen default not representable: '+str(why))
        replay=validation.check(c['identity'],gpu,request=identity)
        require(validation.signature(default)==validation.signature(replay), 'screen identity replay mismatch')
        original_sig=search_controller.source_signature(original,original=True)
        descriptions=[r for r in segment if r.get('record_type')=='execution' and r['phase']=='describe']
        require(len(descriptions)==1 and descriptions[0]['candidate_id']==cid and
                (descriptions[0]['workload']['n'],descriptions[0]['workload']['batch'])==(n,batch), 'screen describe missing')
        described=validation.check_describe(descriptions[0],gpu,req); dp=described['precompile_kernels'][0]
        vals=[r for r in segment if r.get('record_type')=='execution' and r['phase']=='validation']
        require(len(vals)==15 and {(r['workload']['pattern'],r['workload']['seed']) for r in vals}==
                {(p,s) for p in frontier.PATTERNS for s in frontier.SEEDS}, 'screen 5x3 validation incomplete')
        cand_sig=None
        for row in vals:
            require(row['candidate_id']==cid and (row['workload']['n'],row['workload']['batch'])==(n,batch),
                    'screen validation candidate mismatch')
            result=validation.check(row,gpu,request=req); cp=result['precompile_kernels'][0]
            require(all(cp[k]==dp[k] for k in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')), 'screen code changed')
            sig=search_controller.source_signature(result,original=False)
            require(cand_sig is None or sig==cand_sig, 'screen candidate signature drift'); cand_sig=sig
        timings=[r for r in segment if r.get('record_type')=='execution' and r['phase']=='timing']
        require(len(timings)==6, 'screen timing pairs incomplete'); ratios=[]
        for round_no in range(3):
            pair=[r for r in timings if r['round']==round_no]
            require(len(pair)==2 and {r['candidate_id'] for r in pair}=={'original',cid} and
                    {r['order'] for r in pair}=={0,1}, 'screen timing pair malformed')
            pair.sort(key=lambda row:row['order'])
            require([row['candidate_id'] for row in pair] ==
                    ['original' if name=='original' else cid for name in frontier.seeded_order(n,batch,cid,round_no)],
                    'screen timing order differs')
            values={}
            for row in pair:
                require((row['workload']['n'],row['workload']['batch'],row['workload']['pattern'],row['workload']['seed'])==
                        (n,batch,'random',42), 'screen timing workload mismatch')
                orig=row['candidate_id']=='original'
                result=validation.check(row,gpu,original=orig,request=None if orig else req,timing=True)
                require(search_controller.source_signature(result,original=orig)==(original_sig if orig else cand_sig), 'screen timing signature drift')
                require(result['warmup_actual']>=5 and result['warmup_wall_ms']>=1000, 'screen warmup short')
                values['original' if orig else 'candidate']=statistics.median(result['exec_ms_samples'])
            ratios.append(values['candidate']/values['original'])
        pair_rows=[r for r in segment if r.get('record_type')=='screen_pair']
        require(len(pair_rows)==3 and {r['round'] for r in pair_rows}=={0,1,2} and
                all((r['n'],r['batch'],r['candidate_id'])==key and
                    math.isclose(r['ratio'],ratios[r['round']],rel_tol=1e-12) for r in pair_rows),
                'screen pair summaries differ')
        disposition=[r for r in segment if r.get('record_type')=='candidate_disposition']
        require(len(disposition)==1 and disposition[0]['status']=='screen_complete' and disposition[0]['request']==req and
                disposition[0]['pair_ratios']==ratios and math.isclose(disposition[0]['median_ratio'],statistics.median(ratios),rel_tol=1e-12) and
                statistics.median(ratios)<1, 'screen winner disposition invalid or not faster')
        verified.append(dict(n=n,batch=batch,candidate_id=cid,request=req,
                             screen_candidate_signature_sha256=signature_hash(cand_sig),
                             screen_original_signature_sha256=signature_hash(original_sig),
                             screen_ratio=statistics.median(ratios)))
    return {'screen_contract':contract,'cases':verified,'source_sha256':actual,'gpu':gpu}
