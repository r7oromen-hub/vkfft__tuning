#!/usr/bin/env python3
"""Independently audit a single guarded frontier admission before accepting it."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

import runner
sys.path.insert(0, str(runner.EXPANDED))
import audit_guard
import search_controller
import validation
import original_compare_controller as comparison


def require(condition, message):
    if not condition: raise ValueError(message)


def audit(rows, guards, incidents, ledger, manifests, gpu, execution_commit, expected_count=7, binaries=None):
    errors = []
    try:
        require(rows and guards and ledger, 'missing run, guard, or budget evidence')
        contract = rows[0]['contract']
        require(all(r.get('contract') == contract for r in rows), 'mixed output contracts')
        require(contract['gpu'] == gpu and contract['execution_commit'] == execution_commit,
                'GPU or execution commit mismatch')
        require(len(execution_commit) == 40 and all(c in '0123456789abcdef' for c in execution_commit),
                'execution commit must be a full lowercase SHA-1')
        expected_hashes = sorted(contract['manifests'].values())
        require(sorted(runner.sha(path) for path in manifests) == expected_hashes, 'manifest hashes mismatch')
        require(binaries is not None and set(binaries) == {'original','expanded'}, 'binary files required')
        require({name:runner.sha(path) for name,path in binaries.items()} == contract['binaries'],
                'binary hashes mismatch')
        guard_audit = audit_guard.audit(rows, guards, incidents, gpu)
        require(guard_audit['status'] == 'pass', 'guard audit: ' + '; '.join(guard_audit.get('errors', [])[:5]))
        executions = [r for r in rows if r.get('record_type') == 'execution']
        for row in executions:
            require(row.get('phase') in ('identity','calibration','describe','validation','timing') and
                    row.get('status') == 'ok' and row.get('returncode') == 0,
                    'unexpected phase or failed execution')
            parsed = [json.loads(line) for line in row['stdout'].splitlines() if line.strip()]
            require(len(parsed) == 1 and parsed[0] == row['result'], 'result differs from raw stdout')
            kind = 'original' if row['candidate_id'] == 'original' else 'expanded'
            require(row['executable_sha256'] == contract['binaries'][kind], 'execution binary hash mismatch')
        require(not any(r.get('record_type') == 'budget_exhausted' for r in rows), 'admission budget exhausted')
        require(ledger[0]['event'] == 'start' and ledger[-1]['state']['inflight'] is None, 'budget not cleanly closed')
        require(all(x['state']['contract'] == contract for x in ledger), 'budget contract mismatch')
        starts = [x for x in ledger if x['event'] == 'work_start']
        ends = [x for x in ledger if x['event'] == 'work_finish']
        require(len(starts) == len(ends) == len(executions), 'budget and execution counts differ')
        require(all(start['state']['inflight'] == end['identity'] for start,end in zip(starts,ends)),
                'budget work identities differ')
        require(all(end['status'] == row['status'] for end,row in zip(ends,executions)), 'budget status differs')
        require(all((event['identity']['n'],event['identity']['batch'],event['identity']['candidate_id'],
                     event['identity']['phase'],event['identity']['round']) ==
                    (row['workload']['n'],row['workload']['batch'],row['candidate_id'],row['phase'],row['round'])
                    for event,row in zip(ends,executions)), 'budget work/execution identity mismatch')
        expected = {}
        for path in manifests:
            n,batch,selected = runner.load_manifest(path)
            require((n,batch) not in expected, 'duplicate workload manifests')
            expected[(n,batch)] = selected
        require(sum(map(len,expected.values())) == expected_count, 'selected candidate count mismatch')
        require(len({(r['n'],r['batch']) for r in rows if r.get('record_type') == 'workload_status'}) == len(expected),
                'workload status missing or unexpected')

        groups = {}
        active = None
        for row in rows:
            if row.get('record_type') == 'candidate_start':
                key = (row['n'],row['batch'],row['candidate_id'])
                require(key not in groups, 'duplicate candidate start')
                active = key; groups[key] = [row]
            elif row.get('record_type') == 'execution' and row.get('phase') == 'identity':
                active = None
            elif active is not None:
                groups[active].append(row)
        expected_keys = {(n,b,c['candidate_id']) for (n,b),selected in expected.items() for c in selected}
        require(set(groups) == expected_keys, 'candidate starts do not match manifest selection')
        require(not any(r.get('status') == 'numerically_invalid' for r in rows), 'numerically invalid candidate')
        for (n,batch), selected in expected.items():
            identity_rows = [r for r in executions if r['workload']['n'] == n and r['workload']['batch'] == batch and r['phase'] == 'identity']
            require(Counter(r['candidate_id'] for r in identity_rows) == Counter({'original':1,'expanded_default':1,'identity':1}),
                    f'{n}/{batch}: missing identity controls')
            controls = {r['candidate_id']:r for r in identity_rows}
            original = validation.check(controls['original'],gpu,original=True)
            default = validation.check(controls['expanded_default'],gpu)
            require(all((r['workload']['pattern'],r['workload']['seed']) == ('random',42) for r in identity_rows),
                    'identity input differs from protocol')
            comparison.compare(default,original)
            request,reason = runner.representable(default)
            require(request is not None, f'unsupported identity: {reason}')
            replay = validation.check(controls['identity'],gpu,request=request)
            require(validation.signature(default) == validation.signature(replay), 'identity replay differs')
            status = [r for r in rows if r.get('record_type') == 'workload_status' and (r['n'],r['batch']) == (n,batch)]
            require(len(status) == 1 and status[0]['status'] == 'identity_passed' and status[0]['request'] == request,
                    'identity disposition mismatch')
            original_sig = search_controller.source_signature(original,original=True)
            calibration = [r for r in executions if r['phase'] == 'calibration' and (r['workload']['n'],r['workload']['batch']) == (n,batch)]
            require(calibration, 'calibration missing')
            for index,row in enumerate(calibration):
                result = validation.check(row,gpu,original=True)
                require(row['workload']['graph_repeats'] == 64 * 2**index <= 2048, 'invalid calibration progression')
                require(search_controller.source_signature(result,original=True) == original_sig, 'calibration plan drift')
                if index < len(calibration)-1: require(min(result['raw_interval_ms_samples']) < 2, 'calibration overshot')
            require(min(result['raw_interval_ms_samples']) >= 2, 'calibration interval too short')
            graph = calibration[-1]['workload']['graph_repeats']
            for candidate in selected:
                cid,req = candidate['candidate_id'],candidate['request']
                segment = groups[(n,batch,cid)]
                require(segment[0]['request'] == req, 'candidate start request differs')
                descriptions = [r for r in segment if r.get('record_type') == 'execution' and r['phase'] == 'describe']
                require(len(descriptions) == 1 and descriptions[0]['candidate_id'] == cid, 'describe missing')
                desc = validation.check_describe(descriptions[0],gpu,req)
                dp = desc['precompile_kernels'][0]
                vals = [r for r in segment if r.get('record_type') == 'execution' and r['phase'] == 'validation']
                require(len(vals) == 15 and {(r['workload']['pattern'],r['workload']['seed']) for r in vals} ==
                        {(p,s) for p in runner.PATTERNS for s in runner.SEEDS}, 'candidate 5x3 checks incomplete')
                sig = None
                for row in vals:
                    require(row['candidate_id'] == cid and row['workload']['graph_repeats'] == graph, 'validation request identity differs')
                    result = validation.check(row,gpu,request=req)
                    cp = result['precompile_kernels'][0]
                    require(all(cp[k] == dp[k] for k in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')), 'describe/execute code differs')
                    actual = search_controller.source_signature(result,original=False)
                    require(sig is None or sig == actual, 'validation signature drift'); sig = actual
                timings = [r for r in segment if r.get('record_type') == 'execution' and r['phase'] == 'timing']
                require(len(timings) == 6, 'timing pair count mismatch')
                ratios = []
                for round_no in range(3):
                    pair = [r for r in timings if r['round'] == round_no]
                    require(len(pair) == 2 and {r['candidate_id'] for r in pair} == {'original',cid} and
                            {r['order'] for r in pair} == {0,1}, 'pair structure mismatch')
                    pair.sort(key=lambda r:r['order'])
                    order = runner.seeded_order(n,batch,cid,round_no)
                    require([r['candidate_id'] for r in pair] == [('original' if x == 'original' else cid) for x in order],
                            'pair order mismatch')
                    values = {}
                    for row in pair:
                        is_original = row['candidate_id'] == 'original'
                        require((row['workload']['pattern'],row['workload']['seed'],row['workload']['graph_repeats']) == ('random',42,graph),
                                'timing workload differs')
                        result = validation.check(row,gpu,original=is_original,request=None if is_original else req,timing=True)
                        require(result['warmup_actual'] >= 5 and result['warmup_wall_ms'] >= 1000, 'timing warmup too short')
                        require(search_controller.source_signature(result,original=is_original) == (original_sig if is_original else sig),
                                'timing plan drift')
                        values['original' if is_original else 'candidate'] = statistics.median(result['exec_ms_samples'])
                    ratios.append(values['candidate']/values['original'])
                pair_records = [r for r in segment if r.get('record_type') == 'screen_pair']
                require(len(pair_records) == 3 and {r['round'] for r in pair_records} == {0,1,2} and
                        all((r['n'],r['batch'],r['candidate_id']) == (n,batch,cid) for r in pair_records) and
                        all(math.isclose(r['ratio'],ratios[r['round']],rel_tol=1e-12) for r in pair_records), 'reported pair ratios differ')
                dispositions = [r for r in segment if r.get('record_type') == 'candidate_disposition']
                require(len(dispositions) == 1 and dispositions[0]['status'] == 'screen_complete' and
                        (dispositions[0]['n'],dispositions[0]['batch'],dispositions[0]['candidate_id']) == (n,batch,cid) and
                        dispositions[0]['request'] == req and dispositions[0]['pair_ratios'] == ratios and
                        math.isclose(dispositions[0]['median_ratio'],statistics.median(ratios),rel_tol=1e-12),
                        'candidate disposition differs')
        return {'schema':'frontier-admission-gate/v1', 'status':'pass', 'gpu':gpu,
                'execution_commit':execution_commit, 'candidates':len(expected_keys), 'workloads':len(expected),
                'scope':'one guarded session; all selected candidates validated and screened; no general device or performance guarantee',
                'binary_hashes_rechecked':True, 'guard_audit':guard_audit}
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(f'{type(error).__name__}: {error}')
        return {'schema':'frontier-admission-gate/v1', 'status':'fail', 'errors':errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('rows','guard','incidents','budget','manifest'):
        p.add_argument('--'+field, type=Path, action='append' if field == 'manifest' else 'store', required=True)
    p.add_argument('--gpu',required=True); p.add_argument('--execution-commit',required=True)
    p.add_argument('--expected-count',type=int,default=7)
    p.add_argument('--original',type=Path,required=True); p.add_argument('--expanded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = audit(runner.read_log(args.rows),runner.read_log(args.guard),runner.read_log(args.incidents),
                   runner.read_log(args.budget),args.manifest,args.gpu,args.execution_commit,args.expected_count,
                   {'original':args.original,'expanded':args.expanded})
    result['evidence_sha256'] = {str(path):runner.sha(path) for path in [args.rows,args.guard,args.incidents,args.budget,*args.manifest]}
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,indent=2,sort_keys=True); out.write('\n')
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__': sys.exit(main())
