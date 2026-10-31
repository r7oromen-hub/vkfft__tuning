#!/usr/bin/env python3
"""Aggregate a fixed 28-workload screen and independently re-audited confirmations."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
CONFIRM = HERE.parent / 'frontier_confirmation'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONFIRM))
import analyze
import evidence
spec = importlib.util.spec_from_file_location('frontier_confirmation_aggregate_audit', CONFIRM / 'audit.py')
confirmation_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation_audit)

WORKLOADS = tuple((n, b) for n in (128, 256, 512, 1024, 2048, 4096, 8192)
                  for b in (1, 16, 256, 4096))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return analyze.sha(path)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def confirm_files(report, plan_path, screen_paths, first_paths, second_paths, binaries):
    """Re-audit raw evidence and compare it with the supplied confirmation report."""
    need(report.get('schema') == 'frontier-confirmation-audit-report/v1' and report.get('status') == 'pass',
         'confirmation report did not pass')
    files = [plan_path, *screen_paths.values(), *first_paths.values(), *second_paths.values(), *binaries.values()]
    need(Counter(report.get('evidence_sha256', {}).values()) == Counter(sha(p) for p in files),
         'confirmation report evidence hashes differ from supplied files')
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    screen = evidence.audit_screen(plan, screen_paths, binaries)
    screen_contract = screen['screen_contract']
    need(plan.get('screen_execution_commit') == screen_contract.get('execution_commit') and
         plan.get('screen_binaries') == screen_contract.get('binaries'), 'plan screen provenance mismatch')
    need(report.get('screen') == screen, 'screen re-audit differs from confirmation report')
    proofs = {(c['n'], c['batch'], c['candidate_id']): c for c in screen['cases']}
    need(len(proofs) == len(screen['cases']), 'duplicate confirmation proof')
    sessions = []
    for kind, paths in (('first', first_paths), ('second', second_paths)):
        logs = {k: evidence.frontier.read_log(v) for k, v in paths.items()}
        need(logs['rows'], 'missing confirmation rows')
        contract = logs['rows'][0]['contract']
        generated = contract.get('generated_provenance') or {}
        need(contract.get('binaries') == screen_contract.get('binaries') and
             contract.get('vkfft_commit') == screen_contract.get('vkfft_commit') and
             generated.get('sha256') and generated.get('sha256') ==
             (screen_contract.get('generated_provenance') or {}).get('sha256'),
             'confirmation provenance differs from screen')
        audit = confirmation_audit.audit_session(logs['rows'], logs['guard'], logs['incidents'], logs['budget'],
                    plan, sha(plan_path), screen['gpu'], contract['execution_commit'], binaries, proofs, evidence.pair_order)
        need(audit.get('status') == 'pass' and audit.get('session_kind') == kind and audit == report.get(kind),
             kind + ' raw re-audit failed or differs from confirmation report')
        sessions.append(audit)
    first_hashes = {k: sha(v) for k, v in first_paths.items()}
    combined = confirmation_audit.audit_two(sessions[0], sessions[1], first_hashes)
    need(combined.get('status') == 'pass' and combined == report.get('combined'),
         'two-session raw re-audit failed or differs from confirmation report')
    # Catch a file changed during re-audit; the report is bound to the bytes verified above.
    need(Counter(report['evidence_sha256'].values()) == Counter(sha(p) for p in files),
         'confirmation evidence changed during re-audit')
    return report


def aggregate(coverage, rows, rows_sha256, confirmation_report=None, confirmation_report_sha256=None):
    """Aggregate audit outputs; callers must first call confirm_files for confirmation inputs."""
    need(digest(rows_sha256), 'invalid raw rows hash')
    screen = analyze.analyze(coverage, rows, rows_sha256, 20261031)
    gpu, screen_commit = screen['gpu'], screen['execution_commit']
    by_work = {(w['n'], w['batch']): w for w in screen['workloads']}
    need(set(by_work) <= set(WORKLOADS), 'screen has a workload outside the fixed 28')
    statuses = {(r['n'], r['batch']):r for r in rows if r.get('record_type') == 'workload_status'}
    identities = {k:v['request'] for k,v in statuses.items() if v.get('status') == 'identity_passed'}
    winners = {}
    nondefault_best = {}
    identity_counts = Counter()
    nondefault_counts = Counter()
    nondefault_items = []
    for r in rows:
        if r.get('record_type') != 'candidate_disposition' or r.get('status') != 'screen_complete':
            continue
        key = (r['n'], r['batch'])
        need(key in identities, 'screen candidate lacks a passing identity')
        if r['request'] == identities[key]:
            identity_counts[key] += 1
            continue
        nondefault_counts[key] += 1
        nondefault_items.append({'ratio':r['median_ratio']})
        if key not in nondefault_best or (r['median_ratio'], r['candidate_id']) < (nondefault_best[key]['median_ratio'], nondefault_best[key]['candidate_id']):
            nondefault_best[key] = r
        if r['median_ratio'] >= 1:
            continue
        if key not in winners or (r['median_ratio'],r['candidate_id']) < (winners[key]['median_ratio'],winners[key]['candidate_id']):
            winners[key] = r
    confirmed = {}
    negative_controls = []
    confirmation_commit = None
    if confirmation_report is not None:
        need(digest(confirmation_report_sha256), 'invalid confirmation report hash')
        report = confirmation_report
        need(report.get('schema') == 'frontier-confirmation-audit-report/v1' and report.get('status') == 'pass',
             'confirmation audit report did not pass')
        source = report.get('screen') or {}
        contract = source.get('screen_contract') or {}
        need(source.get('gpu') == gpu and contract.get('gpu') == gpu and
             contract.get('execution_commit') == screen_commit and
             (source.get('source_sha256') or {}).get('rows') == rows_sha256,
             'confirmation screen GPU, commit or raw rows hash mismatch')
        first, second = report.get('first') or {}, report.get('second') or {}
        need(first.get('gpu') == second.get('gpu') == gpu and
             first.get('source_sha256') == second.get('source_sha256') == source.get('source_sha256') and
             first.get('binaries') == second.get('binaries') == contract.get('binaries') and
             first.get('vkfft_commit') == second.get('vkfft_commit') == contract.get('vkfft_commit'),
             'confirmation source, binary, VkFFT commit or GPU mismatch')
        confirmation_commit = first.get('execution_commit')
        need(isinstance(confirmation_commit, str) and len(confirmation_commit) == 40 and
             all(c in '0123456789abcdef' for c in confirmation_commit) and
             second.get('execution_commit') == confirmation_commit, 'confirmation execution commits differ')
        first_hashes = (second.get('first_session') or {}).get('evidence_sha256')
        need(isinstance(first_hashes, dict) and set(first_hashes) == {'rows','guard','incidents','budget'} and
             all(digest(v) for v in first_hashes.values()), 'missing first session evidence hashes')
        combined = confirmation_audit.audit_two(first, second, first_hashes)
        need(combined.get('status') == 'pass' and combined == report.get('combined'),
             'confirmation two-session audit mismatch')
        proofs = {(p['n'],p['batch'],p['candidate_id']):p for p in source.get('cases', [])}
        need(len(proofs) == len(source.get('cases', [])), 'duplicate confirmation screen proof')
        need(len(combined['results']) == len(proofs), 'confirmation proof count mismatch')
        for result in combined['results']:
            key=(result['n'],result['batch']); cid=result['candidate_id']
            proof=proofs.get((*key,cid)); role=result.get('role','winner')
            need(key in WORKLOADS and proof is not None and result['request'] == proof['request'] and
                 role == proof.get('role','winner'), 'confirmation case differs from screen proof')
            if role == 'negative_control':
                negative_controls.append(result)
                continue
            need(role == 'winner' and key in winners and winners[key]['candidate_id'] == cid and
                 winners[key]['request'] == result['request'] and key not in confirmed,
                 'confirmed winner differs from selected screen winner')
            r1 = result['first']['geometric_mean_ratio']; r2 = result['second']['geometric_mean_ratio']
            need(positive(r1) and positive(r2), 'invalid confirmation ratios')
            confirmed[key] = {'candidate_id':cid, 'request':result['request'],
                              'first':result['first'], 'second':result['second'],
                              'equal_weight_geometric_mean_ratio': math.exp((math.log(r1)+math.log(r2))/2),
                              'same_improvement_direction': result['same_improvement_direction'],
                              'both_sessions_practical':result['both_sessions_practical']}
    reports=[]
    for n,batch in WORKLOADS:
        key=(n,batch); w=by_work.get(key); count=w['outcomes']['screen_complete_count'] if w else 0
        dispositions=Counter(c['status'] for c in coverage['cases'] if (c['n'],c['batch']) == key)
        default_status=w['default_status'] if w else 'not_observed'
        default_valid=default_status in ('identity_passed','unsupported_identity')
        winner=winners.get(key); confirmation=confirmed.get(key)
        nondefault_count=nondefault_counts[key]
        if not nondefault_count: state='not_covered'
        elif winner is None: state='no_screen_winner'
        elif confirmation is None: state='not_confirmed'
        elif confirmation['both_sessions_practical']: state='confirmed_practical'
        else: state='confirmed_not_practical'
        fallback=None
        if default_valid:
            if confirmation and confirmation['both_sessions_practical']:
                fallback={'kind':'confirmed_candidate','ratio':confirmation['equal_weight_geometric_mean_ratio'],
                          'candidate_id':confirmation['candidate_id']}
            else: fallback={'kind':'default','ratio':1.0}
        best_nondefault=nondefault_best.get(key)
        if nondefault_count and default_valid:
            if best_nondefault['median_ratio'] < 1:
                best_with_default={'kind':'candidate','n':n,'batch':batch,
                    'candidate_id':best_nondefault['candidate_id'], 'request':best_nondefault['request'],
                    'ratio':best_nondefault['median_ratio']}
            else:
                best_with_default={'kind':'default','ratio':1.0}
        else:
            best_with_default=None
        reports.append({'n':n,'batch':batch,'default_status':default_status,
                        'screen_selected_count':sum(dispositions.values()),
                        'screen_dispositions':{status:dispositions[status] for status in
                            ('screen_complete','scheduler_rejected','resource_rejected','partial','selected_not_started')},
                        'screen_complete_count':count,
                        'identity_screen_complete_count':identity_counts[key],
                        'nondefault_screen_complete_count':nondefault_count,
                        'coverage_reason':('nondefault_screened' if nondefault_count else
                                           'identity_only' if identity_counts[key] else 'no_complete_candidates'),
                        'screen_best':w['best_screened'] if w else None,
                        'screen_best_nondefault':({'n':n,'batch':batch,
                            'candidate_id':best_nondefault['candidate_id'], 'request':best_nondefault['request'],
                            'ratio':best_nondefault['median_ratio']} if best_nondefault else None),
                        'screen_best_with_default':best_with_default,
                        'screen_winner_candidate_id':winner['candidate_id'] if winner else None,
                        'confirmation_status':state,'confirmation':confirmation,'fallback':fallback})
    state_counts=Counter(w['confirmation_status'] for w in reports)
    covered=sum(w['nondefault_screen_complete_count']>0 for w in reports)
    known_screen=[w for w in reports if w['screen_best_with_default'] is not None]
    fallback_values=[w['fallback']['ratio'] for w in reports if w['fallback'] is not None]
    all_defaults=len(fallback_values)==len(WORKLOADS)
    faster=sum(w['screen_best_with_default'] is not None and w['screen_best_with_default']['ratio']<1 for w in reports)
    two_percent=sum(w['screen_best_with_default'] is not None and w['screen_best_with_default']['ratio']<=.98 for w in reports)
    def stats(values, complete):
        return {'complete_28':complete,'known_count':len(values),
                'mean_latency_reduction':statistics.mean(1-r for r in values) if complete else None,
                'geometric_mean_speedup':math.exp(-statistics.mean(math.log(r) for r in values)) if complete else None}
    return {'schema':'frontier-confirmed-aggregate/v1','gpu':gpu,
            'screen_execution_commit':screen_commit,'confirmation_execution_commit':confirmation_commit,
            'source_rows_sha256':rows_sha256,'confirmation_report_sha256':confirmation_report_sha256,
            'candidate_stats':screen['overall'],
            'nondefault_candidate_stats':analyze.ratio_stats(nondefault_items),
            'workload_stats':{'denominator':28,'covered_count':covered,
                'screen_faster_count':faster,'screen_faster_fraction_of_28':faster/28,
                'screen_at_least_2pct_count':two_percent,'screen_at_least_2pct_fraction_of_28':two_percent/28,
                'screen_winner_count':len(winners),'screen_winner_fraction_of_28':len(winners)/28,
                'confirmation_status_counts':{k:state_counts[k] for k in
                    ('not_covered','no_screen_winner','not_confirmed','confirmed_not_practical','confirmed_practical')},
                'confirmed_practical_fraction_of_28':state_counts['confirmed_practical']/28,
                'unknown_screen_count':28-len(known_screen)},
            'screen_summary':stats([w['screen_best_with_default']['ratio'] for w in known_screen],len(known_screen)==28),
            'fallback_summary':stats(fallback_values,all_defaults), 'workloads':reports,
            'negative_controls':negative_controls,
            'scope':{'screen':'best among completed budget-limited nondefault screening candidates and the default; requires at least one nondefault candidate, subject to selection optimism; not a global optimum or independent confirmation',
                     'confirmation':'equal weight to two independent session log ratios; within-session intervals do not correct screening or multiple testing',
                     'workload_counts':'fractions over 28 count observed results only; unobserved and unconfirmed workloads remain unknown',
                     'fallback':'deployment policy estimate: use candidate only if both sessions meet practical criterion, otherwise observed default; unknown default remains unknown',
                     'weighting':'28 workloads have equal weight; mean latency reduction is arithmetic mean of 1-r, geometric speedup is exp(-mean(log r))'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('coverage','rows','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('confirmation-audit','plan','screen-guard','screen-incidents','screen-budget',
                 'first-dir','second-dir','original','expanded'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    confirmation_args=[a.confirmation_audit,a.plan,a.screen_guard,a.screen_incidents,a.screen_budget,
                       a.first_dir,a.second_dir,a.original,a.expanded]
    need(not any(confirmation_args) or all(confirmation_args), 'confirmation requires report, plan, all raw evidence and binaries')
    coverage_bytes=a.coverage.read_bytes(); coverage=json.loads(coverage_bytes)
    rows_sha=sha(a.rows); rows=analyze.read_rows(a.rows); need(sha(a.rows)==rows_sha, 'screen rows changed while reading')
    report=None; report_sha=None
    if a.confirmation_audit:
        data=a.confirmation_audit.read_bytes();report_sha=hashlib.sha256(data).hexdigest();report=json.loads(data)
        paths={'rows':a.rows,'guard':a.screen_guard,'incidents':a.screen_incidents,'budget':a.screen_budget}
        first={k:a.first_dir/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
        second={k:a.second_dir/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
        confirm_files(report,a.plan,paths,first,second,{'original':a.original,'expanded':a.expanded})
        need(sha(a.confirmation_audit)==report_sha, 'confirmation report changed while reading')
    result=aggregate(coverage,rows,rows_sha,report,report_sha)
    result['coverage_sha256']=hashlib.sha256(coverage_bytes).hexdigest()
    need(sha(a.coverage)==result['coverage_sha256'], 'coverage report changed while reading')
    with a.output.open('x',encoding='utf8') as out: json.dump(result,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
