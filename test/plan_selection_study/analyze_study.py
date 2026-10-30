#!/usr/bin/env python3
"""Reproduce the Chinese H1-H3 report and scientific figures from one study root.

No GPU access. Run with a Python providing scipy and matplotlib. Missing stages
remain incomplete; measurements from another root or a pilot are never searched.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import platform
import statistics
import sys

from cost_audit import audit_pair
from report_helpers import STAGES, digest, geometric, headline_stat, holm, paired, read_evidence, selection_eligible
from study_common import load_design, write_json
from trace_audit import audit_trace, fingerprint, plan_fingerprints


def optional_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def h1_analysis(design, rows, selection):
    effects, screen, comparisons = [], [], []
    for w in design['workloads']:
        wid = w['id']
        sr = [r for r in rows['screen'] if r['workload_id'] == wid and selection_eligible(r) and r['candidate_id'] != 'default_replica']
        ms = {r['candidate_id']:r['result']['exec_ms_median'] for r in sr}
        if ms and 'default' in ms:
            fastest = min(ms,key=lambda c:(ms[c],c != 'default',c))
            gap = ms['default']/ms[fastest]
            screen.append(dict(workload_id=wid,n=w['n'],batch=w['batch'],legal_candidates=len(ms),
                               fastest_tested=fastest,default_over_tested_min=gap,
                               default_within2pct=gap <= 1.02,default_within5pct=gap <= 1.05,
                               candidate_default_ratios={c:ms['default']/v for c,v in ms.items()},
                               caveat='selected screen minimum; optimistic tested-set diagnostic'))
        choice = (selection or {}).get('workloads',{}).get(wid)
        if not choice:
            effects.append(dict(workload_id=wid,n=w['n'],batch=w['batch'],status='incomplete_missing_frozen_selection'))
            continue
        cr = [r for r in rows['confirmation'] if r['workload_id'] == wid]
        pair = paired(cr,'default',choice['winner'],3)
        effects.append(dict(workload_id=wid,n=w['n'],batch=w['batch'],winner=choice['winner'],
                            status='complete' if pair else 'unestimable_incomplete_three_round_pair',**(pair or {})))
        for cid in choice['candidates']:
            if cid == 'default_replica':
                continue
            effect = paired(cr,'default',cid,3)
            comparisons.append(dict(workload_id=wid,candidate_id=cid,status='complete' if effect else 'incomplete',**(effect or {})))
    complete = [e for e in effects if e['status'] == 'complete']
    groups = []
    for n in sorted({w['n'] for w in design['workloads']}):
        values = [e['speedup'] for e in complete if e['n'] == n]
        groups.append(dict(n=n,confirmed_workloads=len(values),expected_workloads=sum(w['n'] == n for w in design['workloads']),
                           geometric_speedup=geometric(values)))
    return dict(effects=effects,screen=screen,comparisons=comparisons,N_groups=groups,
                confirmed_workloads=len(complete),expected_workloads=len(effects),
                equal_workload_geometric_speedup=geometric(e['speedup'] for e in complete),
                median_workload_speedup=statistics.median(e['speedup'] for e in complete) if complete else None,
                aggregate_scope='complete_fixed_matrix' if len(complete) == len(effects) else 'partial_observed_workloads_only',
                regressions=[e['workload_id'] for e in complete if e['speedup'] < 1],
                nominal_point_estimate_below_one=[e['workload_id'] for e in complete if e['speedup'] < 1],
                latency_increase_at_least5pct=[e['workload_id'] for e in complete if 1/e['speedup'] >= 1.05],
                worst_nominal_latency_increase=max((max(0.,1/e['speedup']-1) for e in complete),default=None),
                decline_significance='not_tested_no_predeclared_directional_decline_test',
                near_best2=sum(e['default_within2pct'] for e in screen),near_best5=sum(e['default_within5pct'] for e in screen))


def h2_analysis(design, rows):
    structures, counts, contrasts, crossovers, group_gains, contrast_coverage = [], [], [], [], [], []
    requests = {c['id']:c for c in design['candidates']}
    for w in design['workloads']:
        good = [r for r in rows['screen'] if r['workload_id'] == w['id'] and r['status'] == 'ok']
        local = []
        for r in good:
            fp = plan_fingerprints(r['result'])
            item = dict(workload_id=w['id'],candidate_id=r['candidate_id'],requested=r.get('config',{}),
                        resolved=r['result'].get('resolved_config'),requested_class=fingerprint(r.get('config',{})),
                        resolved_class=fingerprint(r['result'].get('resolved_config')),core=fp['core'],
                        canonical=fp['canonical'],source_compiler=fp['source_compiler'],
                        canonical_descriptor=fp['canonical_descriptor'],core_descriptor=fp['core_descriptor'],
                        compiled_diagnostics=r['result'].get('compiled_diagnostics'),
                        kernel_attributes=r['result'].get('kernel_attributes'),record_sha256=r['record_sha256'])
            item['selection_eligible'] = selection_eligible(r)
            structures.append(item)
            local.append(item)
        counts.append(dict(workload_id=w['id'],n=w['n'],batch=w['batch'],accepted_successful=len(local),
                           classes={key:len({x[key] for x in local}) for key in ('requested_class','resolved_class','core','canonical','source_compiler')}))
        groups = dict(NP0=lambda c:'_np0' in c['id'],coalescing=lambda c:c['coalesced_memory'] != 0,
                      LUT4=lambda c:c['use_lut_4step'] != 0,forced_stage=lambda c:c['swap_to_2stage'] != 0 or c['swap_to_3stage'] != 0)
        for label,predicate in groups.items():
            inside = [x for x in local if predicate(requests[x['candidate_id']])]
            outside = [x for x in local if not predicate(requests[x['candidate_id']])]
            group_gains.append(dict(workload_id=w['id'],group=label,successful_candidates=len(inside),
                                    unique_classes_added={key:len({x[key] for x in inside}-{x[key] for x in outside}) for key in ('core','source_compiler')},
                                    interpretation='set difference against all other frozen requests; not an additive attribution'))
    for contrast in design['structure_analysis']['predeclared_batch_contrasts']:
        ref, target = contrast['reference'], contrast['candidate']
        for w in design['workloads']:
            if target.endswith('s2_8192') and w['n'] < 8192 or target.endswith('s3_65536') and w['n'] < 65536:
                continue
            if target == 'a128_r4_l1_np0' and w['n'] & (w['n']-1) == 0:
                continue
            sr = [r for r in rows['screen'] if r['workload_id'] == w['id']]
            cr = [r for r in rows['confirmation'] if r['workload_id'] == w['id']]
            s, c = paired(sr,ref,target,1), paired(cr,ref,target,3)
            contrasts.append(dict(workload_id=w['id'],n=w['n'],batch=w['batch'],reference=ref,target=target,
                                  screen=s,confirmation=c,missing_confirmation=not bool(c)))
        complete_endpoints,eligible_lengths = [],[]
        for n in sorted({w['n'] for w in design['workloads']}):
            group = sorted([x for x in contrasts if x['n'] == n and x['reference'] == ref and x['target'] == target],key=lambda x:x['batch'])
            if len(group) < 2:
                continue
            eligible_lengths.append(n)
            a,b = group[0], group[-1]
            if not all(x['screen'] and x['confirmation'] for x in (a,b)):
                continue
            complete_endpoints.append(n)
            sx,sy = a['screen']['speedup'],b['screen']['speedup']
            cx,cy = a['confirmation']['speedup'],b['confirmation']['speedup']
            opposing = ((sx >= 1.05 and cx >= 1.05 and sy <= 1/1.05 and cy <= 1/1.05) or
                        (sy >= 1.05 and cy >= 1.05 and sx <= 1/1.05 and cx <= 1/1.05))
            if opposing:
                crossovers.append(dict(n=n,reference=ref,target=target,low_batch=a,high_batch=b,
                                       status='exploratory_material_reversal_until_both_headline_directions_pass'))
        contrast_coverage.append(dict(reference=ref,target=target,eligible_N=eligible_lengths,
                                      complete_two_endpoint_confirmation_N=complete_endpoints,
                                      missing_two_endpoint_confirmation_N=sorted(set(eligible_lengths)-set(complete_endpoints))))
    return dict(structures=structures,counts=counts,group_class_gains=group_gains,contrasts=contrasts,
                contrast_coverage=contrast_coverage,exploratory_crossovers=crossovers)


def headline_analysis(rows, manifest, gate, design=None):
    tests, controls = [], []
    if not manifest:
        return dict(status='incomplete_missing_frozen_manifest',tests=[],negative_controls=[],family_size=0)
    for w in manifest['workloads']:
        rs = [r for r in rows if r['workload_id'] == w['workload_id']]
        control = paired(rs,'default','default_replica',10)
        valid = bool(control and all(abs(math.log(q)) <= math.log(1.05) for q in control['ratios']))
        required = {(k,c) for k in range(10) for c in w.get('candidates',[])}
        found = [(r['round'],r['candidate_id']) for r in rs]
        complete = bool(required and set(found) == required and len(found) == len(required) and all(r['status'] == 'ok' for r in rs))
        controls.append(dict(workload_id=w['workload_id'],valid=valid,candidate_round_coverage_complete=complete,comparison=control))
    control_valid = {x['workload_id']:x['valid'] for x in controls}
    for spec in manifest['directional_tests']:
        rs = [r for r in rows if r['workload_id'] == spec['workload_id']]
        effect = paired(rs,spec['reference'],spec['target'],10)
        stat = headline_stat(effect['ratios'] if effect else [])
        inference_reason = stat.pop('reason')
        tests.append(dict(spec,**stat,selection_reason=spec.get('reason'),inference_reason=inference_reason,
                          ratios=effect['ratios'] if effect else [],
                          record_hashes=effect['record_hashes'] if effect else [],
                          quality_valid=bool(gate and gate.get('passed') and control_valid.get(spec['workload_id']))))
    holm(tests)
    # A frozen winner test may be deduplicated with a crossover direction and
    # retain its earlier reason. Identify exact pair/directions, not reason text.
    cross_pairs = []
    if design and manifest.get('crossovers'):
        lookup = {(x['workload_id'],x['reference'],x['target']):x for x in tests}
        for pair in manifest['crossovers']:
            directions = pair.get('directions',[])
            matched = [lookup.get((d['workload_id'],d['reference'],d['target'])) for d in directions]
            if len(matched) == 2 and all(matched):
                a,b = matched
                valid_pair = a['reference'] == b['target'] and a['target'] == b['reference']
                cross_pairs.append(dict(forward=a,reverse=b,confirmed=valid_pair and a['clear_gain'] and b['clear_gain']))
    elif design and manifest.get('missing_crossover') is False:
        by_wid = {w['id']:w for w in design['workloads']}
        permitted = {frozenset((c['reference'],c['candidate'])) for c in design['structure_analysis']['predeclared_batch_contrasts']}
        for i,a in enumerate(tests):
            for b in tests[i+1:]:
                aw,bw = by_wid.get(a['workload_id']),by_wid.get(b['workload_id'])
                if not aw or not bw or aw['n'] != bw['n'] or aw['batch'] == bw['batch']:
                    continue
                endpoints = sorted(w['batch'] for w in design['workloads'] if w['n'] == aw['n'])
                if sorted([aw['batch'],bw['batch']]) != [endpoints[0],endpoints[-1]]:
                    continue
                if (a['reference'] == b['target'] and a['target'] == b['reference'] and
                        frozenset((a['reference'],a['target'])) in permitted):
                    cross_pairs.append(dict(forward=a,reverse=b,confirmed=a['clear_gain'] and b['clear_gain']))
    crossover_valid = any(x['confirmed'] for x in cross_pairs)
    return dict(status='complete' if gate and gate.get('passed') and all(c['candidate_round_coverage_complete'] for c in controls) and all(len(x['ratios']) == 10 for x in tests) else 'incomplete',
                tests=tests,negative_controls=controls,family_size=len(tests),confirmed_crossover=crossover_valid,crossover_pairs=cross_pairs,
                unit='10 paired independent process medians; sample loops are not replications')


def h3_analysis(design, rows, selection, gate, confirmation=()):
    pairs, pipelines = [], []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['workload_id'],r['round']].append(r)
    for wid in design['cost_profile']['workloads']:
        for k in range(3):
            group = groups[wid,k]
            modes = {m:[r for r in group if r['candidate_id'] == m] for m in ('full-search-cost','describe-select-cost')}
            choice = (selection or {}).get('workloads',{}).get(wid,{}).get('winner')
            if any(len(v) != 1 for v in modes.values()) or not choice:
                pairs.append(dict(workload_id=wid,round=k,ok=False,errors=['incomplete_pipeline_pair_or_frozen_choice']))
                continue
            full, desc = modes['full-search-cost'][0], modes['describe-select-cost'][0]
            audit = audit_pair(full.get('result',{}),desc.get('result',{}),full.get('result',{}).get('candidate_order_ids',[]),choice)
            expected = {c['id'] for c in design['candidates']}
            actual_order = full.get('result',{}).get('candidate_order_ids',[])
            if set(actual_order) != expected or len(actual_order) != len(expected):
                audit['ok'] = False
                audit['errors'].append('not_all_frozen_candidates')
            if full['status'] != 'ok' or desc['status'] != 'ok':
                audit['ok'] = False
                audit['errors'].append('pipeline_process_failed')
            fw, dw = full.get('result',{}).get('context_pipeline_wall_ms'), desc.get('result',{}).get('context_pipeline_wall_ms')
            pairs.append(dict(workload_id=wid,round=k,**audit,full_wall_ms=fw,describe_known_choice_wall_ms=dw,
                              construction_ratio=fw/dw if audit['ok'] and fw and dw else None))
    for r in rows:
        s = r.get('result',{})
        totals = collections.defaultdict(float)
        apps = []
        roots = 0.
        for app in s.get('apps',[]):
            tr = audit_trace(app.get('result',{}).get('init_trace',{}))
            roots += tr['root_inclusive_ms']
            for stage,value in tr['stage_totals'].items():
                totals[stage] += value['exclusive_ms']
            apps.append(dict(candidate_id=app['candidate_id'],role=app['role'],sequence_index=app['sequence_index'],
                             first_nvrtc_app_in_context=app.get('first_nvrtc_app_in_context'),
                             nvrtc_invocations_before_app=app.get('nvrtc_invocations_before_app'),trace_ok=tr['ok'],
                             exclusive_stages_ms={k:v['exclusive_ms'] for k,v in tr['stage_totals'].items()},
                             root_inclusive_ms=tr['root_inclusive_ms']))
        wall = s.get('context_pipeline_wall_ms')
        pipelines.append(dict(workload_id=r['workload_id'],round=r['round'],mode=s.get('mode'),status=r['status'],
                              context_pipeline_wall_ms=wall,context_bootstrap_ms=s.get('context_bootstrap_ms'),
                              shared_setup_ms=s.get('shared_buffer_stream_setup_ms'),shared_cleanup_ms=s.get('shared_buffer_stream_cleanup_ms'),
                              subprocess_wall_ms=1000*r['wall_seconds'] if r.get('wall_seconds') is not None else None,exclusive_stages_ms=dict(totals),
                              traced_roots_ms=roots,wall_minus_traced_roots_ms=wall-roots if wall is not None else None,
                              apps=apps,record_sha256=r['record_sha256']))
    stage_summary, amortization, enumeration_diagnostic = [], [], []
    valid_keys = {(p['workload_id'],p['round']) for p in pairs if p['ok']}
    compilation_stages = {'nvrtc_program_create','nvrtc_compile','nvrtc_program_destroy','binary_extract','module_load','module_get_function','function_attribute_setup'}
    for mode in ('full-search-cost','describe-select-cost'):
        good = [p for p in pipelines if p['mode'] == mode and (p['workload_id'],p['round']) in valid_keys]
        totals = collections.defaultdict(float)
        for p in good:
            for stage,value in p['exclusive_stages_ms'].items():
                totals[stage] += value
        traced = sum(totals.values())
        wall = sum(p['context_pipeline_wall_ms'] for p in good)
        compiled = sum(v for k,v in totals.items() if k in compilation_stages)
        stage_summary.append(dict(mode=mode,processes=len(good),exclusive_stages_sum_ms=dict(totals),
                                  compilation_loading_exclusive_ms=compiled,trace_exclusive_ms=traced,pipeline_wall_ms=wall,
                                  compilation_fraction_of_traced=compiled/traced if traced else None,
                                  compilation_fraction_of_pipeline=compiled/wall if wall else None))
    for wid in design['cost_profile']['workloads']:
        winner = (selection or {}).get('workloads',{}).get(wid,{}).get('winner')
        deltas = []
        for r in rows:
            if r['workload_id'] != wid or r['candidate_id'] != 'full-search-cost' or (wid,r['round']) not in valid_keys:
                continue
            apps = {a['candidate_id']:a['result'] for a in r['result']['apps']}
            if winner in apps and 'default' in apps:
                if all(isinstance(apps[c].get('initialize_ms'),(int,float)) for c in ('default',winner)):
                    deltas.append(apps[winner]['initialize_ms']-apps['default']['initialize_ms'])
        cr = [r for r in confirmation if r['workload_id'] == wid]
        effect = paired(cr,'default',winner,3) if winner else None
        execution_delta = None
        if effect:
            by = {(r['round'],r['candidate_id']):r for r in cr if r['status'] == 'ok'}
            execution_delta = statistics.mean(by[k,'default']['result']['exec_ms_median']-by[k,winner]['result']['exec_ms_median'] for k in range(3))
        setup_delta = statistics.mean(deltas) if len(deltas) == 3 else None
        amortization.append(dict(workload_id=wid,winner=winner,setup_delta_ms=setup_delta,execution_delta_ms=execution_delta,
                                 break_even_executions=math.ceil(max(0.,setup_delta)/execution_delta) if setup_delta is not None and execution_delta is not None and execution_delta > 0 else None,
                                 setup_policy='mean initialize_ms difference in full-search-cost shared-context candidate order; includes tracing/hash and final initialization sync; excludes cleanup/context bootstrap/search benchmarking/model scoring',
                                 limitation='component estimate using independent three-round execution deltas; no finite value for nonpositive or unresolved delta'))
        described = [r for r in rows if r['workload_id'] == wid and r['candidate_id'] == 'describe-select-cost' and (wid,r['round']) in valid_keys]
        summed_initialization = []
        for r in described:
            values = [a['result'].get('initialize_ms') for a in r['result']['apps']]
            if len(values) == len(design['candidates'])+1 and all(isinstance(v,(int,float)) and math.isfinite(v) and v >= 0 for v in values):
                summed_initialization.append(sum(values))
        defaults = [r for r in cr if r['candidate_id'] == 'default' and selection_eligible(r)]
        d_values = [r['result'].get('initialize_ms') for r in defaults]
        components_complete = (len(summed_initialization) == 3 and len(defaults) == 3 and
                               {r['round'] for r in defaults} == {0,1,2} and
                               all(isinstance(v,(int,float)) and math.isfinite(v) and v >= 0 for v in d_values))
        enumeration_mean = statistics.mean(summed_initialization) if len(summed_initialization) == 3 else None
        default_mean = statistics.mean(d_values) if components_complete else None
        extra = enumeration_mean-default_mean if components_complete else None
        enumeration_diagnostic.append(dict(workload_id=wid,winner=winner,components_complete=components_complete,
            S3_describe26_plus_materialize_initialize_ms=summed_initialization,
            S3_mean_summed_initialize_ms=enumeration_mean,S2_mean_default_initialize_ms=default_mean,
            additional_initialize_component_ms=extra,independent_execution_delta_ms=execution_delta,
            component_break_even_executions=math.ceil(max(0.,extra)/execution_delta) if extra is not None and execution_delta is not None and execution_delta > 0 else None,
            initialization_component_does_not_increase=extra is not None and extra <= 0,
            S3_source_record_hashes=[r['record_sha256'] for r in described],S2_default_source_record_hashes=[r['record_sha256'] for r in defaults],
            setup_policy='nonpaired cross-stage component diagnostic: mean S3 sum of 26 describe + 1 materialize initialize_ms minus mean S2 single-default initialize_ms; excludes context/bootstrap/shared setup/cleanup/scoring/validation/fallback',
            limitation='not whole pipeline wall minus one initialization; not default end-to-end planning speedup or online ML result'))
    return dict(status='established_fixed_matrix' if gate and gate.get('passed') and pairs and all(p['ok'] for p in pairs) else 'unresolved_or_incomplete',
                pairs=pairs,pipelines=pipelines,stage_summary=stage_summary,amortization=amortization,
                nonpaired_enumeration_vs_single_default_component_diagnostic=enumeration_diagnostic,expected_pairs=24,
                primary_pipeline_processes=len(rows),expected_primary_pipeline_processes=48,passed_pairs=sum(p['ok'] for p in pairs),
                interpretation='actual describe-all plus offline known-choice construction lower bound; not an online selector')


def cost_smoke_analysis(design, rows):
    """Engineering gate only: deliberately omit latency/cost effect estimators."""
    spec = design['preflight']['cost_smoke']
    expected = {c['id'] for c in design['candidates']}
    pairs, applications = [], []
    for wid in spec['workloads']:
        modes = {m:[r for r in rows if r['workload_id'] == wid and r['candidate_id'] == m]
                 for m in ('full-search-cost','describe-select-cost')}
        if any(len(v) != 1 for v in modes.values()):
            pairs.append(dict(workload_id=wid,ok=False,errors=['incomplete_smoke_pair']))
            continue
        full,desc = modes['full-search-cost'][0],modes['describe-select-cost'][0]
        order = full['result']['candidate_order_ids']
        result = audit_pair(full['result'],desc['result'],order,spec['known_selected_candidate'])
        if len(order) != len(expected) or set(order) != expected:
            result['ok'] = False
            result['errors'].append('smoke_not_all_frozen_requests')
        if full['status'] != 'ok' or desc['status'] != 'ok':
            result['ok'] = False
            result['errors'].append('smoke_process_failed')
        pairs.append(dict(workload_id=wid,**result,record_hashes=[full['record_sha256'],desc['record_sha256']]))
    for row in rows:
        for app in row.get('result',{}).get('apps',[]):
            result = app['result']
            trace = audit_trace(result.get('init_trace',{}))
            applications.append(dict(workload_id=row['workload_id'],mode=row['candidate_id'],candidate_id=app['candidate_id'],
                                      role=app['role'],trace_ok=trace['ok'],trace_errors=trace['errors'],
                                      counters=result.get('init_trace',{}).get('counters',{}),
                                      uploads=len(result.get('precompile_kernels',[])),
                                      first_nvrtc_app_in_context=app.get('first_nvrtc_app_in_context'),
                                      nvrtc_invocations_before_app=app.get('nvrtc_invocations_before_app'),
                                      record_sha256=row['record_sha256']))
    descriptions = [a for a in applications if a['role'] == 'describe']
    return dict(status='passed_engineering_only' if len(rows) == 4 and all(p['ok'] for p in pairs) and all(a['trace_ok'] for a in applications) else 'incomplete_or_failed',
                processes=len(rows),expected_processes=spec['pipeline_processes'],pairs=pairs,applications=applications,
                describe_apps=len(descriptions),full_and_materialize_apps=len(applications)-len(descriptions),
                describe_counter_totals={k:sum(a['counters'].get(k,0) for a in descriptions) for k in ('nvrtc_compile','module_load','driver_kernel_launch')},
                actual_descriptor_source_compiler_comparisons=sum(len(p.get('comparisons',[])) for p in pairs),
                excluded_from_H3_primary_cost_estimates=True,
                scope='two-workload preflight engineering smoke; does not replace eight workloads x three rounds x two pipelines')


def make_figures(out, design, evidence, h1, h2, h3, headline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    paths = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png','svg'):
            path = folder/(name+'.'+ext)
            fig.savefig(path,dpi=200,bbox_inches='tight')
            paths.append(path)
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,3.5))
    counts = [evidence['coverage'][s]['accepted_records'] for s in STAGES]
    eligible = [evidence['coverage'][s]['eligible_records'] for s in STAGES]
    ax.bar(STAGES,eligible,color='#497a91',label='Successful / eligible')
    ax.bar(STAGES,[n-e for n,e in zip(counts,eligible)],bottom=eligible,color='#bd6658',label='Failed / excluded')
    ax.set_ylim(0,max(1,max(counts,default=0)*1.15))
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Audited accepted process records')
    ax.tick_params(axis='x',rotation=25)
    ax.set_title('Retained evidence coverage (zero means missing, not a null effect)')
    ax.legend(frameon=False,ncol=2,fontsize=8)
    save(fig,'evidence_coverage')
    effects = [e for e in h1['effects'] if e['status'] == 'complete']
    if effects:
        fig,ax = plt.subplots(figsize=(12,5))
        for i,e in enumerate(effects):
            ax.scatter([i]*len(e['ratios']),e['ratios'],s=10,color='#9aafba',alpha=.8,zorder=2)
            color = '#287a70' if e['speedup'] > 1.02 else '#b14e47' if e['speedup'] < .98 else '#777777'
            ax.scatter(i,e['speedup'],s=30,color=color,zorder=3)
        ax.axhline(1,color='#222222',lw=1)
        ax.axhline(1.02,color='#888888',lw=.7,ls=':')
        ax.set_xticks(range(len(effects)),[e['workload_id'] for e in effects],rotation=90,fontsize=7)
        ax.set_ylabel('Default / frozen-winner latency')
        ax.set_title('Independent confirmation: points = 3 process rounds; large point = geometric mean')
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=label) for c,label in
                           [('#287a70','Point ratio > 1.02'),('#777777','Point ratio 0.98 to 1.02'),('#b14e47','Point ratio < 0.98')]],
                  title='Descriptive colors; no significance claim',frameon=False,fontsize=8,title_fontsize=8)
        save(fig,'h1_frozen_winner_effects')
        fig,(ax,bx) = plt.subplots(1,2,figsize=(10,3.7))
        x = sorted(e['speedup'] for e in effects)
        ax.step(x,[(i+1)/len(x) for i in range(len(x))],where='post',color='#287a70')
        ax.axvline(1,color='#555555',lw=.8)
        ax.set_xlabel('Confirmed default / frozen winner')
        ax.set_ylabel('Observed workload fraction')
        groups = [g for g in h1['N_groups'] if g['geometric_speedup'] is not None]
        bx.plot([g['n'] for g in groups],[g['geometric_speedup'] for g in groups],marker='o',color='#497a91')
        bx.set_xscale('log',base=2)
        bx.axhline(1,color='#555555',lw=.8)
        bx.set_xlabel('N (equal workload weights within N)')
        bx.set_ylabel('Geometric speedup')
        save(fig,'h1_distribution_by_N')
    counts = [c for c in h2['counts'] if c['accepted_successful']]
    if counts:
        fig,ax = plt.subplots(figsize=(12,4.5))
        for key,label in [('requested_class','Requested'),('resolved_class','Resolved'),('core','Core'),('source_compiler','Source/compiler')]:
            ax.plot(range(len(counts)),[c['classes'][key] for c in counts],marker='.',label=label)
        ax.set_xticks(range(len(counts)),[c['workload_id'] for c in counts],rotation=90,fontsize=7)
        ax.set_ylabel('Distinct classes among successful accepted candidates')
        ax.legend(ncol=4,frameon=False)
        ax.set_title('Request folding; core and source equality are different equivalence relations')
        save(fig,'h2_structure_folding')
        fig,ax = plt.subplots(figsize=(13,7))
        ws = [w['id'] for w in design['workloads']]
        cs = [c['id'] for c in design['candidates']]
        raw = {(r['workload_id'],r['candidate_id']):r for r in evidence['rows']['screen']}
        matrix = [[1 if selection_eligible(raw.get((w,c),{})) else 0 if (w,c) in raw else np.nan for w in ws] for c in cs]
        from matplotlib.colors import ListedColormap
        im = ax.imshow(matrix,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap=ListedColormap(['#bd6658','#497a91']))
        ax.set_xticks(range(len(ws)),ws,rotation=90,fontsize=7)
        ax.set_yticks(range(len(cs)),cs,fontsize=7)
        ax.set_title('Candidate eligibility: blue = eligible; red = failed/excluded; white = absent accepted block')
        save(fig,'h2_candidate_validity')
    good = [p for p in h3['pairs'] if p['ok']]
    if good:
        fig,ax = plt.subplots(figsize=(10,4))
        labels = sorted({p['workload_id'] for p in good})
        for i,wid in enumerate(labels):
            for mode,field,color,offset in [('Full','full_wall_ms','#497a91',-.12),('Describe + known choice','describe_known_choice_wall_ms','#c18b3b',.12)]:
                vals = [p[field] for p in good if p['workload_id'] == wid]
                ax.scatter([i+offset]*len(vals),vals,color=color,s=24,label=mode if i == 0 else None)
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
        ax.set_yscale('log')
        ax.set_ylabel('Context pipeline wall time (ms, log scale)')
        ax.legend(frameon=False)
        ax.set_title('Construction component only; 3 fresh-process pairs per workload')
        save(fig,'h3_pipeline_construction')
        # Display only verified pairs; exclusive stage sums never double count
        # nested parent/child spans. Residual remains explicitly separate.
        keys = {(p['workload_id'],p['round']) for p in good}
        pp = [p for p in h3['pipelines'] if (p['workload_id'],p['round']) in keys]
        stage_mass = collections.defaultdict(float)
        for p in pp:
            for stage,value in p['exclusive_stages_ms'].items():
                stage_mass[stage] += value
        top = sorted(stage_mass,key=lambda s:-stage_mass[s])[:6]
        bars = sorted({(p['workload_id'],p['mode']) for p in pp})
        fig,ax = plt.subplots(figsize=(12,5))
        bottom = np.zeros(len(bars))
        colors = ['#497a91','#c18b3b','#287a70','#956687','#7a898a','#b5685a','#cccccc']
        for stage,color in zip(top+['Other exclusive'],colors):
            vals = []
            for wid,mode in bars:
                source = [p for p in pp if p['workload_id'] == wid and p['mode'] == mode]
                vals.append(statistics.mean(p['exclusive_stages_ms'].get(stage,0) if stage != 'Other exclusive' else
                                            sum(v for k,v in p['exclusive_stages_ms'].items() if k not in top) for p in source))
            ax.bar(range(len(bars)),vals,bottom=bottom,label=stage,color=color)
            bottom += np.array(vals)
        ax.set_xticks(range(len(bars)),[wid+'\n'+('Full' if mode == 'full-search-cost' else 'Describe+choice') for wid,mode in bars],rotation=55,ha='right',fontsize=7)
        ax.set_ylabel('Mean traced exclusive time (ms; 3 processes)')
        ax.legend(frameon=False,ncol=3,fontsize=7)
        ax.set_title('Exclusive app stages; shared work and unattributed pipeline residual are reported separately')
        save(fig,'h3_exclusive_stages')
    tests = [x for x in headline['tests'] if x['speedup'] is not None]
    if tests:
        fig,ax = plt.subplots(figsize=(9,max(3,len(tests)*.45)))
        for i,row in enumerate(tests):
            color = '#287a70' if row['clear_gain'] else '#777777'
            ci = row['interval95']
            if ci:
                ax.plot(ci,[i,i],color=color,lw=1.5)
            ax.scatter(row['speedup'],i,color=color,s=30,marker='o' if row['inferential'] else 'x')
        ax.axvline(1.02,color='#888888',ls=':',label='2% null')
        ax.set_yticks(range(len(tests)),[x['workload_id']+' '+x['reference']+' / '+x['target'] for x in tests],fontsize=7)
        ax.set_xlabel('Geometric paired ratio; descriptive paired-log t 95% interval')
        ax.set_title('Frozen headline family: green requires >=5% and Holm p < .05 plus quality gates')
        save(fig,'headline_frozen_tests')
    return paths


def write_report(out, root, design, evidence, h1, h2, h3, smoke, headline, h4, inventory, figures):
    complete = h1['confirmed_workloads'] == len(design['workloads']) and evidence['coverage']['confirmation']['gate_passed']
    h1_status = '完整固定矩阵独立确认' if complete else 'incomplete：独立确认矩阵尚不完整'
    lines = ['# VkFFT 计划选择研究：离线可复现报告','',f"研究：`{design['study_id']}`；设计 SHA256：`{design['_sha256']}`。",
             '',f"H1：**{h1_status}**。H3：**{h3['status']}**。Headline：**{headline['status']}**。",
             '',f"仅读取指定数据目录 `{root}`。不搜索其他目录，不使用旧 pilot 数据，不以缺失数据替代零效果。当前独立确认 {h1['confirmed_workloads']}/{len(design['workloads'])} 个 workload，证据审计问题 {len(evidence['errors'])} 项。",
             '', '## 主要发现与证据状态','',
             f"H1：{h1['confirmed_workloads']}/{h1['expected_workloads']} 个冻结选择有完整3轮独立确认。观测集合的等workload几何比率 default/winner 为 {fmt(h1['equal_workload_geometric_speedup'])}，workload中位比率 {fmt(h1['median_workload_speedup'])}；二者不同，不能用几何均值代表每个区域均有收益。",
             '',f"下降方向：{len(h1['nominal_point_estimate_below_one'])} 个 workload 仅有名义点估计比率<1，最差名义延迟增幅 {fmt(100*h1['worst_nominal_latency_increase']) if h1['worst_nominal_latency_increase'] is not None else '—'}%；达到≥5%延迟增加的实质下降 {len(h1['latency_increase_at_least5pct'])} 个。这里没有进行下降方向的显著性检验；点估计<1不等同统计显著下降。",
             '',f"H2：结构/请求折叠可描述；原预设同ID对照中，有双方候选在两个批量端点均获得完整3轮确认的 N 对照数为 {sum(len(x['complete_two_endpoint_confirmation_N']) for x in h2['contrast_coverage'])}。缺失覆盖不构成‘未发现反转’证据。原headline状态 `{headline['status']}`，独立posthoc复核如另行执行，必须保持其单独身份与family。",
             '',f"H3：主成本仅按显式accepted pair保留 {h3['primary_pipeline_processes']}/48 个pipeline，状态 `{h3['status']}`；full26对比describe26+known-choice1只回答穷举构造组件，默认单plan是另一基线。工程smoke不替代主成本。",
             '',f"H4：`{h4['status']}`；仅在主阶段结束后评估给定离线合法候选集合的条件排序，不证明编译前合法性、单候选上线成功或端到端收益。Headline检验在其固定10轮完成前保持pending。",
             '', '## 范围与审计','',
             '固定单 GPU、CUDA FP32、2/3/5 smooth 1D C2C forward，独立输入输出、自然顺序、输入保持不变。36 个有意选择的 workload，16 个 N 组，不代表所有 FFT 使用场景。独立重复单位是进程中位数，图内采样次数不增加统计样本量。',
             '', '| 阶段 | 原始记录 | 可审计 | 已接受 | 成功/失败 | eligible | 协议状态 | protocol_valid | all_candidates_passed |','|---|---:|---:|---:|---|---:|---|---|---|']
    for stage,c in evidence['coverage'].items():
        lines.append(f"| {stage} | {c['raw_records']} | {c['audited_records']} | {c['accepted_records']} | {c['successful_records']}/{c['unsuccessful_records']} | {c['eligible_records']} | {c['state']} | {c['protocol_valid']} | {c['all_candidates_passed']} |")
    lines += ['', '| 阶段 | 唯一已尝试项 | 当前阶段计划进程数 |','|---|---:|---:|']
    for stage,c in evidence['coverage'].items():
        lines.append(f"| {stage} | {c['unique_attempted_cells']} | {c['expected_processes'] if c['expected_processes'] is not None else '尚无stage status'} |")
    failures = [r for r in evidence['ledger'] if r['status'] != 'ok']
    failure_appendix = []
    if failures:
        failure_appendix += ['', '## 附录：保留的失败尝试（不含时延估计）','', '| 阶段/输入 | Workload | 候选 | 失败边界 | 观察错误 |','|---|---|---|---|---|']
        for r in failures:
            failure_appendix.append(f"| {r['phase']} | {r['workload_id']} | {r['candidate_id']} | {r['failure_phase']} | {r['error']} |")
    lines += ['', '接受的 screen/confirmation/headline 来自 quality.ok 的完整块及其 record_hashes；重新核验原始记录 SHA、guard ID/run ID、before/after 采样和 quiet gate。失败或未接受尝试保留在 attempt_ledger.json。有限轮询仍有采样盲区。',
              '', 'protocol_valid / protocol completed 与 all_candidates_passed 分开：允许的候选局部失效仍是失败记录，不重新标为数值通过。S1 要求完整36×26请求尝试覆盖；合法/eligible子集可少于请求数，且任何固定语义probe失败使精确workload/candidate不可用于选择或学习，即便其后随机输入执行成功。不得外推整列/其他batch无效，也不以无效候选补零时延。诊断进程仅用于工程判定，不进入性能、成本或学习标签。',
              '', '### Cost smoke独立工程复审','',
              f"状态：`{smoke['status']}`；{smoke['processes']}/{smoke['expected_processes']} 个进程；{smoke['describe_apps']} 个 describe app、{smoke['full_and_materialize_apps']} 个 full/materialize app。describe编译/模块加载/kernel launch总计分别为 {smoke['describe_counter_totals']['nvrtc_compile']}/{smoke['describe_counter_totals']['module_load']}/{smoke['describe_counter_totals']['driver_kernel_launch']}。实际同请求descriptor/source/compiler-input对照 {smoke['actual_descriptor_source_compiler_comparisons']} 项；逐app counters与trace复核见 [cost_smoke_audit.json](cost_smoke_audit.json)。",
              '', '该smoke仅为两个预定workload的一轮工程可行性验收，不是H3主成本估计；不把其初始化时延作为S3八workload×三轮×双pipeline的替代，也不提供搜索收益结论。',
              '', '## H1：冻结选择的独立效果','',
              'S1 仅冻结 winner；主效果使用其后 3 轮 fresh-process 配对 log 比率的几何均值。winner=default 时效果按定义为 1，不产生 self 比较显著性。每 workload 等权；未完成时汇总只描述已观测子集。',
              '',f"观测集合几何均值：{fmt(h1['equal_workload_geometric_speedup'])}。名义点估计<1的 workload：{', '.join(h1['nominal_point_estimate_below_one']) or '无 / 尚无数据'}；这不是显著下降名单，≥5%延迟增加的实质下降数量为 {len(h1['latency_increase_at_least5pct'])}。",
              '',f"Screen 中 default 距已测最小值 ≤2%：{h1['near_best2']}/{len(h1['screen'])}；≤5%：{h1['near_best5']}/{len(h1['screen'])}。这些已选最小值是乐观的 tested-set 诊断，不替代独立效果。",
              '', '| Workload | 冻结 winner | 3轮几何比率 | 各轮比率 / 缺失状态 |','|---|---|---:|---|']
    for e in h1['effects']:
        lines.append(f"| {e['workload_id']} | {e.get('winner','—')} | {fmt(e.get('speedup'))} | {', '.join(fmt(x) for x in e.get('ratios',[])) or e['status']} |")
    if h1['screen']:
        lines += ['', '### S1 screen headroom（待独立确认）','',
                  '这里只报告已测合法候选集合内 default/最快者 的一次screen比率；该最小值经过选择，不能称为独立确认收益。eligible数排除default_replica；完整请求覆盖仍包含它及所有失败候选。',
                  '', '| Workload | Eligible候选数（不含replica） | Screen最快ID | Default / screen最小值 | Default在2%内 |','|---|---:|---|---:|---|']
        for s in h1['screen']:
            lines.append(f"| {s['workload_id']} | {s['legal_candidates']} | {s['fastest_tested']} | {fmt(s['default_over_tested_min'])} | {s['default_within2pct']} |")
    lines += ['', '## H2：结构折叠与同 ID 批量对照','',
              'requested、resolved、core 投影、完整 canonical、source/compiler-input 是不同等价关系。完整 descriptor、生成源码及 compiler input 指纹见 h2.json；物理寄存器/local memory 与逻辑 registerBoost 分开保存。结构变化与延迟关联不能单独证明某个资源是增益原因。',
              '',f"已观察到 {len(h2['structures'])} 个成功候选结构；固定同 ID workload对照 {len(h2['contrasts'])} 项。双方ID在两个批量端点均有完整确认的N对照 {sum(len(x['complete_two_endpoint_confirmation_N']) for x in h2['contrast_coverage'])} 项；缺失确认保持 missing，不换ID凑crossover。完整覆盖内探索性 reversal 记录数 {len(h2['exploratory_crossovers'])}，不能把缺失覆盖解释为‘未发现反转’。",
              '',f"独立 headline 两个反向检验均通过：{'是' if headline.get('confirmed_crossover') else '否 / 尚未完成'}。未满足时只报告探索性反转。",
              '', '## Headline：固定10轮检验','',
              f"冻结 directional family 共 {headline['family_size']} 项；所有冻结方向均保留，缺失/失败方向以 p=1 进入 Holm family 计数。普通 paired-log Student-t 95% 区间仅作描述；单侧检验 H0 为 mean(log(reference/target)) ≤ log(1.02)。clear gain 要求几何比率≥1.05、Holm p<0.05 且 guard/负对照/gate有效。",
              '', 'paired-log range ≤16×float64 epsilon×max(1,maxabs(logratio)) 时标记 noninferential/quantization-limited：不生成 t、p 或区间推断；仅为校正计数保留 p=1。default_replica 是质量诊断，不是收益假设。',
              '', '| Workload / 方向 | 比率 | 描述95%区间 | Holm p | clear gain | 状态 |','|---|---:|---|---:|---|---|']
    for x in headline['tests']:
        lines.append(f"| {x['workload_id']} {x['reference']} / {x['target']} | {fmt(x['speedup'])} | {', '.join(fmt(v) for v in x['interval95']) if x['interval95'] else '不可推断'} | {fmt(x['holm_p'])} | {x['clear_gain']} | {x.get('inference_reason') or 'inferential'} |")
    lines += ['', '## H3：实际 describe/full 同构与构造成本','',
              f"主成本固定 8 workload×3 paired rounds×2 pipelines，实际保留 {h3['primary_pipeline_processes']}/{h3['expected_primary_pipeline_processes']} 个pipeline进程，共 {h3['passed_pairs']}/{h3['expected_pairs']} 对通过实际 descriptor/source/compiler-input 等价与编译/加载/launch计数检查。缺失或失败保留，H3 不据此补数据；四个工程smoke进程不在该计数内。",
              '', '每 pipeline 是一个新的进程/context；其内部逐一构造和清理所有26候选。describe-all 后完整物化离线已知 winner 是 known-choice 构造组件下界，不是在线选择器或端到端搜索收益。describe 仍包含配置、规划、host codegen、LUT/资源分配与传输；不称为免费。',
              '', '成本基线边界：full26 versus describe26+known-choice1 仅对比穷举候选构造组件；当前默认行为只构造一个默认plan。因此两pipeline的比率不能表述为“默认规划提速”。单默认plan、候选全构造与describe-all是不同工作量。若使用S2 default initialize_ms与S3逐app initialize_ms总和，只能作同组件、非配对诊断；不得用完整pipeline wall直接减单次default initialize_ms，也不能据此声称端到端ML调优收益。',
              '', '只汇总 exclusive span，禁止把所有嵌套 inclusive 相加。h3.json 保存每 app exclusive 阶段、context bootstrap、共享 buffer/stream setup/cleanup、完整 pipeline wall 及 wall−traced roots 残差；残差包含未覆盖工作，不强行归零。first/later NVRTC 仅指同 context 中的位置，不宣称 driver/compiler/OS cache cold。',
              '', '| Workload | Round | Full ms | Describe+known choice ms | 构造比率 | 审核 |','|---|---:|---:|---:|---:|---|']
    c = evidence['coverage']['cost']
    cost_note = ['', '### 成本原始账本与显式配对选择','',
                 f"原始成本记录 {c['raw_records']} 条；只按 accepted_pairs.json 明确列出的hash保留 {c['accepted_records']} 条、{c['accepted_pairs']} 个完整pair。选择状态：`{c['selection_state']}`。不按status成功、最新attempt或单条可用性补齐配对。原始记录均保留在尝试账本。"]
    if c['A6_recovery_admitted']:
        cost_note += ['', f"A6已核验：raw账本中的 {c['engineering_only_raw_records']} 条未入估计，包含原中断pair的orphan；另有 {c['quarantine_records_outside_main_ledger']} 个quarantine进程在主JSONL之外。当前physical进程计数 {c['physical_processes_estimate']}（完整恢复目标为raw49 / accepted48 / physical50）。旧quarantine不追认为合法，不从中提取成本估计。",
                      '', f"原aborted guard session中仅 {len(c['old_closed_interval_session_exception_record_hashes'])} 个已关闭且干净的精确冻结hash由A6特例保留；这不是把整个旧session标为正常。除此之外，新session都必须完成并通过guard。A6 admission SHA：`{c['admission_sha256']}`。"]
    table_start = lines.index('| Workload | Round | Full ms | Describe+known choice ms | 构造比率 | 审核 |')
    lines[table_start:table_start] = cost_note + ['']
    for p in h3['pairs']:
        lines.append(f"| {p['workload_id']} | {p['round']} | {fmt(p.get('full_wall_ms'))} | {fmt(p.get('describe_known_choice_wall_ms'))} | {fmt(p.get('construction_ratio'))} | {'pass' if p['ok'] else '; '.join(p['errors'])} |")
    lines += ['', '### Exclusive 阶段总账','',
              '下表只汇总通过配对同构审计的进程；compile/load 范围包含 NVRTC create/compile/destroy、binary extract、module load、function lookup/attribute setup。比例分别以全部 app exclusive 和完整 pipeline wall 为分母。',
              '', '| Pipeline | 进程数 | Traced exclusive ms | Compile/load ms | 占traced | 占pipeline |','|---|---:|---:|---:|---:|---:|']
    for s in h3['stage_summary']:
        lines.append(f"| {s['mode']} | {s['processes']} | {fmt(s['trace_exclusive_ms'])} | {fmt(s['compilation_loading_exclusive_ms'])} | {fmt(s['compilation_fraction_of_traced'])} | {fmt(s['compilation_fraction_of_pipeline'])} |")
    stages = sorted({k for s in h3['stage_summary'] for k in s['exclusive_stages_sum_ms']})
    lines += ['', '| Exclusive阶段 | Full总ms | Describe+choice总ms |','|---|---:|---:|']
    for stage in stages:
        values = [s['exclusive_stages_sum_ms'].get(stage,0.) for s in h3['stage_summary']]
        lines.append(f"| {stage} | {fmt(values[0])} | {fmt(values[1])} |")
    lines += ['', '### 已知单候选组件的摊销诊断','',
              'setup policy：完整 all26 pipeline 同context冻结顺序中，winner与default的 initialize_ms 差，取3轮均值；包含trace/hash和初始化同步，排除cleanup、context bootstrap、完整搜索benchmark及model scoring。执行差使用独立3轮 default_ms−winner_ms 的均值。仅当差为正且setup齐全时按 ceil(max(0,setup差)/执行差) 给出组件执行次数，否则不生成有限值；它不是端到端调优break-even。',
              '', '| Workload | setup差ms | exec差ms | 组件摊销执行次数 |','|---|---:|---:|---:|']
    for a in h3['amortization']:
        lines.append(f"| {a['workload_id']} | {fmt(a['setup_delta_ms'])} | {fmt(a['execution_delta_ms'])} | {a['break_even_executions'] if a['break_even_executions'] is not None else '不可估/无有限值'} |")
    lines += ['', '### Describe枚举相对默认单plan：非配对组件诊断','',
              '另列新增枚举工作：取S3每轮26个describe加1个known-choice物化 app的 initialize_ms 总和，再取3轮均值，与S2默认单plan的3个独立fresh-process initialize_ms均值相减。它是跨阶段非配对的同组件差，不是完整pipeline wall减单次默认初始化。只有组件均完整、独立执行差>0才计算有限摊销；组件差≤0可记0并标记“组件不增”。排除context、共享setup/cleanup、scoring、验证及fallback成本；不能称默认端到端规划提速或ML上线收益。',
              '', '| Workload | S3枚举+物化init均值ms | S2默认单plan init均值ms | 额外init组件ms | 独立exec差ms | 组件摊销次数 |','|---|---:|---:|---:|---:|---:|']
    for a in h3['nonpaired_enumeration_vs_single_default_component_diagnostic']:
        lines.append(f"| {a['workload_id']} | {fmt(a['S3_mean_summed_initialize_ms'])} | {fmt(a['S2_mean_default_initialize_ms'])} | {fmt(a['additional_initialize_component_ms'])} | {fmt(a['independent_execution_delta_ms'])} | {a['component_break_even_executions'] if a['component_break_even_executions'] is not None else '不可估/无有限值'} |")
    h4_rows = []
    if h4.get('status') == 'completed_auxiliary':
        h4_rows = ['', '| H4条件策略 | 可用workload数 | unavailable数 | 可用集合几何speedup |', '|---|---:|---:|---:|']
        for outcome in h4.get('outcomes',[]):
            m = outcome['metrics']
            h4_rows.append(f"| {outcome['target']} / {outcome['ablation']} | {m.get('workloads',0)} | {m.get('unavailable',0)} | {fmt(m.get('equal_workload_geometric_speedup'))} |")
        for name,baseline in h4.get('baselines',{}).items():
            m = baseline['metrics']
            h4_rows.append(f"| {name} | {m.get('workloads',0)} | {m.get('unavailable',0)} | {fmt(m.get('equal_workload_geometric_speedup'))} |")
    lines += ['', '## H4、硬件计数器与限制','',
              f"H4 状态：`{h4['status']}`。辅助学习仅在固定主阶段完成后，按冻结4-fold exact-N分组、固定 ExtraTrees 参数和预定义消融输出；至少12个完整 N 组，不插补无效候选，不将默认延迟、timing、编译资源、有效性标记或 source ID 作为输入特征。完整覆盖、训练-only global baseline、默认/oracle与top2/4 shortlist诊断见 [h4.json](h4.json)。",
              '', 'H4 评估给定离线验证通过候选集合后的条件选择能力。每个留出 N 的候选集也先使用该 workload 的执行、数值和语义预检结果过滤；模型只在剩余候选中排序。有效性标记不进入输入特征，但筛选本身使用了留出组事后已知的有效性信息，因此这是对合法集合已知这一条件下的性能排序评估。它不证明模型能在编译前识别无效候选，也不证明上线时可以只编译一个候选且避免执行失败。',
              '', '本轮已有 describe/compile 成功但执行失败的配置；候选生成、运行正确性验证和性能评分是三个不同问题。错误候选发现、验证、fallback 及其成本未在 H4 评估。',
              '', 'training-only global 在留出 workload 无效时保留 unavailable，模型则在预先过滤的合法集合内选择。这不构成模型避免错误或合法性更好的证据。各 aggregate 必须并列可用覆盖与 unavailable 数；分母不同的 available 均值不能直接作为模型性能优势，unavailable 不按零性能计。默认、screen oracle、所有模型及 top2/4 都沿用“给定合法集合”条件；topk 仍是假设 shortlist，不含验证和 fallback 成本。',
              *h4_rows,
              '', '可选 ncu 计数器未执行：严格 profiler 进程链保护尚未验收。任何 profiler/replay 时间均不得合入主延迟；未观察 counters 不能支持 DRAM/L2/occupancy 因果解释。',
              '', '保留 default near-best 区域、回归、请求折叠、无效候选、未确认 crossover、非显著检验、昂贵 describe 阶段和未完成 gate。此单GPU固定矩阵不支持跨设备或全部FFT泛化，缺失实验不是等价证据。',
              '', '## 原始证据与复现','',
              '`analyze_study.py --input <本次数据目录> --output <报告目录>`；使用 scipy 的 Student-t 和标准 matplotlib 生成独立 PNG/SVG。`reproduction.json` 保存解释器/依赖版本、输入目录及脚本 SHA；`raw_inventory.json` 保存所有被读取证据的相对路径、字节数和 SHA256。',
              '', '[机器可读摘要](summary.json) · [完整尝试账本](attempt_ledger.json) · [审计问题](audit_errors.json) · [原始证据索引](raw_inventory.json)', '']
    for f in figures:
        if f.suffix == '.png':
            lines += [f'![{f.stem}](figures/{f.name})','']
    lines += failure_appendix
    lines += ['### 原始文件链接','']
    for item in inventory:
        path = root/item['path']
        lines.append(f"- [{item['path']}]({path.resolve().as_posix()}) (`{item['sha256']}`, {item['bytes']} bytes)")
    (out/'研究报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def fmt(value):
    return '—' if value is None else f'{value:.5g}'


def run(root, out, design_path, learning=False):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    design = load_design(design_path)
    design['_sha256'] = digest(design_path)
    evidence = read_evidence(root,design)
    selection = optional_json(root/'screen_selection.json')
    manifest = optional_json(root/'headline_selection.json')
    screen_gate = evidence['gates']['screen'] or {}
    if selection and (selection.get('study_id') != design['study_id'] or selection.get('design_sha256') != design['_sha256']
                      or not screen_gate.get('passed') or screen_gate.get('selection_sha256') != digest(root/'screen_selection.json')):
        evidence['errors'].append(dict(error='selection_identity_mismatch'))
        selection = None
    if manifest:
        a3 = design_path.with_name('research_addendum_A3.json')
        a3_freeze = optional_json(design_path.with_name('research_addendum_A3_freeze.json')) or {}
        confirmation_gate = evidence['gates']['confirmation'] or {}
        linked = (manifest.get('study_id') == design['study_id'] and
                  (root/'screen_selection.json').is_file() and manifest.get('selection_sha256') == digest(root/'screen_selection.json') and
                  (root/'confirmation_summary.json').is_file() and manifest.get('confirmation_sha256') == digest(root/'confirmation_summary.json') and
                  manifest.get('rounds') == 10 and
                  confirmation_gate.get('headline_selection_sha256') == digest(root/'headline_selection.json') and
                  a3.is_file() and manifest.get('interpretation_addendum_sha256') == digest(a3) and
                  a3_freeze.get('addendum_sha256') == digest(a3) and a3_freeze.get('original_design_sha256') == design['_sha256'])
        if not linked:
            evidence['errors'].append(dict(error='headline_manifest_identity_or_source_link_mismatch'))
            # Retain all family directions but disable inference/claims.
            evidence['gates']['headline'] = None
    h1 = h1_analysis(design,evidence['rows'],selection)
    h2 = h2_analysis(design,evidence['rows'])
    h3 = h3_analysis(design,evidence['rows']['cost'],selection,evidence['gates']['cost'],evidence['rows']['confirmation'])
    smoke = cost_smoke_analysis(design,evidence['rows']['cost_smoke'])
    headline = headline_analysis(evidence['rows']['headline'],manifest,evidence['gates']['headline'],design)
    h4 = dict(status='not_requested_auxiliary',legacy_data_used=False)
    if learning:
        finished = all((optional_json(root/stage/'status.json') or {}).get('state') in ('complete','complete_with_limitations')
                       for stage in ('screen','confirmation','cost','headline'))
        if not finished:
            h4 = dict(status='not_run_primary_sequence_pending',legacy_data_used=False)
        else:
            from learning_check import run_learning
            verified = evidence['identity_ok'] and not any(e.get('stage') == 'screen' or 'screen' in e.get('file','') for e in evidence['errors'])
            h4 = run_learning(design,evidence['rows']['screen'],evidence_verified=verified,
                              frozen_global_candidate=(selection or {}).get('global_candidate'))
    previous_h4 = optional_json(out/'h4.json')
    if previous_h4 and previous_h4.get('status') == 'completed_auxiliary':
        if learning and previous_h4 != h4:
            raise ValueError('completed H4 differs from retained first analysis; preserve the first artifact')
        if not learning:
            h4 = previous_h4
    summary = dict(study_id=design['study_id'],design_sha256=design['_sha256'],legacy_data_used=False,
                   evidence_root=str(root),coverage=evidence['coverage'],audit_error_count=len(evidence['errors']),
                   H1='complete' if h1['confirmed_workloads'] == len(design['workloads']) and evidence['coverage']['confirmation']['gate_passed'] else 'incomplete',
                   H2='observed_fixed_contrasts' if h2['structures'] else 'incomplete',H3=h3['status'],
                   headline=headline['status'],H4=h4['status'],profiler='not_performed',
                   equal_workload_speedup=h1['equal_workload_geometric_speedup'],aggregate_scope=h1['aggregate_scope'])
    for name,value in [('summary',summary),('attempt_ledger',evidence['ledger']),('audit_errors',evidence['errors']),('h1',h1),('h2',h2),('h3',h3),('cost_smoke_audit',smoke),('headline',headline),('h4',h4)]:
        write_json(out/(name+'.json'),value)
    write_json(out/'cost_recovery_report.json',evidence['coverage']['cost'])
    write_json(out/'h4_interpretation.json',dict(evaluation_executed=h4.get('status') == 'completed_auxiliary',
               conditional_on_offline_validated_candidate_set=True,heldout_validity_used_for_filtering=True,
               validity_is_model_input_feature=False,invalid_candidate_detection_evaluated=False,
               validation_and_fallback_costs_measured=False,global_invalid_outcome='unavailable_not_zero_or_model_win',
               aggregate_comparison_requires_parallel_coverage=True,
               interpretation='Conditional ranking within the offline validated heldout candidate set; not precompile validity prediction or a demonstrated single-compile online policy.'))
    files = [root/'manifest.json',root/'research_design.json',root/'research_freeze.json',root/'screen_selection.json',root/'confirmation_summary.json',root/'headline_selection.json',root/'graph_repeats.json']
    for stage in STAGES:
        files += list((root/stage).glob('*.json')) + list((root/stage).glob('*.jsonl')) + list((root/stage/'blocks').glob('*.json'))
    files += list((root/'guards').glob('*.jsonl'))
    files += list(root.glob('research_addendum_*.json'))
    files += [root/'cost_recovery_admission.json'] + list((root/'cost/engineering_archive').glob('*.json'))
    inventory = [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files)) if p.is_file()]
    write_json(out/'raw_inventory.json',inventory)
    figures = make_figures(out,design,evidence,h1,h2,h3,headline)
    import scipy, matplotlib
    dependencies = ['analyze_study.py','report_helpers.py','study_common.py','guard_audit.py','trace_audit.py','cost_audit.py','cost_recovery.py','learning_check.py']
    write_json(out/'reproduction.json',dict(command=[sys.executable,str(Path(__file__).resolve()),'--input',str(root),'--output',str(out),'--design',str(design_path.resolve())]+(['--learning'] if learning else []),input=str(root),output=str(out),
               python=platform.python_version(),python_executable=sys.executable,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
               design_source=str(design_path.resolve()),design_sha256=digest(design_path),
               scripts={name:digest(Path(__file__).with_name(name)) for name in dependencies if Path(__file__).with_name(name).is_file()}))
    write_report(out,root,design,evidence,h1,h2,h3,smoke,headline,h4,inventory,figures)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=Path(__file__).with_name('research_design.json'))
    parser.add_argument('--learning',action='store_true',help='Run fixed auxiliary H4 only after all primary stage statuses finish')
    args = parser.parse_args()
    print(json.dumps(run(args.input,args.output,args.design,args.learning),indent=2))
