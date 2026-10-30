#!/usr/bin/env python3
"""Fail-closed, GPU-free analysis of exactly one independent cloud cohort."""
import argparse
import collections
import json
import math
from pathlib import Path
import random
import statistics

from cloud_reproduction import CloudController, build_schedule, cost_classification, json_file, validate_design
from cost_audit import audit_pair
from guard_audit import completed_session, read_guard, resolve_log
from report_helpers import headline_stat, holm
from study_common import block_quality, candidate_config, canonical, execution_errors, input_seed, read_jsonl, sha, write_json
from trace_audit import audit_trace, plan_fingerprints


def audit_execution_record(root,stage,design,row,calibration_rows):
    if stage=='cost': return []
    catalog = {w['id']:w for w in design['calibration']['workloads']}
    for section in ('equivalence','semantic_probes'):
        for w in design['preflight'][section]['cases' if section=='equivalence' else 'groups']:
            catalog[w['id']] = w
    wid = row['workload_id']
    original = catalog[wid]
    spec = row['workload']
    cid = row['candidate_id']
    candidate = next(c for c in design['candidates'] if c['id']==cid)
    config = candidate_config(candidate)
    if row.get('config') != config: return ['frozen_candidate_config_mismatch']
    if stage in ('equivalence','semantics'):
        repeats,samples = 16,1
    elif stage=='calibration':
        repeats = 1
        by = {(r['workload_id'],r['round']):r for r in calibration_rows}
        for k in range(row['round']):
            previous = by[wid,k]
            raw = previous['result']['raw_interval_ms_median']
            if not isinstance(raw,(int,float)) or not math.isfinite(raw) or raw<=0:
                return ['invalid_prior_calibration']
            cfg = design['measurement']['repeats']
            if raw >= cfg['min_raw_ms'] or repeats==cfg['cap']:
                return ['calibration_continued_after_stop']
            repeats = min(cfg['cap'],max(repeats+1,math.ceil(1.1*cfg['min_raw_ms']*repeats/raw)))
        samples = design['measurement']['screen_samples']
    else:
        repeats = json_file(root/'graph_repeats.json')[wid]['graph_repeats']
        samples = design['measurement']['confirmation_samples']
    if any(spec.get(k)!=original[k] for k in ('id','n','batch','input_bytes')) or spec.get('graph_repeats')!=repeats or spec.get('samples')!=samples:
        return ['frozen_workload_or_measurement_mismatch']
    result = row['result']
    if stage=='equivalence':
        expected_pattern, warmup, warmup_ms = 'random',0,0
        if spec.get('binary_role')!=row['phase'].split('/',1)[1]: return ['binary_role_phase_mismatch']
    elif stage=='semantics':
        expected_pattern, warmup, warmup_ms = row['phase'].split('/',1)[1],0,0
    else:
        expected_pattern, warmup, warmup_ms = 'random',5,design['measurement']['warmup_ms']
    if (result.get('pattern')!=expected_pattern or spec.get('warmup',5)!=warmup or
        spec.get('warmup_ms',design['measurement']['warmup_ms'])!=warmup_ms or
        (stage in ('equivalence','semantics') and spec.get('pattern')!=expected_pattern)):
        return ['frozen_pattern_or_warmup_mismatch']
    errors = execution_errors(result,dict(original,graph_repeats=repeats),samples,instrumented=not (stage=='equivalence' and row['phase']=='equivalence/plain'))
    names = dict(aim_threads='aimThreads',register_boost='registerBoost',use_lut='useLUT',register_boost_nonpow2='registerBoostNonPow2',
                 coalesced_memory='coalescedMemory',use_lut_4step='useLUT_4step',swap_to_2stage='swapTo2Stage4Step',swap_to_3stage='swapTo3Stage4Step')
    expected = {'makeForwardPlanOnly':1}
    expected.update({names[k]:v for k,v in config.items()})
    if result.get('requested_config')!=expected: errors.append('requested_configuration_mismatch')
    if result.get('seed')!=row.get('input_seed'): errors.append('result_seed_mismatch')
    return errors


def stage_rows(root, stage, design, manifest_sha):
    base = root/stage
    gate_path = base/'gate.json'
    rows = read_jsonl(base/'measurements.jsonl')
    if not gate_path.exists():
        return rows, ['stage_gate_missing']
    gate = json_file(gate_path)
    errors = []
    if gate.get('passed') is not True or gate.get('record_count') != len(rows) or not rows:
        errors.append('stage_gate_or_count_invalid')
    sessions = {(s['guard_id'],s['guard_log']) for s in gate.get('guard_sessions',[])}
    if not sessions or sessions != {(r.get('guard_id'),r.get('guard_log')) for r in rows}:
        errors.append('stage_sessions_invalid')
    cache = {}
    for guard_id, log in sessions:
        try:
            path = resolve_log({'guard_log':log},root)
            if not completed_session(path,guard_id): errors.append('guard_session_incomplete')
            cache[(guard_id,log)] = [e for e in read_guard(path) if e.get('guard_id') == guard_id]
            events = cache[(guard_id,log)]
            starts = [e for e in events if e.get('event')=='guard_start']
            if len(starts)!=1 or starts[0].get('idle_seconds',0)<30 or not any(e.get('event')=='idle_complete' for e in events):
                errors.append('guard_start_or_quiet_invalid')
        except (ValueError,OSError):
            errors.append('guard_log_missing')
    for r in rows:
        digest = sha(canonical({k:v for k,v in r.items() if k != 'record_sha256'}))
        if digest != r.get('record_sha256') or r.get('study_id') != design['study_id'] or r.get('manifest_sha256') != manifest_sha or r.get('status') != 'ok' or r.get('guard_clean') is not True or r.get('guard_gpu_uuid') != design['scope']['device_uuid']:
            errors.append('row_identity_status_or_hash_failed')
            continue
        phase = r.get('phase')
        valid_phase = (isinstance(phase,str) and (phase in ('equivalence/plain','equivalence/instrumented') if stage=='equivalence' else
                       phase in ('semantics/zero','semantics/impulse','semantics/constant','semantics/quarter-tone') if stage=='semantics' else phase==stage))
        expected_seed = input_seed(design,'equivalence' if stage=='equivalence' else phase,r.get('workload_id',''),r.get('round',-1)) if valid_phase else None
        if expected_seed is None or r.get('input_seed') != expected_seed:
            errors.append('row_frozen_seed_mismatch')
        try:
            protocol = audit_execution_record(root,stage,design,r,rows)
            if protocol: errors.extend('execution_protocol:'+x for x in protocol)
        except (OSError,KeyError,TypeError,ValueError,StopIteration) as exc:
            errors.append('execution_protocol_unavailable:'+type(exc).__name__)
        events = cache.get((r.get('guard_id'),r.get('guard_log')),[])
        run_id = r.get('guard_run_id')
        starts = [e for e in events if e.get('event')=='run_one_start' and (e.get('active_run') or {}).get('guard_run_id')==run_id]
        ends = [e for e in events if e.get('event')=='run_one_end' and e.get('guard_run_id')==run_id and e.get('clean') is True]
        samples = [e for e in events if e.get('event')=='sample' and (e.get('active_run') or {}).get('guard_run_id')==run_id]
        if not run_id or len(starts)!=1 or len(ends)!=1 or not {'before','after'} <= {e.get('boundary') for e in samples} or any(e.get('ok') is not True or e.get('foreign_compute_pids') or e.get('violation_reasons') or e.get('monitor_error') is not None or e.get('gpu_uuid') != design['scope']['device_uuid'] or e.get('gpu',{}).get('memory_used_mib',float('inf')) > 2048 for e in samples):
            errors.append('row_guard_failed')
    return rows, sorted(set(errors))


def execution_blocks(root, design, rows):
    by_hash = {r['record_sha256']:r for r in rows}
    if len(by_hash) != len(rows):
        raise ValueError('duplicate_execution_record')
    accepted = []
    expected = build_schedule(design)
    for b in expected:
        name = f"execution__{b['workload_id']}__r{b['round']}.json"
        meta = json_file(root/'execution'/'blocks'/name)
        hashes = meta['record_hashes']
        if len(set(hashes)) != len(hashes) or not all(h in by_hash for h in hashes):
            raise ValueError('execution_block_hash_invalid')
        group = [by_hash[h] for h in hashes]
        if (meta['candidate_ids'] != b['candidate_ids'] or [r['candidate_id'] for r in group] != b['candidate_ids'] or
            any(r['phase']!='execution' or r['workload_id']!=b['workload_id'] or r['round']!=b['round'] or r.get('block_attempt',0)!=0 for r in group) or
            len({r['input_seed'] for r in group}) != 1 or not block_quality(group)['ok']):
            raise ValueError('execution_block_invalid')
        accepted.extend(group)
    if len(accepted) != 199 or set(r['record_sha256'] for r in accepted) != set(by_hash):
        raise ValueError('execution_coverage_invalid')
    return accepted


def paired(rows, wid, reference, target, rounds):
    by = collections.defaultdict(list)
    for r in rows:
        if r['workload_id'] == wid:
            by[r['round'],r['candidate_id']].append(r)
    ratios, hashes = [], []
    for k in range(rounds):
        a,b = by[k,reference],by[k,target]
        if len(a)!=1 or len(b)!=1 or a[0]['input_seed']!=b[0]['input_seed']:
            return None
        x,y = a[0]['result']['exec_ms_median'],b[0]['result']['exec_ms_median']
        if not all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in (x,y)):
            return None
        ratios.append(x/y)
        hashes.append([a[0]['record_sha256'],b[0]['record_sha256']])
    return dict(ratios=ratios,record_hashes=hashes,speedup=math.exp(statistics.mean(map(math.log,ratios))))


def analyze_cost(root, design, rows, execution):
    by = collections.defaultdict(list)
    for row in rows:
        by[row['workload_id'],row['round']].append(row)
    results, amortization, pipeline_details = [], [], []
    expected_ids = {c['id'] for c in design['candidates']}
    for wid in design['cost_profile']['workloads']:
        choice = design['cost_profile']['selected_candidates'][wid]
        setup_deltas, enumeration_sums = [], []
        for k in range(3):
            group = by[wid,k]
            modes = {r['candidate_id']:r for r in group}
            if len(group)!=2 or set(modes)!={'full-search-cost','describe-select-cost'}:
                raise ValueError('incomplete_cost_pair')
            full, desc = modes['full-search-cost']['result'], modes['describe-select-cost']['result']
            order = full.get('candidate_order_ids',[])
            expected_order = sorted(expected_ids)
            random.Random(input_seed(design,'cost/candidate_order',wid,k)).shuffle(expected_order)
            if order != expected_order:
                raise ValueError('invalid_cost_candidate_coverage')
            quality = audit_pair(full,desc,order,choice)
            classification = cost_classification(quality)
            if classification == 'fatal':
                raise ValueError('cost_protocol_invalid')
            stored = json_file(root/'cost'/'pairs'/f'{wid}_r{k}.json')
            expected_modes = ['full-search-cost','describe-select-cost'] if k%2==0 else ['describe-select-cost','full-search-cost']
            if (stored.get('classification') != classification or stored.get('candidate_ids')!=expected_modes or
                stored.get('record_hashes') != [modes[x]['record_sha256'] for x in expected_modes]):
                raise ValueError('cost_pair_manifest_invalid')
            fw,dw = full.get('context_pipeline_wall_ms'),desc.get('context_pipeline_wall_ms')
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in (fw,dw)):
                raise ValueError('invalid_cost_pipeline_wall')
            results.append(dict(workload_id=wid,round=k,classification=classification,quality=quality,
                                full_wall_ms=fw,describe_known_choice_wall_ms=dw,
                                construction_ratio=fw/dw if classification=='equivalent' and isinstance(fw,(int,float)) and isinstance(dw,(int,float)) and dw>0 else None,
                                full_unique_core_classes=len({plan_fingerprints(a['result'])['core'] for a in full['apps']}),
                                full_unique_source_compiler_classes=len({plan_fingerprints(a['result'])['source_compiler'] for a in full['apps']}),
                                alias_scope='Counts are observed full constructions; any deduplication savings are hypothetical.'))
            for mode, row in modes.items():
                result = row['result']
                totals = collections.defaultdict(float)
                roots, apps = 0., []
                for app in result['apps']:
                    trace = audit_trace(app['result']['init_trace'])
                    if not trace['ok']: raise ValueError('cost_trace_invalid')
                    roots += trace['root_inclusive_ms']
                    for name,value in trace['stage_totals'].items(): totals[name] += value['exclusive_ms']
                    apps.append(dict(candidate_id=app['candidate_id'],role=app['role'],sequence_index=app['sequence_index'],
                                     first_nvrtc_app_in_context=app.get('first_nvrtc_app_in_context'),
                                     nvrtc_invocations_before_app=app.get('nvrtc_invocations_before_app'),
                                     initialize_ms=app['result'].get('initialize_ms'), counters=app['result']['init_trace']['counters']))
                wall = result.get('context_pipeline_wall_ms')
                pipeline_details.append(dict(workload_id=wid,round=k,mode=mode,record_sha256=row['record_sha256'],
                                             context_pipeline_wall_ms=wall,context_bootstrap_ms=result.get('context_bootstrap_ms'),
                                             external_process_wall_ms=1000*row['wall_seconds'] if isinstance(row.get('wall_seconds'),(int,float)) else None,
                                             cpp_total_process_wall_ms=result.get('total_process_wall_ms'),
                                             wall_scope='External wall covers process launch through exit; C++ main wall excludes OS loader and CUDA context teardown; pipeline wall covers same-context pipeline.',
                                             shared_setup_ms=result.get('shared_buffer_stream_setup_ms'),shared_cleanup_ms=result.get('shared_buffer_stream_cleanup_ms'),
                                             exclusive_stages_ms=dict(totals),traced_roots_ms=roots,
                                             wall_minus_traced_roots_ms=wall-roots if isinstance(wall,(int,float)) else None,apps=apps))
            # Inequivalent pairs are retained as negative evidence, not a validated policy cost.
            if classification=='equivalent':
                apps = {a['candidate_id']:a['result'] for a in full['apps']}
                sv,dv = apps[choice].get('initialize_ms'),apps['default'].get('initialize_ms')
                enum = [a['result'].get('initialize_ms') for a in desc['apps']]
                if all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in (sv,dv)):
                    setup_deltas.append(sv-dv)
                if len(enum)==27 and all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in enum):
                    enumeration_sums.append(sum(enum))
        exec_spec = next(x for x in design['execution']['workloads'] if x['workload_id']==wid)
        exec_rows = {(r['round'],r['candidate_id']):r for r in execution if r['workload_id']==wid}
        delta = None
        default_init = []
        if all((k,'default') in exec_rows and (k,choice) in exec_rows for k in range(exec_spec['rounds'])):
            delta = statistics.mean(exec_rows[k,'default']['result']['exec_ms_median']-exec_rows[k,choice]['result']['exec_ms_median'] for k in range(exec_spec['rounds']))
            default_init = [exec_rows[k,'default']['result'].get('initialize_ms') for k in range(exec_spec['rounds'])]
        valid_init = len(default_init)==exec_spec['rounds'] and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in default_init)
        candidate_extra = statistics.mean(setup_deltas) if len(setup_deltas)==3 else None
        described_extra = statistics.mean(enumeration_sums)-statistics.mean(default_init) if len(enumeration_sums)==3 and valid_init else None
        components = []
        for name,extra in [('single_candidate_initialize_component',candidate_extra),('describe_policy_initialize_component',described_extra)]:
            components.append(dict(id=name, setup_delta_ms=extra, execution_delta_ms=delta,
                                   break_even_executions=math.ceil(max(0.,extra)/delta) if extra is not None and delta is not None and delta>0 else None,
                                   no_extra_initialize_cost=extra is not None and extra<=0))
        amortization.append(dict(workload_id=wid,winner=choice,execution_rounds=exec_spec['rounds'],components=components,
            limitation='Nominal same-cohort initialization components only; excludes search benchmarking, scoring, fallback and other stated overheads. No finite estimate for nonpositive execution delta.'))
    if len(results)!=24 or len(rows)!=48:
        raise ValueError('cost_coverage_invalid')
    valid_keys = {(x['workload_id'],x['round']) for x in results if x['classification']=='equivalent'}
    compilation_stages = {'nvrtc_program_create','nvrtc_compile','nvrtc_program_destroy','binary_extract','module_load','module_get_function','function_attribute_setup'}
    stage_summary = []
    for mode in ('full-search-cost','describe-select-cost'):
        selected = [x for x in pipeline_details if x['mode']==mode and (x['workload_id'],x['round']) in valid_keys]
        totals = collections.defaultdict(float)
        for item in selected:
            for name,value in item['exclusive_stages_ms'].items(): totals[name] += value
        traced = sum(totals.values())
        wall = sum(x['context_pipeline_wall_ms'] for x in selected if isinstance(x['context_pipeline_wall_ms'],(int,float)))
        compile_ms = sum(v for k,v in totals.items() if k in compilation_stages)
        stage_summary.append(dict(mode=mode,processes=len(selected),exclusive_stages_sum_ms=dict(totals),
                                  compilation_loading_exclusive_ms=compile_ms,trace_exclusive_ms=traced,pipeline_wall_ms=wall,
                                  compilation_fraction_of_traced=compile_ms/traced if traced else None,
                                  compilation_fraction_of_pipeline=compile_ms/wall if wall else None))
    return dict(pairs=results,pipelines=pipeline_details,stage_summary=stage_summary,amortization=amortization,
                feasibility='negative' if any(r['classification']=='negative' for r in results) else 'equivalent',
                interpretation='26 full constructions versus 26 descriptions plus a known offline choice; not current-default planning versus an online selector.')


def analyze(root):
    root = Path(root)
    design_path, freeze_path, manifest_path = root/'design.json', root/'freeze.json', root/'manifest.json'
    d, freeze, manifest = json_file(design_path),json_file(freeze_path),json_file(manifest_path)
    validate_design(d)
    if freeze.get('design_sha256') != sha(design_path.read_bytes()) or freeze.get('study_id') != d['study_id'] or manifest.get('design_sha256') != freeze['design_sha256'] or manifest.get('study_id') != d['study_id'] or manifest.get('gpu_uuid') != d['scope']['device_uuid'] or manifest.get('build_provenance_sha256') != sha((root/'build_provenance.json').read_bytes()):
        raise ValueError('cohort_identity_mismatch')
    manifest_sha = sha(manifest_path.read_bytes())
    stages, errors = {}, {}
    for stage in d['stage_order']:
        try:
            rows, found = stage_rows(root,stage,d,manifest_sha)
            stages[stage] = rows
            if found: errors[stage] = found
        except (ValueError,KeyError,OSError,TypeError) as e:
            stages[stage] = []
            errors[stage] = [f'{type(e).__name__}:{e}']
    for stage, count in [('equivalence',8),('semantics',28),('health',30),('execution',199),('cost',48)]:
        if len(stages[stage]) != count:
            errors.setdefault(stage,[]).append('declared_process_count_incomplete')
    if not 11 <= len(stages['calibration']) <= 33:
        errors.setdefault('calibration',[]).append('declared_calibration_count_invalid')
    checker = object.__new__(CloudController)
    checker.design, checker.output = d, root
    for stage in ('equivalence','semantics','calibration','health','execution'):
        if stage not in errors:
            try: checker.check_coverage(stage, stages[stage])
            except (ValueError,KeyError,OSError,TypeError,RuntimeError) as e: errors[stage] = [f'{type(e).__name__}:{e}']
    if 'calibration' not in errors:
        calibration_gate = json_file(root/'calibration'/'gate.json')
        if calibration_gate.get('graph_repeats_sha256') != sha((root/'graph_repeats.json').read_bytes()):
            errors['calibration'] = ['calibration_file_hash_mismatch']
    execution = []
    if not any(stage in errors for stage in ('equivalence','semantics','calibration','health','execution')):
        try:
            execution = execution_blocks(root,d,stages['execution'])
        except (ValueError,KeyError,OSError,TypeError) as e:
            errors['execution'] = [f'{type(e).__name__}:{e}']
    tests = []
    for spec in d['execution']['directional_tests']:
        pair = paired(execution,spec['workload_id'],spec['reference'],spec['target'],10) if execution else None
        result = headline_stat(pair['ratios'] if pair else [])
        tests.append(dict(**spec,**result,quality_valid=pair is not None,record_hashes=pair['record_hashes'] if pair else []))
    holm(tests)
    choices = []
    structures = []
    for spec in d['execution']['workloads']:
        effect = paired(execution,spec['workload_id'],'default',spec['winner'],spec['rounds']) if execution else None
        choices.append(dict(workload_id=spec['workload_id'],winner=spec['winner'],rounds=spec['rounds'],effect=effect,
                            interpretation='default-selected ratio is 1 by definition' if spec['winner']=='default' else 'fixed old-screen winner, newly measured'))
        for cid in spec['candidates']:
            matching = [r for r in execution if r['workload_id']==spec['workload_id'] and r['candidate_id']==cid]
            if not matching: continue
            fingerprints = [plan_fingerprints(r['result']) for r in matching]
            first = matching[0]['result']
            structures.append(dict(workload_id=spec['workload_id'],candidate_id=cid,
                core_fingerprints=sorted({x['core'] for x in fingerprints}),
                source_compiler_fingerprints=sorted({x['source_compiler'] for x in fingerprints}),
                representative_core_descriptor=fingerprints[0]['core_descriptor'],
                requested_config=first.get('requested_config'),resolved_config=first.get('resolved_config'),
                compiled_diagnostics=first.get('compiled_diagnostics'),
                representative_record_sha256=matching[0]['record_sha256']))
    cost = None
    if 'cost' not in errors and execution:
        try: cost = analyze_cost(root,d,stages['cost'],execution)
        except (ValueError,KeyError,OSError,TypeError) as e: errors['cost'] = [f'{type(e).__name__}:{e}']
    quarantined = collections.Counter()
    quarantine_ids = set()
    for path in (root/'guards').glob('*'):
        if not path.is_file(): continue
        try: guard_events = read_guard(path)
        except (OSError,ValueError,json.JSONDecodeError) as exc:
            errors.setdefault('guards',[]).append(f'{path.name}:{type(exc).__name__}:{exc}')
            continue
        for event in guard_events:
            if event.get('event')=='quarantined_run':
                active = event.get('active_run') or {}
                key = (event.get('guard_id'),active.get('guard_run_id'))
                if key in quarantine_ids: continue
                quarantine_ids.add(key)
                phase = active.get('phase','')
                stage = 'equivalence' if phase.startswith('equivalence/') else 'semantics' if phase.startswith('semantics/') else phase
                quarantined[stage] += 1
    return dict(schema_version='cloud-reproduction-report/v1',study_id=d['study_id'],design_sha256=freeze['design_sha256'],
                gpu_uuid=d['scope']['device_uuid'],manifest_sha256=manifest_sha,stage_errors=errors,
                selection_origin=d['selection_origin'],old_cohort_link=d['reporting']['old_cohort_link'],
                stage_record_counts={stage:len(rows) for stage,rows in stages.items()},
                stage_raw_line_counts={stage:sum(1 for _ in (root/stage/'measurements.jsonl').open('rb')) if (root/stage/'measurements.jsonl').exists() else 0 for stage in d['stage_order']},
                stage_accepted_counts={stage:(len(rows) if stage not in errors else 0) for stage,rows in stages.items()},
                stage_quarantined_guard_run_counts={stage:quarantined[stage] for stage in d['stage_order']},directional_tests=tests,
                crossover_confirmed=len([t for t in tests if t['hypothesis']=='H2'])==2 and all(t['clear_gain'] for t in tests if t['hypothesis']=='H2'),
                fixed_choice_effects=choices,structures=structures,cost=cost,
                boundaries=['Old-cohort data are not combined with this cohort.', 'Choices were informed by old data and frozen before new data; no new H4 training.',
                            'A known-choice construction component is not an online selector or a current-default planner comparison.',
                            'Scientific conclusions require all declared guard, numerical and whole-block checks.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = analyze(args.root)
    write_json(args.output, report)
    print(json.dumps(dict(output=str(args.output),stage_errors=report['stage_errors']),indent=2))


if __name__ == '__main__':
    main()
