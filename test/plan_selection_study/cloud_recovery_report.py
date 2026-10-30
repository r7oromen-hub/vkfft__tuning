#!/usr/bin/env python3
"""GPU-free evidence audit and scientific report for the frozen E1 recovery."""
import argparse
import json
from pathlib import Path
import types

import cloud_reproduction_report as report
from cloud_recovery import RecoveryController, audit_interrupted, remaining_schedule, validate_root_layout
from cloud_reproduction import CloudController, build_schedule, json_file, validate_design
from study_common import canonical, read_jsonl, sha, write_json
from trace_audit import plan_fingerprints


def analyze_recovery(original_root,recovery_root,admission_path,freeze_path):
    original,recovery = Path(original_root).resolve(),Path(recovery_root).resolve()
    admission_path,freeze_path = Path(admission_path),Path(freeze_path)
    admission,freeze = json_file(admission_path),json_file(freeze_path)
    if (sha(freeze_path.read_bytes())!='e5f4f9013889db853f6a64ad0d6ce89d5fdab964f8e2bc4fcefa913fb04071e1' or
        freeze.get('admission_sha256')!=sha(admission_path.read_bytes()) or freeze.get('recovery_id')!=admission.get('recovery_id')):
        raise ValueError('recovery_admission_freeze_mismatch')
    here = Path(__file__).parent.resolve()
    base = validate_root_layout(here,original,recovery,admission)
    errors = {}
    for item in admission['source']['files']:
        path = (base/item['path']).resolve()
        if not path.is_relative_to(original) or not path.is_file() or sha(path.read_bytes())!=item['sha256']:
            errors.setdefault('original',[]).append('source_hash_mismatch:'+item['path'])
    design_path = original/'design.json'
    d = json_file(design_path)
    validate_design(d)
    if sha(design_path.read_bytes())!=admission['primary_design_sha256'] or d['study_id']!=admission['study_id']:
        errors.setdefault('original',[]).append('original_design_identity_mismatch')
    original_manifest_sha = sha((original/'manifest.json').read_bytes())
    original_manifest = json_file(original/'manifest.json')
    for name,digest in original_manifest.get('controller_files_sha256',{}).items():
        path = here/name
        if not path.is_file() or sha(path.read_bytes())!=digest:
            errors.setdefault('original',[]).append('controller_source_hash_mismatch:'+name)
    guard_copy = here/'gpu_guard_cloud_recovery.py'
    if not guard_copy.is_file() or sha(guard_copy.read_bytes())!=admission['guards']['recovery_guard_sha256']:
        errors.setdefault('recovery',[]).append('recovery_guard_source_hash_mismatch')
    reviewer = object.__new__(CloudController)
    reviewer.design,reviewer.output = d,original
    for stage in ('equivalence','semantics','calibration','health'):
        try:
            status = json_file(original/stage/'status.json')
            if status.get('state')!='complete': raise ValueError('original_stage_not_complete')
            reviewer.require(stage)
        except (ValueError,KeyError,OSError,TypeError,RuntimeError) as exc:
            errors.setdefault('original',[]).append(stage+':'+str(exc))
    try:
        preserved = audit_interrupted(original,d,admission)
        remaining_schedule(d,admission)
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as exc:
        errors.setdefault('original',[]).append('interrupted_prefix:'+str(exc))
        preserved = dict(rows=[],orphan_record_hashes=[],quarantined_guard_run_id=None)
    recovery_manifest_path = recovery/'recovery_manifest.json'
    recovery_manifest = json_file(recovery_manifest_path) if recovery_manifest_path.exists() else {}
    expected = dict(recovery_id=admission['recovery_id'],admission_sha256=sha(admission_path.read_bytes()),
                    freeze_sha256=sha(freeze_path.read_bytes()),original_manifest_sha256=original_manifest_sha,
                    graph_repeats_sha256=admission['source']['graph_repeats_sha256'],
                    recovery_guard_sha256=admission['guards']['recovery_guard_sha256'],device_uuid=d['scope']['device_uuid'])
    if any(recovery_manifest.get(k)!=v for k,v in expected.items()) or not recovery_manifest_path.exists():
        errors.setdefault('recovery',[]).append('recovery_manifest_identity_mismatch')
    elif recovery_manifest.get('recovery_controller_sha256')!=sha((here/'cloud_recovery.py').read_bytes()):
        errors.setdefault('recovery',[]).append('recovery_controller_hash_mismatch')
    if (recovery/'manifest.json').exists() and sha((recovery/'manifest.json').read_bytes())!=original_manifest_sha:
        errors.setdefault('recovery',[]).append('recovery_runtime_identity_mismatch')
    if not (recovery/'graph_repeats.json').is_file() or sha((recovery/'graph_repeats.json').read_bytes())!=admission['source']['graph_repeats_sha256']:
        errors.setdefault('recovery',[]).append('recovery_calibration_hash_mismatch')
    rmsha = sha(recovery_manifest_path.read_bytes()) if recovery_manifest_path.exists() else None
    stages = {}
    for stage,count in [('health',30),('execution',100),('cost',48)]:
        try:
            rows,issues = report.stage_rows(recovery,stage,d,original_manifest_sha)
            stages[stage] = rows
            if issues: errors.setdefault(stage,[]).extend(issues)
            if len(rows)!=count: errors.setdefault(stage,[]).append('declared_process_count_incomplete')
            status_path = recovery/stage/'status.json'
            if not status_path.exists() or json_file(status_path).get('state')!='complete':
                errors.setdefault(stage,[]).append('stage_not_complete')
            if any(r.get('recovery_manifest_sha256')!=rmsha for r in rows):
                errors.setdefault(stage,[]).append('recovery_record_manifest_binding_failed')
        except (ValueError,KeyError,OSError,TypeError) as exc:
            stages[stage] = []
            errors.setdefault(stage,[]).append(f'{type(exc).__name__}:{exc}')
    if 'health' not in errors:
        try:
            recovery_reviewer = object.__new__(CloudController)
            recovery_reviewer.design,recovery_reviewer.output = d,recovery
            recovery_reviewer.require('health')
        except (ValueError,KeyError,OSError,TypeError,RuntimeError) as exc:
            errors['health'] = [str(exc)]
    accepted = []
    selection_path = recovery/'execution_selection.json'
    if not any(x in errors for x in ('original','recovery','health','execution')):
        try:
            checker = object.__new__(RecoveryController)
            checker.design,checker.output,checker.original,checker.admission,checker.preserved = d,recovery,original,admission,preserved
            checker.args = types.SimpleNamespace(admission=admission_path)
            checker.candidates = {x['id']:x for x in d['candidates']}
            checker.workloads = {x['id']:x for x in d['calibration']['workloads']}
            checker.workloads.update({x['id']:x for x in d['workloads']})
            checker.require_recovery_execution()
            selection = json_file(selection_path)
            old = {r['record_sha256']:r for r in preserved['rows']}
            new = {r['record_sha256']:r for r in stages['execution']}
            for block in selection['blocks']:
                lookup = old if block['source_root']=='original' else new
                accepted.extend(lookup[h] for h in block['record_hashes'])
            if len(accepted)!=199 or len(selection['blocks'])!=65:
                raise ValueError('selected_coverage_mismatch')
        except (ValueError,KeyError,OSError,TypeError,RuntimeError) as exc:
            errors['execution'] = [str(exc)]
            accepted = []
    tests = []
    for spec in d['execution']['directional_tests']:
        pair = report.paired(accepted,spec['workload_id'],spec['reference'],spec['target'],10) if accepted else None
        tests.append(dict(**spec,**report.headline_stat(pair['ratios'] if pair else []),
                          quality_valid=pair is not None,record_hashes=pair['record_hashes'] if pair else []))
    report.holm(tests)
    choices,structures = [],[]
    for spec in d['execution']['workloads']:
        effect = report.paired(accepted,spec['workload_id'],'default',spec['winner'],spec['rounds']) if accepted else None
        choices.append(dict(workload_id=spec['workload_id'],winner=spec['winner'],rounds=spec['rounds'],effect=effect,
                            interpretation='default-selected ratio is 1 by definition' if spec['winner']=='default' else 'fixed old-screen winner, newly measured'))
        for cid in spec['candidates']:
            matching = [r for r in accepted if r['workload_id']==spec['workload_id'] and r['candidate_id']==cid]
            if not matching: continue
            fps = [plan_fingerprints(r['result']) for r in matching]
            first = matching[0]['result']
            structures.append(dict(workload_id=spec['workload_id'],candidate_id=cid,
                core_fingerprints=sorted({x['core'] for x in fps}),source_compiler_fingerprints=sorted({x['source_compiler'] for x in fps}),
                representative_core_descriptor=fps[0]['core_descriptor'],requested_config=first.get('requested_config'),
                resolved_config=first.get('resolved_config'),compiled_diagnostics=first.get('compiled_diagnostics'),
                representative_record_sha256=matching[0]['record_sha256']))
    cost = None
    if 'cost' not in errors and accepted and not any(x in errors for x in ('original','recovery','health','execution')):
        try: cost = report.analyze_cost(recovery,d,stages['cost'],accepted)
        except (ValueError,KeyError,OSError,TypeError) as exc: errors['cost'] = [str(exc)]
    return dict(schema_version='cloud-reproduction-report/v1',study_id=d['study_id'],recovery_id=admission['recovery_id'],
        admission_sha256=sha(admission_path.read_bytes()),design_sha256=admission['primary_design_sha256'],gpu_uuid=d['scope']['device_uuid'],
        manifest_sha256=original_manifest_sha,recovery_manifest_sha256=rmsha,
        execution_selection_sha256=sha(selection_path.read_bytes()) if selection_path.exists() else None,
        selection_origin=d['selection_origin'],old_cohort_link=d['reporting']['old_cohort_link'],stage_errors=errors,
        original_preflight_record_counts={stage:len(read_jsonl(original/stage/'measurements.jsonl')) for stage in ('equivalence','semantics','calibration','health')},
        stage_record_counts={stage:len(rows) for stage,rows in stages.items()},old_preserved_count=len(preserved['rows']),
        selected_record_count=len(accepted),selected_block_count=65 if len(accepted)==199 else 0,
        directional_tests=tests,crossover_confirmed=len([x for x in tests if x['hypothesis']=='H2'])==2 and all(x['clear_gain'] for x in tests if x['hypothesis']=='H2'),
        fixed_choice_effects=choices,structures=structures,cost=cost,
        boundaries=['Original execution remains stopped; this is a separately frozen bounded recovery.',
                    'Exactly 99 old closed-interval records and 100 recovery records; two old orphan records and the quarantined run are excluded.',
                    'Old-host and cloud observations are not pooled; known-choice costs are not an online selector.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    value = analyze_recovery(args.original,args.recovery,args.admission,args.freeze)
    write_json(args.output,value)
    print(json.dumps(dict(output=str(args.output),stage_errors=value['stage_errors']),indent=2))


if __name__=='__main__': main()
