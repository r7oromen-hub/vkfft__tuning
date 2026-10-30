#!/usr/bin/env python3
"""One bounded CPU-only H4 run on independently audited old S1, after admission."""
import argparse
import collections
import json
import math
import os
from pathlib import Path

from candidate_policy import execution_invalid, init_errors
from guard_audit import completed_session, resolve_log
from learning_check import ABLATIONS, DESCRIPTOR_FIELDS, MODEL, prepare_rows, run_learning
from report_helpers import guard_check
from selection_analysis import freeze_screen
from study_common import (block_candidates, block_quality, candidate_config, canonical,
                          execution_errors, input_seed, load_design, read_jsonl, sha, utc, write_json)


class AdmissionError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return sha(Path(path).read_bytes())


def audit_old_screen(root, design, admission):
    """Reconstruct exactly the old accepted S1 blocks, never read cloud labels."""
    root = Path(root)
    manifest = read(root/'manifest.json')
    gate = read(root/'screen'/'gate.json')
    status = read(root/'screen'/'status.json')
    selection = read(root/'screen_selection.json')
    semantic = read(root/'semantics'/'gate.json')
    if (manifest.get('study_id')!=design['study_id'] or manifest.get('design_sha256')!=design['_sha256'] or
        manifest.get('gpu_uuid')!=design['scope']['device_uuid'] or gate.get('passed') is not True or
        status.get('state') not in ('complete','complete_with_limitations') or semantic.get('protocol_valid') is not True or
        selection.get('study_id')!=design['study_id'] or selection.get('design_sha256')!=design['_sha256'] or
        gate.get('selection_sha256')!=digest(root/'screen_selection.json')):
        raise AdmissionError('old_study_gate_or_identity_mismatch')
    semantic_cells = {(x['workload_id'],x['candidate_id']) for x in semantic.get('semantic_invalid_cells',[])}
    if len(semantic_cells)!=len(semantic.get('semantic_invalid_cells',[])):
        raise AdmissionError('duplicate_semantic_invalid_cells')
    rows = read_jsonl(root/'screen'/'measurements.jsonl')
    screen = [r for r in rows if r.get('phase')=='screen']
    if len(screen)!=936 or len(rows)!=1007 or sum(r.get('phase')=='calibration' for r in rows)!=71:
        raise AdmissionError('old_screen_raw_coverage_mismatch')
    byhash = {r.get('record_sha256'):r for r in screen}
    if None in byhash or len(byhash)!=936:
        raise AdmissionError('old_screen_duplicate_or_missing_hash')
    candidates = {c['id']:c for c in design['candidates']}
    repeats = read(root/'graph_repeats.json')
    guard_cache = {}
    blocks = {}
    accepted = []
    paths = list((root/'screen'/'blocks').glob('*.json'))
    if len(paths)!=36:
        raise AdmissionError('old_screen_block_count_mismatch')
    for w in design['workloads']:
        wid = w['id']
        block = read(root/'screen'/'blocks'/f'screen__{wid}__r0.json')
        expected_ids = block_candidates(design,wid,'screen',0)
        hashes = block.get('record_hashes',[])
        if (block.get('candidate_ids')!=expected_ids or len(hashes)!=26 or len(set(hashes))!=26 or
            not all(h in byhash for h in hashes)):
            raise AdmissionError('old_screen_block_membership_mismatch:'+wid)
        group = [byhash[h] for h in hashes]
        for pos,row in enumerate(group):
            cid = row['candidate_id']
            if (cid!=expected_ids[pos] or row.get('study_id')!=design['study_id'] or row.get('phase')!='screen' or
                row.get('workload_id')!=wid or row.get('round')!=0 or row.get('block_attempt',0)!=0 or
                row.get('position_in_block')!=pos or row.get('config')!=candidate_config(candidates[cid]) or
                row.get('input_seed')!=input_seed(design,'screen',wid,0) or
                sha(canonical({k:v for k,v in row.items() if k!='record_sha256'}))!=row['record_sha256'] or
                guard_check(row,design['scope']['device_uuid'],root,guard_cache)):
                raise AdmissionError('old_screen_row_identity_or_guard_failed:'+wid+':'+cid)
            workload = row['workload']
            if (any(workload.get(k)!=w[k] for k in ('id','n','batch','input_bytes')) or
                workload.get('graph_repeats')!=repeats[wid]['graph_repeats'] or workload.get('samples')!=design['measurement']['screen_samples']):
                raise AdmissionError('old_screen_measurement_settings_failed:'+wid+':'+cid)
            is_semantic_invalid = (wid,cid) in semantic_cells
            if is_semantic_invalid != (row.get('selection_eligible') is False):
                raise AdmissionError('old_screen_semantic_eligibility_mismatch:'+wid+':'+cid)
            names = dict(aim_threads='aimThreads',register_boost='registerBoost',use_lut='useLUT',
                         register_boost_nonpow2='registerBoostNonPow2',coalesced_memory='coalescedMemory',
                         use_lut_4step='useLUT_4step',swap_to_2stage='swapTo2Stage4Step',swap_to_3stage='swapTo3Stage4Step')
            expected_request = {'makeForwardPlanOnly':1}
            expected_request.update({names[k]:v for k,v in row['config'].items()})
            result = row['result']
            if (result.get('requested_config')!=expected_request or result.get('seed')!=row['input_seed'] or
                result.get('gpu',{}).get('uuid')!=design['scope']['device_uuid']):
                raise AdmissionError('old_screen_result_request_seed_or_uuid_mismatch:'+wid+':'+cid)
            if row.get('status')=='ok':
                if execution_errors(row['result'],workload,design['measurement']['screen_samples']):
                    raise AdmissionError('old_screen_success_protocol_failed:'+wid+':'+cid)
            elif row.get('status')=='failed' and cid not in ('default','default_replica'):
                result = row.get('result',{})
                normal = (result.get('failure_phase')=='single_forward_validation' and result.get('single_forward_input_unchanged') is True and
                          result.get('error')=='Single out-of-place forward failed immutable-input or numerical validation' and
                          result.get('single_forward_correctness',{}).get('passed') is False and not init_errors(result))
                if not normal and not (row.get('candidate_validity_class')=='execution_invalid_single_forward_A5' and execution_invalid(row)):
                    raise AdmissionError('old_screen_failure_classification_unverified:'+wid+':'+cid)
            else:
                raise AdmissionError('old_screen_default_or_status_failed:'+wid+':'+cid)
        quality = block_quality(group)
        if not quality['ok']:
            raise AdmissionError('old_screen_block_quality_failed:'+wid)
        blocks[wid] = group
        accepted.extend(group)
    if len(accepted)!=936 or {r['record_sha256'] for r in accepted}!=set(byhash):
        raise AdmissionError('old_screen_accepted_coverage_mismatch')
    sessions = {(r['guard_id'],r['guard_log']) for r in accepted}
    if sessions != {(s['guard_id'],s['guard_log']) for s in gate['guard_sessions']}:
        raise AdmissionError('old_screen_guard_session_set_mismatch')
    for gid,log in sessions:
        if not completed_session(resolve_log({'guard_log':log},root),gid):
            raise AdmissionError('old_screen_guard_session_incomplete')
    recomputed = freeze_screen(design,blocks)
    if any(recomputed[k]!=selection.get(k) for k in ('global_candidate','global_scores','workloads')):
        raise AdmissionError('old_screen_frozen_selection_mismatch')
    prepared,coverage,excluded = prepare_rows(design,accepted)
    if len(coverage)!=36 or excluded:
        raise AdmissionError('old_screen_learning_coverage_mismatch')
    if admission:
        source = admission['source']
        if (sorted(r['record_sha256'] for r in prepared)!=source['model_source_record_hashes'] or
            coverage!=source['eligible_candidate_sets'] or len(prepared)!=source['eligible_model_rows'] or
            selection['global_candidate']!=source['frozen_global_candidate']):
            raise AdmissionError('old_screen_frozen_model_inputs_mismatch')
        for item in source['blocks']:
            if [r['record_sha256'] for r in blocks[item['workload_id']]]!=item['record_hashes']:
                raise AdmissionError('old_screen_frozen_block_hashes_mismatch')
    return dict(rows=accepted,global_candidate=selection['global_candidate'],
                audit=dict(screen_records=len(accepted),raw_records=len(rows),calibration_records=len(rows)-len(screen),
                           blocks=len(blocks),screen_successes=sum(r['status']=='ok' for r in accepted),
                           screen_failures=sum(r['status']!='ok' for r in accepted),
                           semantic_invalid_cells=[list(x) for x in sorted(semantic_cells)],
                           legal_candidate_counts={x['workload_id']:len(x['legal_candidates']) for x in coverage},
                           guard_sessions=[dict(guard_id=g,guard_log=p) for g,p in sorted(sessions)]))


def check_admission(admission_path, freeze_path, old_root, cloud_root, output):
    here = Path(__file__).parent.resolve()
    freeze = read(freeze_path)
    admission = read(admission_path)
    if (digest(freeze_path)!='f1f364113e9e710349066b91a8a969f710f0763fc88c4060f1c59eca1f3d2b55' or
        freeze.get('admission_sha256')!=digest(admission_path) or freeze.get('analysis_id')!=admission.get('analysis_id')):
        raise AdmissionError('admission_freeze_mismatch')
    if (Path(old_root).resolve() != (here/admission['source']['root']).resolve() or
        Path(cloud_root).resolve() != (here/admission['activation']['primary_root']).resolve() or
        Path(output).resolve() != (here/admission['output']['root']).resolve()):
        raise AdmissionError('frozen_input_or_output_path_mismatch')
    analysis = admission['analysis']
    if (analysis['model']!=MODEL or analysis['ablations']!=list(ABLATIONS) or
        analysis['precompile_descriptor_allowlist']!=list(DESCRIPTOR_FIELDS) or analysis['fits_max']!=24 or
        analysis['folds']!=4 or analysis['targets']!=2 or analysis['ablations_per_target']!=3 or
        analysis['real_data_fitting_attempts_max']!=1 or analysis['gpu_processes']!=0):
        raise AdmissionError('frozen_model_or_budget_mismatch')
    for item in admission['source']['files']+admission['source']['blocks']:
        source = (here/item['path']).resolve()
        if not source.is_relative_to(here) or not source.is_file() or digest(source)!=item['sha256']:
            raise AdmissionError('frozen_source_file_mismatch:'+item['path'])
    return admission


def check_primary(admission, cloud_root, disposition_path=None):
    root = Path(cloud_root)
    spec = admission['activation']
    design_path = root/'design.json'
    if not design_path.is_file() or digest(design_path)!=spec['primary_design_sha256']:
        raise AdmissionError('cloud_primary_design_mismatch')
    design = read(design_path)
    if design.get('study_id')!=spec['primary_study_id']:
        raise AdmissionError('cloud_primary_study_mismatch')
    stages = spec['primary_stages']
    statuses,bindings = {},{}
    for stage in stages:
        path = root/stage/'status.json'
        if path.exists():
            statuses[stage] = read(path)
            bindings[stage] = digest(path)
            if statuses[stage].get('study_id')!=spec['primary_study_id'] or statuses[stage].get('stage')!=stage:
                raise AdmissionError('cloud_primary_status_identity_mismatch:'+stage)
    if any(s.get('state') in ('starting','running') for s in statuses.values()):
        raise AdmissionError('cloud_primary_sequence_still_running')
    if disposition_path is None:
        if len(statuses)!=len(stages) or any(statuses[s].get('state')!='complete' for s in stages):
            raise AdmissionError('cloud_primary_sequence_not_complete')
        from cloud_reproduction_report import analyze
        audit = analyze(root)
        if audit.get('study_id')!=spec['primary_study_id'] or audit.get('design_sha256')!=spec['primary_design_sha256'] or audit.get('stage_errors'):
            raise AdmissionError('cloud_primary_reaudit_failed')
        return dict(schema_version='cloud-primary-terminal-disposition/v1',study_id=spec['primary_study_id'],
                    design_sha256=spec['primary_design_sha256'],sequence_ended=True,disposition='complete',
                    status_file_bindings=bindings,reason='All six stages completed and passed independent re-audit.',
                    missing_primary_evidence=[])
    value = read(disposition_path)
    if any(k not in value for k in spec['required_terminal_disposition_fields']):
        raise AdmissionError('terminal_disposition_fields_missing')
    if (value.get('study_id')!=spec['primary_study_id'] or value.get('design_sha256')!=spec['primary_design_sha256'] or
        value.get('sequence_ended') is not True or value.get('disposition')!='technical_stop' or
        not isinstance(value.get('schema_version'),str) or not value['schema_version'] or
        not isinstance(value.get('reason'),str) or not value['reason'].strip() or
        not isinstance(value.get('missing_primary_evidence'),list) or not value['missing_primary_evidence'] or
        value.get('status_file_bindings')!=bindings or not any(s.get('state')=='stopped' for s in statuses.values())):
        raise AdmissionError('terminal_technical_stop_not_established')
    if any(s.get('state') not in ('complete','stopped') for s in statuses.values()):
        raise AdmissionError('cloud_primary_status_not_terminal')
    return value


def validate_result(result,admission):
    analysis = admission['analysis']
    source = admission['source']
    outcomes = result.get('outcomes',[])
    expected = {(t,a) for t in (analysis['primary_target'],analysis['secondary_target']) for a in analysis['ablations']}
    if (result.get('status')!='completed_auxiliary' or result.get('study_id')!=source['study_id'] or
        result.get('model')!=analysis['model'] or result.get('distinct_N')!=16 or
        result.get('source_record_hashes')!=source['model_source_record_hashes'] or
        result.get('coverage')!=source['eligible_candidate_sets'] or result.get('excluded_workloads') or
        len(outcomes)!=6 or {(x['target'],x['ablation']) for x in outcomes}!=expected or
        set(result.get('baselines',{}))!={'default','training_only_global','frozen_S1_global_diagnostic','screen_oracle_diagnostic'} or
        {str(k):v for k,v in result.get('folds',{}).items()}!=analysis['fold_assignment_by_exact_N']):
        raise AdmissionError('completed_learning_result_does_not_match_freeze')
    by_wid = {x['workload_id']:x for x in source['eligible_candidate_sets']}
    fold_by_n = {int(k):v for k,v in analysis['fold_assignment_by_exact_N'].items()}
    for outcome in outcomes:
        folds = outcome.get('fold_details',[])
        if len(folds)!=4 or {f['fold'] for f in folds}!={0,1,2,3} or len(outcome.get('workloads',[]))!=36:
            raise AdmissionError('completed_learning_fold_or_coverage_mismatch')
        for f in folds:
            expected_test = sorted(n for n,k in fold_by_n.items() if k==f['fold'])
            expected_train = sorted(n for n,k in fold_by_n.items() if k!=f['fold'])
            if f.get('test_N')!=expected_test or f.get('train_N')!=expected_train:
                raise AdmissionError('completed_learning_fold_assignment_mismatch')
        predictions = outcome['workloads']
        if len({r['workload_id'] for r in predictions})!=36 or {r['workload_id'] for r in predictions}!=set(by_wid):
            raise AdmissionError('completed_learning_workload_coverage_mismatch')
        for r in predictions:
            frozen = by_wid[r['workload_id']]
            if (r.get('n')!=frozen['n'] or r.get('batch')!=frozen['batch'] or r.get('fold')!=fold_by_n[frozen['n']] or
                r.get('selected_candidate') not in frozen['legal_candidates']):
                raise AdmissionError('completed_learning_prediction_identity_mismatch')
    for name,baseline in result['baselines'].items():
        rows = baseline.get('workloads',[])
        if len(rows)!=36 or len({r['workload_id'] for r in rows})!=36 or {r['workload_id'] for r in rows}!=set(by_wid):
            raise AdmissionError('completed_learning_baseline_coverage_mismatch')
        for r in rows:
            frozen = by_wid[r['workload_id']]
            if r.get('fold')!=fold_by_n[frozen['n']] or (r.get('available',True) and r.get('selected_candidate') not in frozen['legal_candidates']):
                raise AdmissionError('completed_learning_baseline_identity_mismatch')
            if name=='default' and (not r.get('available',True) or r.get('selected_candidate')!='default'):
                raise AdmissionError('completed_learning_default_baseline_mismatch')


def run(args):
    admission = check_admission(args.admission,args.freeze,args.old_root,args.cloud_root,args.output)
    design = load_design(args.design)
    design['_sha256'] = digest(args.design)
    if design['_sha256']!=admission['source']['design_sha256'] or design['study_id']!=admission['source']['study_id'] or design['scope']['device_uuid']!=admission['source']['gpu_uuid']:
        raise AdmissionError('old_design_identity_mismatch')
    expected_folds = {str(n):i%4 for i,n in enumerate(sorted({w['n'] for w in design['workloads']}))}
    if expected_folds!=admission['analysis']['fold_assignment_by_exact_N']:
        raise AdmissionError('old_exact_N_fold_assignment_mismatch')
    verified = audit_old_screen(args.old_root,design,admission)
    if getattr(args,'audit_only',False):
        return dict(status='audit_passed_no_fitting',analysis_id=admission['analysis_id'],audit=verified['audit'])
    terminal = check_primary(admission,args.cloud_root,getattr(args,'primary_disposition',None))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    marker = output/admission['output']['reservation']
    if any((output/name).exists() for name in (admission['output']['learning_result'],admission['output']['provenance'],'analysis_failed.json')):
        raise AdmissionError('prior_analysis_artifact_exists_no_automatic_retry')
    # Exclusive marker is created before any fit; an interrupted attempt never reruns silently.
    import platform
    import numpy, scipy, sklearn
    provenance = dict(schema_version='old-screen-h4-provenance/v1',analysis_id=admission['analysis_id'],
                      created_utc=utc(),admission_sha256=digest(args.admission),freeze_sha256=digest(args.freeze),
                      wrapper_sha256=digest(Path(__file__)),old_root=str(Path(args.old_root).resolve()),
                      cloud_root=str(Path(args.cloud_root).resolve()),terminal_disposition=terminal,
                      terminal_disposition_file_sha256=digest(args.primary_disposition) if getattr(args,'primary_disposition',None) else None,
                      model_source_record_hashes=admission['source']['model_source_record_hashes'],
                      source_bindings=admission['source']['files'],block_bindings=admission['source']['blocks'],
                      additional_audit_hashes=dict(candidate_policy_py=digest(Path(__file__).with_name('candidate_policy.py')),
                                                   cloud_reproduction_report_py=digest(Path(__file__).with_name('cloud_reproduction_report.py')),
                                                   old_semantics_gate=digest(Path(args.old_root)/'semantics'/'gate.json')),
                      fold_assignment_by_exact_N=expected_folds,old_screen_audit=verified['audit'],
                      versions=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                      interpretation=admission['interpretation'])
    try:
        with marker.open('x',encoding='utf-8') as handle:
            json.dump(dict(analysis_id=admission['analysis_id'],reserved_utc=utc(),admission_sha256=digest(args.admission)),handle)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AdmissionError('real_data_fit_already_reserved_no_automatic_retry') from exc
    try:
        write_json(output/admission['output']['provenance'],provenance)
        value = run_learning(design,verified['rows'],evidence_verified=True,frozen_global_candidate=verified['global_candidate'])
        validate_result(value,admission)
        target = output/admission['output']['learning_result']
        if target.exists(): raise AdmissionError('learning_result_already_exists')
        write_json(target,value)
        return dict(status=value['status'],analysis_id=admission['analysis_id'],output=str(target),
                    result_sha256=digest(target),fits=24)
    except BaseException as exc:
        write_json(output/'analysis_failed.json',dict(analysis_id=admission['analysis_id'],failed_utc=utc(),
                   reason=f'{type(exc).__name__}:{exc}',automatic_retry=False))
        raise


def main():
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',type=Path,default=here/'old_screen_h4_admission.json')
    p.add_argument('--freeze',type=Path,default=here/'old_screen_h4_admission_freeze.json')
    p.add_argument('--old-root',type=Path,default=here/'results/resume_20261028_attempt2')
    p.add_argument('--cloud-root',type=Path,default=here/'results/cloud_reproduction_20261028_r1')
    p.add_argument('--output',type=Path,default=here/'results/old_screen_h4_20261028_r1')
    p.add_argument('--design',type=Path,default=here/'research_design.json')
    p.add_argument('--primary-disposition',type=Path)
    p.add_argument('--audit-only',action='store_true',help='Re-audit old inputs only; never fit or reserve.')
    args = p.parse_args()
    print(json.dumps(run(args),indent=2))


if __name__=='__main__':
    main()
