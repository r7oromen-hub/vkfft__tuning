#!/usr/bin/env python3
"""Activate the unchanged old-S1 H4 only after independently audited E1 completion."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import old_screen_learning as old
from study_common import canonical, sha


HERE = Path(__file__).parent
C1_FREEZE_SHA256 = 'cff6d74e62d67e59b622d5ea9bcc5895110ed5c1baa24e6e1af81ff24828fdac'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return sha(Path(path).read_bytes())


def check_attachment(args):
    c1,freeze = read(args.attachment),read(args.attachment_freeze)
    if (digest(args.attachment_freeze)!=C1_FREEZE_SHA256 or
        freeze.get('attachment_sha256')!=digest(args.attachment) or
        freeze.get('attachment_id')!=c1.get('attachment_id') or freeze.get('analysis_id')!=c1.get('analysis_id') or
        freeze.get('model_or_data_changes') is not False or freeze.get('additional_fits')!=0):
        raise old.AdmissionError('recovery_activation_attachment_freeze_mismatch')
    for path,key in ((args.admission,'base_admission_sha256'),(args.freeze,'base_freeze_sha256'),
                     (args.recovery_admission,'recovery_admission_sha256'),(args.recovery_freeze,'recovery_freeze_sha256')):
        if digest(path)!=c1[key]: raise old.AdmissionError('recovery_activation_source_hash_mismatch:'+key)
    if (Path(args.cloud_root).resolve()!=(HERE/c1['primary_root']).resolve() or
        Path(args.recovery_root).resolve()!=(HERE/c1['recovery_root']).resolve() or
        getattr(args,'primary_disposition',None) is not None):
        raise old.AdmissionError('recovery_activation_root_or_disposition_mismatch')
    base,recovery = read(args.admission),read(args.recovery_admission)
    if (base.get('analysis_id')!=c1['analysis_id'] or recovery.get('recovery_id')!=c1['recovery_id'] or
        recovery.get('study_id')!=c1['primary_study_id'] or recovery.get('primary_design_sha256')!=c1['primary_design_sha256']):
        raise old.AdmissionError('recovery_activation_identity_mismatch')
    return c1


def verify_completion(args):
    """Independent E1 evidence audit only; do not return any new labels to H4."""
    c1 = check_attachment(args)
    original,recovered = Path(args.cloud_root),Path(args.recovery_root)
    design_path = original/'design.json'
    if not design_path.is_file() or digest(design_path)!=c1['primary_design_sha256']:
        raise old.AdmissionError('recovery_activation_primary_design_mismatch')
    design = read(design_path)
    from cloud_recovery_report import analyze_recovery
    report = analyze_recovery(original,recovered,args.recovery_admission,args.recovery_freeze)
    tests = report.get('directional_tests',[])
    expected_tests = design.get('execution',{}).get('directional_tests',[])
    fields = ('test_id','hypothesis','workload_id','reference','target')
    if (report.get('study_id')!=c1['primary_study_id'] or report.get('design_sha256')!=c1['primary_design_sha256'] or
        report.get('gpu_uuid')!=design.get('scope',{}).get('device_uuid') or
        report.get('recovery_id')!=c1['recovery_id'] or report.get('stage_errors') or
        report.get('admission_sha256')!=c1['recovery_admission_sha256'] or
        report.get('stage_record_counts')!={'health':30,'execution':100,'cost':48} or
        report.get('old_preserved_count')!=99 or report.get('selected_record_count')!=199 or
        report.get('selected_block_count')!=65 or report.get('cost') is None or
        len(tests)!=5 or len(expected_tests)!=5 or
        [{k:t.get(k) for k in fields} for t in tests] != [{k:t.get(k) for k in fields} for t in expected_tests]):
        raise old.AdmissionError('recovery_activation_combined_audit_failed')
    selection_path = recovered/'execution_selection.json'
    manifest_path = recovered/'recovery_manifest.json'
    selection = read(selection_path)
    blocks = selection.get('blocks',[])
    hashes = [h for b in blocks for h in b.get('record_hashes',[])]
    if (report.get('execution_selection_sha256')!=digest(selection_path) or
        report.get('recovery_manifest_sha256')!=digest(manifest_path) or
        selection.get('recovery_id')!=c1['recovery_id'] or len(blocks)!=65 or len(hashes)!=199 or
        len(set(hashes))!=199 or selection.get('selected_old_records')!=99 or selection.get('selected_new_records')!=100 or
        set(hashes)&set(selection.get('excluded_orphan_record_hashes',[]))):
        raise old.AdmissionError('recovery_activation_selection_mismatch')
    statuses = {}
    for label,root,stages in (('original',original,('equivalence','semantics','calibration','health','execution','cost')),
                              ('recovery',recovered,('health','execution','cost'))):
        for stage in stages:
            path = root/stage/'status.json'
            if not path.is_file():
                if (label,stage)==('original','cost'): continue
                raise old.AdmissionError('recovery_activation_status_missing:'+label+':'+stage)
            value = read(path)
            if value.get('study_id')!=c1['primary_study_id'] or value.get('stage')!=stage:
                raise old.AdmissionError('recovery_activation_status_identity_mismatch')
            required = 'stopped' if (label,stage)==('original','execution') else 'complete'
            if value.get('state')!=required:
                raise old.AdmissionError('recovery_activation_status_not_complete:'+label+':'+stage)
            statuses[label+'/'+stage] = digest(path)
    if (original/'cost'/'status.json').exists():
        raise old.AdmissionError('original_cost_status_must_remain_absent')
    for stage,count in (('health',30),('execution',100),('cost',48)):
        gate = read(recovered/stage/'gate.json')
        if gate.get('passed') is not True or gate.get('record_count')!=count:
            raise old.AdmissionError('recovery_activation_gate_count_mismatch:'+stage)
    # Report is re-created in memory and represented only by its canonical hash;
    # cloud timing/validity records are never passed to the learning function.
    return dict(schema_version='cloud-primary-recovery-terminal-disposition/v1',study_id=c1['primary_study_id'],
                design_sha256=c1['primary_design_sha256'],sequence_ended=True,disposition='complete',
                reason='Original execution remains stopped; exact frozen E1 recovery completed and was independently re-audited.',
                missing_primary_evidence=[],status_file_bindings=statuses,recovery_id=c1['recovery_id'],
                base_admission_sha256=c1['base_admission_sha256'],activation_attachment_sha256=digest(args.attachment),
                activation_freeze_sha256=digest(args.attachment_freeze),recovery_admission_sha256=c1['recovery_admission_sha256'],
                recovery_freeze_sha256=c1['recovery_freeze_sha256'],recovery_manifest_sha256=digest(manifest_path),
                execution_selection_sha256=digest(selection_path),recovered_report_canonical_sha256=sha(canonical(report)),
                recovery_report_code_sha256=digest(HERE/'cloud_recovery_report.py'),
                activation_adapter_sha256=digest(Path(__file__)),base_wrapper_sha256=digest(HERE/'old_screen_learning.py'))


def run(args):
    check_attachment(args)
    if getattr(args,'verify_only',False):
        return verify_completion(args)
    def checked_primary(admission,cloud_root,disposition_path=None):
        if (admission.get('analysis_id')!=read(args.attachment)['analysis_id'] or
            Path(cloud_root).resolve()!=Path(args.cloud_root).resolve() or disposition_path is not None):
            raise old.AdmissionError('recovery_activation_call_identity_mismatch')
        return copy.deepcopy(verify_completion(args))
    # The original wrapper performs source/eligibility checks, owns the one-shot
    # reservation and calls the unchanged run_learning. Only its administrative
    # primary-completion check is temporarily replaced by the audited E1 branch.
    with patch.object(old,'check_primary',side_effect=checked_primary):
        return old.run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attachment',type=Path,default=HERE/'old_screen_h4_recovery_activation_C1.json')
    p.add_argument('--attachment-freeze',type=Path,default=HERE/'old_screen_h4_recovery_activation_C1_freeze.json')
    p.add_argument('--recovery-admission',type=Path,default=HERE/'cloud_reproduction_recovery_E1.json')
    p.add_argument('--recovery-freeze',type=Path,default=HERE/'cloud_reproduction_recovery_E1_freeze.json')
    p.add_argument('--recovery-root',type=Path,default=HERE/'results/cloud_reproduction_20261028_recovery1')
    p.add_argument('--admission',type=Path,default=HERE/'old_screen_h4_admission.json')
    p.add_argument('--freeze',type=Path,default=HERE/'old_screen_h4_admission_freeze.json')
    p.add_argument('--old-root',type=Path,default=HERE/'results/resume_20261028_attempt2')
    p.add_argument('--cloud-root',type=Path,default=HERE/'results/cloud_reproduction_20261028_r1')
    p.add_argument('--output',type=Path,default=HERE/'results/old_screen_h4_20261028_r1')
    p.add_argument('--design',type=Path,default=HERE/'research_design.json')
    p.add_argument('--primary-disposition',type=Path)
    p.add_argument('--verify-only',action='store_true',help='Audit recovery completion without reserving or fitting.')
    args = p.parse_args()
    print(json.dumps(run(args),indent=2))


if __name__=='__main__':
    main()
