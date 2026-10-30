"""A5 fail-closed diagnostic admission and exact-cell eligibility; no GPU calls."""
import json
from pathlib import Path
from guard_audit import audit_row,read_guard,resolve_log,completed_session
from study_common import canonical,execution_errors,read_jsonl,sha
from trace_audit import audit_trace


BAD="a128_r4_l1_np1_s2_8192"


def init_errors(result):
    errors=[]
    if result.get("header_instrumentation") is not True: errors.append("instrumentation_role")
    kernels=result.get("precompile_kernels",[])
    uploads=result.get("structural_features",{}).get("numAxisUploads")
    if not isinstance(uploads,int) or uploads<1 or len(kernels)!=uploads: errors.append("kernel_coverage")
    if result.get("descriptor_validation",{}).get("all_match") is not True: errors.append("canonical_validation")
    counters=result.get("init_trace",{}).get("counters",{})
    if counters.get("nvrtc_compile")!=uploads or counters.get("module_load")!=uploads or counters.get("driver_kernel_launch")!=0:
        errors.append("initialization_counters")
    if any(k.get("compiler_options_match") is not True or k.get("actual_compiler_options")!=k.get("compiler_input",{}).get("options") for k in kernels):
        errors.append("actual_compiler_options")
    for kernel in kernels:
        source=kernel.get("source_sha256")
        if not isinstance(source,str) or len(source)!=64 or any(c not in "0123456789abcdef" for c in source):
            errors.append("missing_source_hash")
    errors.extend(audit_trace(result.get("init_trace",{}))["errors"])
    return errors


def execution_invalid(row):
    """Narrow single-forward CUDA fault, after validated init, nondefault only."""
    if row.get("candidate_id") in ("default","default_replica"): return False
    result=row.get("result",{})
    if result.get("failure_phase")!="single_forward_validation": return False
    if result.get("single_forward_input_unchanged") is False or result.get("input_unchanged") is False: return False
    message=str(result.get("error","")).lower()
    if not message.startswith("synchronize one-forward validation:") or "illegal memory access" not in message: return False
    hard=("workload_mismatch","requested_configuration_mismatch","input_seed_mismatch","GPU_UUID_mismatch","protocol_exception:")
    if any(e.startswith(hard) for e in row.get("validation_errors",[])): return False
    return not init_errors(result)


def evidence_row(row,design,root):
    if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"):
        raise ValueError("A5_record_hash_mismatch")
    guard=audit_row(row,design["scope"]["device_uuid"],root)
    if not guard["ok"]: raise ValueError("A5_guard_unclean:"+str(guard))


def admit_diagnostic(root,design,design_path,expected_inst_sha):
    """Must be explicitly invoked after the frozen four outcomes exist."""
    root=Path(root).resolve(strict=True)
    design_path=Path(design_path)
    addendum=design_path.with_name("research_addendum_A5.json")
    frozen=json.loads(design_path.with_name("research_addendum_A5_freeze.json").read_text(encoding="utf-8"))
    if frozen["addendum_sha256"]!=sha(addendum.read_bytes()) or frozen["original_design_sha256"]!=sha(design_path.read_bytes()):
        raise ValueError("A5_freeze_mismatch")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("study_id")!=design["study_id"] or manifest.get("executable_sha256")!=expected_inst_sha:
        raise ValueError("A5_runtime_identity_mismatch")
    rows=read_jsonl(root/"candidate_diagnostic"/"measurements.jsonl")
    expected=[("instrumented","default"),("plain","default"),("instrumented",BAD),("plain",BAD)]
    if [(r.get("workload",{}).get("binary_role"),r.get("candidate_id")) for r in rows]!=expected:
        raise ValueError("A5_exact_four_process_matrix_required")
    pairs={}
    for row in rows:
        evidence_row(row,design,root)
        r=row.get("result",{})
        w=row["workload"]
        expected_values=dict(n=16384,batch=1,graph_repeats=16,seed=2235648248,pattern="zero",buffer_bytes=131072,
                             allocated_primary_buffers_bytes=262144,timing_mode="graph",placement="outofplace")
        if any(r.get(k)!=v for k,v in expected_values.items()) or row.get("input_seed")!=2235648248:
            raise ValueError("A5_exact_request_seed_or_capacity_mismatch")
        if any(w.get(k)!=v for k,v in dict(samples=1,warmup=0,warmup_ms=0,pattern="zero",graph_repeats=16).items()):
            raise ValueError("A5_exact_diagnostic_settings_mismatch")
        role=w["binary_role"]
        if row["candidate_id"]=="default":
            if row["status"]!="ok" or execution_errors(r,w,1,instrumented=role!="plain"):
                raise ValueError("A5_default_control_failed")
        else:
            if r.get("failure_phase")!="single_forward_validation" or "illegal memory access" not in str(r.get("error","")).lower():
                raise ValueError("A5_matching_single_forward_fault_not_observed")
            if role=="instrumented" and not execution_invalid(row): raise ValueError("A5_instrumented_init_integrity_not_established")
        pairs[row["candidate_id"],role]=r
    for cid in ("default",BAD):
        plain=pairs[cid,"plain"]
        inst=pairs[cid,"instrumented"]
        if plain.get("requested_config")!=inst.get("requested_config"): raise ValueError("A5_requested_config_disagreement")
        a=sorted(plain.get("postinit_source_kernels",[]),key=lambda k:(k["axis"],k["upload"]))
        b=sorted(inst.get("precompile_kernels",[]),key=lambda k:(k["axis"],k["upload"]))
        if not a or len(a)!=len(b): raise ValueError("A5_kernel_coverage_disagreement")
        for pa,ib in zip(a,b):
            if any(pa.get(k)!=ib.get(k) for k in ("axis","upload","canonical_descriptor","source_sha256")):
                raise ValueError("A5_source_or_canonical_disagreement")
            expected=pa.get("expected_compiler_input",{})
            observed=ib.get("compiler_input",{})
            for field in ("source_sha256","function","options","original_compile_header_sha256"):
                if expected.get(field)!=observed.get(field) or expected.get(field) is None:
                    raise ValueError("A5_compiler_inputs_disagree")
            if expected.get("options")!=ib.get("actual_compiler_options"):
                raise ValueError("A5_actual_options_disagree")
            if pa.get("snapshot_stage")!="post_initialize_retained_source" or pa.get("actual_compiler_options") is not None:
                raise ValueError("A5_plain_observation_limit_violated")
    for row in rows:
        if not completed_session(resolve_log(row,root),row["guard_id"]):
            raise ValueError("A5_diagnostic_guard_session_not_completed")
    return dict(branch="matching_failure",A5_sha256=sha(addendum.read_bytes()),diagnostic_root=str(root),
                diagnostic_record_hashes=[r["record_sha256"] for r in rows],diagnostic_manifest_sha256=sha((root/"candidate_diagnostic"/"diagnostic_manifest.json").read_bytes()),
                executable_sha256=expected_inst_sha,attribution="Execution-invalid under this harness/task; matching failures do not prove an upstream bug or exclude a shared harness issue.")


def admit_decision(path,proof,design,semantic_root):
    path=Path(path).resolve(strict=True)
    decision=json.loads(path.read_text(encoding="utf-8"))
    if (decision.get("enabled") is not True or decision.get("decision")!="enable_candidate_local_execution_invalid_classification"
            or decision.get("study_id")!=design["study_id"] or decision.get("addendum_sha256")!=proof["A5_sha256"]):
        raise ValueError("A5_explicit_review_decision_not_enabled_or_mismatched")
    hashes=decision.get("independent_evidence_audit",{}).get("diagnostic_record_sha256",{})
    if set(hashes.values())!=set(proof["diagnostic_record_hashes"]): raise ValueError("A5_decision_diagnostic_hashes_mismatch")
    cells=decision.get("exact_semantic_invalid_cells",[])
    if len(cells)!=1 or cells[0].get("workload_id")!="n16384_b1" or cells[0].get("candidate_id")!=BAD:
        raise ValueError("A5_decision_original_cell_mismatch")
    original=cells[0]["source_record_sha256"]
    rows=read_jsonl(Path(semantic_root)/"semantics"/"measurements.jsonl")
    source=next((r for r in rows if r.get("record_sha256")==original),None)
    if source is None or not execution_invalid(source): raise ValueError("A5_original_fault_missing_or_not_classifiable")
    evidence_row(source,design,semantic_root)
    semantic_session_admission(rows,semantic_root,original)
    return dict(proof,decision_file=str(path),decision_sha256=sha(path.read_bytes()),original_semantic_fault_record_sha256=original)


def semantic_invalid_cells(rows):
    cells={}
    for row in rows:
        if row["status"]=="ok": continue
        key=(row["workload_id"],row["candidate_id"])
        entry=cells.setdefault(key,dict(workload_id=key[0],candidate_id=key[1],source_record_hashes=[],patterns=[],
                                        reason="fixed_semantic_probe_failed_exact_cell_ineligible"))
        entry["source_record_hashes"].append(row["record_sha256"])
        entry["patterns"].append(row["workload"].get("pattern"))
    return [cells[k] for k in sorted(cells)]


def semantic_session_admission(rows,root,allowed_fault_hash):
    """Only the exact reviewed controller exception may replace global completion."""
    fault=next((r for r in rows if r.get("record_sha256")==allowed_fault_hash),None)
    if fault is None: raise ValueError("A5_original_semantic_fault_missing")
    expected_error="StudyStop: fatal_execution_protocol_or_baseline_failure:n16384_b1:"+BAD
    expected_incident="guard_or_target_exception:StudyStop:fatal_execution_protocol_or_baseline_failure:n16384_b1:"+BAD
    sessions={(r["guard_id"],r["guard_log"]) for r in rows}
    for guard_id,log in sessions:
        path=resolve_log(dict(guard_log=log),root)
        if completed_session(path,guard_id): continue
        if guard_id!=fault["guard_id"]: raise ValueError("A5_unreviewed_incomplete_semantic_session")
        events=[e for e in read_guard(path) if e.get("guard_id")==guard_id]
        aborts=[e for e in events if e.get("event")=="guard_abort"]
        incidents=[e for e in events if e.get("event")=="incident"]
        if len(aborts)!=1 or aborts[0].get("error")!=expected_error:
            raise ValueError("A5_abort_not_exact_reviewed_target_exception")
        if len(incidents)!=1 or incidents[0].get("reasons")!=[expected_incident] or incidents[0].get("active_run") is not None:
            raise ValueError("A5_real_or_unreviewed_guard_incident")
        for event in events:
            if event.get("event")=="sample" and (event.get("ok") is not True or event.get("foreign_compute_pids") or event.get("violation_reasons") or event.get("monitor_error") is not None):
                raise ValueError("A5_unclean_sample_in_original_semantic_session")
