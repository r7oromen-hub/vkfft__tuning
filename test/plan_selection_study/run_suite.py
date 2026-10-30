#!/usr/bin/env python3
"""Guard-required, staged independent plan-selection study. No legacy data reuse."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from guard_audit import audit_row,completed_session,resolve_log
from cost_audit import audit_pair
from selection_analysis import freeze_screen,confirmation_summary,freeze_headline
from candidate_policy import admit_diagnostic,admit_decision,execution_invalid,evidence_row,semantic_invalid_cells,semantic_session_admission,init_errors
from cost_recovery import admit_recovery,selected_cost_rows,modes as cost_modes,AFFECTED as COST_RECOVERY_PAIR
from study_common import (TUNING_FIELDS,append_json,block_candidates,block_quality,candidate_config,
                          canonical,execution_errors,input_seed,load_design,read_jsonl,record_key,
                          sha,utc,workload_order,write_json)


class StudyStop(RuntimeError):
    pass


def run_one(args,workload,candidate,phase,round_no,order):
    """Sole GPU subprocess entry point; gpu_guard wraps this exact signature."""
    cid,config=candidate
    mode=workload.get("study_mode","execute")
    samples=workload.get("samples",args.design_data["measurement"]["screen_samples"])
    seed_phase=workload.get("seed_phase",phase)
    seed=workload.get("input_seed_override",input_seed(args.design_data,seed_phase,workload["id"],round_no))
    command=[str(Path(args.exe).resolve()),"--n",str(workload["n"]),"--batch",str(workload["batch"]),
             "--seed",str(seed),
             "--timing-mode","graph","--placement","outofplace","--graph-repeats",str(workload.get("graph_repeats",1)),
             "--warmup",str(workload.get("warmup",5)),"--warmup-ms",str(workload.get("warmup_ms",args.design_data["measurement"]["warmup_ms"])),
             "--repeats",str(samples),"--study-mode",mode]
    if workload.get("pattern"):
        command += ["--pattern",workload["pattern"]]
    if mode!="execute":
        command += ["--candidate-file",workload["candidate_file"],"--selected-id",workload["selected_id"],
                    "--init-rounds","1","--candidate-order","forward"]
    for field,value in config.items():
        command += ["--"+field.replace("_","-"),str(value)]
    row=dict(schema_version="plan-selection-process/v1",study_id=args.design_data["study_id"],
             phase=phase,workload_id=workload["id"],workload=workload,candidate_id=cid,config=config,
             round=round_no,order=order,block_id=workload.get("block_id"),block_attempt=workload.get("block_attempt",0),
             position_in_block=workload.get("position_in_block"),input_seed=seed,
             command=command,started_utc=utc())
    started=time.perf_counter()
    try:
        if getattr(args,"expected_executable_sha256",None)!=sha(Path(args.exe).read_bytes()):
            raise ValueError("executable bytes changed after runtime freeze")
        process=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout,check=False)
        row.update(returncode=process.returncode,stderr=process.stderr[-20000:])
        if workload.get("diagnostic_capture"):
            row.update(stdout=process.stdout,stderr=process.stderr)
        objects=[]
        for line in process.stdout.splitlines():
            try:
                value=json.loads(line)
                if isinstance(value,dict):
                    objects.append(value)
            except json.JSONDecodeError:
                pass
        if len(objects)!=1:
            raise ValueError(f"expected one harness JSON object, received {len(objects)}")
        result=objects[0]
        row["result"]=result
        if mode=="execute":
            errors=execution_errors(result,workload,samples,instrumented=workload.get("binary_role")!="plain")
            expected_request={"makeForwardPlanOnly":1}
            names=dict(aim_threads="aimThreads",register_boost="registerBoost",use_lut="useLUT",
                       register_boost_nonpow2="registerBoostNonPow2",coalesced_memory="coalescedMemory",
                       use_lut_4step="useLUT_4step",swap_to_2stage="swapTo2Stage4Step",swap_to_3stage="swapTo3Stage4Step")
            expected_request.update({names[field]:value for field,value in config.items()})
            if result.get("requested_config")!=expected_request:
                errors.append("requested_configuration_mismatch")
            if result.get("seed")!=row["input_seed"]:
                errors.append("input_seed_mismatch")
        else:
            errors=[]
            if result.get("record_type")!="context_profile_summary" or result.get("n")!=workload["n"] or result.get("batch")!=workload["batch"]:
                errors.append("cost_summary_schema_mismatch")
            if result.get("status")!="ok":
                errors.append("cost_pipeline_failed")
        if process.returncode:
            errors.append(f"harness_exit:{process.returncode}")
        if result.get("gpu",{}).get("uuid")!=args.design_data["scope"]["device_uuid"]:
            errors.append("GPU_UUID_mismatch")
        row.update(status="ok" if not errors else "failed",validation_errors=errors)
    except subprocess.TimeoutExpired as error:
        row.update(status="timeout",validation_errors=[f"process_timeout:{args.timeout}"])
        if workload.get("diagnostic_capture"):
            row.update(stdout=(error.stdout or b"").decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout,
                       stderr=(error.stderr or b"").decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr)
    except (OSError,ValueError,TypeError,KeyError) as error:
        row.update(status="failed",validation_errors=[f"protocol_exception:{type(error).__name__}:{error}"])
    row.update(ended_utc=utc(),wall_seconds=time.perf_counter()-started)
    return row


class Study:
    def __init__(self,args):
        self.args=args
        self.design=load_design(args.design)
        args.design_data=self.design
        self.output=args.output
        self.stage=args.stage
        self.stage_path=self.output/self.stage
        self.stage_path.mkdir(parents=True,exist_ok=True)
        status_before=self.stage_path/"status.json"
        self.previous_status_bytes=status_before.read_bytes() if status_before.exists() else None
        self.measurements=self.stage_path/"measurements.jsonl"
        self.rows=read_jsonl(self.measurements)
        self.by_key={record_key(r):r for r in self.rows}
        if len(self.by_key)!=len(self.rows):
            raise StudyStop("duplicate_process_keys")
        self.candidates={c["id"]:c for c in self.design["candidates"]}
        self.workloads={w["id"]:w for w in self.design["workloads"]}
        self.active=None
        self.expected=None
        self.guard_id=os.environ.get("VKFFT_STUDY_GUARD_ID")
        self.status("starting")
        if not self.guard_id or os.environ.get("VKFFT_STUDY_GUARD_UUID")!=self.design["scope"]["device_uuid"] or not os.environ.get("VKFFT_STUDY_GUARD_LOG"):
            raise StudyStop("GPU_guard_required; launch through gpu_guard.py")
        self.freeze_manifest()
        self.controller_snapshot()
        self.cost_recovery=None
        self.cost_pair_records={}
        recovery_path=getattr(args,"cost_recovery_admission",None)
        recovery_file=self.output/"cost_recovery_admission.json"
        if self.stage=="cost" and recovery_path is not None:
            admission=admit_recovery(recovery_path,args.design,self.output,json.loads((self.output/"manifest.json").read_text(encoding="utf-8")),
                                     self.guard_id,self.previous_status_bytes)
            admission.update(recovery_controller_sha256=self.controller_sha256,deployed_guard_sha256=sha(Path(__file__).with_name("gpu_guard.py").read_bytes()))
            self.frozen_file(recovery_file,admission)
            self.cost_recovery=admission
            self.cost_pair_records={(p["workload_id"],p["round"]):p for p in admission["preserved_pairs"]}
            archive=self.stage_path/"engineering_archive"
            for name,data in (("pair_audits_before_A6.json",admission["original_pair_audits_bytes"].encode("utf-8")),
                              ("status_before_A6.json",admission["original_status_bytes"].encode("utf-8")),
                              ("measurements_before_A6.jsonl",self.measurements.read_bytes()[:admission["original_prefix_bytes"]])):
                target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and target.read_bytes()!=data: raise StudyStop("A6_original_snapshot_changed:"+name)
                if not target.exists(): target.write_bytes(data)
            for source in (Path(recovery_path),Path(recovery_path).with_name("research_addendum_A6_freeze.json")):
                target=self.output/source.name
                if target.exists() and target.read_bytes()!=source.read_bytes(): raise StudyStop("A6_frozen_sidecar_changed")
                if not target.exists(): target.write_bytes(source.read_bytes())
            for pair in admission["preserved_pairs"]:
                self.frozen_file(self.stage_path/"pair_attempts"/f"{pair['workload_id']}_r{pair['round']}_a0.json",pair)
            self.frozen_file(self.stage_path/"pair_attempts"/"n16384_b2048_r1_a0.json",dict(workload_id="n16384_b2048",round=1,block_attempt=0,
                candidate_ids=["describe-select-cost"],record_hashes=admission["engineering_only_record_hashes"],engineering_only=True,
                quarantined_guard_run_id=admission["quarantined_guard_run_id"],quality=dict(ok=False,errors=["interrupted_pair_quarantined_partner"])))
            self.save_cost_selection()
        elif self.stage=="cost" and recovery_file.exists():
            raise StudyStop("A6_cost_recovery_requires_explicit_cli_admission_and_allows_only_one_guard_session")
        self.candidate_fault_admission=None
        self.semantic_invalid_cells={}
        admission_path=self.output/"candidate_fault_admission.json"
        diagnostic_root=getattr(args,"candidate_fault_diagnostics",None)
        decision_file=getattr(args,"candidate_fault_decision",None)
        if diagnostic_root is None and admission_path.exists():
            old_admission=json.loads(admission_path.read_text(encoding="utf-8"))
            diagnostic_root=old_admission["diagnostic_root"]
            decision_file=decision_file or old_admission["decision_file"]
        if diagnostic_root is not None:
            if decision_file is None: raise StudyStop("A5_explicit_independent_decision_required")
            admission=admit_diagnostic(diagnostic_root,self.design,args.design,args.expected_executable_sha256)
            admission=admit_decision(decision_file,admission,self.design,self.output)
            self.frozen_file(admission_path,admission)
            self.candidate_fault_admission=admission
        if getattr(args,"import_calibration_from",None):
            self.import_calibration(args.import_calibration_from)

    def controller_snapshot(self):
        files=("run_suite.py","study_common.py","selection_analysis.py","candidate_policy.py","cost_recovery.py","cost_audit.py","guard_audit.py","trace_audit.py","gpu_guard.py")
        hashes={name:sha(Path(__file__).with_name(name).read_bytes()) for name in files}
        for path in sorted(Path(__file__).parent.glob("research_addendum_*.json")):
            hashes[path.name]=sha(path.read_bytes())
        digest=sha(canonical(hashes))
        path=self.stage_path/"controller_snapshots"/(digest+".json")
        if not path.exists():
            write_json(path,dict(created_utc=utc(),controller_sha256=digest,files=hashes))
        self.controller_sha256=digest

    def status(self,state,reason=None,**extra):
        if self.stage=="cost":
            extra.update(raw_recorded_processes=len(self.rows),accepted_processes=2*len(getattr(self,"cost_pair_records",{})),
                         engineering_only_records=1 if getattr(self,"cost_recovery",None) else 0,
                         physical_processes_estimate=len(self.rows)+(1 if getattr(self,"cost_recovery",None) else 0))
        write_json(self.stage_path/"status.json",dict(study_id=self.design["study_id"],stage=self.stage,state=state,
            updated_utc=utc(),exit_reason=reason,expected_processes=self.expected,recorded_processes=len(self.rows),
            successful=sum(r["status"]=="ok" for r in self.rows),failed=sum(r["status"]!="ok" for r in self.rows),
            active=self.active,guard_id=self.guard_id,**extra))

    def freeze_manifest(self):
        executable=Path(self.args.exe).resolve(strict=True)
        manifest=dict(schema_version="plan-selection-runtime/v1",study_id=self.design["study_id"],
                      design_sha256=sha(self.args.design.read_bytes()),executable=str(executable),
                      executable_sha256=sha(executable.read_bytes()),gpu_uuid=self.design["scope"]["device_uuid"],
                      seed_namespace=self.design["measurement"]["seed_namespace"],source=self.design["source"])
        path=self.output/"manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=manifest:
            raise StudyStop("resume_manifest_mismatch")
        write_json(path,manifest)
        self.args.expected_executable_sha256=manifest["executable_sha256"]
        (self.output/"research_freeze.json").write_bytes(self.args.design.with_name("research_freeze.json").read_bytes())
        copied=self.output/"research_design.json"
        if copied.exists() and copied.read_bytes()!=self.args.design.read_bytes():
            raise StudyStop("copied_design_changed")
        copied.write_bytes(self.args.design.read_bytes())

    def require(self,stage):
        gate_path=self.output/stage/"gate.json"
        if not gate_path.exists():
            raise StudyStop("missing_prerequisite_gate:"+stage)
        gate=json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("passed") is not True:
            raise StudyStop("failed_prerequisite_gate:"+stage)
        if stage=="semantics" and gate.get("protocol_valid") is True and self.candidate_fault_admission:
            rows=read_jsonl(self.output/stage/"measurements.jsonl")
            if len(rows)!=self.design["preflight"]["semantic_probes"]["processes"]:
                raise StudyStop("semantic_protocol_coverage_not_complete")
            for row in rows:
                evidence_row(row,self.design,self.output)
                if row["status"]!="ok" and not self.allowed_candidate_failure(row):
                    raise StudyStop("semantic_failure_outside_frozen_A5_policy")
            semantic_session_admission(rows,self.output,self.candidate_fault_admission["original_semantic_fault_record_sha256"])
            return
        selection_name={"screen":"screen_selection.json","confirmation":"headline_selection.json"}.get(stage)
        digest_field={"screen":"selection_sha256","confirmation":"headline_selection_sha256"}.get(stage)
        if selection_name and sha((self.output/selection_name).read_bytes())!=gate.get(digest_field):
            raise StudyStop("prerequisite_selection_hash_changed:"+stage)
        if stage in ("screen","confirmation","headline"):
            # Accepted closed run intervals survive unrelated later session aborts.
            selected=self.accepted_blocks(stage,stage,3 if stage=="confirmation" else 1) if stage!="headline" else None
            if selected is not None: return
        for session in gate.get("guard_sessions",[]):
            row=dict(guard_log=session["guard_log"])
            if not completed_session(resolve_log(row,self.output),session["guard_id"]):
                raise StudyStop("prerequisite_guard_not_complete:"+stage)

    def require_main(self):
        addendum=self.args.design.with_name("research_addendum_A3.json")
        frozen=json.loads(self.args.design.with_name("research_addendum_A3_freeze.json").read_text(encoding="utf-8"))
        if frozen["addendum_sha256"]!=sha(addendum.read_bytes()) or frozen["original_design_sha256"]!=sha(self.args.design.read_bytes()):
            raise StudyStop("A3_selection_interpretation_freeze_mismatch")
        for stage in ("health","semantics","equivalence"):
            self.require(stage)
        semantic_gate=json.loads((self.output/"semantics"/"gate.json").read_text(encoding="utf-8"))
        self.semantic_invalid_cells={(x["workload_id"],x["candidate_id"]):x for x in semantic_gate.get("semantic_invalid_cells",[])}

    def allowed_candidate_failure(self,row):
        if row.get("candidate_id") in ("default","default_replica"): return False
        result=row.get("result",{})
        failure=result.get("failure_phase")
        errors=row.get("validation_errors",[])
        hard=("workload_mismatch","requested_configuration_mismatch","input_seed_mismatch","GPU_UUID_mismatch","protocol_exception:")
        if any(e.startswith(hard) for e in errors): return False
        numerical=bool(errors) and all(e.startswith(("numerical_failure:","harness_status:","harness_exit:")) for e in errors)
        return bool((failure=="initialize" and str(result.get("error","")).startswith("initializeVkFFT failed with code ")) or numerical or
                    (failure=="single_forward_validation" and result.get("single_forward_input_unchanged") is True
                     and result.get("error")=="Single out-of-place forward failed immutable-input or numerical validation"
                     and result.get("single_forward_correctness",{}).get("passed") is False and not init_errors(result)) or
                    (self.candidate_fault_admission and execution_invalid(row)))

    def frozen_file(self,path,value):
        """Scientific selections are immutable, including on resume."""
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8"))!=value:
                raise StudyStop("frozen_selection_changed:"+str(path))
        else:
            write_json(path,value)
        return value

    def import_calibration(self,source):
        """Reuse clean closed calibration runs; a later session abort is not retroactive."""
        source=source.resolve(strict=True)
        original=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
        current=json.loads((self.output/"manifest.json").read_text(encoding="utf-8"))
        if any(original.get(field)!=current.get(field) for field in ("study_id","design_sha256","executable_sha256","gpu_uuid")):
            raise StudyStop("calibration_import_scientific_identity_mismatch")
        raw=(source/"graph_repeats.json").read_bytes()
        frozen=json.loads(raw)
        available=[]
        for path in source.glob("*/measurements.jsonl"):
            available.extend(read_jsonl(path))
        by_hash={row.get("record_sha256"):row for row in available}
        evidence=[]
        for wid,entry in frozen.items():
            row=by_hash.get(entry["source_record_sha256"])
            if row is None or row["phase"]!="calibration" or row["workload_id"]!=wid or row["status"]!="ok":
                raise StudyStop("missing_clean_imported_calibration:"+wid)
            unhashed={k:v for k,v in row.items() if k!="record_sha256"}
            audit=audit_row(row,current["gpu_uuid"],source)
            if sha(canonical(unhashed))!=row["record_sha256"] or not audit["ok"] or row["result"]["graph_repeats"]!=entry["graph_repeats"]:
                raise StudyStop("calibration_import_provenance_failed:"+wid)
            evidence.append(dict(workload_id=wid,record_sha256=row["record_sha256"],guard_audit=audit,
                                 guard_log=row["guard_log"],graph_repeats=entry["graph_repeats"]))
        target=self.output/"graph_repeats.json"
        if target.exists() and target.read_bytes()!=raw:
            raise StudyStop("calibration_import_would_change_frozen_R")
        target.write_bytes(raw)
        write_json(self.output/"calibration_import_provenance.json",dict(source=str(source),
            source_manifest_sha256=sha((source/"manifest.json").read_bytes()),graph_repeats_sha256=sha(raw),
            created_utc=utc(),source_runs=evidence,source_global_guard_success_required=False,
            rule="Only earlier closed clean calibration run intervals are imported; all health pairs are newly executed. No old execution outcome enters study selection."))

    def attempt(self,w,cid,phase,round_no):
        k=(phase,w["id"],cid,round_no,w.get("block_attempt",0))
        if k in self.by_key:
            existing=self.by_key[k]
            if (sha(canonical({name:value for name,value in existing.items() if name!="record_sha256"}))!=existing.get("record_sha256")
                    or not audit_row(existing,self.design["scope"]["device_uuid"],self.output)["ok"]):
                raise StudyStop("retained_process_provenance_failed")
            if existing["status"]!="ok" and not self.allowed_candidate_failure(existing):
                raise StudyStop("retained_failure_requires_admitted_classification:"+w["id"]+":"+cid)
            return existing
        self.active=dict(workload_id=w["id"],candidate_id=cid,phase=phase,round=round_no,block_id=w.get("block_id"))
        self.status("running")
        config=candidate_config(self.candidates[cid]) if cid in self.candidates else {}
        original_exe,original_sha=self.args.exe,self.args.expected_executable_sha256
        try:
            if w.get("binary_role")=="plain":
                self.args.exe=str(Path(self.args.plain_exe).resolve(strict=True))
                self.args.expected_executable_sha256=self.plain_sha256
            row=run_one(self.args,w,(cid,config),phase,round_no,len(self.rows))
        finally:
            self.args.exe,self.args.expected_executable_sha256=original_exe,original_sha
        audit=audit_row(row,self.design["scope"]["device_uuid"],self.output)
        if not audit["ok"]:
            append_json(self.stage_path/"unaccepted_records.jsonl",dict(row=row,guard_audit=audit))
            raise StudyStop("guard_provenance_rejected:"+";".join(audit["errors"]))
        row["guard_audit"]=audit
        row["controller_sha256"]=self.controller_sha256
        if self.candidate_fault_admission and row["status"]!="ok" and execution_invalid(row):
            row["candidate_validity_class"]="execution_invalid_single_forward_A5"
        exclusion=self.semantic_invalid_cells.get((w["id"],cid))
        if phase=="screen" and exclusion:
            row.update(selection_eligible=False,preflight_ineligible=exclusion)
        row["record_sha256"]=sha(canonical(row))
        append_json(self.measurements,row)
        self.rows.append(row)
        self.by_key[k]=row
        self.active=None
        self.status("running")
        print(f"{self.stage} {len(self.rows)}/{self.expected or '?'} {phase} {w['id']} {cid} round={round_no} {row['status']}",flush=True)
        if w.get("study_mode","execute")=="execute" and row["status"]!="ok":
            # Protocol damage never masquerades as a legally rejected candidate.
            if not self.allowed_candidate_failure(row):
                raise StudyStop("fatal_execution_protocol_or_baseline_failure:"+w["id"]+":"+cid)
        return row

    def calibrate(self,workloads):
        path=self.output/"graph_repeats.json"
        frozen=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        probes=self.design["measurement"]["repeats"]
        for original in workloads:
            wid=original["id"]
            if wid in frozen:
                continue
            repeats=1
            for probe in range(probes["max_probes"]):
                w=dict(original,graph_repeats=repeats,samples=self.design["measurement"]["screen_samples"],
                       block_id=f"calibration/{wid}/{probe}")
                row=self.attempt(w,"default","calibration",probe)
                if row["status"]!="ok":
                    raise StudyStop("calibration_failed:"+wid)
                raw=row["result"].get("raw_interval_ms_median")
                if not isinstance(raw,(int,float)) or not math.isfinite(raw) or raw<=0:
                    raise StudyStop("invalid_calibration_raw_interval:"+wid)
                reached=raw>=probes["min_raw_ms"]
                if reached or repeats==probes["cap"] or probe==probes["max_probes"]-1:
                    frozen[wid]=dict(graph_repeats=repeats,raw_ms=raw,target_met=reached,capped=repeats==probes["cap"],
                                     probes=probe+1,source_record_sha256=row["record_sha256"])
                    write_json(path,frozen)
                    break
                repeats=min(probes["cap"],max(repeats+1,math.ceil(1.1*probes["min_raw_ms"]*repeats/raw)))
        return frozen

    def block(self,original,phase,round_no,cids,repeats,samples):
        block_id=f"{phase}/{original['id']}/r{round_no}"
        prior=[r for r in self.rows if r["phase"]==phase and r["workload_id"]==original["id"] and r["round"]==round_no]
        attempt=max([r.get("block_attempt",0) for r in prior],default=0)
        existing=[self.by_key.get((phase,original["id"],cid,round_no,attempt)) for cid in cids]
        interrupted=any(existing) and (not all(existing) or not block_quality([r for r in existing if r])["ok"])
        if interrupted:
            requested=getattr(self.args,"retry_block",None)
            if phase!="screen" or requested!=block_id:
                raise StudyStop("whole_block_new_attempt_requires_explicit_retry_and_fresh_health:"+block_id)
            retries={ (r["workload_id"],r["round"],r.get("block_attempt",0)) for r in self.rows if r["phase"]=="screen" and r.get("block_attempt",0)>0 }
            if len(retries)>=self.design["budgets"]["screen_block_retries_max"]:
                raise StudyStop("screen_whole_block_retry_cap_exhausted")
            health=json.loads((self.output/"health"/"gate.json").read_text(encoding="utf-8"))
            if not health.get("passed") or health["created_utc"]<=max(r.get("ended_utc",r.get("started_utc","")) for r in prior):
                raise StudyStop("fresh_health_required_after_interrupted_block")
            attempt+=1
        rows=[]
        for position,cid in enumerate(cids):
            w=dict(original,graph_repeats=repeats,samples=samples,block_id=block_id,position_in_block=position,block_attempt=attempt)
            rows.append(self.attempt(w,cid,phase,round_no))
        quality=block_quality(rows)
        block_record=dict(block_id=block_id,candidate_ids=cids,quality=quality,
                          block_attempt=attempt,record_hashes=[r["record_sha256"] for r in rows])
        write_json(self.stage_path/"block_attempts"/(block_id.replace("/","__")+f"__a{attempt}.json"),block_record)
        write_json(self.stage_path/"blocks"/(block_id.replace("/","__")+".json"),block_record)
        if not quality["ok"]:
            raise StudyStop("whole_block_quality_failed:"+block_id+":"+";".join(quality["errors"]))
        return rows,quality

    def gate(self,passed,**extra):
        sessions={(r["guard_id"],r["guard_log"]) for r in self.rows}
        value=dict(passed=passed,stage=self.stage,created_utc=utc(),record_count=len(self.rows),
                   guard_sessions=[dict(guard_id=g,guard_log=p) for g,p in sorted(sessions)],
                   guard_completion="Check completed guard sessions before using this gate in the next stage")
        value.update(extra)
        path=self.stage_path/"gate.json"
        if path.exists():
            original=json.loads(path.read_text(encoding="utf-8"))
            if {k:v for k,v in original.items() if k!="created_utc"}=={k:v for k,v in value.items() if k!="created_utc"}:
                return original # A no-op resume must never manufacture a fresh health gate.
        write_json(path,value)
        return value

    def health(self):
        health=self.design["guard"]["health_gate"]
        workloads=[self.workloads[wid] for wid in health["workloads"]]
        self.expected=len(workloads)*health["paired_process_rounds"]*2+len(workloads)*self.design["measurement"]["repeats"]["max_probes"]
        frozen=self.calibrate(workloads)
        self.expected=sum(r["phase"]=="calibration" for r in self.rows)+len(workloads)*health["paired_process_rounds"]*2
        ratios={w["id"]:[] for w in workloads}
        for round_no in range(health["paired_process_rounds"]):
            for w in workloads:
                cids=block_candidates(self.design,w["id"],"health",round_no,["default","default_replica"])
                _,quality=self.block(w,"health",round_no,cids,frozen[w["id"]]["graph_repeats"],self.design["measurement"]["confirmation_samples"])
                ratios[w["id"]].append(quality["default_over_replica"])
        means={wid:math.exp(statistics.mean(map(math.log,values))) for wid,values in ratios.items()}
        passed=all(.98<=x<=1.02 for x in means.values())
        self.gate(passed,paired_ratios=ratios,ratio_gmeans=means)
        if not passed:
            raise StudyStop("health_geometric_mean_outside_2pct")

    def semantics(self):
        probes=self.design["preflight"]["semantic_probes"]
        self.expected=probes["processes"]
        failures=[]
        for group in probes["groups"]:
            n,b,cids=group["n"],group["batch"],group["candidates"]
            for pattern in probes["patterns"]:
                for cid in cids:
                    w=dict(id=f"n{n}_b{b}",n=n,batch=b,input_bytes=n*b*8,graph_repeats=probes["graph_repeats"],samples=probes["samples"],
                           warmup=0,warmup_ms=probes["warmup_ms"],pattern=pattern,block_id=f"semantic/{n}/{b}/{pattern}")
                    row=self.attempt(w,cid,"semantic/"+pattern,0)
                    if row["status"]!="ok":
                        failures.append(dict(workload_id=w["id"],candidate_id=cid,pattern=pattern,errors=row["validation_errors"],
                            source_record_sha256=row["record_sha256"],candidate_local_classified=self.allowed_candidate_failure(row)))
        protocol_valid=not failures or bool(self.candidate_fault_admission and all(x["candidate_local_classified"] for x in failures))
        cells=semantic_invalid_cells(self.rows)
        self.gate(protocol_valid,protocol_valid=protocol_valid,all_candidates_passed=not failures,failures=failures,
                  candidate_validation_failures=failures,semantic_invalid_cells=cells,
                  candidate_fault_admission_sha256=sha((self.output/"candidate_fault_admission.json").read_bytes()) if self.candidate_fault_admission else None)
        if not protocol_valid:
            raise StudyStop("semantic_probe_failures")
        if failures:
            return dict(stage_checks_passed=False,exit_reason="semantic_protocol_complete_with_candidate_invalid_cells")

    def candidate_file(self,wid,round_no,phase):
        order=sorted(self.candidates)
        random.Random(input_seed(self.design,phase+"/candidate_order",wid,round_no)).shuffle(order)
        path=self.stage_path/"candidate_orders"/f"{wid}_r{round_no}.tsv"
        path.parent.mkdir(parents=True,exist_ok=True)
        columns=("candidate_id",)+TUNING_FIELDS+("disable_reorder_four_step",)
        lines=["\t".join(columns)]
        lines += ["\t".join([cid]+[str(self.candidates[cid].get(field,0)) for field in columns[1:]]) for cid in order]
        value=("\n".join(lines)+"\n").encode()
        if path.exists() and path.read_bytes()!=value:
            raise StudyStop("candidate_order_file_changed")
        path.write_bytes(value)
        return path,order

    def equivalence(self):
        if not self.args.plain_exe:
            raise StudyStop("A1_requires_plain_exe")
        addendum_path=self.args.design.with_name("research_addendum_A1.json")
        addendum=json.loads(addendum_path.read_text(encoding="utf-8"))
        if addendum["original_design_sha256"]!=sha(self.args.design.read_bytes()):
            raise StudyStop("A1_design_hash_mismatch")
        self.plain_sha256=sha(Path(self.args.plain_exe).read_bytes())
        write_json(self.stage_path/"manifest.json",dict(addendum_sha256=sha(addendum_path.read_bytes()),
            main_manifest_sha256=sha((self.output/"manifest.json").read_bytes()),plain_executable=str(Path(self.args.plain_exe).resolve()),
            plain_executable_sha256=self.plain_sha256,instrumented_executable_sha256=self.args.expected_executable_sha256))
        (self.stage_path/"research_addendum_A1.json").write_bytes(addendum_path.read_bytes())
        self.expected=addendum["processes_max"]
        pairs=[]
        for case in addendum["cases"]:
            for cid in case["candidates"]:
                results={}
                for role in ("plain","instrumented"):
                    w=dict(id=case["workload_id"],n=case["n"],batch=case["batch"],input_bytes=8*case["n"]*case["batch"],
                           graph_repeats=addendum["graph_repeats"],samples=addendum["samples"],warmup=0,warmup_ms=0,
                           binary_role=role,seed_phase="instrumentation_equivalence",pattern=addendum["pattern"])
                    results[role]=self.attempt(w,cid,"equivalence/"+role,0)
                errors=[]
                if any(row["status"]!="ok" for row in results.values()):
                    errors.append("A1_execution_or_numerical_failure")
                plain=results["plain"].get("result",{}).get("postinit_source_kernels",[])
                inst=results["instrumented"].get("result",{}).get("precompile_kernels",[])
                a=sorted(plain,key=lambda k:(k["axis"],k["upload"]))
                b=sorted(inst,key=lambda k:(k["axis"],k["upload"]))
                if not a or len(a)!=len(b):
                    errors.append("A1_kernel_coverage_mismatch")
                for pa,ib in zip(a,b):
                    if pa["canonical_descriptor"]!=ib["canonical_descriptor"] or pa["source_sha256"]!=ib["source_sha256"]:
                        errors.append("A1_canonical_or_source_mismatch")
                    options=pa["expected_compiler_input"].get("options")
                    if options!=ib["compiler_input"].get("options") or options!=ib.get("actual_compiler_options"):
                        errors.append("A1_compiler_options_mismatch")
                    if pa.get("actual_compiler_options") is not None:
                        errors.append("plain_actual_options_must_remain_unobserved")
                    original_header=pa["expected_compiler_input"].get("original_compile_header_sha256")
                    if not isinstance(original_header,str) or len(original_header)!=64 or original_header!=ib["compiler_input"].get("original_compile_header_sha256"):
                        errors.append("A1_original_compile_header_mismatch")
                    if pa.get("snapshot_stage")!="post_initialize_retained_source":
                        errors.append("A1_plain_snapshot_stage_mismatch")
                pairs.append(dict(workload_id=case["workload_id"],candidate_id=cid,ok=not errors,errors=errors))
        passed=all(pair["ok"] for pair in pairs)
        self.gate(passed,pairs=pairs,excluded_from_performance_analysis=True)
        if not passed:
            raise StudyStop("A1_instrumentation_equivalence_failed")

    def cost_smoke(self):
        self.require("health")
        self.require("semantics")
        smoke=self.design["preflight"]["cost_smoke"]
        self.expected=smoke["pipeline_processes"]
        qualities=[]
        for wid in smoke["workloads"]:
            path,order=self.candidate_file(wid,0,"cost_smoke")
            results={}
            failed_processes=[]
            for mode in ("full-search-cost","describe-select-cost"):
                w=dict(self.workloads[wid],graph_repeats=1,samples=1,warmup=0,warmup_ms=0,study_mode=mode,
                       candidate_file=str(path.resolve()),selected_id=smoke["known_selected_candidate"],block_id=f"cost_smoke/{wid}")
                row=self.attempt(w,mode,"cost_smoke",0)
                results[mode]=row.get("result",{})
                if row["status"]!="ok":
                    failed_processes.append(mode)
            quality=audit_pair(results["full-search-cost"],results["describe-select-cost"],order,smoke["known_selected_candidate"])
            if failed_processes:
                quality["ok"]=False
                quality["errors"].extend("process_status_failed:"+mode for mode in failed_processes)
            qualities.append(dict(workload_id=wid,**quality))
        passed=all(q["ok"] for q in qualities)
        self.gate(passed,H3_feasibility="passed_engineering_smoke" if passed else "unresolved_or_limited",pairs=qualities)
        return dict(stage_checks_passed=passed,exit_reason="cost_smoke_passed" if passed else "H3_unresolved_full_H1_H2_not_replaced")

    def accepted_blocks(self,stage,phase,rounds=1):
        base=self.output/stage
        rows=read_jsonl(base/"measurements.jsonl")
        hashes={r["record_sha256"]:r for r in rows}
        blocks={}
        for path in sorted((base/"blocks").glob("*.json")):
            meta=json.loads(path.read_text(encoding="utf-8"))
            if not meta.get("quality",{}).get("ok"): continue
            selected=[hashes[h] for h in meta["record_hashes"]]
            if [r["candidate_id"] for r in selected]!=meta["candidate_ids"] or not block_quality(selected)["ok"]:
                raise StudyStop("accepted_block_quality_reaudit_failed")
            for row in selected:
                if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row["record_sha256"]:
                    raise StudyStop("source_record_hash_changed")
                if not audit_row(row,self.design["scope"]["device_uuid"],self.output)["ok"]:
                    raise StudyStop("source_record_guard_reaudit_failed")
            if selected and selected[0]["phase"]==phase:
                key=(selected[0]["workload_id"],selected[0]["round"])
                if key in blocks: raise StudyStop("multiple_accepted_attempts_for_block")
                blocks[key]=selected
        if set(blocks)!={(wid,k) for wid in self.workloads for k in range(rounds)}:
            raise StudyStop("incomplete_accepted_blocks:"+stage)
        return blocks

    def screen(self):
        self.require_main()
        self.expected=self.design["budgets"]["screen_processes"]
        frozen=self.calibrate(list(self.workloads.values()))
        blocks={}
        for w in workload_order(self.design,"screen"):
            cids=block_candidates(self.design,w["id"],"screen",0)
            rows,_=self.block(w,"screen",0,cids,frozen[w["id"]]["graph_repeats"],self.design["measurement"]["screen_samples"])
            blocks[w["id"]]=rows
        selection=freeze_screen(self.design,blocks)
        selection.update(study_id=self.design["study_id"],design_sha256=sha(self.args.design.read_bytes()),controller_sha256=self.controller_sha256)
        self.frozen_file(self.output/"screen_selection.json",selection)
        self.gate(True,selection_sha256=sha((self.output/"screen_selection.json").read_bytes()))

    def confirmation(self):
        self.require_main()
        self.require("screen")
        selection=json.loads((self.output/"screen_selection.json").read_text(encoding="utf-8"))
        frozen=json.loads((self.output/"graph_repeats.json").read_text(encoding="utf-8"))
        rounds=self.design["selection"]["confirmation_rounds"]
        self.expected=sum(len(x["candidates"]) for x in selection["workloads"].values())*rounds
        blocks={}
        for k in range(rounds):
            for w in workload_order(self.design,"confirmation",k):
                cids=block_candidates(self.design,w["id"],"confirmation",k,selection["workloads"][w["id"]]["candidates"])
                rows,_=self.block(w,"confirmation",k,cids,frozen[w["id"]]["graph_repeats"],self.design["measurement"]["confirmation_samples"])
                blocks[w["id"],k]=rows
        summary=confirmation_summary(self.design,selection,blocks)
        self.frozen_file(self.output/"confirmation_summary.json",summary)
        unestimable=[wid for wid,entry in summary.items() if entry["primary"] is None]
        screen={wid:rows for (wid,_),rows in self.accepted_blocks("screen","screen").items()}
        headline=freeze_headline(self.design,selection,summary,screen)
        headline.update(study_id=self.design["study_id"],selection_sha256=sha((self.output/"screen_selection.json").read_bytes()),
                        confirmation_sha256=sha((self.output/"confirmation_summary.json").read_bytes()),controller_sha256=self.controller_sha256,
                        interpretation_addendum_sha256=sha(self.args.design.with_name("research_addendum_A3.json").read_bytes()))
        self.frozen_file(self.output/"headline_selection.json",headline)
        self.gate(True,headline_selection_sha256=sha((self.output/"headline_selection.json").read_bytes()),unestimable_primary_workloads=unestimable)
        if unestimable:
            return dict(stage_checks_passed=False,exit_reason="confirmation_complete_with_unestimable_primary_workloads")

    def headline(self):
        self.require_main()
        self.require("confirmation")
        selected=json.loads((self.output/"headline_selection.json").read_text(encoding="utf-8"))
        frozen=json.loads((self.output/"graph_repeats.json").read_text(encoding="utf-8"))
        self.expected=selected["rounds"]*sum(len(x["candidates"]) for x in selected["workloads"])
        lookup={x["workload_id"]:x for x in selected["workloads"]}
        for k in range(selected["rounds"]):
            for w in workload_order(self.design,"headline",k):
                if w["id"] not in lookup: continue
                cids=block_candidates(self.design,w["id"],"headline",k,lookup[w["id"]]["candidates"])
                self.block(w,"headline",k,cids,frozen[w["id"]]["graph_repeats"],self.design["measurement"]["headline_samples"])
        self.gate(True)

    def cost(self):
        self.require_main()
        self.require("screen")
        selection=json.loads((self.output/"screen_selection.json").read_text(encoding="utf-8"))
        spec=self.design["cost_profile"]
        self.expected=len(spec["workloads"])*spec["paired_rounds"]*2
        qualities=[]
        for k in range(spec["paired_rounds"]):
            for wid in spec["workloads"]:
                path,order=self.candidate_file(wid,k,"cost")
                selected=selection["workloads"][wid]["winner"]
                modes=cost_modes(k)
                attempt=1 if self.cost_recovery and (wid,k)==COST_RECOVERY_PAIR else 0
                if self.cost_recovery and (wid,k) in self.cost_pair_records:
                    pair=self.cost_pair_records[wid,k]
                    qualities.append(dict(workload_id=wid,round=k,**pair["quality"]))
                    continue
                old=[r for r in self.rows if r["phase"]=="cost" and r["workload_id"]==wid and r["round"]==k]
                if old and not self.cost_recovery and len(old)!=2:
                    raise StudyStop("partial_cost_pair_requires_explicit_frozen_A6_admission")
                if self.cost_recovery:
                    if len(self.rows)>49 or len(self.rows)-21>=28: raise StudyStop("A6_new_pipeline_budget_exhausted")
                    if any(r.get("block_attempt",0)==attempt for r in old): raise StudyStop("A6_partial_or_completed_recovery_pair_not_rerunnable")
                results={}
                failed=[]
                pair_rows=[]
                for mode in modes:
                    if self.cost_recovery and (len(self.rows)>=49 or len(self.rows)-21>=28): raise StudyStop("A6_new_pipeline_budget_exhausted")
                    w=dict(self.workloads[wid],graph_repeats=1,samples=1,warmup=0,warmup_ms=0,study_mode=mode,
                           candidate_file=str(path.resolve()),selected_id=selected,block_id=f"cost/{wid}/r{k}",block_attempt=attempt)
                    row=self.attempt(w,mode,"cost",k)
                    pair_rows.append(row)
                    results[mode]=row.get("result",{})
                    if row["status"]!="ok": failed.append(mode)
                    if self.cost_recovery and (row["status"]!="ok" or row.get("returncode")!=0 or row.get("validation_errors")):
                        raise StudyStop("A6_new_pipeline_execution_or_protocol_failure_stops_recovery")
                quality=audit_pair(results["full-search-cost"],results["describe-select-cost"],order,selected)
                if failed:
                    quality["ok"]=False
                    quality["errors"].extend("process_status_failed:"+mode for mode in failed)
                if self.cost_recovery and any(not (error.startswith("describe_full_inequivalent:") or error=="selected_materialization_changed_plan_or_source") for error in quality["errors"]):
                    raise StudyStop("A6_new_cost_pair_protocol_failure_stops_recovery")
                pair=dict(workload_id=wid,round=k,block_attempt=attempt,candidate_ids=modes,record_hashes=[r["record_sha256"] for r in pair_rows],quality=quality)
                self.frozen_file(self.stage_path/"pair_attempts"/f"{wid}_r{k}_a{attempt}.json",pair)
                self.cost_pair_records[wid,k]=pair
                self.save_cost_selection()
                qualities.append(dict(workload_id=wid,round=k,**quality))
                write_json(self.stage_path/"pair_audits.json",qualities)
        passed=all(q["ok"] for q in qualities)
        accepted,_,_=selected_cost_rows(self.output,require_complete=True)
        self.gate(passed,H3_feasibility="established_in_fixed_cost_matrix" if passed else "unresolved_or_limited",pairs=qualities,
                  record_count=len(accepted),raw_record_count=len(self.rows),accepted_pair_selection_sha256=sha((self.stage_path/"accepted_pairs.json").read_bytes()),
                  cost_recovery_admission_sha256=sha((self.output/"cost_recovery_admission.json").read_bytes()) if self.cost_recovery else None,
                  accepted_record_hashes=[r["record_sha256"] for r in accepted])
        return dict(stage_checks_passed=passed,exit_reason="cost_completed" if passed else "H3_unresolved_or_limited")

    def save_cost_selection(self):
        spec=self.design["cost_profile"]
        pairs=[self.cost_pair_records[wid,k] for k in range(spec["paired_rounds"]) for wid in spec["workloads"] if (wid,k) in self.cost_pair_records]
        path=self.output/"cost_recovery_admission.json"
        write_json(self.stage_path/"accepted_pairs.json",dict(schema_version="plan-selection-cost-pairs/v1",study_id=self.design["study_id"],
            admission_sha256=sha(path.read_bytes()) if self.cost_recovery else None,pairs=pairs,accepted_processes=len(pairs)*2,
            raw_records=len(self.rows),engineering_only_record_hashes=self.cost_recovery["engineering_only_record_hashes"] if self.cost_recovery else [],
            physical_processes_estimate=len(self.rows)+(1 if self.cost_recovery else 0),complete=len(pairs)==24))

    def candidate_diagnostic(self):
        """Four fixed fresh-process probes; never an eligibility or performance gate."""
        if not self.args.plain_exe:
            raise StudyStop("candidate_diagnostic_requires_registered_plain_exe")
        bad="a128_r4_l1_np1_s2_8192"
        schedule=[("instrumented","default"),("plain","default"),("instrumented",bad),("plain",bad)]
        self.expected=4
        self.plain_sha256=sha(Path(self.args.plain_exe).resolve(strict=True).read_bytes())
        self.frozen_file(self.stage_path/"diagnostic_manifest.json",dict(
            purpose="Fixed original-header versus instrumented reproduction of the retained semantic fault; no candidate-validity reclassification.",
            selected_candidate=bad,schedule=[dict(binary_role=role,candidate_id=cid,config=candidate_config(self.candidates[cid])) for role,cid in schedule],
            input_seed=2235648248,n=16384,batch=1,graph_repeats=16,pattern="zero",samples=1,
            warmup=0,warmup_ms=0,controller_sha256=self.controller_sha256,
            instrumented_executable_sha256=self.args.expected_executable_sha256,plain_executable_sha256=self.plain_sha256,
            excluded_from_selection_and_performance=True))
        outcomes=[]
        for index,(role,cid) in enumerate(schedule):
            config=candidate_config(self.candidates[cid])
            phase="candidate_diagnostic/"+role
            key=(phase,"n16384_b1",cid,0,0)
            if key in self.by_key:
                outcomes.append(self.by_key[key]);continue
            w=dict(id="n16384_b1",n=16384,batch=1,input_bytes=131072,graph_repeats=16,pattern="zero",samples=1,warmup=0,warmup_ms=0,
                   input_seed_override=2235648248,binary_role=role,diagnostic_capture=True)
            original_exe,original_sha=self.args.exe,self.args.expected_executable_sha256
            self.active=dict(phase=phase,workload_id=w["id"],candidate_id=cid,binary_role=role)
            self.status("running")
            try:
                if role=="plain":
                    self.args.exe=str(Path(self.args.plain_exe).resolve(strict=True))
                    self.args.expected_executable_sha256=self.plain_sha256
                row=run_one(self.args,w,(cid,config),phase,0,index)
            finally:
                self.args.exe,self.args.expected_executable_sha256=original_exe,original_sha
            audit=audit_row(row,self.design["scope"]["device_uuid"],self.output)
            if not audit["ok"]:
                append_json(self.stage_path/"unaccepted_records.jsonl",dict(row=row,guard_audit=audit))
                raise StudyStop("diagnostic_guard_provenance_rejected")
            row.update(guard_audit=audit,controller_sha256=self.controller_sha256,excluded_from_selection_and_performance=True)
            row["record_sha256"]=sha(canonical(row))
            append_json(self.measurements,row)
            self.rows.append(row);self.by_key[key]=row;outcomes.append(row)
            self.active=None
            self.status("running")
        write_json(self.stage_path/"diagnostic_summary.json",dict(
            outcomes=[dict(binary_role=r["workload"]["binary_role"],candidate_id=r["candidate_id"],status=r["status"],returncode=r.get("returncode"),
                failure_phase=r.get("result",{}).get("failure_phase"),error=r.get("result",{}).get("error"),record_sha256=r["record_sha256"]) for r in outcomes],
            no_scientific_gate_written=True,eligibility_decision="pending independent review",excluded_from_selection_and_performance=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,default=Path(__file__).with_name("research_design.json"))
    p.add_argument("--exe")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=("plan","health","semantics","equivalence","cost_smoke","screen","confirmation","cost","headline","candidate_diagnostic"),required=True)
    p.add_argument("--timeout",type=float,default=180)
    p.add_argument("--import-calibration-from",type=Path,help="Explicit clean same-study calibration import; requires exact row/hash/guard audit")
    p.add_argument("--plain-exe",help="Original-header control binary, only for the A1 engineering gate")
    p.add_argument("--retry-block",help="Explicit whole screening block retry after a newly passed health gate; maximum two blocks total")
    p.add_argument("--candidate-fault-diagnostics",type=Path,help="Explicit A5 matching-four-diagnostic admission; all guards/source/default controls are reaudited before activation")
    p.add_argument("--candidate-fault-decision",type=Path,help="Explicit independent frozen A5 matching-failure decision; its bytes and diagnostic hashes are retained")
    p.add_argument("--cost-recovery-admission",type=Path,help="Explicit one-time frozen A6 whole-pair engineering recovery; never a generic retry")
    args=p.parse_args()
    if args.stage=="plan":
        design=load_design(args.design)
        value=dict(study_id=design["study_id"],design_sha256=sha(args.design.read_bytes()),budgets=design["budgets"],
                   stages=design["stages"],GPU_work_started=False)
        write_json(args.output/"plan.json",value)
        print(json.dumps(value,indent=2))
        return
    if not args.exe:
        p.error("--exe is required for guarded stages")
    study=None
    try:
        study=Study(args)
        outcome=getattr(study,args.stage)()
        if isinstance(outcome,dict) and outcome.get("stage_checks_passed") is False:
            study.status("complete_with_limitations",reason=outcome["exit_reason"])
        else:
            study.status("complete",reason="all_declared_stage_checks_passed")
    except BaseException as error:
        if study is not None:
            study.status("stopped",reason=f"{type(error).__name__}:{error}")
        else:
            write_json(args.output/args.stage/"status.json",dict(stage=args.stage,state="stopped",updated_utc=utc(),
                       exit_reason=f"{type(error).__name__}:{error}"))
        raise


if __name__=="__main__":
    main()
