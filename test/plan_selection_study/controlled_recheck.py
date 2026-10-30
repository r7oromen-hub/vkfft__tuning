#!/usr/bin/env python3
"""Independent bounded H2 follow-up; GPU execution only through gpu_guard.py."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import uuid

from run_suite import run_one
from cost_audit import audit_pair
from cost_recovery import selected_cost_rows,cost_session_allowed
import guard_audit
from guard_audit import audit_row,completed_session,resolve_log,read_guard
from study_common import append_json,block_candidates,block_quality,candidate_config,canonical,read_jsonl,sha,utc,write_json,workload_order,input_seed,execution_errors


class RecheckStop(RuntimeError):
    pass


@contextlib.contextmanager
def closed_log_cache():
    """Reuse immutable completed log reads; audit_row itself remains unchanged."""
    original=guard_audit.read_guard
    cached={}
    def read(path):
        path=Path(path)
        if path not in cached: cached[path]=original(path)
        return cached[path]
    guard_audit.read_guard=read
    try: yield
    finally: guard_audit.read_guard=original


def independent_destination(primary,output):
    primary=primary.resolve(strict=True)
    destination=output.resolve()
    if primary==destination or primary in destination.parents or destination in primary.parents:
        raise RecheckStop("followup_requires_disjoint_output_directory")


def load_recheck(path,freeze_path):
    design=json.loads(path.read_text(encoding="utf-8"))
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("study_id")!=design["study_id"] or freeze.get("design_sha256")!=sha(path.read_bytes()):
        raise RecheckStop("independent_design_freeze_mismatch")
    c1_path=path.with_name("h2_recheck_activation_C1.json")
    c1=json.loads(c1_path.read_text(encoding="utf-8"))
    if c1.get("study_id")!=design["study_id"] or c1.get("design_sha256")!=sha(path.read_bytes()):
        raise RecheckStop("activation_C1_design_identity_mismatch")
    if design["study_id"]=="plan_selection_h2_followup_20261028_r1" and sha(c1_path.read_bytes())!="b1c2259a40b7fbf4f3659b8d5655cc51083ee6eb4269d523647e0671d4646210":
        raise RecheckStop("activation_C1_frozen_hash_mismatch")
    if design["study_id"]=="plan_selection_20261027_r2" or design["measurement"]["seed_namespace"]=="plan_selection_20261027_r2/v1":
        raise RecheckStop("followup_requires_independent_identity_and_seed_namespace")
    if {(w["n"],w["batch"]) for w in design["workloads"]}!={(16384,1),(16384,2048)} or len(design["workloads"])!=2:
        raise RecheckStop("followup_must_not_expand_two_fixed_endpoints")
    if {c["id"] for c in design["candidates"]}!={"default","default_replica","a128_r4_l1_np1"}:
        raise RecheckStop("followup_must_not_expand_candidate_pair_and_control")
    for candidate in design["candidates"]:
        expected={} if candidate["id"] in ("default","default_replica") else dict(aim_threads=128,register_boost=4,use_lut=1,register_boost_nonpow2=1)
        if candidate_config(candidate)!=expected or candidate.get("disable_reorder_four_step",0)!=0:
            raise RecheckStop("followup_candidate_requests_changed")
    if design["measurement"]["screen_samples"]!=101 or design["measurement"]["confirmation_samples"]!=101 or design["measurement"]["warmup_ms"]!=1000:
        raise RecheckStop("followup_measurement_settings_changed")
    if (len(design["health"]["workloads"])!=3 or {w["id"] for w in design["health"]["workloads"]}!={"n1024_b1","n81920_b409","n1048576_b32"}
            or design["health"]["paired_process_rounds"]!=5 or design["budgets"]["health_processes_max"]!=30):
        raise RecheckStop("followup_health_budget_or_workloads_changed")
    if design["rounds"]!=10 or len(design["directional_tests"])!=2:
        raise RecheckStop("followup_requires_ten_rounds_and_family_two")
    directions={(t["workload_id"],t["reference"],t["target"]) for t in design["directional_tests"]}
    if directions!={("n16384_b1","a128_r4_l1_np1","default"),("n16384_b2048","default","a128_r4_l1_np1")}:
        raise RecheckStop("followup_directional_tests_must_match_frozen_opposite_directions")
    if design["budgets"]["recheck_processes"]!=60 or design["budgets"]["total_processes_max"]>90 or design["budgets"].get("retries",0)!=0:
        raise RecheckStop("followup_budget_expansion_forbidden")
    return design


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=value: raise RecheckStop("immutable_artifact_changed:"+str(path))
    else: write_json(path,value)


class Recheck:
    def __init__(self,args,require_guard=True):
        self.args=args
        self.design=load_recheck(args.design,args.freeze)
        args.design_data=self.design
        self.output=args.output
        self.stage=args.stage
        independent_destination(args.primary_output,self.output)
        self.stage_path=self.output/self.stage
        self.stage_path.mkdir(parents=True,exist_ok=True)
        self.path=self.stage_path/"measurements.jsonl"
        self.rows=read_jsonl(self.path)
        self.candidates={c["id"]:c for c in self.design["candidates"]}
        self.guard_id=os.environ.get("VKFFT_STUDY_GUARD_ID")
        if require_guard and (not self.guard_id or os.environ.get("VKFFT_STUDY_GUARD_UUID")!=self.design["scope"]["device_uuid"]
                or not os.environ.get("VKFFT_STUDY_GUARD_LOG")):
            raise RecheckStop("unchanged_GPU_guard_required")
        with closed_log_cache(): self.manifest()

    def manifest(self):
        primary=self.args.primary_output.resolve(strict=True)
        independent_destination(primary,self.output)
        main=json.loads((primary/"manifest.json").read_text(encoding="utf-8"))
        executable=Path(self.args.exe).resolve(strict=True)
        exe_sha=sha(executable.read_bytes())
        if (main.get("study_id")!="plan_selection_20261027_r2" or main.get("gpu_uuid")!=self.design["scope"]["device_uuid"]
                or main.get("executable_sha256")!=exe_sha or exe_sha!=self.design["scope"]["executable_sha256"]
                or main.get("design_sha256")!=self.design["selection_origin"]["primary_design_sha256"]):
            raise RecheckStop("primary_and_followup_binary_GPU_identity_mismatch")
        sources={"manifest.json":sha((primary/"manifest.json").read_bytes()),"graph_repeats.json":sha((primary/"graph_repeats.json").read_bytes())}
        recovery=None
        if (primary/"cost_recovery_admission.json").exists():
            c2_path=self.args.design.with_name("h2_recheck_activation_C2.json")
            c2=json.loads(c2_path.read_text(encoding="utf-8"))
            if (sha(c2_path.read_bytes())!="4ae470bb0d86306ed767d4c1fa189e68da72c3e42a0b02de9e2916bb7beb77f5"
                    or c2.get("design_sha256")!=sha(self.args.design.read_bytes())):
                raise RecheckStop("A6_recovered_primary_requires_frozen_H2_C2")
            _,_,recovery=selected_cost_rows(primary,require_complete=True,require_sessions=True)
            if recovery["A6_sha256"]!=c2["A6_sha256"]: raise RecheckStop("H2_C2_A6_hash_mismatch")
            for name in ("research_addendum_A6.json","research_addendum_A6_freeze.json","cost_recovery_admission.json","cost/accepted_pairs.json",
                         "guards/cost_01.jsonl","guards/cost_01_incidents.jsonl"):
                sources[name]=sha((primary/name).read_bytes())
            sources["H2_C2"]=sha(c2_path.read_bytes())
            for path in sorted((primary/"cost/pair_attempts").glob("*.json")):
                sources[str(path.relative_to(primary))]=sha(path.read_bytes())
        for stage in ("screen","confirmation","cost","headline"):
            status_path=primary/stage/"status.json"
            status=json.loads(status_path.read_text(encoding="utf-8"))
            state=status.get("state")
            if state!="complete" and not (stage=="cost" and state=="complete_with_limitations" and status.get("exit_reason")=="H3_unresolved_or_limited"):
                raise RecheckStop("primary_sequence_not_finished:"+stage)
            gate_path=primary/stage/"gate.json"
            gate=json.loads(gate_path.read_text(encoding="utf-8"))
            if state=="complete" and gate.get("passed") is not True:
                raise RecheckStop("primary_completed_stage_gate_not_passed:"+stage)
            if state=="complete_with_limitations" and gate.get("passed") is not False:
                raise RecheckStop("primary_limited_cost_requires_explicit_negative_gate")
            if not gate.get("guard_sessions"): raise RecheckStop("primary_stage_guard_sessions_empty:"+stage)
            for session in gate.get("guard_sessions",[]):
                if stage=="cost" and recovery and session["guard_id"]==recovery["original_guard_id"]:
                    if sha(resolve_log(session,primary).read_bytes())!=recovery["original_guard_log_sha256"]:
                        raise RecheckStop("H2_C2_original_cost_guard_changed")
                    continue
                if not completed_session(resolve_log(session,primary),session["guard_id"]):
                    raise RecheckStop("primary_stage_guard_not_completed:"+stage)
            sources[stage+"/status.json"]=sha(status_path.read_bytes())
            sources[stage+"/gate.json"]=sha(gate_path.read_bytes())
            sources[stage+"/measurements.jsonl"]=sha((primary/stage/"measurements.jsonl").read_bytes())
        sources["cost/pair_audits.json"]=sha((primary/"cost/pair_audits.json").read_bytes())
        self.verify_primary_execution(primary,main)
        self.verify_primary_cost(primary,main)
        repeats=json.loads((primary/"graph_repeats.json").read_text(encoding="utf-8"))
        if sources["graph_repeats.json"]!=self.design["measurement"]["repeats_source_sha256"]:
            raise RecheckStop("frozen_R_source_file_hash_changed")
        requested=self.design["workloads"]+self.design["health"]["workloads"]
        self.repeats={w["id"]:repeats[w["id"]] for w in requested}
        if {wid:r["graph_repeats"] for wid,r in self.repeats.items()}!=self.design["measurement"]["repeats_frozen"]:
            raise RecheckStop("frozen_R_entries_changed")
        for filename in ("screen_selection.json","headline_selection.json"):
            sources[filename]=sha((primary/filename).read_bytes())
        sources["screen/measurements.jsonl"]=sha((primary/"screen/measurements.jsonl").read_bytes())
        if (sources["screen_selection.json"]!=self.design["selection_origin"]["screen_selection_sha256"]
                or sources["screen/measurements.jsonl"]!=self.design["selection_origin"]["screen_measurements_sha256"]):
            raise RecheckStop("posthoc_S1_selection_source_changed")
        provenance=self.repeat_provenance(primary,main)
        code_files=("controlled_recheck.py","run_suite.py","study_common.py","selection_analysis.py","candidate_policy.py","cost_recovery.py","cost_audit.py","guard_audit.py","trace_audit.py","gpu_guard.py")
        controller={name:sha(Path(__file__).with_name(name).read_bytes()) for name in code_files}
        self.controller_sha256=sha(canonical(controller))
        value=dict(schema_version="independent-H2-followup-runtime/v1",study_id=self.design["study_id"],
            design_sha256=sha(self.args.design.read_bytes()),freeze_sha256=sha(self.args.freeze.read_bytes()),
            activation_C1_sha256=sha(self.args.design.with_name("h2_recheck_activation_C1.json").read_bytes()),
            executable=str(executable),executable_sha256=exe_sha,gpu_uuid=self.design["scope"]["device_uuid"],
            primary_output=str(primary),primary_source_sha256=sources,graph_repeats=self.repeats,
            repeat_provenance=provenance,
            controller_sha256=self.controller_sha256,controller_files=controller,
            independent_posthoc_followup=True,primary_effects_and_manifests_unchanged=True)
        immutable_json(self.output/"manifest.json",value)
        immutable_json(self.output/"frozen_design.json",self.design)
        immutable_json(self.output/"frozen_design_hash.json",json.loads(self.args.freeze.read_text(encoding="utf-8")))
        for filename,source in (("h2_recheck_design.json",self.args.design),("h2_recheck_freeze.json",self.args.freeze),
                                ("h2_recheck_activation_C1.json",self.args.design.with_name("h2_recheck_activation_C1.json"))):
            target=self.output/filename
            if target.exists() and target.read_bytes()!=source.read_bytes(): raise RecheckStop("exact_frozen_source_changed:"+filename)
            if not target.exists(): target.write_bytes(source.read_bytes())
        if recovery:
            source=self.args.design.with_name("h2_recheck_activation_C2.json")
            target=self.output/source.name
            if target.exists() and target.read_bytes()!=source.read_bytes(): raise RecheckStop("H2_C2_source_changed")
            if not target.exists(): target.write_bytes(source.read_bytes())
        self.args.expected_executable_sha256=exe_sha

    def verify_primary_execution(self,primary,main):
        design=json.loads((primary/"research_design.json").read_text(encoding="utf-8"))
        screen_selection=json.loads((primary/"screen_selection.json").read_text(encoding="utf-8"))
        headline=json.loads((primary/"headline_selection.json").read_text(encoding="utf-8"))
        choices={"screen":{w["id"]:[c["id"] for c in design["candidates"]] for w in design["workloads"]},
                 "confirmation":{wid:x["candidates"] for wid,x in screen_selection["workloads"].items()},
                 "headline":{x["workload_id"]:x["candidates"] for x in headline["workloads"]}}
        for stage,rounds in (("screen",1),("confirmation",3),("headline",10)):
            rows=[r for r in read_jsonl(primary/stage/"measurements.jsonl") if r.get("phase")==stage]
            expected={(wid,k,cid) for wid,cids in choices[stage].items() for k in range(rounds) for cid in cids}
            keys=[(r.get("workload_id"),r.get("round"),r.get("candidate_id")) for r in rows]
            if len(rows)!=len(expected) or len(set(keys))!=len(expected) or set(keys)!=expected:
                raise RecheckStop("primary_frozen_execution_coverage_failed:"+stage)
            hashes={r.get("record_sha256"):r for r in rows}
            for row in rows:
                if row.get("study_id")!=main["study_id"] or sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"):
                    raise RecheckStop("primary_execution_record_identity_or_hash_changed:"+stage)
                if not audit_row(row,main["gpu_uuid"],primary)["ok"]: raise RecheckStop("primary_execution_row_guard_unclean:"+stage)
            block_keys=set();selected_hashes=[]
            for path in (primary/stage/"blocks").glob("*.json"):
                block=json.loads(path.read_text(encoding="utf-8"))
                selected=[hashes[h] for h in block["record_hashes"] if h in hashes]
                if not selected: raise RecheckStop("primary_empty_block:"+stage)
                wid,k=selected[0]["workload_id"],selected[0]["round"]
                order=block_candidates(design,wid,stage,k,choices[stage][wid])
                if ((wid,k) in block_keys or [r["candidate_id"] for r in selected]!=order or block.get("candidate_ids")!=order
                        or any((r["workload_id"],r["round"])!=(wid,k) or r.get("position_in_block")!=i for i,r in enumerate(selected))
                        or not block_quality(selected)["ok"]): raise RecheckStop("primary_execution_block_integrity_failed:"+stage)
                block_keys.add((wid,k));selected_hashes.extend(block["record_hashes"])
            if block_keys!={(wid,k) for wid in choices[stage] for k in range(rounds)} or set(selected_hashes)!=set(hashes) or len(selected_hashes)!=len(rows):
                raise RecheckStop("primary_execution_accepted_block_coverage_failed:"+stage)

    def verify_primary_cost(self,primary,main):
        d=json.loads((primary/"research_design.json").read_text(encoding="utf-8"))
        if sha((primary/"research_design.json").read_bytes())!=main["design_sha256"]:
            raise RecheckStop("primary_design_source_hash_changed")
        spec=d["cost_profile"]
        admission=None
        if (primary/"cost_recovery_admission.json").exists():
            rows,_,admission=selected_cost_rows(primary,require_complete=True,require_sessions=True)
        else: rows=read_jsonl(primary/"cost/measurements.jsonl")
        expected={(wid,k,mode) for wid in spec["workloads"] for k in range(3) for mode in ("full-search-cost","describe-select-cost")}
        keys=[(r.get("workload_id"),r.get("round"),r.get("candidate_id")) for r in rows]
        if len(rows)!=48 or len(set(keys))!=48 or set(keys)!=expected: raise RecheckStop("primary_cost_exact48_coverage_failed")
        lookup=dict(zip(keys,rows))
        for row in rows:
            if (row.get("status")!="ok" or row.get("returncode")!=0 or row.get("validation_errors") or row.get("phase")!="cost"
                    or row.get("study_id")!=main["study_id"]): raise RecheckStop("primary_cost_pipeline_execution_or_protocol_failed")
            if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"):
                raise RecheckStop("primary_cost_record_hash_changed")
            if not audit_row(row,main["gpu_uuid"],primary)["ok"]: raise RecheckStop("primary_cost_row_guard_unclean")
            if not cost_session_allowed(row,primary,admission): raise RecheckStop("primary_cost_row_session_not_complete")
        stored=json.loads((primary/"cost/pair_audits.json").read_text(encoding="utf-8"))
        if len(stored)!=24 or {(x["workload_id"],x["round"]) for x in stored}!={(w,k) for w in spec["workloads"] for k in range(3)}:
            raise RecheckStop("primary_cost_exact24_pair_audits_missing")
        frozen=json.loads((primary/"screen_selection.json").read_text(encoding="utf-8"))
        limited=json.loads((primary/"cost/status.json").read_text(encoding="utf-8"))["state"]=="complete_with_limitations"
        negative_pairs=0
        for wid,k in sorted({(w,k) for w,k,_ in expected}):
            full=lookup[wid,k,"full-search-cost"]["result"]
            desc=lookup[wid,k,"describe-select-cost"]["result"]
            order=sorted(c["id"] for c in d["candidates"])
            random.Random(input_seed(d,"cost/candidate_order",wid,k)).shuffle(order)
            if full.get("candidate_order_ids")!=order or desc.get("candidate_order_ids")!=order:
                raise RecheckStop("primary_cost_frozen_candidate_order_changed")
            quality=audit_pair(full,desc,order,frozen["workloads"][wid]["winner"])
            saved=next(x for x in stored if x["workload_id"]==wid and x["round"]==k)
            if any(saved.get(key)!=value for key,value in quality.items()): raise RecheckStop("primary_cost_stored_pair_audit_disagreement")
            allowed=all(e.startswith("describe_full_inequivalent:") or e=="selected_materialization_changed_plan_or_source" for e in quality["errors"])
            if not quality["ok"] and (not limited or not allowed): raise RecheckStop("primary_cost_protocol_or_nonpermitted_audit_failure")
            if not quality["ok"]: negative_pairs+=1
        if limited and negative_pairs==0: raise RecheckStop("primary_limited_cost_requires_actual_equivalence_negative")

    def repeat_provenance(self,primary,main):
        roots=[primary]
        imported=primary/"calibration_import_provenance.json"
        if imported.exists():
            source=Path(json.loads(imported.read_text(encoding="utf-8"))["source"])
            if not source.is_dir():
                # Local analysis copies may retain the source under the same results parent.
                source=primary.parent/source.name
            if not source.is_dir(): raise RecheckStop("original_imported_R_source_unavailable")
            roots.append(source)
        records={}
        for root in roots:
            manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
            if any(manifest.get(k)!=main.get(k) for k in ("study_id","design_sha256","executable_sha256","gpu_uuid")):
                raise RecheckStop("R_provenance_scientific_identity_mismatch")
            for stage in ("health","screen"):
                for row in read_jsonl(root/stage/"measurements.jsonl"):
                    if row.get("phase")=="calibration": records[row.get("record_sha256")]=(root,row)
        evidence=[]
        for wid,entry in self.repeats.items():
            source,row=records.get(entry["source_record_sha256"],(None,None))
            if (row is None or row.get("status")!="ok" or row.get("candidate_id")!="default" or row.get("workload_id")!=wid
                    or row.get("result",{}).get("graph_repeats")!=entry["graph_repeats"]):
                raise RecheckStop("missing_original_measured_R_provenance:"+wid)
            if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row["record_sha256"]:
                raise RecheckStop("original_R_record_hash_changed:"+wid)
            audit=audit_row(row,self.design["scope"]["device_uuid"],source)
            if not audit["ok"]: raise RecheckStop("original_R_guard_interval_not_clean:"+wid)
            evidence.append(dict(workload_id=wid,source_record_sha256=row["record_sha256"],source_root=str(source),guard_audit=audit))
        return evidence

    def status(self,state,reason=None):
        write_json(self.stage_path/"status.json",dict(study_id=self.design["study_id"],stage=self.stage,state=state,
            exit_reason=reason,updated_utc=utc(),recorded_processes=len(self.rows),successful=sum(r["status"]=="ok" for r in self.rows),
            guard_id=self.guard_id,no_retries=True))

    def enter_stage(self):
        status_path=self.stage_path/"status.json"
        reservation_path=self.stage_path/"attempt_reservation.json"
        if not reservation_path.exists(): raise RecheckStop("one_time_launcher_reservation_required")
        reservation=json.loads(reservation_path.read_text(encoding="utf-8"))
        if reservation.get("reservation_id")!=getattr(self.args,"reservation_id",None):
            raise RecheckStop("one_time_launcher_reservation_mismatch")
        if self.rows:
            status=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if status.get("state")=="complete":
                for row in self.rows: self.audit_record(row)
                return False
            raise RecheckStop("interrupted_stage_not_resumable_no_retry_budget")
        status=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if status.get("state")!="reserved" or (self.stage_path/"unaccepted_records.jsonl").exists():
            raise RecheckStop("zero_row_attempt_not_resumable_no_retry_budget")
        log=Path(os.environ["VKFFT_STUDY_GUARD_LOG"]).resolve()
        if str(log)!=reservation.get("guard_log"): raise RecheckStop("reservation_guard_log_mismatch")
        events=read_guard(log)
        starts=[e for e in events if e.get("event")=="guard_start"]
        if len(starts)!=1 or starts[0].get("guard_id")!=self.guard_id or any(e.get("event")=="guard_abort" for e in events):
            raise RecheckStop("guard_attempt_already_used_no_retry_budget")
        immutable_json(self.stage_path/"guard_admission.json",dict(reservation_id=reservation["reservation_id"],guard_id=self.guard_id,guard_log=str(log)))
        self.status("running")
        return True

    def audit_record(self,row):
        if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"):
            raise RecheckStop("followup_record_hash_changed")
        if not audit_row(row,self.design["scope"]["device_uuid"],self.output)["ok"]:
            raise RecheckStop("followup_guard_record_not_clean")

    def attempt(self,original,cid,phase,round_no,position):
        total=sum(len(read_jsonl(self.output/stage/"measurements.jsonl")) for stage in ("health","recheck"))
        if total>=self.design["budgets"]["total_processes_max"]: raise RecheckStop("followup_total_process_budget_exhausted")
        w=dict(original,graph_repeats=self.repeats[original["id"]]["graph_repeats"],samples=101,warmup=5,warmup_ms=1000,
               block_id=f"{phase}/{original['id']}/r{round_no}",position_in_block=position,diagnostic_capture=True)
        row=run_one(self.args,w,(cid,candidate_config(self.candidates[cid])),phase,round_no,len(self.rows))
        audit=audit_row(row,self.design["scope"]["device_uuid"],self.output)
        if not audit["ok"]:
            append_json(self.stage_path/"unaccepted_records.jsonl",dict(row=row,guard_audit=audit))
            raise RecheckStop("followup_guard_rejected_process")
        row.update(guard_audit=audit,controller_sha256=self.controller_sha256,independent_posthoc_followup=True)
        row["record_sha256"]=sha(canonical(row))
        append_json(self.path,row);self.rows.append(row)
        self.status("running")
        print(f"{self.stage} {len(self.rows)} {w['id']} {cid} round={round_no} {row['status']}",flush=True)
        if row["status"]!="ok": raise RecheckStop("followup_any_execution_or_protocol_failure_stops:"+w["id"]+":"+cid)
        return row

    def block(self,w,phase,k,ids):
        order=block_candidates(self.design,w["id"],phase,k,ids)
        rows=[self.attempt(w,cid,phase,k,i) for i,cid in enumerate(order)]
        quality=block_quality(rows)
        write_json(self.stage_path/"blocks"/f"{w['id']}_r{k}.json",dict(workload_id=w["id"],round=k,candidate_ids=order,
            quality=quality,record_hashes=[r["record_sha256"] for r in rows]))
        if not quality["ok"]: raise RecheckStop("followup_whole_block_quality_failure:"+str(quality["errors"]))
        return quality

    def gate(self,passed,**extra):
        sessions={(r["guard_id"],r["guard_log"]) for r in self.rows}
        immutable_json(self.stage_path/"gate.json",dict(passed=passed,study_id=self.design["study_id"],stage=self.stage,
            record_count=len(self.rows),guard_sessions=[dict(guard_id=g,guard_log=p) for g,p in sorted(sessions)],**extra))

    def health(self):
        if not self.design["health"]["enabled"]: raise RecheckStop("new_health_not_enabled_in_frozen_followup")
        if not self.enter_stage(): return
        ratios={w["id"]:[] for w in self.design["health"]["workloads"]}
        for k in range(5):
            for w in self.design["health"]["workloads"]:
                q=self.block(w,"health",k,["default","default_replica"])
                ratios[w["id"]].append(q["default_over_replica"])
        means={wid:math.exp(statistics.mean(map(math.log,rs))) for wid,rs in ratios.items()}
        passed=all(.98<=v<=1.02 for v in means.values())
        self.gate(passed,paired_ratios=ratios,ratio_gmeans=means)
        if not passed: raise RecheckStop("followup_health_geometric_mean_outside_two_percent")
        self.status("complete","fixed_30_process_health_passed")

    def recheck(self):
        if self.design["health"]["enabled"]:
            with closed_log_cache():
                verify_health(self.design,self.output,json.loads((self.output/"manifest.json").read_text(encoding="utf-8")))
        if not self.enter_stage(): return
        for k in range(10):
            for w in workload_order(self.design,"h2_recheck",k): self.block(w,"h2_recheck",k,list(self.candidates))
        self.gate(True,directional_tests=self.design["directional_tests"],family_size=2)
        self.status("complete","fixed_60_process_followup_finished")


def verify_followup_row(design,manifest,row,phase):
    if row.get("study_id")!=design["study_id"] or row.get("phase")!=phase or row.get("controller_sha256")!=manifest["controller_sha256"]:
        raise RecheckStop("followup_row_study_phase_or_controller_mismatch")
    workloads=design["health"]["workloads"] if phase=="health" else design["workloads"]
    w=next((w for w in workloads if w["id"]==row.get("workload_id")),None)
    candidate=next((c for c in design["candidates"] if c["id"]==row.get("candidate_id")),None)
    k=row.get("round")
    if w is None or candidate is None or not isinstance(k,int) or k not in range(5 if phase=="health" else 10):
        raise RecheckStop("followup_row_workload_candidate_or_round_mismatch")
    seed=input_seed(design,phase,w["id"],k)
    result=row.get("result",{})
    repeats=design["measurement"]["repeats_frozen"][w["id"]]
    expected=dict(w,graph_repeats=repeats,samples=101)
    names=dict(aim_threads="aimThreads",register_boost="registerBoost",use_lut="useLUT",register_boost_nonpow2="registerBoostNonPow2",
               coalesced_memory="coalescedMemory",use_lut_4step="useLUT_4step",swap_to_2stage="swapTo2Stage4Step",swap_to_3stage="swapTo3Stage4Step")
    config=candidate_config(candidate)
    request=dict(makeForwardPlanOnly=1,**{names[key]:value for key,value in config.items()})
    if (row.get("input_seed")!=seed or result.get("seed")!=seed or row.get("config")!=config or result.get("requested_config")!=request
            or result.get("gpu",{}).get("uuid")!=design["scope"]["device_uuid"] or row.get("status")!="ok"
            or row.get("returncode")!=0 or row.get("validation_errors") or execution_errors(result,expected,101)):
        raise RecheckStop("followup_row_seed_R_request_or_execution_validation_failed")


def verify_health(design,root,manifest):
    gate=json.loads((root/"health/gate.json").read_text(encoding="utf-8"))
    status=json.loads((root/"health/status.json").read_text(encoding="utf-8"))
    rows=read_jsonl(root/"health/measurements.jsonl")
    if gate.get("passed") is not True or gate.get("record_count")!=30 or len(rows)!=30 or status.get("state")!="complete":
        raise RecheckStop("followup_health_exact30_complete_required")
    if not gate.get("guard_sessions"): raise RecheckStop("followup_health_guard_sessions_empty")
    for session in gate["guard_sessions"]:
        if not completed_session(resolve_log(session,root),session["guard_id"]): raise RecheckStop("followup_health_guard_not_completed")
    groups={};hashes=set()
    for row in rows:
        verify_followup_row(design,manifest,row,"health")
        rh=row.get("record_sha256")
        if rh in hashes or sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=rh: raise RecheckStop("health_record_duplicate_or_hash_changed")
        hashes.add(rh)
        if not audit_row(row,design["scope"]["device_uuid"],root)["ok"]: raise RecheckStop("health_row_guard_unclean")
        groups.setdefault((row["workload_id"],row["round"]),[]).append(row)
    expected={(w["id"],k) for w in design["health"]["workloads"] for k in range(5)}
    if set(groups)!=expected: raise RecheckStop("health_exact_workload_round_coverage_failed")
    ratios={w["id"]:[] for w in design["health"]["workloads"]}
    for (wid,k),paired_rows in groups.items():
        paired_rows.sort(key=lambda row:row["position_in_block"])
        order=block_candidates(design,wid,"health",k,["default","default_replica"])
        if len(paired_rows)!=2 or [r["candidate_id"] for r in paired_rows]!=order or [r["position_in_block"] for r in paired_rows]!=[0,1]:
            raise RecheckStop("health_exact_candidate_order_failed")
        quality=block_quality(paired_rows)
        if not quality["ok"]: raise RecheckStop("health_recomputed_block_quality_failed")
        ratios[wid].append(quality["default_over_replica"])
    means={wid:math.exp(statistics.mean(map(math.log,rs))) for wid,rs in ratios.items()}
    if any(not .98<=v<=1.02 for v in means.values()): raise RecheckStop("health_recomputed_geometric_mean_failed")
    return dict(record_count=30,ratio_gmeans=means,verified=True)


def analyze(args):
    from report_helpers import paired,headline_stat,holm
    design=load_recheck(args.design,args.freeze)
    manifest=json.loads((args.output/"manifest.json").read_text(encoding="utf-8"))
    if manifest["design_sha256"]!=sha(args.design.read_bytes()): raise RecheckStop("analysis_design_mismatch")
    rows=read_jsonl(args.output/"recheck"/"measurements.jsonl")
    errors=[]
    if design["health"]["enabled"]:
        try:
            with closed_log_cache(): verify_health(design,args.output,manifest)
        except (OSError,ValueError,KeyError,RecheckStop) as error:
            errors.append("health_reaudit:"+str(error))
    hashes={}
    for row in rows:
        try: verify_followup_row(design,manifest,row,"h2_recheck")
        except (ValueError,KeyError,TypeError,RecheckStop) as error: errors.append("recheck_metadata:"+str(error))
        if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"): errors.append("record_hash_changed")
        if not audit_row(row,design["scope"]["device_uuid"],args.output)["ok"]: errors.append("guard_row_unclean")
        if not completed_session(resolve_log(row,args.output),row["guard_id"]): errors.append("guard_session_not_complete")
        hashes[row.get("record_sha256")]=row
    complete=len(rows)==60 and len(hashes)==60 and not errors and all(r["status"]=="ok" for r in rows)
    blocks=list((args.output/"recheck"/"blocks").glob("*.json"))
    if len(blocks)!=20: complete=False
    block_keys=set()
    block_hashes=[]
    for path in blocks:
        b=json.loads(path.read_text(encoding="utf-8"))
        selected=[hashes[h] for h in b["record_hashes"] if h in hashes]
        key=(b["workload_id"],b["round"])
        if key in block_keys: complete=False
        block_keys.add(key);block_hashes.extend(b["record_hashes"])
        order=block_candidates(design,key[0],"h2_recheck",key[1],[c["id"] for c in design["candidates"]])
        if (len(selected)!=3 or [r["candidate_id"] for r in selected]!=order or b.get("candidate_ids")!=order
                or any((r["workload_id"],r["round"])!=key or r.get("position_in_block")!=i for i,r in enumerate(selected))
                or not block_quality(selected)["ok"]): complete=False
    if block_keys!={(w["id"],k) for w in design["workloads"] for k in range(10)} or len(set(block_hashes))!=60:
        complete=False
    tests=[]
    for direction in design["directional_tests"]:
        selected=[r for r in rows if r["workload_id"]==direction["workload_id"]]
        effect=paired(selected,direction["reference"],direction["target"],10)
        statistic=headline_stat(effect["ratios"] if effect and complete else [],10)
        tests.append(dict(direction,**statistic,quality_valid=complete,ratios=effect["ratios"] if effect else None))
    holm(tests)
    summary=dict(study_id=design["study_id"],design_sha256=manifest["design_sha256"],independent_posthoc_followup=True,
        family_size=2,complete_clean_followup=complete,audit_errors=sorted(set(errors)),tests=tests,
        rank_reversal_established=complete and all(t["clear_gain"] for t in tests),
        primary_manifest_and_effects_unchanged=True,
        measurement_controller_sha256=manifest["controller_sha256"],analysis_controller_sha256=sha(Path(__file__).read_bytes()),
        analysis_statistics_helper_sha256=sha(Path(__file__).with_name("report_helpers.py").read_bytes()),
        scope="Independent fresh process evidence at two S1-selected endpoints on the same GPU. Both frozen directions must independently pass the same Holm family; no broader-domain or causal claim.")
    write_json(args.output/"analysis"/"summary.json",summary)
    print(json.dumps(summary,indent=2))


def launch(args):
    """Reserve one attempt before unchanged guard quiet-start, including zero-row failures."""
    import gpu_guard
    stage=args.stage.removeprefix("launch-")
    child=argparse.Namespace(**vars(args));child.stage=stage
    study=Recheck(child,require_guard=False)
    reservation_path=study.stage_path/"attempt_reservation.json"
    if (reservation_path.exists() or (study.stage_path/"status.json").exists() or study.rows
            or (study.stage_path/"unaccepted_records.jsonl").exists()):
        raise RecheckStop("stage_attempt_already_reserved_or_used_no_retry")
    guard_log=study.output/"guards"/(stage+"_01.jsonl")
    incidents=study.output/"guards"/(stage+"_01_incidents.jsonl")
    if guard_log.exists() or incidents.exists(): raise RecheckStop("guard_attempt_path_already_used_no_retry")
    reservation_id=uuid.uuid4().hex
    immutable_json(reservation_path,dict(reservation_id=reservation_id,stage=stage,study_id=study.design["study_id"],
        reserved_utc=utc(),guard_log=str(guard_log.resolve()),controller_sha256=study.controller_sha256,no_retry=True))
    study.status("reserved","one_time_attempt_reserved_before_guard_quiet_start")
    command=[str(Path(gpu_guard.__file__).resolve()),"--gpu",study.design["scope"]["device_uuid"],
        "--log",str(guard_log.resolve()),"--incidents",str(incidents.resolve()),"--idle-seconds","30",
        "--script",str(Path(__file__).resolve()),"--","--design",str(args.design.resolve()),"--freeze",str(args.freeze.resolve()),
        "--output",str(args.output.resolve()),"--primary-output",str(args.primary_output.resolve()),"--exe",str(Path(args.exe).resolve()),
        "--stage",stage,"--timeout",str(args.timeout),"--reservation-id",reservation_id]
    original_argv=sys.argv
    try:
        sys.argv=command
        code=gpu_guard.main()
    except BaseException as error:
        study.rows=read_jsonl(study.path)
        study.status("stopped","guard_launch_or_quiet_start_failure:"+str(error))
        raise
    finally: sys.argv=original_argv
    if code:
        study.rows=read_jsonl(study.path)
        study.status("stopped","unchanged_guard_rejected_attempt_including_possible_zero_row_quiet_start_failure")
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,required=True)
    p.add_argument("--freeze",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--primary-output",type=Path)
    p.add_argument("--exe")
    p.add_argument("--stage",choices=("launch-health","launch-recheck","health","recheck","analyze"),required=True)
    p.add_argument("--timeout",type=float,default=180)
    p.add_argument("--reservation-id",help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.stage=="analyze": analyze(args);return
    if not args.exe or not args.primary_output: p.error("guarded stages require --exe and --primary-output")
    if not 0<args.timeout<=180: p.error("--timeout must not exceed the frozen 180-second process budget")
    independent_destination(args.primary_output,args.output)
    if args.stage.startswith("launch-"): return launch(args)
    study=None
    try:
        study=Recheck(args)
        getattr(study,args.stage)()
    except BaseException as error:
        if study is not None: study.status("stopped",f"{type(error).__name__}:{error}")
        else: write_json(args.output/args.stage/"status.json",dict(state="stopped",exit_reason=f"{type(error).__name__}:{error}",updated_utc=utc()))
        raise


if __name__=="__main__": raise SystemExit(main())
