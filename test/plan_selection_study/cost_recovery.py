"""Single A6 engineering recovery: append-only raw rows, explicit whole cost pairs."""
import json
from pathlib import Path
import random

from cost_audit import audit_pair
from guard_audit import audit_row,completed_session,read_guard,resolve_log
from study_common import canonical,read_jsonl,sha,input_seed


AFFECTED=("n16384_b2048",1)
OLD_GUARD="65fb86d7befa46b8829730cb819ff9d5"
QUARANTINED_RUN=OLD_GUARD+":8d1736ca0edb410e8be84f61a2c7994d"
ORPHAN="730865a20b6f808755a86f566b4caf796c3a7be63ca07d7bbd522f8f8c92c199"


def modes(round_no):
    return ["describe-select-cost","full-search-cost"] if round_no%2 else ["full-search-cost","describe-select-cost"]


def source_hash(addendum,suffix):
    fields={"cost/measurements.jsonl":"source_cost_measurements_sha256","cost/pair_audits.json":"source_pair_audits_sha256",
            "guards/cost_01.jsonl":"original_guard_log_sha256"}
    if suffix in fields: return addendum[fields[suffix]]
    found=[v for k,v in addendum.get("source_files",{}).items() if k.replace("\\","/").endswith(suffix)]
    if len(found)!=1: raise ValueError("A6_missing_unique_frozen_source:"+suffix)
    return found[0]


def validate_row(row,uuid,root):
    if sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))!=row.get("record_sha256"):
        raise ValueError("cost_record_hash_changed")
    if not audit_row(row,uuid,root)["ok"]: raise ValueError("cost_row_closed_interval_guard_not_clean")


def admit_recovery(addendum_path,design_path,root,manifest,guard_id,original_status_bytes=None):
    """No timing or quarantined output is used to choose the accepted replacement."""
    addendum_path=Path(addendum_path).resolve(strict=True)
    root=Path(root)
    a=json.loads(addendum_path.read_text(encoding="utf-8"))
    f=json.loads(addendum_path.with_name("research_addendum_A6_freeze.json").read_text(encoding="utf-8"))
    if (f.get("addendum_sha256")!=sha(addendum_path.read_bytes()) or a.get("study_id")!=manifest["study_id"]
            or f.get("original_design_sha256")!=sha(Path(design_path).read_bytes())):
        raise ValueError("A6_addendum_freeze_or_identity_mismatch")
    if (a.get("enabled") is not True or a.get("orphan_record_sha256")!=ORPHAN or a.get("original_guard_id")!=OLD_GUARD
            or a.get("quarantined_guard_run_id")!=QUARANTINED_RUN or a.get("max_new_processes")!=28 or a.get("physical_cap")!=50
            or a.get("accepted48")!=48 or a.get("raw49")!=49 or a.get("no_new_health") is not True
            or a.get("affected_pair",{}).get("recovery_attempt")!=1):
        raise ValueError("A6_exact_authorized_recovery_contract_mismatch")
    data=(root/"cost/measurements.jsonl").read_bytes()
    prefix=b"".join(data.splitlines(keepends=True)[:21])
    if sha(prefix)!=source_hash(a,"cost/measurements.jsonl"): raise ValueError("A6_original21_prefix_changed")
    original=[json.loads(line) for line in prefix.splitlines()]
    if len(original)!=21 or original[-1].get("record_sha256")!=ORPHAN or len({r["record_sha256"] for r in original})!=21:
        raise ValueError("A6_original21_identity_mismatch")
    orphan=original[-1]
    if (orphan["workload_id"],orphan["round"],orphan["candidate_id"],orphan.get("block_attempt",0))!=(*AFFECTED,"describe-select-cost",0):
        raise ValueError("A6_original_orphan_identity_mismatch")
    old_log=resolve_log(orphan,root)
    if sha(old_log.read_bytes())!=source_hash(a,"guards/cost_01.jsonl"): raise ValueError("A6_reviewed_guard_log_changed")
    events=[e for e in read_guard(old_log) if e.get("guard_id")==OLD_GUARD]
    quarantined=[e for e in events if e.get("event")=="quarantined_run" and (e.get("active_run") or {}).get("guard_run_id")==QUARANTINED_RUN]
    if len(quarantined)!=1: raise ValueError("A6_exact_quarantined_run_missing")
    if sha(canonical(quarantined[0]["quarantined_row"]))!=a["quarantined_row_sha256"]:
        raise ValueError("A6_quarantined_evidence_hash_changed")
    incident_path=root/"guards/cost_01_incidents.jsonl"
    if sha(incident_path.read_bytes())!=a["original_guard_incidents_sha256"]:
        raise ValueError("A6_original_incident_log_changed")
    if guard_id==OLD_GUARD: raise ValueError("A6_requires_fresh_guard_quiet_start")
    # A second recovery guard, including a prior zero-row refusal, is not authorized.
    for path in (root/"guards").glob("*.jsonl"):
        for event in read_guard(path):
            if event.get("event")!="guard_start": continue
            argv=event.get("script_args",[])
            stage=argv[argv.index("--stage")+1] if "--stage" in argv and argv.index("--stage")+1<len(argv) else None
            if stage=="cost" and event.get("guard_id") not in (OLD_GUARD,guard_id):
                raise ValueError("A6_only_one_new_cost_guard_attempt_allowed")
    archive=root/"cost/engineering_archive/pair_audits_before_A6.json"
    audits_bytes=archive.read_bytes() if archive.exists() else (root/"cost/pair_audits.json").read_bytes()
    if sha(audits_bytes)!=source_hash(a,"cost/pair_audits.json"): raise ValueError("A6_original10_pair_audits_changed")
    original_pairs=json.loads(audits_bytes)
    if len(original_pairs)!=10 or any(p.get("ok") is not True for p in original_pairs): raise ValueError("A6_requires_exact10_completed_original_pairs")
    records={}
    for row in original:
        validate_row(row,manifest["gpu_uuid"],root)
        records.setdefault((row["workload_id"],row["round"]),[]).append(row)
    selected=[]
    for pair in original_pairs:
        key=(pair["workload_id"],pair["round"])
        rows=records.get(key,[])
        ordered=sorted(rows,key=lambda r:modes(key[1]).index(r["candidate_id"]))
        if len(ordered)!=2 or [r["candidate_id"] for r in ordered]!=modes(key[1]): raise ValueError("A6_original_pair_incomplete")
        full=next(r["result"] for r in ordered if r["candidate_id"]=="full-search-cost")
        described=next(r["result"] for r in ordered if r["candidate_id"]=="describe-select-cost")
        quality=audit_pair(full,described,full["candidate_order_ids"],full["selected_id"])
        if not quality["ok"] or any(pair.get(k)!=v for k,v in quality.items()): raise ValueError("A6_original_pair_reaudit_failed")
        selected.append(dict(workload_id=key[0],round=key[1],block_attempt=0,candidate_ids=modes(key[1]),
                             record_hashes=[r["record_sha256"] for r in ordered],quality=quality))
    accepted=[h for p in selected for h in p["record_hashes"]]
    if len(set(accepted))!=20 or ORPHAN in accepted: raise ValueError("A6_original_closed_pair_partition_failed")
    projection=[{key:p[key] for key in ("workload_id","round","block_attempt","candidate_ids","record_hashes")} for p in selected]
    if projection!=a["preserved_pair_record_hashes"]: raise ValueError("A6_preserved_pair_membership_changed")
    archived_status=root/"cost/engineering_archive/status_before_A6.json"
    status_bytes=archived_status.read_bytes() if archived_status.exists() else original_status_bytes
    if status_bytes is None: status_bytes=(root/"cost/status.json").read_bytes()
    if sha(status_bytes)!=a["source_status_sha256"]: raise ValueError("A6_original_stopped_status_changed")
    candidate_path=root/"cost/candidate_orders/n16384_b2048_r1.tsv"
    if sha(candidate_path.read_bytes())!=a["affected_pair"]["candidate_order_sha256"]:
        raise ValueError("A6_affected_candidate_order_file_changed")
    admission=dict(schema_version="A6-cost-recovery-admission/v1",study_id=manifest["study_id"],addendum_file=str(addendum_path),
        A6_sha256=sha(addendum_path.read_bytes()),original_prefix_sha256=sha(prefix),original_prefix_bytes=len(prefix),
        original_guard_id=OLD_GUARD,original_guard_log_sha256=sha(old_log.read_bytes()),quarantined_guard_run_id=QUARANTINED_RUN,
        engineering_only_record_hashes=[ORPHAN],preserved_record_hashes=accepted,preserved_pairs=selected,
        affected_pair=dict(workload_id=AFFECTED[0],round=AFFECTED[1],block_attempt=1),recovery_guard_id=guard_id,
        accepted_processes_target=48,accepted_pairs_target=24,raw_records_cap=49,physical_process_cap=50,new_process_cap=28,
        original_pair_audits_sha256=sha(audits_bytes),original_pair_audits_bytes=audits_bytes.decode("utf-8"))
    admission.update(original_status_sha256=sha(status_bytes),original_status_bytes=status_bytes.decode("utf-8"),
                     affected_candidate_order_sha256=sha(candidate_path.read_bytes()),affected_input_seed=a["affected_pair"]["input_seed"],
                     original_guard_incidents_sha256=a["original_guard_incidents_sha256"])
    admission.update(executable_sha256=manifest["executable_sha256"],gpu_uuid=manifest["gpu_uuid"],original_design_sha256=manifest["design_sha256"])
    return admission


def cost_session_allowed(row,root,admission):
    path=resolve_log(row,root)
    if completed_session(path,row["guard_id"]): return True
    return bool(admission and row.get("guard_id")==admission["original_guard_id"]
        and row.get("record_sha256") in admission["preserved_record_hashes"]
        and sha(path.read_bytes())==admission["original_guard_log_sha256"])


def verify_recovery_admission(root):
    root=Path(root)
    stored=json.loads((root/"cost_recovery_admission.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    proof=admit_recovery(root/"research_addendum_A6.json",root/"research_design.json",root,manifest,stored["recovery_guard_id"])
    if any(stored.get(k)!=v for k,v in proof.items() if k!="addendum_file"):
        raise ValueError("A6_immutable_admission_does_not_reconstruct")
    snapshot=root/"cost/controller_snapshots"/(stored["recovery_controller_sha256"]+".json")
    controller=json.loads(snapshot.read_text(encoding="utf-8"))
    if (sha(canonical(controller["files"]))!=stored["recovery_controller_sha256"]
            or controller["files"].get("gpu_guard.py")!=stored["deployed_guard_sha256"]):
        raise ValueError("A6_deployed_controller_guard_provenance_mismatch")
    return stored


def selected_cost_rows(root,require_complete=False,require_sessions=False):
    root=Path(root)
    raw=read_jsonl(root/"cost/measurements.jsonl")
    selection=json.loads((root/"cost/accepted_pairs.json").read_text(encoding="utf-8"))
    admission_path=root/"cost_recovery_admission.json"
    admission=verify_recovery_admission(root) if admission_path.exists() else None
    if admission and selection.get("admission_sha256")!=sha(admission_path.read_bytes()): raise ValueError("cost_pair_selection_admission_hash_changed")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    design=json.loads((root/"research_design.json").read_text(encoding="utf-8"))
    expected={(wid,k) for wid in design["cost_profile"]["workloads"] for k in range(3)}
    if admission:
        if len(raw)>49 or len(raw)<21: raise ValueError("A6_raw_ledger_count_outside_contract")
        for row in raw[21:]:
            key=(row["workload_id"],row["round"])
            expected_attempt=1 if key==AFFECTED else 0
            if key not in expected or row.get("block_attempt",0)!=expected_attempt or row.get("guard_id")!=admission["recovery_guard_id"]:
                raise ValueError("A6_new_rows_must_use_only_unique_recovery_guard_and_fixed_attempt")
    if len({r.get("record_sha256") for r in raw})!=len(raw): raise ValueError("duplicate_raw_cost_record_hash")
    by_hash={r["record_sha256"]:r for r in raw}
    accepted=[];seen=set()
    for pair in selection["pairs"]:
        key=(pair["workload_id"],pair["round"])
        if key not in expected or key in seen: raise ValueError("duplicate_or_out_of_design_cost_pair")
        seen.add(key)
        attempt=1 if admission and key==AFFECTED else 0
        if pair["block_attempt"]!=attempt or pair["candidate_ids"]!=modes(key[1]) or len(pair["record_hashes"])!=2:
            raise ValueError("cost_pair_attempt_or_order_mismatch")
        rows=[by_hash[h] for h in pair["record_hashes"]]
        for i,row in enumerate(rows):
            if ((row["workload_id"],row["round"])!=key or row.get("block_attempt",0)!=attempt or row["candidate_id"]!=modes(key[1])[i]
                    or row.get("phase")!="cost" or row.get("study_id")!=manifest["study_id"]): raise ValueError("cost_pair_row_identity_mismatch")
            if admission and row["record_sha256"] in admission["engineering_only_record_hashes"]: raise ValueError("engineering_orphan_cannot_enter_cost_estimators")
            validate_row(row,manifest["gpu_uuid"],root)
            if require_sessions and not cost_session_allowed(row,root,admission): raise ValueError("cost_pair_session_not_admissible")
            if row.get("input_seed")!=input_seed(design,"cost",key[0],key[1]): raise ValueError("cost_pair_frozen_input_seed_changed")
        by_mode={row["candidate_id"]:row["result"] for row in rows}
        order=sorted(c["id"] for c in design["candidates"])
        random.Random(input_seed(design,"cost/candidate_order",key[0],key[1])).shuffle(order)
        selected=json.loads((root/"screen_selection.json").read_text(encoding="utf-8"))["workloads"][key[0]]["winner"]
        quality=audit_pair(by_mode["full-search-cost"],by_mode["describe-select-cost"],order,selected)
        if quality!=pair["quality"]: raise ValueError("cost_selected_pair_quality_does_not_recompute")
        if admission and key in {(p["workload_id"],p["round"]) for p in admission["preserved_pairs"]}:
            original=next(p for p in admission["preserved_pairs"] if (p["workload_id"],p["round"])==key)
            if pair!=original: raise ValueError("A6_preserved_pair_membership_changed")
        accepted.extend(rows)
    if require_complete and (seen!=expected or len(accepted)!=48): raise ValueError("accepted_cost_matrix_not_complete48")
    if len({r["record_sha256"] for r in accepted})!=len(accepted): raise ValueError("cost_record_reused_between_pairs")
    if require_complete and admission:
        if len(raw)!=49 or {r["record_sha256"] for r in raw}-{r["record_sha256"] for r in accepted}!={ORPHAN}:
            raise ValueError("A6_complete_ledger_requires_raw49_selected48_exact_orphan")
    gate_path=root/"cost/gate.json"
    if gate_path.exists():
        gate=json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("accepted_pair_selection_sha256")!=sha((root/"cost/accepted_pairs.json").read_bytes()):
            raise ValueError("cost_gate_pair_selection_hash_changed")
    elif require_complete and require_sessions: raise ValueError("completed_cost_gate_missing")
    return accepted,selection,admission
