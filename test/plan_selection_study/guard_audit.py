"""Audit exact guard/run linkage without contacting a GPU."""
import json
from pathlib import Path


def read_guard(path):
    data=path.read_bytes()
    lines=data.splitlines()
    if data and not data.endswith(b"\n"):
        lines=lines[:-1]  # A live monitor may be appending; never edit its log.
    return [json.loads(line) for line in lines if line.strip()]


def resolve_log(row,root=None):
    path=Path(row.get("guard_log",""))
    if path.is_file():
        return path
    if root:
        text=row.get("guard_log","").replace("\\","/")
        suffix=text.split("/guards/",1)[-1] if "/guards/" in text else text.rsplit("/",1)[-1]
        local=Path(root)/"guards"/suffix
        if local.is_file():
            return local
    raise ValueError("guard log not found")


def audit_row(row,gpu_uuid,root=None):
    errors=[]
    if row.get("guard_clean") is not True or row.get("guard_gpu_uuid")!=gpu_uuid:
        return dict(ok=False,errors=["missing_clean_guard_linkage"])
    try:
        events=read_guard(resolve_log(row,root))
    except (OSError,ValueError,json.JSONDecodeError) as error:
        return dict(ok=False,errors=["guard_log:"+str(error)])
    session=[e for e in events if e.get("guard_id")==row.get("guard_id")]
    run_id=row.get("guard_run_id")
    starts=[e for e in session if e.get("event")=="run_one_start" and e.get("active_run",{}).get("guard_run_id")==run_id]
    ends=[e for e in session if e.get("event")=="run_one_end" and e.get("guard_run_id")==run_id and e.get("clean") is True]
    if len(starts)!=1 or len(ends)!=1 or not run_id:
        errors.append("missing_unique_guard_run_boundaries")
    if not any(e.get("event")=="guard_start" for e in session) or not any(e.get("event")=="idle_complete" for e in session):
        errors.append("missing_guard_start_or_quiet_gate")
    samples=[e for e in session if e.get("event")=="sample" and (e.get("active_run") or {}).get("guard_run_id")==run_id]
    boundaries={e.get("boundary") for e in samples}
    if not {"before","after"}<=boundaries:
        errors.append("missing_before_after_samples")
    for sample in samples:
        if (sample.get("ok") is not True or sample.get("foreign_compute_pids") or sample.get("violation_reasons")
                or sample.get("monitor_error") is not None or sample.get("gpu_uuid")!=gpu_uuid
                or sample.get("gpu",{}).get("memory_used_mib",float("inf"))>2048):
            errors.append("unclean_guard_sample")
    return dict(ok=not errors,errors=errors,guard_id=row.get("guard_id"),guard_run_id=run_id,
                samples=len(samples),boundaries=sorted(str(x) for x in boundaries))


def completed_session(log,guard_id):
    events=[e for e in read_guard(log) if e.get("guard_id")==guard_id]
    return (any(e.get("event")=="guard_complete" and e.get("success") is True and e.get("clean") is True
                and e.get("script_exit_code")==0 for e in events)
            and not any(e.get("event")=="guard_abort" for e in events))
