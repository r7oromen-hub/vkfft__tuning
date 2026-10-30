"""Deterministic study records and protocol helpers, usable without a GPU."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from trace_audit import audit_trace,plan_fingerprints


TUNING_FIELDS=("aim_threads","register_boost","use_lut","register_boost_nonpow2",
               "coalesced_memory","use_lut_4step","swap_to_2stage","swap_to_3stage")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def append_json(path,value):
    import os
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("ab") as f:
        f.write(canonical(value)+b"\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    data=path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"incomplete final JSONL line; preserve/recover explicitly: {path}")
    rows=[]
    for number,line in enumerate(data.splitlines(),1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {number}: {path}") from error
    return rows


def load_design(path):
    design=json.loads(path.read_text(encoding="utf-8"))
    freeze_path=path.with_name("research_freeze.json")
    if not freeze_path.is_file():
        raise ValueError("missing independent research_freeze.json")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("study_id")!=design["study_id"] or freeze.get("design_sha256")!=sha(path.read_bytes()):
        raise ValueError("design differs from preregistered freeze")
    ws=design["workloads"]
    cs=design["candidates"]
    if len({w["id"] for w in ws})!=len(ws) or len({c["id"] for c in cs})!=len(cs):
        raise ValueError("duplicate design workload/candidate IDs")
    if len(ws)*len(cs)!=design["budgets"]["screen_processes"]:
        raise ValueError("screen budget differs from frozen matrix")
    for w in ws:
        if (w["id"]!=f"n{w['n']}_b{w['batch']}" or w["input_bytes"]!=8*w["n"]*w["batch"]
                or w["input_bytes"]>design["scope"]["max_input_bytes"]):
            raise ValueError("invalid workload dimensions/cap")
    by_id={c["id"]:c for c in cs}
    for cid in ("default","default_replica"):
        if any(by_id[cid][field]!=0 for field in TUNING_FIELDS):
            raise ValueError("default/control must have the same empty tuning request")
    if any(c.get("disable_reorder_four_step",0)!=0 for c in cs):
        raise ValueError("natural-order task forbids disable_reorder_four_step")
    return design


def input_seed(design,phase,wid,round_no):
    text=f"{design['measurement']['seed_namespace']}:{phase}:{wid}:{round_no}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],"little")


def candidate_config(candidate):
    return {field:candidate[field] for field in TUNING_FIELDS if candidate[field]!=0}


def workload_order(design,phase,round_no=0):
    grouped={}
    for workload in design["workloads"]:
        grouped.setdefault(workload["n"],[]).append(workload)
    rng=random.Random(input_seed(design,phase,"N_blocks",round_no))
    lengths=sorted(grouped)
    rng.shuffle(lengths)
    result=[]
    for n in lengths:
        batches=sorted(grouped[n],key=lambda w:w["batch"])
        rng.shuffle(batches)
        result.extend(batches)
    return result


def block_candidates(design,wid,phase,round_no,candidate_ids=None):
    ids=list(candidate_ids or [c["id"] for c in design["candidates"]])
    middle=sorted(cid for cid in ids if cid not in ("default","default_replica"))
    random.Random(input_seed(design,phase+"/order",wid,round_no)).shuffle(middle)
    anchors=["default","default_replica"]
    if phase!="screen" and round_no%2:
        anchors.reverse()
    return [anchors[0]]+middle+[anchors[1]]


def record_key(row):
    return (row["phase"],row["workload_id"],row["candidate_id"],row["round"],row.get("block_attempt",0))


def execution_errors(result,workload,expected_samples=None,instrumented=True):
    errors=[]
    samples=result.get("exec_ms_samples",[])
    if result.get("status")!="ok":
        errors.append("harness_status:"+str(result.get("status")))
    if result.get("n")!=workload["n"] or result.get("batch")!=workload["batch"]:
        errors.append("workload_mismatch")
    if result.get("timing_mode")!="graph" or result.get("placement")!="outofplace":
        errors.append("execution_protocol_mismatch")
    if result.get("graph_repeats")!=workload["graph_repeats"]:
        errors.append("graph_repeats_mismatch")
    if (not samples or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in samples)
            or (expected_samples is not None and len(samples)!=expected_samples)):
        errors.append("invalid_samples")
    elif not math.isclose(statistics.median(samples),result.get("exec_ms_median",float("nan")),rel_tol=1e-10,abs_tol=1e-12):
        errors.append("median_mismatch")
    for name in ("single_forward_correctness","correctness"):
        check=result.get(name,{})
        if check.get("passed") is not True or check.get("criterion")!="criterion_v2":
            errors.append("numerical_failure:"+name)
    if result.get("single_forward_input_unchanged") is not True or result.get("input_unchanged") is not True:
        errors.append("input_changed_or_unchecked")
    if result.get("header_instrumentation") is not instrumented:
        errors.append("header_instrumentation_role_mismatch")
    if instrumented and result.get("descriptor_validation",{}).get("all_match") is not True:
        errors.append("precompile_postinit_descriptor_mismatch")
    kernels=result.get("precompile_kernels",[])
    uploads=result.get("structural_features",{}).get("numAxisUploads")
    counters=result.get("init_trace",{}).get("counters",{})
    if instrumented and (not isinstance(uploads,int) or uploads<1 or len(kernels)!=uploads
            or counters.get("nvrtc_compile")!=uploads or counters.get("module_load")!=uploads
            or counters.get("driver_kernel_launch")!=0):
        errors.append("full_initialization_counter_positive_control_failed")
    if instrumented and any(k.get("compiler_options_match") is not True or k.get("actual_compiler_options")!=k.get("compiler_input",{}).get("options") for k in kernels):
        errors.append("actual_compiler_options_mismatch")
    if (not isinstance(uploads,int) or result.get("graph_kernel_count")!=workload["graph_repeats"]*uploads
            or result.get("graph_event_record_count")!=2
            or result.get("graph_node_count")!=workload["graph_repeats"]*uploads+2):
        errors.append("bounded_domain_serialized_graph_count_failed")
    if instrumented:
        trace=audit_trace(result.get("init_trace",{}))
        errors.extend("trace:"+error for error in trace["errors"])
        try:
            plan_fingerprints(result)
        except (KeyError,ValueError,TypeError) as error:
            errors.append("missing_plan_fingerprint:"+str(error))
    elif not result.get("postinit_source_kernels"):
        errors.append("missing_plain_retained_postinit_sources")
    return errors


def process_drift(result):
    samples=result.get("exec_ms_samples",[])
    if len(samples)<4:
        return None
    quarter=max(1,len(samples)//4)
    return statistics.median(samples[-quarter:])/statistics.median(samples[:quarter])


def block_quality(rows,health=False):
    usable={row["candidate_id"]:row for row in rows if row["status"]=="ok"}
    errors=[]
    if "default" not in usable or "default_replica" not in usable:
        return dict(ok=False,errors=["missing_valid_default_or_replica"])
    a,b=usable["default"]["result"],usable["default_replica"]["result"]
    if plan_fingerprints(a)["source_compiler"]!=plan_fingerprints(b)["source_compiler"]:
        errors.append("default_replica_source_compiler_mismatch")
    ratio=a["exec_ms_median"]/b["exec_ms_median"]
    if abs(math.log(ratio))>math.log(1.05):
        errors.append("default_replica_ratio_exceeds_5pct")
    for row in rows:
        if row["status"]=="ok":
            drift=process_drift(row["result"])
            if drift is None or abs(math.log(drift))>math.log(1.05):
                errors.append("nonstationary_quartiles:"+row["candidate_id"])
    return dict(ok=not errors,errors=errors,default_over_replica=ratio,
                interpretation="Whole-block quality gate; never retain only a favorable row.")
