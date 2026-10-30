"""Pure audits and fingerprints for instrumented planning records; no GPU calls."""
import collections
import hashlib
import json
import math


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def plan_fingerprints(result):
    kernels=sorted(result.get("precompile_kernels",[]),key=lambda k:(k["axis"],k["upload"]))
    if not kernels:
        raise ValueError("no precompile kernel descriptors")
    canonical=[dict(axis=k["axis"],upload=k["upload"],descriptor=k["canonical_descriptor"]) for k in kernels]
    source=[dict(axis=k["axis"],upload=k["upload"],source_sha256=k["source_sha256"],compiler_input=k["compiler_input"],
                 logical_binding_layout=k["canonical_descriptor"]) for k in kernels]
    structure=result.get("structural_features",{})
    first=kernels[0]["canonical_descriptor"]
    core=dict(numAxisUploads=structure.get("numAxisUploads",first.get("numAxisUploads")),
              axisSplit=structure.get("axisSplit",first.get("axisSplit")),kernels=[])
    for kernel in kernels:
        descriptor=kernel["canonical_descriptor"]
        core["kernels"].append(dict(axis=kernel["axis"],upload=kernel["upload"],
            **{field:descriptor.get(field) for field in ("fftDim","stageRadix","localSize","axisBlock","registerBoost")}))
    return dict(canonical=fingerprint(canonical),source_compiler=fingerprint(source),core=fingerprint(core),
                core_descriptor=core,canonical_descriptor=canonical,source_compiler_descriptor=source)


def audit_trace(trace,tolerance_ms=1e-6):
    """Validate a forest of synchronous nested spans, including separate cleanup."""
    spans=trace.get("spans",[])
    errors=[]
    if not spans:
        return dict(ok=False,errors=["missing_spans"],stage_totals={},root_inclusive_ms=0.,exclusive_sum_ms=0.)
    ids=[s["id"] for s in spans]
    if len(set(ids))!=len(ids):
        errors.append("duplicate_span_id")
    by_id={s["id"]:s for s in spans}
    for span in spans:
        visited=set()
        current=span["id"]
        while current in by_id and current!=-1:
            if current in visited:
                errors.append(f"parent_cycle:{span['id']}")
                break
            visited.add(current)
            current=by_id[current]["parent_id"]
    children=collections.defaultdict(list)
    roots=[]
    for span in spans:
        if span["end_ns"]<span["start_ns"]:
            errors.append(f"negative_duration:{span['id']}")
        if span["parent_id"]==-1:
            roots.append(span)
        elif span["parent_id"] not in by_id:
            errors.append(f"missing_parent:{span['id']}")
        else:
            parent=by_id[span["parent_id"]]
            children[parent["id"]].append(span)
            if parent["id"]==span["id"] or not(parent["start_ns"]<=span["start_ns"]<=span["end_ns"]<=parent["end_ns"]):
                errors.append(f"invalid_nesting:{span['id']}")
    totals=collections.defaultdict(lambda:dict(calls=0,inclusive_ms=0.,exclusive_ms=0.))
    sorted_roots=sorted(roots,key=lambda s:s["start_ns"])
    if any(a["end_ns"]>b["start_ns"] for a,b in zip(sorted_roots,sorted_roots[1:])):
        errors.append("overlapping_synchronous_roots")
    for span in spans:
        direct=sorted(children[span["id"]],key=lambda s:s["start_ns"])
        if any(a["end_ns"]>b["start_ns"] for a,b in zip(direct,direct[1:])):
            errors.append(f"overlapping_direct_children:{span['id']}")
        inclusive=(span["end_ns"]-span["start_ns"])/1e6
        exclusive=inclusive-sum((c["end_ns"]-c["start_ns"])/1e6 for c in direct)
        if not math.isclose(span["inclusive_ms"],inclusive,rel_tol=1e-10,abs_tol=tolerance_ms):
            errors.append(f"inclusive_mismatch:{span['id']}")
        if exclusive < -tolerance_ms or not math.isclose(span["exclusive_ms"],exclusive,rel_tol=1e-10,abs_tol=tolerance_ms):
            errors.append(f"exclusive_mismatch:{span['id']}")
        total=totals[span["stage"]]
        total["calls"]+=1
        total["inclusive_ms"]+=inclusive
        total["exclusive_ms"]+=exclusive
    for stage,total in totals.items():
        reported=trace.get("stage_totals",{}).get(stage)
        if reported is None or reported.get("calls")!=total["calls"]:
            errors.append(f"stage_calls_mismatch:{stage}")
            continue
        for metric in ("inclusive_ms","exclusive_ms"):
            if not math.isclose(reported[metric],total[metric],rel_tol=1e-10,abs_tol=tolerance_ms):
                errors.append(f"stage_total_mismatch:{stage}:{metric}")
    root_sum=sum((s["end_ns"]-s["start_ns"])/1e6 for s in roots)
    exclusive_sum=sum(t["exclusive_ms"] for t in totals.values())
    if not math.isclose(root_sum,exclusive_sum,rel_tol=1e-10,abs_tol=tolerance_ms):
        errors.append("forest_partition_mismatch")
    for counter in ("nvrtc_compile","module_load","driver_kernel_launch"):
        if trace.get("counters",{}).get(counter)!=totals.get(counter,{}).get("calls",0):
            errors.append(f"counter_mismatch:{counter}")
    return dict(ok=not errors,errors=errors,stage_totals=dict(totals),root_inclusive_ms=root_sum,
                exclusive_sum_ms=exclusive_sum,root_stages=[s["stage"] for s in roots],
                interpretation="Exclusive sums partition separate synchronous root spans; inclusive stage totals overlap and must not be added.")
