"""Audit same-context construction pipelines and actual describe/full equivalence."""
from trace_audit import audit_trace,plan_fingerprints


def audit_pipeline(summary,expected_ids,selected_id):
    errors=[]
    mode=summary.get("mode")
    if mode not in ("full-search-cost","describe-select-cost"):
        errors.append("unknown_pipeline_mode")
    apps=summary.get("apps",[])
    expected=list(expected_ids)+([selected_id] if mode=="describe-select-cost" else [])
    if summary.get("candidate_order_ids")!=list(expected_ids) or [a.get("candidate_id") for a in apps]!=expected:
        errors.append("pipeline_candidate_order_or_coverage_mismatch")
    if summary.get("init_rounds")!=1 or not summary.get("context_id"):
        errors.append("pipeline_context_or_round_unit_mismatch")
    if summary.get("selected_id")!=selected_id:
        errors.append("selected_id_mismatch")
    for index,app in enumerate(apps):
        result=app.get("result",{})
        expected_role=("full" if mode=="full-search-cost" else ("describe" if index<len(expected_ids) else "materialize_selected"))
        if app.get("role")!=expected_role or result.get("shared_context_verified") is not True:
            errors.append(f"app_role_or_shared_context_not_verified:{index}")
        if app.get("sequence_index")!=index or app.get("round")!=0 or app.get("pipeline")!=mode:
            errors.append(f"app_sequence_mismatch:{index}")
        if result.get("status")!="ok":
            errors.append(f"app_failed:{index}:{app.get('candidate_id')}")
        trace=audit_trace(result.get("init_trace",{}))
        errors.extend(f"app{index}:{error}" for error in trace["errors"])
        kernels=result.get("precompile_kernels",[])
        uploads=result.get("structural_features",{}).get("numAxisUploads")
        if uploads is None and kernels:
            uploads=kernels[0].get("canonical_descriptor",{}).get("numAxisUploads")
        if not isinstance(uploads,int) or uploads<1 or len(kernels)!=uploads:
            errors.append(f"app_kernel_coverage_mismatch:{index}")
        counters=result.get("init_trace",{}).get("counters",{})
        if app.get("role")=="describe":
            if any(counters.get(k)!=0 for k in ("nvrtc_compile","module_load","driver_kernel_launch")):
                errors.append(f"describe_side_effect_counter_nonzero:{index}")
        else:
            if counters.get("nvrtc_compile")!=uploads or counters.get("module_load")!=uploads or counters.get("driver_kernel_launch")!=0:
                errors.append(f"full_positive_counter_failed:{index}")
            if any(k.get("compiler_options_match") is not True or k.get("actual_compiler_options")!=k.get("compiler_input",{}).get("options") for k in kernels):
                errors.append(f"full_compiler_options_failed:{index}")
        try:
            plan_fingerprints(result)
        except (KeyError,TypeError,ValueError) as error:
            errors.append(f"app_fingerprint_unavailable:{index}:{error}")
    return dict(ok=not errors,errors=errors,apps=len(apps),context_id=summary.get("context_id"),
                scope="One fresh process per pipeline; all candidate apps inside it share one context.")


def audit_pair(full,described,expected_ids,selected_id):
    a=audit_pipeline(full,expected_ids,selected_id)
    b=audit_pipeline(described,expected_ids,selected_id)
    errors=["full:"+x for x in a["errors"]]+["describe:"+x for x in b["errors"]]
    if full.get("context_id")==described.get("context_id"):
        errors.append("paired_pipelines_must_use_fresh_distinct_contexts")
    full_apps={x["candidate_id"]:x["result"] for x in full.get("apps",[]) if x.get("role")=="full"}
    desc_apps={x["candidate_id"]:x["result"] for x in described.get("apps",[]) if x.get("role")=="describe"}
    comparisons=[]
    for cid in expected_ids:
        if cid not in full_apps or cid not in desc_apps:
            errors.append("missing_equivalence_candidate:"+cid)
            continue
        try:
            fp,dp=plan_fingerprints(full_apps[cid]),plan_fingerprints(desc_apps[cid])
            equal=fp["canonical"]==dp["canonical"] and fp["source_compiler"]==dp["source_compiler"]
            comparisons.append(dict(candidate_id=cid,canonical_equal=fp["canonical"]==dp["canonical"],
                                    source_compiler_equal=fp["source_compiler"]==dp["source_compiler"]))
            if not equal:
                errors.append("describe_full_inequivalent:"+cid)
        except (KeyError,TypeError,ValueError) as error:
            errors.append("equivalence_unavailable:"+cid+":"+str(error))
    materialized=[a for a in described.get("apps",[]) if a.get("role")=="materialize_selected"]
    if len(materialized)!=1 or materialized[0].get("candidate_id")!=selected_id:
        errors.append("missing_unique_selected_materialization")
    elif selected_id in full_apps and selected_id in desc_apps:
        try:
            rebuilt=plan_fingerprints(materialized[0]["result"])
            original=plan_fingerprints(full_apps[selected_id])
            drafted=plan_fingerprints(desc_apps[selected_id])
            if any(rebuilt[field]!=original[field] or rebuilt[field]!=drafted[field] for field in ("canonical","source_compiler")):
                errors.append("selected_materialization_changed_plan_or_source")
        except (KeyError,TypeError,ValueError) as error:
            errors.append("selected_materialization_unavailable:"+str(error))
    return dict(ok=not errors,errors=errors,comparisons=comparisons,
                interpretation="Actual describe-all plus known-choice construction component, not an online selector demonstration.")
