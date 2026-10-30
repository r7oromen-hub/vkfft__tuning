#!/usr/bin/env python3
"""Fixed, CPU-only auxiliary H4 experiment on audited new-study screen blocks."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from study_common import TUNING_FIELDS,canonical,load_design,sha,write_json


MODEL=dict(n_estimators=256,max_depth=8,min_samples_leaf=2,max_features=1.0,random_state=20261027)
ABLATIONS=("workload_knobs","workload_structure","workload_knobs_structure")
# Explicit allowlist of values captured before compilation; no hashes or compiled attributes.
DESCRIPTOR_FIELDS=("axis","upload","FFTdim","actualNumBatches","resolved_numberBatches","numAxisUploads",
    "fftDim","fft_dim_full","precision","complexSize","inverse","actualInverse","localSize","axisBlock",
    "groupedBatch","numStages","stageRadix","logical_registers_per_thread","min_logical_registers_per_thread",
    "registers_per_radix","usedLocRegs","registerBoost","dynamic_shared_bytes","sharedMemSize","sharedMemSizePow2",
    "LUT","LUT_4step","reorderFourStep","readToRegisters","writeFromRegisters","stridedSharedLayout",
    "inputNumberByteSize","outputNumberByteSize","inputOffset","outputOffset","performR2C","performDCT",
    "performDST","useRader","useBluesteinFFT","bufferLUTSize","axisSplit","actualFFTSizePerAxis",
    "inputStride","outputStride","isInputFormatted","isOutputFormatted")


def flattened_numeric(value,prefix):
    result={}
    if isinstance(value,(list,tuple)):
        for i,v in enumerate(value): result.update(flattened_numeric(v,f"{prefix}/{i}"))
    elif isinstance(value,(int,float)) and math.isfinite(value): result[prefix]=float(value)
    elif value is not None:
        raise ValueError("nonnumeric_precompile_feature:"+prefix)
    return result


def feature_sets(design,row):
    """Only workload values, frozen requested knobs and allowlisted precompile fields."""
    w=next(w for w in design["workloads"] if w["id"]==row["workload_id"])
    c=next(c for c in design["candidates"] if c["id"]==row["candidate_id"])
    base=dict(n=float(w["n"]),batch=float(w["batch"]))
    knobs={"request/"+k:float(c.get(k,0)) for k in TUNING_FIELDS}
    descriptors={}
    kernels=sorted(row["result"].get("precompile_kernels",[]),key=lambda k:(k["axis"],k["upload"]))
    if not kernels: raise ValueError("missing_precompile_descriptors")
    for index,kernel in enumerate(kernels):
        descriptor=kernel["canonical_descriptor"]
        for field in DESCRIPTOR_FIELDS:
            if field in descriptor:
                descriptors.update(flattened_numeric(descriptor[field],f"kernel/{index}/{field}"))
    return [dict(base,**knobs),dict(base,**descriptors),dict(base,**knobs,**descriptors)]


def prepare_rows(design,screen_rows):
    by=collections.defaultdict(list)
    for row in screen_rows: by[row.get("workload_id")].append(row)
    expected={c["id"] for c in design["candidates"]}
    records=[]
    coverage=[]
    excluded=[]
    for w in design["workloads"]:
        wid=w["id"]
        raw=by.get(wid,[])
        ids=[r.get("candidate_id") for r in raw]
        if (len(raw)!=len(expected) or set(ids)!=expected or len(set(ids))!=len(ids)
                or any(r.get("phase")!="screen" or r.get("round")!=0 or r.get("guard_clean") is not True
                       or r.get("guard_gpu_uuid")!=design["scope"]["device_uuid"]
                       or r.get("study_id")!=design["study_id"] for r in raw)
                or len({r.get("block_attempt",0) for r in raw})!=1):
            excluded.append(dict(workload_id=wid,reason="incomplete_or_not_verified_clean_screen_block"))
            continue
        legal={r["candidate_id"]:r for r in raw if r.get("status")=="ok" and r.get("selection_eligible",True) and r["candidate_id"]!="default_replica"}
        if "default" not in legal:
            excluded.append(dict(workload_id=wid,reason="no_valid_default"));continue
        baseline=legal["default"]["result"].get("exec_ms_median")
        local=[]
        try:
            for cid,row in sorted(legal.items()):
                latency=row["result"].get("exec_ms_median")
                if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in (latency,baseline)):
                    raise ValueError("invalid_latency_label")
                local.append(dict(workload_id=wid,n=w["n"],batch=w["batch"],candidate_id=cid,time=latency,default_time=baseline,
                    features=feature_sets(design,row),record_sha256=row["record_sha256"]))
        except (ValueError,TypeError,KeyError) as error:
            excluded.append(dict(workload_id=wid,reason="feature_or_label_schema:"+str(error)));continue
        coverage.append(dict(workload_id=wid,n=w["n"],batch=w["batch"],legal_candidates=sorted(legal),
                             excluded_candidates=sorted(expected-set(legal)-{"default_replica"})))
        records.extend(local)
    # N is the grouping unit: every declared batch at that N must have a clean complete block.
    covered={x["workload_id"] for x in coverage}
    incomplete_n={w["n"] for w in design["workloads"] if w["id"] not in covered}
    for x in coverage:
        if x["n"] in incomplete_n: excluded.append(dict(workload_id=x["workload_id"],reason="another_declared_batch_missing_from_exact_N_group"))
    return ([r for r in records if r["n"] not in incomplete_n],
            [x for x in coverage if x["n"] not in incomplete_n],excluded)


def train_global(train):
    """Candidate valid across all training workloads; no held-out timing used."""
    by=collections.defaultdict(dict)
    for r in train: by[r["workload_id"]][r["candidate_id"]]=r
    common=set.intersection(*(set(rows) for rows in by.values()))
    scores={cid:statistics.mean(math.log(rows[cid]["time"]/rows[cid]["default_time"]) for rows in by.values()) for cid in common}
    winner=min(scores,key=lambda cid:(scores[cid],cid!="default",cid))
    return winner,scores


def metrics(chosen,group,shortlist):
    fastest=min(r["time"] for r in group)
    ratio=chosen["time"]/fastest
    speedup=chosen["default_time"]/chosen["time"]
    return dict(selected_candidate=chosen["candidate_id"],speedup=speedup,regret=ratio,
                relative_excess_latency=ratio-1,within2pct=ratio<=1.02,within5pct=ratio<=1.05,
                regression=max(0.,1/speedup-1),top2_hypothetical_shortlist_oracle_regret=min(r["time"] for r in shortlist[:2])/fastest,
                top4_hypothetical_shortlist_oracle_regret=min(r["time"] for r in shortlist[:4])/fastest)


def aggregate(rows):
    available=[r for r in rows if r.get("available",True)]
    if not available: return dict(workloads=0,unavailable=len(rows))
    return dict(workloads=len(available),unavailable=len(rows)-len(available),
        equal_workload_geometric_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in available)),
        geometric_regret=math.exp(statistics.mean(math.log(r["regret"]) for r in available)),
        within2pct_fraction=statistics.mean(r["within2pct"] for r in available),within5pct_fraction=statistics.mean(r["within5pct"] for r in available),
        worst_regression=max(r["regression"] for r in available),worst_speedup=min(r["speedup"] for r in available),
        geometric_top2_shortlist_regret=math.exp(statistics.mean(math.log(r["top2_hypothetical_shortlist_oracle_regret"]) for r in available)),
        geometric_top4_shortlist_regret=math.exp(statistics.mean(math.log(r["top4_hypothetical_shortlist_oracle_regret"]) for r in available)))


def run_learning(design,screen_rows,*,evidence_verified=False,frozen_global_candidate=None):
    result=dict(schema_version="plan-selection-learning/v1",study_id=design["study_id"],model=MODEL,
                legacy_data_used=False,replication_unit="workload, grouped by exact N",auxiliary_only=True,
                evaluation_condition="Selection conditional on the offline execution-, numerical- and semantic-validated eligible candidate set, including held-out workloads.",
                heldout_validity_used_for_candidate_filter=True,heldout_validity_used_for_filtering=True,
                validity_not_a_model_feature=True,validity_prediction_evaluated=False,invalid_candidate_detection_evaluated=False,
                deployment_end_to_end_evaluated=False,validation_fallback_costs_measured=False)
    if not evidence_verified:
        return dict(result,status="not_run_unverified_evidence",reason="Requires accepted blocks from the report evidence audit.")
    records,coverage,excluded=prepare_rows(design,screen_rows)
    lengths=sorted({r["n"] for r in records})
    result.update(coverage=coverage,excluded_workloads=excluded,distinct_N=len(lengths),training_rows=len(records))
    if len(lengths)<12:
        return dict(result,status="not_run_coverage_gate",reason="At least twelve exact-N groups from complete clean screening blocks required.")
    import sklearn
    import numpy
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction import DictVectorizer
    # Freeze assignment from the declared N set, even when coverage excludes a group.
    folds={n:i%4 for i,n in enumerate(sorted({w["n"] for w in design["workloads"]}))}
    result["folds"]=folds
    result["runtime_versions"]=dict(sklearn=sklearn.__version__,numpy=numpy.__version__)
    result["feature_schema"]=dict(workload=["n","batch"],requested=list(TUNING_FIELDS),precompile_allowlist=list(DESCRIPTOR_FIELDS),
        excluded=["R","timing","physical_registers","compiled_local_memory","validity","candidate_ID","source_hash","process_order"])
    result["source_record_hashes"]=sorted(r["record_sha256"] for r in records)
    outcomes=[]
    baselines={"default":[],"training_only_global":[],"frozen_S1_global_diagnostic":[],"screen_oracle_diagnostic":[]}
    for target in ("log_candidate_over_default","log_absolute_latency"):
        for ablation,name in enumerate(ABLATIONS):
            predictions=[]
            fold_details=[]
            for fold in range(4):
                train=[r for r in records if folds[r["n"]]!=fold]
                test=[r for r in records if folds[r["n"]]==fold]
                if not test:
                    fold_details.append(dict(fold=fold,train_N=sorted({r["n"] for r in train}),test_N=[],status="empty_heldout_fold_skipped_fixed_assignment"))
                    continue
                vectorizer=DictVectorizer(sparse=False)
                x=vectorizer.fit_transform([r["features"][ablation] for r in train])
                y=[math.log(r["time"]/r["default_time"] if target=="log_candidate_over_default" else r["time"]) for r in train]
                model=ExtraTreesRegressor(**MODEL)
                model.fit(x,y)
                predicted=model.predict(vectorizer.transform([r["features"][ablation] for r in test]))
                global_id,scores=train_global(train)
                fold_details.append(dict(fold=fold,train_N=sorted({r["n"] for r in train}),test_N=sorted({r["n"] for r in test}),
                    training_only_global_id=global_id,training_global_scores=scores,features=vectorizer.get_feature_names_out().tolist()))
                grouped=collections.defaultdict(list)
                for row,pred in zip(test,predicted): grouped[row["workload_id"]].append((float(pred),row))
                for wid,group in sorted(grouped.items()):
                    group.sort(key=lambda p:(p[0],p[1]["candidate_id"]!="default",p[1]["candidate_id"]))
                    ordered=[r for _,r in group]
                    metric=metrics(ordered[0],ordered,ordered)
                    predictions.append(dict(workload_id=wid,n=ordered[0]["n"],batch=ordered[0]["batch"],fold=fold,**metric))
                    if target=="log_candidate_over_default" and ablation==0:
                        for baseline in baselines:
                            if baseline=="default": chosen=next(r for r in ordered if r["candidate_id"]=="default")
                            elif baseline=="screen_oracle_diagnostic": chosen=min(ordered,key=lambda r:(r["time"],r["candidate_id"]!="default",r["candidate_id"]))
                            elif baseline=="frozen_S1_global_diagnostic": chosen=next((r for r in ordered if r["candidate_id"]==frozen_global_candidate),None)
                            else: chosen=next((r for r in ordered if r["candidate_id"]==global_id),None)
                            if chosen is None:
                                selected=frozen_global_candidate if baseline=="frozen_S1_global_diagnostic" else global_id
                                baselines[baseline].append(dict(workload_id=wid,fold=fold,available=False,selected_candidate=selected,
                                    reason="frozen_S1_global_manifest_not_supplied" if selected is None else "fixed_global_invalid_on_heldout_workload"))
                            else:
                                baselines[baseline].append(dict(workload_id=wid,fold=fold,available=True,**metrics(chosen,ordered,[chosen])))
            outcomes.append(dict(target=target,priority="primary" if target=="log_candidate_over_default" else "secondary_sensitivity",
                                 ablation=name,fold_details=fold_details,workloads=predictions,metrics=aggregate(predictions)))
    result.update(status="completed_auxiliary",outcomes=outcomes,baselines={k:dict(workloads=v,metrics=aggregate(v)) for k,v in baselines.items()},
        limitation="Fixed small domain on one GPU. Ranking is conditional on the offline validated eligible set, including held-out workloads. Validity is not a model feature, but held-out execution/numerical/semantic outcomes filter candidate choices. This does not evaluate precompile validity detection, avoiding runtime faults or end-to-end deployment. Successful compilation/description does not establish correct execution. Candidate-invalid held-out global choices remain unavailable, with coverage reported alongside metrics. Frozen S1 global and screen oracle use screen labels and are optimistic diagnostics, not leakage-free baselines. Shortlist oracle excludes online construction, validation, fallback and search costs.")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True,help="One explicit new-study result root, never a pilot directory")
    p.add_argument("--design",type=Path,default=Path(__file__).with_name("research_design.json"))
    args=p.parse_args()
    from report_helpers import read_evidence
    d=load_design(args.design)
    d["_sha256"]=sha(args.design.read_bytes())
    evidence=read_evidence(args.output,d)
    screen_errors=[e for e in evidence["errors"] if e.get("stage")=="screen" or "screen" in e.get("file","")]
    verified=evidence["identity_ok"] and not screen_errors
    finished=True
    for stage in ("screen","confirmation","cost","headline"):
        status_path=args.output/stage/"status.json"
        status=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if status.get("state") not in ("complete","complete_with_limitations"): finished=False
    if finished:
        selection_path=args.output/"screen_selection.json"
        selection=json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        value=run_learning(d,evidence["rows"]["screen"],evidence_verified=verified,frozen_global_candidate=selection.get("global_candidate"))
    else:
        value=dict(status="not_run_primary_sequence_pending",study_id=d["study_id"],auxiliary_only=True,legacy_data_used=False)
    target=args.output/"report"/"learning_check.json"
    # Preserve the first fixed analysis and reject a different second result.
    if target.exists():
        old=json.loads(target.read_text(encoding="utf-8"))
        if old!=value:
            if old.get("status")=="completed_auxiliary":
                raise ValueError("learning_output_already_exists_with_different_evidence; preserve first result")
            write_json(target.parent/"learning_attempts"/(sha(canonical(old))+".json"),old)
            write_json(target,value)
    else: write_json(target,value)
    print(json.dumps(dict(status=value["status"],distinct_N=value.get("distinct_N"),artifact=str(target)),indent=2))


if __name__=="__main__": main()
