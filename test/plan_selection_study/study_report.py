#!/usr/bin/env python3
"""Audit retained evidence and report H1-H4 without invoking a GPU."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from guard_audit import audit_row
from selection_analysis import paired_effect
from study_common import canonical,load_design,read_jsonl,sha,write_json
from trace_audit import audit_trace,plan_fingerprints


def headline_test(ratios):
    logs=list(map(math.log,ratios))
    effect=math.exp(statistics.mean(logs))
    limited=max(logs)-min(logs)<=16*sys.float_info.epsilon*max(1,max(map(abs,logs)))
    if limited:
        return dict(speedup=effect,p=1.,interval95=None,inferential=False,reason="quantization_or_zero_variance")
    from scipy.stats import t
    se=statistics.stdev(logs)/math.sqrt(len(logs))
    mean=statistics.mean(logs)
    critical=float(t.ppf(.975,len(logs)-1))
    return dict(speedup=effect,p=float(t.sf((mean-math.log(1.02))/se,len(logs)-1)),
                interval95=[math.exp(mean-critical*se),math.exp(mean+critical*se)],inferential=True)


def holm(tests):
    ordered=sorted(range(len(tests)),key=lambda i:tests[i]["p"])
    previous=0.
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1.,tests[i]["p"]*(len(tests)-rank)))
        tests[i]["holm_p"]=previous
        tests[i]["clear_gain"]=bool(tests[i]["inferential"] and tests[i]["speedup"]>=1.05 and previous<.05)
    return tests


def audit_evidence(root,design):
    accepted={}
    ledger=[]
    errors=[]
    for stage in ("health","semantics","equivalence","cost_smoke","screen","confirmation","cost","headline"):
        rows=read_jsonl(root/stage/"measurements.jsonl")
        by_hash={r.get("record_sha256"):r for r in rows}
        for row in rows:
            hashed=sha(canonical({k:v for k,v in row.items() if k!="record_sha256"}))==row.get("record_sha256")
            guard=audit_row(row,design["scope"]["device_uuid"],root)
            clean=hashed and guard["ok"]
            if not clean: errors.append(dict(stage=stage,record=row.get("record_sha256"),hash_ok=hashed,guard=guard))
            ledger.append(dict(stage=stage,phase=row["phase"],workload_id=row["workload_id"],candidate_id=row["candidate_id"],round=row["round"],
                               block_attempt=row.get("block_attempt",0),status=row["status"],clean=clean,validation_errors=row.get("validation_errors",[])))
        stage_rows=[]
        if stage in ("screen","confirmation","headline"):
            for path in sorted((root/stage/"blocks").glob("*.json")):
                block=json.loads(path.read_text(encoding="utf-8"))
                if not block["quality"]["ok"]: continue
                rs=[by_hash[h] for h in block["record_hashes"]]
                if not all(sha(canonical({k:v for k,v in r.items() if k!="record_sha256"}))==r["record_sha256"] and audit_row(r,design["scope"]["device_uuid"],root)["ok"] for r in rs):
                    continue
                stage_rows.extend(rs)
        else:
            stage_rows=rows
        accepted[stage]=stage_rows
    return accepted,ledger,errors


def learning(design,screen):
    """Fixed grouped-N ablations; no postcompile or timing inference features."""
    by=collections.defaultdict(dict)
    for r in screen:
        if r["status"]=="ok" and r["candidate_id"]!="default_replica": by[r["workload_id"]][r["candidate_id"]]=r
    expected={c["id"] for c in design["candidates"] if c["id"]!="default_replica"}
    complete={wid:rows for wid,rows in by.items() if set(rows)==expected}
    lengths=sorted({r["workload"]["n"] for rows in complete.values() for r in rows.values()})
    if len(lengths)<12: return dict(status="not_run_coverage_gate",distinct_N=len(lengths),complete_workloads=sorted(complete))
    try:
        import numpy as np
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_extraction import DictVectorizer
    except ImportError as error:
        return dict(status="not_run_dependency_unavailable",reason=str(error))
    requested={c["id"]:c for c in design["candidates"]}
    def flatten(v,prefix=""):
        output={}
        if isinstance(v,dict):
            for k,x in v.items(): output.update(flatten(x,prefix+"/"+k))
        elif isinstance(v,list):
            for k,x in enumerate(v): output.update(flatten(x,prefix+"/"+str(k)))
        elif isinstance(v,(int,float)) and not isinstance(v,bool): output[prefix]=v
        elif isinstance(v,str): output[prefix]=v
        return output
    records=[]
    for wid,rows in sorted(complete.items()):
        for cid,r in sorted(rows.items()):
            base=dict(n=r["workload"]["n"],batch=r["workload"]["batch"])
            knobs=flatten({k:v for k,v in requested[cid].items() if k not in ("id","role")},"request")
            desc=flatten(plan_fingerprints(r["result"])["canonical_descriptor"],"descriptor")
            records.append(dict(wid=wid,cid=cid,n=base["n"],time=r["result"]["exec_ms_median"],default=rows["default"]["result"]["exec_ms_median"],
                                features=[dict(base,**knobs),dict(base,**desc),dict(base,**knobs,**desc)]))
    outcomes=[]
    folds={n:i%4 for i,n in enumerate(lengths)}
    for target in ("log_relative","log_absolute"):
        for ablation,name in enumerate(("workload_knobs","workload_structure","workload_knobs_structure")):
            predictions=[]
            for fold in range(4):
                train=[r for r in records if folds[r["n"]]!=fold]
                test=[r for r in records if folds[r["n"]]==fold]
                vec=DictVectorizer(sparse=False)
                x=vec.fit_transform([r["features"][ablation] for r in train])
                y=np.array([math.log(r["time"]/r["default"] if target=="log_relative" else r["time"]) for r in train])
                model=ExtraTreesRegressor(n_estimators=256,max_depth=8,min_samples_leaf=2,max_features=1.0,random_state=20261027)
                model.fit(x,y)
                pred=model.predict(vec.transform([r["features"][ablation] for r in test]))
                groups=collections.defaultdict(list)
                for r,p in zip(test,pred): groups[r["wid"]].append((float(p),r))
                scores={cid:statistics.mean(math.log(r["time"]/r["default"]) for r in train if r["cid"]==cid) for cid in expected}
                global_id=min(scores,key=lambda c:(scores[c],c!="default",c))
                for wid,group in groups.items():
                    group.sort(key=lambda pr:(pr[0],pr[1]["cid"]!="default",pr[1]["cid"]))
                    winner=group[0][1]
                    fastest=min(r["time"] for _,r in group)
                    global_row=next(r for _,r in group if r["cid"]==global_id)
                    predictions.append(dict(workload_id=wid,fold=fold,candidate_id=winner["cid"],speedup=winner["default"]/winner["time"],
                        regret=winner["time"]/fastest,within2pct=winner["time"]<=1.02*fastest,within5pct=winner["time"]<=1.05*fastest,
                        training_only_global_id=global_id,global_speedup=global_row["default"]/global_row["time"],
                        default_speedup=1.,screen_oracle_speedup=winner["default"]/fastest,
                        top2_shortlist_oracle_regret=min(r["time"] for _,r in group[:2])/fastest,
                        top4_shortlist_oracle_regret=min(r["time"] for _,r in group[:4])/fastest))
            outcomes.append(dict(target=target,ablation=name,predictions=predictions,
                geometric_speedup=math.exp(statistics.mean(math.log(p["speedup"]) for p in predictions)),worst_speedup=min(p["speedup"] for p in predictions)))
    return dict(status="completed_auxiliary",folds=folds,complete_workloads=sorted(complete),outcomes=outcomes,
                limitation="One GPU, sixteen fixed N groups; shortlist oracle is hypothetical and excludes online construction/search costs.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--design",type=Path,default=Path(__file__).with_name("research_design.json"))
    p.add_argument("--learning",action="store_true",help="Run the fixed auxiliary H4 analysis once after H1-H3")
    args=p.parse_args()
    root=args.output
    d=load_design(args.design)
    rows,ledger,errors=audit_evidence(root,d)
    report=root/"report"
    report.mkdir(exist_ok=True)
    write_json(report/"attempt_validity_ledger.json",ledger)
    write_json(report/"audit_errors.json",errors)
    selection=json.loads((root/"screen_selection.json").read_text(encoding="utf-8")) if (root/"screen_selection.json").exists() else None
    structures=[]
    for r in rows["screen"]:
        if r["status"]!="ok": continue
        fp=plan_fingerprints(r["result"])
        structures.append(dict(workload_id=r["workload_id"],candidate_id=r["candidate_id"],requested=r.get("config",{}),
                               resolved=r["result"].get("resolved_config"),canonical=fp["canonical"],core=fp["core"],source_compiler=fp["source_compiler"],
                               descriptor=fp["canonical_descriptor"],compiled_resources=r["result"].get("kernel_attributes")))
    write_json(report/"structure_coverage.json",structures)
    effects=[]
    if selection:
        for w in d["workloads"]:
            wid=w["id"]
            rs=[r for r in rows["confirmation"] if r["workload_id"]==wid]
            winner=selection["workloads"][wid]["winner"]
            effect=paired_effect(rs,"default",winner,3)
            if effect: effects.append(dict(workload_id=wid,n=w["n"],batch=w["batch"],winner=winner,**effect))
    write_json(report/"independent_effects.json",effects)
    tests=[]
    if (root/"headline_selection.json").exists():
        headlines=json.loads((root/"headline_selection.json").read_text(encoding="utf-8"))
        for test in headlines["directional_tests"]:
            rs=[r for r in rows["headline"] if r["workload_id"]==test["workload_id"]]
            pair=paired_effect(rs,test["reference"],test["target"],headlines["rounds"])
            if pair: tests.append(dict(test,**headline_test(pair["ratios"]),ratios=pair["ratios"]))
            else: tests.append(dict(test,p=1.,speedup=None,inferential=False,reason="incomplete_headline_pair"))
        holm(tests)
    write_json(report/"headline_tests.json",tests)
    cost=[]
    for r in rows["cost"]:
        s=r.get("result",{})
        applications=[]
        for a in s.get("apps",[]):
            tr=audit_trace(a.get("result",{}).get("init_trace",{}))
            applications.append(dict(candidate_id=a["candidate_id"],role=a["role"],sequence_index=a["sequence_index"],
                                     stage_totals=tr["stage_totals"],root_inclusive_ms=tr["root_inclusive_ms"],trace_ok=tr["ok"],
                                     candidate_wall_ms=a.get("result",{}).get("candidate_wall_ms")))
        cost.append(dict(workload_id=r["workload_id"],round=r["round"],status=r["status"],mode=s.get("mode"),
                         context_pipeline_wall_ms=s.get("context_pipeline_wall_ms"),context_bootstrap_ms=s.get("context_bootstrap_ms"),
                         shared_buffer_stream_setup_ms=s.get("shared_buffer_stream_setup_ms"),shared_buffer_stream_cleanup_ms=s.get("shared_buffer_stream_cleanup_ms"),
                         apps=applications))
    write_json(report/"cost_breakdown.json",cost)
    if args.learning:
        h4=learning(d,rows["screen"])
        write_json(report/"learning.json",h4)
    gates={stage:json.loads((root/stage/"gate.json").read_text(encoding="utf-8")) if (root/stage/"gate.json").exists() else None for stage in rows}
    complete_screen=len({r["workload_id"] for r in rows["screen"]})==len(d["workloads"])
    summary=dict(study_id=d["study_id"],design_sha256=sha(args.design.read_bytes()),audit_errors=len(errors),screen_complete=complete_screen,
        accepted_screen_processes=len(rows["screen"]),independently_confirmed_workloads=len(effects),headline_tests=tests,
        equal_workload_geometric_speedup=math.exp(statistics.mean(math.log(e["speedup"]) for e in effects)) if effects else None,
        H1="complete_independent_confirmation" if len(effects)==36 and not errors else "incomplete",
        H2="structure_and_exact_ID_contrasts_reported; crossover_requires_two_clear_opposing_tests",
        H3="fixed_cost_matrix_feasibility_established" if gates["cost"] and gates["cost"].get("passed") and not errors else "unresolved_or_limited",
        H4="auxiliary_see_learning_artifact" if args.learning else "not_requested_auxiliary",
        legacy_data_used=False,profiler="not_performed_unless_separate_guarded_evidence_present")
    write_json(report/"summary.json",summary)
    text=["# VkFFT fixed plan-selection study", "", f"Study: `{d['study_id']}`. CUDA FP32 smooth 1D, immutable input, natural-order out-of-place forward, one fixed GPU.",
          "",f"Accepted screen processes: {len(rows['screen'])}/936. Independently confirmed workloads: {len(effects)}/36. Evidence audit errors: {len(errors)}.",
          "",f"H1: {summary['H1']}. H3: {summary['H3']}.",
          "", "The screen winner is selected once before three fresh confirmation rounds. Process medians are the replication unit. Default-selected effects equal one by definition. Screen minima are optimistic tested-set diagnostics.",
          "", "| Workload | Frozen winner | Independent speedup | Round ratios |", "|---|---|---:|---|"]
    text += [f"| {e['workload_id']} | {e['winner']} | {e['speedup']:.4f} | {', '.join(format(x,'.4f') for x in e['ratios'])} |" for e in effects]
    text += ["", "H2 structure classes are recorded in structure_coverage.json. Source/compiler equality and core projection equality are separate. No single-resource causal attribution is established by structural association.",
             "", "Headline benefit tests use the frozen directions, ten new paired process rounds, a 2% one-sided log threshold and Holm correction across the entire frozen family. Zero variance is noninferential. A crossover requires both opposite directions to pass.",
             "", "Cost pipelines share one context within each fresh process. Describe-all plus an offline known winner is a construction lower bound. Exclusive stage totals partition nested spans; inclusive times overlap. Raw pipeline totals include bookkeeping and shared buffer setup/cleanup. Compiler/driver caches are not claimed cold.",
             "", "Negative evidence includes near-default domains, regressions, request folding, failed candidates, unresolved H3 pairs, missing crossovers and nonsignificant headline tests; see the complete attempt ledger. No legacy pilot measurement enters this report.",
             "", "Unfinished gates remain open questions; missing experiments are not evidence of equivalence. Guard sampling has a finite polling blind spot. This deliberately chosen matrix on one GPU supports no all-FFT or cross-device generalization."]
    (report/"study_report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if effects:
            fig,ax=plt.subplots(figsize=(12,5))
            ax.bar(range(len(effects)),[e["speedup"] for e in effects],color=["#197c75" if e["speedup"]>=1 else "#bd554c" for e in effects])
            ax.axhline(1,color="black",linewidth=1)
            ax.set_xticks(range(len(effects)),[e["workload_id"] for e in effects],rotation=90,fontsize=7)
            ax.set_ylabel("Default / frozen-winner latency (3 fresh rounds)")
            fig.tight_layout();fig.savefig(report/"independent_effects.png",dpi=180);plt.close(fig)
    except ImportError:
        write_json(report/"plot_unavailable.json",dict(reason="matplotlib_not_installed"))
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
