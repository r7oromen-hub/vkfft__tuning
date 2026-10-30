"""Deterministic frozen selections and independent process-level summaries."""
import math
import statistics
from trace_audit import plan_fingerprints


def tie_key(cid):
    return (cid != "default", cid)


def freeze_screen(design, blocks):
    selections={}
    eligible={}
    for w in design["workloads"]:
        rows=blocks[w["id"]]
        legal={r["candidate_id"]:r for r in rows if r["status"]=="ok" and r.get("selection_eligible",True) and r["candidate_id"]!="default_replica"}
        ranked=sorted(legal,key=lambda c:(legal[c]["result"]["exec_ms_median"],tie_key(c)))
        winner=ranked[0]
        fp=plan_fingerprints(legal[winner]["result"])
        alt=next((c for c in ranked if all(plan_fingerprints(legal[c]["result"])[f]!=fp[f] for f in ("core","source_compiler"))),None)
        eligible[w["id"]]=legal
        selections[w["id"]]=dict(winner=winner,alternative=alt,screen_ms={c:r["result"]["exec_ms_median"] for c,r in legal.items()},
                                  record_hashes=[r["record_sha256"] for r in rows])
    common=set.intersection(*(set(x) for x in eligible.values()))
    scores={c:statistics.mean(math.log(rows[c]["result"]["exec_ms_median"]/rows["default"]["result"]["exec_ms_median"]) for rows in eligible.values()) for c in common}
    global_id=min(scores,key=lambda c:(scores[c],tie_key(c))) if scores else None
    for s in selections.values():
        s["candidates"]=sorted({c for c in ("default","default_replica",s["winner"],s["alternative"],global_id) if c})
    return dict(global_candidate=global_id,global_scores=scores,workloads=selections,
                rule="Frozen screening choices; confirmation never reselects the primary winner.")


def paired_effect(rows,reference,target,rounds):
    by={(r["round"],r["candidate_id"]):r for r in rows if r["status"]=="ok"}
    ratios=[]
    for k in range(rounds):
        if (k,reference) not in by or (k,target) not in by:
            return None
        ratios.append(by[k,reference]["result"]["exec_ms_median"]/by[k,target]["result"]["exec_ms_median"])
    return dict(ratios=ratios,speedup=math.exp(statistics.mean(map(math.log,ratios))),
                reference=reference,target=target,unit="independent paired process medians")


def confirmation_summary(design,selection,blocks):
    result={}
    for w in design["workloads"]:
        wid=w["id"]
        s=selection["workloads"][wid]
        rows=[r for (bw,_),rs in blocks.items() if bw==wid for r in rs]
        result[wid]=dict(winner=s["winner"],primary=paired_effect(rows,"default",s["winner"],design["selection"]["confirmation_rounds"]),
                         comparisons={c:paired_effect(rows,"default",c,design["selection"]["confirmation_rounds"]) for c in s["candidates"] if c!="default_replica"})
    return result


def freeze_headline(design,selection,summary,screen):
    ws={w["id"]:w for w in design["workloads"]}
    estimable=[wid for wid in ws if summary[wid]["primary"] is not None]
    rank=sorted(estimable,key=lambda wid:(-summary[wid]["primary"]["speedup"],ws[wid]["n"],ws[wid]["batch"]))
    chosen=[]
    tests=[]
    if not rank:
        return dict(workloads=[],directional_tests=[],crossovers=[],rounds=design["headline"]["rounds"],missing_crossover=True,
                    selection_uses_only_screen_and_confirmation=True,unestimable_workloads=sorted(ws))
    def add(wid,reason,reference="default",target=None):
        target=target or selection["workloads"][wid]["winner"]
        if wid not in [x["workload_id"] for x in chosen]:
            winner=selection["workloads"][wid]["winner"]
            chosen.append(dict(workload_id=wid,reason=reason,candidates=sorted({"default","default_replica",winner})))
            if winner!="default":
                tests.append(dict(workload_id=wid,reference="default",target=winner,reason="frozen_winner_effect"))
        entry=next(x for x in chosen if x["workload_id"]==wid)
        entry["candidates"]=sorted(set(entry["candidates"]+[reference,target]))
        if target!=reference and not any(t["workload_id"]==wid and t["reference"]==reference and t["target"]==target for t in tests):
            tests.append(dict(workload_id=wid,reference=reference,target=target,reason=reason))
    add(rank[0],"largest_confirmed_frozen_winner_effect")
    n0=ws[rank[0]]["n"]
    def changed(wid):
        rows={r["candidate_id"]:r for r in screen[wid]}
        winner=selection["workloads"][wid]["winner"]
        return plan_fingerprints(rows[winner]["result"])["core"]!=plan_fingerprints(rows["default"]["result"])["core"]
    def odd(n):
        while n%2==0: n//=2
        return n
    second=[wid for wid in rank if ws[wid]["n"]!=n0 and changed(wid)]
    second.sort(key=lambda wid:(odd(ws[wid]["n"])==odd(n0),-summary[wid]["primary"]["speedup"],ws[wid]["n"],ws[wid]["batch"]))
    if second: add(second[0],"different_N_core_change")
    # Exact predeclared candidate identities; both screen and confirmation must qualify.
    cross=[]
    for priority,c in enumerate(design["structure_analysis"]["predeclared_batch_contrasts"]):
        ref,target=c["reference"],c["candidate"]
        for n in sorted({w["n"] for w in ws.values()}):
            if target=="a128_r4_l1_np0" and n&(n-1)==0: continue
            if target.endswith("s2_8192") and n<8192: continue
            if target.endswith("s3_65536") and n<65536: continue
            endpoints=sorted([w for w in ws.values() if w["n"]==n],key=lambda w:w["batch"])
            a,b=endpoints[0]["id"],endpoints[-1]["id"]
            if a not in estimable or b not in estimable: continue
            effects=[]
            for wid in (a,b):
                cmp=summary[wid]["comparisons"]
                if not cmp.get(ref) or not cmp.get(target): break
                conf=cmp[target]["speedup"]/cmp[ref]["speedup"]
                sr=selection["workloads"][wid]["screen_ms"]
                if ref not in sr or target not in sr: break
                effects.append((conf,sr[ref]/sr[target]))
            if len(effects)==2:
                x,y=effects
                if x[0]>=1.05 and x[1]>=1.05 and y[0]<=1/1.05 and y[1]<=1/1.05:
                    cross.append((priority!=0,-min(x[0],1/y[0]),n,ws[a]["batch"],ws[b]["batch"],ref,target,priority,a,b))
                elif y[0]>=1.05 and y[1]>=1.05 and x[0]<=1/1.05 and x[1]<=1/1.05:
                    cross.append((priority!=0,-min(y[0],1/x[0]),n,ws[b]["batch"],ws[a]["batch"],ref,target,priority,b,a))
    crossovers=[]
    if cross:
        _,_,n,_,_,ref,target,index,a,b=min(cross)
        add(a,"exact_ID_crossover_forward",ref,target)
        add(b,"exact_ID_crossover_reverse",target,ref)
        crossovers.append(dict(n=n,contrast_index=index,directions=[dict(workload_id=a,reference=ref,target=target),
                                                                   dict(workload_id=b,reference=target,target=ref)]))
    worst=min(estimable,key=lambda wid:(summary[wid]["primary"]["speedup"],ws[wid]["n"],ws[wid]["batch"]))
    add(worst,"weakest_frozen_winner_effect")
    defaults=[wid for wid in estimable if selection["workloads"][wid]["winner"]=="default"]
    defaults.sort(key=lambda wid:(ws[wid]["n"],ws[wid]["batch"]))
    if defaults:
        lengths=sorted({ws[wid]["n"] for wid in defaults})
        median_n=lengths[(len(lengths)-1)//2]
        final=min((wid for wid in defaults if ws[wid]["n"]==median_n),key=lambda wid:ws[wid]["batch"])
    else: final=min(estimable,key=lambda wid:(abs(math.log(summary[wid]["primary"]["speedup"])),ws[wid]["n"],ws[wid]["batch"]))
    add(final,"default_best_or_nearest_null")
    return dict(workloads=chosen,directional_tests=tests,crossovers=crossovers,rounds=design["headline"]["rounds"],
                missing_crossover=not bool(cross),selection_uses_only_screen_and_confirmation=True,
                unestimable_workloads=sorted(set(ws)-set(estimable)))
