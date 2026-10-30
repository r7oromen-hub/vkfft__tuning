#!/usr/bin/env python3
"""Select predeclared independent follow-up cases; generate commands, never run GPU work."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import posixpath
import shlex

from run_suite import key, negative_control, result_ok, workload_id


def read_strict_records(path):
    """Completed verification inputs may not silently skip malformed records."""
    records=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            record=json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed completed JSONL: {path}:{line_no}") from exc
        if not isinstance(record,dict):
            raise ValueError(f"non-object completed JSONL: {path}:{line_no}")
        records.append(record)
    return records


def gmean(values):
    return math.exp(sum(math.log(x) for x in values)/len(values))


def family(n):
    while n%2==0:
        n//=2
    return n


def core_structure(result):
    s=result.get("structural_features",{})
    return dict(numAxisUploads=s.get("numAxisUploads"),kernels=[
        {k:kernel.get(k) for k in ("stageRadix","localSize","axisBlock")} for kernel in s.get("kernels",[])])


def valid(row):
    return row is not None and row.get("status")=="ok" and result_ok(row.get("result",{}))


def load_complete_main(folder):
    names=("manifest.json","measurements.jsonl","selected.json","graph_repeats.json")
    for name in names:
        if not (folder/name).is_file():
            raise ValueError(f"complete main input missing {name}")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("confirm_rounds")!=3 or manifest.get("timing_mode")!="graph" or manifest.get("placement")!="outofplace":
        raise ValueError("main must use the declared graph/outofplace protocol and three confirmation rounds")
    selected=json.loads((folder/"selected.json").read_text(encoding="utf-8"))
    frozen=json.loads((folder/"graph_repeats.json").read_text(encoding="utf-8"))
    candidates=[c[0] for c in manifest["candidates"]]
    ws={workload_id(w):w for w in manifest["workloads"]}
    if set(selected)!=set(ws) or set(frozen)!=set(ws):
        raise ValueError("selected/frozen-R maps must cover exactly the planned workloads")
    rows=list({key(r):r for r in read_strict_records(folder/"measurements.jsonl")}.values())
    lookup={key(r):r for r in rows}
    cases={}
    for wid,w in ws.items():
        if not isinstance(frozen[wid],int) or frozen[wid]<1:
            raise ValueError(f"invalid frozen R for {wid}")
        screen=[]
        for cid in candidates:
            row=lookup.get(("screen",wid,cid,0))
            if row is None:
                raise ValueError(f"incomplete screen: {wid}/{cid}")
            if valid(row):
                screen.append(row)
        timing={r["candidate_id"]:r["result"]["exec_ms_median"] for r in screen}
        if "default" not in timing or selected[wid] not in timing or timing[selected[wid]]!=min(timing.values()):
            raise ValueError(f"invalid main-screen default/winner for {wid}")
        control=negative_control(screen)
        expected={"default",selected[wid]}|({control} if control else set())
        ratios=[]
        for round_no in range(3):
            for cid in expected:
                if not valid(lookup.get(("confirm",wid,cid,round_no))):
                    raise ValueError(f"incomplete/invalid confirmation: {wid}/{cid}/round{round_no}")
            ratios.append(lookup[("confirm",wid,"default",round_no)]["result"]["exec_ms_median"]/
                          lookup[("confirm",wid,selected[wid],round_no)]["result"]["exec_ms_median"])
        cases[wid]=dict(workload_id=wid,n=w["n"],batch=w["batch"],family=family(w["n"]),
                        winner=selected[wid],score=gmean(ratios),graph_repeats=frozen[wid],control=control,
                        timing=timing,rows={r["candidate_id"]:r for r in screen},confirmation_ratios=ratios)
    gpu_uuids=set()
    for row in rows:
        if row["workload_id"] not in ws or row["phase"] not in ("screen","confirm") or row["candidate_id"] not in candidates:
            raise ValueError("main contains an unplanned workload, phase or candidate")
        if valid(row):
            result=row["result"]
            gpu_uuid=result.get("gpu",{}).get("uuid")
            if not isinstance(gpu_uuid,str) or not gpu_uuid.startswith("GPU-"):
                raise ValueError("main successful records require actual GPU UUID provenance")
            gpu_uuids.add(gpu_uuid)
            expected_r=frozen[row["workload_id"]]
            if (result.get("graph_repeats")!=expected_r or row["workload"].get("graph_repeats",expected_r)!=expected_r
                    or result.get("placement")!="outofplace" or result.get("timing_mode")!="graph"):
                raise ValueError(f"main frozen R/protocol mismatch: {row['workload_id']}")
    if len(gpu_uuids)!=1:
        raise ValueError("main successful records must come from one GPU UUID")
    provenance={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    return manifest,cases,provenance


def select_cases(cases):
    picked={}
    evidence={}
    def order(c):
        return c["n"],c["batch"],c["winner"]
    def add(case,reason,targets,detail=None):
        wid=case["workload_id"]
        if wid not in picked:
            if len(picked)>=6:
                return False
            picked[wid]=dict(source=case,reasons=[],targets=set())
        picked[wid]["reasons"].append(dict(reason=reason,evidence=detail or {}))
        picked[wid]["targets"].update(cid for cid in targets if cid!="default")
        return True
    nondefault=[c for c in cases.values() if c["winner"]!="default"]
    strongest=min(nondefault,key=lambda c:(-c["score"],*order(c))) if nondefault else None
    if strongest:
        add(strongest,"highest_confirmed_nondefault",[strongest["winner"]],{"confirmed_speedup":strongest["score"]})
        structural=[c for c in nondefault if c["n"]!=strongest["n"] and
                    core_structure(c["rows"][c["winner"]]["result"])!=core_structure(c["rows"]["default"]["result"])]
        if structural:
            second=min(structural,key=lambda c:(c["family"]==strongest["family"],-c["score"],*order(c)))
            add(second,"different_N_structural_change",[second["winner"]],
                dict(confirmed_speedup=second["score"],different_family=second["family"]!=strongest["family"],
                     default_core=core_structure(second["rows"]["default"]["result"]),
                     target_core=core_structure(second["rows"][second["winner"]]["result"])))
    crossovers=[]
    for a,b in itertools.combinations(sorted(cases.values(),key=order),2):
        for left,right in itertools.combinations(sorted(set(a["timing"])&set(b["timing"])),2):
            ratio_a=a["timing"][right]/a["timing"][left]
            ratio_b=b["timing"][left]/b["timing"][right]
            if ratio_a>=1.05 and ratio_b>=1.05:
                directions=[(a,left,right,ratio_a),(b,right,left,ratio_b)]
            elif 1/ratio_a>=1.05 and 1/ratio_b>=1.05:
                directions=[(a,right,left,1/ratio_a),(b,left,right,1/ratio_b)]
            else:
                continue
            same_n=a["n"]==b["n"] and a["batch"]!=b["batch"]
            crossovers.append(((not same_n,-min(d[3] for d in directions),a["n"],a["batch"],b["n"],b["batch"],left,right),directions))
    if crossovers:
        _,directions=min(crossovers,key=lambda item:item[0])
        if len(set(picked)|{d[0]["workload_id"] for d in directions})<=6:
            evidence["crossover"]=dict(weaker_screen_directional_ratio=min(d[3] for d in directions),
                same_N_different_B=directions[0][0]["n"]==directions[1][0]["n"],directions=[])
            for case,winner,loser,ratio in directions:
                detail=dict(target=winner,reference=loser,screen_directional_ratio=ratio)
                evidence["crossover"]["directions"].append(dict(workload_id=case["workload_id"],**detail))
                add(case,"screen_rank_crossover",[winner,loser],detail)
    if nondefault:
        worst=min(nondefault,key=lambda c:(c["score"],*order(c)))
        add(worst,"worst_confirmed_regression" if worst["score"]<1 else "weakest_observed_gain",
            [worst["winner"]],{"confirmed_speedup":worst["score"]})
    default_fast=[c for c in cases.values() if c["timing"]["default"]==min(c["timing"].values())]
    if default_fast:
        ordered=sorted(default_fast,key=order)
        representative=ordered[(len(ordered)-1)//2]
        competitors=[cid for cid in representative["timing"] if cid!="default"]
        if competitors:
            target=min(competitors,key=lambda cid:(representative["timing"][cid],cid))
            add(representative,"default_screen_fastest_representative",[target],
                {"target":target,"selection":"lower median in numeric N,B order","default_fastest_pool":len(ordered)})
    elif cases:
        representative=min(cases.values(),key=lambda c:(abs(c["score"]-1),*order(c)))
        add(representative,"confirmed_speedup_closest_to_one",[representative["winner"]],
            {"confirmed_speedup":representative["score"]})
    return picked,evidence


def build_plan(main):
    manifest,cases,provenance=load_complete_main(main)
    picked,evidence=select_cases(cases)
    design_path=Path(__file__).with_name("headline_design.json")
    design=json.loads(design_path.read_text(encoding="utf-8"))
    output_cases=[]
    comparisons={}
    for index,(wid,chosen) in enumerate(picked.items(),1):
        case=chosen["source"]
        if case["control"] is None:
            raise ValueError(f"selected case {wid} has no exact resolved+descriptor default control")
        case_id=f"case{index:02d}_{wid}"
        targets=sorted(chosen["targets"])
        candidate_ids=list(dict.fromkeys(["default"]+targets+[case["control"]]))
        output_cases.append(dict(case_id=case_id,workload_id=wid,n=case["n"],batch=case["batch"],
            family=f"odd_part_{case['family']}",graph_repeats=case["graph_repeats"],
            candidates=candidate_ids,target_candidates=targets,control=case["control"],
            seed=design["append_protocol"]["seed_base"]+index,reasons=chosen["reasons"],
            main_winner=case["winner"],main_confirmed_speedup=case["score"],
            main_confirmation_ratios=case["confirmation_ratios"]))
        requests=[(target,"default","target_vs_default") for target in targets]
        requests += [(r["evidence"]["target"],r["evidence"]["reference"],"crossover_direction")
                     for r in chosen["reasons"] if r["reason"]=="screen_rank_crossover"]
        for target,reference,reason in requests:
            if target==reference:
                continue
            cid=f"{case_id}__{target}__vs__{reference}"
            control_diagnostic=({reference,target}=={"default",case["control"]})
            if cid not in comparisons:
                comparisons[cid]=dict(comparison_id=cid,case_id=case_id,workload_id=wid,target=target,reference=reference,
                                      role="control_diagnostic" if control_diagnostic else "benefit",reasons=[])
            comparisons[cid]["reasons"].append(reason)
    return dict(schema_version=1,design=design,design_sha256=hashlib.sha256(design_path.read_bytes()).hexdigest(),
                main_provenance_sha256=provenance,main_executable=manifest["executable"],
                main_executable_sha256=manifest["executable_sha256"],source_commit=manifest.get("source_commit","unknown"),
                main_gpu_uuid=next(iter(cases.values()))["rows"]["default"]["result"]["gpu"]["uuid"],
                main_candidate_configs=dict(manifest["candidates"]),
                cases=output_cases,comparisons=list(comparisons.values()),selection_evidence=evidence,
                distinct_workloads=len(output_cases),gpu_work_started=False)


def write_plan(plan,output,runner=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output plan directory must be new or empty; never overwrite measured follow-ups")
    output.mkdir(parents=True,exist_ok=True)
    (output/"selection.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    (output/"headline_design.json").write_text(json.dumps(plan["design"],indent=2)+"\n",encoding="utf-8")
    runner=runner or posixpath.join(posixpath.dirname(posixpath.dirname(plan["main_executable"])),"test/plan_tuning/controlled_recheck.py")
    lines=["#!/usr/bin/env bash","set -euo pipefail",'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
           "HARNESS="+shlex.quote(plan["main_executable"]),"RECHECK="+shlex.quote(runner),
           'if [[ $# -ge 1 ]]; then HARNESS=$1; fi','if [[ $# -ge 2 ]]; then RECHECK=$2; fi',
           "export CUDA_VISIBLE_DEVICES="+shlex.quote(plan["main_gpu_uuid"]),""]
    for case in plan["cases"]:
        folder=output/"cases"/case["case_id"]
        folder.mkdir(parents=True)
        (folder/"workloads.json").write_text(json.dumps([dict(n=case["n"],batch=case["batch"],family=case["family"])],indent=2)+"\n",encoding="utf-8")
        lines.append(f"# {case['case_id']}: R is frozen from main, no calibration")
        lines.append('python3 "$RECHECK" --exe "$HARNESS" '
            f'--output "$SCRIPT_DIR/measurements/{case["case_id"]}" --workloads "$SCRIPT_DIR/cases/{case["case_id"]}/workloads.json" '
            f'--candidates {shlex.quote(",".join(case["candidates"]))} --rounds 10 --warmup 5 --warmup-ms 1000 --repeats 101 '
            f'--timing-mode graph --placement outofplace --graph-repeats {case["graph_repeats"]} '
            f'--seed {case["seed"]} --source-commit {shlex.quote(plan["source_commit"])}')
        lines.append("")
    with (output/"run_checks.sh").open("w",encoding="utf-8",newline="\n") as f:
        f.write("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("main",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--runner",help="Runtime POSIX path to controlled_recheck.py; defaults beside main project")
    args=p.parse_args()
    plan=build_plan(args.main)
    write_plan(plan,args.output,args.runner)
    print(json.dumps(dict(distinct_workloads=plan["distinct_workloads"],cases=[c["case_id"] for c in plan["cases"]],
                          shell_script=str(args.output/"run_checks.sh"),gpu_work_started=False),indent=2))


if __name__=="__main__":
    main()
