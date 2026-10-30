#!/usr/bin/env python3
"""Analyze exactly ten predeclared follow-up rounds, separately from main results."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from analyze import dump_csv
from run_suite import key, result_ok
from make_headline_checks import read_strict_records


def paired_statistics(ratios,inferential=True):
    values=np.asarray(ratios,dtype=float)
    if values.ndim!=1 or len(values)==0 or not np.all(np.isfinite(values)) or np.any(values<=0):
        raise ValueError("paired ratios must be a nonempty finite positive vector")
    logs=np.log(values)
    mean=float(np.mean(logs))
    out=dict(rounds=len(ratios),speedup_gmean=math.exp(mean),observed_range=[float(min(ratios)),float(max(ratios))],
             log_mean=mean,log_sample_sd=float(np.std(logs,ddof=1)) if len(logs)>1 else None,
             paired_log_t_95ci=None,one_sided_p_vs_1p02=None)
    # Algebraically equal ratios can differ by a few floating-point ULPs.
    # This threshold removes arithmetic noise, not observed timing variation.
    zero_spread_tolerance=16*np.finfo(float).eps*max(1.,float(np.max(np.abs(logs))))
    out["zero_log_spread_tolerance"]=float(zero_spread_tolerance)
    if len(logs)<2 or float(np.ptp(logs))<=zero_spread_tolerance:
        out["inference_status"]="zero_variance_noninferential" if len(logs)>1 else "insufficient_rounds"
        return out
    se=float(np.std(logs,ddof=1)/math.sqrt(len(logs)))
    critical=float(stats.t.ppf(.975,len(logs)-1))
    out["paired_log_t_95ci"]=[math.exp(mean-critical*se),math.exp(mean+critical*se)]
    out["inference_status"]="approximate_paired_log_t" if inferential else "diagnostic_interval_only"
    if inferential:
        out["one_sided_p_vs_1p02"]=float(stats.t.sf((mean-math.log(1.02))/se,len(logs)-1))
    return out


def holm_adjust(results,alpha=.05):
    """Keep noninferential comparisons in family with conservative p=1."""
    benefit=[r for r in results if r["role"]=="benefit"]
    ordered=sorted(benefit,key=lambda r:(r["one_sided_p_vs_1p02"] if r["one_sided_p_vs_1p02"] is not None else 1.,r["comparison_id"]))
    adjusted=0.
    for index,row in enumerate(ordered):
        p=row["one_sided_p_vs_1p02"]
        adjusted=max(adjusted,min(1.,(len(ordered)-index)*(p if p is not None else 1.)))
        row["holm_adjusted_p"]=adjusted if p is not None else None
        row["holm_reject_2pct_null"]=p is not None and adjusted<=alpha
        row["clear_gain"]=row["speedup_gmean"]>=1.05 and row["holm_reject_2pct_null"]
    for row in results:
        if row["role"]!="benefit":
            row.update(holm_adjusted_p=None,holm_reject_2pct_null=False,clear_gain=False)
    return len(benefit)


def load_followups(plan_root,plan):
    all_rows={}
    provenance={}
    for case in plan["cases"]:
        folder=plan_root/"measurements"/case["case_id"]
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        expected=dict(rounds=10,warmup=5,warmup_ms=1000,repeats=101,timing_mode="graph",placement="outofplace",
                      graph_repeats_requested=case["graph_repeats"],seed=case["seed"],executable_sha256=plan["main_executable_sha256"],
                      cuda_visible_devices=plan["main_gpu_uuid"],source_commit=plan["source_commit"])
        for field,value in expected.items():
            if manifest.get(field)!=value:
                raise ValueError(f"follow-up protocol mismatch: {case['case_id']}/{field}")
        expected_configs={cid:plan["main_candidate_configs"][cid] for cid in case["candidates"]}
        if len(manifest["candidates"])!=len(expected_configs) or dict(manifest["candidates"])!=expected_configs:
            raise ValueError("follow-up candidate settings differ from predeclared case")
        rows=read_strict_records(folder/"measurements.jsonl")
        expected_keys={("control",case["workload_id"],cid,round_no) for cid in case["candidates"] for round_no in range(10)}
        if len(rows)!=len(expected_keys) or {key(r) for r in rows}!=expected_keys:
            raise ValueError(f"follow-up must contain exactly ten complete rounds: {case['case_id']}")
        lookup={}
        for row in rows:
            result=row.get("result",{})
            if (row.get("status")!="ok" or not result_ok(result) or len(result["exec_ms_samples"])!=101
                    or result.get("graph_repeats")!=case["graph_repeats"] or result.get("timing_mode")!="graph"
                    or result.get("placement")!="outofplace" or row["workload"].get("n")!=case["n"]
                    or row["workload"].get("batch")!=case["batch"] or result.get("n")!=case["n"]
                    or result.get("batch")!=case["batch"] or result.get("gpu",{}).get("uuid")!=plan["main_gpu_uuid"]
                    or row["workload"].get("graph_repeats",case["graph_repeats"])!=case["graph_repeats"]):
                raise ValueError(f"invalid or mixed follow-up record: {case['case_id']}")
            if (result.get("correctness",{}).get("criterion")!="criterion_v2"
                    or result.get("single_forward_correctness",{}).get("passed") is not True
                    or result.get("single_forward_correctness",{}).get("criterion")!="criterion_v2"
                    or result.get("single_forward_input_unchanged") is not True or result.get("input_unchanged") is not True):
                raise ValueError("follow-up requires both v2 numerical checks and immutable input")
            seed_text=f"{case['seed']}:control:{case['workload_id']}:{row['round']}"
            expected_seed=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4],"little")
            if row.get("input_seed")!=expected_seed or result.get("seed")!=expected_seed:
                raise ValueError("follow-up input seed differs from the predeclared paired round")
            measured=float(np.median(result["exec_ms_samples"]))
            if not math.isclose(measured,result["exec_ms_median"],rel_tol=1e-12,abs_tol=1e-15):
                raise ValueError("reported process median differs from raw samples")
            lookup[(row["candidate_id"],row["round"])]=measured
        for round_no in range(10):
            baseline=next(r for r in rows if r["candidate_id"]=="default" and r["round"]==round_no)["result"]
            control=next(r for r in rows if r["candidate_id"]==case["control"] and r["round"]==round_no)["result"]
            if (not baseline.get("resolved_config") or not baseline.get("structural_features")
                    or baseline.get("resolved_config")!=control.get("resolved_config")
                    or baseline.get("structural_features")!=control.get("structural_features")):
                raise ValueError("predeclared negative control no longer matches recorded default descriptors")
        all_rows[case["case_id"]]=dict(lookup=lookup,records=rows)
        for name in ("manifest.json","measurements.jsonl"):
            provenance[f"{case['case_id']}/{name}"]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    return all_rows,provenance


def analyze(plan_root):
    plan=json.loads((plan_root/"selection.json").read_text(encoding="utf-8"))
    copied_design=json.loads((plan_root/"headline_design.json").read_text(encoding="utf-8"))
    if copied_design!=plan["design"]:
        raise ValueError("selection and copied predeclared design differ")
    data,provenance=load_followups(plan_root,plan)
    results,round_rows,controls,control_rounds=[],[],[],[]
    for comparison in plan["comparisons"]:
        lookup=data[comparison["case_id"]]["lookup"]
        ratios=[]
        for round_no in range(10):
            reference=lookup[(comparison["reference"],round_no)]
            target=lookup[(comparison["target"],round_no)]
            ratio=reference/target
            ratios.append(ratio)
            round_rows.append(dict(comparison_id=comparison["comparison_id"],case_id=comparison["case_id"],
                                   role=comparison["role"],round=round_no,target=comparison["target"],reference=comparison["reference"],
                                   target_median_ms=target,reference_median_ms=reference,speedup=ratio,log_speedup=math.log(ratio)))
        results.append(dict(comparison,**paired_statistics(ratios,comparison["role"]=="benefit")))
    family_size=holm_adjust(results)
    for case in plan["cases"]:
        lookup=data[case["case_id"]]["lookup"]
        ratios=[lookup[("default",r)]/lookup[(case["control"],r)] for r in range(10)]
        controls.append(dict(case_id=case["case_id"],workload_id=case["workload_id"],control=case["control"],
                             **paired_statistics(ratios,False)))
        control_rounds += [dict(case_id=case["case_id"],round=r,control=case["control"],ratio=ratio) for r,ratio in enumerate(ratios)]
    cross=[r for r in results if "crossover_direction" in r["reasons"]]
    report=dict(schema_version=1,design=plan["design"],cases=len(plan["cases"]),rounds_per_case=10,
                comparisons=results,negative_controls=controls,holm_family_size=family_size,
                holm_noninferential_policy="Planned zero-variance comparisons retain family burden using p=1 internally; no inferential p-value or success is reported.",
                crossover=dict(directional_comparisons=[r["comparison_id"] for r in cross],
                               both_directions_clear_gain=len(cross)==2 and all(r["clear_gain"] for r in cross)),
                provenance_sha256=dict(provenance,selection=hashlib.sha256((plan_root/"selection.json").read_bytes()).hexdigest()),
                limitations=["Paired log t intervals/tests use finite-round approximate normality and independence assumptions; ordinary 95% intervals are not simultaneous intervals.",
                             "Zero observed variance, including floating-point arithmetic-only spread, is noninferential, not infinite significance.",
                             "Negative controls diagnose noise/drift and provide no tuning-gain evidence.",
                             "The clear_gain flag applies the statistical rule only; measurement validity also requires reviewing the separately reported negative-control behavior.",
                             "Nonsignificance in a default-fastest case establishes neither equivalence nor global optimality.",
                             "These deliberately selected extremes/representatives are not an unbiased aggregate performance sample or independent model validation.",
                             "Main model results remain offline screen cross-validation. No online compile savings are measured."])
    return report,round_rows,control_rounds


def write_output(report,round_rows,control_rounds,output):
    output.mkdir(parents=True,exist_ok=True)
    dump_csv(output/"all_round_ratios.csv",round_rows)
    dump_csv(output/"control_round_ratios.csv",control_rounds)
    control_statistics=[]
    for row in report["negative_controls"]:
        flat={k:v for k,v in row.items() if k not in ("observed_range","paired_log_t_95ci")}
        flat["ci_low"],flat["ci_high"]=(row["paired_log_t_95ci"] or [None,None])
        flat["range_low"],flat["range_high"]=row["observed_range"]
        control_statistics.append(flat)
    dump_csv(output/"control_statistics.csv",control_statistics)
    flattened=[]
    for row in report["comparisons"]:
        flat={k:v for k,v in row.items() if k not in ("reasons","observed_range","paired_log_t_95ci")}
        flat["reasons"]=";".join(row["reasons"])
        flat["ci_low"],flat["ci_high"]=(row["paired_log_t_95ci"] or [None,None])
        flat["range_low"],flat["range_high"]=row["observed_range"]
        flattened.append(flat)
    dump_csv(output/"comparison_statistics.csv",flattened)
    (output/"headline_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# 独立追加检查", "", f"共 {report['cases']} 个定向选择的工作负载，每个固定 10 轮。预定收益比较 family 大小为 {report['holm_family_size']}。",
           "", "每轮使用新进程样本中位数形成配对 log 比率；区间为普通配对 log t 95% 区间，p 值对不超过 2% 收益的单侧原假设作 Holm 校正。零方差仅作描述，不能推断显著性。",
           "", "|比较|几何均值|普通 95% 区间|Holm p|满足预定明显收益规则|", "|---|---:|---|---:|---|"]
    for row in report["comparisons"]:
        ci=row["paired_log_t_95ci"]
        interval=f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "不可推断"
        p=f"{row['holm_adjusted_p']:.5g}" if row["holm_adjusted_p"] is not None else "不适用/不可推断"
        lines.append(f"|{row['comparison_id']}|{row['speedup_gmean']:.4f}|{interval}|{p}|{row['clear_gain']}|")
    lines += ["",f"预定排名交叉的两个方向均通过规则：{report['crossover']['both_directions_clear_gain']}。",
              "", "明显收益规则：几何均值至少 1.05，并且 Holm 校正后拒绝收益不超过 2% 的原假设。负对照不作收益证据。默认案例未显著不代表等价或全空间最优。这些定向案例不构成模型独立验证，也不是总体平均收益估计。"]
    lines += ["", "|负对照 workload|默认/对照几何均值|十轮范围|", "|---|---:|---|"]
    for row in report["negative_controls"]:
        low,high=row["observed_range"]
        lines.append(f"|{row['workload_id']}|{row['speedup_gmean']:.4f}|[{low:.4f}, {high:.4f}]|")
    lines += ["", "clear_gain 仅表示满足预定统计规则；解释性能收益前还需检查上表的负对照波动，不能用显著性替代测量有效性检查。"]
    (output/"headline_results_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report,round_rows,control_rounds=analyze(args.plan)
    output=args.output or args.plan/"analysis"
    write_output(report,round_rows,control_rounds,output)
    print(json.dumps(dict(cases=report["cases"],holm_family_size=report["holm_family_size"],crossover=report["crossover"],
                          summary=str(output/"headline_summary.json")),indent=2))


if __name__=="__main__":
    main()
