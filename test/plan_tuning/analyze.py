#!/usr/bin/env python3
"""Analyze observed VkFFT device timings without claiming plan-avoidance savings."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import warnings

import numpy as np

from run_suite import CANDIDATES, key, read_records, negative_control


def dump_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = [float(v) for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    if not values:
        return {"count":0}
    return dict(count=len(values),mean=float(np.mean(values)),
                **dict(zip(("min","p10","median","p90","max"),map(float,np.quantile(values,[0,.1,.5,.9,1])))))


def validation_analysis(records, calibration_records, manifest, output):
    """Numerical, interval, cost and temporal diagnostics; no savings estimate."""
    numerical, process_drift = [], []
    for source, rows in (("measurements",records),("calibration",calibration_records)):
        for row in rows:
            result = row.get("result",{})
            for field, label, unchanged in (("single_forward_correctness","single_forward","single_forward_input_unchanged"),
                                             ("correctness","final_graph_or_forward","input_unchanged")):
                check = result.get(field)
                if not isinstance(check,dict):
                    continue
                numerical.append(dict(source=source,phase=row["phase"],workload_id=row["workload_id"],
                                      candidate_id=row["candidate_id"],round=row["round"],check=label,
                                      process_status=row["status"],passed=check.get("passed"),
                                      criterion=check.get("criterion","legacy_or_unknown"),
                                      checked_elements=check.get("checked_elements"),
                                      nonfinite_count=check.get("nonfinite_count"),
                                      failing_batches=check.get("failing_batches"),
                                      pointwise_passed=check.get("pointwise_passed"),
                                      pointwise_failed_elements=check.get("failed_elements"),
                                      worst_batch_l2_scaled_error=check.get("worst_batch_l2_scaled_error"),
                                      worst_batch_linf_input_scaled_error=check.get("worst_batch_linf_input_scaled_error"),
                                      l2_threshold=check.get("l2_scaled_error_threshold"),
                                      linf_threshold=check.get("linf_input_scaled_error_threshold"),
                                      input_unchanged=result.get(unchanged)))
    for row in records:
        samples = row.get("result",{}).get("exec_ms_samples",[])
        if row["status"] != "ok" or len(samples) < 6:
            continue
        third = max(1,len(samples)//3)
        early,late = statistics.median(samples[:third]),statistics.median(samples[-third:])
        process_drift.append(dict(phase=row["phase"],workload_id=row["workload_id"],candidate_id=row["candidate_id"],
                                  round=row["round"],order=row.get("order"),samples=len(samples),
                                  early_third_median_ms=early,late_third_median_ms=late,late_over_early=late/early))
    dump_csv(output / "numerical_checks.csv",numerical)
    dump_csv(output / "process_drift.csv",process_drift)
    numerical_summary = {}
    for source in ("measurements","calibration"):
        numerical_summary[source] = {}
        for check in ("single_forward","final_graph_or_forward"):
            rows = [r for r in numerical if r["source"]==source and r["check"]==check]
            numerical_summary[source][check] = dict(checks=len(rows),passed=sum(r["passed"] is True for r in rows),
                failed=sum(r["passed"] is False for r in rows),
                missing_passed=sum(r["passed"] is None for r in rows),
                nonfinite_outputs=sum(r["nonfinite_count"] or 0 for r in rows),
                failing_batches=sum(r["failing_batches"] or 0 for r in rows),
                pointwise_diagnostic_failed_checks=sum(r["pointwise_passed"] is False for r in rows),
                pointwise_diagnostic_failed_elements=sum(r["pointwise_failed_elements"] or 0 for r in rows),
                immutable_input_failures=sum(r["input_unchanged"] is False for r in rows),
                immutable_input_checks=sum(r["input_unchanged"] is not None for r in rows),
                criteria=sorted({r["criterion"] for r in rows}),
                worst_batch_l2_scaled_error=distribution([r["worst_batch_l2_scaled_error"] for r in rows]),
                worst_batch_linf_input_scaled_error=distribution([r["worst_batch_linf_input_scaled_error"] for r in rows]),
                l2_thresholds=sorted({r["l2_threshold"] for r in rows if r["l2_threshold"] is not None}),
                linf_thresholds=sorted({r["linf_threshold"] for r in rows if r["linf_threshold"] is not None}))
    valid = [r for r in records if r["status"]=="ok"]
    raw_rows = [r for r in valid if isinstance(r.get("result",{}).get("raw_interval_ms_median"),(int,float))]
    target = manifest.get("graph_target_ms")
    interval_rows = []
    for row in raw_rows:
        result = row["result"]
        raw = result.get("raw_interval_ms_samples",[])
        interval_rows.append(dict(phase=row["phase"],workload_id=row["workload_id"],candidate_id=row["candidate_id"],
                                  round=row["round"],graph_repeats=result.get("graph_repeats",1),
                                  raw_interval_ms_median=result["raw_interval_ms_median"],target_ms=target,
                                  median_below_target=(result["raw_interval_ms_median"]<target) if target else None,
                                  samples=len(raw),samples_below_target=sum(x<target for x in raw) if target else None))
    dump_csv(output / "raw_intervals.csv",interval_rows)
    all_raw = [x for r in raw_rows for x in r["result"].get("raw_interval_ms_samples",[])]
    timing = dict(raw_process_medians_ms=distribution([r["result"]["raw_interval_ms_median"] for r in raw_rows]),
                  raw_samples_ms=distribution(all_raw),calibration_default_target_ms=target,
                  process_medians_below_target=sum(r["median_below_target"] is True for r in interval_rows),
                  sample_fraction_below_target=(sum(x<target for x in all_raw)/len(all_raw)) if target and all_raw else None,
                  graph_repeats=distribution([r["result"].get("graph_repeats",1) for r in raw_rows]),
                  interpretation="Default calibration target is a timing diagnostic, not a guaranteed lower bound for all candidates/samples.")
    costs = {}
    for phase in sorted({r["phase"] for r in records}):
        rows = [r for r in records if r["phase"]==phase]
        successful = [r for r in rows if r["status"]=="ok"]
        starts,ends = [],[]
        for row in rows:
            try:
                starts.append(dt.datetime.fromisoformat(row["started_utc"]))
                ends.append(dt.datetime.fromisoformat(row["ended_utc"]))
            except (KeyError,ValueError):
                pass
        costs[phase] = dict(processes=len(rows),successful=len(successful),
                           subprocess_wall_seconds=sum(r.get("wall_seconds",0) for r in rows),
                           first_start_to_last_end_seconds=(max(ends)-min(starts)).total_seconds() if starts and ends else None,
                           initialize_including_compile_ms=distribution([r["result"].get("initialize_ms") for r in successful]),
                           graph_build_ms=distribution([r["result"].get("graph_build_ms") for r in successful]),
                           initialization_over_single_fft=distribution([r["result"]["initialize_ms"]/r["result"]["exec_ms_median"]
                               for r in successful if isinstance(r["result"].get("initialize_ms"),(int,float))]))
    drift = dict(late_over_early=distribution([r["late_over_early"] for r in process_drift]),
                 late_over_early_gmean=geomean([r["late_over_early"] for r in process_drift]),
                 processes_over_5pct_change=sum(abs(r["late_over_early"]-1)>.05 for r in process_drift),
                 interpretation="Within-process last-third / first-third sample medians after warmup; diagnostic, not causal attribution.")
    return dict(numerical=numerical_summary,timing_intervals=timing,measured_costs=costs,within_process_drift=drift,
                calibration_subprocess_wall_seconds=sum(r.get("wall_seconds",0) for r in calibration_records),
                cost_interpretation="Observed initialization, graph construction and process wall time are separate measurements; no online compile-avoidance savings measured.")


def write_chinese_report(report,output):
    def fmt(value,digits=4):
        return f"{value:.{digits}f}" if isinstance(value,(int,float)) else "未测得"
    validation=report["validation"]
    lines=["# VkFFT 规划动机实验结果", "", "本报告仅汇总当前结果目录；旧版试验和不同计时协议未合并。",
           "",f"成功记录 {report['valid']}/{report['measurements']}；失败 {report['failures']}。完整筛选工作负载 {report['complete_screen_workloads']}/{report['planned_screen_workloads']}；复测记录可用 {report['confirmed_workloads']} 个，其中完成全部预定轮次 {report['complete_confirmed_workloads']} 个。",
           f"筛选中的默认/最佳已测配置时间几何均值为 {fmt(report['screen_oracle_speedup_gmean'])} 倍；该值包含选优偏差。"]
    if report["confirmation_primary_ready"]:
        interval=report["confirmation_gmean_designed_suite_n_cluster_bootstrap_95ci"]
        lines += [f"独立交错复测的几何均值为 {fmt(report['confirmation_speedup_gmean'])} 倍。设计工作负载内按 N 分组重采样的 95% 区间为 [{fmt(interval[0])}, {fmt(interval[1])}]；这不代表任意 FFT 工作负载总体的区间。",
                  f"复测出现回退的工作负载 {report['confirmation_regressions']} 个，其中时间增加超过 5% 的为 {report['confirmation_regressions_over_5pct']} 个。"]
    elif report["confirmed_workloads"]:
        lines += ["预定独立复测尚未结束；不将这些早期记录解释为完整三轮确认或最终总体收益。可用轮次数见 confirmation.csv。"]
    lines += ["", "## 数值与输入不变性检查", "", "|检查|记录数|通过|失败|非有限输出|输入改变|最大归一化 L2 误差|最大归一化峰值误差|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,label in (("single_forward","单次前向"),("final_graph_or_forward","最终图/前向")):
        item=validation["numerical"]["measurements"][name]
        lines.append(f"|{label}|{item['checks']}|{item['passed']}|{item['failed']}|{item['nonfinite_outputs']}|{item['immutable_input_failures']}|{item['worst_batch_l2_scaled_error'].get('max','未记录')}|{item['worst_batch_linf_input_scaled_error'].get('max','未记录')}|")
    lines += ["", "criterion_v2 按每个 batch 检查有限性、L2(error)/(sqrt(N)·L2(input)) ≤ 1e-5、L∞(error)/L2(input) ≤ 1e-4；零输入要求精确零输出。阈值属于参考旧版试验后固定的 FP32 工程标准，不是理论误差界。旧版逐点阈值只作诊断；缺少相应字段时不推断通过。",
              "", "## 计时与成本审计", ""]
    timing=validation["timing_intervals"]
    raw=timing["raw_process_medians_ms"]
    lines += [f"原始整图时间中位数的范围为 [{fmt(raw.get('min'))}, {fmt(raw.get('max'))}] ms；低于默认校准目标 {timing['calibration_default_target_ms']} ms 的进程记录 {timing['process_medians_below_target']} 个。校准目标只验证默认配置，速度更快的候选或后续样本可能低于目标。"]
    for phase,cost in validation["measured_costs"].items():
        lines += [f"{phase}：进程累计墙钟 {fmt(cost['subprocess_wall_seconds'],2)} s；初始化（含编译）中位数 {fmt(cost['initialize_including_compile_ms'].get('median'),3)} ms；图构建中位数 {fmt(cost['graph_build_ms'].get('median'),3)} ms。这些成本分别记录，均不等于模型在线开销或已实现的编译节省。"]
    lines += [f"预热后每个进程末三分之一/首三分之一样本中位数比值的几何均值为 {fmt(validation['within_process_drift']['late_over_early_gmean'])}；变化幅度超过 5% 的进程 {validation['within_process_drift']['processes_over_5pct_change']} 个。此项用于发现漂移，不做因果归因。",
              "", "## 负对照", ""]
    for phase,item in report["negative_controls"].items():
        lines += [f"{phase}：匹配解析配置与已记录规划描述符的显式配置共有 {item['pairs']} 对，默认/对照时间比的几何均值 {fmt(item['ratio_gmean'])}，观察范围 {item['ratio_range']}。这些比值用于检查噪声和漂移，不作为调优收益。"]
    lines += ["", "## 相同学习器下的特征比较", "", "|划分|方法|工作负载数|相对默认速度几何均值|相对 oracle 时间几何均值|距 oracle ≤2%|≤5%|",
              "|---|---|---:|---:|---:|---:|---:|"]
    labels={"default":"默认","best_global_train":"训练集最优固定配置","raw_config":"原始配置参数","planner_structural":"规划结构描述符","combined":"配置参数与结构描述符"}
    for mode,model in report["models"].items():
        if not isinstance(model,dict) or model.get("status")!="ok":
            continue
        for name,label in labels.items():
            item=model["results"][name]
            lines.append(f"|{mode}|{label}|{item['workloads']}|{fmt(item['speedup_vs_default_gmean'])}|{fmt(item['slowdown_vs_oracle_gmean'])}|{100*item['fraction_within_2pct']:.1f}%|{100*item['fraction_within_5pct']:.1f}%|")
    lines += ["", "模型设计在主筛选产生完整工作负载之前固定（见 model_design.json）：主要标签为 log(t_candidate/t_default)，绝对 log 时间仅作为单独的敏感性分析。默认计时只构造训练与评价标签，不作推理特征。每个工作负载具有相等训练权重。",
              "", "主分析按精确 N 留出，所有 batch 和候选配置共同分组；更严格的次分析按 N 的奇数部分留出整个长度族。两者均只用训练数据拟合特征词表、模型与最优固定配置。结构模型排除所有 cuda_* 编译器资源字段，使用相同 ExtraTrees 学习器。模型结果基于筛选数据，属于探索性验证。",
              "", "top2/top4 指模型给出的短名单内再使用已测 oracle 选优的假设结果；实际实验初始化了所有计划。未测量计划生成但不编译的路径、实际编译规避收益或模型在线端到端开销。",
              "", "图表：`screen_oracle_gap.png`、`confirmed_speedup.png`、`candidate_rank_heatmap.png`、`model_ablation.png`、`model_shortlist_hypothetical.png`、`within_process_drift.png`、`negative_control_ratios.png`（有数据时生成）。"]
    (output / "results_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def quality_audit(records,calibration_records,manifest,output,provenance=None):
    """Protocol checks for partial snapshots; deliberately no tuning/model metrics."""
    validation=validation_analysis(records,calibration_records,manifest,output)
    (output / "validation.json").write_text(json.dumps(validation,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    groups={}
    for row in records:
        if row["status"]=="ok":
            groups.setdefault((row["phase"],row["workload_id"],row["round"]),[]).append(row)
    noops=[]
    for (phase,wid,round_no),rows in groups.items():
        control=negative_control(rows)
        if control is None:
            continue
        baseline=next(r for r in rows if r["candidate_id"]=="default")
        explicit=next(r for r in rows if r["candidate_id"]==control)
        noops.append(dict(phase=phase,workload_id=wid,round=round_no,candidate_id=control,
                          descriptor_equal=True,resolved_equal=True,selected_control=True,
                          default_over_candidate=baseline["result"]["exec_ms_median"]/explicit["result"]["exec_ms_median"],
                          order_difference=explicit.get("order",0)-baseline.get("order",0)))
    dump_csv(output / "negative_controls.csv",noops)
    planned=manifest.get("workloads",[])
    support=[]
    for w in planned:
        wid=f"n{w['n']}_b{w['batch']}"
        rows=[r for r in records if r["phase"]=="screen" and r["workload_id"]==wid]
        valid=[r for r in rows if r["status"]=="ok"]
        support.append(dict(workload_id=wid,n=w["n"],batch=w["batch"],attempted=len(rows),valid=len(valid),
                            failed=len(rows)-len(valid),expected=len(manifest.get("candidates",CANDIDATES)),
                            valid_default=any(r["candidate_id"]=="default" for r in valid)))
    dump_csv(output / "workload_coverage.csv",support)
    ratios=[r["default_over_candidate"] for r in noops]
    report=dict(mode="quality_only",measurements=len(records),valid=sum(r["status"]=="ok" for r in records),
                failed=sum(r["status"]!="ok" for r in records),planned_workloads=len(planned),
                complete_workloads=sum(r["valid"]==r["expected"] for r in support),
                incomplete_workloads=sum(r["valid"]!=r["expected"] for r in support),
                attempted_workloads=sum(r["attempted"]>0 for r in support),
                unattempted_workloads=sum(r["attempted"]==0 for r in support),
                missing_valid_default=sum(not r["valid_default"] for r in support),
                executable_sha256=manifest.get("executable_sha256"),
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),snapshot_provenance=provenance or {},
                negative_control=dict(pairs=len(noops),ratio_gmean=geomean(ratios),ratio_distribution=distribution(ratios),
                                      interpretation="Measurement noise/drift diagnostic only; no tuning gain claim"),
                validation=validation,models="skipped_quality_only",tuning_performance="deferred; incomplete snapshot")
    report["plots"]=plots(output,[],[],[],negative_controls=noops)
    raw=[x for r in records if r["status"]=="ok" for x in r.get("result",{}).get("raw_interval_ms_samples",[])]
    if raw:
        try:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(8,4))
            ax.hist(np.log10(raw),bins=40,color="#577bb6",alpha=.8)
            target=manifest.get("graph_target_ms")
            if target:
                ax.axvline(math.log10(target),color="black",linewidth=.8,label=f"default calibration target {target} ms")
                ax.legend()
            ax.set(xlabel="log10(raw graph interval in ms)",ylabel="Samples",title="Partial-snapshot timing interval audit")
            fig.savefig(output / "raw_interval_distribution.png",dpi=140,bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass
    (output / "quality_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def geomean(xs):
    return float(np.exp(np.mean(np.log(xs)))) if len(xs) else None


def bootstrap_geomean(xs, rng, count=4000):
    if not len(xs):
        return [None, None]
    logs = np.log(xs)
    means = [float(np.exp(np.mean(rng.choice(logs, size=len(logs), replace=True))))
             for _ in range(count)]
    return [float(x) for x in np.quantile(means, [.025, .975])]


def cluster_n_bootstrap(rows, rng, count=4000):
    if not rows:
        return [None, None]
    groups = {}
    for row in rows:
        groups.setdefault(row["n"], []).append(math.log(row["speedup"]))
    keys = list(groups)
    means = [math.exp(float(np.mean([x for n in rng.choice(keys, len(keys), replace=True)
                                    for x in groups[n]]))) for _ in range(count)]
    return [float(x) for x in np.quantile(means, [.025, .975])]


def signature(result):
    # Preserve nested arrays so distinct stage/kernel structures stay distinct.
    obj = result.get("structural_features", {})
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def flatten_features(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for k, v in sorted(value.items()):
            flat.update(flatten_features(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(value, (list, tuple)):
        flat[prefix + ".length"] = float(len(value))
        for i, v in enumerate(value):
            flat.update(flatten_features(v, f"{prefix}.{i}"))
    elif isinstance(value, (int, float, bool)) and math.isfinite(value):
        flat[prefix] = float(value)
    return flat


def precompile_features(structure):
    """Exclude CUDA compiler/resource observations from planner descriptors."""
    flat = flatten_features(structure)
    return {k:v for k,v in flat.items()
            if not any(part.startswith("cuda_") for part in k.split("."))}


def workload_features(w):
    n = w["n"]
    out = dict(log2_n=math.log2(n), log2_batch=math.log2(w["batch"]),
               log2_elements=math.log2(n * w["batch"]))
    for prime in (2, 3, 5, 7):
        factor = 0
        while n % prime == 0:
            factor += 1
            n //= prime
        out[f"factor_{prime}"] = factor
    out["remaining_factor"] = n
    return out


def model_analysis(rows, output, seed, group_mode="exact_n",target_mode="relative"):
    """Same regressor/splits for raw knobs and post-compilation structural data.

    Exact-N holds every batch/candidate of N together; family-OOD holds every
    length/batch/candidate of an odd-part family together. All feature fitting
    uses training rows only. Structural inputs require actual initialized plans.
    """
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return dict(status="skipped", reason="scikit-learn unavailable")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["workload_id"], []).append(row)
    # Restrict all baselines/models to identical candidate support.
    eligible = [r for rs in grouped.values()
                if len(rs) == len(CANDIDATES) and any(r["candidate_id"] == "default" for r in rs)
                for r in rs]
    eligible.sort(key=lambda r:(r["workload_id"],r["candidate_id"]!="default",r["candidate_id"]))
    def n_family(row):
        n = row["workload"]["n"]
        while n % 2 == 0:
            n //= 2
        return f"odd_part_{n}"
    groups = np.array([n_family(r) if group_mode == "family_ood" else r["workload"]["n"] for r in eligible])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return dict(status="skipped", reason=f"need at least three complete {group_mode} groups")
    default_times={r["workload_id"]:r["result"]["exec_ms_median"] for r in eligible if r["candidate_id"]=="default"}
    y = np.log([r["result"]["exec_ms_median"]/(default_times[r["workload_id"]] if target_mode=="relative" else 1.) for r in eligible])
    raw, structural,combined = [], [], []
    for row in eligible:
        base = workload_features(row["workload"])
        config = row["config"]
        raw.append(dict(base, default=int(row["candidate_id"] == "default"),
                        **{k: config.get(k, 0) for k in ("aim_threads", "register_boost", "use_lut")}))
        structural.append(dict(base, **precompile_features(row["result"].get("structural_features", {}))))
        combined.append(dict(raw[-1],**structural[-1]))
    features_by_set=(("raw_config",raw),("planner_structural",structural),("combined",combined))
    forbidden={"graph_repeats","exec_ms_median","exec_ms_samples","initialize_ms","graph_build_ms","raw_interval_ms_median"}
    if any(any(part in forbidden or part.startswith("cuda_") for part in field.split("."))
           for _,features in features_by_set for feature in features for field in feature):
        raise ValueError("measurement or compiler field leaked into model features")
    splits = list(GroupKFold(n_splits=min(5, len(unique_groups))).split(raw, y, groups))
    selections = []
    for fold, (train, test) in enumerate(splits):
        train_workloads = {}
        for idx in train:
            r = eligible[idx]
            train_workloads.setdefault(r["workload_id"], {})[r["candidate_id"]] = r["result"]["exec_ms_median"]
        fixed = min((c[0] for c in CANDIDATES), key=lambda cid: geomean([
            timings[cid] / timings["default"] for timings in train_workloads.values()]))
        predictions = {}
        for name, features in features_by_set:
            vectorizer = DictVectorizer(sparse=False)
            x_train = vectorizer.fit_transform([features[i] for i in train])
            x_test = vectorizer.transform([features[i] for i in test])
            # Equal capacity and training groups; modest fixed hyperparameters.
            model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2,
                                        max_features=1.0, random_state=seed, n_jobs=1)
            model.fit(x_train, y[train],sample_weight=np.full(len(train),1./len(CANDIDATES)))
            predictions[name] = dict(zip(test.tolist(), model.predict(x_test).tolist()))
        for wid in sorted({eligible[i]["workload_id"] for i in test}):
            indices = [i for i in test if eligible[i]["workload_id"] == wid]
            timings = {eligible[i]["candidate_id"]: eligible[i]["result"]["exec_ms_median"] for i in indices}
            oracle = min(timings.values())
            choices = {"default": "default", "best_global_train": fixed}
            choices.update({name: eligible[min(indices, key=lambda i: pred[i])]["candidate_id"]
                            for name, pred in predictions.items()})
            for name, cid in choices.items():
                selections.append(dict(fold=fold, workload_id=wid, model=name, candidate_id=cid,
                                       exec_ms=timings[cid], oracle_ms=oracle,
                                       speedup_vs_default=timings["default"] / timings[cid],
                                       slowdown_vs_oracle=timings[cid] / oracle,
                                       within_2pct=int(timings[cid] / oracle <= 1.02),
                                       within_5pct=int(timings[cid] / oracle <= 1.05)))
            for name, pred in predictions.items():
                ordered = sorted(indices, key=lambda i:pred[i])
                for shortlist_size in (2, 4):
                    selected_i = min(ordered[:shortlist_size], key=lambda i:eligible[i]["result"]["exec_ms_median"])
                    cid = eligible[selected_i]["candidate_id"]
                    selections.append(dict(fold=fold, workload_id=wid,
                                           model=f"{name}_hypothetical_top{shortlist_size}_oracle",
                                           candidate_id=cid, exec_ms=timings[cid], oracle_ms=oracle,
                                           speedup_vs_default=timings["default"] / timings[cid],
                                           slowdown_vs_oracle=timings[cid] / oracle,
                                           within_2pct=int(timings[cid] / oracle <= 1.02),
                                           within_5pct=int(timings[cid] / oracle <= 1.05)))
    dump_csv(output / f"model_selections_{target_mode}_{group_mode}.csv", selections)
    aggregate = {}
    for name in sorted({r["model"] for r in selections}):
        rs = [r for r in selections if r["model"] == name]
        aggregate[name] = dict(workloads=len(rs),
                               speedup_vs_default_gmean=geomean([r["speedup_vs_default"] for r in rs]),
                               slowdown_vs_oracle_gmean=geomean([r["slowdown_vs_oracle"] for r in rs]),
                               fraction_within_2pct=float(np.mean([r["within_2pct"] for r in rs])),
                               fraction_within_5pct=float(np.mean([r["within_5pct"] for r in rs])))
    return dict(status="ok", split=("GroupKFold by N family (odd part), all lengths/batches/candidates together" if group_mode == "family_ood"
                                   else "GroupKFold by exact N, all batches/candidates together"),
                target="log(t_candidate/t_default)" if target_mode=="relative" else "log(t_candidate_ms), sensitivity only",
                default_timing_usage="training/evaluation labels only; never an inference feature",
                weighting="each workload total training weight 1; each of 19 candidates weight 1/19",
                feature_sets=[name for name,_ in features_by_set],
                validation="screen timings only; exploratory, not independent confirmation",
                structural_features_require_plan_initialization=True,
                compiler_resource_features_excluded="all nested cuda_* fields",
                shortlist_interpretation="hypothetical oracle within model top2/top4; all plans were actually initialized; shortlist online costs unmeasured",
                compile_avoidance_measured=False, online_cost_measured=False, results=aggregate)


def plots(output, summaries, confirmed, rows, models=None, negative_controls=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable"
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    if summaries:
        fig, ax = plt.subplots(figsize=(8, 4))
        families = sorted({s.get("family", "unknown") for s in summaries})
        colors = {family:f"C{i}" for i,family in enumerate(families)}
        for family in families:
            for is_single in (True, False):
                ss = sorted([s for s in summaries if s.get("family", "unknown") == family
                             and (s["batch"] == 1) == is_single], key=lambda s:s["n"])
                if ss:
                    ax.scatter([s["n"] for s in ss], [s["oracle_speedup"] for s in ss],
                               marker="o" if is_single else "^", color=colors[family], s=25,
                               label=f"{family}; {'B=1' if is_single else 'throughput batch'}")
        ax.set(xscale="log", xlabel="Transform length N", ylabel="Default / best tested execution time",
               title="Screened opportunity (selection-biased)")
        ax.axhline(1, color="black", linewidth=.8)
        ax.legend()
        fig.savefig(output / "screen_oracle_gap.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter([s["unique_structures"] for s in summaries], [s["oracle_speedup"] for s in summaries], s=20)
        ax.set(xlabel="Distinct effective structural fingerprints", ylabel="Default / best tested execution time",
               title="Requested configurations and observed descriptors")
        fig.savefig(output / "structure_opportunity.png")
        plt.close(fig)
        complete_ids = [s["workload_id"] for s in sorted(summaries, key=lambda s:(s["n"],s["batch"])) if s["complete"]]
        if complete_ids:
            rank_matrix = np.full((len(CANDIDATES), len(complete_ids)), np.nan)
            lookup = {(r["workload_id"], r["candidate_id"]):r["result"]["exec_ms_median"] for r in rows}
            for j, wid in enumerate(complete_ids):
                ordered = sorted(CANDIDATES, key=lambda c:lookup[(wid,c[0])])
                for rank, (cid, _) in enumerate(ordered, 1):
                    rank_matrix[[c[0] for c in CANDIDATES].index(cid),j] = rank
            fig, ax = plt.subplots(figsize=(12, 6))
            plot = ax.imshow(rank_matrix, aspect="auto", interpolation="nearest", vmin=1, vmax=len(CANDIDATES), cmap="viridis_r")
            ax.set_yticks(np.arange(len(CANDIDATES)), [c[0] for c in CANDIDATES], fontsize=7)
            stride = max(1,len(complete_ids)//16)
            ax.set_xticks(np.arange(0,len(complete_ids),stride), complete_ids[::stride], rotation=65, ha="right", fontsize=7)
            ax.set(title="Candidate execution ranks depend on workload (screened)", xlabel="Workload", ylabel="Requested configuration")
            fig.colorbar(plot, ax=ax, label="Rank (1 fastest)")
            fig.savefig(output / "candidate_rank_heatmap.png")
            plt.close(fig)
    if confirmed:
        fig, ax = plt.subplots(figsize=(8, 4))
        ss = sorted(confirmed, key=lambda s:s["speedup"])
        x = np.arange(len(ss))
        ax.errorbar(x, [s["speedup"] for s in ss],
                    yerr=np.array([[s["speedup"] - s["range_low"] for s in ss],
                                   [s["range_high"] - s["speedup"] for s in ss]]), fmt="o", markersize=3, capsize=2)
        ax.axhline(1, color="black", linewidth=.8)
        ax.set(xlabel="Workloads ordered by confirmed speedup", ylabel="Default / selected execution time",
               title=("Partial independent confirmation; available-round range" if any(s["rounds"]<3 for s in confirmed)
                      else "Independent interleaved confirmation (observed paired-round range)"))
        fig.savefig(output / "confirmed_speedup.png")
        plt.close(fig)
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter([r["result"]["exec_ms_median"] for r in rows],
                   [r["result"].get("initialize_ms", float("nan")) for r in rows], s=8, alpha=.5)
        ax.set(xscale="log", yscale="log", xlabel="Median device execution (ms)",
               ylabel="Initialization including compile (ms)", title="Measured planning cost and device execution")
        fig.savefig(output / "initialize_vs_execution.png")
        plt.close(fig)
    modes = [mode for mode in ("exact_n","family_ood") if (models or {}).get(mode,{}).get("status")=="ok"]
    if modes:
        names = ["default","best_global_train","raw_config","planner_structural","combined"]
        labels = ["Default","Best fixed\n(training)","Raw knobs","Planner\nstructure","Knobs +\nstructure"]
        fig, axes = plt.subplots(1,len(modes),figsize=(6*len(modes),4),squeeze=False)
        for ax,mode in zip(axes[0],modes):
            results = models[mode]["results"]
            heights = [results[name]["slowdown_vs_oracle_gmean"] for name in names]
            ax.bar(labels,heights,color=["#999999","#d8a34d","#577bb6","#3b9c88","#9169ab"])
            ax.axhline(1,color="black",linewidth=.8)
            ax.set(title="Exact N held out" if mode=="exact_n" else "Odd-part family held out",
                   ylabel="Selected / oracle time (geometric mean)")
            for x,height in enumerate(heights):
                ax.text(x,height,f"{height:.3f}",ha="center",va="bottom",fontsize=9)
        fig.suptitle("Primary relative-log target; same learner and grouped splits (exploratory)")
        fig.tight_layout()
        fig.savefig(output / "model_ablation.png")
        plt.close(fig)
        fig, axes = plt.subplots(1,len(modes),figsize=(6*len(modes),4),squeeze=False)
        shortlist_names = ["raw_config","raw_config_hypothetical_top2_oracle","raw_config_hypothetical_top4_oracle",
                           "planner_structural","planner_structural_hypothetical_top2_oracle","planner_structural_hypothetical_top4_oracle",
                           "combined","combined_hypothetical_top2_oracle","combined_hypothetical_top4_oracle"]
        for ax,mode in zip(axes[0],modes):
            results=models[mode]["results"]
            heights=[100*results[name]["fraction_within_2pct"] for name in shortlist_names]
            ax.bar(["R1","R2","R4","S1","S2","S4","C1","C2","C4"],heights,
                   color=["#577bb6"]*3+["#3b9c88"]*3+["#9169ab"]*3)
            ax.set(ylim=(0,108),ylabel="Workloads within 2% of oracle (%)",
                   title="Exact N held out" if mode=="exact_n" else "Odd-part family held out")
        fig.suptitle("Top2/4 use a hypothetical shortlist oracle; all plans were actually initialized")
        fig.tight_layout()
        fig.savefig(output / "model_shortlist_hypothetical.png")
        plt.close(fig)
    drift_path=output / "process_drift.csv"
    if drift_path.exists():
        with drift_path.open(encoding="utf-8") as f:
            ratios=[float(r["late_over_early"]) for r in csv.DictReader(f)]
        if ratios:
            fig,ax=plt.subplots(figsize=(8,4))
            ax.hist(np.log10(ratios),bins=40,color="#577bb6",alpha=.8)
            ax.axvline(0,color="black",linewidth=.8)
            ax.set(xlabel="log10(last-third / first-third median execution time)",ylabel="Processes",
                   title="Within-process sample drift after warmup (descriptive)")
            fig.savefig(output / "within_process_drift.png")
            plt.close(fig)
    controls=[r for r in (negative_controls or []) if r["resolved_equal"] and r["descriptor_equal"] and r["selected_control"]]
    if controls:
        fig,ax=plt.subplots(figsize=(8,4))
        for phase in sorted({r["phase"] for r in controls}):
            values=[r["default_over_candidate"] for r in controls if r["phase"]==phase]
            ax.hist(values,bins=min(20,max(5,len(values)//3)),alpha=.55,label=phase)
        ax.axvline(1,color="black",linewidth=.8)
        ax.set(xlabel="Default / explicit matching-descriptor candidate",ylabel="Pairs",
               title="Negative-control ratios diagnose noise/drift; not tuning gains")
        ax.legend()
        fig.savefig(output / "negative_control_ratios.png")
        plt.close(fig)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="suite output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--quality-only",action="store_true",help="Partial-snapshot validation only; no tuning performance or model evaluation")
    args = parser.parse_args()
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    records = list({key(r):r for r in read_records(args.input / "measurements.jsonl")}.values())
    if not records:
        parser.error("no measurements found")
    valid = [r for r in records if r["status"] == "ok"]
    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    calibration_records=list({key(r):r for name in ("calibration.jsonl","calibration_verification.jsonl")
                              for r in read_records(args.input / name)}.values())
    if args.quality_only:
        provenance=dict(measurements_sha256=hashlib.sha256((args.input / "measurements.jsonl").read_bytes()).hexdigest(),
                        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None)
        print(json.dumps(quality_audit(records,calibration_records,manifest,output,provenance),indent=2,allow_nan=False))
        return
    attempted_screen_ids = {r["workload_id"] for r in records if r["phase"] == "screen"}
    planned_workloads = manifest.get("workloads", [])
    planned_screen_count = len(planned_workloads) if planned_workloads else len(attempted_screen_ids)
    screen = [r for r in valid if r["phase"] == "screen"]
    rng = np.random.default_rng(args.seed)
    summary, detailed, structures, noops = [], [], [], []
    by_workload = {}
    for row in screen:
        by_workload.setdefault(row["workload_id"], []).append(row)
    for wid, rows in sorted(by_workload.items()):
        baseline = next((r for r in rows if r["candidate_id"] == "default"), None)
        if baseline is None:
            continue
        baseline_ms = baseline["result"]["exec_ms_median"]
        control = negative_control(rows)
        for candidate in rows:
            if candidate["candidate_id"] == "default":
                continue
            observed_equal = bool(baseline["result"].get("structural_features")) and candidate["result"].get("structural_features") == baseline["result"]["structural_features"]
            resolved_equal = bool(baseline["result"].get("resolved_config")) and candidate["result"].get("resolved_config") == baseline["result"]["resolved_config"]
            if observed_equal:
                noops.append(dict(phase="screen", workload_id=wid, candidate_id=candidate["candidate_id"], round=0,
                                  descriptor_equal=True, resolved_equal=resolved_equal,
                                  selected_control=candidate["candidate_id"] == control,
                                  default_over_candidate=baseline_ms / candidate["result"]["exec_ms_median"],
                                  order_difference=candidate.get("order", 0) - baseline.get("order", 0)))
        ranked = sorted(rows, key=lambda r:r["result"]["exec_ms_median"])
        best = ranked[0]
        fingerprint_groups = {}
        for rank, row in enumerate(ranked, 1):
            result = row["result"]
            fingerprint = signature(result)
            fingerprint_groups.setdefault(fingerprint, []).append(row["candidate_id"])
            detailed.append(dict(workload_id=wid, n=row["workload"]["n"], batch=row["workload"]["batch"],
                                 candidate_id=row["candidate_id"], rank=rank,
                                 exec_ms=result["exec_ms_median"], initialize_ms=result.get("initialize_ms"),
                                 graph_build_ms=result.get("graph_build_ms"), timing_mode=result.get("timing_mode", "direct"),
                                 graph_repeats=result.get("graph_repeats",1),placement=result.get("placement","inplace"),
                                 raw_interval_ms=result.get("raw_interval_ms_median"),
                                 speedup_vs_default=baseline_ms / result["exec_ms_median"],
                                 structural_fingerprint=fingerprint))
        summary.append(dict(workload_id=wid, **best["workload"],
                            default_ms=baseline_ms, best_ms=best["result"]["exec_ms_median"],
                            winner=best["candidate_id"], oracle_speedup=baseline_ms / best["result"]["exec_ms_median"],
                            valid_candidates=len(rows), total_candidates=len(CANDIDATES),
                            complete=int(len(rows) == len(CANDIDATES)), unique_structures=len(fingerprint_groups)))
        for fingerprint, candidates in fingerprint_groups.items():
            structures.append(dict(workload_id=wid, fingerprint=fingerprint, count=len(candidates),
                                   candidates=";".join(sorted(candidates))))
    confirmed = []
    confirmations = [r for r in valid if r["phase"] == "confirm"]
    for row in summary:
        wid, winner = row["workload_id"], row["winner"]
        observations = {(r["round"], r["candidate_id"]):r for r in confirmations if r["workload_id"] == wid}
        control = negative_control(by_workload[wid])
        ratios = []
        for round_no in sorted({r["round"] for r in confirmations if r["workload_id"] == wid}):
            baseline, chosen = observations.get((round_no, "default")), observations.get((round_no, winner))
            if baseline is not None and chosen is not None:
                ratios.append(baseline["result"]["exec_ms_median"] / chosen["result"]["exec_ms_median"])
            explicit = observations.get((round_no, control)) if control else None
            if baseline is not None and explicit is not None:
                noops.append(dict(phase="confirm", workload_id=wid, candidate_id=control, round=round_no,
                                  descriptor_equal=explicit["result"].get("structural_features") == baseline["result"].get("structural_features"),
                                  resolved_equal=explicit["result"].get("resolved_config") == baseline["result"].get("resolved_config"),
                                  selected_control=True,
                                  default_over_candidate=baseline["result"]["exec_ms_median"] / explicit["result"]["exec_ms_median"],
                                  order_difference=explicit.get("order", 0) - baseline.get("order", 0)))
        if ratios:
            confirmed.append(dict(workload_id=wid, n=row["n"], batch=row["batch"], winner=winner,
                                  rounds=len(ratios), speedup=geomean(ratios), range_low=min(ratios), range_high=max(ratios),
                                  regression=int(geomean(ratios) < 1),
                                  regression_over_5pct=int(geomean(ratios) < 1/1.05),
                                  all_rounds_regression=int(max(ratios) < 1),
                                  screen_speedup=row["oracle_speedup"]))
    for filename, rows in (("workloads.csv", summary), ("candidate_ranks.csv", detailed),
                           ("effective_structures.csv", structures), ("confirmation.csv", confirmed),
                           ("negative_controls.csv", noops)):
        dump_csv(output / filename, rows)
    complete = [r for r in summary if r["complete"]]
    gaps = [r["oracle_speedup"] for r in complete]
    speeds = [r["speedup"] for r in confirmed]
    failures = [dict(phase=r["phase"], workload_id=r["workload_id"], candidate_id=r["candidate_id"],
                     round=r["round"], status=r["status"], error=r.get("error", ""))
                for r in records if r["status"] != "ok"]
    dump_csv(output / "failures.csv", failures)
    # Expose rank/workload dependence through per-config aggregate and rank range.
    config_stats = []
    for cid, _ in CANDIDATES:
        rs = [r for r in detailed if r["candidate_id"] == cid and r["workload_id"] in {s["workload_id"] for s in complete}]
        if rs:
            config_stats.append(dict(candidate_id=cid, workloads=len(rs),
                                     speedup_gmean=geomean([r["speedup_vs_default"] for r in rs]),
                                     rank_min=min(r["rank"] for r in rs), rank_max=max(r["rank"] for r in rs),
                                     wins=sum(r["rank"] == 1 for r in rs)))
    dump_csv(output / "fixed_configs.csv", config_stats)
    report = dict(schema_version=1, measurements=len(records), valid=len(valid), failures=len(failures),
                  screen_workloads_with_default=len(summary), complete_screen_workloads=len(complete),
                  planned_screen_workloads=planned_screen_count,
                  incomplete_screen_workloads=planned_screen_count - len(complete),
                  attempted_screen_workloads=len(attempted_screen_ids),
                  screen_workloads_missing_valid_default=len(attempted_screen_ids) - len(summary),
                  screen_workloads_not_attempted=planned_screen_count - len(attempted_screen_ids),
                  screen_oracle_speedup_gmean=geomean(gaps),
                  screen_oracle_speedup_quantiles=(dict(zip(("p0", "p25", "p50", "p75", "p90", "p100"),
                                                          map(float, np.quantile(gaps, [0, .25, .5, .75, .9, 1])))) if gaps else {}),
                  confirmed_workloads=len(confirmed), confirmation_speedup_gmean=geomean(speeds),
                  confirmation_gmean_designed_suite_n_cluster_bootstrap_95ci=cluster_n_bootstrap(confirmed, rng),
                  confirmation_regressions=sum(r["regression"] for r in confirmed),
                  confirmation_regressions_over_5pct=sum(r["regression_over_5pct"] for r in confirmed),
                  confirmation_all_rounds_regressions=sum(r["all_rounds_regression"] for r in confirmed),
                  mean_unique_observed_descriptors=float(np.mean([r["unique_structures"] for r in summary])) if summary else None,
                  best_global_screen_config=max(config_stats, key=lambda x:x["speedup_gmean"]) if config_stats else None,
                  devices=[json.loads(x) for x in sorted({json.dumps({k:v for k,v in r["result"].get("gpu",{}).items()
                            if k not in ("free_memory_bytes",)},sort_keys=True) for r in valid})],
                  measurement_metadata={k:sorted({str(r["result"].get(k, "unknown")) for r in valid})
                                        for k in ("backend", "vkfft_version", "upstream_commit", "cuda_runtime_version", "cuda_driver_version", "nvrtc_version", "precision", "timing_scope", "timing_metric", "timing_mode", "placement", "input_cache_regime")},
                  correctness_criteria=sorted({r.get("result",{}).get("correctness",{}).get("pass_rule","unknown") for r in records}),
                  limitations=["Screen winner and oracle are selection-biased; independent confirmation is primary.",
                               "Device timings use input restored before each sample, with restore outside CUDA events. Graph repeats use immutable out-of-place input and per-FFT normalization.",
                               "Initialization includes compilation; structural features require initialized plans.",
                               "Graph construction, when used, is separately recorded; device replay timings exclude graph construction and input restore.",
                               "Descriptor fingerprints identify matching recorded fields, not proven equivalent generated kernels.",
                               "Negative controls with matching resolved config and recorded descriptors diagnose noise/drift; their ratios are not tuning gains.",
                               "No measured plan-only pathway, compile avoidance, or online model end-to-end overhead.",
                               "Per-workload confirmation bars are observed paired-round ranges, not confidence intervals.",
                               "Aggregate bootstrap resamples designed-suite N clusters with batches together; this is not population inference.",
                               "Primary models hold out exact N; related lengths may cross folds. Secondary family-OOD models have only three family groups."])
    report["negative_controls"] = {}
    required_rounds=manifest.get("confirm_rounds",3)
    report["confirmation_required_rounds"]=required_rounds
    report["complete_confirmed_workloads"]=sum(r["rounds"]>=required_rounds for r in confirmed)
    report["partial_confirmation_workloads"]=sum(r["rounds"]<required_rounds for r in confirmed)
    report["confirmation_primary_ready"]=(planned_screen_count>0 and report["complete_confirmed_workloads"]==planned_screen_count)
    for phase in ("screen", "confirm"):
        selected = [r for r in noops if r["phase"]==phase and r["resolved_equal"] and r["descriptor_equal"] and r["selected_control"]]
        values = [r["default_over_candidate"] for r in selected]
        order_differences=[r["order_difference"] for r in selected]
        correlation=(float(np.corrcoef(order_differences,np.log(values))[0,1])
                     if len(values)>2 and np.std(order_differences)>0 and np.std(np.log(values))>0 else None)
        report["negative_controls"][phase] = dict(pairs=len(values), ratio_gmean=geomean(values),
                                                  ratio_range=[min(values), max(values)] if values else [],
                                                  fraction_over_2pct_deviation=(sum(abs(x-1)>.02 for x in values)/len(values)) if values else None,
                                                  fraction_over_5pct_deviation=(sum(abs(x-1)>.05 for x in values)/len(values)) if values else None,
                                                  signed_order_difference_log_ratio_correlation=correlation,
                                                  candidate_after_default_ratio_gmean=geomean([r["default_over_candidate"] for r in selected if r["order_difference"]>0]),
                                                  candidate_before_default_ratio_gmean=geomean([r["default_over_candidate"] for r in selected if r["order_difference"]<0]),
                                                  interpretation="noise/drift diagnostic, not tuning gain")
    report["validation"]=validation_analysis(records,calibration_records,manifest,output)
    (output / "validation.json").write_text(json.dumps(report["validation"],indent=2,allow_nan=False)+"\n",encoding="utf-8")
    design_path=Path(__file__).with_name("model_design.json")
    report["model_design"]=json.loads(design_path.read_text(encoding="utf-8")) if design_path.exists() else {"status":"missing"}
    (output / "model_design.json").write_text(json.dumps(report["model_design"],indent=2)+"\n",encoding="utf-8")
    report["models"] = ({mode:model_analysis(screen, output, args.seed, mode,"relative") for mode in ("exact_n", "family_ood")}
                        if not args.skip_models else {"status":"skipped"})
    report["models_absolute_sensitivity"] = ({mode:model_analysis(screen, output, args.seed, mode,"absolute") for mode in ("exact_n", "family_ood")}
                                             if not args.skip_models else {"status":"skipped"})
    report["plots"] = plots(output, summary, confirmed, screen,report["models"],noops)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = ["# VkFFT plan-tuning measurements", "", f"Successful measurements: {len(valid)}/{len(records)}; failures: {len(failures)}.",
            f"Complete screened workloads: {len(complete)}; workloads with available confirmation: {len(confirmed)}; planned-round complete: {report['complete_confirmed_workloads']}."]
    if gaps:
        text += [f"Screened default/best-tested geometric mean: {geomean(gaps):.4f}x (selection-biased)."]
    if speeds and report["confirmation_primary_ready"]:
        ci = report["confirmation_gmean_designed_suite_n_cluster_bootstrap_95ci"]
        text += [f"Independent confirmation geometric mean: {geomean(speeds):.4f}x; designed-suite N-cluster bootstrap 95% interval [{ci[0]:.4f}, {ci[1]:.4f}].",
                 f"Regressions: {report['confirmation_regressions']}; over 5% slowdown: {report['confirmation_regressions_over_5pct']}."]
    elif speeds:
        text += ["Independent confirmation schedule is incomplete; available early rounds are not the final planned confirmation result."]
    text += ["", "## Interpretation limits", ""] + [f"- {s}" for s in report["limitations"]]
    (output / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_chinese_report(report,output)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
