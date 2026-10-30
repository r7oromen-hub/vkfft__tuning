#!/usr/bin/env python3
"""Compare recorded negative controls and telemetry, separate from tuning results."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze import distribution, dump_csv, geomean
from run_suite import key, read_records


def control_set(folder):
    manifest=json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows=list({key(r):r for r in read_records(folder / "measurements.jsonl")}.values())
    lookup={key(r):r for r in rows}
    pairs=[]
    for row in rows:
        if row["candidate_id"]=="default" or row["status"]!="ok":
            continue
        baseline=lookup.get((row["phase"],row["workload_id"],"default",row["round"]))
        if baseline is None or baseline["status"]!="ok":
            continue
        a,b=baseline["result"],row["result"]
        pairs.append(dict(protocol=folder.name,warmup_ms=manifest["warmup_ms"],workload_id=row["workload_id"],
                          round=row["round"],candidate_id=row["candidate_id"],
                          ratio=a["exec_ms_median"]/b["exec_ms_median"],
                          descriptor_equal=a.get("structural_features")==b.get("structural_features"),
                          resolved_equal=a.get("resolved_config")==b.get("resolved_config"),
                          graph_repeats=b.get("graph_repeats"),
                          default_ms=a["exec_ms_median"],candidate_ms=b["exec_ms_median"],
                          default_raw_interval_ms=a.get("raw_interval_ms_median"),
                          candidate_raw_interval_ms=b.get("raw_interval_ms_median")))
    summary={}
    for wid in sorted({p["workload_id"] for p in pairs}):
        selected=[p for p in pairs if p["workload_id"]==wid]
        ratios=[p["ratio"] for p in selected]
        summary[wid]=dict(pairs=len(ratios),ratio_gmean=geomean(ratios),ratio_range=[min(ratios),max(ratios)],
                          ratio_median=statistics.median(ratios),sample_std_log_ratio=statistics.stdev(map(math.log,ratios)) if len(ratios)>1 else None,
                          max_absolute_deviation_from_one=max(abs(x-1) for x in ratios),
                          fraction_over_2pct=sum(abs(x-1)>.02 for x in ratios)/len(ratios),
                          all_descriptors_equal=all(p["descriptor_equal"] for p in selected),
                          all_resolved_equal=all(p["resolved_equal"] for p in selected),
                          graph_repeats=sorted({p["graph_repeats"] for p in selected}))
    drift=[]
    numerical=[]
    for row in rows:
        result=row.get("result",{})
        samples=result.get("exec_ms_samples",[])
        if len(samples)>=6:
            third=len(samples)//3
            drift.append(statistics.median(samples[-third:])/statistics.median(samples[:third]))
        for field in ("single_forward_correctness","correctness"):
            check=result.get(field)
            if isinstance(check,dict):
                numerical.append(dict(check=field,passed=check.get("passed"),criterion=check.get("criterion"),
                                      pointwise_passed=check.get("pointwise_passed")))
    return dict(warmup_ms=manifest["warmup_ms"],measurements=len(rows),successful=sum(r["status"]=="ok" for r in rows),
                executable_sha256=manifest.get("executable_sha256"),timing_mode=manifest.get("timing_mode"),
                placement=manifest.get("placement"),numerical_checks=len(numerical),
                numerical_failed=sum(n["passed"] is False for n in numerical),
                pointwise_diagnostic_failed=sum(n["pointwise_passed"] is False for n in numerical),
                workloads=summary,within_process_late_over_early=distribution(drift)),pairs,rows


def read_telemetry(path):
    records=[]
    zone=dt.timezone(dt.timedelta(hours=8))
    with path.open(encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row={k.strip():v.strip() for k,v in raw.items() if k is not None and v is not None}
            if not row.get("timestamp"):
                continue
            timestamp=dt.datetime.strptime(row["timestamp"],"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone).astimezone(dt.timezone.utc)
            def numeric(name):
                match=re.search(r"-?\d+(?:\.\d+)?",row.get(name,""))
                return float(match.group()) if match else None
            records.append(dict(timestamp_utc=timestamp,pstate=row.get("pstate"),
                                sm_mhz=numeric("clocks.current.sm [MHz]"),
                                memory_mhz=numeric("clocks.current.memory [MHz]"),
                                gpu_util_percent=numeric("utilization.gpu [%]"),power_w=numeric("power.draw [W]")))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    output=args.output or args.results
    output.mkdir(parents=True,exist_ok=True)
    folders=[args.results/name for name in ("controls_outofplace_graph_v2","controls_hot_1024","controls_hot_81920")]
    sets,pairs,all_rows={},{},{}
    provenance={}
    for folder in folders:
        if not folder.exists():
            continue
        sets[folder.name],pairs[folder.name],all_rows[folder.name]=control_set(folder)
        provenance[str(folder / "measurements.jsonl")]=hashlib.sha256((folder / "measurements.jsonl").read_bytes()).hexdigest()
    common=set(sets.get("controls_outofplace_graph_v2",{}).get("workloads",{}))
    hot_names=[name for name in sets if name.startswith("controls_hot_")]
    common &= {wid for name in hot_names for wid in sets[name]["workloads"]}
    comparison=[]
    for wid in sorted(common):
        for name in sets:
            if wid in sets[name]["workloads"]:
                comparison.append(dict(protocol=name,warmup_ms=sets[name]["warmup_ms"],workload_id=wid,
                                       **sets[name]["workloads"][wid]))
    telemetry_path=args.results / "controls_hot_telemetry.csv"
    telemetry=read_telemetry(telemetry_path) if telemetry_path.exists() else []
    hot_rows=[r for name in hot_names for r in all_rows[name]]
    window_rows=[]
    for row in hot_rows:
        start,end=dt.datetime.fromisoformat(row["started_utc"]),dt.datetime.fromisoformat(row["ended_utc"])
        samples=[s for s in telemetry if start<=s["timestamp_utc"]<=end]
        window_rows.append(dict(workload_id=row["workload_id"],candidate_id=row["candidate_id"],round=row["round"],
                                telemetry_samples=len(samples),sm_mhz_min=min((s["sm_mhz"] for s in samples if s["sm_mhz"] is not None),default=None),
                                sm_mhz_median=statistics.median([s["sm_mhz"] for s in samples if s["sm_mhz"] is not None]) if samples else None,
                                power_w_median=statistics.median([s["power_w"] for s in samples if s["power_w"] is not None]) if samples else None))
    dump_csv(output / "validation_control_comparison.csv",comparison)
    dump_csv(output / "validation_control_pairs.csv",[r for group in pairs.values() for r in group])
    dump_csv(output / "validation_telemetry_windows.csv",window_rows)
    analytic_path=args.results / "analytic_probes" / "analytic_probes_summary.json"
    analytic=json.loads(analytic_path.read_text(encoding="utf-8")) if analytic_path.exists() else {}
    report=dict(protocols=sets,common_workload_comparison=comparison,
                same_executable_for_compared_controls=len({sets[name]["executable_sha256"] for name in sets})==1,
                telemetry=dict(samples=len(telemetry),timestamp_assumption="timezone-less CSV interpreted as Asia/Shanghai UTC+08:00 to align with process UTC timestamps",
                               sm_mhz=distribution([s["sm_mhz"] for s in telemetry]),power_w=distribution([s["power_w"] for s in telemetry]),
                               pstate_counts={state:sum(s["pstate"]==state for s in telemetry) for state in sorted({s["pstate"] for s in telemetry})},
                               process_windows=len(window_rows),scope="Whole process windows include setup, compilation, warmup, timed samples and verification; no claim of kernel-only clock attribution."),
                analytic_validation={k:analytic.get(k) for k in ("total","passed","failed","criterion","patterns","lengths","batches","graph_repeats","pattern_metrics")},
                provenance_sha256=provenance,
                interpretation=["Negative-control ratios diagnose measurement repeatability and drift, not tuning gains.",
                                "100 ms and 1000 ms groups have different process-pair counts; observed ranges and log standard deviations are descriptive.",
                                "Exact ratio 1.0 can reflect finite timing precision; it does not prove zero variability in all settings.",
                                "Matching resolved configuration and recorded descriptors does not prove identical generated code.",
                                "This validation audit is separate from the main tuning experiment and its model results."])
    (output / "validation_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if common:
        fig,axes=plt.subplots(1,len(common),figsize=(6*len(common),4),squeeze=False)
        for ax,wid in zip(axes[0],sorted(common)):
            for x,warmup in enumerate((100,1000)):
                values=[r["ratio"] for group in pairs.values() for r in group if r["workload_id"]==wid and r["warmup_ms"]==warmup]
                jitter=[(i-(len(values)-1)/2)*.025 for i in range(len(values))]
                ax.scatter([x+j for j in jitter],values,color="#577bb6" if x==0 else "#3b9c88",s=30,alpha=.8)
                if values:
                    ax.hlines(geomean(values),x-.2,x+.2,color="black",linewidth=1.4)
            ax.axhline(1,color="gray",linewidth=.8,linestyle="--")
            ax.set_xticks([0,1],["100 ms warmup\n5 process pairs","1000 ms warmup\n10 process pairs"])
            ax.set(title=wid,ylabel="Default / explicit matching-descriptor control")
        fig.suptitle("Negative-control repeatability; observed pairs, not tuning gains")
        fig.tight_layout()
        fig.savefig(output / "validation_control_warmup_comparison.png",dpi=150,bbox_inches="tight")
        plt.close(fig)
    if telemetry:
        t0=telemetry[0]["timestamp_utc"]
        seconds=[(r["timestamp_utc"]-t0).total_seconds() for r in telemetry]
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        axes[0].plot(seconds,[r["sm_mhz"] for r in telemetry],color="#577bb6",linewidth=1)
        axes[1].plot(seconds,[r["power_w"] for r in telemetry],color="#3b9c88",linewidth=1)
        axes[0].set(ylabel="SM clock (MHz)",title="1000 ms control-run telemetry (whole process context)")
        axes[1].set(ylabel="Power (W)",xlabel="Seconds from telemetry start")
        fig.tight_layout()
        fig.savefig(output / "validation_control_telemetry.png",dpi=150,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(dict(common_workload_comparison=comparison,telemetry=report["telemetry"],
                         analytic_validation=report["analytic_validation"]),indent=2))


if __name__=="__main__":
    main()
