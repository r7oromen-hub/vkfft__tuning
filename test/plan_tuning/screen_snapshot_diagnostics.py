#!/usr/bin/env python3
"""Read-only diagnostics for a completed screen with incomplete confirmation."""
import argparse
import json
from pathlib import Path
import collections

from make_headline_checks import core_structure
from run_suite import key,read_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot",type=Path)
    args=p.parse_args()
    output=args.snapshot/"analysis"
    summary=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    anomalies=json.loads((output/"anomalous_negative_controls.json").read_text(encoding="utf-8"))
    rows=list({key(r):r for r in read_records(args.snapshot/"measurements.jsonl")}.values())
    screen=[r for r in rows if r["phase"]=="screen"]
    core_groups=collections.defaultdict(set)
    for row in screen:
        core_groups[row["workload_id"]].add(json.dumps(core_structure(row["result"]),sort_keys=True))
    core_counts=dict(collections.Counter(map(len,core_groups.values())))
    primary={mode:{name:stats for name,stats in summary["models"][mode]["results"].items()
                   if name in ("default","best_global_train","raw_config","planner_structural","combined")}
             for mode in ("exact_n","family_ood")}
    review=dict(status="screen_complete_confirmation_incomplete_quality_warning",screen_records=len(screen),
                early_confirmation_records=sum(r["phase"]=="confirm" for r in rows),
                complete_three_round_workloads=summary["complete_confirmed_workloads"],
                core_structure_count_histogram=core_counts,primary_relative_model_results=primary,
                anomalous_exact_controls=[x["workload_id"] for x in anomalies["anomalies"]],
                warning="Large apparent screen gains overlap exact-control anomalies and late extra GPU memory occupancy. No tuning-gain or model-superiority attribution is supported by those outliers; no records were removed or model rules changed.",
                primary_target_unchanged=True,all_records_preserved=True)
    (output/"screen_snapshot_review.json").write_text(json.dumps(review,indent=2)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(anomalies["anomalies"]),1,figsize=(11,12),squeeze=False)
    for ax,item in zip(axes[:,0],anomalies["anomalies"]):
        for record in item["records"]:
            ax.plot(range(1,len(record["raw_samples"])+1),record["raw_samples"],"o-",markersize=3,
                    label=f"{record['candidate_id']} order {record['order']} {record['started_utc'][11:23]} UTC")
        ax.set(title=f"{item['workload_id']}; frozen R={item['records'][0]['graph_repeats']}",ylabel="Raw graph interval (ms)")
        ax.legend(fontsize=8)
    axes[-1,0].set_xlabel("Timed sample within process")
    fig.suptitle("Exact recorded default/control matches: late processes contain timing state changes")
    fig.tight_layout()
    fig.savefig(output/"anomalous_control_sample_sequences.png",dpi=140,bbox_inches="tight")
    plt.close(fig)
    ordered=sorted(screen,key=lambda r:r["order"])
    bad_orders={max(item["records"],key=lambda r:r["order"])["order"] for item in anomalies["anomalies"]}
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot([r["order"] for r in ordered],[r["result"]["gpu"]["free_memory_bytes"]/2**30 for r in ordered],".",markersize=3)
    bad=[r for r in ordered if r["order"] in bad_orders]
    ax.scatter([r["order"] for r in bad],[r["result"]["gpu"]["free_memory_bytes"]/2**30 for r in bad],color="#c54040",s=30,label="late anomalous exact-control process")
    ax.set(xlabel="Screen process order",ylabel="Reported GPU free memory (GiB)",title="GPU metadata changes coincide with the late control anomalies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"gpu_free_memory_process_order.png",dpi=140,bbox_inches="tight")
    plt.close(fig)
    lines=["# 完整初筛快照：质量警告", "", "本目录含 912/912 初筛及 6 条早期 round0 confirm；完成完整三轮确认的工作负载为 0。",
           "", review["warning"], "", "精确 resolved+planner 匹配的 48 对负对照中，5 对偏离超过 5%。这些异常的较晚进程位于 UTC09:43–09:45，GPU 可用显存比早期配对少约 11GB；每对 graph R、节点/核数量、GPU UUID、编译器资源诊断一致。序列图显示晚进程内部同时出现正常与变慢状态。原始数据与既定模型均未修改。",
           "", "19 个请求配置平均折叠为 2.6667 类记录描述符；37/48 个工作负载只有 1 类 upload/radix/block 核心结构。筛选 oracle 差距含选择偏差，而且本快照的大尾部被负对照异常污染。排名热图中许多差异很小，不能直接解释成实质交叉。",
           "", "主要 relative-target exactN 结果：raw、planner、combined 相对默认几何均值分别为 0.978597、0.961381、1.040885；familyOOD 分别为 1.027436、1.002686、0.999151。结构单独并未占优，combined 也没有跨 family 稳定优势。绝对时间敏感性分析改变排名，不能事后改选它作主目标。所有数字仍受到上述筛选标签污染限制。",
           "", "图表 `anomalous_control_sample_sequences.png`、`gpu_free_memory_process_order.png` 与 `anomalous_negative_controls.json` 提供审计证据。须等待独立复测和预定追加检查，不将这些初筛异常称为调参收益或模型独立验证。"]
    (output/"SCREEN_SNAPSHOT_NOTES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(review,indent=2))


if __name__=="__main__":
    main()
