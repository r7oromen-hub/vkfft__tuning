"""Synthetic checks for predeclared follow-up selection and paired analysis."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import make_headline_checks as headline
import analyze_headline_checks as headline_analysis


A="a64_r1_l-1"
B="a256_r4_l1"
CONTROL="a128_r1_l-1"
GPU_UUID="GPU-synthetic-main-device"
CANDIDATES=[("default",{}),(A,{"aim_threads":64}),(B,{"aim_threads":256}),(CONTROL,{"aim_threads":128})]


def synthetic_main(folder):
    specifications=[(256,1,[10,5,8,10],2.1),(256,16,[10,9,5,10],1.8),
                    (768,1,[10,6,9,10],1.5),(1024,1,[10,4,9,10],1.9),
                    (1280,1,[10,9,9.5,10],.95),(2048,1,[5,6,7,5],1.)]
    ws=[dict(n=n,batch=batch) for n,batch,_,_ in specifications]
    manifest=dict(workloads=ws,candidates=CANDIDATES,confirm_rounds=3,timing_mode="graph",placement="outofplace",
                  executable="/synthetic/project/build/vkfft_plan_benchmark",executable_sha256="0"*64,source_commit="synthetic")
    selected,frozen,rows={},{},[]
    def result(w,cid,milliseconds,repeats):
        local=64 if cid in ("default",CONTROL) else (128 if cid==A else 256)
        return dict(status="ok",n=w["n"],batch=w["batch"],gpu=dict(uuid=GPU_UUID),exec_ms_samples=[milliseconds]*3,exec_ms_median=milliseconds,
                    graph_repeats=repeats,timing_mode="graph",placement="outofplace",
                    correctness=dict(passed=True,criterion="criterion_v2"),
                    resolved_config=dict(aimThreads=128 if cid in ("default",CONTROL) else local),
                    structural_features=dict(numAxisUploads=1,LUT=0,kernels=[dict(stageRadix=[4,4],localSize=[local,1,1],axisBlock=[local,1,1,1])]))
    for i,(n,batch,times,score) in enumerate(specifications):
        w=ws[i]
        wid=headline.workload_id(w)
        frozen[wid]=i+2
        winner=min(zip(CANDIDATES,times),key=lambda item:item[1])[0][0]
        selected[wid]=winner
        for (cid,config),milliseconds in zip(CANDIDATES,times):
            rows.append(dict(phase="screen",workload_id=wid,workload=dict(w,graph_repeats=i+2),candidate_id=cid,
                             config=config,round=0,status="ok",result=result(w,cid,milliseconds,i+2)))
        for round_no in range(3):
            for cid in dict.fromkeys(["default",winner,CONTROL]):
                milliseconds=10/score if cid==winner and winner!="default" else 10.
                rows.append(dict(phase="confirm",workload_id=wid,workload=dict(w,graph_repeats=i+2),candidate_id=cid,
                                 config=dict(CANDIDATES)[cid],round=round_no,status="ok",result=result(w,cid,milliseconds,i+2)))
    (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    (folder/"selected.json").write_text(json.dumps(selected),encoding="utf-8")
    (folder/"graph_repeats.json").write_text(json.dumps(frozen),encoding="utf-8")
    write_rows(folder,rows)
    return rows


def write_rows(folder,rows):
    (folder/"measurements.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")


class HeadlineSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_deduplicated_and_directional(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            rows=synthetic_main(folder)
            plan=headline.build_plan(folder)
            write_rows(folder,list(reversed(rows)))
            reversed_plan=headline.build_plan(folder)
            self.assertEqual(plan["cases"],reversed_plan["cases"])
            self.assertEqual(plan["distinct_workloads"],5)
            self.assertLessEqual(plan["distinct_workloads"],6)
            self.assertEqual(len({c["workload_id"] for c in plan["cases"]}),5)
            second=next(c for c in plan["cases"] if any(r["reason"]=="different_N_structural_change" for r in c["reasons"]))
            self.assertEqual(second["n"],768)  # different family outranks higher same-family score
            cross=plan["selection_evidence"]["crossover"]
            self.assertTrue(cross["same_N_different_B"])
            self.assertEqual({d["target"] for d in cross["directions"]},{A,B})
            self.assertTrue(all(d["screen_directional_ratio"]>=1.05 for d in cross["directions"]))
            self.assertTrue(all("default" in c["candidates"] and CONTROL in c["candidates"] for c in plan["cases"]))
            self.assertTrue(any(r["reason"]=="worst_confirmed_regression" for c in plan["cases"] for r in c["reasons"]))

    def test_completeness_and_frozen_R_gates(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            rows=synthetic_main(folder)
            write_rows(folder,rows[:-1])
            with self.assertRaisesRegex(ValueError,"confirmation"):
                headline.build_plan(folder)
            write_rows(folder,rows[1:])
            with self.assertRaisesRegex(ValueError,"screen"):
                headline.build_plan(folder)
            changed=copy.deepcopy(rows)
            changed[0]["result"]["graph_repeats"]+=1
            write_rows(folder,changed)
            with self.assertRaisesRegex(ValueError,"frozen R"):
                headline.build_plan(folder)

    def test_LUT_only_is_not_core_structure_change_and_commands_freeze_R(self):
        base=dict(structural_features=dict(numAxisUploads=1,LUT=0,kernels=[dict(stageRadix=[4,4],localSize=[64,1,1],axisBlock=[64,1,1,1])]))
        variant=copy.deepcopy(base)
        variant["structural_features"]["LUT"]=1
        self.assertEqual(headline.core_structure(base),headline.core_structure(variant))
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            synthetic_main(folder)
            plan=headline.build_plan(folder)
            output=folder/"plan"
            headline.write_plan(plan,output)
            script=(output/"run_checks.sh").read_text()
            self.assertIn("--rounds 10",script)
            self.assertIn("--warmup-ms 1000",script)
            self.assertIn(f"export CUDA_VISIBLE_DEVICES={GPU_UUID}",script)
            for case in plan["cases"]:
                self.assertIn(f"--graph-repeats {case['graph_repeats']}",script)
                self.assertNotEqual(case["seed"],20261027)
                self.assertTrue((output/"cases"/case["case_id"] / "workloads.json").exists())
            self.assertFalse((output/"measurements").exists())

    def test_both_directions_of_default_control_are_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            rows=synthetic_main(folder)
            for row in rows:
                if row["phase"]!="screen" or row["workload"]["n"]!=256:
                    continue
                if ((row["workload"]["batch"]==1 and row["candidate_id"]==CONTROL)
                        or (row["workload"]["batch"]==16 and row["candidate_id"]=="default")):
                    row["result"]["exec_ms_median"]=5.
                    row["result"]["exec_ms_samples"]=[5.]*3
            write_rows(folder,rows)
            plan=headline.build_plan(folder)
            matched=[c for c in plan["comparisons"] if {c["target"],c["reference"]}=={"default",CONTROL}]
            self.assertTrue(any(c["target"]=="default" for c in matched))
            self.assertTrue(all(c["role"]=="control_diagnostic" for c in matched))

    def test_main_gpu_consistency_and_malformed_records(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            rows=synthetic_main(folder)
            rows[0]["result"]["gpu"]["uuid"]="GPU-other-device"
            write_rows(folder,rows)
            with self.assertRaisesRegex(ValueError,"one GPU UUID"):
                headline.build_plan(folder)
            synthetic_main(folder)
            with (folder/"measurements.jsonl").open("a",encoding="utf-8") as f:
                f.write('{"interrupted":')
            with self.assertRaisesRegex(ValueError,"malformed completed JSONL"):
                headline.build_plan(folder)


def synthetic_followups(plan_root,plan,main_folder):
    original=headline.load_complete_main(main_folder)[1]
    for case in plan["cases"]:
        folder=plan_root/"measurements"/case["case_id"]
        folder.mkdir(parents=True)
        manifest=dict(rounds=10,warmup=5,warmup_ms=1000,repeats=101,timing_mode="graph",placement="outofplace",
                      graph_repeats_requested=case["graph_repeats"],seed=case["seed"],executable_sha256=plan["main_executable_sha256"],
                      cuda_visible_devices=GPU_UUID,source_commit=plan["source_commit"],
                      candidates=[(cid,dict(CANDIDATES)[cid]) for cid in case["candidates"]])
        (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        rows=[]
        for round_no in range(10):
            seed_text=f"{case['seed']}:control:{case['workload_id']}:{round_no}"
            input_seed=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4],"little")
            default_time=10*(1+.001*round_no)
            for cid in case["candidates"]:
                if cid in ("default",CONTROL):
                    milliseconds=default_time
                elif case["n"]==256 and case["batch"]==1:
                    milliseconds=default_time/(1.18+.002*(round_no-4.5) if cid==A else 1.1)
                elif case["n"]==256 and case["batch"]==16:
                    milliseconds=default_time/(1.20+.002*(round_no-4.5) if cid==B else 1.08)
                elif case["n"]==1280:
                    milliseconds=default_time/(.99+.001*(round_no-4.5))
                else:
                    milliseconds=default_time/(1.2+.002*(round_no-4.5))
                result=copy.deepcopy(original[case["workload_id"]]["rows"][cid]["result"])
                result.update(seed=input_seed,exec_ms_samples=[milliseconds]*101,exec_ms_median=milliseconds,
                              single_forward_correctness=dict(passed=True,criterion="criterion_v2"),
                              single_forward_input_unchanged=True,input_unchanged=True)
                rows.append(dict(phase="control",workload_id=case["workload_id"],workload=dict(n=case["n"],batch=case["batch"],graph_repeats=case["graph_repeats"]),
                                 candidate_id=cid,round=round_no,input_seed=input_seed,status="ok",result=result))
        write_rows(folder,rows)


class HeadlineAnalysisTests(unittest.TestCase):
    def test_zero_variance_is_noninferential_and_holm_keeps_family_burden(self):
        zero=headline_analysis.paired_statistics([1.1]*10)
        self.assertIsNone(zero["paired_log_t_95ci"])
        self.assertIsNone(zero["one_sided_p_vs_1p02"])
        arithmetic_only=headline_analysis.paired_statistics([1.1,math.nextafter(1.1,math.inf)]*5)
        self.assertEqual(arithmetic_only["inference_status"],"zero_variance_noninferential")
        self.assertIsNone(arithmetic_only["one_sided_p_vs_1p02"])
        rows=[dict(comparison_id="a",role="benefit",speedup_gmean=1.1,one_sided_p_vs_1p02=.01),
              dict(comparison_id="b",role="benefit",speedup_gmean=1.1,one_sided_p_vs_1p02=.04),
              dict(comparison_id="c",role="benefit",speedup_gmean=1.1,one_sided_p_vs_1p02=None),
              dict(comparison_id="ctrl",role="control_diagnostic",speedup_gmean=1.1,one_sided_p_vs_1p02=.001)]
        self.assertEqual(headline_analysis.holm_adjust(rows),3)
        self.assertAlmostEqual(rows[0]["holm_adjusted_p"],.03)
        self.assertAlmostEqual(rows[1]["holm_adjusted_p"],.08)
        self.assertTrue(rows[0]["clear_gain"])
        self.assertFalse(rows[1]["clear_gain"])
        self.assertFalse(rows[2]["clear_gain"])
        self.assertFalse(rows[3]["clear_gain"])

    def test_followup_duplicate_gpu_and_candidate_settings_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            synthetic_main(folder)
            plan=headline.build_plan(folder)
            plan_root=folder/"plan"
            headline.write_plan(plan,plan_root)
            synthetic_followups(plan_root,plan,folder)
            case_folder=plan_root/"measurements"/plan["cases"][0]["case_id"]
            records=headline.read_strict_records(case_folder/"measurements.jsonl")
            write_rows(case_folder,records+[copy.deepcopy(records[0])])
            with self.assertRaisesRegex(ValueError,"exactly ten complete rounds"):
                headline_analysis.analyze(plan_root)
            write_rows(case_folder,records)
            changed=copy.deepcopy(records)
            changed[0]["result"]["gpu"]["uuid"]="GPU-other-device"
            write_rows(case_folder,changed)
            with self.assertRaisesRegex(ValueError,"invalid or mixed"):
                headline_analysis.analyze(plan_root)
            changed=copy.deepcopy(records)
            changed[0]["input_seed"]+=1
            write_rows(case_folder,changed)
            with self.assertRaisesRegex(ValueError,"input seed"):
                headline_analysis.analyze(plan_root)
            write_rows(case_folder,records)
            manifest_path=case_folder/"manifest.json"
            manifest=json.loads(manifest_path.read_text())
            manifest["candidates"][0][1]["aim_threads"]=512
            manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"candidate settings"):
                headline_analysis.analyze(plan_root)

    def test_independent_append_analysis_and_complete_round_gate(self):
        with tempfile.TemporaryDirectory(prefix="headline_synthetic_",dir=Path(__file__).parent) as temp:
            folder=Path(temp)
            synthetic_main(folder)
            plan=headline.build_plan(folder)
            plan_root=folder/"plan"
            headline.write_plan(plan,plan_root)
            synthetic_followups(plan_root,plan,folder)
            report,ratios,controls=headline_analysis.analyze(plan_root)
            self.assertEqual(report["cases"],5)
            self.assertTrue(report["crossover"]["both_directions_clear_gain"])
            self.assertEqual(len(controls),50)
            self.assertEqual(len(ratios),10*len(plan["comparisons"]))
            self.assertTrue(all(c["inference_status"]=="zero_variance_noninferential" for c in report["negative_controls"]))
            headline_analysis.write_output(report,ratios,controls,plan_root/"analysis")
            self.assertTrue((plan_root/"analysis"/"all_round_ratios.csv").exists())
            last_case=plan["cases"][-1]["case_id"]
            measurements=plan_root/"measurements"/last_case/"measurements.jsonl"
            raw=measurements.read_text().splitlines()
            measurements.write_text("\n".join(raw[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"exactly ten complete rounds"):
                headline_analysis.analyze(plan_root)


if __name__=="__main__":
    unittest.main()
