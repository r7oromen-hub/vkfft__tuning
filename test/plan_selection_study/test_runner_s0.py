import argparse
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import run_suite
from study_common import candidate_config,input_seed,load_design,read_jsonl


ROOT=Path(__file__).parent
DESIGN=ROOT/"research_design.json"


def synthetic_result(w,cid,config,seed):
    options=["--gpu-architecture=sm_89"]
    spans=[dict(id=0,parent_id=-1,stage="initialize",start_ns=0,end_ns=6_000_000,inclusive_ms=6.,exclusive_ms=4.),
           dict(id=1,parent_id=0,stage="nvrtc_compile",start_ns=1_000_000,end_ns=2_000_000,inclusive_ms=1.,exclusive_ms=1.),
           dict(id=2,parent_id=0,stage="module_load",start_ns=3_000_000,end_ns=4_000_000,inclusive_ms=1.,exclusive_ms=1.)]
    names=dict(aim_threads="aimThreads",register_boost="registerBoost",use_lut="useLUT",register_boost_nonpow2="registerBoostNonPow2",
               coalesced_memory="coalescedMemory",use_lut_4step="useLUT_4step",swap_to_2stage="swapTo2Stage4Step",swap_to_3stage="swapTo3Stage4Step")
    request=dict(makeForwardPlanOnly=1,**{names[k]:v for k,v in config.items()})
    numerical=dict(passed=True,criterion="criterion_v2")
    return dict(status="ok",n=w["n"],batch=w["batch"],seed=seed,requested_config=request,header_instrumentation=True,
        timing_mode="graph",placement="outofplace",graph_repeats=w["graph_repeats"],
        exec_ms_samples=[.6]*w["samples"],exec_ms_median=.6,raw_interval_ms_median=.6*w["graph_repeats"],
        single_forward_correctness=numerical,correctness=numerical,single_forward_input_unchanged=True,input_unchanged=True,
        descriptor_validation=dict(all_match=True),structural_features=dict(numAxisUploads=1,axisSplit=[w["n"]]),
        graph_kernel_count=w["graph_repeats"],graph_event_record_count=2,graph_node_count=w["graph_repeats"]+2,
        gpu=dict(uuid=load_design(DESIGN)["scope"]["device_uuid"]),
        init_trace=dict(spans=spans,stage_totals={s["stage"]:dict(calls=1,inclusive_ms=s["inclusive_ms"],exclusive_ms=s["exclusive_ms"]) for s in spans},
                        counters=dict(nvrtc_compile=1,module_load=1,driver_kernel_launch=0)),
        precompile_kernels=[dict(axis=0,upload=0,raw_descriptor={},canonical_descriptor=dict(numAxisUploads=1,fftDim=w["n"],axisSplit=[w["n"]],
                                localSize=[64,1,1],axisBlock=[64,1,1,w["n"]],registerBoost=1,stageRadix=[4,4]),
                                source_sha256="a"*64,compiler_input=dict(options=options,original_compile_header_sha256="c"*64),actual_compiler_options=options,compiler_options_match=True)])


class RunnerS0Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="s0_synthetic_",dir=ROOT)
        self.folder=Path(self.temp.name)
        self.exe=self.folder/"synthetic_executable"
        self.exe.write_bytes(b"synthetic test fixture, never executed")
        self.env=dict(VKFFT_STUDY_GUARD_ID="synthetic-guard",VKFFT_STUDY_GUARD_UUID=load_design(DESIGN)["scope"]["device_uuid"],
                      VKFFT_STUDY_GUARD_LOG=str(self.folder/"guard.jsonl"))

    def tearDown(self):
        self.temp.cleanup()

    def args(self,stage):
        return argparse.Namespace(design=DESIGN,exe=str(self.exe),output=self.folder/"study",stage=stage,timeout=180,
                                  import_calibration_from=None,plain_exe=None)

    def mock_run(self,args,w,candidate,phase,round_no,order):
        cid,config=candidate
        result=synthetic_result(w,cid,config,input_seed(args.design_data,w.get("seed_phase",phase),w["id"],round_no))
        return dict(phase=phase,workload_id=w["id"],workload=w,candidate_id=cid,round=round_no,order=order,status="ok",result=result,
                    guard_id="synthetic-guard",guard_gpu_uuid=self.env["VKFFT_STUDY_GUARD_UUID"],guard_log=self.env["VKFFT_STUDY_GUARD_LOG"],
                    guard_run_id=f"{phase}:{w['id']}:{cid}:{round_no}",guard_clean=True)

    def test_health_and_resume_do_not_duplicate_complete_blocks(self):
        args=self.args("health")
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=self.mock_run) as mocked,
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            study.health()
            gate_bytes=(args.output/"health"/"gate.json").read_bytes()
            resumed=run_suite.Study(args)
            resumed.health()
            self.assertEqual((args.output/"health"/"gate.json").read_bytes(),gate_bytes)
        self.assertEqual(mocked.call_count,33) # 3 R1 calibration probes + 30 health processes
        rows=read_jsonl(args.output/"health"/"measurements.jsonl")
        self.assertEqual(len(rows),33)
        self.assertTrue(json.loads((args.output/"health"/"gate.json").read_text())["passed"])

    def test_semantics_is_exact_frozen64_probes(self):
        args=self.args("semantics")
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=self.mock_run),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            study.semantics()
        rows=read_jsonl(args.output/"semantics"/"measurements.jsonl")
        self.assertEqual(len(rows),64)
        self.assertTrue(all(r["workload"]["samples"]==1 and r["workload"]["warmup_ms"]==0 and r["workload"]["graph_repeats"]==16 for r in rows))

    def test_missing_guard_and_modified_design_are_rejected(self):
        with patch.dict(os.environ,{k:"" for k in self.env}):
            with self.assertRaisesRegex(run_suite.StudyStop,"GPU_guard_required"):
                run_suite.Study(self.args("health"))
        changed=self.folder/"research_design.json"
        changed.write_bytes(DESIGN.read_bytes()+b" ")
        (self.folder/"research_freeze.json").write_bytes((ROOT/"research_freeze.json").read_bytes())
        with self.assertRaisesRegex(ValueError,"preregistered freeze"):
            load_design(changed)

    def test_A1_switches_guard_expected_executable_and_has_eight_processes(self):
        args=self.args("equivalence")
        plain=self.folder/"plain_executable"
        plain.write_bytes(b"synthetic plain control, never executed")
        args.plain_exe=str(plain)
        seen=[]
        def mocked(args,w,candidate,phase,round_no,order):
            seen.append((w["binary_role"],args.exe))
            row=self.mock_run(args,w,candidate,phase,round_no,order)
            if w["binary_role"]=="plain":
                result=row["result"]
                result["postinit_source_kernels"]=[dict(axis=k["axis"],upload=k["upload"],canonical_descriptor=k["canonical_descriptor"],
                    source_sha256=k["source_sha256"],expected_compiler_input=k["compiler_input"],actual_compiler_options=None,snapshot_stage="post_initialize_retained_source")
                    for k in result["precompile_kernels"]]
                result["precompile_kernels"]=[]
                result["header_instrumentation"]=False
            return row
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=mocked),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            study.equivalence()
        self.assertEqual(len(seen),8)
        self.assertTrue(all(Path(exe)==(plain if role=="plain" else self.exe) for role,exe in seen))
        self.assertTrue(json.loads((args.output/"equivalence"/"gate.json").read_text())["passed"])

    def test_cost_smoke_is_four_context_pipelines_with_shared_candidate_order(self):
        args=self.args("cost_smoke")
        def mocked(args,w,candidate,phase,round_no,order):
            cid,config=candidate
            table=Path(w["candidate_file"]).read_text().splitlines()
            ids=[line.split("\t")[0] for line in table[1:]]
            mode=w["study_mode"]
            applications=[]
            for index,request_id in enumerate(ids+(["default"] if mode=="describe-select-cost" else [])):
                role="describe" if mode=="describe-select-cost" and index<len(ids) else ("full" if mode=="full-search-cost" else "materialize_selected")
                result=synthetic_result(w,request_id,{},0)
                result["shared_context_verified"]=True
                if role=="describe":
                    span=dict(id=0,parent_id=-1,stage="initialize",start_ns=0,end_ns=6_000_000,inclusive_ms=6.,exclusive_ms=6.)
                    result["init_trace"]=dict(spans=[span],stage_totals=dict(initialize=dict(calls=1,inclusive_ms=6.,exclusive_ms=6.)),
                                              counters=dict(nvrtc_compile=0,module_load=0,driver_kernel_launch=0))
                applications.append(dict(candidate_id=request_id,round=0,pipeline=mode,role=role,sequence_index=index,result=result))
            row=self.mock_run(args,w,candidate,phase,round_no,order)
            row["result"]=dict(record_type="context_profile_summary",status="ok",mode=mode,n=w["n"],batch=w["batch"],
                                context_id=mode+":"+w["id"],candidate_order_ids=ids,selected_id="default",init_rounds=1,apps=applications)
            return row
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=mocked) as calls,
              patch.object(run_suite.Study,"require",return_value=None),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            outcome=study.cost_smoke()
        self.assertEqual(calls.call_count,4)
        self.assertTrue(outcome["stage_checks_passed"])

    def test_screen_confirmation_headline_freezes_and_fresh_rounds(self):
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=self.mock_run),
              patch.object(run_suite.Study,"require",return_value=None),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            screen=run_suite.Study(self.args("screen"))
            (screen.output/"semantics").mkdir()
            (screen.output/"semantics"/"gate.json").write_text(json.dumps(dict(passed=True)))
            screen.design=deepcopy(screen.design)
            screen.design["workloads"]=screen.design["workloads"][:2]
            screen.design["candidates"]=screen.design["candidates"][:4]
            screen.workloads={w["id"]:w for w in screen.design["workloads"]}
            screen.candidates={c["id"]:c for c in screen.design["candidates"]}
            screen.screen()
            selected=json.loads((screen.output/"screen_selection.json").read_text())
            self.assertEqual(selected["global_candidate"],"default")
            self.assertTrue(all(s["winner"]=="default" for s in selected["workloads"].values()))
            confirmation=run_suite.Study(self.args("confirmation"))
            confirmation.design=screen.design
            confirmation.workloads=screen.workloads
            confirmation.candidates=screen.candidates
            confirmation.confirmation()
            self.assertEqual(len(confirmation.rows),12)
            headline=run_suite.Study(self.args("headline"))
            headline.design=screen.design
            headline.workloads=screen.workloads
            headline.candidates=screen.candidates
            headline.headline()
            self.assertEqual(len(headline.rows),20) # deduplicated null/tie slots, no extra winners
            self.assertEqual({r["round"] for r in headline.rows},set(range(10)))
            self.assertNotEqual(confirmation.rows[0]["result"]["seed"],headline.rows[0]["result"]["seed"])

    def test_partial_screen_is_never_prefix_spliced_and_needs_new_health(self):
        args=self.args("screen")
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=self.mock_run),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            w=next(iter(study.workloads.values()))
            row=study.attempt(dict(w,graph_repeats=1,samples=31,block_id="partial"),"default","screen",0)
            with self.assertRaisesRegex(run_suite.StudyStop,"explicit_retry"):
                study.block(w,"screen",0,["default","default_replica"],1,31)
            args.retry_block=f"screen/{w['id']}/r0"
            health=study.output/"health"/"gate.json"
            health.parent.mkdir()
            health.write_text(json.dumps(dict(passed=True,created_utc="2000-01-01T00:00:00+00:00")))
            # mock rows omit process timestamps, add a known timestamp to the retained partial row.
            row["ended_utc"]="2026-10-28T00:00:00+00:00"
            with self.assertRaisesRegex(run_suite.StudyStop,"fresh_health_required"):
                study.block(w,"screen",0,["default","default_replica"],1,31)
            health.write_text(json.dumps(dict(passed=True,created_utc="2026-10-28T00:10:00+00:00")))
            new,_=study.block(w,"screen",0,["default","default_replica"],1,31)
            self.assertEqual(len(study.rows),3)
            self.assertTrue(all(r["workload"]["block_attempt"]==1 for r in new))

    def test_candidate_diagnostic_preserves_four_probes_without_scientific_gate(self):
        args=self.args("candidate_diagnostic")
        plain=self.folder/"plain_executable"
        plain.write_bytes(b"synthetic plain diagnostic")
        args.plain_exe=str(plain)
        seen=[]
        def fault(args,w,candidate,phase,round_no,order):
            seen.append((w,args.exe,candidate[0]))
            row=self.mock_run(args,w,candidate,phase,round_no,order)
            if candidate[0]!="default":
                row.update(status="failed",returncode=1,stdout="full raw diagnostic stdout",stderr="full stderr")
                row["result"].update(status="error",failure_phase="single_forward_validation",error="CUDA illegal memory access")
            return row
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=fault),
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(args)
            study.candidate_diagnostic()
        self.assertEqual(len(study.rows),4)
        self.assertEqual([cid for _,_,cid in seen],["default","default","a128_r4_l1_np1_s2_8192","a128_r4_l1_np1_s2_8192"])
        self.assertEqual({Path(exe) for _,exe,_ in seen},{self.exe,plain})
        self.assertTrue(all(w["input_seed_override"]==2235648248 and w["graph_repeats"]==16 and w["samples"]==1 and w["pattern"]=="zero" for w,_,_ in seen))
        self.assertFalse((study.stage_path/"gate.json").exists())
        self.assertEqual([r["status"] for r in study.rows],["ok","ok","failed","failed"])
        self.assertTrue(all(r["excluded_from_selection_and_performance"] for r in study.rows))

    def test_run_one_diagnostic_uses_exact_seed_and_keeps_fault_output(self):
        from study_common import sha
        args=self.args("candidate_diagnostic")
        args.design_data=load_design(DESIGN)
        args.expected_executable_sha256=sha(self.exe.read_bytes())
        w=dict(id="n16384_b1",n=16384,batch=1,graph_repeats=16,samples=1,warmup=0,warmup_ms=0,
               pattern="zero",input_seed_override=2235648248,diagnostic_capture=True)
        fault=dict(status="error",failure_phase="single_forward_validation",error="CUDA illegal memory access",
                   gpu=dict(uuid=args.design_data["scope"]["device_uuid"]))
        raw=json.dumps(fault)+"\n"
        with patch.object(run_suite.subprocess,"run",return_value=subprocess.CompletedProcess([],1,raw,"diagnostic stderr")) as invoke:
            row=run_suite.run_one(args,w,("a128_r4_l1_np1_s2_8192",dict(aim_threads=128,register_boost=4,use_lut=1,
                register_boost_nonpow2=1,swap_to_2stage=8192)),"candidate_diagnostic/instrumented",0,0)
        command=invoke.call_args.args[0]
        self.assertEqual(command[command.index("--seed")+1],"2235648248")
        self.assertEqual(row["input_seed"],2235648248)
        self.assertEqual(row["stdout"],raw)
        self.assertEqual(row["stderr"],"diagnostic stderr")
        self.assertEqual(row["result"]["error"],"CUDA illegal memory access")

    def test_A5_resume_preserves_original19_and_only_appends_remaining45(self):
        args=self.args("semantics")
        bad="a128_r4_l1_np1_s2_8192"
        def mocked(args,w,candidate,phase,round_no,order):
            row=self.mock_run(args,w,candidate,phase,round_no,order)
            if w["id"]=="n16384_b1" and candidate[0]==bad and w["pattern"]=="zero":
                row.update(status="failed",validation_errors=["harness_status:error","invalid_samples"])
                row["result"].update(status="error",failure_phase="single_forward_validation",
                    error="synchronize one-forward validation: an illegal memory access was encountered")
                row["result"].pop("single_forward_input_unchanged",None)
            return row
        with (patch.dict(os.environ,self.env),patch.object(run_suite,"run_one",side_effect=mocked) as calls,
              patch.object(run_suite,"audit_row",return_value=dict(ok=True,errors=[])),contextlib.redirect_stdout(io.StringIO())):
            first=run_suite.Study(args)
            with self.assertRaisesRegex(run_suite.StudyStop,"fatal_execution_protocol"):
                first.semantics()
            self.assertEqual(calls.call_count,19)
            before=first.measurements.read_bytes()
            resumed=run_suite.Study(args)
            resumed.candidate_fault_admission=dict(branch="matching_failure")
            (resumed.output/"candidate_fault_admission.json").write_text(json.dumps(resumed.candidate_fault_admission))
            resumed.semantics()
            self.assertEqual(calls.call_count,64)
            self.assertTrue(resumed.measurements.read_bytes().startswith(before))
            gate=json.loads((resumed.stage_path/"gate.json").read_text())
            self.assertTrue(gate["passed"] and gate["protocol_valid"])
            self.assertFalse(gate["all_candidates_passed"])
            self.assertEqual(len(gate["candidate_validation_failures"]),1)
            self.assertEqual(gate["semantic_invalid_cells"][0]["candidate_id"],bad)


if __name__=="__main__":
    unittest.main()
