import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from learning_check import feature_sets,prepare_rows,run_learning,train_global,MODEL
from study_common import load_design


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.design=load_design(Path(__file__).with_name("research_design.json"))

    def rows(self):
        rows=[]
        for w in self.design["workloads"]:
            for c in self.design["candidates"]:
                latency=.5 if c["id"]=="default" else .7
                rows.append(dict(study_id=self.design["study_id"],phase="screen",round=0,workload_id=w["id"],
                    candidate_id=c["id"],block_attempt=0,status="ok",guard_clean=True,guard_gpu_uuid=self.design["scope"]["device_uuid"],record_sha256=w["id"]+c["id"],
                    result=dict(exec_ms_median=latency,graph_repeats=999,physical_registers=999,
                        precompile_kernels=[dict(axis=0,upload=0,source_sha256="timing_secret",canonical_descriptor=dict(fftDim=w["n"],
                            numAxisUploads=1,localSize=[64,1,1],logical_registers_per_thread=8,physical_registers=999,
                            exec_ms_median=1234,source_sha256="secret",validity=1))])))
        return rows

    def test_feature_allowlist_has_no_label_or_postcompile_leakage(self):
        row=self.rows()[0]
        original=feature_sets(self.design,row)
        changed=copy.deepcopy(row)
        changed["result"]["exec_ms_median"]=9e5
        changed["result"]["physical_registers"]=7
        changed["result"]["graph_repeats"]=1
        kernel=changed["result"]["precompile_kernels"][0]
        kernel["source_sha256"]="another"
        for field in ("physical_registers","exec_ms_median","source_sha256","validity"): kernel["canonical_descriptor"][field]=0
        self.assertEqual(feature_sets(self.design,changed),original)
        self.assertTrue(any("logical_registers_per_thread" in f for f in original[1]))
        self.assertFalse(any("physical_registers" in f or "source_sha256" in f for f in original[1]))

    def test_complete_blocks_gate_no_pilot_and_no_imputation(self):
        rows=self.rows()
        rows[2]["status"]="failed"
        prepared,coverage,_=prepare_rows(self.design,rows)
        self.assertEqual(len(coverage),36)
        self.assertNotIn(rows[2]["candidate_id"],coverage[0]["legal_candidates"])
        self.assertFalse(any(r["record_sha256"]==rows[2]["record_sha256"] for r in prepared))
        without_one=rows[1:]
        _,coverage,excluded=prepare_rows(self.design,without_one)
        self.assertEqual(len(coverage),34) # both batches of incomplete exact-N group are excluded
        self.assertEqual(len(excluded),2)
        pilot=copy.deepcopy(rows)
        for r in pilot: r["study_id"]="old_pilot"
        self.assertEqual(run_learning(self.design,pilot,evidence_verified=True)["status"],"not_run_coverage_gate")
        self.assertEqual(run_learning(self.design,rows)["status"],"not_run_unverified_evidence")

    def test_training_only_global_never_uses_heldout_labels(self):
        train=[dict(workload_id="train1",candidate_id="default",time=2.,default_time=2.),
               dict(workload_id="train1",candidate_id="x",time=1.,default_time=2.),
               dict(workload_id="train2",candidate_id="default",time=3.,default_time=3.),
               dict(workload_id="train2",candidate_id="x",time=1.,default_time=3.)]
        self.assertEqual(train_global(train)[0],"x")
        self.assertEqual(train_global(train[:-1])[0],"default") # candidate absent on one train workload is ineligible

    def test_exact_semantic_invalid_success_is_excluded_from_legal_targets(self):
        rows=self.rows()
        forbidden=rows[2]
        forbidden["selection_eligible"]=False
        forbidden["result"]["exec_ms_median"]=.000001
        records,coverage,excluded=prepare_rows(self.design,rows)
        self.assertEqual(len(coverage),36)
        self.assertFalse(excluded)
        self.assertFalse(any(r["record_sha256"]==forbidden["record_sha256"] for r in records))
        self.assertIn(forbidden["candidate_id"],coverage[0]["excluded_candidates"])

    def test_twelve_complete_N_with_empty_fixed_fold_does_not_reassign_or_predict_empty(self):
        import numpy as np
        lengths=sorted({w["n"] for w in self.design["workloads"]})
        removed={n for i,n in enumerate(lengths) if i%4==0}
        rows=[r for r in self.rows() if next(w for w in self.design["workloads"] if w["id"]==r["workload_id"])["n"] not in removed]
        class Model:
            def __init__(self,**params): pass
            def fit(self,x,y): return self
            def predict(self,x):
                if not len(x): raise AssertionError("must not predict an empty held-out fold")
                return np.zeros(len(x))
        with patch("sklearn.ensemble.ExtraTreesRegressor",Model):
            result=run_learning(self.design,rows,evidence_verified=True)
        self.assertEqual(result["distinct_N"],12)
        self.assertEqual(result["folds"],{n:i%4 for i,n in enumerate(lengths)})
        self.assertTrue(all(o["fold_details"][0]["status"]=="empty_heldout_fold_skipped_fixed_assignment" for o in result["outcomes"]))

    def test_all_six_ablations_share_fixed_group_folds_and_capacity(self):
        # Mock only fitting cost; test actual grouped-data assembly, 24 fits and metrics.
        import numpy as np
        fits=[]
        class Model:
            def __init__(self,**params):
                self.params=params
                fits.append(self)
            def fit(self,x,y): self.width=x.shape[1];return self
            def predict(self,x): return np.zeros(x.shape[0])
        with patch("sklearn.ensemble.ExtraTreesRegressor",Model):
            result=run_learning(self.design,self.rows(),evidence_verified=True)
        self.assertEqual(result["status"],"completed_auxiliary")
        self.assertEqual(len(fits),24)
        self.assertTrue(all(m.params==MODEL for m in fits))
        self.assertEqual(len(result["outcomes"]),6)
        canonical_folds=[(f["train_N"],f["test_N"]) for f in result["outcomes"][0]["fold_details"]]
        for o in result["outcomes"]:
            self.assertEqual([(f["train_N"],f["test_N"]) for f in o["fold_details"]],canonical_folds)
            for f in o["fold_details"]: self.assertFalse(set(f["train_N"])&set(f["test_N"]))
            self.assertEqual(len(o["workloads"]),36)
            self.assertTrue(all(r["selected_candidate"]=="default" for r in o["workloads"]))
        self.assertEqual(result["baselines"]["default"]["metrics"]["equal_workload_geometric_speedup"],1.)


if __name__=="__main__": unittest.main()
