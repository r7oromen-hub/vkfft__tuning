import math
import unittest
from copy import deepcopy
from pathlib import Path
from selection_analysis import freeze_screen,freeze_headline
from study_common import load_design
from study_report import headline_test,holm
from test_runner_s0 import synthetic_result


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.design=load_design(Path(__file__).with_name("research_design.json"))

    def blocks(self):
        blocks={}
        for w in self.design["workloads"]:
            rows=[]
            for c in self.design["candidates"]:
                r=synthetic_result(dict(w,graph_repeats=1,samples=31),c["id"],{},1)
                rows.append(dict(candidate_id=c["id"],status="ok",result=r,record_sha256=c["id"]+w["id"]))
            blocks[w["id"]]=rows
        return blocks

    def test_replica_excluded_ties_default_and_core_distinct_alternative(self):
        blocks=self.blocks()
        for rows in blocks.values():
            rows[1]["result"]["exec_ms_median"]=.01 # negative control may never win
            rows[2]["result"]["exec_ms_median"]=.5
            rows[2]["result"]["precompile_kernels"][0]["source_sha256"]="b"*64 # LUT-only source change
            rows[3]["result"]["precompile_kernels"][0]["canonical_descriptor"]["localSize"]=[128,1,1]
        frozen=freeze_screen(self.design,blocks)
        self.assertEqual(frozen["global_candidate"],self.design["candidates"][2]["id"])
        for s in frozen["workloads"].values():
            self.assertEqual(s["winner"],self.design["candidates"][2]["id"])
            self.assertEqual(s["alternative"],self.design["candidates"][3]["id"])
        for rows in blocks.values():
            for r in rows: r["result"]["exec_ms_median"]=.6
        self.assertTrue(all(s["winner"]=="default" for s in freeze_screen(self.design,blocks)["workloads"].values()))

    def test_crossover_includes_third_winner_and_all_predeclared_tests(self):
        blocks=self.blocks()
        selection=freeze_screen(self.design,blocks)
        a,b="n16384_b1","n16384_b2048"
        target="a128_r4_l1_np1"
        winner="a64_r1_lm1"
        for wid in (a,b):
            selection["workloads"][wid]["winner"]=winner
            selection["workloads"][wid]["screen_ms"][target]=.3 if wid==a else 1.2
        summary={wid:dict(primary=dict(speedup=1.),comparisons={"default":dict(speedup=1.)}) for wid in blocks}
        for wid in (a,b):
            summary[wid]["primary"]["speedup"]=1.1
            summary[wid]["comparisons"][target]=dict(speedup=2. if wid==a else .5)
        frozen=freeze_headline(self.design,selection,summary,blocks)
        self.assertFalse(frozen["missing_crossover"])
        for wid in (a,b):
            entry=next(x for x in frozen["workloads"] if x["workload_id"]==wid)
            self.assertTrue({winner,target,"default","default_replica"}<=set(entry["candidates"]))
            self.assertTrue(any(t["workload_id"]==wid and t["target"]==winner and t["reference"]=="default" for t in frozen["directional_tests"]))

    def test_zero_variance_is_noninferential_and_counts_in_holm(self):
        x=headline_test([1.2]*10)
        self.assertEqual(x["p"],1.)
        self.assertIsNone(x["interval95"])
        tests=holm([dict(p=.02,speedup=1.1,inferential=True),x,dict(p=.03,speedup=1.1,inferential=True)])
        self.assertAlmostEqual(tests[0]["holm_p"],.06)
        self.assertFalse(any(t["clear_gain"] for t in tests))


if __name__=="__main__": unittest.main()
