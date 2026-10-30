import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('frontier_phase_test',ROOT/'tools/frontier_search_phase.py')
phase=importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


class PhaseBudgetTests(unittest.TestCase):
    def test_manifest_time_is_charged_and_global_reserve_preserved(self):
        self.assertEqual(phase.available_seconds(100,1000,20000,400),700)
        self.assertEqual(phase.available_seconds(100,1000,8000,400),400)
        self.assertLess(phase.available_seconds(100,1000,7500,400),0)

    def test_historical_anchor_precedes_neighbors_and_other_candidates(self):
        rows=[{'selection':'selected','selection_index':i,'selection_reasons':labels}
              for i,labels in enumerate([['identity_neighbor'],['stratified_cost_balance'],['historical'],['identity']])]
        phase.prioritize_anchors(rows)
        self.assertEqual([r['selection_index'] for r in rows],[2,3,0,1])


if __name__=='__main__':unittest.main()
