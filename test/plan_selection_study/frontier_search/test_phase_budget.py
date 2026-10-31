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

    def test_default_reserve_is_backward_compatible(self):
        for args in [(100,1000,20000,400),(100,1000,8000,400),(100,1000,7500,400)]:
            with self.subTest(args=args):
                self.assertEqual(phase.available_seconds(*args),
                                 phase.available_seconds(*args,reserve_seconds=7200))

    def test_zero_or_small_reserve_does_not_bypass_global_deadline(self):
        # Phase budget alone would permit 9700 seconds; the global limit wins.
        self.assertEqual(phase.available_seconds(100,10000,800,400,reserve_seconds=0),400)
        self.assertEqual(phase.available_seconds(100,10000,800,400,reserve_seconds=100),300)
        self.assertLess(phase.available_seconds(100,10000,800,900,reserve_seconds=0),0)

    def test_initial_budget_obeys_both_phase_limit_and_global_deadline(self):
        self.assertEqual(phase.initial_phase_seconds('exhaust128', 30000, 100), 7200)
        self.assertEqual(phase.initial_phase_seconds('breadth', 30000, 100), 10800)
        self.assertEqual(phase.initial_phase_seconds('exhaust128', 1100, 100, 0), 1000)
        self.assertEqual(phase.initial_phase_seconds('breadth', 1100, 100, 250), 750)
        self.assertEqual(phase.initial_phase_seconds('breadth', 100, 101, 0), -1)

    def test_manifest_time_is_charged_with_custom_reserve(self):
        self.assertEqual(phase.available_seconds(100,1000,5000,400,reserve_seconds=0),700)
        self.assertEqual(phase.available_seconds(100,1000,1000,400,reserve_seconds=100),500)
        self.assertEqual(phase.available_seconds(100,1000,5000,550,reserve_seconds=20),550)

    def test_reserve_parser_and_direct_call_reject_invalid_values(self):
        self.assertEqual(phase.parse_reserve('0'),0)
        self.assertEqual(phase.parse_reserve('12.5'),12.5)
        self.assertEqual(phase.parse_reserve(7200),7200)
        for bad in ('-1','nan','inf','-inf',-1,float('nan'),float('inf')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phase.parse_reserve(bad)
                with self.assertRaises(ValueError):
                    phase.available_seconds(100,1000,5000,400,reserve_seconds=bad)
                with self.assertRaises(ValueError):
                    phase.initial_phase_seconds('exhaust128',5000,400,reserve_seconds=bad)

    def test_historical_anchor_precedes_neighbors_and_other_candidates(self):
        rows=[{'selection':'selected','selection_index':i,'selection_reasons':labels}
              for i,labels in enumerate([['identity_neighbor'],['stratified_cost_balance'],['historical'],['identity']])]
        phase.prioritize_anchors(rows)
        self.assertEqual([r['selection_index'] for r in rows],[2,3,0,1])


if __name__=='__main__':unittest.main()
