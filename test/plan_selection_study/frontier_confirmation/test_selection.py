import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('frontier_confirmation_phase',ROOT/'tools'/'frontier_confirmation_phase.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
plan_spec=importlib.util.spec_from_file_location('frontier_confirmation_make_plan',Path(__file__).with_name('make_plan.py'))
plan_module=importlib.util.module_from_spec(plan_spec);plan_spec.loader.exec_module(plan_module)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.default={'radix':[8,16], 'logical_registers':16,'grouped_batch':1}
        self.winner={'radix':[16,8], 'logical_registers':16,'grouped_batch':1}
        self.control={'radix':[4,4,8], 'logical_registers':8,'grouped_batch':1}
        self.rows=[{'record_type':'workload_status','n':128,'batch':16,'status':'identity_passed','request':self.default},
                   {'record_type':'candidate_disposition','n':128,'batch':16,'status':'screen_complete',
                    'candidate_id':'a'*64,'request':self.winner,'median_ratio':.9},
                   {'record_type':'candidate_disposition','n':128,'batch':16,'status':'screen_complete',
                    'candidate_id':'b'*64,'request':self.control,'median_ratio':1.2},
                   {'record_type':'candidate_disposition','n':128,'batch':16,'status':'screen_complete',
                    'candidate_id':'c'*64,'request':self.default,'median_ratio':1.1}]
    def test_explicit_negative_control_keeps_separate_role(self):
        cases=driver.select_cases(self.rows,['128:16:'+'b'*64])
        self.assertEqual([c['role'] for c in cases],['winner','negative_control'])
        self.assertEqual(cases[1]['request'],self.control)
        self.assertEqual([c['role'] for c in driver.select_cases(self.rows,[])],['winner'])

    def test_rejects_identity_unknown_duplicate_and_wrong_direction(self):
        for item in ['128:16:'+'c'*64,'128:16:'+'d'*64,'128:16:'+'a'*64,'bad']:
            with self.subTest(item=item),self.assertRaises(ValueError):driver.select_cases(self.rows,[item])
        with self.assertRaises(ValueError):driver.select_cases(self.rows,['128:16:'+'b'*64]*2)

    def test_controls_can_be_run_without_winners(self):
        rows=[row for row in self.rows if row.get('candidate_id')!='a'*64]
        self.assertEqual(driver.select_cases(rows,['128:16:'+'b'*64])[0]['role'],'negative_control')

    def test_frozen_plan_separates_controls_from_winner_summary(self):
        selected=driver.select_cases(self.rows,['128:16:'+'b'*64])
        proof={'screen_contract':{'execution_commit':'a'*40,'binaries':{'original':'a','expanded':'b'}},
               'cases':selected}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            paths={name:Path(tmp)/name for name in ('rows','guard','incidents','budget')}
            for path in paths.values():path.write_text(json.dumps({}),encoding='utf8')
            with mock.patch.object(plan_module.evidence,'audit_screen',return_value=proof):
                plan=plan_module.build({'gpu':'GPU-test','cases':selected},paths,{})
        self.assertEqual([c['role'] for c in plan['screen_winners']],['winner'])
        self.assertEqual([c['role'] for c in plan['screen_negative_controls']],['negative_control'])


if __name__=='__main__':unittest.main()
