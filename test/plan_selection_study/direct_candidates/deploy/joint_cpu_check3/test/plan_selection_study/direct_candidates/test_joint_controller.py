import copy, sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent / 'controller'))
import joint_controller as c
class NumericInvalidTests(unittest.TestCase):
    def test_only_clean_reference_failure_is_classified(self):
        w={'n':4,'batch':2,'seed':42}; a=SimpleNamespace(gpu='GPU-test')
        r={'gpu':{'uuid':'GPU-test'},'requested_config':{'makeForwardPlanOnly':1},'n':4,'batch':2,'seed':42,
           'single_forward_input_unchanged':True,'validation_output_initialization':'0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
           'descriptor_validation':{'all_match':True},'direct_resource_validation':{'threads':8},'precompile_kernels':[{}],
           'status':'error','failure_phase':'single_forward_validation','error':'Single out-of-place forward failed immutable-input or numerical validation',
           'single_forward_correctness':{'passed':False,'checked_elements':8}}
        row={'guard_clean':True,'result':r}
        self.assertTrue(c.numeric_invalid(row,a,w))
        for field,value in [('single_forward_input_unchanged',False),('failure_phase','execution')]:
            bad=copy.deepcopy(row);bad['result'][field]=value;self.assertFalse(c.numeric_invalid(bad,a,w))
        bad=copy.deepcopy(row);bad['guard_clean']=False;self.assertFalse(c.numeric_invalid(bad,a,w))
        bad=copy.deepcopy(row);bad['result']['single_forward_correctness']['checked_elements']=7;self.assertFalse(c.numeric_invalid(bad,a,w))
        final=copy.deepcopy(row);final['result'].update(status='correctness_failed',input_unchanged=True,
            single_forward_correctness={'passed':True,'checked_elements':8},correctness={'passed':False,'checked_elements':8})
        self.assertTrue(c.numeric_invalid(final,a,w))
        final['result']['input_unchanged']=False;self.assertFalse(c.numeric_invalid(final,a,w))
if __name__=='__main__':unittest.main()
