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
           'descriptor_validation':{'all_match':True},'vkfft_result_code':0,'direct_resource_validation':{'threads':8,'function_thread_limit':1024,'dynamic_shared_bytes':0,'static_shared_bytes':0,'function_dynamic_limit_bytes':49152,'device_shared_limit_bytes':49152},'precompile_kernels':[{'compiler_options_match':True}], 'structural_features':{'numAxisUploads':1,'kernels':[{'stageRadix':[2,2],'localSize':[8,1,1],'groupedBatch':1}]},
           'status':'error','failure_phase':'single_forward_validation','error':'Single out-of-place forward failed immutable-input or numerical validation',
           'single_forward_correctness':{'passed':False,'checked_elements':8,'nonfinite_count':0,'failing_batches':1}}
        row={'guard_clean':True,'returncode':1,'phase':'numerical','direct_environment':{'VKFFT_DIRECT_RADIX':'2,2','VKFFT_DIRECT_BATCH':'1','VKFFT_DIRECT_BLOCK':'8,1,1'},'result':r}
        self.assertTrue(c.numeric_invalid(row,a,w))
        for field,value in [('single_forward_input_unchanged',False),('failure_phase','execution')]:
            bad=copy.deepcopy(row);bad['result'][field]=value;self.assertFalse(c.numeric_invalid(bad,a,w))
        bad=copy.deepcopy(row);bad['guard_clean']=False;self.assertFalse(c.numeric_invalid(bad,a,w))
        bad=copy.deepcopy(row);bad['result']['single_forward_correctness']['checked_elements']=7;self.assertFalse(c.numeric_invalid(bad,a,w))
        final=copy.deepcopy(row);final['result'].update(status='correctness_failed',input_unchanged=True,
            single_forward_correctness={'passed':True,'checked_elements':8,'nonfinite_count':0,'failing_batches':0},correctness={'passed':False,'checked_elements':8,'nonfinite_count':0,'failing_batches':1})
        self.assertTrue(c.numeric_invalid(final,a,w))
        final['result']['input_unchanged']=False;self.assertFalse(c.numeric_invalid(final,a,w))
if __name__=='__main__':unittest.main()
