import copy
import unittest
import validation


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.gpu='GPU-test';self.req={'radix':[16,16], 'logical_registers':16, 'grouped_batch':4}
        self.row={'guard_clean':True,'guard_gpu_uuid':self.gpu,'returncode':0,
                  'direct_environment':validation.request_env(self.req),'workload':{'n':256,'batch':16},
                  'result':{'status':'ok','study_mode':'describe','describe_only':True,'n':256,'batch':16,
                            'gpu':{'uuid':self.gpu},'init_trace':{'counters':{'nvrtc_compile':0,'module_load':0,'driver_kernel_launch':0}},
                            'descriptor_validation':{'all_match':True},
                            'structural_features':{'numAxisUploads':1,'kernels':[{'stageRadix':[16,16],'groupedBatch':4,'localSize':[4,16,1]}]},
                            'precompile_kernels':[{'raw_descriptor':{'logical_registers_per_thread':16,'registerBoost':1,'axisSwapped':1}}]}}

    def test_describe_guard_and_execution_counters_fail_closed(self):
        validation.check_describe(self.row,self.gpu,self.req)
        for field,value in [('guard_clean',False),('guard_gpu_uuid','GPU-other')]:
            row=copy.deepcopy(self.row);row[field]=value
            with self.assertRaises(ValueError): validation.check_describe(row,self.gpu,self.req)
        row=copy.deepcopy(self.row);row['result']['init_trace']['counters']['nvrtc_compile']=1
        with self.assertRaises(ValueError): validation.check_describe(row,self.gpu,self.req)

    def test_only_known_clean_rejection(self):
        row={'guard_clean':True,'guard_gpu_uuid':self.gpu,'returncode':1,
             'result':{'status':'error','failure_phase':'initialize','error':'initializeVkFFT failed with code 3002'}}
        self.assertTrue(validation.known_rejection(row,self.gpu))
        for error in ('initializeVkFFT failed with code 5001','out of memory','NVRTC failed'):
            row['result']['error']=error;self.assertFalse(validation.known_rejection(row,self.gpu))
        row['result']['error']='initializeVkFFT failed with code 3002';row['guard_clean']=False
        self.assertFalse(validation.known_rejection(row,self.gpu))

    def test_identity_signature_includes_resources_and_configuration(self):
        base={'structural_features':{},'compiled_diagnostics':{'registers':4},'resolved_config':{'x':1},
              'precompile_kernels':[{'canonical_descriptor':{},'source_sha256':'a','source_bytes':1,
                                     'compiler_input':{},'actual_compiler_options':[]}]}
        other=copy.deepcopy(base);other['compiled_diagnostics']['registers']=5
        self.assertNotEqual(validation.signature(base),validation.signature(other))
        other=copy.deepcopy(base);other['resolved_config']['x']=2
        self.assertNotEqual(validation.signature(base),validation.signature(other))

    def test_missing_instrumentation_fails_closed(self):
        row={'guard_clean':True,'guard_gpu_uuid':self.gpu,'returncode':0,'status':'ok',
             'workload':{'n':256,'batch':16,'seed':42,'pattern':'random'},
             'result':{'status':'ok','gpu':{'uuid':self.gpu,'cuda_visible_devices':self.gpu},
                       'n':256,'batch':16,'seed':42,'pattern':'random','requested_config':{'makeForwardPlanOnly':1}}}
        with self.assertRaisesRegex(ValueError, 'instrumentation'): validation.check(row,self.gpu)


if __name__=='__main__': unittest.main()
