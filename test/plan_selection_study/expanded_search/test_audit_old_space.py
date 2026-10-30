import copy
import unittest
import audit_old_space


class OldSpaceTest(unittest.TestCase):
    def fixture(self):
        table=[8 if i in (2,4,8) else 0 for i in range(33)]
        return {'n':256,'batch':16,'structural_features':{'numAxisUploads':1,'kernels':[
            {'stageRadix':[8,8,4],'localSize':[32,1,1],'groupedBatch':1}]},
            'precompile_kernels':[{'raw_descriptor':{'registerBoost':1,'fftDim':256,'complexSize':8,
                'axisSwapped':0,'registers_per_radix':table,'min_logical_registers_per_thread':8,'logical_registers_per_thread':8}}],
            'resolved_config':{'sharedMemorySize':49152,'maxThreadsNum':1024}}

    def test_old_structural_subset_and_endpoint(self):
        base=self.fixture();request={'radix':[8,4,2,4],'logical_registers':8,'grouped_batch':1}
        self.assertEqual(audit_old_space.membership(request,base,[32,1,1])['membership'],'old_structural_subset')
        request['radix']=[4,8,8]
        self.assertEqual(audit_old_space.membership(request,base)['reason'],'fixed_endpoints_or_two_stage_rule')

    def test_old_table_and_actual_block_must_match(self):
        base=self.fixture();request={'radix':[8,8,4],'logical_registers':16,'grouped_batch':1}
        self.assertEqual(audit_old_space.membership(request,base)['reason'],'register_table_or_envelope_changed')
        request['logical_registers']=8
        self.assertEqual(audit_old_space.membership(request,base,[1,32,1])['reason'],'actual_block_differs_from_old_request')

    def test_two_stage_baseline_cannot_refactor(self):
        base=copy.deepcopy(self.fixture());base['structural_features']['kernels'][0]['stageRadix']=[16,16]
        request={'radix':[8,8,4],'logical_registers':8,'grouped_batch':1}
        self.assertEqual(audit_old_space.membership(request,base)['reason'],'fixed_endpoints_or_two_stage_rule')


if __name__=='__main__':unittest.main()
