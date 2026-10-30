import unittest
import describe_confirmed_results as subject


class ConfirmedDescriptionTest(unittest.TestCase):
    def test_reports_actual_resources_and_structure(self):
        signature={'structural_features':{'kernels':[{'stageRadix':[8,8,2],'groupedBatch':4,'localSize':[4,16,1],
                    'registerBoost':1,'dynamic_shared_bytes':1152}]},
                   'canonical_descriptor':{'logical_registers_per_thread':8,'min_logical_registers_per_thread':8},
                   'compiled_diagnostics':[{'cuda_static_shared_bytes':0,'cuda_num_regs':54,'cuda_local_bytes':0,'cuda_max_threads_per_block':1024}],
                   'source_sha256':'example'}
        out=subject.structure(signature)
        self.assertEqual(out['radix'],[8,8,2]);self.assertEqual(out['local_size'],[4,16,1]);self.assertEqual(out['cuda_registers_per_thread'],54)
        self.assertEqual(out['logical_registers'],8);self.assertEqual(out['dynamic_shared_bytes'],1152)


if __name__=='__main__':unittest.main()
