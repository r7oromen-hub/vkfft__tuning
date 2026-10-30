import copy
import unittest
import identity


class IdentityTest(unittest.TestCase):
    def test_full_table_and_layout_are_required(self):
        regs=[16 if i in (2,4,8,16) else 0 for i in range(33)]
        result={'n':256,'batch':16,'structural_features':{'numAxisUploads':1,'kernels':[{'stageRadix':[16,16],
                'groupedBatch':8,'localSize':[8,16,1]}]},'precompile_kernels':[{'raw_descriptor':{'registerBoost':1,
                'logical_registers_per_thread':16,'min_logical_registers_per_thread':16,
                'registers_per_radix':regs,'axisSwapped':1}}]}
        req,reason=identity.representable(result)
        self.assertIsNone(reason);self.assertEqual(req['grouped_batch'],8)
        changed=copy.deepcopy(result);changed['precompile_kernels'][0]['raw_descriptor']['registers_per_radix'][2]=0
        self.assertEqual(identity.representable(changed)[1],'nonuniform_register_table')
        changed=copy.deepcopy(result);changed['structural_features']['kernels'][0]['localSize']=[16,8,1]
        self.assertEqual(identity.representable(changed)[1],'unsupported_local_layout')


if __name__=='__main__': unittest.main()
