import unittest
import enumerate_candidates as enum


class EnumerationTest(unittest.TestCase):
    def test_ordered_decompositions_include_new_endpoints_and_multisets(self):
        values=list(enum.decompositions(256))
        self.assertEqual(len(values), 120)
        self.assertIn((16,16), values)
        self.assertIn((8,4,8), values)
        self.assertIn((2,2,2,2,2,2,2,2), values)
        self.assertIn((32,8), values)
        self.assertEqual(len(values),len(set(values)))

    def test_complete_product_space_and_request_identity(self):
        values=list(enum.candidates(256, 16))
        self.assertEqual(len(values),120*6*5)
        self.assertEqual(len({r['candidate_id'] for r in values}),len(values))
        rows=[r for r in values if r['request']=={'radix':[8,4,8], 'logical_registers':8, 'grouped_batch':4}]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['planned_fft_threads'],32)
        self.assertEqual(rows[0]['status'],'pending_scheduler_and_device_validation')
        rows=[r for r in values if r['request']=={'radix':[32,8], 'logical_registers':8, 'grouped_batch':4}]
        self.assertEqual(rows[0]['reason'],'radix_exceeds_registers')

    def test_input_range(self):
        with self.assertRaises(ValueError): list(enum.decompositions(257))
        with self.assertRaises(ValueError): list(enum.candidates(256,0))


if __name__=='__main__': unittest.main()
