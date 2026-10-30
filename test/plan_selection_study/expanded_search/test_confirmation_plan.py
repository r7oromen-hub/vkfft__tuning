import copy
import unittest
import confirmation_plan


class PlanTest(unittest.TestCase):
    def fixture(self):
        p={'record_type':'search_protocol','session_id':'s','binaries':{'original':'o','expanded':'e'},
           'sha256':{'original':'a','expanded':'b'},'protocol':{'batches':[1],'lengths':[128],'selection_seed':1}}
        rows=[p];ratios=[]
        for i in range(3):
            o={'record_type':'execution','guard_run_id':f'o{i}','guard_clean':True,'candidate_id':'original',
               'phase':'screen_timing','round':i,'order':0,'workload':{'n':128,'batch':1},'result':{'exec_ms_samples':[1,1,1]}}
            c={'record_type':'execution','guard_run_id':f'c{i}','guard_clean':True,'candidate_id':'x',
               'phase':'screen_timing','round':i,'order':1,'workload':{'n':128,'batch':1},'result':{'exec_ms_samples':[.9,.9,.9]}}
            rows += [o,c,{'record_type':'screen_pair','n':128,'batch':1,'candidate_id':'x','round':i,
                          'original_guard_run_id':f'o{i}','candidate_guard_run_id':f'c{i}',
                          'original_median_ms':1,'candidate_median_ms':.9,'candidate_over_original':.9}]
            ratios.append(.9)
        rows += [{'record_type':'candidate_disposition','n':128,'batch':1,'candidate_id':'x','status':'screen_complete',
                  'median_ratio':.9,'pair_ratios':ratios,'request':{},'graph_repeats':64,
                  'candidate_signature':{},'original_signature':{}},
                 {'record_type':'search_complete','status':'budget_exhausted'}]
        return rows

    def test_raw_recomputed_and_fallback(self):
        rows=self.fixture();plan=confirmation_plan.make(rows,'hash')
        self.assertEqual(plan['workloads'][0]['candidate']['candidate_id'],'x')
        rows[-2]['median_ratio']=1.01
        with self.assertRaises(ValueError):confirmation_plan.make(rows,'hash')

    def test_missing_or_forged_pair_fails(self):
        rows=self.fixture();rows[3]['candidate_over_original']=.8
        with self.assertRaises(ValueError):confirmation_plan.make(rows,'hash')
        rows=self.fixture();rows=[r for r in rows if r.get('guard_run_id')!='c0']
        with self.assertRaises(KeyError):confirmation_plan.make(rows,'hash')

    def test_faster_first_and_at_most_two_default_controls(self):
        allrows=[]
        for n,ratio in ((128,1.2),(256,1.05),(512,.95),(1024,1.1),(2048,.9)):
            rows=self.fixture()
            for row in rows:
                if row.get('n')==128:row['n']=n
                if row.get('workload',{}).get('n')==128:row['workload']['n']=n
                if row.get('record_type')=='execution':
                    row['guard_run_id']+=str(n)
                    if row['candidate_id']=='x':row['result']['exec_ms_samples']=[ratio]*3
                if row.get('record_type')=='screen_pair':
                    row['original_guard_run_id']+=str(n);row['candidate_guard_run_id']+=str(n)
                    row['candidate_median_ms']=ratio;row['candidate_over_original']=ratio
                if row.get('record_type')=='candidate_disposition':row['pair_ratios']=[ratio]*3;row['median_ratio']=ratio
            if not allrows:
                rows[0]['protocol']['lengths']=[128,256,512,1024,2048];allrows=rows
            else:allrows[-1:-1]=rows[1:-1]
        plan=confirmation_plan.make(allrows,'hash')
        self.assertEqual([(x['n'],x['role']) for x in plan['confirmation_order']],
                         [(2048,'screen_faster'),(512,'screen_faster'),(256,'default_faster_control'),(1024,'default_faster_control')])


if __name__=='__main__':unittest.main()
