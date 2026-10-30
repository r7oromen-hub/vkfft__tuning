import importlib.util, json, math, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('factorial_controller', HERE/'joint_controller.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class FactorialTest(unittest.TestCase):
    def test_randomized_complete_blocks_and_interaction(self):
        env={k+str(g): {'radix':v,'grouped_batch':g,'block':[g,128,1]}
             for k,v in c.RADICES.items() for g in (4,8)}
        seen=[]
        def fake_run(args,w,candidate,phase,round_no,order):
            name,_=candidate;seen.append((name,phase,w['seed'],round_no,order))
            req=env.get(name,{'radix':[8,8,8,2],'grouped_batch':1,'block':[128,1,1]})
            t={'default':1,'identity':1,'A4':1,'B4':2,'A8':4,'B8':2}[name]
            return {'result': {'signature':name if name in env else 'default',
                'structural_features':{'kernels':[{'stageRadix':req['radix'],'groupedBatch':req['grouped_batch'],'localSize':req['block']}]},
                'raw_interval_ms_samples':[3.0]*w['repeats'],'exec_ms_samples':[float(t)]*w['repeats']}}
        def fake_shape(result,req):
            k=result['structural_features']['kernels'][0]
            assert (k['stageRadix'],k['groupedBatch'],k['localSize'])==(req['radix'],req['grouped_batch'],req['block'])
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            out=Path(tmp)/'out.jsonl'; exe=Path(tmp)/'binary';exe.write_bytes(b'example')
            argv=['joint_controller.py','--exe',str(exe),'--gpu','GPU-test','--output',str(out),'--rounds','10']
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'id','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(c.suite,'run_one',side_effect=fake_run),patch.object(c.core,'validate',side_effect=lambda r,a,n:r['result']), \
                 patch.object(c.core,'signature',side_effect=lambda r:r['signature']),patch.object(c.core,'assert_shape',side_effect=fake_shape), \
                 patch.object(c.core.enum,'candidates',return_value=[{'request':v,'reason':None} for v in env.values()]):
                c.main()
            rows=[json.loads(x) for x in out.read_text().splitlines()]
        rounds=[r for r in rows if r.get('record_type')=='paired_round']
        self.assertEqual(len(rounds),10)
        self.assertTrue(all(abs(r['ratio_of_ratios']-4)<1e-12 for r in rounds))
        self.assertTrue(all(set(r['order'])=={'default','A4','A8','B4','B8'} for r in rounds))
        for r in rounds:
            block=[x for x in seen if x[1]=='paired_timing' and x[3]==r['round']]
            self.assertEqual(len(block),5);self.assertEqual({x[2] for x in block},{r['seed']})
        self.assertTrue(math.isclose(rows[-1]['geometric_interaction'],4.0))

    def test_reject_without_matching_guard(self):
        with patch.object(sys,'argv',['joint_controller.py','--exe','no','--gpu','GPU-test','--output','out']), \
             patch.dict(os.environ,{},clear=True):
            with self.assertRaises(SystemExit): c.main()


if __name__=='__main__': unittest.main()
