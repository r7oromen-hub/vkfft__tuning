import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('knob_controller',HERE/'joint_controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class KnobTest(unittest.TestCase):
    def test_requested_knobs_and_randomized_blocks(self):
        seen=[]
        def run(args,w,candidate,phase,round_no,order):
            name,_=candidate;config=w['config'];seen.append((name,phase,w['seed'],round_no,order))
            requested={'makeForwardPlanOnly':1}
            if config:requested.update(aimThreads=config['aim_threads'],registerBoost=config['register_boost'],useLUT=config['use_lut'])
            return {'direct_environment':{},'result':{'requested_config':requested,'signature':name,
                'raw_interval_ms_samples':[3.0]*w['repeats'],'exec_ms_samples':[1.0]*w['repeats']}}
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            exe=Path(d)/'binary';exe.write_bytes(b'example');out=Path(d)/'rows.jsonl'
            argv=['joint_controller.py','--exe',str(exe),'--gpu','GPU-test','--output',str(out),'--case','small']
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'id','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(c.suite,'run_one',side_effect=run),patch.object(c.core,'validate',side_effect=lambda r,a,n:r['result']), \
                 patch.object(c.core,'signature',side_effect=lambda r:r['signature']):
                c.main()
            rows=[json.loads(x) for x in out.read_text().splitlines()]
        rounds=[r for r in rows if r.get('record_type')=='paired_round'];self.assertEqual(len(rounds),10)
        for row in rounds:
            block=[x for x in seen if x[1]=='paired_timing' and x[3]==row['round']]
            self.assertEqual(len(block),9);self.assertEqual({x[2] for x in block},{row['seed']})
            self.assertEqual(set(row['order']),{'default',*c.CONFIGS})
        self.assertEqual(rows[-1]['record_type'],'controller_complete')

    def test_wrong_requested_knobs_fail_closed(self):
        row={'direct_environment':{},'result':{'requested_config':{'makeForwardPlanOnly':1}}}
        with self.assertRaises(ValueError):c.checked(row,None,1,{'aim_threads':64,'register_boost':1,'use_lut':1})

if __name__=='__main__':unittest.main()
