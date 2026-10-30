import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import default_controller as c
import default_gpu_guard as guard
import run_suite as suite

HERE = Path(__file__).resolve().parent

def successful(args, w, candidate, phase, round_no, order):
    stage = [8, 8, 4]
    local = [32, 1, 1]
    source = 'a' * 64
    correct = dict(passed=True, nonfinite_count=0, failing_batches=0, checked_elements=w['n']*w['batch'])
    pre = dict(raw_descriptor={'localSize': local}, canonical_descriptor={'stageRadix': stage, 'localSize': local},
               source_sha256=source, source_bytes=123, compiler_options_match=True,
               actual_compiler_options=['--gpu-architecture=sm_89'],
               compiler_input={'source_sha256': source, 'options':['--gpu-architecture=sm_89']})
    result = dict(status='ok', gpu={'uuid': args.gpu, 'name':'NVIDIA GeForce RTX 4090'}, n=w['n'], batch=w['batch'], seed=w['seed'],
                  requested_config={'makeForwardPlanOnly': 1}, validation_output_initialization='0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
                  single_forward_input_unchanged=True, input_unchanged=True, single_forward_correctness=correct, correctness=correct,
                  descriptor_validation={'all_match': True}, structural_features={'numAxisUploads':1, 'kernels':[{'stageRadix':stage, 'localSize':local}]},
                  precompile_kernels=[pre], graph_repeats=1, graph_kernel_count=1, graph_event_record_count=2, exec_ms_samples=[0.1])
    return dict(status='ok', guard_clean=True, direct_environment=candidate[1], result=result, workload=w)

class Tests(unittest.TestCase):
    def test_all_nine_cases_guard_and_complete(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            for n in c.LENGTHS:
                for batch in c.BATCHES:
                    path = Path(temp) / f'{n}_{batch}.jsonl'
                    argv = ['default_controller.py', '--exe',sys.executable, '--gpu','GPU-test', '--n',str(n), '--batch',str(batch), '--output',str(path)]
                    with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'VKFFT_STUDY_GUARD_ID':'test', 'VKFFT_STUDY_GUARD_UUID':'GPU-test'}), patch.object(c.suite,'run_one',successful):
                        c.main()
                    rows = [json.loads(x) for x in path.read_text().splitlines()]
                    self.assertEqual(rows[-1]['record_type'], 'controller_complete')
                    self.assertEqual(rows[0]['workload']['config'], {})
                    self.assertEqual(rows[0]['direct_environment'], {})

    def test_validation_fails_closed(self):
        class Args: gpu='GPU-test'
        w = dict(n=256,batch=16,seed=42,config={})
        for mutate in (
            lambda r: r.update(guard_clean=False),
            lambda r: r['result']['requested_config'].update(aimThreads=128),
            lambda r: r['result']['correctness'].update(checked_elements=1),
            lambda r: r['result']['precompile_kernels'][0].update(source_sha256='bad'),
            lambda r: r['result']['structural_features']['kernels'][0].update(localSize=[64,1,1]),
            lambda r: r.update(direct_environment={'VKFFT_DIRECT_BLOCK':'32,1,1'}),
        ):
            row=successful(Args(),w,('default',{}),'default_probe',0,0)
            mutate(row)
            with self.assertRaises(ValueError): c.validate(row,Args(),w)

    def test_runner_clears_injection_and_does_not_pass_knobs(self):
        class Args:
            exe=sys.executable
            import hashlib
            exe_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
            timeout=3
        class Process:
            returncode=0; stdout='{}\n'; stderr=''
        w=dict(n=256,batch=1,seed=42,config={},pattern='random',graph_repeats=1,warmup=0,warmup_ms=0,repeats=1)
        def fake(command, **kwargs):
            for field in suite.FIELDS: self.assertNotIn(field, kwargs['env'])
            for flag in ('--aim-threads','--register-boost','--use-lut','--register-boost-nonpow2'): self.assertNotIn(flag,command)
            return Process()
        with patch.dict(os.environ, {x:'inherited-bad-value' for x in suite.FIELDS}), patch.object(suite.subprocess,'run',fake):
            self.assertEqual(suite.run_one(Args(),w,('default',{}),'probe',0,0)['status'],'ok')

    def test_guard_wraps_exact_runner(self):
        class Args: gpu='GPU-test'; log=Path('unused'); incidents=Path('unused')
        g=guard.Guard(Args(),query=lambda *a:None)
        g.emit=lambda *a,**kw:None
        g.sample=lambda b:{'utc':'sample'}
        class Target: exe=sys.executable
        row=g.wrapped(lambda *a:{'status':'ok'})(Target(), {'n':256,'batch':1},('default',{}),'probe',0,0)
        self.assertTrue(row['guard_clean'])
        self.assertTrue(row['guard_run_id'])
        self.assertIsNone(g.active)

if __name__ == '__main__': unittest.main()
