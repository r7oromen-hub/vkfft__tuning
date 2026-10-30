import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import describe_controller as controller
import test_probe as fixtures


def probe_record():
    record = fixtures.ComparisonTest().record()
    record.update({'requested_config': {}, 'placement': 'outofplace', 'backend': 'CUDA_NVRTC', 'precision': 'complex_fp32',
                   'cuda_runtime_version': 12060, 'cuda_driver_version': 12060, 'nvrtc_version': '12.6',
                   'gpu': {'uuid': 'GPU-test', 'compute_capability': '8.0'}})
    kernel = record['precompile_kernels'][0]
    kernel.update({'canonical_descriptor': kernel['raw_descriptor'], 'source_sha256': 'abc', 'source_bytes': 3, 'compiler_input': {}})
    return record


class DescribeControllerTest(unittest.TestCase):
    def test_subprocess_clears_overrides_and_uses_describe(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            exe = Path(tmp) / 'exe'; exe.write_bytes(b'fixture')
            args = SimpleNamespace(exe=exe, exe_sha256=controller.digest(exe), timeout=10)
            response = SimpleNamespace(returncode=0, stdout='{}\n', stderr='')
            with patch.dict(os.environ, {'VKFFT_DIRECT_RADIX': '8,4'}), patch.object(controller.subprocess, 'run', return_value=response) as run:
                row = controller.run_one(args, {'n': 256, 'batch': 16}, ('probe', {}), 'describe', 0, 0)
            self.assertEqual(row['status'], 'ok')
            self.assertNotIn('VKFFT_DIRECT_RADIX', run.call_args.kwargs['env'])
            self.assertEqual(run.call_args.args[0][-4:], ['--study-mode', 'describe', '--placement', 'outofplace'])

    def test_pairing_and_fail_closed_output(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                root = Path(tmp)
                probe, baseline = root/'probe', root/'baseline'
                probe.write_bytes(b'probe'); baseline.write_bytes(b'baseline')
                args = SimpleNamespace(gpu='GPU-test', probe=probe, baseline=baseline, output=root/'out', case=['256:16'], timeout=10)
                def fake_run(_a, _workload, candidate, *_rest):
                    record = probe_record()
                    if candidate[0] == 'baseline':
                        del record['precompile_kernels'][0]['precodegen_raw_descriptor']
                        if fail:
                            record['precompile_kernels'][0]['source_sha256'] = 'mismatch'
                    return {'status': 'ok', 'guard_clean': True, 'result': record, 'stdout': json.dumps(record)}
                with patch.dict(os.environ, {'VKFFT_STUDY_GUARD_ID': 'guard', 'VKFFT_STUDY_GUARD_UUID': 'GPU-test'}), patch.object(controller, 'run_one', side_effect=fake_run):
                    if fail:
                        with self.assertRaises(ValueError): controller.collect(args)
                        self.assertFalse((args.output/'comparison.json').exists())
                    else:
                        self.assertEqual(controller.collect(args)['kernel_pairs'], 1)
                        self.assertTrue((args.output/'comparison.json').exists())
                        self.assertEqual(len((args.output/'probe.jsonl').read_text().splitlines()), 1)
                    self.assertEqual(len((args.output/'runs.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
