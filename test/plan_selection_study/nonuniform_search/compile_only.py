"""Generate and build host harnesses without executing them or touching a GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cuda-root', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    output = args.output.resolve(strict=True)
    generated, build = output / 'generated', output / 'build'
    if generated.exists() or build.exists():
        raise ValueError('build and generated outputs must not already exist')
    commands = [
        [str(args.cuda_root / 'bin/nvcc'), '--version'],
        ['g++', '--version'],
        [sys.executable, str(here / 'prepare_headers.py'), '--source', str(root / 'VkFFT/vkFFT'),
         '--output', str(generated / 'include')],
        ['cmake', '-S', str(here), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DNONUNIFORM_GENERATED_ROOT=' + str(generated),
         '-DCUDAToolkit_ROOT=' + str(args.cuda_root), '-DPython3_EXECUTABLE=' + sys.executable],
        ['cmake', '--build', str(build), '--parallel', '2'],
    ]
    for command in commands:
        print(json.dumps(command), flush=True)
        subprocess.run(command, check=True)
    print(json.dumps({'status': 'host_harnesses_built_only', 'executed_harnesses': False,
                      'generated_cuda_kernels_compiled': False, 'gpu_validation': False}))


if __name__ == '__main__':
    main()
