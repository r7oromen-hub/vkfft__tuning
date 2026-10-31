"""Run registered CPU checks only; does not compile CUDA or use a device."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    artifacts = Path(os.environ['EXPERIMENT_ARTIFACT_DIR']).resolve()
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ is required for the independent C++ request checks')
    environment = dict(os.environ)
    # MinGW's compiler subprocesses resolve their runtime DLLs through PATH.
    # An absolute g++ path alone does not ensure cc1plus can load those DLLs.
    environment['PATH'] = str(Path(compiler).parent) + os.pathsep + environment.get('PATH', '')
    executable = artifacts / ('candidate_cpu.exe' if os.name == 'nt' else 'candidate_cpu')
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(here), '-p', 'test_*.py', '-v'],
        [compiler, '-std=c++17', '-O2', '-UNDEBUG', *(['-static'] if os.name == 'nt' else []),
         str(here / 'test_candidate.cpp'), '-o', str(executable)],
        [str(executable)],
    ]
    for command in commands:
        print(repr(command), flush=True)
        subprocess.run(command, check=True, env=environment)


if __name__ == '__main__':
    main()
