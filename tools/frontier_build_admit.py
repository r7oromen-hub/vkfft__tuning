"""Build and admit a new frontier prototype from the actual committed checkout."""
import hashlib
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / 'test/plan_selection_study/frontier_search'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'experiments/frontier_20261031_protocol.json')
    args=parser.parse_args()
    protocol = args.protocol.resolve(strict=True)
    if not protocol.is_relative_to(ROOT):raise ValueError('protocol must be inside checkout')
    cfg = json.loads(protocol.read_text())
    migration=cfg.get('migration_budget')
    if migration:
        if datetime.now(timezone.utc)>datetime.fromisoformat(migration['launch_not_after']):
            raise ValueError('migration launch allowance expired; recompute carried budget')
        if cfg['total_reserved_wall_seconds']+migration['charged_elapsed_including_300s_transfer_margin']!=migration['original_cap_seconds']:
            raise ValueError('migration budget does not reconcile')
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ)
    env['PATH'] = cfg['cuda_root'] + '/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = cfg['cuda_root'] + '/lib64:' + env.get('LD_LIBRARY_PATH', '')
    sys.path.insert(0, str(ROOT/'test/plan_selection_study/expanded_search'))
    import device_lock
    lockpath = Path('/run/lock') / ('vkfft-study-' + cfg['gpu_uuid'] + '.lock')
    build = output / 'build'
    generated = output / 'generated'
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(FRONT), '-p', 'test_*.py'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'test/plan_selection_study/frontier_execution'), '-p', 'test_*.py'],
        [sys.executable, str(FRONT/'prepare_headers.py'), '--source', str(ROOT/'VkFFT/vkFFT'), '--output', str(generated/'include')],
        ['cmake','-S',str(FRONT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
         '-DFRONTIER_GENERATED_ROOT='+str(generated),'-DCUDAToolkit_ROOT='+cfg['cuda_root'], '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
        ['cmake','--build',str(build),'--parallel','2'],
        ['ctest','--test-dir',str(build),'--output-on-failure'],
    ]
    with device_lock.DeviceLock(lockpath):
        query = subprocess.run(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],
                               capture_output=True,text=True,check=True)
        if query.stdout.strip(): raise RuntimeError('GPU has an active compute process')
        for i, cmd in enumerate(commands):
            t = time.monotonic()
            available=cfg['total_reserved_wall_seconds']-(t-started)
            if available<=360:raise RuntimeError('campaign budget exhausted before build step')
            r = subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=min(900,available-360))
            record = dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
                          seconds=time.monotonic()-t, execution_commit=os.environ['EXPERIMENT_CODE_COMMIT'])
            (output/f'build_step_{i}.json').write_text(json.dumps(record,indent=2)+'\n')
            print('build_step',i,'exit',r.returncode,flush=True)
            if r.returncode: return r.returncode
    spec = importlib.util.spec_from_file_location('frontier_manifest',FRONT/'manifest.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    manifests = []
    for w in cfg['admission_workloads']:
        header, rows = mod.build(w['n'],w['batch'],None,w['requests'],len(w['requests']))
        path = output/f"n{w['n']}_b{w['batch']}.manifest.jsonl"
        with path.open('x') as f:
            for row in [header,*rows]: f.write(json.dumps(row,sort_keys=True)+'\n')
        manifests += ['--manifest',str(path)]
    elapsed = time.monotonic()-started
    (output/'campaign_start.json').write_text(json.dumps(dict(start_monotonic=started,build_elapsed_seconds=elapsed,
        total_reserved_wall_seconds=cfg['total_reserved_wall_seconds'],gpu=cfg['gpu_uuid'],
        execution_commit=os.environ['EXPERIMENT_CODE_COMMIT']),indent=2)+'\n')
    remaining = min(cfg['initial_admission_seconds'], cfg['total_reserved_wall_seconds']-elapsed)
    if remaining <= 360: raise RuntimeError('Insufficient admission budget after build')
    cmd = [sys.executable,str(ROOT/'test/plan_selection_study/frontier_execution/runner.py'),
           '--gpu',cfg['gpu_uuid'],'--original',str(build/'vkfft_frontier_original'),
           '--expanded',str(build/'vkfft_frontier_v2'),*manifests,
           '--source-file',str(protocol),'--source-file',str(FRONT/'CMakeLists.txt'),
           '--artifact-file',str(build/'compile_commands.json'),
           '--generated-provenance',str(generated/'expanded_provenance.json'),
           '--output',str(output/'admission.jsonl'),'--budget-ledger',str(output/'budget.jsonl'),
           '--log',str(output/'guard.jsonl'),'--incidents',str(output/'incidents.jsonl'),
           '--lock',str(lockpath),'--seconds',str(remaining),'--timeout','300']
    (output/'admission_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
