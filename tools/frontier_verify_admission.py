"""Offline audit of the frozen frontier admission and formal controller tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    for name in ('frontier_search','frontier_execution','frontier_knobs'):
        cmd = [sys.executable,'-m','unittest','discover','-s',str(ROOT/'test/plan_selection_study'/name),'-p','test_*.py','-v']
        p = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        (output/(name+'.json')).write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
        if p.returncode: return p.returncode
    record = json.loads((ROOT/'experiments/records/frontier-20261031-admission-001.json').read_text())
    if record['status'] != 'completed': raise ValueError('admission run did not complete')
    for item in record['artifacts']:
        p = ROOT/item['path']
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('archived evidence differs: '+item['path'])
    base = ROOT/'experiments/artifacts/frontier-20261031-admission-001'
    dest = output/'admission_gate.json'
    cmd = [sys.executable,str(ROOT/'test/plan_selection_study/frontier_execution/admission_gate.py'),
           '--rows',str(base/'admission.jsonl'),'--guard',str(base/'guard.jsonl'),
           '--incidents',str(base/'incidents.jsonl'),'--budget',str(base/'budget.jsonl'),
           '--gpu','GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b',
           '--execution-commit',record['execution_commit'],'--expected-count','7',
           '--original',str(base/'build/vkfft_frontier_original'),
           '--expanded',str(base/'build/vkfft_frontier_v2'),'--output',str(dest)]
    for p in sorted(base.glob('*.manifest.jsonl')): cmd += ['--manifest',str(p)]
    return subprocess.run(cmd,cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
