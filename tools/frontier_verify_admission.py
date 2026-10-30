"""Offline audit of the frozen frontier admission and formal controller tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frontier_campaign_inputs', ROOT/'tools/frontier_campaign_inputs.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'experiments/frontier_20261031_protocol.json')
    parser.add_argument('--admission-run', default='frontier-20261031-admission-001')
    args = parser.parse_args(argv)
    protocol, config = campaign.load_protocol(ROOT, args.protocol)
    record, base, _ = campaign.load_completed_run(ROOT, args.admission_run, protocol)
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    for name in ('frontier_search','frontier_execution','frontier_knobs','frontier_confirmation'):
        cmd = [sys.executable,'-m','unittest','discover','-s',str(ROOT/'test/plan_selection_study'/name),'-p','test_*.py','-v']
        p = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        (output/(name+'.json')).write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
        if p.returncode: return p.returncode
    campaign.verify_artifacts(ROOT, record, [ROOT/item['path'] for item in record['artifacts']])
    dest = output/'admission_gate.json'
    cmd = [sys.executable,str(ROOT/'test/plan_selection_study/frontier_execution/admission_gate.py'),
           '--rows',str(base/'admission.jsonl'),'--guard',str(base/'guard.jsonl'),
           '--incidents',str(base/'incidents.jsonl'),'--budget',str(base/'budget.jsonl'),
           '--gpu',config['gpu_uuid'],
           '--execution-commit',record['execution_commit'],
           '--expected-count',str(sum(len(w['requests']) for w in config['admission_workloads'])),
           '--original',str(base/'build/vkfft_frontier_original'),
           '--expanded',str(base/'build/vkfft_frontier_v2'),'--output',str(dest)]
    for p in sorted(base.glob('*.manifest.jsonl')): cmd += ['--manifest',str(p)]
    return subprocess.run(cmd,cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
