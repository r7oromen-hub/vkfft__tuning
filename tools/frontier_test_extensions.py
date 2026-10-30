"""Run the committed CPU checks for joint knobs and independent confirmation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    protocol = json.loads((ROOT/'experiments/frontier_extensions_cpu.json').read_text())
    results = []
    for name in protocol['suites']:
        cmd = [sys.executable, '-m', 'unittest', 'discover', '-s',
               str(ROOT/'test/plan_selection_study'/name), '-p', 'test_*.py', '-v']
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        item = dict(suite=name, command=cmd, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        results.append(item)
        (output/(name+'.json')).write_text(json.dumps(item, indent=2)+'\n', encoding='utf-8')
        print(name, 'exit', r.returncode, flush=True)
        if r.returncode:
            return r.returncode
    (output/'summary.json').write_text(json.dumps(dict(status='pass', gpu_execution=False,
        execution_commit=os.environ['EXPERIMENT_CODE_COMMIT'], suites=[r['suite'] for r in results]),indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
