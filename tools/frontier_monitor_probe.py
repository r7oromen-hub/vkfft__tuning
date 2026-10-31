"""Record read-only monitor latency and parsing on the selected device."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('monitor_guard',ROOT/'test/plan_selection_study/original_compare/original_compare_guard.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',required=True)
    args=parser.parse_args()
    output=Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    query=guard.GPUQuery(args.gpu)
    measurements=[]
    for index in range(30):
        begin=time.monotonic()
        gpu,apps=query(None,os.getpid())
        elapsed=time.monotonic()-begin
        row={'sample':index,'seconds':elapsed,'gpu':gpu,'apps':apps}
        with (output/'samples.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n')
        if elapsed>2 or apps:
            raise RuntimeError('probe exceeded unchanged 2s limit or observed foreign compute')
        measurements.append(elapsed)
    result={'samples':len(measurements),'median_seconds':statistics.median(measurements),
            'max_seconds':max(measurements),'gpu':args.gpu,
            'scope':'read-only idle-device query; not active-harness or FFT performance validation'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
