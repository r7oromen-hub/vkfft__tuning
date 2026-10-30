#!/usr/bin/env python3
"""Build a separate wide-R research header tree, never modifying upstream."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'expanded_search'
sys.path.insert(0, str(HERE))
import rejection_instrumentation

spec = importlib.util.spec_from_file_location('expanded_v1_prepare', OLD / 'prepare_headers.py')
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    output = Path(output).resolve()
    for filename in ('expanded_benchmark.cpp', 'expanded_provenance.json'):
        if (output.parent / filename).exists():
            raise ValueError(f'output parent already contains {filename}')
    # v1 requires an empty header destination and generates an isolated tree.
    v1.prepare(source, output)
    (output / 'candidate.hpp').write_bytes((HERE / 'candidate.hpp').read_bytes())
    rejection_instrumentation.instrument(output)
    provenance_file = output.parent / 'expanded_provenance.json'
    provenance = json.loads(provenance_file.read_text(encoding='utf8'))
    for rel in list(provenance['expanded']):
        path = output.parent / rel
        provenance['expanded'][rel] = sha(path)
    provenance['frontier_v2'] = {
        'scope': 'single upload, boost 1, power-of-two length; logical R up to 256',
        'device_validation': 'not performed by generator',
        'generator_sha256': sha(Path(__file__)),
        'candidate_sha256': sha(HERE / 'candidate.hpp'),
        'rejection_instrumentation_sha256': sha(HERE / 'rejection_instrumentation.py'),
        'rejection_header_sha256': sha(HERE / 'rejection.hpp'),
        'generated_rejection_header_sha256': sha(output / 'rejection.hpp'),
    }
    provenance_file.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
