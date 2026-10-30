#!/usr/bin/env python3
"""Make an isolated v2-derived LUT/coalescing research benchmark."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'frontier_search'
spec = importlib.util.spec_from_file_location('frontier_v2_generator_for_knobs', V2 / 'prepare_headers.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one knobs anchor, found {text.count(old)}: {old!r}')
    return text.replace(old, new, 1)


def prepare(source, output):
    output = Path(output).resolve()
    # The v2 generator enforces an empty output and exclusive sibling artifacts.
    v2.prepare(source, output)
    benchmark_file = output.parent / 'expanded_benchmark.cpp'
    benchmark = benchmark_file.read_text(encoding='utf8')
    parent_benchmark_sha = sha(benchmark_file)
    old = 'o.aim || o.boost || o.lut || o.nonpow2 || o.coalesced || o.lut4 || o.swap2 || o.swap3)'
    new = 'o.aim || o.boost || o.nonpow2 || o.lut4 || o.swap2 || o.swap3)'
    benchmark = once(benchmark, old, new)
    benchmark = once(benchmark,
        'expanded candidate requires valid request, execute or describe/outofplace, and default config',
        'expanded candidate requires valid request, execute or describe/outofplace, and only LUT/coalesced config overrides')
    anchor = '        result.add("resolved_config", resolved(resources.app.configuration));'
    benchmark = once(benchmark, anchor, anchor + '''
        // Audit explicit knobs for original, experimental default and candidate.
        // Zero requests the upstream device-dependent default.
        if (o.lut && resources.app.configuration.useLUT != o.lut)
            throw std::runtime_error("requested LUT override differs from resolved configuration");
        if (o.coalesced && resources.app.configuration.coalescedMemory != (pfUINT)o.coalesced)
            throw std::runtime_error("requested coalesced-memory override differs from resolved configuration");''')
    benchmark_file.write_text(benchmark, encoding='utf8')
    provenance_file = output.parent / 'expanded_provenance.json'
    provenance = json.loads(provenance_file.read_text(encoding='utf8'))
    for rel in provenance['expanded']:
        provenance['expanded'][rel] = sha(output.parent / rel)
    provenance['frontier_knobs'] = {
        'schema': 'frontier-knobs-generated/v1',
        'scope': 'v2 boost 1, one upload; LUT and coalesced-memory overrides only',
        'generator_sha256': sha(Path(__file__)),
        'parent_v2_generator_sha256': sha(V2 / 'prepare_headers.py'),
        'parent_generated_benchmark_sha256': parent_benchmark_sha,
        'generated_benchmark_sha256': sha(benchmark_file),
        'device_validation': 'not performed by generator',
    }
    provenance_file.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
