"""Audit recorded host build inputs; never load or execute the built binaries."""
import argparse
import json
from pathlib import Path

import admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-run', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    record_path = root / 'experiments/records' / (args.build_run + '.json')
    record = json.loads(record_path.read_text(encoding='utf8'))
    admission.need(record['status'] == 'completed' and record['returncode'] == 0,
                   'source build did not complete')
    directory = root / 'experiments/artifacts' / args.build_run
    for entry in record['artifacts']:
        path = root / entry['path']
        admission.need(path.resolve().is_relative_to(directory.resolve()), 'artifact outside build run')
        admission.need(path.stat().st_size == entry['bytes'] and admission.sha(path) == entry['sha256'],
                       'source build artifact differs: ' + entry['path'])
    original = root / 'VkFFT/vkFFT'
    upstream = admission.v2.git(original, 'rev-parse', 'HEAD')
    generated = admission.verify_generated(root, directory / 'generated/expanded_provenance.json', upstream)
    dependencies = list((directory / 'build/CMakeFiles/vkfft_nonuniform_original.dir').rglob('*.o.d'))
    admission.need(len(dependencies) == 1, 'expected exactly one original target dependency file')
    dependency_proof = admission.verify_build_dependencies(directory / 'build/compile_commands.json',
        dependencies[0], directory / 'generated', original, root / 'test/plan_selection_study/instrumentation')
    result = {'status': 'pass', 'build_run': args.build_run,
              'build_execution_commit': record['execution_commit'], 'record_sha256': admission.sha(record_path),
              'verified_artifacts': len(record['artifacts']),
              'tree_sha256': {name: generated[name + '_tree']['sha256']
                             for name in ('original', 'generated', 'instrumentation')},
              'dependency_proof': dependency_proof,
              'scope': 'host build inventory and include/dependency provenance only; no GPU execution or kernel JIT'}
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
