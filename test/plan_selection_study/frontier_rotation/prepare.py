#!/usr/bin/env python3
"""Generate an isolated L2-rotation benchmark from a frozen frontier v2 tree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OLD = STUDY / 'expanded_search'
REPO = HERE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root):
    """Hash the entire input tree; do not follow links outside the evidence."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('input tree must be a directory')
    entries = {}
    for path in sorted(root.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError('symlinks are not allowed in input trees')
        if path.is_file():
            entries[path.relative_to(root).as_posix()] = sha(path)
        elif not path.is_dir():
            raise ValueError('unsupported file type in input tree')
    if not entries:
        raise ValueError('input tree is empty')
    payload = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return {'files': entries, 'sha256': hashlib.sha256(payload).hexdigest()}


def verify_frontier(generated):
    """Verify v2's declared generated outputs before deriving a new benchmark."""
    generated = Path(generated).resolve(strict=True)
    provenance = json.loads((generated / 'expanded_provenance.json').read_text(encoding='utf8'))
    recorded = provenance.get('expanded')
    front = provenance.get('frontier_v2')
    if not isinstance(recorded, dict) or not recorded or not isinstance(front, dict):
        raise ValueError('frontier v2 provenance is required')
    required = {'expanded_benchmark.cpp', 'include/candidate.hpp', 'include/trace.hpp'}
    if not required <= set(recorded):
        raise ValueError('frontier provenance lacks required generated files')
    for rel, digest in recorded.items():
        target = (generated / rel).resolve(strict=True)
        if not target.is_relative_to(generated) or not target.is_file() or sha(target) != digest:
            raise ValueError('frontier generated source hash mismatch: ' + str(rel))
    source_dir = STUDY / 'frontier_search'
    for field, name in (('generator_sha256', 'prepare_headers.py'), ('candidate_sha256', 'candidate.hpp'),
                        ('rejection_instrumentation_sha256', 'rejection_instrumentation.py'),
                        ('rejection_header_sha256', 'rejection.hpp')):
        if sha(source_dir / name) != front.get(field):
            raise ValueError('frontier generator source hash mismatch: ' + name)
    if sha(generated / 'include/rejection.hpp') != front.get('generated_rejection_header_sha256'):
        raise ValueError('frontier rejection header hash mismatch')
    return provenance


def load_rotation_v2():
    # Resolve the old module without allowing an unrelated preloaded module to
    # supply its `prepare_rotation` dependency.
    old_name = 'prepare_rotation'
    prior = sys.modules.pop(old_name, None)
    sys.path.insert(0, str(OLD))
    try:
        spec = importlib.util.spec_from_file_location('frontier_rotation_legacy_v2', OLD / 'prepare_rotation_v2.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(old_name, None)
        if prior is not None:
            sys.modules[old_name] = prior
    if Path(module.v1.__file__).resolve() != (OLD / 'prepare_rotation.py').resolve():
        raise ValueError('unexpected legacy rotation generator')
    return module


def prepare(generated, output, original, instrumentation):
    generated = Path(generated).resolve(strict=True)
    original = Path(original).resolve(strict=True)
    instrumentation = Path(instrumentation).resolve(strict=True)
    output = Path(output).resolve()
    inputs = (generated, original, instrumentation)
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in inputs):
        raise ValueError('output must be disjoint from every input tree')
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for index, a in enumerate(inputs) for b in inputs[index + 1:]):
        raise ValueError('generated, original and instrumentation trees must be disjoint')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be new or empty')
    if not (original / 'vkFFT.h').is_file() or not (instrumentation / 'cpu_reference.hpp').is_file():
        raise ValueError('original VkFFT and instrumentation headers are required')
    frontier = verify_frontier(generated)
    before = {name: tree_inventory(path) for name, path in
              (('frontier_generated', generated), ('original_vkfft', original), ('instrumentation', instrumentation))}
    legacy = load_rotation_v2()
    provenance = legacy.prepare(generated, output)
    after = {name: tree_inventory(path) for name, path in
             (('frontier_generated', generated), ('original_vkfft', original), ('instrumentation', instrumentation))}
    if before != after:
        raise RuntimeError('an input tree changed during generation')
    cmake = (output / 'CMakeLists.txt').read_text(encoding='utf8')
    original_anchor = 'target_include_directories(${target} PRIVATE "${VKFFT_INCLUDE_ROOT}")'
    generated_anchor = 'target_include_directories(${target} PRIVATE "${ROTATION_GENERATED_ROOT}" "${EXPANDED_GENERATED_ROOT}/include"'
    if original_anchor not in cmake or generated_anchor not in cmake or cmake.index(original_anchor) >= cmake.index(generated_anchor):
        raise ValueError('rotation CMake original include order is not established')
    outputs = {path.name: sha(path) for path in sorted(output.iterdir())
               if path.is_file() and path.name != 'rotation_provenance.json'}
    provenance['frontier_rotation'] = {
        'schema': 'frontier-rotation-generated/v1',
        'device_validation': 'not performed',
        'generator_sha256': sha(Path(__file__)),
        'frontier_upstream_commit': frontier.get('upstream_commit'),
        'legacy_sources': {name: sha(OLD / name) for name in
                           ('prepare_rotation.py', 'prepare_rotation_v2.py', 'rotation_pool.hpp',
                            'rotation_cuda_compat.hpp', 'CMakeLists_rotation.txt')},
        'inputs': before,
        'output_files': outputs,
        'input_tree_hash_encoding': 'sha256 of UTF-8 compact sorted JSON mapping relative file paths to SHA-256',
        'original_include_precedence': 'CMake lists original VkFFT before generated include for original target; audit compile dependencies',
    }
    (output / 'rotation_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf8')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded-generated-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--original-include-root', type=Path, default=REPO / 'VkFFT/vkFFT')
    parser.add_argument('--instrumentation-root', type=Path, default=STUDY / 'instrumentation')
    args = parser.parse_args()
    doc = prepare(args.expanded_generated_root, args.output, args.original_include_root, args.instrumentation_root)
    print(json.dumps({'schema': doc['frontier_rotation']['schema'],
                      'rotation_provenance_sha256': sha(args.output / 'rotation_provenance.json'),
                      'output_files': doc['frontier_rotation']['output_files']}, sort_keys=True))


if __name__ == '__main__':
    main()
