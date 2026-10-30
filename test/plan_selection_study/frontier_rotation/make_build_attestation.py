#!/usr/bin/env python3
"""Record a declarative rotation build association for a future guarded run."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-commit', required=True); p.add_argument('--vkfft-commit', required=True)
    p.add_argument('--rotation-root', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True); p.add_argument('--expanded', type=Path, required=True)
    p.add_argument('--build-command', action='append', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if any(not command.strip() for command in a.build_command): p.error('build commands must be nonempty')
    if not all(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
               for commit in (a.code_commit, a.vkfft_commit)):
        p.error('commit arguments must be full lowercase SHA-1 IDs')
    if a.original.resolve(strict=True) == a.expanded.resolve(strict=True): p.error('rotation binaries must differ')
    doc = {'schema': 'frontier-rotation-build/v1', 'code_commit': a.code_commit, 'vkfft_commit': a.vkfft_commit,
           'rotation_provenance_sha256': sha(a.rotation_root / 'rotation_provenance.json'),
           'binaries': {'original': sha(a.original), 'expanded': sha(a.expanded)}, 'build_commands': a.build_command}
    with a.output.open('x', encoding='utf8', newline='\n') as file:
        json.dump(doc, file, indent=2, sort_keys=True); file.write('\n')


if __name__ == '__main__':
    main()
