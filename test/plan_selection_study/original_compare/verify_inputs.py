#!/usr/bin/env python3
"""Verify original headers and the frozen Linux joint-generated harness."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[3]
manifest=json.loads((Path(__file__).parent/'input_manifest.json').read_text(encoding='utf8'))
errors=[]
for relative, expected in sorted(manifest.items()):
    path=root/relative
    if not path.is_file():
        errors.append(f'missing: {relative}')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(f'hash mismatch: {relative}')
if errors:
    print('\n'.join(errors),file=sys.stderr)
    sys.exit(1)
print(f'verified {len(manifest)} original inputs and Linux-generated harness files')
