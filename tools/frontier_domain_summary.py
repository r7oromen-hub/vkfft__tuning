"""Count the declared request domain; no claim of device validity or performance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'test/plan_selection_study/frontier_search/manifest.py'


def main():
    spec = importlib.util.spec_from_file_location('frontier_manifest', GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    rows = []
    for n in (128, 256, 512, 1024, 2048, 4096, 8192):
        radix = list(generator.prototype.decompositions(n))
        exponent = n.bit_length() - 1
        counts = [1] + [0] * exponent
        for i in range(1, exponent + 1):
            counts[i] = sum(counts[i-j] for j in range(1, min(i, 5)+1))
        if len(radix) != counts[-1] or len(set(radix)) != len(radix):
            raise ValueError('radix enumeration differs from independent composition count')
        for batch in (1, 16, 256, 4096):
            groups = batch.bit_length()
            old = sum(max(d) <= r <= n for d in radix for r in (2,4,8,16,32,64)) * groups
            new = sum(max(d) <= r <= n for d in radix for r in generator.REGISTERS) * groups
            rows.append({'n': n, 'batch': batch, 'ordered_radix_sequences': len(radix),
                         'group_options': groups, 'old_static_eligible': old,
                         'expanded_static_eligible': new, 'additional_static_eligible': new-old})
    result = {'schema': 'frontier-domain-summary/v1',
              'execution_commit': os.environ['EXPERIMENT_CODE_COMMIT'],
              'generator_sha256': hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
              'scope': 'power-of-two core matrix; ordered radix 2/4/8/16/32; uniform R through 256; boost1',
              'interpretation': 'Static eligibility only, not scheduler validity, actual distinct kernels, or measured coverage.',
              'rows': rows,
              'total_old_static_eligible': sum(r['old_static_eligible'] for r in rows),
              'total_expanded_static_eligible': sum(r['expanded_static_eligible'] for r in rows)}
    target = Path(os.environ['EXPERIMENT_ARTIFACT_DIR']) / 'domain_summary.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
