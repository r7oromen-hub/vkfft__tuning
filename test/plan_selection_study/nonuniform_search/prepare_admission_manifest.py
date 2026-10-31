"""Materialize only the preregistered CPU contract cases; never launch a device."""
import json
import os
from pathlib import Path

import admission
import candidate


def main():
    root = Path(__file__).resolve().parents[3]
    contract_path = root / 'experiments/research/nonuniform_register_admission_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf8'))
    cases = contract['proposed_admission_matrix']['nonuniform_cases']
    entries = []
    for case in cases:
        for radix in case['radix_orders']:
            for group in case['groups']:
                table = [0] * 33
                for index, registers in case['active_table'].items():
                    table[int(index)] = registers
                request = {'schema': candidate.SCHEMA,
                           'workload': {'n': case['n'], 'batch': case['batch']},
                           'radix': radix, 'register_table': table,
                           'profile': 'nonuniform_active_only', 'grouped_batch': group}
                actual = candidate.validate_request(request)
                if actual != {'min_registers': case['expected_min'],
                              'max_registers': case['expected_max'],
                              'planned_fft_threads': case['planned_fft_threads']}:
                    raise ValueError('derived planning constraints differ from registered contract')
                entries.append({'candidate_id': candidate.candidate_id(request), 'request': request})
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR']) / 'manifest.json'
    with output.open('x', encoding='utf8') as stream:
        json.dump({'schema': 'nonuniform-admission-manifest/v1', 'candidates': entries}, stream, indent=2)
        stream.write('\n')
    if admission.load_manifest(output) != entries:
        raise ValueError('controller manifest roundtrip mismatch')
    print(json.dumps({'candidate_count': len(entries), 'status': 'static_manifest_validated',
                      'gpu_admitted': False, 'performance_measured': False}))


if __name__ == '__main__':
    main()
