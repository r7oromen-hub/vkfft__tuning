"""Check schema compatibility against hash-bound old default execution records.

This is offline validation of the new checker, not nonuniform GPU admission.
"""
import argparse
import hashlib
import json
from pathlib import Path

import identity
import validation


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    record = json.loads(args.record.read_text(encoding='utf8'))
    relative = args.rows.resolve().relative_to(root).as_posix()
    entries = [item for item in record['artifacts'] if item['path'] == relative]
    if (record['status'] != 'completed' or len(entries) != 1 or
            entries[0]['bytes'] != args.rows.stat().st_size or
            entries[0]['sha256'] != digest(args.rows)):
        raise ValueError('archived input does not match completed source inventory')
    report = {'scope': 'old default records only; no nonuniform GPU evidence',
              'record_sha256': digest(args.record), 'rows_sha256': entries[0]['sha256'],
              'checked': 0, 'skipped': 0, 'failures': [], 'representability': []}
    with args.rows.open(encoding='utf8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if (row.get('record_type') != 'execution' or row.get('direct_environment') != {} or
                    row.get('result', {}).get('study_mode') != 'execute'):
                report['skipped'] += 1
                continue
            result = row['result']
            report['checked'] += 1
            try:
                validation.check(row, row['guard_gpu_uuid'],
                                 original=result['header_instrumentation'] is False)
                if result['header_instrumentation'] is True:
                    request, reason = identity.representable(result)
                    report['representability'].append({'n': result['n'], 'batch': result['batch'],
                                                       'supported': request is not None, 'reason': reason})
            except (ValueError, KeyError, TypeError) as error:
                report['failures'].append({'line': line_number, 'n': result.get('n'),
                                           'batch': result.get('batch'), 'error': str(error)})
    report['status'] = 'pass' if report['checked'] > 0 and not report['failures'] else 'fail'
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'checked': report['checked'],
                      'failures': len(report['failures'])}))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
