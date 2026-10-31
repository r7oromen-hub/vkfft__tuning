"""Re-audit a completed breadth screen and produce CPU-only analysis reports.

Run through tools/run_experiment.py so all outputs are recorded separately from
the immutable source run. This phase does not execute the recorded binaries.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
import frontier_campaign_inputs as campaign_inputs

AUDIT = ROOT / 'test/plan_selection_study/frontier_confirmation/coverage_audit.py'
ANALYSIS = ROOT / 'test/plan_selection_study/frontier_analysis'
LOG_NAMES = ('rows', 'guard', 'incidents', 'budget')


def registered_output(root, protocol, source_run):
    """Require the environment and committed declaration made by run_experiment."""
    root = Path(root).resolve(strict=True)
    run_id = os.environ['EXPERIMENT_ID']
    commit = os.environ['EXPERIMENT_CODE_COMMIT']
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_id)
            or run_id == source_run or not re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise ValueError('invalid or reused postscreen experiment identity')
    expected = root / 'experiments/artifacts' / run_id
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    if output.is_symlink() or expected.is_symlink():
        raise ValueError('postscreen artifact directory cannot be a symlink')
    output = output.resolve(strict=True)
    if output != expected.resolve(strict=True) or not output.is_dir():
        raise ValueError('postscreen output is not its registered artifact directory')
    if campaign_inputs.git(root, 'rev-parse', 'HEAD') != commit:
        raise ValueError('postscreen execution commit does not match HEAD')
    record = campaign_inputs.checked_path(root, root / 'experiments/records' / (run_id + '.json'))
    campaign_inputs.tracked(root, record)
    declaration = json.loads(record.read_text(encoding='utf8'))
    execution = json.loads((output / 'execution.json').read_text(encoding='utf8'))
    execution_declaration = dict(execution)
    started_at = execution_declaration.pop('started_at', None)
    execution_commit = execution_declaration.get('execution_commit')
    execution_declaration['execution_commit'] = None
    info = {'path': Path(protocol).relative_to(root).as_posix(),
            'bytes': Path(protocol).stat().st_size, 'sha256': campaign_inputs.sha(protocol)}
    if (declaration.get('schema') != 'git-experiment/v1' or declaration.get('id') != run_id
            or declaration.get('status') != 'declared' or declaration.get('protocol') != info
            or declaration.get('parent_code_commit') != campaign_inputs.git(root, 'rev-parse', commit + '^')
            or execution_commit != commit or not isinstance(started_at, str) or not started_at
            or execution_declaration != declaration):
        raise ValueError('postscreen declaration or protocol does not match execution')
    return output


def source_inputs(root, context, source_run):
    """Resolve only inventoried source files and bind both sessions to admission."""
    root = Path(root).resolve(strict=True)
    record, source, record_path = campaign_inputs.load_completed_run(root, source_run, context['protocol'])
    sessions = {}
    entries = record.get('artifacts')
    if not isinstance(entries, list):
        raise ValueError('source artifact inventory missing')
    for kind, manifests_dir in (('defaults', 'defaults_manifests'), ('search', 'manifests')):
        paths = {name: source / kind / (name + '.jsonl') for name in LOG_NAMES}
        prefix = (source / manifests_dir).relative_to(root).as_posix() + '/'
        manifests = []
        for entry in entries:
            rel = entry.get('path')
            if isinstance(rel, str) and rel.startswith(prefix):
                name = rel[len(prefix):]
                if not re.fullmatch(r'n[0-9]+_b[0-9]+\.manifest\.jsonl', name):
                    raise ValueError('unexpected source manifest inventory entry')
                manifests.append(root / PurePosixPath(rel))
        if not manifests:
            raise ValueError('source ' + kind + ' manifests missing')
        manifests.sort()
        campaign_inputs.verify_artifacts(root, record, [*paths.values(), *manifests])
        with paths['rows'].open(encoding='utf8') as f:
            first = next((json.loads(line) for line in f if line.strip()), None)
        contract = (first or {}).get('contract') or {}
        admission = context['admission_contract']
        if (first is None or first.get('record_type') != 'session_start'
                or contract.get('protocol') != 'frontier-execution/v1'
                or contract.get('gpu') != context['gpu']
                or contract.get('execution_commit') != record['execution_commit']
                or contract.get('binaries') != admission.get('binaries')
                or contract.get('vkfft_commit') != admission.get('vkfft_commit')
                or (contract.get('generated_provenance') or {}).get('sha256') !=
                   (admission.get('generated_provenance') or {}).get('sha256')):
            raise ValueError('source ' + kind + ' provenance differs from admission or record')
        for item in (context['protocol'], context['gate']):
            if (contract.get('source_files') or {}).get(str(Path(item).resolve())) != campaign_inputs.sha(item):
                raise ValueError('source ' + kind + ' is not bound to protocol and gate')
        sessions[kind] = {'paths': paths, 'manifests': manifests}
    return record, record_path, sessions


def run_step(output, name, command):
    """Capture an exact command and independent logs; stop on any failure."""
    (output / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n', encoding='utf8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with (output / (name + '.stdout.log')).open('xb') as stdout, \
         (output / (name + '.stderr.log')).open('xb') as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / campaign_inputs.DEFAULT_PROTOCOL)
    parser.add_argument('--admission-run', default=campaign_inputs.DEFAULT_ADMISSION_RUN)
    parser.add_argument('--gate', type=Path, default=ROOT / campaign_inputs.DEFAULT_GATE)
    args = parser.parse_args(argv)
    context = campaign_inputs.load_campaign(ROOT, args.protocol, args.admission_run, args.gate)
    output = registered_output(ROOT, context['protocol'], args.source_run)
    record, record_path, sessions = source_inputs(ROOT, context, args.source_run)
    binaries = {'original': context['admission_dir'] / 'build/vkfft_frontier_original',
                'expanded': context['admission_dir'] / 'build/vkfft_frontier_v2'}
    inventory = [path for session in sessions.values()
                 for path in [*session['paths'].values(), *session['manifests']]]
    evidence = {'source_run': args.source_run, 'source_record': str(record_path),
                'source_record_sha256': campaign_inputs.sha(record_path),
                'source_execution_commit': record['execution_commit'],
                'admission_run': args.admission_run, 'gpu': context['gpu'],
                'protocol_sha256': campaign_inputs.sha(context['protocol']),
                'gate_sha256': campaign_inputs.sha(context['gate']),
                'source_artifact_sha256': {str(p): campaign_inputs.sha(p) for p in inventory},
                'binary_sha256': {name: campaign_inputs.sha(path) for name, path in binaries.items()}}
    (output / 'postscreen_inputs.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf8')
    for kind in ('defaults', 'search'):
        session = sessions[kind]
        cmd = [sys.executable, str(AUDIT)]
        for name, path in session['paths'].items(): cmd += ['--' + name, str(path)]
        for name, path in binaries.items(): cmd += ['--' + name, str(path)]
        for path in session['manifests']: cmd += ['--manifest', str(path)]
        cmd += ['--gpu', context['gpu'], '--output', str(output / (kind + '_coverage.json'))]
        run_step(output, kind + '_coverage', cmd)
    coverage = output / 'search_coverage.json'
    paths = sessions['search']['paths']
    for name, script, extra in (
        ('analysis', 'analyze.py', []),
        ('search_costs', 'search_costs.py', ['--budget', str(paths['budget'])]),
        ('aggregate', 'aggregate_confirmed.py', []),
    ):
        cmd = [sys.executable, str(ANALYSIS / script), '--coverage', str(coverage),
               '--rows', str(paths['rows']), *extra, '--output', str(output / (name + '.json'))]
        run_step(output, name, cmd)
    # Recheck the authenticated inputs after all readers have finished.
    campaign_inputs.verify_artifacts(ROOT, record, inventory)
    if (campaign_inputs.sha(record_path) != evidence['source_record_sha256']
            or any(campaign_inputs.sha(path) != evidence['binary_sha256'][name] for name, path in binaries.items())
            or campaign_inputs.sha(context['protocol']) != evidence['protocol_sha256']
            or campaign_inputs.sha(context['gate']) != evidence['gate_sha256']):
        raise ValueError('postscreen input changed during analysis')
    print(json.dumps({'status': 'complete', 'source_run': args.source_run, 'output': str(output)}))


if __name__ == '__main__':
    main()
