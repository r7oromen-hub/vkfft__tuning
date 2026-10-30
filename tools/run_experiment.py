"""Commit a run declaration, execute a command, then commit its evidence index."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run(root, experiment_id, protocol, command):
    root = Path(root).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', experiment_id):
        raise ValueError('invalid experiment ID')
    if not command:
        raise ValueError('missing command')
    protocol = Path(protocol).resolve()
    rel = protocol.relative_to(root).as_posix()
    git(root, 'ls-files', '--error-unmatch', rel)
    if git(root, 'status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none'):
        raise ValueError('commit all changes before running an experiment')
    submodules = git(root, 'submodule', 'status', '--recursive')
    if any(line.startswith(('-', '+', 'U')) for line in submodules.splitlines()):
        raise ValueError('submodule not checked out at committed revision')
    record = root / 'experiments' / 'records' / (experiment_id + '.json')
    artifacts = root / 'experiments' / 'artifacts' / experiment_id
    if record.exists() or artifacts.exists():
        raise ValueError('experiment ID already exists')
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {'schema': 'git-experiment/v1', 'id': experiment_id, 'status': 'declared',
            'declared_at': utc(), 'parent_code_commit': git(root, 'rev-parse', 'HEAD'),
            'protocol': {'path': rel, **digest(protocol)}, 'command': command,
            'submodules': submodules, 'execution_commit': None,
            'remote_source_verification': 'If command dispatches remotely, verify and record remote checkout separately.'}
    record_rel = record.relative_to(root).as_posix()
    save(record, data)
    git(root, 'add', '--', record_rel)
    git(root, 'commit', '-m', f'experiment: declare {experiment_id}', '--', record_rel)
    execution_commit = git(root, 'rev-parse', 'HEAD')
    artifacts.mkdir(parents=True)
    env = dict(os.environ, EXPERIMENT_ID=experiment_id,
               EXPERIMENT_CODE_COMMIT=execution_commit, EXPERIMENT_ARTIFACT_DIR=str(artifacts))
    data.update(execution_commit=execution_commit, started_at=utc())
    save(artifacts / 'execution.json', data)
    # The tracked declaration remains unchanged while the child runs.
    try:
        with (artifacts / 'stdout.log').open('wb') as out, (artifacts / 'stderr.log').open('wb') as err:
            rc = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err).returncode
        data.update(returncode=rc, status='completed' if rc == 0 else 'failed')
    except (OSError, KeyboardInterrupt) as e:
        rc = 130 if isinstance(e, KeyboardInterrupt) else 127
        data.update(returncode=rc, status='interrupted' if rc == 130 else 'launch_failed', error=type(e).__name__)
    dirty = git(root, 'status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none')
    if dirty:
        data['status'] = 'source_changed_during_run'
        data['worktree_changes'] = dirty
        rc = rc or 2
    data.update(finished_at=utc(), artifacts=[{'path': p.relative_to(root).as_posix(), **digest(p)}
                for p in sorted(artifacts.rglob('*')) if p.is_file() and not p.is_symlink()])
    save(record, data)
    git(root, 'add', '--', record_rel)
    git(root, 'commit', '-m', f'experiment: record {experiment_id} ({data["status"]})', '--', record_rel)
    print(json.dumps({'record': record_rel, 'execution_commit': execution_commit,
                      'result_commit': git(root, 'rev-parse', 'HEAD'), 'status': data['status']}))
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    repo = git(Path.cwd(), 'rev-parse', '--show-toplevel')
    sys.exit(run(repo, args.id, args.protocol, command))
