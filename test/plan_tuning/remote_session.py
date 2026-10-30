"""Ephemeral authenticated SSH transport; credentials are read only from the TTY."""
import argparse
import getpass
import json
from pathlib import Path
import sys

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh' / 'known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('SSH password: ')
    client.connect(args.host, username=args.user, password=password,
                   allow_agent=False, look_for_keys=False,
                   timeout=20, auth_timeout=20, banner_timeout=20)
    password = None
    print(json.dumps({'connected': args.host}), flush=True)
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                op = req['op']
                if op == 'close':
                    break
                if op == 'run':
                    _, stdout, stderr = client.exec_command(req['command'], timeout=req.get('timeout', 120))
                    result = {'stdout': stdout.read().decode('utf-8', 'replace'),
                              'stderr': stderr.read().decode('utf-8', 'replace'),
                              'exit_code': stdout.channel.recv_exit_status()}
                elif op in ('put', 'get'):
                    with client.open_sftp() as sftp:
                        if op == 'put':
                            sftp.put(req['local'], req['remote'])
                        else:
                            Path(req['local']).parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(req['remote'], req['local'])
                    result = {'ok': True, 'op': op, 'local': req['local'], 'remote': req['remote']}
                else:
                    raise ValueError('Unknown operation')
                print(json.dumps(result, ensure_ascii=True), flush=True)
            except Exception as exc:
                print(json.dumps({'error': str(exc)}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
