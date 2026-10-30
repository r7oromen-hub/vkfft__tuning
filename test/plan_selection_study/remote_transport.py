"""Strict-key SSH/SFTP JSON transport. Passwords are accepted only from a TTY.

Start with --host HOST --port PORT --user USER --known-hosts FILE. After the
password prompt, send one JSON request per line: run(command, timeout),
put(local, remote), get(remote, local), or close. No credentials are persisted.
"""
import argparse
import getpass
import json
from pathlib import Path
import sys
import time
import warnings

import paramiko


def run(client, command, timeout):
    if not isinstance(command,str) or not command:
        raise ValueError('run requires a nonempty command string')
    if not isinstance(timeout,(int,float)) or not 0 < timeout <= 86400:
        raise ValueError('timeout must be positive and at most 86400 seconds')
    _,stdout,stderr = client.exec_command(command,timeout=timeout)
    channel = stdout.channel
    out,err = bytearray(),bytearray()
    deadline = time.monotonic()+timeout
    try:
        while True:
            while channel.recv_ready():
                out.extend(channel.recv(65536))
            while channel.recv_stderr_ready():
                err.extend(channel.recv_stderr(65536))
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('remote command exceeded its requested timeout')
            time.sleep(.01)
        return dict(stdout=out.decode('utf-8','replace'),stderr=err.decode('utf-8','replace'),
                    exit_code=channel.recv_exit_status())
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,default=22)
    parser.add_argument('--user',required=True)
    parser.add_argument('--known-hosts',type=Path,required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    if not sys.stdin.isatty():
        parser.error('password authentication requires a TTY; do not pipe credentials')
    known_hosts = args.known_hosts.resolve(strict=True)
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    key_name = args.host if args.port == 22 else f'[{args.host}]:{args.port}'
    if client.get_host_keys().lookup(key_name) is None:
        parser.error('host/port has no pinned key in the specified known-hosts file')
    with warnings.catch_warnings():
        warnings.simplefilter('error',getpass.GetPassWarning)
        password = getpass.getpass('SSH password: ')
    try:
        client.connect(args.host,port=args.port,username=args.user,password=password,
                       allow_agent=False,look_for_keys=False,timeout=20,auth_timeout=20,banner_timeout=20)
    except BaseException:
        client.close()
        raise
    finally:
        password = None
    print(json.dumps(dict(connected=args.host,port=args.port,user=args.user,known_hosts=str(known_hosts))),flush=True)
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                op = req['op']
                if op == 'close':
                    print(json.dumps(dict(closed=True)),flush=True)
                    break
                if op == 'run':
                    result = run(client,req['command'],req.get('timeout',120))
                elif op in ('put','get'):
                    with client.open_sftp() as sftp:
                        if op == 'put':
                            sftp.put(req['local'],req['remote'])
                        else:
                            Path(req['local']).parent.mkdir(parents=True,exist_ok=True)
                            sftp.get(req['remote'],req['local'])
                    result = dict(ok=True,op=op,local=req['local'],remote=req['remote'])
                else:
                    raise ValueError('unknown operation')
                print(json.dumps(result,ensure_ascii=True),flush=True)
            except Exception as exc:
                print(json.dumps(dict(error=str(exc)),ensure_ascii=True),flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
