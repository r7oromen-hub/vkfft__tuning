#!/usr/bin/env python3
"""Reuse the joint GPU guard for this directory's describe controller only."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'controller'))
import joint_gpu_guard as common
import describe_controller as controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--incidents', required=True, type=Path)
    parser.add_argument('--idle-seconds', type=float, default=30)
    guard_args, remainder = parser.parse_known_args()
    args = controller.parser().parse_args(remainder)
    if os.name != 'posix' or not Path('/proc/self/stat').is_file():
        parser.error('guard requires Linux /proc')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        parser.error('guard requires Linux pidfd signaling')
    if not args.gpu.startswith('GPU-') or ',' in args.gpu:
        parser.error('--gpu must be one full GPU UUID')
    if not math.isfinite(guard_args.idle_seconds) or not 0 <= guard_args.idle_seconds <= 3600:
        parser.error('invalid idle-seconds')
    if guard_args.log.resolve() == guard_args.incidents.resolve():
        parser.error('log and incidents must differ')
    for path in (guard_args.log, guard_args.incidents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    guard_args.gpu = args.gpu
    guard = common.Guard(guard_args)
    original_popen, original_run_one = subprocess.Popen, controller.run_one
    originals = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                          VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(guard_args.log.resolve()))
        guard.emit('guard_start', script=str(HERE / 'describe_controller.py'), idle_seconds=guard_args.idle_seconds,
                   memory_limit_mib=common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                   parent_pid=os.getpid(), script_args=remainder,
                   own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
        guard.idle_check()
        subprocess.Popen = guard.popen(original_popen)
        controller.run_one = guard.wrapped(original_run_one)
        guard.start()
        controller.collect(args)
        guard.sample('final'); guard.check(); guard.close(); guard.check()
        if guard.thread is not None and guard.thread.is_alive():
            raise common.GuardAbort('GPU monitor did not stop within bounded shutdown timeout')
        guard.emit('guard_complete', success=True, clean=True, script_exit_code=0, sample_count=guard.sample_count)
        return 0
    except BaseException as error:
        if not guard.abort.is_set():
            guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
        guard.close()
        guard.emit('guard_abort', incident=True, success=False, clean=False,
                   error=f'{type(error).__name__}: {error}', sample_count=guard.sample_count)
        print(f'GPU guard stopped its own experiment: {error}', file=sys.stderr)
        return 3
    finally:
        guard.close()
        subprocess.Popen, controller.run_one = original_popen, original_run_one
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    sys.exit(main())
