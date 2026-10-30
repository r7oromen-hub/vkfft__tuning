"""Wrap only expanded_controller with the proven pidfd GPU guard machinery."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'original_compare'))
import original_compare_guard as common
sys.path.insert(0, str(HERE))
import expanded_controller as controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--incidents', required=True, type=Path)
    parser.add_argument('--idle-seconds', type=float, default=30)
    args, remainder = parser.parse_known_args()
    if os.name != 'posix' or not Path('/proc/self/stat').is_file() or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        parser.error('Linux /proc and pidfd signaling required')
    if not args.gpu.startswith('GPU-') or ',' in args.gpu or not math.isfinite(args.idle_seconds) or not 0 <= args.idle_seconds <= 3600:
        parser.error('invalid GPU UUID or idle duration')
    if args.log.resolve() == args.incidents.resolve():
        parser.error('log and incident files must differ')
    for path in (args.log, args.incidents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    guard = common.Guard(args)
    old_popen, old_run_one = subprocess.Popen, controller.suite.run_one
    keys = ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                          VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
        guard.emit('guard_start', script=str(HERE / 'expanded_controller.py'), idle_seconds=args.idle_seconds,
                   memory_limit_mib=common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                   parent_pid=os.getpid(), script_args=remainder,
                   own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
        guard.idle_check()
        subprocess.Popen = guard.popen(old_popen)
        controller.suite.run_one = guard.wrapped(old_run_one)
        guard.start()
        controller.main(['--gpu', args.gpu, *remainder])
        guard.sample('final'); guard.check(); guard.close(); guard.check()
        if guard.thread is not None and guard.thread.is_alive():
            raise common.GuardAbort('monitor failed to stop')
        guard.emit('guard_complete', success=True, clean=True, script_exit_code=0, sample_count=guard.sample_count)
        return 0
    except BaseException as error:
        if not guard.abort.is_set():
            guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
        guard.close()
        guard.emit('guard_abort', incident=True, success=False, clean=False,
                   error=f'{type(error).__name__}: {error}', sample_count=guard.sample_count)
        print(f'GPU guard stopped experiment: {error}', file=sys.stderr)
        return 3
    finally:
        guard.close()
        subprocess.Popen, controller.suite.run_one = old_popen, old_run_one
        for key, value in saved.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


if __name__ == '__main__':
    sys.exit(main())
