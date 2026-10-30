"""Guard a single registered search or confirmation controller session."""
import argparse
import importlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'original_compare'))
import original_compare_guard as common
sys.path.insert(0,str(HERE))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=('search','confirmation'))
    p.add_argument('--gpu',required=True);p.add_argument('--log',required=True,type=Path)
    p.add_argument('--incidents',required=True,type=Path);p.add_argument('--idle-seconds',type=float,default=30)
    a,remainder=p.parse_known_args()
    if os.name!='posix' or not Path('/proc/self/stat').is_file() or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        p.error('Linux /proc and pidfd signaling required')
    if not a.gpu.startswith('GPU-') or ',' in a.gpu or not math.isfinite(a.idle_seconds) or not 0<=a.idle_seconds<=3600:
        p.error('invalid GPU or idle duration')
    if a.log.resolve()==a.incidents.resolve():p.error('log and incidents must differ')
    for path in (a.log,a.incidents):path.parent.mkdir(parents=True,exist_ok=True);path.touch(exist_ok=True)
    controller=importlib.import_module(a.kind+'_controller')
    guard=common.Guard(a);old_popen,old_run= subprocess.Popen,controller.suite.run_one
    keys=('CUDA_VISIBLE_DEVICES','VKFFT_STUDY_GUARD_ID','VKFFT_STUDY_GUARD_UUID','VKFFT_STUDY_GUARD_LOG')
    saved={key:os.environ.get(key) for key in keys}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,VKFFT_STUDY_GUARD_ID=guard.guard_id,
                          VKFFT_STUDY_GUARD_UUID=a.gpu,VKFFT_STUDY_GUARD_LOG=str(a.log.resolve()))
        guard.emit('guard_start',script=str(HERE/(a.kind+'_controller.py')),idle_seconds=a.idle_seconds,
                   memory_limit_mib=common.MEMORY_LIMIT_MIB,monitor_interval_seconds=1,parent_pid=os.getpid(),
                   script_args=remainder,own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
        guard.idle_check();subprocess.Popen=guard.popen(old_popen);controller.suite.run_one=guard.wrapped(old_run)
        guard.start();controller.main(['--gpu',a.gpu,*remainder])
        guard.sample('final');guard.check();guard.close();guard.check()
        if guard.thread is not None and guard.thread.is_alive():raise common.GuardAbort('monitor failed to stop')
        guard.emit('guard_complete',success=True,clean=True,script_exit_code=0,sample_count=guard.sample_count);return 0
    except BaseException as error:
        if not guard.abort.is_set():guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
        guard.close();guard.emit('guard_abort',incident=True,success=False,clean=False,error=f'{type(error).__name__}: {error}',sample_count=guard.sample_count)
        print(f'GPU guard stopped experiment: {error}',file=sys.stderr);return 3
    finally:
        guard.close();subprocess.Popen,controller.suite.run_one=old_popen,old_run
        for key,value in saved.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


if __name__=='__main__':sys.exit(main())
