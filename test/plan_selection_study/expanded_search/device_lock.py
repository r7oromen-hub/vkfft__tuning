"""Kernel-owned exclusive lock, inherited by the guarded child.

The lock file is never removed: kernel flock, not its existence, determines
ownership. If the launcher dies, the child still holds the passed descriptor
until it exits; a second launcher cannot overlap it.
"""
import fcntl
import os
from pathlib import Path


class DeviceLock:
    def __init__(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.fd); self.fd = None
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
