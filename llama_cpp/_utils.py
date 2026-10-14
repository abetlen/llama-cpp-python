import os
import sys
import math
import multiprocessing

from typing import Any, Dict, List, Optional

# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

STDOUT_FILENO = 1
STDERR_FILENO = 2


class suppress_stdout_stderr(object):
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, disable: bool = True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        self.old_stdout_fileno_undup = STDOUT_FILENO
        self.old_stderr_fileno_undup = STDERR_FILENO

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        # In Jupyter notebooks, ipykernel replaces sys.stdout/stderr with
        # OutStream objects that hold their own copy of the original fd in
        # _original_stdstream_copy. This bypasses our dup2 redirect, so we
        # need to point that copy at the real fd temporarily.
        # https://github.com/ipython/ipykernel/blob/912923542d55e6c80c0e7c1f94c648b52a225011/ipykernel/iostream.py#L618-L622
        self._saved_stdout_copy = getattr(
            self.sys.stdout, "_original_stdstream_copy", None
        )
        self._saved_stderr_copy = getattr(
            self.sys.stderr, "_original_stdstream_copy", None
        )
        if self._saved_stdout_copy is not None:
            self.sys.stdout._original_stdstream_copy = self.old_stdout_fileno_undup
        if self._saved_stderr_copy is not None:
            self.sys.stderr._original_stdstream_copy = self.old_stderr_fileno_undup

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr

        self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        self.os.close(self.old_stdout_fileno)
        self.os.close(self.old_stderr_fileno)

        # Restore ipykernel's OutStream fd copies
        if self._saved_stdout_copy is not None:
            self.sys.stdout._original_stdstream_copy = self._saved_stdout_copy
        if self._saved_stderr_copy is not None:
            self.sys.stderr._original_stdstream_copy = self._saved_stderr_copy


class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()


def _read_cgroup_cpu_limit(cgroup_dir: str, v2: bool) -> Optional[float]:
    """Return the CPU limit (in CPUs) set on one cgroup directory, or None."""
    try:
        if v2:
            with open(os.path.join(cgroup_dir, "cpu.max")) as f:
                quota, period = f.read().split()[:2]
            if quota == "max":
                return None
        else:
            with open(os.path.join(cgroup_dir, "cpu.cfs_quota_us")) as f:
                quota = f.read().strip()
            with open(os.path.join(cgroup_dir, "cpu.cfs_period_us")) as f:
                period = f.read().strip()
        if int(quota) <= 0 or int(period) <= 0:
            return None
        return int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit(
    proc_cgroup: str = "/proc/self/cgroup",
    cgroup_root: str = "/sys/fs/cgroup",
) -> Optional[float]:
    """Return the tightest CPU quota (in CPUs) on this process's cgroup, or None.

    Reads cgroup v2 `cpu.max` or cgroup v1 `cpu.cfs_quota_us`/`cpu.cfs_period_us`
    for the process's cgroup and its ancestors (e.g. `docker run --cpus` or a
    Kubernetes CPU limit). Returns None when no quota is set or on non-Linux.
    """
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    limits: List[float] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if controllers == "":
            mounts, v2 = [cgroup_root], True
        elif "cpu" in controllers.split(","):
            names = ("cpu", "cpu,cpuacct", "cpuacct,cpu")
            mounts, v2 = [os.path.join(cgroup_root, n) for n in names], False
        else:
            continue
        for mount in mounts:
            # Walk from the process cgroup up to the mount root. The mount root
            # is also the process cgroup inside containers without a cgroup
            # namespace, where `path` does not exist under the mount.
            rel = path.strip("/")
            while True:
                limit = _read_cgroup_cpu_limit(os.path.join(mount, rel), v2)
                if limit is not None:
                    limits.append(limit)
                if not rel:
                    break
                rel = os.path.dirname(rel)
    return min(limits) if limits else None


def default_cpu_count() -> int:
    """Number of CPUs this process can actually use.

    `multiprocessing.cpu_count()` reports every CPU on the host, even when the
    process is pinned to fewer (`taskset`, `docker run --cpuset-cpus`) or
    limited by a cgroup CPU quota (`docker run --cpus`, Kubernetes limits).
    This uses the CPU affinity mask where available and caps it at the
    cgroup quota rounded up.
    """
    try:
        count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        count = multiprocessing.cpu_count()
    limit = _cgroup_cpu_limit()
    if limit is not None:
        count = min(count, math.ceil(limit))
    return max(count, 1)
