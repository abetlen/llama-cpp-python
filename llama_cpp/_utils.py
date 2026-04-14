import os
import sys

from typing import Any, Dict

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
