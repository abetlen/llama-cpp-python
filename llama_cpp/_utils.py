import os
import sys

import sys
from typing import Any, Dict

# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

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

        # Check if sys.stdout and sys.stderr have fileno method
        if not hasattr(self.sys.stdout, 'fileno') or not hasattr(self.sys.stderr, 'fileno'):
            return self  # Return the instance without making changes

        self.old_stdout_fileno_undup = self.sys.stdout.fileno()
        self.old_stderr_fileno_undup = self.sys.stderr.fileno()

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        # Check if sys.stdout and sys.stderr have fileno method
        if hasattr(self.sys.stdout, 'fileno') and hasattr(self.sys.stderr, 'fileno'):
            self.sys.stdout = self.old_stdout
            self.sys.stderr = self.old_stderr

            self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
            self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

            self.os.close(self.old_stdout_fileno)
            self.os.close(self.old_stderr_fileno)


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
