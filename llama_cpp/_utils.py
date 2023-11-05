import os
import sys


class suppress_stdout_stderr(object):
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    open = open
    sys = sys
    os = os

    def __init__(self, disable: bool = True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        self.outnull_file = self.open(self.os.devnull, "w")
        self.errnull_file = self.open(self.os.devnull, "w")

        self.old_stdout_fileno_undup = self.sys.stdout.fileno()
        self.old_stderr_fileno_undup = self.sys.stderr.fileno()

        self.old_stdout_fileno = self.os.dup(self.sys.stdout.fileno())
        self.old_stderr_fileno = self.os.dup(self.sys.stderr.fileno())

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = self.outnull_file
        self.sys.stderr = self.errnull_file
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

        self.outnull_file.close()
        self.errnull_file.close()
