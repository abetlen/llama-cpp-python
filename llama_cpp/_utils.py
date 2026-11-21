import os
import sys
from contextlib import ExitStack
from typing import Any, Dict

STDOUT_FILENO = 1
STDERR_FILENO = 2


class suppress_stdout_stderr(object):
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os
    _open = staticmethod(open)

    def __init__(self, disable: bool = True):
        self.disable = disable

    def __enter__(self):
        if self.disable:
            return self

        with ExitStack() as redirects:
            # An explicit encoding also works when called during interpreter shutdown.
            null_stream = redirects.enter_context(
                self._open(self.os.devnull, "w", encoding="utf-8")
            )
            for name, fd in (("stdout", STDOUT_FILENO), ("stderr", STDERR_FILENO)):
                original_fd = self.os.dup(fd)
                redirects.callback(self.os.close, original_fd)
                self.os.dup2(null_stream.fileno(), fd)
                redirects.callback(self.os.dup2, original_fd, fd)

                stream = getattr(self.sys, name)
                # ipykernel's OutStream also writes through its own saved fd.
                saved_copy = getattr(stream, "_original_stdstream_copy", None)
                if saved_copy is not None:
                    stream._original_stdstream_copy = fd
                    redirects.callback(
                        setattr, stream, "_original_stdstream_copy", saved_copy
                    )
                setattr(self.sys, name, null_stream)
                redirects.callback(setattr, self.sys, name, stream)
            self._redirects = redirects.pop_all()
        return self

    def __exit__(self, *_):
        if not self.disable:
            self._redirects.close()


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
