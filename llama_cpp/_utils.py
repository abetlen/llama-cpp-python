import os
import sys

import sys
from typing import Any, Dict

class NullDevice():
    def write(self, s):
        pass

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
        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.sys.stdout = NullDevice()
        self.sys.stderr = NullDevice()
        return self

    def __exit__(self, *_):
        if self.disable:
            return
        
        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr


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
