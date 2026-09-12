import ctypes
import os
import subprocess
import sys

if len(sys.argv) == 1:
    for iteration in range(5):
        print(f"Captured test iteration {iteration + 1}", flush=True)
        result = subprocess.run([sys.executable, __file__, "once"])
        if result.returncode:
            raise SystemExit(result.returncode)
    raise SystemExit(0)

import llama_cpp.llama_cpp as lc
import pytest

stderr = os.dup(2)


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def report_abort(message):
    os.write(stderr, b"NATIVE ABORT: " + message + b"\n")


lc._lib.ggml_set_abort_callback.argtypes = [type(report_abort)]
lc._lib.ggml_set_abort_callback.restype = ctypes.c_void_p
lc._lib.ggml_set_abort_callback(report_abort)
raise SystemExit(pytest.main(["-q", "-x"]))
