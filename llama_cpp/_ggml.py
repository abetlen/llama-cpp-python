"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""

import ctypes
import os
import pathlib
from typing import NewType, Optional

import llama_cpp._ctypes_extensions as ctypes_ext

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)


ctypes_function = ctypes_ext.ctypes_function_for_shared_library(libggml)

# Opaque native registry handle, owned by GGML rather than Python.
ggml_backend_reg_t = NewType("ggml_backend_reg_t", int)
ggml_backend_reg_t_ctypes = ctypes.c_void_p


@ctypes_function("ggml_backend_load", [ctypes.c_char_p], ggml_backend_reg_t_ctypes)
def ggml_backend_load(path: bytes, /) -> Optional[ggml_backend_reg_t]:
    """Load and register a backend module; return None if loading fails.

    Pass a filesystem-encoded path (for example, os.fsencode(path)).
    Keep the returned handle alive until all models and backend resources close.
    """
    ...


@ctypes_function("ggml_backend_unload", [ggml_backend_reg_t_ctypes], None)
def ggml_backend_unload(reg: ggml_backend_reg_t, /) -> None:
    """Unload a handle returned by ggml_backend_load after its resources close.

    Do not reuse the handle after unloading, or unload a backend still in use.
    """
    ...


@ctypes_function("ggml_backend_load_all", [], None)
def ggml_backend_load_all() -> None:
    """Register backends using GGML's native default search paths."""
    ...


@ctypes_function("ggml_backend_load_all_from_path", [ctypes.c_char_p], None)
def ggml_backend_load_all_from_path(dir_path: Optional[bytes], /) -> None:
    """Register backends from a filesystem-encoded directory.

    None requests GGML's native default search paths instead.
    """
    ...
