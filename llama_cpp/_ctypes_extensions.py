from __future__ import annotations

import sys
import os
import ctypes
import functools
import pathlib

from typing import (
    Any,
    Callable,
    List,
    Union,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias


_EMSCRIPTEN_SIDE_MODULE_SUFFIX = ".cpython-00-wasm32-emscripten.so"


# Load the library
def load_shared_library(lib_base_name: str, base_path: pathlib.Path):
    """Platform independent shared library loader"""
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform == "emscripten":
        # Use a CPython-style tag that Pyodide skips during package auto-load.
        lib_paths += [
            base_path / f"lib{lib_base_name}{_EMSCRIPTEN_SIDE_MODULE_SUFFIX}",
        ]
    elif sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_paths += [
            base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        lib_paths += [
            base_path / f"lib{lib_base_name}.so",
            base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        lib_paths += [
            base_path / f"{lib_base_name}.dll",
            base_path / f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    cdll_args = dict()  # type: ignore

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32":
        os.add_dll_directory(str(base_path))
        os.environ["PATH"] = str(base_path) + os.pathsep + os.environ["PATH"]

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))
        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    if sys.platform == "emscripten":
        cdll_args["mode"] = ctypes.RTLD_GLOBAL
        lib_dir = str(base_path)
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in ld_library_path.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                lib_dir
                if not ld_library_path
                else f"{lib_dir}{os.pathsep}{ld_library_path}"
            )

        emscripten_dependencies = {
            "llama": ("ggml-base", "ggml-cpu", "ggml"),
            "mtmd": ("ggml-base", "ggml-cpu", "ggml", "llama"),
        }
        for dependency in emscripten_dependencies.get(lib_base_name, ()):
            dependency_path = (
                base_path / f"lib{dependency}{_EMSCRIPTEN_SIDE_MODULE_SUFFIX}"
            )
            if dependency_path.exists():
                try:
                    ctypes.CDLL(str(dependency_path), **cdll_args)  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load shared library '{dependency_path}': {e}"
                    )

    # Try to load the shared library, handling potential errors
    for lib_path in lib_paths:
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path), **cdll_args)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# ctypes sane type hint helpers
#
# - Generic Pointer and Array types
# - PointerOrRef type with a type hinted byref function
#
# NOTE: Only use these for static type checking not for runtime checks
# no good will come of that

if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[
        CtypesPointer[CtypesCData], CtypesRef[CtypesCData]
    ]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    """Decorator for defining ctypes functions with type hints"""

    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function


def _byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = _byref if TYPE_CHECKING else ctypes.byref
