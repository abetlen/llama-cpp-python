from __future__ import annotations

import ctypes
import functools

from typing import Any, Iterable, Union

from . import llama_cpp

_lib = llama_cpp._lib


def _ctypes_function_from_names(
    names: Iterable[str],
    argtypes: list[Any],
    restype: Any,
):
    """Decorator for extension functions whose exported symbol name can vary by ABI."""

    def decorator(f):
        missing: list[str] = []
        for name in names:
            try:
                func = getattr(_lib, name)
            except AttributeError:
                missing.append(name)
                continue
            func.argtypes = argtypes
            func.restype = restype
            functools.wraps(f)(func)
            return func
        raise AttributeError(
            f"None of the shared library symbols were found: {', '.join(missing)}"
        )

    return decorator


# LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);
@_ctypes_function_from_names(
    (
        "llama_set_embeddings_pre_norm",
        "_Z29llama_set_embeddings_pre_normP13llama_contextbb",
        "?llama_set_embeddings_pre_norm@@YAXPEAUllama_context@@_N1@Z",
    ),
    [llama_cpp.llama_context_p_ctypes, ctypes.c_bool, ctypes.c_bool],
    None,
)
def llama_set_embeddings_pre_norm(
    ctx: llama_cpp.llama_context_p,
    value: bool,
    masked: bool,
    /,
):
    """Set whether the context outputs pre-norm embeddings or not."""
    ...


# LLAMA_API float * llama_get_embeddings_pre_norm(struct llama_context * ctx);
@_ctypes_function_from_names(
    (
        "llama_get_embeddings_pre_norm",
        "_Z29llama_get_embeddings_pre_normP13llama_context",
        "?llama_get_embeddings_pre_norm@@YAPEAMPEAUllama_context@@@Z",
    ),
    [llama_cpp.llama_context_p_ctypes],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_pre_norm(
    ctx: llama_cpp.llama_context_p,
    /,
):
    """Get the pre-norm embeddings from the last evaluation."""
    ...


# LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);
@_ctypes_function_from_names(
    (
        "llama_get_embeddings_pre_norm_ith",
        "_Z33llama_get_embeddings_pre_norm_ithP13llama_contexti",
        "?llama_get_embeddings_pre_norm_ith@@YAPEAMPEAUllama_context@@H@Z",
    ),
    [llama_cpp.llama_context_p_ctypes, ctypes.c_int32],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_pre_norm_ith(
    ctx: llama_cpp.llama_context_p,
    i: Union[ctypes.c_int32, int],
    /,
):
    """Get the pre-norm embeddings for the ith output row from the last evaluation."""
    ...
