"""Experimental bindings for non-public llama.cpp APIs from `llama-ext.h`.

This module is not part of the stable llama-cpp-python public API.
Downstream code should not import or depend on it directly.
"""

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


# LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
@_ctypes_function_from_names(
    (
        "llama_set_embeddings_nextn",
        "_Z26llama_set_embeddings_nextnP13llama_contextbb",
        "?llama_set_embeddings_nextn@@YAXPEAUllama_context@@_N1@Z",
    ),
    [llama_cpp.llama_context_p_ctypes, ctypes.c_bool, ctypes.c_bool],
    None,
)
def llama_set_embeddings_nextn(
    ctx: llama_cpp.llama_context_p,
    value: bool,
    masked: bool,
    /,
):
    """Set whether the context outputs nextn embeddings or not."""
    ...


# LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);
@_ctypes_function_from_names(
    (
        "llama_get_embeddings_nextn",
        "_Z26llama_get_embeddings_nextnP13llama_context",
        "?llama_get_embeddings_nextn@@YAPEAMPEAUllama_context@@@Z",
    ),
    [llama_cpp.llama_context_p_ctypes],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_nextn(
    ctx: llama_cpp.llama_context_p,
    /,
):
    """Get the nextn embeddings from the last evaluation."""
    ...


# LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);
@_ctypes_function_from_names(
    (
        "llama_get_embeddings_nextn_ith",
        "_Z30llama_get_embeddings_nextn_ithP13llama_contexti",
        "?llama_get_embeddings_nextn_ith@@YAPEAMPEAUllama_context@@H@Z",
    ),
    [llama_cpp.llama_context_p_ctypes, ctypes.c_int32],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_nextn_ith(
    ctx: llama_cpp.llama_context_p,
    i: Union[ctypes.c_int32, int],
    /,
):
    """Get the nextn embeddings for the ith output row from the last evaluation."""
    ...


# LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);
@_ctypes_function_from_names(
    (
        "llama_get_ctx_other",
        "_Z19llama_get_ctx_otherP13llama_context",
        "?llama_get_ctx_other@@YAPEAUllama_context@@PEAU1@@Z",
    ),
    [llama_cpp.llama_context_p_ctypes],
    llama_cpp.llama_context_p_ctypes,
)
def llama_get_ctx_other(
    ctx: llama_cpp.llama_context_p,
    /,
):
    """Get the context linked through llama_context_params.ctx_other."""
    ...
