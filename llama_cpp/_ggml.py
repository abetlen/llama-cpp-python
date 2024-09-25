"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import os
import pathlib

import ctypes

import llama_cpp._ctypes_extensions as ctypes_ext

import numpy as np


libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

ggml_function = ctypes_ext.ctypes_function_for_shared_library(libggml)


# define GGML_MAX_DIMS           4
GGML_MAX_DIMS = 4

# define GGML_MAX_OP_PARAMS      64
GGML_MAX_OP_PARAMS = 64

# define GGML_MAX_SRC            10
GGML_MAX_SRC = 10

# define GGML_MAX_NAME           64
GGML_MAX_NAME = 64


# // n-dimensional tensor
# struct ggml_tensor {
#     enum ggml_type         type;
#
#     GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor");
#
#     struct ggml_backend_buffer * buffer;
#
#     int64_t ne[GGML_MAX_DIMS]; // number of elements
#     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
#                                // nb[0] = ggml_type_size(type)
#                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
#                                // nb[i] = nb[i-1] * ne[i-1]
#
#     // compute data
#     enum ggml_op op;
#
#     // op params - allocated as int32_t for alignment
#     int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
#
#     int32_t flags;
#
#     struct ggml_tensor * grad;
#     struct ggml_tensor * src[GGML_MAX_SRC];
#
#     // source tensor and offset for views
#     struct ggml_tensor * view_src;
#     size_t               view_offs;
#
#     void * data;
#
#     char name[GGML_MAX_NAME];
#
#     void * extra; // extra things e.g. for ggml-cuda.cu
#
#     // char padding[4];
# };
class ggml_tensor(ctypes.Structure):
    __fields__ = [
        ("type", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("ne", ctypes.c_int64 * 8),
        ("nb", ctypes.c_size_t * 8),
        ("op", ctypes.c_int),
        ("op_params", ctypes.c_int32 * 8),
        ("flags", ctypes.c_int32),
        ("grad", ctypes.c_void_p),
        ("src", ctypes.c_void_p * 8),
        ("view_src", ctypes.c_void_p),
        ("view_offs", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("name", ctypes.c_char * 64),
        ("extra", ctypes.c_void_p),
    ]


ggml_tensor_p = ctypes_ext.CtypesPointer[ggml_tensor]
ggml_tensor_p_ctypes = ctypes.POINTER(ggml_tensor)


# GGML_API GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_get",
    [ggml_tensor_p_ctypes, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
    ctypes.c_void_p,
)
def ggml_backend_tensor_get(
    tensor: ggml_tensor_p, data: ctypes.c_void_p, offset: int, size: int
) -> None:
    ...


# GGML_API GGML_CALL size_t  ggml_nbytes      (const struct ggml_tensor * tensor);
@ggml_function(
    "ggml_nbytes",
    [ggml_tensor_p_ctypes],
    ctypes.c_size_t,
)
def ggml_nbytes(tensor: ggml_tensor_p) -> int:
    ...


# GGML_API GGML_CALL int64_t ggml_nelements   (const struct ggml_tensor * tensor);
@ggml_function(
    "ggml_nelements",
    [ggml_tensor_p_ctypes],
    ctypes.c_int64,
)
def ggml_nelements(tensor: ggml_tensor_p) -> int:
    ...


# GGML_API           int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
@ggml_function(
    "ggml_n_dims",
    [ggml_tensor_p_ctypes],
    ctypes.c_int,
)
def ggml_n_dims(tensor: ggml_tensor_p) -> int:
    ...


def ggml_tensor_to_numpy(tensor: ggml_tensor_p):
    nbytes = ggml_nbytes(tensor)
    nelements = ggml_nelements(tensor)
    data = np.empty(nelements, dtype=np.float32)
    ggml_backend_tensor_get(
        tensor, ctypes.cast(data.ctypes.data, ctypes.c_void_p), 0, nbytes
    )
    return data.reshape(tensor.contents.ne[: ggml_n_dims(tensor)])
