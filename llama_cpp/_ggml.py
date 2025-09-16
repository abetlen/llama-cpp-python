"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import enum
import os
import pathlib
import ctypes

import llama_cpp._ctypes_extensions as ctypes_ext

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_DEBUG = 1,
#     GGML_LOG_LEVEL_INFO  = 2,
#     GGML_LOG_LEVEL_WARN  = 3,
#     GGML_LOG_LEVEL_ERROR = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };

class GGMLLogLevel(enum.IntEnum):
    GGML_LOG_LEVEL_NONE = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT = 5 # continue previous log

# // ====== ggml-opt.h ======

# enum ggml_opt_build_type {
#     GGML_OPT_BUILD_TYPE_FORWARD = 10,
#     GGML_OPT_BUILD_TYPE_GRAD    = 20,
#     GGML_OPT_BUILD_TYPE_OPT     = 30,
# };
class GGMLOptBuildType(enum.IntEnum):
    GGML_OPT_BUILD_TYPE_FORWARD = 10
    GGML_OPT_BUILD_TYPE_GRAD = 20
    GGML_OPT_BUILD_TYPE_OPT = 30


# // built-in loss types, i.e. the built-in quantities minimized by the optimizer
# // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
# enum ggml_opt_loss_type {
#     GGML_OPT_LOSS_TYPE_MEAN,
#     GGML_OPT_LOSS_TYPE_SUM,
#     GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
#     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
# };
class GGMLOptLossType(enum.IntEnum):
    GGML_OPT_LOSS_TYPE_MEAN = 0
    GGML_OPT_LOSS_TYPE_SUM = 1
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY = 2
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3


# // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
# struct ggml_opt_optimizer_params {
#     // AdamW optimizer parameters
#     struct {
#         float alpha; // learning rate
#         float beta1;
#         float beta2;
#         float eps;   // epsilon for numerical stability
#         float wd;    // weight decay for AdamW, use 0.0f to disable
#     } adamw;
# };
class ggml_opt_adamw_params(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float), # learning rate
        ('beta1', ctypes.c_float),
        ('beta2', ctypes.c_float),
        ('eps',   ctypes.c_float), # epsilon for numerical stability
        ('wd',    ctypes.c_float), # weight decay for AdamW, use 0.0f to disable
    ]

class ggml_opt_optimizer_params(ctypes.Structure):
    _fields_ = [
        ('adamw', ggml_opt_adamw_params), # Nested AdamW parameters
    ]


# // callback to calculate optimizer parameters prior to a backward pass
# // userdata can be used to pass arbitrary data
# typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
ggml_opt_get_optimizer_params = ctypes.CFUNCTYPE(
    ctypes.POINTER(ggml_opt_optimizer_params), ctypes.c_void_p
)
