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
    NewType,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
            _base_path / f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    if "LLAMA_CPP_LIB" in os.environ:
        lib_base_name = os.environ["LLAMA_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))
        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)


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


ctypes_function = ctypes_function_for_shared_library(_lib)


def byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = ctypes.byref  # type: ignore


# from ggml-backend.h
# typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)

# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

# llama.h bindings

_lib.llama_max_devices.argtypes = []
_lib.llama_max_devices.restype = ctypes.c_size_t

LLAMA_MAX_DEVICES = _lib.llama_max_devices()

# define LLAMA_DEFAULT_SEED 0xFFFFFFFF
LLAMA_DEFAULT_SEED = 0xFFFFFFFF

# define LLAMA_MAX_RNG_STATE (64*1024)
LLAMA_MAX_RNG_STATE = 64 * 1024

# define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
LLAMA_FILE_MAGIC_GGLA = 0x67676C61

# define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
LLAMA_FILE_MAGIC_GGSN = 0x6767736E

# define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
# define LLAMA_SESSION_VERSION 4
LLAMA_SESSION_VERSION = 4


# struct llama_model;
llama_model_p = NewType("llama_model_p", int)
llama_model_p_ctypes = ctypes.c_void_p

# struct llama_context;
llama_context_p = NewType("llama_context_p", int)
llama_context_p_ctypes = ctypes.c_void_p


# typedef int32_t llama_pos;
llama_pos = ctypes.c_int32
# typedef int32_t llama_token;
llama_token = ctypes.c_int32
llama_token_p = ctypes.POINTER(llama_token)
# typedef int32_t llama_seq_id;
llama_seq_id = ctypes.c_int32


# enum llama_vocab_type {
#     LLAMA_VOCAB_TYPE_SPM = 0, // SentencePiece
#     LLAMA_VOCAB_TYPE_BPE = 1, // Byte Pair Encoding
#     LLAMA_VOCAB_TYPE_WPM = 2, // WordPiece
# };
LLAMA_VOCAB_TYPE_SPM = 0
LLAMA_VOCAB_TYPE_BPE = 1
LLAMA_VOCAB_TYPE_WPM = 2


# // note: these values should be synchronized with ggml_rope
# // TODO: maybe move this enum to ggml.h (ggml_rope_type)
# enum llama_rope_type {
#     LLAMA_ROPE_TYPE_NONE = -1,
#     LLAMA_ROPE_TYPE_NORM =  0,
#     LLAMA_ROPE_TYPE_NEOX =  2,
#     LLAMA_ROPE_TYPE_GLM  =  4,
# };
LLAMA_ROPE_TYPE_NONE = -1
LLAMA_ROPE_TYPE_NORM = 0
LLAMA_ROPE_TYPE_NEOX = 2
LLAMA_ROPE_TYPE_GLM = 4


# enum llama_token_type {
#     LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
#     LLAMA_TOKEN_TYPE_NORMAL       = 1,
#     LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
#     LLAMA_TOKEN_TYPE_CONTROL      = 3,
#     LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
#     LLAMA_TOKEN_TYPE_UNUSED       = 5,
#     LLAMA_TOKEN_TYPE_BYTE         = 6,
# };
LLAMA_TOKEN_TYPE_UNDEFINED = 0
LLAMA_TOKEN_TYPE_NORMAL = 1
LLAMA_TOKEN_TYPE_UNKNOWN = 2
LLAMA_TOKEN_TYPE_CONTROL = 3
LLAMA_TOKEN_TYPE_USER_DEFINED = 4
LLAMA_TOKEN_TYPE_UNUSED = 5
LLAMA_TOKEN_TYPE_BYTE = 6


# // model file types
# enum llama_ftype {
#     LLAMA_FTYPE_ALL_F32              = 0,
#     LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
#     // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
#     // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
#     LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors

#     LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
# };
LLAMA_FTYPE_ALL_F32 = 0
LLAMA_FTYPE_MOSTLY_F16 = 1
LLAMA_FTYPE_MOSTLY_Q4_0 = 2
LLAMA_FTYPE_MOSTLY_Q4_1 = 3
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
LLAMA_FTYPE_MOSTLY_Q8_0 = 7
LLAMA_FTYPE_MOSTLY_Q5_0 = 8
LLAMA_FTYPE_MOSTLY_Q5_1 = 9
LLAMA_FTYPE_MOSTLY_Q2_K = 10
LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
LLAMA_FTYPE_MOSTLY_Q6_K = 18
LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
LLAMA_FTYPE_MOSTLY_IQ2_XS = 20
LLAMA_FTYPE_MOSTLY_Q2_K_S = 21
LLAMA_FTYPE_MOSTLY_IQ3_XS = 22
LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23
LLAMA_FTYPE_MOSTLY_IQ1_S = 24
LLAMA_FTYPE_MOSTLY_IQ4_NL = 25
LLAMA_FTYPE_MOSTLY_IQ3_S = 26
LLAMA_FTYPE_MOSTLY_IQ3_M = 27
LLAMA_FTYPE_MOSTLY_IQ2_S = 28
LLAMA_FTYPE_MOSTLY_IQ2_M = 29
LLAMA_FTYPE_MOSTLY_IQ4_XS = 30
LLAMA_FTYPE_GUESSED = 1024

# enum llama_rope_scaling_type {
#     LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
#     LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
#     LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
#     LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
#     LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_YARN,
# };
LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
LLAMA_ROPE_SCALING_TYPE_NONE = 0
LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
LLAMA_ROPE_SCALING_TYPE_YARN = 2
LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_YARN

# enum llama_pooling_type {
#     LLAMA_POOLING_TYPE_NONE = 0,
#     LLAMA_POOLING_TYPE_MEAN = 1,
#     LLAMA_POOLING_TYPE_CLS  = 2,
# };
LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2

# enum llama_split_mode {
#     LLAMA_SPLIT_MODE_NONE    = 0, // single GPU
#     LLAMA_SPLIT_MODE_LAYER   = 1, // split layers and KV across GPUs
#     LLAMA_SPLIT_MODE_ROW     = 2, // split rows across GPUs
# };
LLAMA_SPLIT_MODE_NONE = 0
LLAMA_SPLIT_MODE_LAYER = 1
LLAMA_SPLIT_MODE_ROW = 2


# typedef struct llama_token_data {
#     llama_token id; // token id
#     float logit;    // log-odds of the token
#     float p;        // probability of the token
# } llama_token_data;
class llama_token_data(ctypes.Structure):
    """Used to store token data

    Attributes:
        id (llama_token): token id
        logit (float): log-odds of the token
        p (float): probability of the token"""

    _fields_ = [
        ("id", llama_token),
        ("logit", ctypes.c_float),
        ("p", ctypes.c_float),
    ]


llama_token_data_p = ctypes.POINTER(llama_token_data)


# typedef struct llama_token_data_array {
#     llama_token_data * data;
#     size_t size;
#     bool sorted;
# } llama_token_data_array;
class llama_token_data_array(ctypes.Structure):
    """Used to sample tokens given logits

    Attributes:
        data (ctypes.Array[llama_token_data]): token data
        size (int): size of the array
        sorted (bool): whether the array is sorted"""

    _fields_ = [
        ("data", llama_token_data_p),
        ("size", ctypes.c_size_t),
        ("sorted", ctypes.c_bool),
    ]


llama_token_data_array_p = ctypes.POINTER(llama_token_data_array)

# typedef bool (*llama_progress_callback)(float progress, void *ctx);
llama_progress_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_float, ctypes.c_void_p
)


# // Input data for llama_decode
# // A llama_batch object can contain input about one or many sequences
# // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
# //
# // - token  : the token ids of the input (used when embd is NULL)
# // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
# // - pos    : the positions of the respective token in the sequence
# // - seq_id : the sequence to which the respective token belongs
# // - logits : if zero, the logits for the respective token will not be output
# //
# typedef struct llama_batch {
#     int32_t n_tokens;

#     llama_token  *  token;
#     float        *  embd;
#     llama_pos    *  pos;
#     int32_t      *  n_seq_id;
#     llama_seq_id ** seq_id;
#     int8_t       *  logits;


#     // NOTE: helpers for smooth API transition - can be deprecated in the future
#     //       for future-proof code, use the above fields instead and ignore everything below
#     //
#     // pos[i] = all_pos_0 + i*all_pos_1
#     //
#     llama_pos    all_pos_0;  // used if pos == NULL
#     llama_pos    all_pos_1;  // used if pos == NULL
#     llama_seq_id all_seq_id; // used if seq_id == NULL
# } llama_batch;
class llama_batch(ctypes.Structure):
    """Input data for llama_decode

    A llama_batch object can contain input about one or many sequences

    The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens

    Attributes:
        token (ctypes.Array[llama_token]): the token ids of the input (used when embd is NULL)
        embd (ctypes.Array[ctypes.ctypes.c_float]): token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        pos (ctypes.Array[ctypes.Array[llama_pos]]): the positions of the respective token in the sequence
        seq_id (ctypes.Array[ctypes.Array[llama_seq_id]]): the sequence to which the respective token belongs
    """

    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
        ("all_pos_0", llama_pos),
        ("all_pos_1", llama_pos),
        ("all_seq_id", llama_seq_id),
    ]


# enum llama_model_kv_override_type {
#     LLAMA_KV_OVERRIDE_TYPE_INT,
#     LLAMA_KV_OVERRIDE_TYPE_FLOAT,
#     LLAMA_KV_OVERRIDE_TYPE_BOOL,
# };
LLAMA_KV_OVERRIDE_TYPE_INT = 0
LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
LLAMA_KV_OVERRIDE_TYPE_BOOL = 2


# struct llama_model_kv_override {
#     char key[128];
#     enum llama_model_kv_override_type tag;
#     union {
#         int64_t int_value;
#         double float_value;
#         bool bool_value;
#     };
# };
class llama_model_kv_override_value(ctypes.Union):
    _fields_ = [
        ("int_value", ctypes.c_int64),
        ("float_value", ctypes.c_double),
        ("bool_value", ctypes.c_bool),
    ]


class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_char * 128),
        ("tag", ctypes.c_int),
        ("value", llama_model_kv_override_value),
    ]


# struct llama_model_params {
#     int32_t n_gpu_layers; // number of layers to store in VRAM
#     enum llama_split_mode split_mode; // how to split the model across multiple GPUs

#     // main_gpu interpretation depends on split_mode:
#     // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
#     // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
#     // LLAMA_SPLIT_LAYER: ignored
#     int32_t main_gpu;

#     // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
#     const float * tensor_split;

#     // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
#     // If the provided progress_callback returns true, model loading continues.
#     // If it returns false, model loading is immediately aborted.
#     llama_progress_callback progress_callback;

#     // context pointer passed to the progress callback
#     void * progress_callback_user_data;

#     // override key-value pairs of the model meta data
#     const struct llama_model_kv_override * kv_overrides;


#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool vocab_only; // only load the vocabulary, no weights
#     bool use_mmap;   // use mmap if possible
#     bool use_mlock;  // force system to keep model in RAM
# };
class llama_model_params(ctypes.Structure):
    """Parameters for llama_model

    Attributes:
        n_gpu_layers (int): number of layers to store in VRAM
        split_mode (int): how to split the model across multiple GPUs
        main_gpu (int): the GPU that is used for the entire model. main_gpu interpretation depends on split_mode: LLAMA_SPLIT_NONE: the GPU that is used for the entire model LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results LLAMA_SPLIT_LAYER: ignored
        tensor_split (ctypes.Array[ctypes.ctypes.c_float]): proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        progress_callback (llama_progress_callback): called with a progress value between 0.0 and 1.0. Pass NULL to disable. If the provided progress_callback returns true, model loading continues. If it returns false, model loading is immediately aborted.
        progress_callback_user_data (ctypes.ctypes.c_void_p): context pointer passed to the progress callback
        kv_overrides (ctypes.Array[llama_model_kv_override]): override key-value pairs of the model meta data
        vocab_only (bool): only load the vocabulary, no weights
        use_mmap (bool): use mmap if possible
        use_mlock (bool): force system to keep model in RAM"""

    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
    ]


# struct llama_context_params {
#     uint32_t seed;              // RNG seed, -1 for random
#     uint32_t n_ctx;             // text context, 0 = from model
#     uint32_t n_batch;           // prompt processing maximum batch size
#     uint32_t n_threads;         // number of threads to use for generation
#     uint32_t n_threads_batch;   // number of threads to use for batch processing
#     int32_t  rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`

#     // ref: https://github.com/ggerganov/llama.cpp/pull/2054
#     float    rope_freq_base;   // RoPE base frequency, 0 = from model
#     float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
#     float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
#     float    yarn_attn_factor; // YaRN magnitude scaling factor
#     float    yarn_beta_fast;   // YaRN low correction dim
#     float    yarn_beta_slow;   // YaRN high correction dim
#     uint32_t yarn_orig_ctx;    // YaRN original context size
#     float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

#     ggml_backend_sched_eval_callback cb_eval;
#     void * cb_eval_user_data;

#     enum ggml_type type_k; // data type for K cache
#     enum ggml_type type_v; // data type for V cache

#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
#     bool embedding;   // embedding mode only
#     bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
#     bool do_pooling;  // whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)

#     // Abort callback
#     // if it returns true, execution of llama_decode() will be aborted
#     // currently works only with CPU execution
#     ggml_abort_callback abort_callback;
#     void *              abort_callback_data;
# };
class llama_context_params(ctypes.Structure):
    """Parameters for llama_context

    Attributes:
        seed (int): RNG seed, -1 for random
        n_ctx (int): text context, 0 = from model
        n_batch (int): prompt processing maximum batch size
        n_threads (int): number of threads to use for generation
        n_threads_batch (int): number of threads to use for batch processing
        rope_scaling_type (int): RoPE scaling type, from `enum llama_rope_scaling_type`
        rope_freq_base (float): RoPE base frequency, 0 = from model
        rope_freq_scale (float): RoPE frequency scaling factor, 0 = from model
        yarn_ext_factor (float): YaRN extrapolation mix factor, negative = from model
        yarn_attn_factor (float): YaRN magnitude scaling factor
        yarn_beta_fast (float): YaRN low correction dim
        yarn_beta_slow (float): YaRN high correction dim
        yarn_orig_ctx (int): YaRN original context size
        defrag_thold (float): defragment the KV cache if holes/size > thold, < 0 disabled (default)
        cb_eval (ggml_backend_sched_eval_callback): callback for scheduling eval
        cb_eval_user_data (ctypes.ctypes.c_void_p): user data for cb_eval
        type_k (int): data type for K cache
        type_v (int): data type for V cache
        logits_all (bool): the llama_eval() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        embedding (bool): embedding mode only
        offload_kqv (bool): whether to offload the KQV ops (including the KV cache) to GPU
        do_pooling (bool): whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
        abort_callback (ggml_abort_callback): abort callback if it returns true, execution of llama_decode() will be aborted
        abort_callback_data (ctypes.ctypes.c_void_p): data for abort_callback
    """

    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_threads", ctypes.c_uint32),
        ("n_threads_batch", ctypes.c_uint32),
        ("rope_scaling_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("logits_all", ctypes.c_bool),
        ("embedding", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("do_pooling", ctypes.c_bool),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
    ]


# // Signature for logging events
# // Note that text includes the new line character at the end for most events.
# // If your logging mechanism cannot handle that, check if the last character is '\n' and strip it
# // if it exists.
# // It might not exist for progress report where '.' is output repeatedly.
# typedef void (*llama_log_callback)(enum llama_log_level level, const char * text, void * user_data);
llama_log_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)
"""Signature for logging events
Note that text includes the new line character at the end for most events.
If your logging mechanism cannot handle that, check if the last character is '\n' and strip it
if it exists.
It might not exist for progress report where '.' is output repeatedly."""


# // model quantization parameters
# typedef struct llama_model_quantize_params {
#     int32_t nthread;             // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
#     enum llama_ftype ftype;      // quantize to this llama_ftype
#     bool allow_requantize;       // allow quantizing non-f32/f16 tensors
#     bool quantize_output_tensor; // quantize output.weight
#     bool only_copy;              // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
#     bool pure;                   // disable k-quant mixtures and quantize all tensors to the same type
#     void * imatrix;              // pointer to importance matrix data
# } llama_model_quantize_params;
class llama_model_quantize_params(ctypes.Structure):
    """Parameters for llama_model_quantize

    Attributes:
        nthread (int): number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ftype (int): quantize to this llama_ftype
        allow_requantize (bool): allow quantizing non-f32/f16 tensors
        quantize_output_tensor (bool): quantize output.weight
        only_copy (bool): only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        pure (bool): disable k-quant mixtures and quantize all tensors to the same type
        imatrix (ctypes.ctypes.c_void_p): pointer to importance matrix data
    """

    _fields_ = [
        ("nthread", ctypes.c_int32),
        ("ftype", ctypes.c_int),
        ("allow_requantize", ctypes.c_bool),
        ("quantize_output_tensor", ctypes.c_bool),
        ("only_copy", ctypes.c_bool),
        ("pure", ctypes.c_bool),
        ("imatrix", ctypes.c_void_p),
    ]


# // grammar types
# struct llama_grammar;
llama_grammar_p = ctypes.c_void_p

# // grammar element type
# enum llama_gretype {
#     // end of rule definition
#     LLAMA_GRETYPE_END            = 0,

#     // start of alternate definition for rule
#     LLAMA_GRETYPE_ALT            = 1,

#     // non-terminal element: reference to rule
#     LLAMA_GRETYPE_RULE_REF       = 2,

#     // terminal element: character (code point)
#     LLAMA_GRETYPE_CHAR           = 3,

#     // inverse char(s) ([^a], [^a-b] [^abc])
#     LLAMA_GRETYPE_CHAR_NOT       = 4,

#     // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
#     // be an inclusive range ([a-z])
#     LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

#     // modifies a preceding LLAMA_GRETYPE_CHAR or
#     // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
#     LLAMA_GRETYPE_CHAR_ALT       = 6,
# };
LLAMA_GRETYPE_END = 0
LLAMA_GRETYPE_ALT = 1
LLAMA_GRETYPE_RULE_REF = 2
LLAMA_GRETYPE_CHAR = 3
LLAMA_GRETYPE_CHAR_NOT = 4
LLAMA_GRETYPE_CHAR_RNG_UPPER = 5
LLAMA_GRETYPE_CHAR_ALT = 6


# typedef struct llama_grammar_element {
#     enum llama_gretype type;
#     uint32_t           value; // Unicode code point or rule ID
# } llama_grammar_element;
class llama_grammar_element(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("value", ctypes.c_uint32),
    ]


llama_grammar_element_p = ctypes.POINTER(llama_grammar_element)

# // performance timing information
# struct llama_timings {
#     double t_start_ms;
#     double t_end_ms;
#     double t_load_ms;
#     double t_sample_ms;
#     double t_p_eval_ms;
#     double t_eval_ms;


#     int32_t n_sample;
#     int32_t n_p_eval;
#     int32_t n_eval;
# };
class llama_timings(ctypes.Structure):
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_end_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_sample_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_sample", ctypes.c_int32),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
    ]


# // used in chat template
# typedef struct llama_chat_message {
#     const char * role;
#     const char * content;
# } llama_chat_message;
class llama_chat_message(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


# // Helpers for getting default parameters
# LLAMA_API struct llama_model_params llama_model_default_params(void);
@ctypes_function(
    "llama_model_default_params",
    [],
    llama_model_params,
)
def llama_model_default_params() -> llama_model_params:
    """Get default parameters for llama_model"""
    ...


# LLAMA_API struct llama_context_params llama_context_default_params(void);
@ctypes_function(
    "llama_context_default_params",
    [],
    llama_context_params,
)
def llama_context_default_params() -> llama_context_params:
    """Get default parameters for llama_context"""
    ...


# LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);
@ctypes_function(
    "llama_model_quantize_default_params",
    [],
    llama_model_quantize_params,
)
def llama_model_quantize_default_params() -> llama_model_quantize_params:
    """Get default parameters for llama_model_quantize"""
    ...


# // Initialize the llama + ggml backend
# // If numa is true, use NUMA optimizations
# // Call once at the start of the program
# LLAMA_API void llama_backend_init(bool numa);
# LLAMA_API void llama_backend_init(void);
@ctypes_function(
    "llama_backend_init",
    [],
    None,
)
def llama_backend_init():
    """Initialize the llama + ggml backend
    If numa is true, use NUMA optimizations
    Call once at the start of the program"""
    ...


# // numa strategies
# enum ggml_numa_strategy {
#     GGML_NUMA_STRATEGY_DISABLED   = 0,
#     GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
#     GGML_NUMA_STRATEGY_ISOLATE    = 2,
#     GGML_NUMA_STRATEGY_NUMACTL    = 3,
#     GGML_NUMA_STRATEGY_MIRROR     = 4,
#     GGML_NUMA_STRATEGY_COUNT
# };
GGML_NUMA_STRATEGY_DISABLED = 0
GGML_NUMA_STRATEGY_DISTRIBUTE = 1
GGML_NUMA_STRATEGY_ISOLATE = 2
GGML_NUMA_STRATEGY_NUMACTL = 3
GGML_NUMA_STRATEGY_MIRROR = 4
GGML_NUMA_STRATEGY_COUNT = 5


# //optional:
# LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);
@ctypes_function(
    "llama_numa_init",
    [ctypes.c_int],
    None,
)
def llama_numa_init(numa: int, /):
    ...


# // Call once at the end of the program - currently only used for MPI
# LLAMA_API void llama_backend_free(void);
@ctypes_function(
    "llama_backend_free",
    [],
    None,
)
def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    ...


# LLAMA_API struct llama_model * llama_load_model_from_file(
#                          const char * path_model,
#         struct llama_model_params     params);
@ctypes_function(
    "llama_load_model_from_file",
    [ctypes.c_char_p, llama_model_params],
    llama_model_p_ctypes,
)
def llama_load_model_from_file(
    path_model: bytes, params: llama_model_params, /
) -> Optional[llama_model_p]:
    ...


# LLAMA_API void llama_free_model(struct llama_model * model);
@ctypes_function(
    "llama_free_model",
    [llama_model_p_ctypes],
    None,
)
def llama_free_model(model: llama_model_p, /):
    ...


# LLAMA_API struct llama_context * llama_new_context_with_model(
#                  struct llama_model * model,
#         struct llama_context_params   params);
@ctypes_function(
    "llama_new_context_with_model",
    [llama_model_p_ctypes, llama_context_params],
    llama_context_p_ctypes,
)
def llama_new_context_with_model(
    model: llama_model_p, params: llama_context_params, /
) -> Optional[llama_context_p]:
    ...


# // Frees all allocated memory
# LLAMA_API void llama_free(struct llama_context * ctx);
@ctypes_function(
    "llama_free",
    [llama_context_p_ctypes],
    None,
)
def llama_free(ctx: llama_context_p, /):
    """Frees all allocated memory"""
    ...


# LLAMA_API int64_t llama_time_us(void);
@ctypes_function(
    "llama_time_us",
    [],
    ctypes.c_int64,
)
def llama_time_us() -> int:
    ...


# LLAMA_API size_t llama_max_devices(void);
@ctypes_function("llama_max_devices", [], ctypes.c_size_t)
def llama_max_devices() -> int:
    ...


# LLAMA_API bool llama_supports_mmap       (void);
@ctypes_function("llama_supports_mmap", [], ctypes.c_bool)
def llama_supports_mmap() -> bool:
    ...


# LLAMA_API bool llama_supports_mlock      (void);
@ctypes_function("llama_supports_mlock", [], ctypes.c_bool)
def llama_supports_mlock() -> bool:
    ...


# LLAMA_API bool llama_supports_gpu_offload(void);
@ctypes_function("llama_supports_gpu_offload", [], ctypes.c_bool)
def llama_supports_gpu_offload() -> bool:
    ...


# LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);
@ctypes_function("llama_get_model", [llama_context_p_ctypes], llama_model_p_ctypes)
def llama_get_model(ctx: llama_context_p, /) -> Optional[llama_model_p]:
    ...


# LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
@ctypes_function("llama_n_ctx", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_ctx(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
@ctypes_function("llama_n_batch", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_batch(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_model * model);
@ctypes_function("llama_vocab_type", [llama_model_p_ctypes], ctypes.c_int)
def llama_vocab_type(model: llama_model_p, /) -> int:
    ...


# LLAMA_API enum llama_rope_type  llama_rope_type (const struct llama_model * model);
@ctypes_function("llama_rope_type", [llama_model_p_ctypes], ctypes.c_int)
def llama_rope_type(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_n_vocab    (const struct llama_model * model);
@ctypes_function("llama_n_vocab", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_vocab(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model);
@ctypes_function("llama_n_ctx_train", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_ctx_train(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_n_embd     (const struct llama_model * model);
@ctypes_function("llama_n_embd", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_embd(model: llama_model_p, /) -> int:
    ...


# // Get the model's RoPE frequency scaling factor
# LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);
@ctypes_function("llama_rope_freq_scale_train", [llama_model_p_ctypes], ctypes.c_float)
def llama_rope_freq_scale_train(model: llama_model_p, /) -> float:
    """Get the model's RoPE frequency scaling factor"""
    ...


# // Functions to access the model's GGUF metadata scalar values
# // - The functions return the length of the string on success, or -1 on failure
# // - The output string is always null-terminated and cleared on failure
# // - GGUF array values are not supported by these functions


# // Get metadata value as a string by key name
# LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);
@ctypes_function(
    "llama_model_meta_val_str",
    [
        llama_model_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_model_meta_val_str(
    model: llama_model_p,
    key: Union[ctypes.c_char_p, bytes],
    buf: bytes,
    buf_size: int,
    /,
) -> int:
    """Get metadata value as a string by key name"""
    ...


# // Get the number of metadata key/value pairs
# LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);
@ctypes_function("llama_model_meta_count", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_meta_count(model: llama_model_p, /) -> int:
    """Get the number of metadata key/value pairs"""
    ...


# // Get metadata key name by index
# LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
@ctypes_function(
    "llama_model_meta_key_by_index",
    [
        llama_model_p_ctypes,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_model_meta_key_by_index(
    model: llama_model_p,
    i: Union[ctypes.c_int, int],
    buf: Union[bytes, CtypesArray[ctypes.c_char]],
    buf_size: int,
    /,
) -> int:
    """Get metadata key name by index"""
    ...


# // Get metadata value as a string by index
# LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
@ctypes_function(
    "llama_model_meta_val_str_by_index",
    [
        llama_model_p_ctypes,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_model_meta_val_str_by_index(
    model: llama_model_p,
    i: Union[ctypes.c_int, int],
    buf: Union[bytes, CtypesArray[ctypes.c_char]],
    buf_size: int,
    /,
) -> int:
    """Get metadata value as a string by index"""
    ...


# // Get a string describing the model type
# LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
@ctypes_function(
    "llama_model_desc",
    [llama_model_p_ctypes, ctypes.c_char_p, ctypes.c_size_t],
    ctypes.c_int32,
)
def llama_model_desc(
    model: llama_model_p,
    buf: Union[bytes, CtypesArray[ctypes.c_char]],
    buf_size: Union[ctypes.c_size_t, int],
    /,
) -> int:
    """Get a string describing the model type"""
    ...


# // Returns the total size of all the tensors in the model in bytes
# LLAMA_API uint64_t llama_model_size(const struct llama_model * model);
@ctypes_function("llama_model_size", [llama_model_p_ctypes], ctypes.c_uint64)
def llama_model_size(model: llama_model_p, /) -> int:
    """Returns the total size of all the tensors in the model in bytes"""
    ...


# // Returns the total number of parameters in the model
# LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);
@ctypes_function("llama_model_n_params", [llama_model_p_ctypes], ctypes.c_uint64)
def llama_model_n_params(model: llama_model_p, /) -> int:
    """Returns the total number of parameters in the model"""
    ...


# // Get a llama model tensor
# LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);
@ctypes_function(
    "llama_get_model_tensor", [llama_model_p_ctypes, ctypes.c_char_p], ctypes.c_void_p
)
def llama_get_model_tensor(
    model: llama_model_p, name: Union[ctypes.c_char_p, bytes], /
) -> ctypes.c_void_p:
    """Get a llama model tensor"""
    ...


# // Returns 0 on success
# LLAMA_API uint32_t llama_model_quantize(
#         const char * fname_inp,
#         const char * fname_out,
#         const llama_model_quantize_params * params);
@ctypes_function(
    "llama_model_quantize",
    [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(llama_model_quantize_params),
    ],
    ctypes.c_uint32,
)
def llama_model_quantize(
    fname_inp: bytes,
    fname_out: bytes,
    params: CtypesPointerOrRef[llama_model_quantize_params],
    /,
) -> int:
    """Returns 0 on success"""
    ...


# LLAMA_API int32_t llama_model_apply_lora_from_file(
#         const struct llama_model * model,
#                   const char * path_lora,
#                        float   scale,
#                   const char * path_base_model,
#                      int32_t   n_threads);
@ctypes_function(
    "llama_model_apply_lora_from_file",
    [
        llama_model_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_char_p,
        ctypes.c_int32,
    ],
    ctypes.c_int32,
)
def llama_model_apply_lora_from_file(
    model: llama_model_p,
    path_lora: Union[ctypes.c_char_p, bytes],
    scale: Union[ctypes.c_float, float],
    path_base_model: Union[ctypes.c_char_p, bytes, None],
    n_threads: Union[ctypes.c_int32, int],
    /,
) -> int:
    ...


# //
# // KV cache
# //


# // Information associated with an individual cell in the KV cache view.
# struct llama_kv_cache_view_cell {
#     // The position for this cell. Takes KV cache shifts into account.
#     // May be negative if the cell is not populated.
#     llama_pos pos;
# };
class llama_kv_cache_view_cell(ctypes.Structure):
    _fields_ = [("pos", llama_pos)]


# // An updateable view of the KV cache.
# struct llama_kv_cache_view {
#     // Number of KV cache cells. This will be the same as the context size.
#     int32_t n_cells;

#     // Maximum number of sequences that can exist in a cell. It's not an error
#     // if there are more sequences in a cell than this value, however they will
#     // not be visible in the view cells_sequences.
#     int32_t n_max_seq;

#     // Number of tokens in the cache. For example, if there are two populated
#     // cells, the first with 1 sequence id in it and the second with 2 sequence
#     // ids then you'll have 3 tokens.
#     int32_t token_count;

#     // Number of populated cache cells.
#     int32_t used_cells;

#     // Maximum contiguous empty slots in the cache.
#     int32_t max_contiguous;

#     // Index to the start of the max_contiguous slot range. Can be negative
#     // when cache is full.
#     int32_t max_contiguous_idx;

#     // Information for an individual cell.
#     struct llama_kv_cache_view_cell * cells;


#     // The sequences for each cell. There will be n_max_seq items per cell.
#     llama_seq_id * cells_sequences;
# };
class llama_kv_cache_view(ctypes.Structure):
    _fields_ = [
        ("n_cells", ctypes.c_int32),
        ("n_max_seq", ctypes.c_int32),
        ("token_count", ctypes.c_int32),
        ("used_cells", ctypes.c_int32),
        ("max_contiguous", ctypes.c_int32),
        ("max_contiguous_idx", ctypes.c_int32),
        ("cells", ctypes.POINTER(llama_kv_cache_view_cell)),
        ("cells_sequences", ctypes.POINTER(llama_seq_id)),
    ]


llama_kv_cache_view_p = ctypes.POINTER(llama_kv_cache_view)


# // Create an empty KV cache view. (use only for debugging purposes)
# LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_max_seq);
@ctypes_function(
    "llama_kv_cache_view_init",
    [llama_context_p_ctypes, ctypes.c_int32],
    llama_kv_cache_view,
)
def llama_kv_cache_view_init(
    ctx: llama_context_p, n_max_seq: Union[ctypes.c_int32, int], /
) -> llama_kv_cache_view:
    """Create an empty KV cache view. (use only for debugging purposes)"""
    ...


# // Free a KV cache view. (use only for debugging purposes)
# LLAMA_API void llama_kv_cache_view_free(struct llama_kv_cache_view * view);
@ctypes_function("llama_kv_cache_view_free", [llama_kv_cache_view_p], None)
def llama_kv_cache_view_free(view: "ctypes.pointer[llama_kv_cache_view]", /):  # type: ignore
    """Free a KV cache view. (use only for debugging purposes)"""
    ...


# // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
# LLAMA_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);
@ctypes_function(
    "llama_kv_cache_view_update", [llama_context_p_ctypes, llama_kv_cache_view_p], None
)
def llama_kv_cache_view_update(ctx: llama_context_p, view: CtypesPointerOrRef[llama_kv_cache_view], /):  # type: ignore
    """Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)"""
    ...


# // Returns the number of tokens in the KV cache (slow, use only for debug)
# // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
# LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);
@ctypes_function(
    "llama_get_kv_cache_token_count", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_get_kv_cache_token_count(ctx: llama_context_p, /) -> int:
    """Returns the number of tokens in the KV cache (slow, use only for debug)
    If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    """
    ...


# // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
# LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);
@ctypes_function(
    "llama_get_kv_cache_used_cells", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_get_kv_cache_used_cells(ctx: llama_context_p, /) -> int:
    """Returns the number of used KV cells (i.e. have at least one sequence assigned to them)"""
    ...


# // Clear the KV cache
# LLAMA_API void llama_kv_cache_clear(
#         struct llama_context * ctx);
@ctypes_function("llama_kv_cache_clear", [llama_context_p_ctypes], None)
def llama_kv_cache_clear(ctx: llama_context_p, /):
    """Clear the KV cache"""
    ...


# // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
# // seq_id < 0 : match any sequence
# // p0 < 0     : [0,  p1]
# // p1 < 0     : [p0, inf)
# LLAMA_API void llama_kv_cache_seq_rm(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id,
#                    llama_pos   p0,
#                    llama_pos   p1);
@ctypes_function(
    "llama_kv_cache_seq_rm",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_pos,
        llama_pos,
    ],
    None,
)
def llama_kv_cache_seq_rm(
    ctx: llama_context_p,
    seq_id: Union[llama_seq_id, int],
    p0: Union[llama_pos, int],
    p1: Union[llama_pos, int],
    /,
):
    """Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    seq_id < 0 : match any sequence
    p0 < 0     : [0,  p1]
    p1 < 0     : [p0, inf)"""
    ...


# // Copy all tokens that belong to the specified sequence to another sequence
# // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
# // p0 < 0 : [0,  p1]
# // p1 < 0 : [p0, inf)
# LLAMA_API void llama_kv_cache_seq_cp(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id_src,
#                 llama_seq_id   seq_id_dst,
#                    llama_pos   p0,
#                    llama_pos   p1);
@ctypes_function(
    "llama_kv_cache_seq_cp",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_seq_id,
        llama_pos,
        llama_pos,
    ],
    None,
)
def llama_kv_cache_seq_cp(
    ctx: llama_context_p,
    seq_id_src: Union[llama_seq_id, int],
    seq_id_dst: Union[llama_seq_id, int],
    p0: Union[llama_pos, int],
    p1: Union[llama_pos, int],
    /,
):
    """Copy all tokens that belong to the specified sequence to another sequence
    Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    ...


# // Removes all tokens that do not belong to the specified sequence
# LLAMA_API void llama_kv_cache_seq_keep(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id);
@ctypes_function(
    "llama_kv_cache_seq_keep", [llama_context_p_ctypes, llama_seq_id], None
)
def llama_kv_cache_seq_keep(ctx: llama_context_p, seq_id: Union[llama_seq_id, int], /):
    """Removes all tokens that do not belong to the specified sequence"""
    ...


# // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
# // If the KV cache is RoPEd, the KV data is updated accordingly:
# //   - lazily on next llama_decode()
# //   - explicitly with llama_kv_cache_update()
# // p0 < 0 : [0,  p1]
# // p1 < 0 : [p0, inf)
# LLAMA_API void llama_kv_cache_seq_add(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id,
#                    llama_pos   p0,
#                    llama_pos   p1,
#                    llama_pos   delta);
@ctypes_function(
    "llama_kv_cache_seq_add",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_pos,
        llama_pos,
        llama_pos,
    ],
    None,
)
def llama_kv_cache_seq_add(
    ctx: llama_context_p,
    seq_id: Union[llama_seq_id, int],
    p0: Union[llama_pos, int],
    p1: Union[llama_pos, int],
    delta: Union[llama_pos, int],
    /,
):
    """Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    If the KV cache is RoPEd, the KV data is updated accordingly:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    ...


# // Integer division of the positions by factor of `d > 1`
# // If the KV cache is RoPEd, the KV data is updated accordingly
# // p0 < 0 : [0,  p1]
# // p1 < 0 : [p0, inf)
# LLAMA_API void llama_kv_cache_seq_div(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id,
#                    llama_pos   p0,
#                    llama_pos   p1,
#                          int   d);
@ctypes_function(
    "llama_kv_cache_seq_div",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_pos,
        llama_pos,
        ctypes.c_int,
    ],
    None,
)
def llama_kv_cache_seq_div(
    ctx: llama_context_p,
    seq_id: Union[llama_seq_id, int],
    p0: Union[llama_pos, int],
    p1: Union[llama_pos, int],
    d: Union[ctypes.c_int, int],
    /,
):
    """Integer division of the positions by factor of `d > 1`
    If the KV cache is RoPEd, the KV data is updated accordingly
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    ...


# // Defragment the KV cache
# // This will be applied:
# //   - lazily on next llama_decode()
# //   - explicitly with llama_kv_cache_update()
# LLAMA_API void llama_kv_cache_defrag(struct llama_context * ctx);
@ctypes_function("llama_kv_cache_defrag", [llama_context_p_ctypes], None)
def llama_kv_cache_defrag(ctx: llama_context_p, /):
    """Defragment the KV cache
    This will be applied:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()"""
    ...


# // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
# LLAMA_API void llama_kv_cache_update(struct llama_context * ctx);
@ctypes_function("llama_kv_cache_update", [llama_context_p_ctypes], None)
def llama_kv_cache_update(ctx: llama_context_p, /):
    """Apply the KV cache updates (such as K-shifts, defragmentation, etc.)"""
    ...


# //
# // State / sessions
# //


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
# LLAMA_API size_t llama_get_state_size(const struct llama_context * ctx);
@ctypes_function("llama_get_state_size", [llama_context_p_ctypes], ctypes.c_size_t)
def llama_get_state_size(ctx: llama_context_p, /) -> int:
    """Returns the maximum size in bytes of the state (rng, logits, embedding
    and kv_cache) - will often be smaller after compacting tokens"""
    ...


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
# LLAMA_API size_t llama_copy_state_data(
#         struct llama_context * ctx,
#                      uint8_t * dst);
@ctypes_function(
    "llama_copy_state_data",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
    ],
    ctypes.c_size_t,
)
def llama_copy_state_data(
    ctx: llama_context_p, dst: CtypesArray[ctypes.c_uint8], /
) -> int:
    """Copies the state to the specified destination address.
    Destination needs to have allocated enough memory.
    Returns the number of bytes copied"""
    ...


# // Set the state reading from the specified address
# // Returns the number of bytes read
# LLAMA_API size_t llama_set_state_data(
#         struct llama_context * ctx,
#                const uint8_t * src);
@ctypes_function(
    "llama_set_state_data",
    [llama_context_p_ctypes, ctypes.POINTER(ctypes.c_uint8)],
    ctypes.c_size_t,
)
def llama_set_state_data(
    ctx: llama_context_p, src: CtypesArray[ctypes.c_uint8], /
) -> int:
    """Set the state reading from the specified address"""
    ...


# Save/load session file
# LLAMA_API bool llama_load_session_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#                  llama_token * tokens_out,
#                       size_t   n_token_capacity,
#                       size_t * n_token_count_out);
@ctypes_function(
    "llama_load_session_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_token_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    ctypes.c_size_t,
)
def llama_load_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens_out: CtypesArray[llama_token],
    n_token_capacity: Union[ctypes.c_size_t, int],
    n_token_count_out: CtypesPointerOrRef[ctypes.c_size_t],
    /,
) -> int:
    ...


# LLAMA_API bool llama_save_session_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#            const llama_token * tokens,
#                       size_t   n_token_count);
@ctypes_function(
    "llama_save_session_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_token_p,
        ctypes.c_size_t,
    ],
    ctypes.c_size_t,
)
def llama_save_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens: CtypesArray[llama_token],
    n_token_count: Union[ctypes.c_size_t, int],
    /,
) -> int:
    ...


# //
# // Decoding
# //


# // Return batch for single sequence of tokens starting at pos_0
# //
# // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
# //
# LLAMA_API struct llama_batch llama_batch_get_one(
#               llama_token * tokens,
#                   int32_t   n_tokens,
#                 llama_pos   pos_0,
#              llama_seq_id   seq_id);
@ctypes_function(
    "llama_batch_get_one",
    [
        llama_token_p,
        ctypes.c_int,
        llama_pos,
        llama_seq_id,
    ],
    llama_batch,
)
def llama_batch_get_one(
    tokens: CtypesArray[llama_token],
    n_tokens: Union[ctypes.c_int, int],
    pos_0: Union[llama_pos, int],
    seq_id: llama_seq_id,
    /,
) -> llama_batch:
    """Return batch for single sequence of tokens starting at pos_0

    NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    """
    ...


# // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
# // Each token can be assigned up to n_seq_max sequence ids
# // The batch has to be freed with llama_batch_free()
# // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
# // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
# // The rest of the llama_batch members are allocated with size n_tokens
# // All members are left uninitialized
# LLAMA_API struct llama_batch llama_batch_init(
#         int32_t n_tokens,
#         int32_t embd,
#         int32_t n_seq_max);
@ctypes_function(
    "llama_batch_init", [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], llama_batch
)
def llama_batch_init(
    n_tokens: Union[ctypes.c_int32, int],
    embd: Union[ctypes.c_int32, int],
    n_seq_max: Union[ctypes.c_int32, int],
    /,
) -> llama_batch:
    """Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    Each token can be assigned up to n_seq_max sequence ids
    The batch has to be freed with llama_batch_free()
    If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    The rest of the llama_batch members are allocated with size n_tokens
    All members are left uninitialized"""
    ...


# // Frees a batch of tokens allocated with llama_batch_init()
# LLAMA_API void llama_batch_free(struct llama_batch batch);
@ctypes_function("llama_batch_free", [llama_batch], None)
def llama_batch_free(batch: llama_batch, /):
    """Frees a batch of tokens allocated with llama_batch_init()"""
    ...


# // Positive return values does not mean a fatal error, but rather a warning.
# //   0 - success
# //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
# // < 0 - error
# LLAMA_API int32_t llama_decode(
#         struct llama_context * ctx,
#           struct llama_batch   batch);
@ctypes_function("llama_decode", [llama_context_p_ctypes, llama_batch], ctypes.c_int32)
def llama_decode(ctx: llama_context_p, batch: llama_batch, /) -> int:
    """Positive return values does not mean a fatal error, but rather a warning.
    0 - success
    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    < 0 - error"""
    ...


# // Set the number of threads used for decoding
# // n_threads is the number of threads used for generation (single token)
# // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
# LLAMA_API void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);
@ctypes_function(
    "llama_set_n_threads",
    [
        llama_context_p_ctypes,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ],
    None,
)
def llama_set_n_threads(
    ctx: llama_context_p,
    n_threads: Union[ctypes.c_uint32, int],
    n_threads_batch: Union[ctypes.c_uint32, int],
    /,
):
    """Set the number of threads used for decoding
    n_threads is the number of threads used for generation (single token)
    n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    """
    ...

# // Set abort callback
# LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);
@ctypes_function(
    "llama_set_abort_callback",
    [llama_context_p_ctypes, ggml_abort_callback, ctypes.c_void_p],
    None,
)
def llama_set_abort_callback(
    ctx: llama_context_p,
    abort_callback: Callable[[ctypes.c_void_p], None],
    abort_callback_data: ctypes.c_void_p,
    /,
):
    """Set abort callback"""
    ...


# // Token logits obtained from the last call to llama_decode()
# // The logits for the last token are stored in the last row
# // Logits for which llama_batch.logits[i] == 0 are undefined
# // Rows: n_tokens provided with llama_batch
# // Cols: n_vocab
# LLAMA_API float * llama_get_logits(struct llama_context * ctx);
@ctypes_function(
    "llama_get_logits", [llama_context_p_ctypes], ctypes.POINTER(ctypes.c_float)
)
def llama_get_logits(ctx: llama_context_p, /) -> CtypesArray[ctypes.c_float]:
    """Token logits obtained from the last call to llama_eval()
    The logits for the last token are stored in the last row
    Logits for which llama_batch.logits[i] == 0 are undefined
    Rows: n_tokens provided with llama_batch
    Cols: n_vocab"""
    ...


# // Logits for the ith token. Equivalent to:
# // llama_get_logits(ctx) + i*n_vocab
# LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
@ctypes_function(
    "llama_get_logits_ith",
    [llama_context_p_ctypes, ctypes.c_int32],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_logits_ith(
    ctx: llama_context_p, i: Union[ctypes.c_int32, int], /
) -> CtypesArray[ctypes.c_float]:
    """Logits for the ith token. Equivalent to:
    llama_get_logits(ctx) + i*n_vocab"""
    ...


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
# LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
@ctypes_function(
    "llama_get_embeddings", [llama_context_p_ctypes], ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings(ctx: llama_context_p, /) -> CtypesArray[ctypes.c_float]:
    """Get the embeddings for the input
    shape: [n_embd] (1-dimensional)"""
    ...


# // Get the embeddings for the ith sequence
# // llama_get_embeddings(ctx) + i*n_embd
# LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
@ctypes_function(
    "llama_get_embeddings_ith",
    [llama_context_p_ctypes, ctypes.c_int32],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_ith(
    ctx: llama_context_p, i: Union[ctypes.c_int32, int], /
) -> CtypesArray[ctypes.c_float]:
    """Get the embeddings for the ith sequence
    llama_get_embeddings(ctx) + i*n_embd"""
    ...


# //
# // Vocab
# //


# LLAMA_API const char * llama_token_get_text(const struct llama_model * model, llama_token token);
@ctypes_function(
    "llama_token_get_text", [llama_model_p_ctypes, llama_token], ctypes.c_char_p
)
def llama_token_get_text(
    model: llama_model_p, token: Union[llama_token, int], /
) -> bytes:
    ...


# LLAMA_API float llama_token_get_score(const struct llama_model * model, llama_token token);
@ctypes_function(
    "llama_token_get_score", [llama_model_p_ctypes, llama_token], ctypes.c_float
)
def llama_token_get_score(
    model: llama_model_p, token: Union[llama_token, int], /
) -> float:
    ...


# LLAMA_API enum llama_token_type llama_token_get_type(const struct llama_model * model, llama_token token);
@ctypes_function(
    "llama_token_get_type", [llama_model_p_ctypes, llama_token], ctypes.c_int
)
def llama_token_get_type(
    model: llama_model_p, token: Union[llama_token, int], /
) -> int:
    ...


# // Special tokens


# LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
@ctypes_function("llama_token_bos", [llama_model_p_ctypes], llama_token)
def llama_token_bos(model: llama_model_p, /) -> int:
    """beginning-of-sentence"""
    ...


# LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence
@ctypes_function("llama_token_eos", [llama_model_p_ctypes], llama_token)
def llama_token_eos(model: llama_model_p, /) -> int:
    """end-of-sentence"""
    ...


# LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line
@ctypes_function("llama_token_nl", [llama_model_p_ctypes], llama_token)
def llama_token_nl(model: llama_model_p, /) -> int:
    """next-line"""
    ...


# // Returns -1 if unknown, 1 for true or 0 for false.
# LLAMA_API int32_t         llama_add_bos_token(const struct llama_model * model);
@ctypes_function("llama_add_bos_token", [llama_model_p_ctypes], ctypes.c_int32)
def llama_add_bos_token(model: llama_model_p, /) -> int:
    """Returns -1 if unknown, 1 for true or 0 for false."""
    ...


# // Returns -1 if unknown, 1 for true or 0 for false.
# LLAMA_API int32_t         llama_add_eos_token(const struct llama_model * model);
@ctypes_function("llama_add_eos_token", [llama_model_p_ctypes], ctypes.c_int32)
def llama_add_eos_token(model: llama_model_p, /) -> int:
    """Returns -1 if unknown, 1 for true or 0 for false."""
    ...


# // codellama infill tokens
# LLAMA_API llama_token llama_token_prefix(const struct llama_model * model); // Beginning of infill prefix
@ctypes_function("llama_token_prefix", [llama_model_p_ctypes], llama_token)
def llama_token_prefix(model: llama_model_p) -> int:
    """codellama infill tokens"""
    ...


# LLAMA_API llama_token llama_token_middle(const struct llama_model * model); // Beginning of infill middle
@ctypes_function("llama_token_middle", [llama_model_p_ctypes], llama_token)
def llama_token_middle(model: llama_model_p, /) -> int:
    ...


# LLAMA_API llama_token llama_token_suffix(const struct llama_model * model); // Beginning of infill suffix
@ctypes_function("llama_token_suffix", [llama_model_p_ctypes], llama_token)
def llama_token_suffix(model: llama_model_p, /) -> int:
    ...


# LLAMA_API llama_token llama_token_eot   (const struct llama_model * model); // End of infill middle
@ctypes_function("llama_token_eot", [llama_model_p_ctypes], llama_token)
def llama_token_eot(model: llama_model_p, /) -> int:
    ...


# //
# // Tokenization
# //


# /// @details Convert the provided text into tokens.
# /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
# /// @return Returns the number of tokens on success, no more than n_max_tokens
# /// @return Returns a negative number on failure - the number of tokens that would have been returned
# /// @param special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
# ///                Does not insert a leading space.
# LLAMA_API int32_t llama_tokenize(
#     const struct llama_model * model,
#                   const char * text,
#                      int32_t   text_len,
#                  llama_token * tokens,
#                      int32_t   n_max_tokens,
#                         bool   add_bos,
#                         bool   special);
@ctypes_function(
    "llama_tokenize",
    [
        llama_model_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int32,
        llama_token_p,
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ],
    ctypes.c_int32,
)
def llama_tokenize(
    model: llama_model_p,
    text: bytes,
    text_len: Union[ctypes.c_int, int],
    tokens: CtypesArray[llama_token],
    n_max_tokens: Union[ctypes.c_int, int],
    add_bos: Union[ctypes.c_bool, bool],
    special: Union[ctypes.c_bool, bool],
    /,
) -> int:
    """Convert the provided text into tokens."""
    ...


# // Token Id -> Piece.
# // Uses the vocabulary in the provided context.
# // Does not write null terminator to the buffer.
# // User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
# LLAMA_API int32_t llama_token_to_piece(
#           const struct llama_model * model,
#                        llama_token   token,
#                               char * buf,
#                            int32_t   length);
@ctypes_function(
    "llama_token_to_piece",
    [
        llama_model_p_ctypes,
        llama_token,
        ctypes.c_char_p,
        ctypes.c_int32,
    ],
    ctypes.c_int32,
)
def llama_token_to_piece(
    model: llama_model_p,
    token: Union[llama_token, int],
    buf: Union[ctypes.c_char_p, bytes, CtypesArray[ctypes.c_char]],
    length: Union[ctypes.c_int, int],
    /,
) -> int:
    """Token Id -> Piece.
    Uses the vocabulary in the provided context.
    Does not write null terminator to the buffer.
    User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    """
    ...


# /// Apply chat template. Inspired by hf apply_chat_template() on python.
# /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
# /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
# /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
# /// @param chat Pointer to a list of multiple llama_chat_message
# /// @param n_msg Number of llama_chat_message in this chat
# /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
# /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
# /// @param length The size of the allocated buffer
# /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
# LLAMA_API int32_t llama_chat_apply_template(
#           const struct llama_model * model,
#                         const char * tmpl,
#    const struct llama_chat_message * chat,
#                             size_t   n_msg,
#                               bool   add_ass,
#                               char * buf,
#                            int32_t   length);
@ctypes_function(
    "llama_chat_apply_template",
    [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(llama_chat_message),
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_chat_apply_template(
    model: llama_model_p,
    tmpl: bytes,
    chat: CtypesArray[llama_chat_message],
    n_msg: int,
    /,
) -> int:
    ...


# //
# // Grammar
# //


# LLAMA_API struct llama_grammar * llama_grammar_init(
#         const llama_grammar_element ** rules,
#                                 size_t    n_rules,
#                                 size_t    start_rule_index);
@ctypes_function(
    "llama_grammar_init",
    [
        ctypes.POINTER(llama_grammar_element_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    llama_grammar_p,
)
def llama_grammar_init(
    rules: CtypesArray[
        CtypesPointer[llama_grammar_element]
    ],  # NOTE: This might be wrong type sig
    n_rules: Union[ctypes.c_size_t, int],
    start_rule_index: Union[ctypes.c_size_t, int],
    /,
) -> llama_grammar_p:
    """Initialize a grammar from a set of rules."""
    ...


# LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);
@ctypes_function(
    "llama_grammar_free",
    [llama_grammar_p],
    None,
)
def llama_grammar_free(grammar: llama_grammar_p, /):
    """Free a grammar."""
    ...


# LLAMA_API struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);
@ctypes_function(
    "llama_grammar_copy",
    [llama_grammar_p],
    llama_grammar_p,
)
def llama_grammar_copy(grammar: llama_grammar_p, /) -> llama_grammar_p:
    """Copy a grammar."""
    ...


# //
# // Sampling functions
# //


# // Sets the current rng seed.
# LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);
@ctypes_function(
    "llama_set_rng_seed",
    [llama_context_p_ctypes, ctypes.c_uint32],
    None,
)
def llama_set_rng_seed(ctx: llama_context_p, seed: Union[ctypes.c_uint32, int], /):
    """Sets the current rng seed."""
    ...


# /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
# /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
# LLAMA_API void llama_sample_repetition_penalties(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#            const llama_token * last_tokens,
#                       size_t   penalty_last_n,
#                        float   penalty_repeat,
#                        float   penalty_freq,
#                        float   penalty_present);
@ctypes_function(
    "llama_sample_repetition_penalties",
    [
        llama_context_p_ctypes,
        llama_token_data_array_p,
        llama_token_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    None,
)
def llama_sample_repetition_penalties(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    last_tokens_data: CtypesArray[llama_token],
    penalty_last_n: Union[ctypes.c_size_t, int],
    penalty_repeat: Union[ctypes.c_float, float],
    penalty_freq: Union[ctypes.c_float, float],
    penalty_present: Union[ctypes.c_float, float],
    /,
):
    """Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    """
    ...


# /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
# /// @param logits Logits extracted from the original generation context.
# /// @param logits_guidance Logits extracted from a separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
# /// @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
# LLAMA_API void llama_sample_apply_guidance(
#           struct llama_context * ctx,
#                          float * logits,
#                          float * logits_guidance,
#                          float   scale);
@ctypes_function(
    "llama_sample_apply_guidance",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ],
    None,
)
def llama_sample_apply_guidance(
    ctx: llama_context_p,
    logits: CtypesArray[ctypes.c_float],
    logits_guidance: CtypesArray[ctypes.c_float],
    scale: Union[ctypes.c_float, float],
    /,
):
    """Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806"""
    ...


# /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
# LLAMA_API void llama_sample_softmax(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates);
@ctypes_function(
    "llama_sample_softmax",
    [llama_context_p_ctypes, llama_token_data_array_p],
    None,
)
def llama_sample_softmax(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    /,
):
    """Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits."""
    ...


# /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_k(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                      int32_t   k,
#                       size_t   min_keep);
@ctypes_function(
    "llama_sample_top_k",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_int32, ctypes.c_size_t],
    None,
)
def llama_sample_top_k(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    k: Union[ctypes.c_int, int],
    min_keep: Union[ctypes.c_size_t, int],
    /,
):
    """Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751"""
    ...


# /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_p(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   p,
#                       size_t   min_keep);
@ctypes_function(
    "llama_sample_top_p",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_float, ctypes.c_size_t],
    None,
)
def llama_sample_top_p(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    p: Union[ctypes.c_float, float],
    min_keep: Union[ctypes.c_size_t, int],
    /,
):
    """Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751"""
    ...


# /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
# LLAMA_API void llama_sample_min_p(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   p,
#                       size_t   min_keep);
@ctypes_function(
    "llama_sample_min_p",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_float, ctypes.c_size_t],
    None,
)
def llama_sample_min_p(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    p: Union[ctypes.c_float, float],
    min_keep: Union[ctypes.c_size_t, int],
    /,
):
    """Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841"""
    ...


# /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
# LLAMA_API void llama_sample_tail_free(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   z,
#                       size_t   min_keep);
@ctypes_function(
    "llama_sample_tail_free",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_float, ctypes.c_size_t],
    None,
)
def llama_sample_tail_free(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    z: Union[ctypes.c_float, float],
    min_keep: Union[ctypes.c_size_t, int],
    /,
):
    """Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/."""
    ...


# /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
# LLAMA_API void llama_sample_typical(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   p,
#                       size_t   min_keep);
@ctypes_function(
    "llama_sample_typical",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_float, ctypes.c_size_t],
    None,
)
def llama_sample_typical(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    p: Union[ctypes.c_float, float],
    min_keep: Union[ctypes.c_size_t, int],
    /,
):
    """Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666."""
    ...


# /// @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.
# LLAMA_API void llama_sample_entropy(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates_p,
#                        float   min_temp,
#                        float   max_temp,
#                        float   exponent_val);
@ctypes_function(
    "llama_sample_entropy",
    [
        llama_context_p_ctypes,
        llama_token_data_array_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    None,
)
def llama_sample_entropy(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    min_temp: Union[ctypes.c_float, float],
    max_temp: Union[ctypes.c_float, float],
    exponent_val: Union[ctypes.c_float, float],
    /,
):
    """Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772."""
    ...


# LLAMA_API void llama_sample_temp(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   temp);
@ctypes_function(
    "llama_sample_temp",
    [llama_context_p_ctypes, llama_token_data_array_p, ctypes.c_float],
    None,
)
def llama_sample_temp(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    temp: Union[ctypes.c_float, float],
    /,
):
    """Temperature sampling described in academic paper "Generating Long Sequences with Sparse Transformers" https://arxiv.org/abs/1904.10509

    Parameters:
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        temp: The temperature value to use for the sampling. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    """
    ...


# /// @details Apply constraints from grammar
# LLAMA_API void llama_sample_grammar(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#   const struct llama_grammar * grammar);
@ctypes_function(
    "llama_sample_grammar",
    [llama_context_p_ctypes, llama_token_data_array_p, llama_grammar_p],
    None,
)
def llama_sample_grammar(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    grammar,  # type: llama_grammar_p
    /,
):
    """Apply constraints from grammar

    Parameters:
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        grammar: A grammar object containing the rules and constraints to apply to the generated text.
    """
    ...


# /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   tau,
#                        float   eta,
#                      int32_t   m,
#                        float * mu);
@ctypes_function(
    "llama_sample_token_mirostat",
    [
        llama_context_p_ctypes,
        llama_token_data_array_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ],
    llama_token,
)
def llama_sample_token_mirostat(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    tau: Union[ctypes.c_float, float],
    eta: Union[ctypes.c_float, float],
    m: Union[ctypes.c_int, int],
    mu: CtypesPointerOrRef[ctypes.c_float],
    /,
) -> int:
    """Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

    Parameters:
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        m: The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
        mu: Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    """
    ...


# /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat_v2(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#                        float   tau,
#                        float   eta,
#                        float * mu);
@ctypes_function(
    "llama_sample_token_mirostat_v2",
    [
        llama_context_p_ctypes,
        llama_token_data_array_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ],
    llama_token,
)
def llama_sample_token_mirostat_v2(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    tau: Union[ctypes.c_float, float],
    eta: Union[ctypes.c_float, float],
    mu: CtypesPointerOrRef[ctypes.c_float],
    /,
) -> int:
    """Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

    Parameters:
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        mu: Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    """
    ...


# /// @details Selects the token with the highest probability.
# ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
# LLAMA_API llama_token llama_sample_token_greedy(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates);
@ctypes_function(
    "llama_sample_token_greedy",
    [llama_context_p_ctypes, llama_token_data_array_p],
    llama_token,
)
def llama_sample_token_greedy(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    /,
) -> int:
    """Selects the token with the highest probability."""
    ...


# /// @details Randomly selects a token from the candidates based on their probabilities.
# LLAMA_API llama_token llama_sample_token(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates);
@ctypes_function(
    "llama_sample_token",
    [llama_context_p_ctypes, llama_token_data_array_p],
    llama_token,
)
def llama_sample_token(
    ctx: llama_context_p,
    candidates: Union[
        CtypesArray[llama_token_data_array], CtypesPointerOrRef[llama_token_data_array]
    ],
    /,
) -> int:
    """Randomly selects a token from the candidates based on their probabilities."""
    ...


# /// @details Accepts the sampled token into the grammar
# LLAMA_API void llama_grammar_accept_token(
#         struct llama_context * ctx,
#         struct llama_grammar * grammar,
#                  llama_token   token);
@ctypes_function(
    "llama_grammar_accept_token",
    [llama_context_p_ctypes, llama_grammar_p, llama_token],
    None,
)
def llama_grammar_accept_token(
    ctx: llama_context_p, grammar: llama_grammar_p, token: Union[llama_token, int], /
) -> None:
    """Accepts the sampled token into the grammar"""
    ...


# //
# // Beam search
# //

# struct llama_beam_view {
#     const llama_token * tokens;


#     size_t n_tokens;
#     float  p;        // Cumulative beam probability (renormalized relative to all beams)
#     bool   eob;      // Callback should set this to true when a beam is at end-of-beam.
# };
class llama_beam_view(ctypes.Structure):
    _fields_ = [
        ("tokens", llama_token_p),
        ("n_tokens", ctypes.c_size_t),
        ("p", ctypes.c_float),
        ("eob", ctypes.c_bool),
    ]


# // Passed to beam_search_callback function.
# // Whenever 0 < common_prefix_length, this number of tokens should be copied from any of the beams
# // (e.g. beams[0]) as they will be removed (shifted) from all beams in all subsequent callbacks.
# // These pointers are valid only during the synchronous callback, so should not be saved.
# struct llama_beams_state {
#     struct llama_beam_view * beam_views;
#     size_t n_beams;               // Number of elements in beam_views[].
#     size_t common_prefix_length;  // Current max length of prefix tokens shared by all beams.
#     bool   last_call;             // True iff this is the last callback invocation.
# };
class llama_beams_state(ctypes.Structure):
    _fields_ = [
        ("beam_views", ctypes.POINTER(llama_beam_view)),
        ("n_beams", ctypes.c_size_t),
        ("common_prefix_length", ctypes.c_size_t),
        ("last_call", ctypes.c_bool),
    ]


# // Type of pointer to the beam_search_callback function.
# // void* callback_data is any custom data passed to llama_beam_search, that is subsequently
# // passed back to beam_search_callback. This avoids having to use global variables in the callback.
# typedef void (*llama_beam_search_callback_fn_t)(void * callback_data, struct llama_beams_state);
llama_beam_search_callback_fn_t = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, llama_beams_state
)


# /// @details Deterministically returns entire sentence constructed by a beam search.
# /// @param ctx Pointer to the llama_context.
# /// @param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
# /// @param callback_data A pointer that is simply passed back to callback.
# /// @param n_beams Number of beams to use.
# /// @param n_past Number of tokens already evaluated.
# /// @param n_predict Maximum number of tokens to predict. EOS may occur earlier.
# /// @param n_threads Number of threads as passed to llama_eval().
# LLAMA_API void llama_beam_search(
#                struct llama_context * ctx,
#     llama_beam_search_callback_fn_t   callback,
#                                void * callback_data,
#                              size_t   n_beams,
#                             int32_t   n_past,
#                             int32_t   n_predict);
@ctypes_function(
    "llama_beam_search",
    [
        llama_context_p_ctypes,
        llama_beam_search_callback_fn_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    None,
)
def llama_beam_search(
    ctx: llama_context_p,
    callback: CtypesFuncPointer,
    callback_data: ctypes.c_void_p,
    n_beams: Union[ctypes.c_size_t, int],
    n_past: Union[ctypes.c_int, int],
    n_predict: Union[ctypes.c_int, int],
    /,
):
    ...


# Performance information


# LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);
@ctypes_function(
    "llama_get_timings",
    [llama_context_p_ctypes],
    llama_timings,
)
def llama_get_timings(ctx: llama_context_p, /) -> llama_timings:
    """Get performance information"""
    ...


# LLAMA_API void llama_print_timings(struct llama_context * ctx);
@ctypes_function(
    "llama_print_timings",
    [llama_context_p_ctypes],
    None,
)
def llama_print_timings(ctx: llama_context_p, /):
    """Print performance information"""
    ...


# LLAMA_API void llama_reset_timings(struct llama_context * ctx);
@ctypes_function(
    "llama_reset_timings",
    [llama_context_p_ctypes],
    None,
)
def llama_reset_timings(ctx: llama_context_p, /):
    """Reset performance information"""
    ...


# Print system information
# LLAMA_API const char * llama_print_system_info(void);
@ctypes_function(
    "llama_print_system_info",
    [],
    ctypes.c_char_p,
)
def llama_print_system_info() -> bytes:
    """Print system information"""
    ...


# NOTE: THIS IS CURRENTLY BROKEN AS ggml_log_callback IS NOT EXPOSED IN LLAMA.H
# // Set callback for all future logging events.
# // If this is not called, or NULL is supplied, everything is output on stderr.
# LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);
@ctypes_function(
    "llama_log_set",
    [ctypes.c_void_p, ctypes.c_void_p],
    None,
)
def llama_log_set(
    log_callback: Optional[CtypesFuncPointer],
    user_data: ctypes.c_void_p,
    /,
):
    """Set callback for all future logging events.

    If this is not called, or NULL is supplied, everything is output on stderr."""
    ...


# LLAMA_API void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);
@ctypes_function(
    "llama_dump_timing_info_yaml",
    [ctypes.c_void_p, llama_context_p_ctypes],
    None,
)
def llama_dump_timing_info_yaml(stream: ctypes.c_void_p, ctx: llama_context_p, /):
    ...
