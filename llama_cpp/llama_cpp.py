import sys
import os
import ctypes
from ctypes import (
    c_double,
    c_int,
    c_float,
    c_char_p,
    c_int32,
    c_uint32,
    c_void_p,
    c_bool,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib
from typing import List, Union


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
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
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

# llama.h bindings

GGML_USE_CUBLAS = hasattr(_lib, "ggml_init_cublas")
GGML_CUDA_MAX_DEVICES = ctypes.c_int(16)
LLAMA_MAX_DEVICES = GGML_CUDA_MAX_DEVICES if GGML_USE_CUBLAS else ctypes.c_int(1)

# #define LLAMA_FILE_MAGIC_GGJT        0x67676a74u // 'ggjt'
LLAMA_FILE_MAGIC_GGJT = ctypes.c_uint(0x67676A74)
# #define LLAMA_FILE_MAGIC_GGLA        0x67676c61u // 'ggla'
LLAMA_FILE_MAGIC_GGLA = ctypes.c_uint(0x67676C61)
# #define LLAMA_FILE_MAGIC_GGMF        0x67676d66u // 'ggmf'
LLAMA_FILE_MAGIC_GGMF = ctypes.c_uint(0x67676D66)
# #define LLAMA_FILE_MAGIC_GGML        0x67676d6cu // 'ggml'
LLAMA_FILE_MAGIC_GGML = ctypes.c_uint(0x67676D6C)
# #define LLAMA_FILE_MAGIC_GGSN        0x6767736eu // 'ggsn'
LLAMA_FILE_MAGIC_GGSN = ctypes.c_uint(0x6767736E)

# #define LLAMA_FILE_VERSION           3
LLAMA_FILE_VERSION = c_int(3)
LLAMA_FILE_MAGIC = LLAMA_FILE_MAGIC_GGJT
LLAMA_FILE_MAGIC_UNVERSIONED = LLAMA_FILE_MAGIC_GGML
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = c_int(1)

# #define LLAMA_DEFAULT_SEED           0xFFFFFFFF
LLAMA_DEFAULT_SEED = c_int(0xFFFFFFFF)

# struct llama_model;
llama_model_p = c_void_p

# struct llama_context;
llama_context_p = c_void_p


# typedef int llama_token;
llama_token = c_int
llama_token_p = POINTER(llama_token)


# typedef struct llama_token_data {
#     llama_token id; // token id
#     float logit;    // log-odds of the token
#     float p;        // probability of the token
# } llama_token_data;
class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),
        ("logit", c_float),
        ("p", c_float),
    ]


llama_token_data_p = POINTER(llama_token_data)


# typedef struct llama_token_data_array {
#     llama_token_data * data;
#     size_t size;
#     bool sorted;
# } llama_token_data_array;
class llama_token_data_array(Structure):
    _fields_ = [
        ("data", llama_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


llama_token_data_array_p = POINTER(llama_token_data_array)

# typedef void (*llama_progress_callback)(float progress, void *ctx);
llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)


# struct llama_context_params {
#     uint32_t seed;         // RNG seed, -1 for random
#     int32_t  n_ctx;        // text context
#     int32_t  n_batch;      // prompt processing batch size
#     int32_t  n_gqa;        // grouped-query attention (TEMP - will be moved to model hparams)
#     float    rms_norm_eps; // rms norm epsilon (TEMP - will be moved to model hparams)
#     int32_t  n_gpu_layers; // number of layers to store in VRAM
#     int32_t  main_gpu;     // the GPU that is used for scratch and small tensors
#
#     const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)

#     // ref: https://github.com/ggerganov/llama.cpp/pull/2054
#     float    rope_freq_base;  // RoPE base frequency
#     float    rope_freq_scale; // RoPE frequency scaling factor

#     // called with a progress value between 0 and 1, pass NULL to disable
#     llama_progress_callback progress_callback;
#     // context pointer passed to the progress callback
#     void * progress_callback_user_data;


#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool low_vram;   // if true, reduce VRAM usage at the cost of performance
#     bool f16_kv;     // use fp16 for KV cache
#     bool logits_all; // the llama_eval() call computes all logits, not just the last one
#     bool vocab_only; // only load the vocabulary, no weights
#     bool use_mmap;   // use mmap if possible
#     bool use_mlock;  // force system to keep model in RAM
#     bool embedding;  // embedding mode only
# };
class llama_context_params(Structure):
    _fields_ = [
        ("seed", c_uint32),
        ("n_ctx", c_int32),
        ("n_batch", c_int32),
        ("n_gqa", c_int32),
        ("rms_norm_eps", c_float),
        ("n_gpu_layers", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", c_void_p),
        ("low_vram", c_bool),
        ("f16_kv", c_bool),
        ("logits_all", c_bool),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("embedding", c_bool),
    ]


llama_context_params_p = POINTER(llama_context_params)

# enum llama_ftype {
#     LLAMA_FTYPE_ALL_F32              = 0,
#     LLAMA_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     // LLAMA_FTYPE_MOSTLY_Q4_2       = 5, // support has been removed
#     // LLAMA_FTYPE_MOSTLY_Q4_3       = 6, // support has been removed
#     LLAMA_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q2_K          = 10,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,// except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q6_K          = 18,// except 1d tensors
# };
LLAMA_FTYPE_ALL_F32 = c_int(0)
LLAMA_FTYPE_MOSTLY_F16 = c_int(1)
LLAMA_FTYPE_MOSTLY_Q4_0 = c_int(2)
LLAMA_FTYPE_MOSTLY_Q4_1 = c_int(3)
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(4)
LLAMA_FTYPE_MOSTLY_Q8_0 = c_int(7)
LLAMA_FTYPE_MOSTLY_Q5_0 = c_int(8)
LLAMA_FTYPE_MOSTLY_Q5_1 = c_int(9)
LLAMA_FTYPE_MOSTLY_Q2_K = c_int(10)
LLAMA_FTYPE_MOSTLY_Q3_K_S = c_int(11)
LLAMA_FTYPE_MOSTLY_Q3_K_M = c_int(12)
LLAMA_FTYPE_MOSTLY_Q3_K_L = c_int(13)
LLAMA_FTYPE_MOSTLY_Q4_K_S = c_int(14)
LLAMA_FTYPE_MOSTLY_Q4_K_M = c_int(15)
LLAMA_FTYPE_MOSTLY_Q5_K_S = c_int(16)
LLAMA_FTYPE_MOSTLY_Q5_K_M = c_int(17)
LLAMA_FTYPE_MOSTLY_Q6_K = c_int(18)


# // model quantization parameters
# typedef struct llama_model_quantize_params {
#     int nthread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
#     enum llama_ftype   ftype;    // quantize to this llama_ftype
#     bool allow_requantize;       // allow quantizing non-f32/f16 tensors
#     bool quantize_output_tensor; // quantize output.weight
# } llama_model_quantize_params;
class llama_model_quantize_params(Structure):
    _fields_ = [
        ("nthread", c_int),
        ("ftype", c_int),
        ("allow_requantize", c_bool),
        ("quantize_output_tensor", c_bool),
    ]


# // grammar types
# struct llama_grammar;
llama_grammar_p = c_void_p

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
LLAMA_GRETYPE_END = c_int(0)
LLAMA_GRETYPE_ALT = c_int(1)
LLAMA_GRETYPE_RULE_REF = c_int(2)
LLAMA_GRETYPE_CHAR = c_int(3)
LLAMA_GRETYPE_CHAR_NOT = c_int(4)
LLAMA_GRETYPE_CHAR_RNG_UPPER = c_int(5)
LLAMA_GRETYPE_CHAR_ALT = c_int(6)


# typedef struct llama_grammar_element {
#     enum llama_gretype type;
#     uint32_t           value; // Unicode code point or rule ID
# } llama_grammar_element;
class llama_grammar_element(Structure):
    _fields_ = [
        ("type", c_int),
        ("value", c_uint32),
    ]


llama_grammar_element_p = POINTER(llama_grammar_element)

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
class llama_timings(Structure):
    _fields_ = [
        ("t_start_ms", c_double),
        ("t_end_ms", c_double),
        ("t_load_ms", c_double),
        ("t_sample_ms", c_double),
        ("t_p_eval_ms", c_double),
        ("t_eval_ms", c_double),
        ("n_sample", c_int32),
        ("n_p_eval", c_int32),
        ("n_eval", c_int32),
    ]


# LLAMA_API int llama_max_devices();
def llama_max_devices() -> int:
    return _lib.llama_max_devices()


_lib.llama_max_devices.argtypes = []
_lib.llama_max_devices.restype = c_int


# LLAMA_API struct llama_context_params llama_context_default_params();
def llama_context_default_params() -> llama_context_params:
    return _lib.llama_context_default_params()


_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params


# LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params();
def llama_model_quantize_default_params() -> llama_model_quantize_params:
    return _lib.llama_model_quantize_default_params()


_lib.llama_model_quantize_default_params.argtypes = []
_lib.llama_model_quantize_default_params.restype = llama_model_quantize_params


# LLAMA_API bool llama_mmap_supported();
def llama_mmap_supported() -> bool:
    return _lib.llama_mmap_supported()


_lib.llama_mmap_supported.argtypes = []
_lib.llama_mmap_supported.restype = c_bool


# LLAMA_API bool llama_mlock_supported();
def llama_mlock_supported() -> bool:
    return _lib.llama_mlock_supported()


_lib.llama_mlock_supported.argtypes = []
_lib.llama_mlock_supported.restype = c_bool


# // TODO: not great API - very likely to change
# // Initialize the llama + ggml backend
# // If numa is true, use NUMA optimizations
# // Call once at the start of the program
# LLAMA_API void llama_backend_init(bool numa);
def llama_backend_init(numa: c_bool):
    return _lib.llama_backend_init(numa)


_lib.llama_backend_init.argtypes = [c_bool]
_lib.llama_backend_init.restype = None


# // Call once at the end of the program - currently only used for MPI
# LLAMA_API void llama_backend_free();
def llama_backend_free():
    return _lib.llama_backend_free()


_lib.llama_backend_free.argtypes = []
_lib.llama_backend_free.restype = None


# LLAMA_API struct llama_model * llama_load_model_from_file(
#                             const char * path_model,
#         struct llama_context_params   params);
def llama_load_model_from_file(
    path_model: bytes, params: llama_context_params
) -> llama_model_p:
    return _lib.llama_load_model_from_file(path_model, params)


_lib.llama_load_model_from_file.argtypes = [c_char_p, llama_context_params]
_lib.llama_load_model_from_file.restype = llama_model_p


# LLAMA_API void llama_free_model(struct llama_model * model);
def llama_free_model(model: llama_model_p):
    return _lib.llama_free_model(model)


_lib.llama_free_model.argtypes = [llama_model_p]
_lib.llama_free_model.restype = None


# LLAMA_API struct llama_context * llama_new_context_with_model(
#                     struct llama_model * model,
#         struct llama_context_params   params);
def llama_new_context_with_model(
    model: llama_model_p, params: llama_context_params
) -> llama_context_p:
    return _lib.llama_new_context_with_model(model, params)


_lib.llama_new_context_with_model.argtypes = [llama_model_p, llama_context_params]
_lib.llama_new_context_with_model.restype = llama_context_p


# LLAMA_API int64_t llama_time_us();
def llama_time_us() -> int:
    return _lib.llama_time_us()


_lib.llama_time_us.argtypes = []
_lib.llama_time_us.restype = ctypes.c_int64


# // Various functions for loading a ggml llama model.
# // Allocate (almost) all memory needed for the model.
# // Return NULL on failure
# LLAMA_API struct llama_context * llama_init_from_file(
#                             const char * path_model,
#         struct llama_context_params   params);
def llama_init_from_file(
    path_model: bytes, params: llama_context_params
) -> llama_context_p:
    return _lib.llama_init_from_file(path_model, params)


_lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
_lib.llama_init_from_file.restype = llama_context_p


# Frees all allocated memory
# LLAMA_API void llama_free(struct llama_context * ctx);
def llama_free(ctx: llama_context_p):
    return _lib.llama_free(ctx)


_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None


# // Returns 0 on success
# LLAMA_API int llama_model_quantize(
#         const char * fname_inp,
#         const char * fname_out,
#         const llama_model_quantize_params * params);
def llama_model_quantize(
    fname_inp: bytes,
    fname_out: bytes,
    params,  # type: POINTER(llama_model_quantize_params) # type: ignore
) -> int:
    return _lib.llama_model_quantize(fname_inp, fname_out, params)


_lib.llama_model_quantize.argtypes = [
    c_char_p,
    c_char_p,
    POINTER(llama_model_quantize_params),
]
_lib.llama_model_quantize.restype = c_int


# Apply a LoRA adapter to a loaded model
# path_base_model is the path to a higher quality model to use as a base for
# the layers modified by the adapter. Can be NULL to use the current loaded model.
# The model needs to be reloaded before applying a new adapter, otherwise the adapter
# will be applied on top of the previous one
# Returns 0 on success
# LLAMA_API int llama_apply_lora_from_file(
#         struct llama_context * ctx,
#                   const char * path_lora,
#                   const char * path_base_model,
#                          int   n_threads);
def llama_apply_lora_from_file(
    ctx: llama_context_p,
    path_lora: c_char_p,
    path_base_model: c_char_p,
    n_threads: c_int,
) -> int:
    return _lib.llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


_lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
_lib.llama_apply_lora_from_file.restype = c_int


# LLAMA_API int llama_model_apply_lora_from_file(
#         const struct llama_model * model,
#                     const char * path_lora,
#                     const char * path_base_model,
#                             int   n_threads);
def llama_model_apply_lora_from_file(
    model: llama_model_p,
    path_lora: Union[c_char_p, bytes],
    path_base_model: Union[c_char_p, bytes],
    n_threads: c_int,
) -> int:
    return _lib.llama_model_apply_lora_from_file(
        model, path_lora, path_base_model, n_threads
    )


_lib.llama_model_apply_lora_from_file.argtypes = [
    llama_model_p,
    c_char_p,
    c_char_p,
    c_int,
]
_lib.llama_model_apply_lora_from_file.restype = c_int


# Returns the number of tokens in the KV cache
# LLAMA_API int llama_get_kv_cache_token_count(const struct llama_context * ctx);
def llama_get_kv_cache_token_count(ctx: llama_context_p) -> int:
    return _lib.llama_get_kv_cache_token_count(ctx)


_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int


# Sets the current rng seed.
# LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, int seed);
def llama_set_rng_seed(ctx: llama_context_p, seed: c_uint32):
    return _lib.llama_set_rng_seed(ctx, seed)


_lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
_lib.llama_set_rng_seed.restype = None


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
# LLAMA_API size_t llama_get_state_size(const struct llama_context * ctx);
def llama_get_state_size(ctx: llama_context_p) -> int:
    return _lib.llama_get_state_size(ctx)


_lib.llama_get_state_size.argtypes = [llama_context_p]
_lib.llama_get_state_size.restype = c_size_t


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
# LLAMA_API size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst);
def llama_copy_state_data(
    ctx: llama_context_p, dst  # type: Array[c_uint8]
) -> int:
    return _lib.llama_copy_state_data(ctx, dst)


_lib.llama_copy_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_copy_state_data.restype = c_size_t


# Set the state reading from the specified address
# Returns the number of bytes read
# LLAMA_API size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src);
def llama_set_state_data(
    ctx: llama_context_p, src  # type: Array[c_uint8]
) -> int:
    return _lib.llama_set_state_data(ctx, src)


_lib.llama_set_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_set_state_data.restype = c_size_t


# Save/load session file
# LLAMA_API bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
def llama_load_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens_out,  # type: Array[llama_token]
    n_token_capacity: c_size_t,
    n_token_count_out,  # type: _Pointer[c_size_t]
) -> int:
    return _lib.llama_load_session_file(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out
    )


_lib.llama_load_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
    c_size_t_p,
]
_lib.llama_load_session_file.restype = c_size_t


# LLAMA_API bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
def llama_save_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens,  # type: Array[llama_token]
    n_token_count: c_size_t,
) -> int:
    return _lib.llama_save_session_file(ctx, path_session, tokens, n_token_count)


_lib.llama_save_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
]
_lib.llama_save_session_file.restype = c_size_t


# Run the llama inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
# LLAMA_API int llama_eval(
#         struct llama_context * ctx,
#            const llama_token * tokens,
#                          int   n_tokens,
#                          int   n_past,
#                          int   n_threads);
def llama_eval(
    ctx: llama_context_p,
    tokens,  # type: Array[llama_token]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> int:
    return _lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)


_lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
_lib.llama_eval.restype = c_int


# // Same as llama_eval, but use float matrix input directly.
# LLAMA_API int llama_eval_embd(
#         struct llama_context * ctx,
#                     const float * embd,
#                             int   n_tokens,
#                             int   n_past,
#                             int   n_threads);
def llama_eval_embd(
    ctx: llama_context_p,
    embd,  # type: Array[c_float]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> int:
    return _lib.llama_eval_embd(ctx, embd, n_tokens, n_past, n_threads)


_lib.llama_eval_embd.argtypes = [llama_context_p, c_float_p, c_int, c_int, c_int]
_lib.llama_eval_embd.restype = c_int


# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
# LLAMA_API int llama_tokenize(
#         struct llama_context * ctx,
#                   const char * text,
#                  llama_token * tokens,
#                          int   n_max_tokens,
#                         bool   add_bos);
def llama_tokenize(
    ctx: llama_context_p,
    text: bytes,
    tokens,  # type: Array[llama_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> int:
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


_lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
_lib.llama_tokenize.restype = c_int


# LLAMA_API int llama_tokenize_with_model(
#     const struct llama_model * model,
#                     const char * text,
#                     llama_token * tokens,
#                             int   n_max_tokens,
#                         bool   add_bos);
def llama_tokenize_with_model(
    model: llama_model_p,
    text: bytes,
    tokens,  # type: Array[llama_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> int:
    return _lib.llama_tokenize_with_model(model, text, tokens, n_max_tokens, add_bos)


# LLAMA_API int llama_n_vocab(const struct llama_context * ctx);
def llama_n_vocab(ctx: llama_context_p) -> int:
    return _lib.llama_n_vocab(ctx)


_lib.llama_n_vocab.argtypes = [llama_context_p]
_lib.llama_n_vocab.restype = c_int


# LLAMA_API int llama_n_ctx  (const struct llama_context * ctx);
def llama_n_ctx(ctx: llama_context_p) -> int:
    return _lib.llama_n_ctx(ctx)


_lib.llama_n_ctx.argtypes = [llama_context_p]
_lib.llama_n_ctx.restype = c_int


# LLAMA_API int llama_n_embd (const struct llama_context * ctx);
def llama_n_embd(ctx: llama_context_p) -> int:
    return _lib.llama_n_embd(ctx)


_lib.llama_n_embd.argtypes = [llama_context_p]
_lib.llama_n_embd.restype = c_int


# LLAMA_API int llama_n_vocab_from_model(const struct llama_model * model);
def llama_n_vocab_from_model(model: llama_model_p) -> int:
    return _lib.llama_n_vocab_from_model(model)


_lib.llama_n_vocab_from_model.argtypes = [llama_model_p]
_lib.llama_n_vocab_from_model.restype = c_int


# LLAMA_API int llama_n_ctx_from_model  (const struct llama_model * model);
def llama_n_ctx_from_model(model: llama_model_p) -> int:
    return _lib.llama_n_ctx_from_model(model)


_lib.llama_n_ctx_from_model.argtypes = [llama_model_p]
_lib.llama_n_ctx_from_model.restype = c_int


# LLAMA_API int llama_n_embd_from_model (const struct llama_model * model);
def llama_n_embd_from_model(model: llama_model_p) -> int:
    return _lib.llama_n_embd_from_model(model)


_lib.llama_n_embd_from_model.argtypes = [llama_model_p]
_lib.llama_n_embd_from_model.restype = c_int


# // Get the vocabulary as output parameters.
# // Returns number of results.
# LLAMA_API int llama_get_vocab(
#         const struct llama_context * ctx,
#                         const char * * strings,
#                                 float * scores,
#                                 int   capacity);
def llama_get_vocab(
    ctx: llama_context_p,
    strings,  # type: Array[c_char_p] # type: ignore
    scores,  # type: Array[c_float] # type: ignore
    capacity: c_int,
) -> int:
    return _lib.llama_get_vocab(ctx, strings, scores, capacity)


_lib.llama_get_vocab.argtypes = [
    llama_context_p,
    POINTER(c_char_p),
    POINTER(c_float),
    c_int,
]
_lib.llama_get_vocab.restype = c_int


# LLAMA_API int llama_get_vocab_from_model(
#             const struct llama_model * model,
#                         const char * * strings,
#                                 float * scores,
#                                 int   capacity);
def llama_get_vocab_from_model(
    model: llama_model_p,
    strings,  # type: Array[c_char_p] # type: ignore
    scores,  # type: Array[c_float] # type: ignore
    capacity: c_int,
) -> int:
    return _lib.llama_get_vocab_from_model(model, strings, scores, capacity)


_lib.llama_get_vocab_from_model.argtypes = [
    llama_model_p,
    POINTER(c_char_p),
    POINTER(c_float),
    c_int,
]
_lib.llama_get_vocab_from_model.restype = c_int


# Token logits obtained from the last call to llama_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
# LLAMA_API float * llama_get_logits(struct llama_context * ctx);
def llama_get_logits(
    ctx: llama_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.llama_get_logits(ctx)


_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = c_float_p


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
# LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
def llama_get_embeddings(
    ctx: llama_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.llama_get_embeddings(ctx)


_lib.llama_get_embeddings.argtypes = [llama_context_p]
_lib.llama_get_embeddings.restype = c_float_p


# // Token Id -> String. Uses the vocabulary in the provided context
# LLAMA_API const char * llama_token_to_str(
#         const struct llama_context * ctx,
#                         llama_token   token);
def llama_token_to_str(ctx: llama_context_p, token: llama_token) -> bytes:
    return _lib.llama_token_to_str(ctx, token)


_lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
_lib.llama_token_to_str.restype = c_char_p


# LLAMA_API const char * llama_token_to_str_with_model(
#             const struct llama_model * model,
#                         llama_token   token);
def llama_token_to_str_with_model(model: llama_model_p, token: llama_token) -> bytes:
    return _lib.llama_token_to_str_with_model(model, token)


_lib.llama_token_to_str_with_model.argtypes = [llama_model_p, llama_token]
_lib.llama_token_to_str_with_model.restype = c_char_p

# Special tokens


# LLAMA_API llama_token llama_token_bos(); // beginning-of-sentence
def llama_token_bos() -> int:
    return _lib.llama_token_bos()


_lib.llama_token_bos.argtypes = []
_lib.llama_token_bos.restype = llama_token


# LLAMA_API llama_token llama_token_eos(); // end-of-sentence
def llama_token_eos() -> int:
    return _lib.llama_token_eos()


_lib.llama_token_eos.argtypes = []
_lib.llama_token_eos.restype = llama_token


# LLAMA_API llama_token llama_token_nl(); // next-line
def llama_token_nl() -> int:
    return _lib.llama_token_nl()


_lib.llama_token_nl.argtypes = []
_lib.llama_token_nl.restype = llama_token


# // Grammar
# //
# LLAMA_API struct llama_grammar * llama_grammar_init(
#         const llama_grammar_element ** rules,
#                                 size_t    n_rules,
#                                 size_t    start_rule_index);
def llama_grammar_init(
    rules,  # type: Array[llama_grammar_element_p] # type: ignore
    n_rules: c_size_t,
    start_rule_index: c_size_t,
) -> llama_grammar_p:
    return _lib.llama_grammar_init(rules, n_rules, start_rule_index)


_lib.llama_grammar_init.argtypes = [
    POINTER(llama_grammar_element_p),
    c_size_t,
    c_size_t,
]
_lib.llama_grammar_init.restype = llama_grammar_p


# LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);
def llama_grammar_free(grammar: llama_grammar_p):
    return _lib.llama_grammar_free(grammar)


_lib.llama_grammar_free.argtypes = [llama_grammar_p]
_lib.llama_grammar_free.restype = None


# Sampling functions


# @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
# LLAMA_API void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty);
def llama_sample_repetition_penalty(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    last_tokens_data,  # type: Array[llama_token]
    last_tokens_size: c_int,
    penalty: c_float,
):
    return _lib.llama_sample_repetition_penalty(
        ctx, candidates, last_tokens_data, last_tokens_size, penalty
    )


_lib.llama_sample_repetition_penalty.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
]
_lib.llama_sample_repetition_penalty.restype = None


# @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
# LLAMA_API void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);
def llama_sample_frequency_and_presence_penalties(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    last_tokens_data,  # type: Array[llama_token]
    last_tokens_size: c_int,
    alpha_frequency: c_float,
    alpha_presence: c_float,
):
    return _lib.llama_sample_frequency_and_presence_penalties(
        ctx,
        candidates,
        last_tokens_data,
        last_tokens_size,
        alpha_frequency,
        alpha_presence,
    )


_lib.llama_sample_frequency_and_presence_penalties.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
    c_float,
]
_lib.llama_sample_frequency_and_presence_penalties.restype = None


# /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
# /// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
# /// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
# /// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
# LLAMA_API void llama_sample_classifier_free_guidance(
#             struct llama_context * ctx,
#         llama_token_data_array * candidates,
#             struct llama_context * guidance_ctx,
#                             float   scale);
def llama_sample_classifier_free_guidance(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    guidance_ctx: llama_context_p,
    scale: c_float,
):
    return _lib.llama_sample_classifier_free_guidance(
        ctx, candidates, guidance_ctx, scale
    )


_lib.llama_sample_classifier_free_guidance.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_context_p,
    c_float,
]
_lib.llama_sample_classifier_free_guidance.restype = None


# @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
# LLAMA_API void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_softmax(
    ctx: llama_context_p, candidates  # type: _Pointer[llama_token_data]
):
    return _lib.llama_sample_softmax(ctx, candidates)


_lib.llama_sample_softmax.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_softmax.restype = None


# @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep);
def llama_sample_top_k(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    k: c_int,
    min_keep: c_size_t,
):
    return _lib.llama_sample_top_k(ctx, candidates, k, min_keep)


_lib.llama_sample_top_k.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_int,
    c_size_t,
]
_lib.llama_sample_top_k.restype = None


# @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
def llama_sample_top_p(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_top_p(ctx, candidates, p, min_keep)


_lib.llama_sample_top_p.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_top_p.restype = None


# @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
# LLAMA_API void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);
def llama_sample_tail_free(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    z: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_tail_free(ctx, candidates, z, min_keep)


_lib.llama_sample_tail_free.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_tail_free.restype = None


# @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
# LLAMA_API void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
def llama_sample_typical(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_typical(ctx, candidates, p, min_keep)


_lib.llama_sample_typical.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_typical.restype = None


# LLAMA_API void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates, float temp);
def llama_sample_temperature(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    temp: c_float,
):
    return _lib.llama_sample_temperature(ctx, candidates, temp)


_lib.llama_sample_temperature.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
]
_lib.llama_sample_temperature.restype = None


# @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu);
def llama_sample_token_mirostat(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    tau: c_float,
    eta: c_float,
    m: c_int,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)


_lib.llama_sample_token_mirostat.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    c_int,
    c_float_p,
]
_lib.llama_sample_token_mirostat.restype = llama_token


# @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu);
def llama_sample_token_mirostat_v2(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    tau: c_float,
    eta: c_float,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)


_lib.llama_sample_token_mirostat_v2.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    c_float_p,
]
_lib.llama_sample_token_mirostat_v2.restype = llama_token


# @details Selects the token with the highest probability.
# LLAMA_API llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_token_greedy(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
) -> int:
    return _lib.llama_sample_token_greedy(ctx, candidates)


_lib.llama_sample_token_greedy.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token_greedy.restype = llama_token


# @details Randomly selects a token from the candidates based on their probabilities.
# LLAMA_API llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_token(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
) -> int:
    return _lib.llama_sample_token(ctx, candidates)


_lib.llama_sample_token.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token.restype = llama_token


# Performance information


# LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);
def llama_get_timings(ctx: llama_context_p) -> llama_timings:
    return _lib.llama_get_timings(ctx)


_lib.llama_get_timings.argtypes = [llama_context_p]
_lib.llama_get_timings.restype = llama_timings


# LLAMA_API void llama_print_timings(struct llama_context * ctx);
def llama_print_timings(ctx: llama_context_p):
    _lib.llama_print_timings(ctx)


_lib.llama_print_timings.argtypes = [llama_context_p]
_lib.llama_print_timings.restype = None


# LLAMA_API void llama_reset_timings(struct llama_context * ctx);
def llama_reset_timings(ctx: llama_context_p):
    _lib.llama_reset_timings(ctx)


_lib.llama_reset_timings.argtypes = [llama_context_p]
_lib.llama_reset_timings.restype = None


# Print system information
# LLAMA_API const char * llama_print_system_info(void);
def llama_print_system_info() -> bytes:
    return _lib.llama_print_system_info()


_lib.llama_print_system_info.argtypes = []
_lib.llama_print_system_info.restype = c_char_p

###################################################################################################


_llama_initialized = False

if not _llama_initialized:
    llama_backend_init(c_bool(False))
    _llama_initialized = True
