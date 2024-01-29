import sys
import os
import ctypes
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_int8,
    c_int32,
    c_uint8,
    c_uint32,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
)
import pathlib
from typing import List, Union

import llama_cpp.llama_cpp as llama_cpp

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

    if "LLAVA_CPP_LIB" in os.environ:
        lib_base_name = os.environ["LLAVA_CPP_LIB"]
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
        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

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
_libllava_base_name = "llava"

# Load the library
_libllava = _load_shared_library(_libllava_base_name)


################################################
# llava.h
################################################

# struct clip_ctx;
clip_ctx_p = c_void_p

# struct llava_image_embed {
#     float * embed;
#     int n_image_pos;
# };
class llava_image_embed(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_image_pos", c_int),
    ]

# /** sanity check for clip <-> llava embed size match */
# LLAVA_API bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);
def llava_validate_embed_size(ctx_llama: llama_cpp.llama_context_p, ctx_clip: clip_ctx_p) -> bool:
    return _libllava.llava_validate_embed_size(ctx_llama, ctx_clip)

_libllava.llava_validate_embed_size.argtypes = [llama_cpp.llama_context_p, clip_ctx_p]
_libllava.llava_validate_embed_size.restype = c_bool

# /** build an image embed from image file bytes */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
def llava_image_embed_make_with_bytes(ctx_clip: clip_ctx_p, n_threads: Union[c_int, int], image_bytes: bytes, image_bytes_length: Union[c_int, int]) -> "_Pointer[llava_image_embed]":
    return _libllava.llava_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length)

_libllava.llava_image_embed_make_with_bytes.argtypes = [clip_ctx_p, c_int, POINTER(c_uint8), c_int]
_libllava.llava_image_embed_make_with_bytes.restype = POINTER(llava_image_embed)

# /** build an image embed from a path to an image filename */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
def llava_image_embed_make_with_filename(ctx_clip: clip_ctx_p, n_threads: Union[c_int, int], image_path: bytes) -> "_Pointer[llava_image_embed]":
    return _libllava.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)

_libllava.llava_image_embed_make_with_filename.argtypes = [clip_ctx_p, c_int, c_char_p]
_libllava.llava_image_embed_make_with_filename.restype = POINTER(llava_image_embed)

# LLAVA_API void llava_image_embed_free(struct llava_image_embed * embed);
# /** free an embedding made with llava_image_embed_make_* */
def llava_image_embed_free(embed: "_Pointer[llava_image_embed]"):
    return _libllava.llava_image_embed_free(embed)

_libllava.llava_image_embed_free.argtypes = [POINTER(llava_image_embed)]
_libllava.llava_image_embed_free.restype = None

# /** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
# LLAVA_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);
def llava_eval_image_embed(ctx_llama: llama_cpp.llama_context_p, embed: "_Pointer[llava_image_embed]", n_batch: Union[c_int, int], n_past: "_Pointer[c_int]") -> bool:
    return _libllava.llava_eval_image_embed(ctx_llama, embed, n_batch, n_past)

_libllava.llava_eval_image_embed.argtypes = [llama_cpp.llama_context_p, POINTER(llava_image_embed), c_int, POINTER(c_int)]
_libllava.llava_eval_image_embed.restype = c_bool


################################################
# clip.h
################################################


# struct clip_vision_hparams {
#     int32_t image_size;
#     int32_t patch_size;
#     int32_t hidden_size;
#     int32_t n_intermediate;
#     int32_t projection_dim;
#     int32_t n_head;
#     int32_t n_layer;
#     float eps;
# };
class clip_vision_hparams(Structure):
    _fields_ = [
        ("image_size", c_int32),
        ("patch_size", c_int32),
        ("hidden_size", c_int32),
        ("n_intermediate", c_int32),
        ("projection_dim", c_int32),
        ("n_head", c_int32),
        ("n_layer", c_int32),
        ("eps", c_float),
    ]

# /** load mmproj model */
# CLIP_API struct clip_ctx * clip_model_load(const char * fname, const int verbosity);
def clip_model_load(fname: bytes, verbosity: Union[c_int, int]) -> clip_ctx_p:
    return _libllava.clip_model_load(fname, verbosity)

_libllava.clip_model_load.argtypes = [c_char_p, c_int]
_libllava.clip_model_load.restype = clip_ctx_p

# /** free mmproj model */
# CLIP_API void clip_free(struct clip_ctx * ctx);
def clip_free(ctx: clip_ctx_p):
    return _libllava.clip_free(ctx)

_libllava.clip_free.argtypes = [clip_ctx_p]
_libllava.clip_free.restype = None

# size_t clip_embd_nbytes(const struct clip_ctx * ctx);
# int clip_n_patches(const struct clip_ctx * ctx);
# int clip_n_mmproj_embd(const struct clip_ctx * ctx);

# // RGB uint8 image
# struct clip_image_u8 {
#     int nx;
#     int ny;
#     uint8_t * data = NULL;
#     size_t size;
# };

# // RGB float32 image (NHWC)
# // Memory layout: RGBRGBRGB...
# struct clip_image_f32 {
#     int nx;
#     int ny;
#     float * data = NULL;
#     size_t size;
# };

# struct clip_image_u8_batch {
#     struct clip_image_u8 * data;
#     size_t size;
# };

# struct clip_image_f32_batch {
#     struct clip_image_f32 * data;
#     size_t size;
# };

# struct clip_image_u8 * make_clip_image_u8();
# struct clip_image_f32 * make_clip_image_f32();
# CLIP_API void clip_image_u8_free(clip_image_u8 * img);
# CLIP_API void clip_image_f32_free(clip_image_f32 * img);
# CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
# /** interpret bytes as an image file with length bytes_length, and use the result to populate img */
# CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);

# bool clip_image_preprocess(const struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32 * res, const bool pad2square);
# bool clip_image_encode(const struct clip_ctx * ctx, const int n_threads, struct clip_image_f32 * img, float * vec);

# bool clip_image_batch_encode(const struct clip_ctx * ctx, const int n_threads, const struct clip_image_f32_batch * imgs,
#                              float * vec);

# bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype);