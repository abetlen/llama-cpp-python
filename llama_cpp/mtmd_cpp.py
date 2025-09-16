from __future__ import annotations

import enum
import os
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_uint,
    c_uint8,
    c_int32,
    c_uint32,
    c_float,
    c_void_p,
    c_size_t,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
)
import pathlib
from typing import (
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

import llama_cpp.llama_cpp as llama_cpp

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
    )


# --- mtmd library loading ---
_libmtmd_base_name = "mtmd"
_libmtmd_override_path = os.environ.get("MTMD_CPP_LIB")
_libmtmd_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libmtmd_override_path is None else pathlib.Path(_libmtmd_override_path).parent

# Load the mtmd library
_libmtmd = load_shared_library(_libmtmd_base_name, _libmtmd_base_path)
ctypes_function_mtmd = ctypes_function_for_shared_library(_libmtmd)


################################################
# mtmd.h
# /**
#  * libmtmd: A library for multimodal support in llama.cpp.
#  *
#  * WARNING: This API is experimental and subject to many BREAKING CHANGES.
#  *          Issues related to API usage may receive lower priority support.
#  *
#  * For the usage, see an example in mtmd-cli.cpp
#  */
################################################


# enum mtmd_input_chunk_type {
#     MTMD_INPUT_CHUNK_TYPE_TEXT,
#     MTMD_INPUT_CHUNK_TYPE_IMAGE,
#     MTMD_INPUT_CHUNK_TYPE_AUDIO,
# };
class mtmd_input_chunk_type(enum.IntEnum):
    MTMD_INPUT_CHUNK_TYPE_TEXT = 0
    MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
    MTMD_INPUT_CHUNK_TYPE_AUDIO = 2

# // opaque types

# struct mtmd_context;
mtmd_context_p = NewType("mtmd_context_p", int)
mtmd_context_p_ctypes = c_void_p

# struct mtmd_bitmap;
mtmd_bitmap_p = NewType("mtmd_bitmap_p", int)
mtmd_bitmap_p_ctypes = c_void_p

# struct mtmd_image_tokens;
mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

# struct mtmd_input_chunk;
mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

# struct mtmd_input_chunks;
mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p


# struct mtmd_input_text {
#     const char * text;
#     bool add_special;
#     bool parse_special;
# };
class mtmd_input_text(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]
mtmd_input_text_p = NewType("mtmd_input_text_p", int)
mtmd_input_text_p_ctypes = POINTER(mtmd_input_text)

# struct mtmd_context_params {
#     bool use_gpu;
#     bool print_timings;
#     int n_threads;
#     enum ggml_log_level verbosity;
#     const char * image_marker; // deprecated, use media_marker instead
#     const char * media_marker;
# };
class mtmd_context_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("verbosity", c_int),
        ("image_marker", c_char_p),
        ("media_marker", c_char_p),
    ]

mtmd_context_params_p = NewType("mtmd_context_params_p", int)
mtmd_context_params_p_ctypes = POINTER(mtmd_context_params)

# MTMD_API const char * mtmd_default_marker(void);
@ctypes_function_mtmd(
    "mtmd_default_marker",
    [],
    c_char_p,
)
def mtmd_default_marker() -> c_char_p:
    ...


# MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
@ctypes_function_mtmd(
    "mtmd_context_params_default",
    [],
    mtmd_context_params,
)
def mtmd_context_params_default() -> mtmd_context_params:
    ...


# // initialize the mtmd context
# // return nullptr on failure
# MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
#                                             const struct llama_model * text_model,
#                                             const struct mtmd_context_params ctx_params);
@ctypes_function_mtmd(
    "mtmd_init_from_file", [
        c_char_p,
        llama_cpp.llama_model_p_ctypes,
        mtmd_context_params,
    ],
    mtmd_context_p_ctypes,
)
def mtmd_init_from_file(
    mmproj_fname: c_char_p,
    text_model: llama_cpp.llama_model_p,
    ctx_params: mtmd_context_params,
    /,
) -> mtmd_context_p:
    """
    initialize the mtmd context
    return nullptr on failure
    """
    ...


# MTMD_API void mtmd_free(mtmd_context * ctx);
@ctypes_function_mtmd("mtmd_free", [mtmd_context_p_ctypes], None)
def mtmd_free(ctx: mtmd_context_p):
    ...

# // whether we need to set non-causal mask before llama_decode
# MTMD_API bool mtmd_decode_use_non_causal(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_decode_use_non_causal", [mtmd_context_p_ctypes], c_bool)
def mtmd_decode_use_non_causal(ctx: mtmd_context_p) -> c_bool:
    ...

# // whether the current model use M-RoPE for llama_decode
# MTMD_API bool mtmd_decode_use_mrope(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_decode_use_mrope", [mtmd_context_p_ctypes], c_bool)
def mtmd_decode_use_mrope(ctx: mtmd_context_p) -> c_bool:
    ...

# // whether the current model supports vision input
# MTMD_API bool mtmd_support_vision(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_support_vision", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_vision(ctx: mtmd_context_p) -> c_bool:
    ...

# // whether the current model supports audio input
# MTMD_API bool mtmd_support_audio(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_support_audio", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_audio(ctx: mtmd_context_p) -> c_bool:
    ...

# // get audio bitrate in Hz, for example 16000 for Whisper
# // return -1 if audio is not supported
# MTMD_API int mtmd_get_audio_bitrate(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_get_audio_bitrate", [mtmd_context_p_ctypes], c_int)
def mtmd_get_audio_bitrate(ctx: mtmd_context_p) -> c_int:
    ...

# // mtmd_bitmap
# //
# // if bitmap is image:
# //     length of data must be nx * ny * 3
# //     the data is in RGBRGBRGB... format
# // if bitmap is audio:
# //     length of data must be n_samples * sizeof(float)
# //     the data is in float format (PCM F32)

# MTMD_API mtmd_bitmap *         mtmd_bitmap_init           (uint32_t nx, uint32_t ny, const unsigned char * data);
@ctypes_function_mtmd(
    "mtmd_bitmap_init", [
        c_uint32,
        c_uint32,
        c_char_p,
    ],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init(
    nx: c_uint32,
    ny: c_uint32,
    data: c_char_p,
    /,
) -> mtmd_bitmap_p:
    ...


# MTMD_API mtmd_bitmap *         mtmd_bitmap_init_from_audio(size_t n_samples,         const float         * data);
@ctypes_function_mtmd(
    "mtmd_bitmap_init_from_audio", [
        c_uint,
        POINTER(c_float)
    ],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init_from_audio(
    n_samples: c_uint,
    data: POINTER(c_float),
    /,
) -> mtmd_bitmap_p:
    ...


# MTMD_API uint32_t              mtmd_bitmap_get_nx     (const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_get_nx", [mtmd_bitmap_p_ctypes], c_uint32)
def mtmd_bitmap_get_nx(bitmap: mtmd_bitmap_p) -> c_uint32:
    ...

# MTMD_API uint32_t              mtmd_bitmap_get_ny     (const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_get_ny", [mtmd_bitmap_p_ctypes], c_uint32)
def mtmd_bitmap_get_ny(bitmap: mtmd_bitmap_p) -> c_uint32:
    ...

# MTMD_API const unsigned char * mtmd_bitmap_get_data   (const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_get_data", [mtmd_bitmap_p_ctypes], c_char_p)
def mtmd_bitmap_get_data(bitmap: mtmd_bitmap_p) -> c_char_p:
    ...

# MTMD_API size_t                mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_get_n_bytes", [mtmd_bitmap_p_ctypes], c_size_t)
def mtmd_bitmap_get_n_bytes(bitmap: mtmd_bitmap_p) -> c_size_t:
    ...

# MTMD_API bool                  mtmd_bitmap_is_audio   (const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_is_audio", [mtmd_bitmap_p_ctypes], c_bool)
def mtmd_bitmap_is_audio(bitmap: mtmd_bitmap_p) -> c_bool:
    ...

# MTMD_API void                  mtmd_bitmap_free       (mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_free", [mtmd_bitmap_p_ctypes], None)
def mtmd_bitmap_free(bitmap: mtmd_bitmap_p):
    ...

# // bitmap ID is optional, but useful for KV cache tracking
# // these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
# MTMD_API const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap);
@ctypes_function_mtmd("mtmd_bitmap_get_id", [mtmd_bitmap_p_ctypes], c_char_p)
def mtmd_bitmap_get_id(bitmap: mtmd_bitmap_p) -> c_char_p:
    """
    bitmap ID is optional, but useful for KV cache tracking
    these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
    """
    ...


# MTMD_API void         mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id);
@ctypes_function_mtmd(
    "mtmd_bitmap_set_id", [
        mtmd_bitmap_p_ctypes,
        c_char_p,
    ], None)
def mtmd_bitmap_set_id(
    bitmap: mtmd_bitmap_p,
    id: c_char_p,
    /,
):
    ...


# // mtmd_input_chunks
# //
# // this is simply a list of mtmd_input_chunk
# // the elements can only be populated via mtmd_tokenize()
# MTMD_API mtmd_input_chunks *      mtmd_input_chunks_init(void);
@ctypes_function_mtmd("mtmd_input_chunks_init", [], mtmd_input_chunks_p_ctypes)
def mtmd_input_chunks_init() -> mtmd_input_chunks_p:
    """
    this is simply a list of mtmd_input_chunk
    the elements can only be populated via mtmd_tokenize()
    """
    ...


# MTMD_API size_t                   mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
@ctypes_function_mtmd("mtmd_input_chunks_size", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_input_chunks_size(chunks: mtmd_input_chunks_p) -> c_size_t:
    ...


# MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get (const mtmd_input_chunks * chunks, size_t idx);
@ctypes_function_mtmd(
    "mtmd_input_chunks_get", [
        mtmd_input_chunks_p_ctypes,
        c_int32,
    ], mtmd_input_chunk_p_ctypes)
def mtmd_input_chunks_get(
    chunks: mtmd_input_chunks_p,
    idx: c_int32,
    /,
) -> mtmd_input_chunk_p:
    ...


# MTMD_API void                     mtmd_input_chunks_free(mtmd_input_chunks * chunks);
@ctypes_function_mtmd("mtmd_input_chunks_free", [mtmd_input_chunks_p_ctypes], None)
def mtmd_input_chunks_free(chunks: mtmd_input_chunks_p):
    ...


# // mtmd_input_chunk
# //
# // the instance will be constructed via mtmd_tokenize()
# // it will be freed along with mtmd_input_chunks
# MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type        (const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_get_type", [mtmd_input_chunk_p_ctypes], c_int32)
def mtmd_input_chunk_get_type(chunk: mtmd_input_chunk_p) -> c_int32:
    """
    the instance will be constructed via mtmd_tokenize()
    it will be freed along with mtmd_input_chunks
    """
    ...

# MTMD_API const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output);
@ctypes_function_mtmd(
    "mtmd_input_chunk_get_tokens_text",
    [mtmd_input_chunk_p_ctypes, POINTER(c_size_t)],
    POINTER(llama_cpp.llama_token)
)
def mtmd_input_chunk_get_tokens_text(
    chunk: mtmd_input_chunk_p, n_tokens_output: "_Pointer[c_size_t]", /
) -> Optional["_Pointer[llama_cpp.llama_token]"]:
    ...

# MTMD_API const mtmd_image_tokens *  mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_get_tokens_image", [mtmd_input_chunk_p_ctypes], mtmd_image_tokens_p_ctypes)
def mtmd_input_chunk_get_tokens_image(chunk: mtmd_input_chunk_p) -> mtmd_image_tokens_p:
    ...

# MTMD_API size_t                     mtmd_input_chunk_get_n_tokens    (const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_get_n_tokens", [mtmd_input_chunk_p_ctypes], c_size_t)
def mtmd_input_chunk_get_n_tokens(chunk: mtmd_input_chunk_p) -> c_size_t:
    ...

# // returns nullptr for ID on text chunk
# MTMD_API const char *               mtmd_input_chunk_get_id          (const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_get_id", [mtmd_input_chunk_p_ctypes], c_char_p)
def mtmd_input_chunk_get_id(chunk: mtmd_input_chunk_p) -> c_char_p:
    """
    returns nullptr for ID on text chunk
    """
    ...

# // number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
# MTMD_API llama_pos                  mtmd_input_chunk_get_n_pos       (const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_get_n_pos", [mtmd_input_chunk_p_ctypes], c_int32)
def mtmd_input_chunk_get_n_pos(chunk: mtmd_input_chunk_p) -> c_int32:
    """
    number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
    """
    ...

# // in case you want to use custom logic to handle the chunk (i.e. KV cache management)
# // you can move the chunk ownership to your own code by copying it
# // remember to free the chunk when you are done with it
# MTMD_API mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_copy", [mtmd_input_chunk_p_ctypes], mtmd_input_chunk_p_ctypes)
def mtmd_input_chunk_copy(chunk: mtmd_input_chunk_p) -> mtmd_input_chunk_p:
    """
    in case you want to use custom logic to handle the chunk (i.e. KV cache management)
    you can move the chunk ownership to your own code by copying it
    remember to free the chunk when you are done with it
    """
    ...

# MTMD_API void               mtmd_input_chunk_free(mtmd_input_chunk * chunk);
@ctypes_function_mtmd("mtmd_input_chunk_free", [mtmd_input_chunk_p_ctypes], None)
def mtmd_input_chunk_free(chunk: mtmd_input_chunk_p):
    """
    remember to free the chunk when you are done with it
    """
    ...


# // mtmd_image_tokens
# //
# // the instance will be constructed via mtmd_tokenize()
# // it will be freed along with mtmd_input_chunk
# MTMD_API size_t       mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens); // TODO: deprecate
@ctypes_function_mtmd(
    "mtmd_image_tokens_get_n_tokens", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_n_tokens(image_tokens: mtmd_image_tokens_p) -> c_size_t:
    ...

# MTMD_API size_t       mtmd_image_tokens_get_nx      (const mtmd_image_tokens * image_tokens);
@ctypes_function_mtmd(
    "mtmd_image_tokens_get_nx", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_nx(image_tokens: mtmd_image_tokens_p) -> c_size_t:
    ...

# MTMD_API size_t       mtmd_image_tokens_get_ny      (const mtmd_image_tokens * image_tokens);
@ctypes_function_mtmd(
    "mtmd_image_tokens_get_ny", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_ny(image_tokens: mtmd_image_tokens_p) -> c_size_t:
    ...

# MTMD_API const char * mtmd_image_tokens_get_id      (const mtmd_image_tokens * image_tokens); // TODO: deprecate
@ctypes_function_mtmd(
    "mtmd_image_tokens_get_id", [mtmd_image_tokens_p_ctypes], c_char_p)
def mtmd_image_tokens_get_id(image_tokens: mtmd_image_tokens_p) -> c_char_p:
    ...

# // number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
# MTMD_API llama_pos    mtmd_image_tokens_get_n_pos   (const mtmd_image_tokens * image_tokens); // TODO: deprecate
@ctypes_function_mtmd(
    "mtmd_image_tokens_get_n_pos", [mtmd_image_tokens_p_ctypes], c_int32)
def mtmd_image_tokens_get_n_pos(image_tokens: mtmd_image_tokens_p) -> c_int32:
    ...

# // tokenize an input text prompt and a list of bitmaps (images/audio)
# // the prompt must have the input image marker (default: "<__media__>") in it
# // the default marker is defined by mtmd_default_marker()
# // the marker will be replaced with the image/audio chunk
# // for example:
# //   "here is an image: <__media__>\ndescribe it in detail."
# //   this will gives 3 chunks:
# //   1. "here is an image: <start_of_image>"
# //   2. (image/audio tokens)
# //   3. "<end_of_image>\ndescribe it in detail."
# // number of bitmaps must be equal to the number of markers in the prompt
# // this function is thread-safe (shared ctx)
# // return values:
# //   0 on success
# //   1 on number of bitmaps not matching the number of markers
# //   2 on image preprocessing error
# MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
#                                mtmd_input_chunks * output,
#                                const mtmd_input_text * text,
#                                const mtmd_bitmap ** bitmaps,
#                                size_t n_bitmaps);
@ctypes_function_mtmd(
    "mtmd_tokenize", [
        mtmd_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        mtmd_input_text_p_ctypes,
        POINTER(mtmd_bitmap_p_ctypes),
        c_uint,
    ],
    c_int32,
)
def mtmd_tokenize(
    ctx: mtmd_context_p,
    output: mtmd_input_chunks_p,
    text: mtmd_input_text_p,
    bitmaps: POINTER(mtmd_bitmap_p),
    n_bitmaps: c_uint,
    /,
) -> c_int32:
    """
    tokenize an input text prompt and a list of bitmaps (images/audio)
    the prompt must have the input image marker (default: "<__media__>") in it
    the default marker is defined by mtmd_default_marker()
    the marker will be replaced with the image/audio chunk
    return values:
      0 on success
      1 on number of bitmaps not matching the number of markers
      2 on image preprocessing error
    """
    ...

# // returns 0 on success
# // TODO: deprecate
# MTMD_API int32_t mtmd_encode(mtmd_context * ctx,
#                              const mtmd_image_tokens * image_tokens);
@ctypes_function_mtmd(
    "mtmd_encode", [
        mtmd_context_p_ctypes,
        mtmd_image_tokens_p_ctypes
    ],
    c_int32,
)
def mtmd_encode(
    ctx: mtmd_context_p,
    image_tokens: mtmd_image_tokens_p,
    /,
) -> c_int32:
    ...


# // returns 0 on success
# MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx,
#                                    const mtmd_input_chunk * chunk);
@ctypes_function_mtmd(
    "mtmd_encode_chunk", [
        mtmd_context_p_ctypes,
        mtmd_input_chunk_p_ctypes
    ],
    c_int32,
)
def mtmd_encode_chunk(
    ctx: mtmd_context_p,
    chunk: mtmd_input_chunk_p,
    /,
) -> c_int32:
    ...

# // get output embeddings from the last encode pass
# // the reading size (in bytes) is equal to:
# // llama_model_n_embd(model) * mtmd_input_chunk_get_n_tokens(chunk) * sizeof(float)
# MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);
@ctypes_function_mtmd(
    "mtmd_get_output_embd", [mtmd_context_p_ctypes], POINTER(c_float))
def mtmd_get_output_embd(ctx: mtmd_context_p) -> POINTER(c_float):
    """
    get output embeddings from the last encode pass
    """
    ...


# // test function, to be used in test-mtmd-c-api.c
# MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);
@ctypes_function_mtmd(
    "mtmd_test_create_input_chunks",
    [],
    mtmd_input_chunk_p_ctypes,
)
def mtmd_test_create_input_chunks() -> mtmd_input_chunk_p:
    ...


# //
# // libmtmd helper functions
# //
# // Please note that these helpers are not guaranteed to be stable.
# // BREAKING CHANGES are expected.
# //


# // helper function to construct a mtmd_bitmap from a file
# // it calls mtmd_helper_bitmap_init_from_buf() internally
# // returns nullptr on failure
# // this function is thread-safe
# MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);

@ctypes_function_mtmd(
    "mtmd_helper_bitmap_init_from_file", [mtmd_context_p_ctypes, c_char_p], mtmd_bitmap_p_ctypes)
def mtmd_helper_bitmap_init_from_file(ctx: mtmd_context_p, fname: c_char_p) -> mtmd_bitmap_p:
    """
    helper function to construct a mtmd_bitmap from a file
    it calls mtmd_helper_bitmap_init_from_buf() internally
    returns nullptr on failure
    """
    ...


# // helper function to construct a mtmd_bitmap from a buffer containing a file
# // supported formats:
# //     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
# //     audio: formats supported by miniaudio: wav, mp3, flac
# // note: audio files will be auto-detected based on magic bytes
# // returns nullptr on failure
# // this function is thread-safe
# MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);
@ctypes_function_mtmd(
    "mtmd_helper_bitmap_init_from_buf", [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t], mtmd_bitmap_p_ctypes)
def mtmd_helper_bitmap_init_from_buf(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    len: c_size_t,
    /,
) -> mtmd_bitmap_p:
    """
    helper function to construct a mtmd_bitmap from a buffer containing a file
    supported formats:
         image: formats supported by stb_image: jpg, png, bmp, gif, etc.
         audio: formats supported by miniaudio: wav, mp3, flac
    note: audio files will be auto-detected based on magic bytes
    returns nullptr on failure
    """
    ...


# // helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
# MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);
@ctypes_function_mtmd(
    "mtmd_helper_get_n_tokens", [mtmd_input_chunk_p_ctypes], c_size_t)
def mtmd_helper_get_n_tokens(chunks: mtmd_input_chunk_p) -> c_size_t:
    """
    helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
    """
    ...


# // helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
# // normally, n_pos is equal to n_tokens, but for M-RoPE it is different
# MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);
@ctypes_function_mtmd(
    "mtmd_helper_get_n_pos", [mtmd_input_chunk_p_ctypes], c_int32)
def mtmd_helper_get_n_pos(chunks: mtmd_input_chunk_p) -> c_int32:
    """
    helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
    normally, n_pos is equal to n_tokens, but for M-RoPE it is different
    """
    ...


# // helper function that automatically:
# // 1. run llama_decode() on text chunks
# // 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
# // if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
# // otherwise, returns 0 on success
# // this function is NOT thread-safe
# MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
#                                          struct llama_context * lctx,
#                                          const mtmd_input_chunks * chunks,
#                                          llama_pos n_past,
#                                          llama_seq_id seq_id,
#                                          int32_t n_batch,
#                                          bool logits_last,
#                                          llama_pos * new_n_past);
@ctypes_function_mtmd(
    "mtmd_helper_eval_chunks", [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        c_int32,
        c_int32,
        c_int32,
        c_bool,
        POINTER(c_int32),
    ],
    c_int32)
def mtmd_helper_eval_chunks(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunks: mtmd_input_chunk_p,
    n_past: c_int32,
    seq_id: c_int32,
    n_batch: c_int32,
    logits_last: c_bool,
    new_n_past: POINTER(c_int32),
    /,
) -> c_int32:
    """
    helper function that automatically:
    1. run llama_decode() on text chunks
    2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
    if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
    otherwise, returns 0 on success
    """
    ...


# // works like mtmd_helper_eval_chunks(), but only for a single chunk
# // this function is NOT thread-safe
# MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
#                                                struct llama_context * lctx,
#                                                const mtmd_input_chunk * chunk,
#                                                llama_pos n_past,
#                                                llama_seq_id seq_id,
#                                                int32_t n_batch,
#                                                bool logits_last,
#                                                llama_pos * new_n_past);
@ctypes_function_mtmd(
    "mtmd_helper_eval_chunk_single", [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        c_int32,
        c_int32,
        c_int32,
        c_bool,
        POINTER(c_int32),
    ],
    c_int32)
def mtmd_helper_eval_chunk_single(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunks: mtmd_input_chunk_p,
    n_past: c_int32,
    seq_id: c_int32,
    n_batch: c_int32,
    logits_last: c_bool,
    new_n_past: POINTER(c_int32),
    /,
) -> c_int32:
    """
    works like mtmd_helper_eval_chunks(), but only for a single chunk
    """
    ...


# // helper function to decode an image whose embeddings have already been calculated
# // this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
# // ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
# MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
#                                                 struct llama_context * lctx,
#                                                 const mtmd_input_chunk * chunk,
#                                                 float * encoded_embd,
#                                                 llama_pos n_past,
#                                                 llama_seq_id seq_id,
#                                                 int32_t n_batch,
#                                                 llama_pos * new_n_past);
@ctypes_function_mtmd(
    "mtmd_helper_decode_image_chunk", [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        POINTER(c_float),
        c_int32,
        c_int32,
        c_int32,
        POINTER(c_int32),
    ],
    c_int32)
def mtmd_helper_decode_image_chunk(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunks: mtmd_input_chunk_p,
    encoded_embd: POINTER(c_float),
    n_past: c_int32,
    seq_id: c_int32,
    n_batch: c_int32,
    new_n_past: c_int32,
    /,
) -> c_int32:
    """
    helper function to decode an image whose embeddings have already been calculated
    this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
    ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
    """
    ...
