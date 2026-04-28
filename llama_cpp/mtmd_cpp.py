from __future__ import annotations

import os
import warnings
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_uint8,
    c_uint32,
    c_size_t,
    c_float,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    byref,
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


# Specify the base name of the shared library to load
_libmtmd_base_name = "mtmd"
_libmtmd_override_path = os.environ.get("MTMD_CPP_LIB")
_libmtmd_base_path = (
    pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
    if _libmtmd_override_path is None
    else pathlib.Path()
)

# Load the library
_libmtmd = load_shared_library(_libmtmd_base_name, _libmtmd_base_path)

ctypes_function = ctypes_function_for_shared_library(_libmtmd)

################################################
# mtmd.h types
################################################

# Opaque types
mtmd_context_p = NewType("mtmd_context_p", int)
mtmd_context_p_ctypes = c_void_p

mtmd_bitmap_p = NewType("mtmd_bitmap_p", int)
mtmd_bitmap_p_ctypes = c_void_p

mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p

# Enums
MTMD_INPUT_CHUNK_TYPE_TEXT = 0
MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
MTMD_INPUT_CHUNK_TYPE_AUDIO = 2


# Structures
class mtmd_context_params(Structure):
    """Context parameters for MTMD initialization.

    `image_marker` is deprecated upstream and kept for compatibility; use
    `media_marker` for multimodal prompt placeholders.
    """

    if TYPE_CHECKING:
        use_gpu: bool
        print_timings: bool
        n_threads: int
        image_marker: Optional[bytes]
        media_marker: Optional[bytes]
        flash_attn_type: int
        warmup: bool
        image_min_tokens: int
        image_max_tokens: int
        cb_eval: llama_cpp.ggml_backend_sched_eval_callback
        cb_eval_user_data: c_void_p

    _fields_ = [
        ("use_gpu", c_bool),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("image_marker", c_char_p),
        ("media_marker", c_char_p),
        ("flash_attn_type", c_int),
        ("warmup", c_bool),
        ("image_min_tokens", c_int),
        ("image_max_tokens", c_int),
        ("cb_eval", llama_cpp.ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", c_void_p),
    ]


class mtmd_input_text(Structure):
    """Text input passed to `mtmd_tokenize`."""

    _fields_ = [
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]


class mtmd_decoder_pos(Structure):
    """Decoder attention position for M-RoPE models."""

    _fields_ = [
        ("t", c_uint32),
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
    ]


################################################
# mtmd.h functions
################################################


# MTMD_API const char * mtmd_default_marker(void);
@ctypes_function("mtmd_default_marker", [], c_char_p)
def mtmd_default_marker() -> bytes:
    """Return the default media marker."""
    ...


# MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
@ctypes_function("mtmd_context_params_default", [], mtmd_context_params)
def mtmd_context_params_default() -> mtmd_context_params:
    """Return the default MTMD context parameters."""
    ...


# MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
#                                             const struct llama_model * text_model,
#                                             const struct mtmd_context_params ctx_params);
@ctypes_function(
    "mtmd_init_from_file",
    [c_char_p, llama_cpp.llama_model_p_ctypes, mtmd_context_params],
    mtmd_context_p_ctypes,
)
def mtmd_init_from_file(
    mmproj_fname: bytes,
    text_model: llama_cpp.llama_model_p,
    ctx_params: mtmd_context_params,
    /,
) -> Optional[mtmd_context_p]:
    """Initialize the MTMD context from a projector file. Returns None on failure."""
    ...


# MTMD_API void mtmd_free(mtmd_context * ctx);
@ctypes_function("mtmd_free", [mtmd_context_p_ctypes], None)
def mtmd_free(ctx: mtmd_context_p, /): ...


# MTMD_API bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_decode_use_non_causal",
    [mtmd_context_p_ctypes, mtmd_input_chunk_p_ctypes],
    c_bool,
)
def mtmd_decode_use_non_causal(
    ctx: mtmd_context_p, chunk: Optional[mtmd_input_chunk_p], /
) -> bool:
    """Check whether MTMD decoding uses non-causal attention."""
    ...


# MTMD_API bool mtmd_decode_use_mrope(const mtmd_context * ctx);
@ctypes_function("mtmd_decode_use_mrope", [mtmd_context_p_ctypes], c_bool)
def mtmd_decode_use_mrope(ctx: mtmd_context_p, /) -> bool:
    """Check whether MTMD decoding uses mRoPE."""
    ...


# MTMD_API bool mtmd_support_vision(const mtmd_context * ctx);
@ctypes_function("mtmd_support_vision", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_vision(ctx: mtmd_context_p, /) -> bool:
    """Check whether the current model supports vision input."""
    ...


# MTMD_API bool mtmd_support_audio(const mtmd_context * ctx);
@ctypes_function("mtmd_support_audio", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_audio(ctx: mtmd_context_p, /) -> bool:
    """Check whether MTMD supports audio."""
    ...


# MTMD_API int mtmd_get_audio_sample_rate(const mtmd_context * ctx);
@ctypes_function("mtmd_get_audio_sample_rate", [mtmd_context_p_ctypes], c_int)
def mtmd_get_audio_sample_rate(ctx: mtmd_context_p, /) -> int:
    """Get the audio sample rate in Hz. Returns -1 if audio is not supported."""
    ...


# Deprecated compatibility wrapper for the renamed mtmd_get_audio_sample_rate().
def mtmd_get_audio_bitrate(ctx: mtmd_context_p, /) -> int:
    warnings.warn(
        "mtmd_get_audio_bitrate is deprecated; use mtmd_get_audio_sample_rate instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return mtmd_get_audio_sample_rate(ctx)


# MTMD_API mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data);
@ctypes_function(
    "mtmd_bitmap_init", [c_uint32, c_uint32, POINTER(c_uint8)], mtmd_bitmap_p_ctypes
)
def mtmd_bitmap_init(
    nx: Union[c_uint32, int],
    ny: Union[c_uint32, int],
    data: CtypesArray[c_uint8],
    /,
) -> Optional[mtmd_bitmap_p]: ...


# MTMD_API mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples, const float * data);
@ctypes_function(
    "mtmd_bitmap_init_from_audio",
    [c_size_t, POINTER(c_float)],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init_from_audio(
    n_samples: Union[c_size_t, int],
    data: CtypesArray[c_float],
    /,
) -> Optional[mtmd_bitmap_p]:
    """Initialize an MTMD bitmap from audio samples."""
    ...


# MTMD_API void mtmd_bitmap_free(mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_free", [mtmd_bitmap_p_ctypes], None)
def mtmd_bitmap_free(bitmap: mtmd_bitmap_p, /): ...


# MTMD_API uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_get_nx", [mtmd_bitmap_p_ctypes], c_uint32)
def mtmd_bitmap_get_nx(bitmap: mtmd_bitmap_p, /) -> int:
    """Get the bitmap width in pixels."""
    ...


# MTMD_API uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_get_ny", [mtmd_bitmap_p_ctypes], c_uint32)
def mtmd_bitmap_get_ny(bitmap: mtmd_bitmap_p, /) -> int:
    """Get the bitmap height in pixels."""
    ...


# MTMD_API const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_get_data", [mtmd_bitmap_p_ctypes], POINTER(c_uint8))
def mtmd_bitmap_get_data(bitmap: mtmd_bitmap_p, /) -> Optional[CtypesArray[c_uint8]]:
    """Get the raw bitmap data buffer."""
    ...


# MTMD_API size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_get_n_bytes", [mtmd_bitmap_p_ctypes], c_size_t)
def mtmd_bitmap_get_n_bytes(bitmap: mtmd_bitmap_p, /) -> int:
    """Get the bitmap data size in bytes."""
    ...


# MTMD_API bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_is_audio", [mtmd_bitmap_p_ctypes], c_bool)
def mtmd_bitmap_is_audio(bitmap: mtmd_bitmap_p, /) -> bool:
    """Check whether the bitmap contains audio data."""
    ...


# MTMD_API const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_get_id", [mtmd_bitmap_p_ctypes], c_char_p)
def mtmd_bitmap_get_id(bitmap: mtmd_bitmap_p, /) -> Optional[bytes]:
    """Get the optional bitmap identifier."""
    ...


# MTMD_API void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id);
@ctypes_function("mtmd_bitmap_set_id", [mtmd_bitmap_p_ctypes, c_char_p], None)
def mtmd_bitmap_set_id(bitmap: mtmd_bitmap_p, id: Optional[bytes], /):
    """Set the optional bitmap identifier."""
    ...


# MTMD_API mtmd_input_chunks * mtmd_input_chunks_init(void);
@ctypes_function("mtmd_input_chunks_init", [], mtmd_input_chunks_p_ctypes)
def mtmd_input_chunks_init() -> Optional[mtmd_input_chunks_p]: ...


# MTMD_API void mtmd_input_chunks_free(mtmd_input_chunks * chunks);
@ctypes_function("mtmd_input_chunks_free", [mtmd_input_chunks_p_ctypes], None)
def mtmd_input_chunks_free(chunks: mtmd_input_chunks_p, /): ...


# MTMD_API size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
@ctypes_function("mtmd_input_chunks_size", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_input_chunks_size(chunks: mtmd_input_chunks_p, /) -> int: ...


# MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx);
@ctypes_function(
    "mtmd_input_chunks_get",
    [mtmd_input_chunks_p_ctypes, c_size_t],
    mtmd_input_chunk_p_ctypes,
)
def mtmd_input_chunks_get(
    chunks: mtmd_input_chunks_p, idx: Union[c_size_t, int], /
) -> Optional[mtmd_input_chunk_p]: ...


# MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
#                                mtmd_input_chunks * output,
#                                const mtmd_input_text * text,
#                                const mtmd_bitmap ** bitmaps,
#                                size_t n_bitmaps);
@ctypes_function(
    "mtmd_tokenize",
    [
        mtmd_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        POINTER(mtmd_input_text),
        POINTER(mtmd_bitmap_p_ctypes),
        c_size_t,
    ],
    c_int,
)
def mtmd_tokenize(
    ctx: mtmd_context_p,
    output: mtmd_input_chunks_p,
    text: "_Pointer[mtmd_input_text]",
    bitmaps: CtypesArray[mtmd_bitmap_p_ctypes],
    n_bitmaps: Union[c_size_t, int],
    /,
) -> int: ...


# MTMD_API size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_get_n_tokens", [mtmd_input_chunk_p_ctypes], c_size_t)
def mtmd_input_chunk_get_n_tokens(chunk: mtmd_input_chunk_p, /) -> int: ...


# MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_get_type", [mtmd_input_chunk_p_ctypes], c_int)
def mtmd_input_chunk_get_type(chunk: mtmd_input_chunk_p, /) -> int: ...


# MTMD_API const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output);
@ctypes_function(
    "mtmd_input_chunk_get_tokens_text",
    [mtmd_input_chunk_p_ctypes, POINTER(c_size_t)],
    POINTER(llama_cpp.llama_token),
)
def mtmd_input_chunk_get_tokens_text(
    chunk: mtmd_input_chunk_p, n_tokens_output: "_Pointer[c_size_t]", /
) -> Optional["_Pointer[llama_cpp.llama_token]"]: ...


# MTMD_API const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_input_chunk_get_tokens_image",
    [mtmd_input_chunk_p_ctypes],
    mtmd_image_tokens_p_ctypes,
)
def mtmd_input_chunk_get_tokens_image(
    chunk: mtmd_input_chunk_p, /
) -> Optional[mtmd_image_tokens_p]: ...


# MTMD_API const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_get_id", [mtmd_input_chunk_p_ctypes], c_char_p)
def mtmd_input_chunk_get_id(chunk: mtmd_input_chunk_p, /) -> Optional[bytes]:
    """Get the optional chunk identifier."""
    ...


# MTMD_API llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_input_chunk_get_n_pos",
    [mtmd_input_chunk_p_ctypes],
    llama_cpp.llama_pos,
)
def mtmd_input_chunk_get_n_pos(chunk: mtmd_input_chunk_p, /) -> int:
    """Get the number of positions consumed by the chunk."""
    ...


# MTMD_API mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_input_chunk_copy", [mtmd_input_chunk_p_ctypes], mtmd_input_chunk_p_ctypes
)
def mtmd_input_chunk_copy(chunk: mtmd_input_chunk_p, /) -> Optional[mtmd_input_chunk_p]:
    """Copy an input chunk and transfer ownership to the caller."""
    ...


# MTMD_API void mtmd_input_chunk_free(mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_free", [mtmd_input_chunk_p_ctypes], None)
def mtmd_input_chunk_free(chunk: mtmd_input_chunk_p, /):
    """Free an owned input chunk."""
    ...


# MTMD_API size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens);
@ctypes_function(
    "mtmd_image_tokens_get_n_tokens", [mtmd_image_tokens_p_ctypes], c_size_t
)
def mtmd_image_tokens_get_n_tokens(image_tokens: mtmd_image_tokens_p, /) -> int:
    """Get the number of image tokens."""
    ...


# DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens),
#            "use mtmd_image_tokens_get_decoder_pos() instead");
@ctypes_function("mtmd_image_tokens_get_nx", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_nx(image_tokens: mtmd_image_tokens_p, /) -> int:
    """Get the image token grid width."""
    ...


# DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens),
#            "use mtmd_image_tokens_get_decoder_pos() instead");
@ctypes_function("mtmd_image_tokens_get_ny", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_ny(image_tokens: mtmd_image_tokens_p, /) -> int:
    """Get the image token grid height."""
    ...


# MTMD_API const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens);
@ctypes_function("mtmd_image_tokens_get_id", [mtmd_image_tokens_p_ctypes], c_char_p)
def mtmd_image_tokens_get_id(image_tokens: mtmd_image_tokens_p, /) -> Optional[bytes]:
    """Get the optional image token identifier."""
    ...


# MTMD_API llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens);
@ctypes_function(
    "mtmd_image_tokens_get_n_pos",
    [mtmd_image_tokens_p_ctypes],
    llama_cpp.llama_pos,
)
def mtmd_image_tokens_get_n_pos(image_tokens: mtmd_image_tokens_p, /) -> int:
    """Get the number of positions consumed by the image tokens."""
    ...


# MTMD_API struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(
#     const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i);
@ctypes_function(
    "mtmd_image_tokens_get_decoder_pos",
    [mtmd_image_tokens_p_ctypes, llama_cpp.llama_pos, c_size_t],
    mtmd_decoder_pos,
)
def mtmd_image_tokens_get_decoder_pos(
    image_tokens: mtmd_image_tokens_p,
    pos_0: llama_cpp.llama_pos,
    i: Union[c_size_t, int],
    /,
) -> mtmd_decoder_pos:
    """Get decoder attention position for an image embedding token."""
    ...


# MTMD_API int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens);
@ctypes_function(
    "mtmd_encode",
    [mtmd_context_p_ctypes, mtmd_image_tokens_p_ctypes],
    c_int,
)
def mtmd_encode(ctx: mtmd_context_p, image_tokens: mtmd_image_tokens_p, /) -> int:
    """Run an MTMD encode pass for image tokens."""
    ...


# MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_encode_chunk",
    [mtmd_context_p_ctypes, mtmd_input_chunk_p_ctypes],
    c_int,
)
def mtmd_encode_chunk(ctx: mtmd_context_p, chunk: mtmd_input_chunk_p, /) -> int:
    """Run an MTMD encode pass for a single chunk."""
    ...


# MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);
@ctypes_function("mtmd_get_output_embd", [mtmd_context_p_ctypes], POINTER(c_float))
def mtmd_get_output_embd(ctx: mtmd_context_p, /) -> Optional[CtypesArray[c_float]]:
    """Get output embeddings from the last encode pass."""
    ...


# MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);
@ctypes_function("mtmd_test_create_input_chunks", [], mtmd_input_chunks_p_ctypes)
def mtmd_test_create_input_chunks() -> Optional[mtmd_input_chunks_p]:
    """Create MTMD test chunks for the C API tests."""
    ...


################################################
# mtmd-helper.h functions
################################################


# MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);
@ctypes_function(
    "mtmd_helper_bitmap_init_from_file",
    [mtmd_context_p_ctypes, c_char_p],
    mtmd_bitmap_p_ctypes,
)
def mtmd_helper_bitmap_init_from_file(
    ctx: mtmd_context_p, fname: bytes, /
) -> Optional[mtmd_bitmap_p]:
    """Initialize an MTMD bitmap from a file."""
    ...


# MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);
@ctypes_function(
    "mtmd_helper_bitmap_init_from_buf",
    [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t],
    mtmd_bitmap_p_ctypes,
)
def mtmd_helper_bitmap_init_from_buf(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    length: Union[c_size_t, int],
    /,
) -> Optional[mtmd_bitmap_p]: ...


# MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);
@ctypes_function("mtmd_helper_get_n_tokens", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_helper_get_n_tokens(chunks: mtmd_input_chunks_p, /) -> int: ...


# MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);
@ctypes_function(
    "mtmd_helper_get_n_pos",
    [mtmd_input_chunks_p_ctypes],
    llama_cpp.llama_pos,
)
def mtmd_helper_get_n_pos(chunks: mtmd_input_chunks_p, /) -> int:
    """Count the total positions consumed by the chunks."""
    ...


# MTMD_API void mtmd_helper_image_get_decoder_pos(
#     const mtmd_image_tokens * image, llama_pos pos_0, struct mtmd_decoder_pos * out_pos);
@ctypes_function(
    "mtmd_helper_image_get_decoder_pos",
    [mtmd_image_tokens_p_ctypes, llama_cpp.llama_pos, POINTER(mtmd_decoder_pos)],
    None,
)
def mtmd_helper_image_get_decoder_pos(
    image: mtmd_image_tokens_p,
    pos_0: llama_cpp.llama_pos,
    out_pos: "_Pointer[mtmd_decoder_pos]",
    /,
):
    """Fill decoder attention positions for all image embedding tokens."""
    ...


# MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
#                                          struct llama_context * lctx,
#                                          const mtmd_input_chunks * chunks,
#                                          llama_pos n_past,
#                                          llama_seq_id seq_id,
#                                          int32_t n_batch,
#                                          bool logits_last,
#                                          llama_pos * new_n_past);
@ctypes_function(
    "mtmd_helper_eval_chunks",
    [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        llama_cpp.llama_pos,
        llama_cpp.llama_seq_id,
        c_int,
        c_bool,
        POINTER(llama_cpp.llama_pos),
    ],
    c_int,
)
def mtmd_helper_eval_chunks(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunks: mtmd_input_chunks_p,
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: Union[c_int, int],
    logits_last: Union[c_bool, bool],
    new_n_past: "_Pointer[llama_cpp.llama_pos]",
    /,
) -> int: ...


# MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
#                                                struct llama_context * lctx,
#                                                const mtmd_input_chunk * chunk,
#                                                llama_pos n_past,
#                                                llama_seq_id seq_id,
#                                                int32_t n_batch,
#                                                bool logits_last,
#                                                llama_pos * new_n_past);
@ctypes_function(
    "mtmd_helper_eval_chunk_single",
    [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        llama_cpp.llama_pos,
        llama_cpp.llama_seq_id,
        c_int,
        c_bool,
        POINTER(llama_cpp.llama_pos),
    ],
    c_int,
)
def mtmd_helper_eval_chunk_single(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunk: mtmd_input_chunk_p,
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: Union[c_int, int],
    logits_last: Union[c_bool, bool],
    new_n_past: "_Pointer[llama_cpp.llama_pos]",
    /,
) -> int: ...


# MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
#                                                 struct llama_context * lctx,
#                                                 const mtmd_input_chunk * chunk,
#                                                 float * encoded_embd,
#                                                 llama_pos n_past,
#                                                 llama_seq_id seq_id,
#                                                 int32_t n_batch,
#                                                 llama_pos * new_n_past);
@ctypes_function(
    "mtmd_helper_decode_image_chunk",
    [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        POINTER(c_float),
        llama_cpp.llama_pos,
        llama_cpp.llama_seq_id,
        c_int,
        POINTER(llama_cpp.llama_pos),
    ],
    c_int,
)
def mtmd_helper_decode_image_chunk(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunk: mtmd_input_chunk_p,
    encoded_embd: CtypesArray[c_float],
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: Union[c_int, int],
    new_n_past: "_Pointer[llama_cpp.llama_pos]",
    /,
) -> int:
    """Decode a pre-encoded image chunk."""
    ...


# MTMD_API void mtmd_log_set(ggml_log_callback log_callback, void * user_data);
@ctypes_function(
    "mtmd_log_set",
    [llama_cpp.llama_log_callback, c_void_p],
    None,
)
def mtmd_log_set(log_callback, user_data: c_void_p, /):
    """Set the MTMD logging callback."""
    ...


# MTMD_API void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data);
@ctypes_function(
    "mtmd_helper_log_set",
    [llama_cpp.llama_log_callback, c_void_p],
    None,
)
def mtmd_helper_log_set(log_callback, user_data: c_void_p, /):
    """Set the MTMD helper logging callback."""
    ...
