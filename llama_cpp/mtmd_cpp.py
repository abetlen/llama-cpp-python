from __future__ import annotations

import os
import warnings
from ctypes import (
    CFUNCTYPE,
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
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
    Callable,
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
    else pathlib.Path(_libmtmd_override_path)
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

mtmd_helper_video_p = NewType("mtmd_helper_video_p", int)
mtmd_helper_video_p_ctypes = c_void_p

mtmd_helper_gen_audio_p = NewType("mtmd_helper_gen_audio_p", int)
mtmd_helper_gen_audio_p_ctypes = c_void_p

mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p

mtmd_batch_p = NewType("mtmd_batch_p", int)
mtmd_batch_p_ctypes = c_void_p

mtmd_audio_stream_p = NewType("mtmd_audio_stream_p", int)
mtmd_audio_stream_p_ctypes = c_void_p

mtmd_gen_state_p = NewType("mtmd_gen_state_p", int)
mtmd_gen_state_p_ctypes = c_void_p

# Enums
MTMD_INPUT_CHUNK_TYPE_TEXT = 0
MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
MTMD_INPUT_CHUNK_TYPE_AUDIO = 2
MTMD_INPUT_CHUNK_TYPE_COUNT = 3

MTMD_VIDEO_TEMPLATE_DEFAULT = 0
MTMD_VIDEO_TEMPLATE_LFM3 = 1
MTMD_VIDEO_TEMPLATE_LFM3_OMNI = 2
MTMD_VIDEO_TEMPLATE_QWEN3VL = 3

MTMD_GEN_AUDIO_TYPE_NONE = 0
MTMD_GEN_AUDIO_TYPE_QWEN3TTS = 1
MTMD_GEN_AUDIO_TYPE_POCKETTTS = 2
MTMD_GEN_AUDIO_TYPE_SOPRANO = 3
MTMD_GEN_AUDIO_TYPE_LFM3A = 4
MTMD_GEN_AUDIO_TYPE_CHATTERBOX = 5
MTMD_GEN_AUDIO_TYPE_DRAMA3 = 6

MTMD_GEN_PROCESS_TYPE_GEN_CODE = 0
MTMD_GEN_PROCESS_TYPE_GEN_WAV = 1

MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM = 0
MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV = 1

MTMD_HELPER_GEN_AUDIO_OUT_AUTO = 0
MTMD_HELPER_GEN_AUDIO_OUT_TEXT = 1
MTMD_HELPER_GEN_AUDIO_OUT_AUDIO = 2
MTMD_HELPER_GEN_AUDIO_OUT_MIXED = 3

mtmd_progress_callback = CFUNCTYPE(c_bool, c_float, c_void_p)


# Structures
class mtmd_context_params(Structure):
    """Context parameters for MTMD initialization.

    `image_marker` is deprecated upstream and kept for compatibility; use
    `media_marker` for multimodal prompt placeholders.
    """

    if TYPE_CHECKING:
        use_gpu: bool
        device: c_void_p
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
        batch_max_tokens: int
        progress_callback: Callable[[float, c_void_p], bool]
        progress_callback_user_data: c_void_p

    _fields_ = [
        ("use_gpu", c_bool),
        ("device", c_void_p),
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
        ("batch_max_tokens", c_int),
        ("progress_callback", mtmd_progress_callback),
        ("progress_callback_user_data", c_void_p),
    ]


class mtmd_input_text(Structure):
    """Text input passed to `mtmd_tokenize`."""

    if TYPE_CHECKING:
        text: Optional[bytes]
        text_len: int
        add_special: bool
        parse_special: bool

    _fields_ = [
        ("text", c_char_p),
        ("text_len", c_size_t),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]


# struct mtmd_input_part {
#     // only text or bitmap can be set, not both
#     const struct mtmd_input_text * text;
#     const struct mtmd_bitmap * bitmap;
# };
class mtmd_input_part(Structure):
    """An input part with only text or bitmap set, not both."""

    if TYPE_CHECKING:
        text: _Pointer[mtmd_input_text]
        bitmap: Optional[mtmd_bitmap_p]

    _fields_ = [
        ("text", POINTER(mtmd_input_text)),
        ("bitmap", mtmd_bitmap_p_ctypes),
    ]


class mtmd_decoder_pos(Structure):
    """Decoder attention position for M-RoPE models."""

    _fields_ = [
        ("t", c_uint32),
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
    ]


# struct mtmd_video_config {
#     enum mtmd_video_template tmpl;
#     float fps;
#     int32_t n_frames;
#     int32_t frame_width;
#     int32_t frame_height;
# };
class mtmd_video_config(Structure):
    if TYPE_CHECKING:
        tmpl: int
        fps: float
        n_frames: int
        frame_width: int
        frame_height: int

    _fields_ = [
        ("tmpl", c_int),
        ("fps", c_float),
        ("n_frames", c_int32),
        ("frame_width", c_int32),
        ("frame_height", c_int32),
    ]


# struct mtmd_audio_stream_params {
#     int32_t step_ms;
#     const llama_token * prompt;
#     size_t n_prompt;
# };
class mtmd_audio_stream_params(Structure):
    if TYPE_CHECKING:
        step_ms: int
        prompt: Optional["_Pointer[llama_cpp.llama_token]"]
        n_prompt: int

    _fields_ = [
        ("step_ms", c_int32),
        ("prompt", llama_cpp.llama_token_p),
        ("n_prompt", c_size_t),
    ]


# struct mtmd_caps {
#     bool inp_vision;
#     bool inp_audio;
#     bool gen_audio;
# };
class mtmd_caps(Structure):
    """Capabilities exposed by an mmproj file."""

    if TYPE_CHECKING:
        inp_vision: bool
        inp_audio: bool
        gen_audio: bool

    _fields_ = [
        ("inp_vision", c_bool),
        ("inp_audio", c_bool),
        ("gen_audio", c_bool),
    ]


# struct mtmd_gen_audio_info {
#     enum mtmd_gen_audio_type type;
#     int32_t sample_rate; // in Hz, for example 24000 for qwen3tts
#     const char * model_variant; // name of the weight variant, can be None if not applicable
#     int32_t n_voices; // predefined voices, see mtmd_gen_audio_voice_name()
# };
class mtmd_gen_audio_info(Structure):
    if TYPE_CHECKING:
        type: int
        sample_rate: int
        model_variant: Optional[bytes]
        n_voices: int

    _fields_ = [
        ("type", c_int),
        ("sample_rate", c_int32),
        ("model_variant", c_char_p),
        ("n_voices", c_int32),
    ]


# struct mtmd_gen_inp {
#     enum mtmd_gen_process_type type;
#
#     // for MTMD_GEN_PROCESS_TYPE_GEN_CODE
#     int32_t code0;  // the sampled codebook 0 entry from backbone
#     float * embd;   // the hidden state from backbone, must have n_text_embd elements
#     int32_t top_k;
#     float   top_p;
#     float   temp; // sampling temperature, or noise scale for flow-matching decoders
#     float   penalty_repeat; // repetition penalty of the sampled codes, <= 1 is off
#     int32_t penalty_last_n; // frames the repetition penalty looks back on
#
#     // for MTMD_GEN_PROCESS_TYPE_GEN_WAV
#     // pass either codes (discrete) or feats (continuous), depending on the pipeline
#     int32_t * codes;
#     size_t    n_codes;
#     const float * feats;
#     size_t        n_feats;
# };
class mtmd_gen_inp(Structure):
    if TYPE_CHECKING:
        type: int
        code0: int
        embd: Optional["_Pointer[c_float]"]
        top_k: int
        top_p: float
        temp: float
        penalty_repeat: float
        penalty_last_n: int
        codes: Optional["_Pointer[c_int32]"]
        n_codes: int
        feats: Optional["_Pointer[c_float]"]
        n_feats: int

    _fields_ = [
        ("type", c_int),
        ("code0", c_int32),
        ("embd", POINTER(c_float)),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("temp", c_float),
        ("penalty_repeat", c_float),
        ("penalty_last_n", c_int32),
        ("codes", POINTER(c_int32)),
        ("n_codes", c_size_t),
        ("feats", POINTER(c_float)),
        ("n_feats", c_size_t),
    ]


# struct mtmd_gen_out {
#     // note: output memory is owned by the state, valid until the next process() call with this state
#
#     // for MTMD_GEN_PROCESS_TYPE_GEN_CODE
#     const int32_t * codes;
#     size_t          n_codes;
#     const float * feats; // continuous counterpart of codes
#     size_t        n_feats;
#     const float * embd; // the generated hidden state, to be fed back to backbone
#                         // it must have n_text_embd elements
#     bool is_eos; // only set by pipelines having the EOS head inside mmproj
#
#     // for MTMD_GEN_PROCESS_TYPE_GEN_WAV
#     const float * audio;
#     size_t        n_samples;
# };
class mtmd_gen_out(Structure):
    if TYPE_CHECKING:
        codes: Optional["_Pointer[c_int32]"]
        n_codes: int
        feats: Optional["_Pointer[c_float]"]
        n_feats: int
        embd: Optional["_Pointer[c_float]"]
        is_eos: bool
        audio: Optional["_Pointer[c_float]"]
        n_samples: int

    _fields_ = [
        ("codes", POINTER(c_int32)),
        ("n_codes", c_size_t),
        ("feats", POINTER(c_float)),
        ("n_feats", c_size_t),
        ("embd", POINTER(c_float)),
        ("is_eos", c_bool),
        ("audio", POINTER(c_float)),
        ("n_samples", c_size_t),
    ]


# struct mtmd_gen_state_params {
#     uint32_t seed; // UINT32_MAX for random
# };
class mtmd_gen_state_params(Structure):
    if TYPE_CHECKING:
        seed: int

    _fields_ = [("seed", c_uint32)]


mtmd_bitmap_lazy_callback = CFUNCTYPE(
    c_int,
    c_size_t,
    c_void_p,
    POINTER(mtmd_bitmap_p_ctypes),
    POINTER(c_char_p),
)


# // one decoded sub-batch of embeddings, passed to mtmd_helper_post_decode_callback
# struct mtmd_helper_embd_batch {
#     int32_t n_tokens;
#     const float     * embd;   // [n_tokens, n_embd]
#     int32_t           n_embd;
#     const llama_pos * pos;    // [n_pos, n_tokens], section-major
#     int32_t           n_pos;  // 4 for M-RoPE models, 1 otherwise
#     llama_seq_id      seq_id;
# };
class mtmd_helper_embd_batch(Structure):
    """One decoded sub-batch of embeddings, passed to mtmd_helper_post_decode_callback."""

    if TYPE_CHECKING:
        n_tokens: int
        embd: "_Pointer[c_float]"
        n_embd: int
        pos: "_Pointer[llama_cpp.llama_pos]"
        n_pos: int
        seq_id: int

    _fields_ = [
        ("n_tokens", c_int32),
        ("embd", POINTER(c_float)),
        ("n_embd", c_int32),
        ("pos", POINTER(llama_cpp.llama_pos)),
        ("n_pos", c_int32),
        ("seq_id", llama_cpp.llama_seq_id),
    ]


# typedef int32_t (*mtmd_helper_post_decode_callback)(const struct mtmd_helper_embd_batch * batch, void * user_data);
mtmd_helper_post_decode_callback = CFUNCTYPE(
    c_int32,
    POINTER(mtmd_helper_embd_batch),
    c_void_p,
)


class mtmd_helper_bitmap_wrapper(Structure):
    """Bitmap wrapper returned by MTMD helper media loaders."""

    if TYPE_CHECKING:
        bitmap: Optional[mtmd_bitmap_p]
        video_ctx: Optional[mtmd_helper_video_p]

    _fields_ = [
        ("bitmap", mtmd_bitmap_p_ctypes),
        ("video_ctx", mtmd_helper_video_p_ctypes),
    ]


class mtmd_helper_video_info(Structure):
    """Metadata for a decoded video stream."""

    if TYPE_CHECKING:
        width: int
        height: int
        fps: float
        n_frames: int

    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("fps", c_float),
        ("n_frames", c_int),
    ]


class mtmd_helper_video_init_params(Structure):
    """Parameters for initializing an MTMD helper video stream."""

    if TYPE_CHECKING:
        fps_target: float
        ffmpeg_bin_dir: Optional[bytes]
        timestamp_interval_ms: int
        no_soundtrack: bool

    _fields_ = [
        ("fps_target", c_float),
        ("ffmpeg_bin_dir", c_char_p),
        ("timestamp_interval_ms", c_int64),
        ("no_soundtrack", c_bool),
    ]


# // opt for mtmd_helper_bitmap_init_from_*()
# struct mtmd_helper_init_opt {
#     struct mtmd_helper_video_init_params video_params;
# };
class mtmd_helper_init_opt(Structure):
    if TYPE_CHECKING:
        video_params: mtmd_helper_video_init_params

    _fields_ = [("video_params", mtmd_helper_video_init_params)]


# struct mtmd_helper_gen_audio_inp {
#     bool stream; // if true, output() must be called after each step_gen()
#     llama_seq_id seq_id;
#
#     const char * prompt;
#     size_t       prompt_len;
#
#     int32_t n_past;
#     enum mtmd_helper_gen_audio_out_mode out_mode;
#
#     mtmd_bitmap * speaker_ref; // optional, can be NULL
#     const char * lang; // optional, can be NULL
#
#     int32_t  top_k;
#     float    top_p;
#     uint32_t seed; // UINT32_MAX for random (default: random)
#
#     enum mtmd_helper_gen_audio_outtype out_type;
# };
class mtmd_helper_gen_audio_inp(Structure):
    if TYPE_CHECKING:
        stream: bool
        seq_id: int
        prompt: Optional[bytes]
        prompt_len: int
        n_past: int
        out_mode: int
        speaker_ref: Optional[mtmd_bitmap_p]
        lang: Optional[bytes]
        top_k: int
        top_p: float
        seed: int
        out_type: int

    _fields_ = [
        ("stream", c_bool),
        ("seq_id", llama_cpp.llama_seq_id),
        ("prompt", c_char_p),
        ("prompt_len", c_size_t),
        ("n_past", c_int32),
        ("out_mode", c_int),
        ("speaker_ref", mtmd_bitmap_p_ctypes),
        ("lang", c_char_p),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("seed", c_uint32),
        ("out_type", c_int),
    ]


class mtmd_helper_cap(Structure):
    """Capabilities exposed by an MTMD model and its projector."""

    if TYPE_CHECKING:
        chat: bool
        vision: bool
        audio: bool
        video: bool
        audio_stream: bool
        text_gen: bool
        audio_gen: bool
        audio_gen_spk_ref: bool

    _fields_ = [
        ("chat", c_bool),
        ("vision", c_bool),
        ("audio", c_bool),
        ("video", c_bool),
        ("audio_stream", c_bool),
        ("text_gen", c_bool),
        ("audio_gen", c_bool),
        ("audio_gen_spk_ref", c_bool),
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


# MTMD_API bool mtmd_support_audio_stream(const mtmd_context * ctx);
@ctypes_function("mtmd_support_audio_stream", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_audio_stream(ctx: mtmd_context_p, /) -> bool:
    """Check whether the current model supports realtime audio input."""
    ...


# MTMD_API int mtmd_get_audio_sample_rate(const mtmd_context * ctx);
@ctypes_function("mtmd_get_audio_sample_rate", [mtmd_context_p_ctypes], c_int)
def mtmd_get_audio_sample_rate(ctx: mtmd_context_p, /) -> int:
    """Get the audio sample rate in Hz. Returns -1 if audio is not supported."""
    ...


# MTMD_API struct mtmd_audio_stream_params mtmd_audio_stream_params_default(void);
@ctypes_function("mtmd_audio_stream_params_default", [], mtmd_audio_stream_params)
def mtmd_audio_stream_params_default() -> mtmd_audio_stream_params:
    """Get default realtime audio stream parameters."""
    ...


# MTMD_API mtmd_audio_stream * mtmd_audio_stream_init(
#     mtmd_context * ctx, struct mtmd_audio_stream_params params);
@ctypes_function(
    "mtmd_audio_stream_init",
    [mtmd_context_p_ctypes, mtmd_audio_stream_params],
    mtmd_audio_stream_p_ctypes,
)
def mtmd_audio_stream_init(
    ctx: mtmd_context_p,
    params: mtmd_audio_stream_params,
    /,
) -> Optional[mtmd_audio_stream_p]:
    """Initialize realtime audio processing, or return None on failure."""
    ...


# MTMD_API void mtmd_audio_stream_free(mtmd_audio_stream * stream);
@ctypes_function("mtmd_audio_stream_free", [mtmd_audio_stream_p_ctypes], None)
def mtmd_audio_stream_free(stream: mtmd_audio_stream_p, /):
    """Free a realtime audio stream."""
    ...


# MTMD_API int32_t mtmd_audio_stream_push(
#     mtmd_audio_stream * stream, const float * samples, size_t n_samples);
@ctypes_function(
    "mtmd_audio_stream_push",
    [mtmd_audio_stream_p_ctypes, POINTER(c_float), c_size_t],
    c_int32,
)
def mtmd_audio_stream_push(
    stream: mtmd_audio_stream_p,
    samples: Optional[CtypesArray[c_float]],
    n_samples: Union[c_size_t, int],
    /,
) -> int:
    """Push PCM F32 samples into a realtime audio stream."""
    ...


# MTMD_API int32_t mtmd_audio_stream_flush(mtmd_audio_stream * stream, bool end);
@ctypes_function(
    "mtmd_audio_stream_flush", [mtmd_audio_stream_p_ctypes, c_bool], c_int32
)
def mtmd_audio_stream_flush(stream: mtmd_audio_stream_p, end: bool, /) -> int:
    """Flush pending audio, optionally ending the stream."""
    ...


# MTMD_API int32_t mtmd_audio_stream_step(mtmd_audio_stream * stream, llama_token last_token);
@ctypes_function(
    "mtmd_audio_stream_step",
    [mtmd_audio_stream_p_ctypes, llama_cpp.llama_token],
    c_int32,
)
def mtmd_audio_stream_step(
    stream: mtmd_audio_stream_p,
    last_token: llama_cpp.llama_token,
    /,
) -> int:
    """Produce the next decoder rows, using the previously sampled token."""
    ...


# MTMD_API const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream * stream);
@ctypes_function(
    "mtmd_audio_stream_get_output_embd",
    [mtmd_audio_stream_p_ctypes],
    POINTER(c_float),
)
def mtmd_audio_stream_get_output_embd(
    stream: mtmd_audio_stream_p,
    /,
) -> Optional[CtypesArray[c_float]]:
    """Get decoder embedding rows produced by the last stream step."""
    ...


# MTMD_API bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream * stream);
@ctypes_function(
    "mtmd_audio_stream_is_seq_start", [mtmd_audio_stream_p_ctypes], c_bool
)
def mtmd_audio_stream_is_seq_start(stream: mtmd_audio_stream_p, /) -> bool:
    """Check whether the last stream step starts a new sequence."""
    ...


# MTMD_API int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream * stream);
@ctypes_function(
    "mtmd_audio_stream_get_n_rows", [mtmd_audio_stream_p_ctypes], c_int64
)
def mtmd_audio_stream_get_n_rows(stream: mtmd_audio_stream_p, /) -> int:
    """Get the number of audio rows whose text has been sampled."""
    ...


# MTMD_API int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream * stream);
@ctypes_function(
    "mtmd_audio_stream_get_n_rows_flushed", [mtmd_audio_stream_p_ctypes], c_int64
)
def mtmd_audio_stream_get_n_rows_flushed(stream: mtmd_audio_stream_p, /) -> int:
    """Get the number of audio rows produced through the last flush."""
    ...


# MTMD_API struct mtmd_video_config mtmd_get_video_config(
#     const mtmd_context * ctx, int32_t src_width, int32_t src_height,
#     int32_t src_n_frames, float src_fps);
@ctypes_function(
    "mtmd_get_video_config",
    [mtmd_context_p_ctypes, c_int32, c_int32, c_int32, c_float],
    mtmd_video_config,
)
def mtmd_get_video_config(
    ctx: mtmd_context_p,
    src_width: int,
    src_height: int,
    src_n_frames: int,
    src_fps: float,
    /,
) -> mtmd_video_config:
    """Get recommended video sampling and frame dimensions for the current model."""
    ...


# MTMD_API const char * mtmd_get_marker(const mtmd_context * ctx);
@ctypes_function("mtmd_get_marker", [mtmd_context_p_ctypes], c_char_p)
def mtmd_get_marker(ctx: mtmd_context_p, /) -> Optional[bytes]:
    """Get the current media marker string."""
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


# MTMD_API bool mtmd_bitmap_is_voice(const mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_is_voice", [mtmd_bitmap_p_ctypes], c_bool)
def mtmd_bitmap_is_voice(bitmap: mtmd_bitmap_p, /) -> bool:
    """Check whether the bitmap contains a predefined voice representation."""
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


# // if true, this bitmap can be merged (temporal merge) with an adjacent mergeable bitmap by certain video input models
# MTMD_API void mtmd_bitmap_set_mergeable(mtmd_bitmap * bitmap, bool mergeable);
@ctypes_function("mtmd_bitmap_set_mergeable", [mtmd_bitmap_p_ctypes, c_bool], None)
def mtmd_bitmap_set_mergeable(bitmap: mtmd_bitmap_p, mergeable: bool, /):
    """Allow temporal merging with an adjacent mergeable bitmap."""
    ...


# MTMD_API void mtmd_bitmap_set_is_video(mtmd_bitmap * bitmap, bool is_video);
@ctypes_function("mtmd_bitmap_set_is_video", [mtmd_bitmap_p_ctypes, c_bool], None)
def mtmd_bitmap_set_is_video(bitmap: mtmd_bitmap_p, is_video: bool, /):
    """Mark a bitmap or lazy bitmap as part of a video input."""
    ...


# MTMD_API mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx,
#                                              const char * id,
#                                              void * user_data,
#                                              mtmd_bitmap_lazy_callback callback);
@ctypes_function(
    "mtmd_bitmap_init_lazy",
    [mtmd_context_p_ctypes, c_char_p, c_void_p, mtmd_bitmap_lazy_callback],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init_lazy(
    ctx: mtmd_context_p,
    id: Optional[bytes],
    user_data: c_void_p,
    callback: mtmd_bitmap_lazy_callback,
    /,
) -> Optional[mtmd_bitmap_p]:
    """Initialize a lazy MTMD bitmap."""
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


# MTMD_API int32_t mtmd_tokenize(const mtmd_context * ctx,
#                                mtmd_input_chunks * output,
#                                const mtmd_input_text * text,
#                                const mtmd_bitmap * const * bitmaps,
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


# // same as mtmd_tokenize(), but takes an array of mtmd_input_part
# // use cases:
# // - when you don't want to use media markers (they will be tokenized as normal text)
# // - when you want to control parse_special for each text part
# // note: per-part add_special will be ignored
# // return 1 if a part has both text and bitmap set (or neither)
# MTMD_API int32_t mtmd_tokenize_from_parts(const mtmd_context * ctx,
#                                          mtmd_input_chunks * output,
#                                          const mtmd_input_part * const * parts,
#                                          size_t n_parts,
#                                          bool add_special);
@ctypes_function(
    "mtmd_tokenize_from_parts",
    [
        mtmd_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        POINTER(POINTER(mtmd_input_part)),
        c_size_t,
        c_bool,
    ],
    c_int32,
)
def mtmd_tokenize_from_parts(
    ctx: mtmd_context_p,
    output: mtmd_input_chunks_p,
    parts: CtypesArray[POINTER(mtmd_input_part)],
    n_parts: Union[c_size_t, int],
    add_special: bool,
    /,
) -> int:
    """Tokenize text and bitmap parts without substituting media markers."""
    ...


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


# // similar to mtmd_input_chunk_copy, but returns a placeholder chunk
# MTMD_API mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_input_chunk_get_placeholder",
    [mtmd_input_chunk_p_ctypes],
    mtmd_input_chunk_p_ctypes,
)
def mtmd_input_chunk_get_placeholder(
    chunk: mtmd_input_chunk_p, /
) -> Optional[mtmd_input_chunk_p]:
    """Like mtmd_input_chunk_copy, but returns a placeholder chunk."""
    ...


# // save/load an input chunk to/from a buffer (useful for KV save/load)
# // important: only chunk's metadata will be saved, the actual image/audio data will not be saved
# // the loaded chunk will always be a placeholder, cannot be used for mtmd_encode() or mtmd_batch_encode()
# // out_buf can be nullptr (to query expected_out_len)
# // returns 0 on success, non-zero on failure
# MTMD_API int32_t mtmd_input_chunk_save(const mtmd_input_chunk * chunk, char * out_buf, size_t out_len, size_t * expected_out_len);
@ctypes_function(
    "mtmd_input_chunk_save",
    [mtmd_input_chunk_p_ctypes, POINTER(c_char), c_size_t, POINTER(c_size_t)],
    c_int32,
)
def mtmd_input_chunk_save(
    chunk: mtmd_input_chunk_p,
    out_buf: Optional[CtypesArray[c_char]],
    out_len: Union[c_size_t, int],
    expected_out_len: "_Pointer[c_size_t]",
    /,
) -> int:
    """Save an input chunk's metadata to a buffer."""
    ...


# // returns nullptr on failure
# MTMD_API mtmd_input_chunk * mtmd_input_chunk_load(const char * buf, size_t len);
@ctypes_function(
    "mtmd_input_chunk_load",
    [c_char_p, c_size_t],
    mtmd_input_chunk_p_ctypes,
)
def mtmd_input_chunk_load(
    buf: bytes,
    length: Union[c_size_t, int],
    /,
) -> Optional[mtmd_input_chunk_p]:
    """Load an input chunk placeholder from saved metadata."""
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
    """Run a deprecated MTMD encode pass for image tokens."""
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


# MTMD_API mtmd_batch * mtmd_batch_init(mtmd_context * ctx);
@ctypes_function("mtmd_batch_init", [mtmd_context_p_ctypes], mtmd_batch_p_ctypes)
def mtmd_batch_init(ctx: mtmd_context_p, /) -> Optional[mtmd_batch_p]:
    """Initialize an MTMD media chunk batch for a context."""
    ...


# MTMD_API void mtmd_batch_free(mtmd_batch * batch);
@ctypes_function("mtmd_batch_free", [mtmd_batch_p_ctypes], None)
def mtmd_batch_free(batch: mtmd_batch_p, /): ...


# MTMD_API int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_batch_add_chunk",
    [mtmd_batch_p_ctypes, mtmd_input_chunk_p_ctypes],
    c_int,
)
def mtmd_batch_add_chunk(
    batch: mtmd_batch_p,
    chunk: mtmd_input_chunk_p,
    /,
) -> int:
    """Add a media chunk to an MTMD batch."""
    ...


# MTMD_API int32_t mtmd_batch_encode(mtmd_batch * batch);
@ctypes_function("mtmd_batch_encode", [mtmd_batch_p_ctypes], c_int)
def mtmd_batch_encode(batch: mtmd_batch_p, /) -> int:
    """Run an MTMD encode pass for all chunks in a batch."""
    ...


# MTMD_API float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk);
@ctypes_function(
    "mtmd_batch_get_output_embd",
    [mtmd_batch_p_ctypes, mtmd_input_chunk_p_ctypes],
    POINTER(c_float),
)
def mtmd_batch_get_output_embd(
    batch: mtmd_batch_p,
    chunk: mtmd_input_chunk_p,
    /,
) -> Optional[CtypesArray[c_float]]:
    """Get output embeddings for a chunk from the last batch encode pass."""
    ...


# MTMD_API struct mtmd_caps mtmd_get_cap_from_file(const char * mmproj_fname);
@ctypes_function("mtmd_get_cap_from_file", [c_char_p], mtmd_caps)
def mtmd_get_cap_from_file(mmproj_fname: bytes, /) -> mtmd_caps:
    """Get mmproj capabilities without initializing a full MTMD context."""
    ...


# MTMD_API struct mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * ctx);
@ctypes_function(
    "mtmd_gen_audio_get_info",
    [mtmd_context_p_ctypes],
    mtmd_gen_audio_info,
)
def mtmd_gen_audio_get_info(ctx: mtmd_context_p, /) -> mtmd_gen_audio_info:
    """Get audio generation information for an MTMD context."""
    ...


# MTMD_API size_t mtmd_gen_audio_read_tensor(
#     mtmd_context * ctx, const char * name, float * out, size_t n_max);
@ctypes_function(
    "mtmd_gen_audio_read_tensor",
    [mtmd_context_p_ctypes, c_char_p, POINTER(c_float), c_size_t],
    c_size_t,
)
def mtmd_gen_audio_read_tensor(
    ctx: mtmd_context_p,
    name: bytes,
    out: Optional[CtypesArray[c_float]],
    n_max: Union[c_size_t, int],
    /,
) -> int:
    """Read a named audio-generation tensor as F32 and return its element count."""
    ...


# MTMD_API const char * mtmd_gen_audio_voice_name(const mtmd_context * ctx, int32_t idx);
@ctypes_function(
    "mtmd_gen_audio_voice_name",
    [mtmd_context_p_ctypes, c_int32],
    c_char_p,
)
def mtmd_gen_audio_voice_name(
    ctx: mtmd_context_p,
    idx: int,
    /,
) -> Optional[bytes]:
    """Get the name of a predefined voice by index."""
    ...


# MTMD_API mtmd_bitmap * mtmd_gen_audio_get_voice(const mtmd_context * ctx, const char * name);
@ctypes_function(
    "mtmd_gen_audio_get_voice",
    [mtmd_context_p_ctypes, c_char_p],
    mtmd_bitmap_p_ctypes,
)
def mtmd_gen_audio_get_voice(
    ctx: mtmd_context_p,
    name: bytes,
    /,
) -> Optional[mtmd_bitmap_p]:
    """Create an owned voice bitmap for a predefined voice name."""
    ...


# MTMD_API struct mtmd_gen_state_params mtmd_gen_state_params_default(void);
@ctypes_function("mtmd_gen_state_params_default", [], mtmd_gen_state_params)
def mtmd_gen_state_params_default() -> mtmd_gen_state_params:
    """Get default audio generation state parameters."""
    ...


# MTMD_API mtmd_gen_state * mtmd_gen_state_init(mtmd_context * ctx, struct mtmd_gen_state_params params);
@ctypes_function(
    "mtmd_gen_state_init",
    [mtmd_context_p_ctypes, mtmd_gen_state_params],
    mtmd_gen_state_p_ctypes,
)
def mtmd_gen_state_init(
    ctx: mtmd_context_p,
    params: mtmd_gen_state_params,
    /,
) -> Optional[mtmd_gen_state_p]:
    """Initialize audio generation state, or return None if unsupported."""
    ...


# MTMD_API void mtmd_gen_state_free(mtmd_gen_state * state);
@ctypes_function("mtmd_gen_state_free", [mtmd_gen_state_p_ctypes], None)
def mtmd_gen_state_free(state: mtmd_gen_state_p, /):
    """Free audio generation state."""
    ...


# MTMD_API void mtmd_gen_state_reset(mtmd_gen_state * state);
@ctypes_function("mtmd_gen_state_reset", [mtmd_gen_state_p_ctypes], None)
def mtmd_gen_state_reset(state: mtmd_gen_state_p, /):
    """Reset audio generation state while preserving its RNG stream."""
    ...


# MTMD_API int32_t mtmd_gen_state_set_ref(mtmd_context * ctx, mtmd_gen_state * state);
@ctypes_function(
    "mtmd_gen_state_set_ref",
    [mtmd_context_p_ctypes, mtmd_gen_state_p_ctypes],
    c_int32,
)
def mtmd_gen_state_set_ref(
    ctx: mtmd_context_p,
    state: mtmd_gen_state_p,
    /,
) -> int:
    """Copy the most recently encoded speaker reference into generation state."""
    ...


# // defaults tuned for the loaded pipeline, callers override only what they care about
# MTMD_API struct mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * ctx);
@ctypes_function(
    "mtmd_gen_inp_default",
    [mtmd_context_p_ctypes],
    mtmd_gen_inp,
)
def mtmd_gen_inp_default(ctx: mtmd_context_p, /) -> mtmd_gen_inp:
    """Get default audio generation input parameters for an MTMD context."""
    ...


# // run one step and update the state in place
# MTMD_API int32_t mtmd_gen_audio_process(mtmd_context * ctx,
#                                 mtmd_gen_state * state,
#                                 const struct mtmd_gen_inp * inp,
#                                 struct mtmd_gen_out * out);
@ctypes_function(
    "mtmd_gen_audio_process",
    [
        mtmd_context_p_ctypes,
        mtmd_gen_state_p_ctypes,
        POINTER(mtmd_gen_inp),
        POINTER(mtmd_gen_out),
    ],
    c_int32,
)
def mtmd_gen_audio_process(
    ctx: mtmd_context_p,
    state: mtmd_gen_state_p,
    inp: "_Pointer[mtmd_gen_inp]",
    out: "_Pointer[mtmd_gen_out]",
    /,
) -> int:
    """Process one audio generation step."""
    ...


# MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);
@ctypes_function("mtmd_test_create_input_chunks", [], mtmd_input_chunks_p_ctypes)
def mtmd_test_create_input_chunks() -> Optional[mtmd_input_chunks_p]:
    """Create MTMD test chunks for the C API tests."""
    ...


################################################
# mtmd-helper.h functions
################################################


# MTMD_API bool mtmd_helper_support_video(const mtmd_context * ctx);
@ctypes_function(
    "mtmd_helper_support_video",
    [mtmd_context_p_ctypes],
    c_bool,
)
def mtmd_helper_support_video(ctx: mtmd_context_p, /) -> bool:
    """Check whether MTMD helper video support is available."""
    ...


# MTMD_API struct mtmd_helper_init_opt mtmd_helper_init_opt_default(void);
@ctypes_function("mtmd_helper_init_opt_default", [], mtmd_helper_init_opt)
def mtmd_helper_init_opt_default() -> mtmd_helper_init_opt:
    """Get default options for initializing media from files or buffers."""
    ...


# MTMD_API struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(
#                     const mtmd_context * ctx,
#                     const char * fname,
#                     bool placeholder,
#                     struct mtmd_helper_init_opt opt);
@ctypes_function(
    "mtmd_helper_bitmap_init_from_file",
    [mtmd_context_p_ctypes, c_char_p, c_bool, mtmd_helper_init_opt],
    mtmd_helper_bitmap_wrapper,
)
def mtmd_helper_bitmap_init_from_file_wrapper(
    ctx: mtmd_context_p,
    fname: bytes,
    placeholder: Union[c_bool, bool],
    opt: mtmd_helper_init_opt,
    /,
) -> mtmd_helper_bitmap_wrapper:
    """Initialize an MTMD bitmap wrapper from a file."""
    ...


def mtmd_helper_bitmap_init_from_file(
    ctx: mtmd_context_p,
    fname: bytes,
    placeholder: Union[c_bool, bool],
    opt: Optional[mtmd_helper_init_opt] = None,
    /,
) -> Optional[mtmd_bitmap_p]:
    """Initialize an MTMD bitmap from a file."""
    if opt is None:
        opt = mtmd_helper_init_opt_default()
    return mtmd_helper_bitmap_init_from_file_wrapper(
        ctx, fname, placeholder, opt
    ).bitmap


# MTMD_API struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
#                     const mtmd_context * ctx,
#                     const unsigned char * buf, size_t len,
#                     bool placeholder,
#                     struct mtmd_helper_init_opt opt);
@ctypes_function(
    "mtmd_helper_bitmap_init_from_buf",
    [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t, c_bool, mtmd_helper_init_opt],
    mtmd_helper_bitmap_wrapper,
)
def mtmd_helper_bitmap_init_from_buf_wrapper(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    length: Union[c_size_t, int],
    placeholder: Union[c_bool, bool],
    opt: mtmd_helper_init_opt,
    /,
) -> mtmd_helper_bitmap_wrapper: ...


def mtmd_helper_bitmap_init_from_buf(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    length: Union[c_size_t, int],
    placeholder: Union[c_bool, bool],
    opt: Optional[mtmd_helper_init_opt] = None,
    /,
) -> Optional[mtmd_bitmap_p]:
    """Initialize an MTMD bitmap from a buffer."""
    if opt is None:
        opt = mtmd_helper_init_opt_default()
    return mtmd_helper_bitmap_init_from_buf_wrapper(
        ctx, buf, length, placeholder, opt
    ).bitmap


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
#                                                 llama_pos * new_n_past,
#                                                 mtmd_helper_post_decode_callback callback,
#                                                 void * user_data);
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
        mtmd_helper_post_decode_callback,
        c_void_p,
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
    callback: Optional[mtmd_helper_post_decode_callback],
    user_data: c_void_p,
    /,
) -> int:
    """Decode a pre-encoded image chunk."""
    ...


# MTMD_API struct mtmd_helper_video_init_params mtmd_helper_video_init_params_default(void);
@ctypes_function(
    "mtmd_helper_video_init_params_default", [], mtmd_helper_video_init_params
)
def mtmd_helper_video_init_params_default() -> mtmd_helper_video_init_params:
    """Return the default MTMD helper video initialization parameters."""
    ...


# MTMD_API mtmd_helper_video * mtmd_helper_video_init(
#                     const struct mtmd_context * mctx,
#                     const char * path,
#                     struct mtmd_helper_video_init_params params);
@ctypes_function(
    "mtmd_helper_video_init",
    [mtmd_context_p_ctypes, c_char_p, mtmd_helper_video_init_params],
    mtmd_helper_video_p_ctypes,
)
def mtmd_helper_video_init(
    ctx: mtmd_context_p,
    path: bytes,
    params: mtmd_helper_video_init_params,
    /,
) -> Optional[mtmd_helper_video_p]:
    """Initialize an MTMD helper video stream from a file path."""
    ...


# MTMD_API mtmd_helper_video * mtmd_helper_video_init_from_buf(
#                     const struct mtmd_context * mctx,
#                     const unsigned char * buf, size_t len,
#                     struct mtmd_helper_video_init_params params);
@ctypes_function(
    "mtmd_helper_video_init_from_buf",
    [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t, mtmd_helper_video_init_params],
    mtmd_helper_video_p_ctypes,
)
def mtmd_helper_video_init_from_buf(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    length: Union[c_size_t, int],
    params: mtmd_helper_video_init_params,
    /,
) -> Optional[mtmd_helper_video_p]:
    """Initialize an MTMD helper video stream from a buffer."""
    ...


# MTMD_API void mtmd_helper_video_free(mtmd_helper_video * ctx);
@ctypes_function("mtmd_helper_video_free", [mtmd_helper_video_p_ctypes], None)
def mtmd_helper_video_free(ctx: mtmd_helper_video_p, /):
    """Free an MTMD helper video stream."""
    ...


# MTMD_API struct mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx);
@ctypes_function(
    "mtmd_helper_video_get_info",
    [mtmd_helper_video_p_ctypes],
    mtmd_helper_video_info,
)
def mtmd_helper_video_get_info(ctx: mtmd_helper_video_p, /) -> mtmd_helper_video_info:
    """Get metadata for an MTMD helper video stream."""
    ...


# MTMD_API int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
#             mtmd_bitmap ** out_bitmap,
#             char ** out_text);
@ctypes_function(
    "mtmd_helper_video_read_next",
    [
        mtmd_helper_video_p_ctypes,
        POINTER(mtmd_bitmap_p_ctypes),
        POINTER(c_char_p),
    ],
    c_int,
)
def mtmd_helper_video_read_next(
    ctx: mtmd_helper_video_p,
    out_bitmap: "_Pointer[mtmd_bitmap_p_ctypes]",
    out_text: "_Pointer[c_char_p]",
    /,
) -> int:
    """Read the next bitmap or text chunk from an MTMD helper video stream."""
    ...


# // return true if model can be used for chat
# MTMD_API bool mtmd_helper_model_can_chat(const struct llama_context * lctx, const struct mtmd_context * mctx);
@ctypes_function(
    "mtmd_helper_model_can_chat",
    [llama_cpp.llama_context_p_ctypes, mtmd_context_p_ctypes],
    c_bool,
)
def mtmd_helper_model_can_chat(
    lctx: llama_cpp.llama_context_p,
    mctx: mtmd_context_p,
    /,
) -> bool:
    """Return whether the model can be used for chat."""
    ...


# MTMD_API struct mtmd_helper_cap mtmd_helper_get_capabilities(
#     const struct llama_model * model, const struct mtmd_context * mctx);
@ctypes_function(
    "mtmd_helper_get_capabilities",
    [llama_cpp.llama_model_p_ctypes, mtmd_context_p_ctypes],
    mtmd_helper_cap,
)
def mtmd_helper_get_capabilities(
    model: llama_cpp.llama_model_p,
    mctx: Optional[mtmd_context_p],
    /,
) -> mtmd_helper_cap:
    """Get model and projector capabilities."""
    ...


# MTMD_API struct mtmd_helper_gen_audio_inp mtmd_helper_gen_audio_inp_default(void);
@ctypes_function(
    "mtmd_helper_gen_audio_inp_default",
    [],
    mtmd_helper_gen_audio_inp,
)
def mtmd_helper_gen_audio_inp_default() -> mtmd_helper_gen_audio_inp:
    """Get default audio generation helper input parameters."""
    ...


# MTMD_API mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(
#                                     struct llama_context * lctx,
#                                     struct mtmd_context * mctx);
@ctypes_function(
    "mtmd_helper_gen_audio_init",
    [llama_cpp.llama_context_p_ctypes, mtmd_context_p_ctypes],
    mtmd_helper_gen_audio_p_ctypes,
)
def mtmd_helper_gen_audio_init(
    lctx: llama_cpp.llama_context_p,
    mctx: mtmd_context_p,
    /,
) -> Optional[mtmd_helper_gen_audio_p]:
    """Initialize an audio generation helper context."""
    ...


# MTMD_API void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx);
@ctypes_function(
    "mtmd_helper_gen_audio_free",
    [mtmd_helper_gen_audio_p_ctypes],
    None,
)
def mtmd_helper_gen_audio_free(ctx: mtmd_helper_gen_audio_p, /): ...


# MTMD_API void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx);
@ctypes_function(
    "mtmd_helper_gen_audio_reset",
    [mtmd_helper_gen_audio_p_ctypes],
    None,
)
def mtmd_helper_gen_audio_reset(ctx: mtmd_helper_gen_audio_p, /): ...


# MTMD_API int32_t mtmd_helper_gen_audio_set_input(
#                         mtmd_helper_gen_audio * ctx,
#                         const struct mtmd_helper_gen_audio_inp * inp);
@ctypes_function(
    "mtmd_helper_gen_audio_set_input",
    [mtmd_helper_gen_audio_p_ctypes, POINTER(mtmd_helper_gen_audio_inp)],
    c_int32,
)
def mtmd_helper_gen_audio_set_input(
    ctx: mtmd_helper_gen_audio_p,
    inp: "_Pointer[mtmd_helper_gen_audio_inp]",
    /,
) -> int:
    """Set the audio generation helper input."""
    ...


# // processes at most n_batch prompt tokens per call
# // returns: >0 = number of prompt tokens remaining, 0 = done, <0 = error
# MTMD_API int32_t mtmd_helper_gen_audio_step_prompt(
#                         mtmd_helper_gen_audio * ctx,
#                         int32_t n_batch);
@ctypes_function(
    "mtmd_helper_gen_audio_step_prompt",
    [mtmd_helper_gen_audio_p_ctypes, c_int32],
    c_int32,
)
def mtmd_helper_gen_audio_step_prompt(
    ctx: mtmd_helper_gen_audio_p,
    n_batch: int,
    /,
) -> int:
    """Process up to n_batch prompt tokens."""
    ...


# // generates one frame; must only be called after step_prompt() has returned 0
# // sampled can be LLAMA_TOKEN_NULL for pipelines with no discrete backbone token
# // out_stop (optional) is set on end-of-speech, the caller must then stop the loop
# // h_state_out is valid until next step_gen() or reset() call, None if no frame is generated
# MTMD_API int32_t mtmd_helper_gen_audio_step_gen(
#                         mtmd_helper_gen_audio * ctx,
#                         llama_token sampled,
#                         const float *  h_state_in,
#                         const float ** h_state_out,
#                         bool * out_stop);
@ctypes_function(
    "mtmd_helper_gen_audio_step_gen",
    [
        mtmd_helper_gen_audio_p_ctypes,
        llama_cpp.llama_token,
        POINTER(c_float),
        POINTER(POINTER(c_float)),
        POINTER(c_bool),
    ],
    c_int32,
)
def mtmd_helper_gen_audio_step_gen(
    ctx: mtmd_helper_gen_audio_p,
    sampled: llama_cpp.llama_token,
    h_state_in: Optional["_Pointer[c_float]"],
    h_state_out: "_Pointer[_Pointer[c_float]]",
    out_stop: Optional["_Pointer[c_bool]"],
    /,
) -> int:
    """Generate one audio frame."""
    ...


# MTMD_API bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio * ctx);
@ctypes_function(
    "mtmd_helper_gen_audio_next_is_text",
    [mtmd_helper_gen_audio_p_ctypes],
    c_bool,
)
def mtmd_helper_gen_audio_next_is_text(ctx: mtmd_helper_gen_audio_p, /) -> bool:
    """Check whether the next audio generation step expects a sampled text token."""
    ...


# MTMD_API const llama_logit_bias * mtmd_helper_gen_audio_get_logit_bias(
#     mtmd_helper_gen_audio * ctx, size_t * n_bias);
@ctypes_function(
    "mtmd_helper_gen_audio_get_logit_bias",
    [mtmd_helper_gen_audio_p_ctypes, POINTER(c_size_t)],
    llama_cpp.llama_logit_bias_p,
)
def mtmd_helper_gen_audio_get_logit_bias(
    ctx: mtmd_helper_gen_audio_p,
    n_bias: "_Pointer[c_size_t]",
    /,
) -> Optional[CtypesArray[llama_cpp.llama_logit_bias]]:
    """Get the logit bias for tokens sampled by the audio generation helper."""
    ...


# MTMD_API int32_t mtmd_helper_audio_stream_get_logit_bias(
#     const mtmd_context * mctx, const struct llama_model * model,
#     llama_logit_bias * out, int32_t n_max);
@ctypes_function(
    "mtmd_helper_audio_stream_get_logit_bias",
    [
        mtmd_context_p_ctypes,
        llama_cpp.llama_model_p_ctypes,
        POINTER(llama_cpp.llama_logit_bias),
        c_int32,
    ],
    c_int32,
)
def mtmd_helper_audio_stream_get_logit_bias(
    mctx: mtmd_context_p,
    model: llama_cpp.llama_model_p,
    out: Optional[CtypesArray[llama_cpp.llama_logit_bias]],
    n_max: int,
    /,
) -> int:
    """Get token biases for text sampled while transcribing an audio stream."""
    ...


# // out_data valid until next get_output() or reset() call
# // out_n_samples (optional, can be NULL) receives the number of generated PCM samples
# MTMD_API int32_t mtmd_helper_gen_audio_get_output(
#                         mtmd_helper_gen_audio * ctx,
#                         int32_t * out_sample_rate,
#                         const char ** out_data,
#                         size_t * out_data_len,
#                         int64_t * out_n_samples);
@ctypes_function(
    "mtmd_helper_gen_audio_get_output",
    [
        mtmd_helper_gen_audio_p_ctypes,
        POINTER(c_int32),
        POINTER(POINTER(c_char)),
        POINTER(c_size_t),
        POINTER(c_int64),
    ],
    c_int32,
)
def mtmd_helper_gen_audio_get_output(
    ctx: mtmd_helper_gen_audio_p,
    out_sample_rate: "_Pointer[c_int32]",
    out_data: "_Pointer[_Pointer[c_char]]",
    out_data_len: "_Pointer[c_size_t]",
    out_n_samples: Optional["_Pointer[c_int64]"],
    /,
) -> int:
    """Get accumulated PCM or WAV audio output."""
    ...


# MTMD_API int32_t mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio * ctx);
@ctypes_function(
    "mtmd_helper_gen_audio_flush", [mtmd_helper_gen_audio_p_ctypes], c_int32
)
def mtmd_helper_gen_audio_flush(ctx: mtmd_helper_gen_audio_p, /) -> int:
    """Flush buffered audio codes through the decoder."""
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
