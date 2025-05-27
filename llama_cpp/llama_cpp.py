from __future__ import annotations

import os
import ctypes
import pathlib

from typing import (
    Callable,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    byref,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesCData,
        CtypesArray,
        CtypesPointer,
        CtypesVoidPointer,
        CtypesRef,
        CtypesPointerOrRef,
        CtypesFuncPointer,
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"
_override_base_path = os.environ.get("LLAMA_CPP_LIB_PATH")
_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _override_base_path is None else pathlib.Path(_override_base_path)
# Load the library
_lib = load_shared_library(_lib_base_name, _base_path)

ctypes_function = ctypes_function_for_shared_library(_lib)


# from ggml.h
# // NOTE: always add types at the end of the enum to keep backward compatibility
# enum ggml_type {
#     GGML_TYPE_F32     = 0,
#     GGML_TYPE_F16     = 1,
#     GGML_TYPE_Q4_0    = 2,
#     GGML_TYPE_Q4_1    = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 = 5, support has been removed
#     GGML_TYPE_Q5_0    = 6,
#     GGML_TYPE_Q5_1    = 7,
#     GGML_TYPE_Q8_0    = 8,
#     GGML_TYPE_Q8_1    = 9,
#     GGML_TYPE_Q2_K    = 10,
#     GGML_TYPE_Q3_K    = 11,
#     GGML_TYPE_Q4_K    = 12,
#     GGML_TYPE_Q5_K    = 13,
#     GGML_TYPE_Q6_K    = 14,
#     GGML_TYPE_Q8_K    = 15,
#     GGML_TYPE_IQ2_XXS = 16,
#     GGML_TYPE_IQ2_XS  = 17,
#     GGML_TYPE_IQ3_XXS = 18,
#     GGML_TYPE_IQ1_S   = 19,
#     GGML_TYPE_IQ4_NL  = 20,
#     GGML_TYPE_IQ3_S   = 21,
#     GGML_TYPE_IQ2_S   = 22,
#     GGML_TYPE_IQ4_XS  = 23,
#     GGML_TYPE_I8      = 24,
#     GGML_TYPE_I16     = 25,
#     GGML_TYPE_I32     = 26,
#     GGML_TYPE_I64     = 27,
#     GGML_TYPE_F64     = 28,
#     GGML_TYPE_IQ1_M   = 29,
#     GGML_TYPE_COUNT,
# };
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_COUNT = 30

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

# define LLAMA_TOKEN_NULL -1
LLAMA_TOKEN_NULL = -1

# define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
LLAMA_FILE_MAGIC_GGLA = 0x67676C61

# define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
LLAMA_FILE_MAGIC_GGSN = 0x6767736E

# define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'
LLAMA_FILE_MAGIC_GGSQ = 0x67677371

# define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
# define LLAMA_SESSION_VERSION 9
LLAMA_SESSION_VERSION = 9

# define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ
# define LLAMA_STATE_SEQ_VERSION 2
LLAMA_STATE_SEQ_VERSION = 2

# struct llama_vocab;
llama_vocab_p = NewType("llama_vocab_p", int)
llama_vocab_p_ctypes = ctypes.c_void_p

# struct llama_model;
llama_model_p = NewType("llama_model_p", int)
llama_model_p_ctypes = ctypes.c_void_p

# struct llama_context;
llama_context_p = NewType("llama_context_p", int)
llama_context_p_ctypes = ctypes.c_void_p

# # struct llama_sampler;
# llama_sampler_p = NewType("llama_sampler_p", int)
# llama_sampler_p_ctypes = ctypes.c_void_p

# struct llama_kv_cache;
llama_kv_cache_p = NewType("llama_kv_cache_p", int)
llama_kv_cache_p_ctypes = ctypes.c_void_p

# typedef int32_t llama_pos;
llama_pos = ctypes.c_int32
# typedef int32_t llama_token;
llama_token = ctypes.c_int32
llama_token_p = ctypes.POINTER(llama_token)
# typedef int32_t llama_seq_id;
llama_seq_id = ctypes.c_int32


# enum llama_vocab_type {
#     LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
#     LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
#     LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
#     LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
#     LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
#     LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
# };
LLAMA_VOCAB_TYPE_NONE = 0
"""For models without vocab"""
LLAMA_VOCAB_TYPE_SPM = 1
"""LLaMA tokenizer based on byte-level BPE with byte fallback"""
LLAMA_VOCAB_TYPE_BPE = 2
"""GPT-2 tokenizer based on byte-level BPE"""
LLAMA_VOCAB_TYPE_WPM = 3
"""BERT tokenizer based on WordPiece"""
LLAMA_VOCAB_TYPE_UGM = 4
"""T5 tokenizer based on Unigram"""
LLAMA_VOCAB_TYPE_RWKV = 5
"""RWKV tokenizer based on greedy tokenization"""


# // pre-tokenization types
# enum llama_vocab_pre_type {
#     LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
#     LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
#     LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
#     LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
#     LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
#     LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
#     LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
#     LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
#     LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
#     LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
#     LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
#     LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
#     LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
#     LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
#     LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
#     LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
#     LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
#     LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
#     LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
#     LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
#     LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
#     LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
#     LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
#     LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23,
#     LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
#     LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25,
#     LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26,
#     LLAMA_VOCAB_PRE_TYPE_MINERVA        = 27,
#     LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28,
#     LLAMA_VOCAB_PRE_TYPE_GPT4O          = 29,
#     LLAMA_VOCAB_PRE_TYPE_SUPERBPE       = 30,
#     LLAMA_VOCAB_PRE_TYPE_TRILLION       = 31,
#     LLAMA_VOCAB_PRE_TYPE_BAILINGMOE     = 32,
#     LLAMA_VOCAB_PRE_TYPE_LLAMA4         = 33,
#     LLAMA_VOCAB_PRE_TYPE_PIXTRAL        = 34,
# };
LLAMA_VOCAB_PRE_TYPE_DEFAULT = 0
LLAMA_VOCAB_PRE_TYPE_LLAMA3 = 1
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM = 2
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3
LLAMA_VOCAB_PRE_TYPE_FALCON = 4
LLAMA_VOCAB_PRE_TYPE_MPT = 5
LLAMA_VOCAB_PRE_TYPE_STARCODER = 6
LLAMA_VOCAB_PRE_TYPE_GPT2 = 7
LLAMA_VOCAB_PRE_TYPE_REFACT = 8
LLAMA_VOCAB_PRE_TYPE_COMMAND_R = 9
LLAMA_VOCAB_PRE_TYPE_STABLELM2 = 10
LLAMA_VOCAB_PRE_TYPE_QWEN2 = 11
LLAMA_VOCAB_PRE_TYPE_OLMO = 12
LLAMA_VOCAB_PRE_TYPE_DBRX = 13
LLAMA_VOCAB_PRE_TYPE_SMAUG = 14
LLAMA_VOCAB_PRE_TYPE_PORO = 15
LLAMA_VOCAB_PRE_TYPE_CHATGLM3 = 16
LLAMA_VOCAB_PRE_TYPE_CHATGLM4 = 17
LLAMA_VOCAB_PRE_TYPE_VIKING = 18
LLAMA_VOCAB_PRE_TYPE_JAIS = 19
LLAMA_VOCAB_PRE_TYPE_TEKKEN = 20
LLAMA_VOCAB_PRE_TYPE_SMOLLM = 21
LLAMA_VOCAB_PRE_TYPE_CODESHELL = 22
LLAMA_VOCAB_PRE_TYPE_BLOOM = 23
LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH = 24
LLAMA_VOCAB_PRE_TYPE_EXAONE = 25
LLAMA_VOCAB_PRE_TYPE_CHAMELEON = 26
LLAMA_VOCAB_PRE_TYPE_MINERVA = 27
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM = 28
LLAMA_VOCAB_PRE_TYPE_GPT4O = 29
LLAMA_VOCAB_PRE_TYPE_SUPERBPE = 30
LLAMA_VOCAB_PRE_TYPE_TRILLION = 31
LLAMA_VOCAB_PRE_TYPE_BAILINGMOE = 32
LLAMA_VOCAB_PRE_TYPE_LLAMA4 = 33
LLAMA_VOCAB_PRE_TYPE_PIXTRAL = 34


# // note: these values should be synchronized with ggml_rope
# // TODO: maybe move this enum to ggml.h (ggml_rope_type)
# enum llama_rope_type {
#     LLAMA_ROPE_TYPE_NONE   = -1,
#     LLAMA_ROPE_TYPE_NORM   = 0,
#     LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
#     LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
#     LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
# };
LLAMA_ROPE_TYPE_NONE = -1
LLAMA_ROPE_TYPE_NORM = 0
LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX = 2
LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE = 8
LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION = 24


# enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
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


# enum llama_token_attr {
#     LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
#     LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
#     LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
#     LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
#     LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
#     LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
#     LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
#     LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
#     LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
#     LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
#     LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
# };
LLAMA_TOKEN_ATTR_UNDEFINED = 0
LLAMA_TOKEN_ATTR_UNKNOWN = 1 << 0
LLAMA_TOKEN_ATTR_UNUSED = 1 << 1
LLAMA_TOKEN_ATTR_NORMAL = 1 << 2
LLAMA_TOKEN_ATTR_CONTROL = 1 << 3
LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
LLAMA_TOKEN_ATTR_BYTE = 1 << 5
LLAMA_TOKEN_ATTR_NORMALIZED = 1 << 6
LLAMA_TOKEN_ATTR_LSTRIP = 1 << 7
LLAMA_TOKEN_ATTR_RSTRIP = 1 << 8
LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 << 9


# // model file types
# enum llama_ftype {
#     LLAMA_FTYPE_ALL_F32              = 0,
#     LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
#     // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
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
#     LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
#     //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
#     //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
#     //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
#     LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
#
#     LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
# };
LLAMA_FTYPE_ALL_F32 = 0
LLAMA_FTYPE_MOSTLY_F16 = 1
LLAMA_FTYPE_MOSTLY_Q4_0 = 2
LLAMA_FTYPE_MOSTLY_Q4_1 = 3
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
LLAMA_FTYPE_MOSTLY_IQ1_M = 31
LLAMA_FTYPE_MOSTLY_BF16 = 32
# LLAMA_FTYPE_MOSTLY_Q4_0_4_4 = 33
# LLAMA_FTYPE_MOSTLY_Q4_0_4_8 = 34
# LLAMA_FTYPE_MOSTLY_Q4_0_8_8 = 35
LLAMA_FTYPE_MOSTLY_TQ1_0 = 36
LLAMA_FTYPE_MOSTLY_TQ2_0 = 37
LLAMA_FTYPE_GUESSED = 1024

# enum llama_rope_scaling_type {
#     LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
#     LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
#     LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
#     LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
#     LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
#     LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_YARN,
# };
LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
LLAMA_ROPE_SCALING_TYPE_NONE = 0
LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
LLAMA_ROPE_SCALING_TYPE_YARN = 2
LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3
LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_YARN

# enum llama_pooling_type {
#     LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
#     LLAMA_POOLING_TYPE_NONE = 0,
#     LLAMA_POOLING_TYPE_MEAN = 1,
#     LLAMA_POOLING_TYPE_CLS  = 2,
#     LLAMA_POOLING_TYPE_LAST = 3,
#     LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
# };
LLAMA_POOLING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2
LLAMA_POOLING_TYPE_LAST = 3
LLAMA_POOLING_TYPE_RANK = 4

# enum llama_attention_type {
#     LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
#     LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
#     LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
# };
LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
LLAMA_ATTENTION_TYPE_CAUSAL = 0
LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1


# enum llama_split_mode {
#     LLAMA_SPLIT_MODE_NONE  = 0, // single GPU
#     LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
#     LLAMA_SPLIT_MODE_ROW   = 2, // split rows across GPUs
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

    if TYPE_CHECKING:
        id: llama_token
        logit: float
        p: float

    _fields_ = [
        ("id", llama_token),
        ("logit", ctypes.c_float),
        ("p", ctypes.c_float),
    ]


llama_token_data_p = ctypes.POINTER(llama_token_data)


# typedef struct llama_token_data_array {
#     // TODO: consider SoA
#     // NOTE: this pointer can be modified by the samplers
#     llama_token_data * data;
#     size_t size;
#     int64_t selected; // this is the index in the data array (i.e. not the token id)
#     bool sorted;
# } llama_token_data_array;
class llama_token_data_array(ctypes.Structure):
    """Used to sample tokens given logits

    Attributes:
        data (ctypes.Array[llama_token_data]): token data
        size (int): size of the array
        selected (int): index in the data array (i.e. not the token id)
        sorted (bool): whether the array is sorted"""

    if TYPE_CHECKING:
        data: CtypesArray[llama_token_data]
        size: int
        selected: int
        sorted: bool

    _fields_ = [
        ("data", llama_token_data_p),
        ("size", ctypes.c_size_t),
        ("selected", ctypes.c_int64),
        ("sorted", ctypes.c_bool),
    ]


llama_token_data_array_p = ctypes.POINTER(llama_token_data_array)

# typedef bool (*llama_progress_callback)(float progress, void * user_data);
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
# //            (if set to NULL, the token position will be tracked automatically by llama_decode)
# // - seq_id : the sequence to which the respective token belongs
# //            (if set to NULL, the sequence ID will be assumed to be 0)
# // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
# //            (if set to NULL, only the logits for last token will be returned)
# //
# typedef struct llama_batch {
#     int32_t n_tokens;

#     llama_token  *  token;
#     float        *  embd;
#     llama_pos    *  pos;
#     int32_t      *  n_seq_id;
#     llama_seq_id ** seq_id;
#     int8_t       *  logits; // TODO: rename this to "output"
# } llama_batch;
class llama_batch(ctypes.Structure):
    """Input data for llama_decode

    A llama_batch object can contain input about one or many sequences

    The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens

    Attributes:
        n_tokens (int): number of tokens
        token (ctypes.Array[llama_token]): the token ids of the input (used when embd is NULL)
        embd (ctypes.Array[ctypes.ctypes.c_float]): token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        pos (ctypes.Array[ctypes.Array[llama_pos]]): the positions of the respective token in the sequence
        seq_id (ctypes.Array[ctypes.Array[llama_seq_id]]): the sequence to which the respective token belongs
        logits (ctypes.Array[ctypes.ctypes.c_int8]): if zero, the logits for the respective token will not be output
    """

    if TYPE_CHECKING:
        n_tokens: int
        token: CtypesArray[llama_token]
        embd: CtypesArray[ctypes.c_float]
        pos: CtypesArray[CtypesArray[llama_pos]]
        n_seq_id: CtypesArray[ctypes.c_int]
        seq_id: CtypesArray[CtypesArray[llama_seq_id]]
        logits: CtypesArray[ctypes.c_int8]

    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


# enum llama_model_kv_override_type {
#     LLAMA_KV_OVERRIDE_TYPE_INT,
#     LLAMA_KV_OVERRIDE_TYPE_FLOAT,
#     LLAMA_KV_OVERRIDE_TYPE_BOOL,
#     LLAMA_KV_OVERRIDE_TYPE_STR,
# };
LLAMA_KV_OVERRIDE_TYPE_INT = 0
LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
LLAMA_KV_OVERRIDE_TYPE_BOOL = 2
LLAMA_KV_OVERRIDE_TYPE_STR = 3


# struct llama_model_kv_override {
#     enum llama_model_kv_override_type tag;

#     char key[128];


#     union {
#         int64_t val_i64;
#         double  val_f64;
#         bool    val_bool;
#         char    val_str[128];
#     };
# };
class llama_model_kv_override_value(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]

    if TYPE_CHECKING:
        val_i64: int
        val_f64: float
        val_bool: bool
        val_str: bytes


class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("key", ctypes.c_char * 128),
        ("value", llama_model_kv_override_value),
    ]

    if TYPE_CHECKING:
        tag: int
        key: bytes
        value: Union[int, float, bool, bytes]


# struct llama_model_tensor_buft_override {
#     const char * pattern;
#     ggml_backend_buffer_type_t buft;
# };


# struct llama_model_params {
#     // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
#     ggml_backend_dev_t * devices;

#     // NULL-terminated list of buffer types to use for tensors that match a pattern
#     const struct llama_model_tensor_buft_override * tensor_buft_overrides;

#     int32_t n_gpu_layers; // number of layers to store in VRAM
#     enum llama_split_mode split_mode; // how to split the model across multiple GPUs

#     // main_gpu interpretation depends on split_mode:
#     // LLAMA_SPLIT_MODE_NONE: the GPU that is used for the entire model
#     // LLAMA_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results
#     // LLAMA_SPLIT_MODE_LAYER: ignored
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
#     bool vocab_only;    // only load the vocabulary, no weights
#     bool use_mmap;      // use mmap if possible
#     bool use_mlock;     // force system to keep model in RAM
#     bool check_tensors; // validate model tensor data
# };
class llama_model_params(ctypes.Structure):
    """Parameters for llama_model

    Attributes:
        devices (ctypes.Array[ggml_backend_dev_t]): NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        tensor_buft_overrides (ctypes.Array[llama_model_tensor_buft_override]): NULL-terminated list of buffer types to use for tensors that match a pattern
        n_gpu_layers (int): number of layers to store in VRAM
        split_mode (int): how to split the model across multiple GPUs
        main_gpu (int): the GPU that is used for the entire model. main_gpu interpretation depends on split_mode: LLAMA_SPLIT_NONE: the GPU that is used for the entire model LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results LLAMA_SPLIT_LAYER: ignored
        tensor_split (ctypes.Array[ctypes.ctypes.c_float]): proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        progress_callback (llama_progress_callback): called with a progress value between 0.0 and 1.0. Pass NULL to disable. If the provided progress_callback returns true, model loading continues. If it returns false, model loading is immediately aborted.
        progress_callback_user_data (ctypes.ctypes.c_void_p): context pointer passed to the progress callback
        kv_overrides (ctypes.Array[llama_model_kv_override]): override key-value pairs of the model meta data
        vocab_only (bool): only load the vocabulary, no weights
        use_mmap (bool): use mmap if possible
        use_mlock (bool): force system to keep model in RAM
        check_tensors (bool): validate model tensor data"""

    if TYPE_CHECKING:
        devices: CtypesArray[ctypes.c_void_p]  # NOTE: unused
        tensor_buft_overrides: CtypesArray[llama_model_tensor_buft_override] # NOTE: unused
        n_gpu_layers: int
        split_mode: int
        main_gpu: int
        tensor_split: CtypesArray[ctypes.c_float]
        progress_callback: Callable[[float, ctypes.c_void_p], bool]
        progress_callback_user_data: ctypes.c_void_p
        kv_overrides: CtypesArray[llama_model_kv_override]
        vocab_only: bool
        use_mmap: bool
        use_mlock: bool
        check_tensors: bool

    _fields_ = [
        ("devices", ctypes.c_void_p), # NOTE: unnused
        ("tensor_buft_overrides", ctypes.c_void_p), # NOTE: unused
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
        ("check_tensors", ctypes.c_bool),
    ]


# // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
# //       https://github.com/ggerganov/llama.cpp/pull/7544
# struct llama_context_params {
#     uint32_t n_ctx;             // text context, 0 = from model
#     uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
#     uint32_t n_ubatch;          // physical maximum batch size
#     uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
#     int32_t  n_threads;         // number of threads to use for generation
#     int32_t  n_threads_batch;   // number of threads to use for batch processing

#     enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
#     enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
#     enum llama_attention_type    attention_type;    // attention type to use for embeddings

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

#     enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
#     enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
#     bool embeddings;  // if true, extract embeddings (together with logits)
#     bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
#     bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
#     bool no_perf;     // whether to measure performance timings


#     // Abort callback
#     // if it returns true, execution of llama_decode() will be aborted
#     // currently works only with CPU execution
#     ggml_abort_callback abort_callback;
#     void *              abort_callback_data;
# };
class llama_context_params(ctypes.Structure):
    """Parameters for llama_context

    Attributes:
        n_ctx (int): text context, 0 = from model
        n_batch (int): logical maximum batch size that can be submitted to llama_decode
        n_ubatch (int): physical maximum batch size
        n_seq_max (int): max number of sequences (i.e. distinct states for recurrent models)
        n_threads (int): number of threads to use for generation
        n_threads_batch (int): number of threads to use for batch processing
        rope_scaling_type (int): RoPE scaling type, from `enum llama_rope_scaling_type`
        pooling_type (int): whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
        attention_type (int): attention type to use for embeddings
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
        logits_all (bool): the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        embeddings (bool): if true, extract embeddings (together with logits)
        offload_kqv (bool): whether to offload the KQV ops (including the KV cache) to GPU
        flash_attn (bool): whether to use flash attention
        no_perf (bool): whether to measure performance timings
        abort_callback (ggml_abort_callback): abort callback if it returns true, execution of llama_decode() will be aborted
        abort_callback_data (ctypes.ctypes.c_void_p): data for abort_callback
    """

    if TYPE_CHECKING:
        n_ctx: int
        n_batch: int
        n_ubatch: int
        n_seq_max: int
        n_threads: int
        n_threads_batch: int
        rope_scaling_type: int
        pooling_type: int
        attention_type: int
        rope_freq_base: float
        rope_freq_scale: float
        yarn_ext_factor: float
        yarn_attn_factor: float
        yarn_beta_fast: float
        yarn_beta_slow: float
        yarn_orig_ctx: int
        defrag_thold: float
        cb_eval: Callable[[ctypes.c_void_p, bool], bool]
        cb_eval_user_data: ctypes.c_void_p
        type_k: int
        type_v: int
        logits_all: bool
        embeddings: bool
        offload_kqv: bool
        flash_attn: bool
        no_perf: bool
        abort_callback: Callable[[ctypes.c_void_p], bool]
        abort_callback_data: ctypes.c_void_p

    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
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
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
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
#     int32_t nthread;                      // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
#     enum llama_ftype ftype;               // quantize to this llama_ftype
#     enum ggml_type output_tensor_type;    // output tensor type
#     enum ggml_type token_embedding_type;  // token embeddings tensor type
#     bool allow_requantize;                // allow quantizing non-f32/f16 tensors
#     bool quantize_output_tensor;          // quantize output.weight
#     bool only_copy;                       // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
#     bool pure;                            // quantize all tensors to the default type
#     bool keep_split;                      // quantize to the same number of shards
#     void * imatrix;                       // pointer to importance matrix data
#     void * kv_overrides;                  // pointer to vector containing overrides
#     void * tensor_types;                  // pointer to vector containing tensor types
# } llama_model_quantize_params;
class llama_model_quantize_params(ctypes.Structure):
    """Parameters for llama_model_quantize

    Attributes:
        nthread (int): number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ftype (int): quantize to this llama_ftype
        output_tensor_type (int): output tensor type
        token_embedding_type (int): token embeddings tensor type
        allow_requantize (bool): allow quantizing non-f32/f16 tensors
        quantize_output_tensor (bool): quantize output.weight
        only_copy (bool): only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        pure (bool): quantize all tensors to the default type
        keep_split (bool): quantize to the same number of shards
        imatrix (ctypes.c_void_p): pointer to importance matrix data
        kv_overrides (ctypes.c_void_p): pointer to vector containing overrides
        tensor_types (ctypes.c_void_p): pointer to vector containing tensor types
    """

    if TYPE_CHECKING:
        nthread: int
        ftype: int
        output_tensor_type: int
        token_embedding_type: int
        allow_requantize: bool
        quantize_output_tensor: bool
        only_copy: bool
        pure: bool
        keep_split: bool
        imatrix: ctypes.c_void_p
        kv_overrides: ctypes.c_void_p
        tensor_types: ctypes.c_void_p

    _fields_ = [
        ("nthread", ctypes.c_int32),
        ("ftype", ctypes.c_int),
        ("output_tensor_type", ctypes.c_int),
        ("token_embedding_type", ctypes.c_int),
        ("allow_requantize", ctypes.c_bool),
        ("quantize_output_tensor", ctypes.c_bool),
        ("only_copy", ctypes.c_bool),
        ("pure", ctypes.c_bool),
        ("keep_split", ctypes.c_bool),
        ("imatrix", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("tensor_types", ctypes.c_void_p),
    ]


# typedef struct llama_logit_bias {
#     llama_token token;
#     float bias;
# } llama_logit_bias;
class llama_logit_bias(ctypes.Structure):
    """Used to store logit bias

    Attributes:
        token (llama_token): token id
        bias (float): bias"""

    if TYPE_CHECKING:
        token: llama_token
        bias: float

    _fields_ = [
        ("token", llama_token),
        ("bias", ctypes.c_float),
    ]


llama_logit_bias_p = ctypes.POINTER(llama_logit_bias)


# typedef struct llama_sampler_chain_params {
#     bool no_perf; // whether to measure performance timings
# } llama_sampler_chain_params;
class llama_sampler_chain_params(ctypes.Structure):
    """Parameters for llama_sampler_chain

    Attributes:
        no_perf (bool): whether to measure performance timings"""

    if TYPE_CHECKING:
        no_perf: bool

    _fields_ = [
        ("no_perf", ctypes.c_bool),
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


# // lora adapter
# struct llama_adapter_lora;
llama_adapter_lora_p = ctypes.c_void_p
llama_adapter_lora_p_ctypes = ctypes.POINTER(ctypes.c_void_p)


# // Helpers for getting default parameters
# LLAMA_API struct llama_model_params          llama_model_default_params(void);
@ctypes_function(
    "llama_model_default_params",
    [],
    llama_model_params,
)
def llama_model_default_params() -> llama_model_params:
    """Get default parameters for llama_model"""
    ...


# LLAMA_API struct llama_context_params        llama_context_default_params(void);
@ctypes_function(
    "llama_context_default_params",
    [],
    llama_context_params,
)
def llama_context_default_params() -> llama_context_params:
    """Get default parameters for llama_context"""
    ...


# LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
@ctypes_function(
    "llama_sampler_chain_default_params",
    [],
    llama_sampler_chain_params,
)
def llama_sampler_chain_default_params() -> llama_sampler_chain_params:
    """Get default parameters for llama_sampler_chain"""
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


# //optional:
# LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);
@ctypes_function(
    "llama_numa_init",
    [ctypes.c_int],
    None,
)
def llama_numa_init(numa: int, /):
    ...


# // Optional: an auto threadpool gets created in ggml if not passed explicitly
# LLAMA_API void llama_attach_threadpool(
#         struct llama_context * ctx,
#            ggml_threadpool_t   threadpool,
#            ggml_threadpool_t   threadpool_batch);
# TODO: Add llama_attach_threadpool


# LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);
# TODO: Add llama_detach_threadpool


# DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
#                          const char * path_model,
#           struct llama_model_params   params),
#         "use llama_model_load_from_file instead");
@ctypes_function(
    "llama_load_model_from_file",
    [ctypes.c_char_p, llama_model_params],
    llama_model_p_ctypes,
)
def llama_load_model_from_file(
    path_model: bytes, params: llama_model_params, /
) -> Optional[llama_model_p]:
    ...


# // Load the model from a file
# // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
# // If the split file name does not follow this pattern, use llama_model_load_from_splits
# LLAMA_API struct llama_model * llama_model_load_from_file(
#                          const char * path_model,
#           struct llama_model_params   params);
@ctypes_function(
    "llama_model_load_from_file",
    [ctypes.c_char_p, llama_model_params],
    llama_model_p_ctypes,
)
def llama_model_load_from_file(
    path_model: bytes, params: llama_model_params, /
) -> Optional[llama_model_p]:
    """Load the model from a file

    If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf

    If the split file name does not follow this pattern, use llama_model_load_from_splits"""
    ...


# // Load the model from multiple splits (support custom naming scheme)
# // The paths must be in the correct order
# LLAMA_API struct llama_model * llama_model_load_from_splits(
#                          const char ** paths,
#                              size_t    n_paths,
#           struct llama_model_params    params);
@ctypes_function(
    "llama_model_load_from_splits",
    [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, llama_model_params],
    llama_model_p_ctypes,
)
def llama_model_load_from_splits(
    paths: List[bytes], n_paths: int, params: llama_model_params, /
) -> Optional[llama_model_p]:
    """Load the model from multiple splits (support custom naming scheme)

    The paths must be in the correct order"""
    ...


# LLAMA_API void llama_free_model(struct llama_model * model);
@ctypes_function(
    "llama_free_model",
    [llama_model_p_ctypes],
    None,
)
def llama_free_model(model: llama_model_p, /):
    ...


# LLAMA_API void llama_model_free(struct llama_model * model);
@ctypes_function(
    "llama_model_free",
    [llama_model_p_ctypes],
    None,
)
def llama_model_free(model: llama_model_p, /):
    ...


# LLAMA_API struct llama_context * llama_init_from_model(
#                  struct llama_model * model,
#         struct llama_context_params   params);
@ctypes_function(
    "llama_init_from_model",
    [llama_model_p_ctypes, llama_context_params],
    llama_context_p_ctypes,
)
def llama_init_from_model(
    model: llama_model_p, params: llama_context_params, /
) -> Optional[llama_context_p]:
    ...


# DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
#                  struct llama_model * model,
#         struct llama_context_params   params),
#         "use llama_init_from_model instead");
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


# LLAMA_API bool llama_supports_rpc        (void);
@ctypes_function("llama_supports_rpc", [], ctypes.c_bool)
def llama_supports_rpc() -> bool:
    ...


# LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
@ctypes_function("llama_n_ctx", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_ctx(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
@ctypes_function("llama_n_batch", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_batch(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
@ctypes_function("llama_n_ubatch", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_ubatch(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);
@ctypes_function("llama_n_seq_max", [llama_context_p_ctypes], ctypes.c_uint32)
def llama_n_seq_max(ctx: llama_context_p, /) -> int:
    ...




# DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
@ctypes_function("llama_n_ctx_train", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_ctx_train(model: llama_model_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
@ctypes_function("llama_n_embd", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_embd(model: llama_model_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
@ctypes_function("llama_n_layer", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_layer(model: llama_model_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");
@ctypes_function("llama_n_head", [llama_model_p_ctypes], ctypes.c_int32)
def llama_n_head(model: llama_model_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");
@ctypes_function("llama_n_vocab", [llama_vocab_p_ctypes], ctypes.c_int32)
def llama_n_vocab(model: llama_vocab_p, /) -> int:
    ...


# LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
@ctypes_function("llama_get_model", [llama_context_p_ctypes], llama_model_p_ctypes)
def llama_get_model(ctx: llama_context_p, /) -> Optional[llama_model_p]:
    ...


# LLAMA_API    struct llama_kv_cache * llama_get_kv_self (      struct llama_context * ctx);
@ctypes_function(
    "llama_get_kv_self",
    [llama_context_p_ctypes],
    llama_kv_cache_p_ctypes,
)
def llama_get_kv_self(ctx: llama_context_p, /) -> Optional[llama_kv_cache_p]:
    """Get the KV cache for self-attention"""
    ...


# LLAMA_API  enum llama_pooling_type    llama_pooling_type(const struct llama_context * ctx);
@ctypes_function("llama_pooling_type", [llama_context_p_ctypes], ctypes.c_int)
def llama_pooling_type(ctx: llama_context_p, /) -> int:
    ...


# LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
@ctypes_function("llama_model_get_vocab", [llama_model_p_ctypes], llama_vocab_p_ctypes)
def llama_model_get_vocab(model: llama_model_p, /) -> Optional[llama_vocab_p]:
    ...


# LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);
@ctypes_function("llama_model_rope_type", [llama_model_p_ctypes], ctypes.c_int)
def llama_model_rope_type(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);
@ctypes_function("llama_model_n_ctx_train", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_n_ctx_train(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
@ctypes_function("llama_model_n_embd", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_n_embd(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_model_n_layer    (const struct llama_model * model);
@ctypes_function("llama_model_n_layer", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_n_layer(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_model_n_head     (const struct llama_model * model);
@ctypes_function("llama_model_n_head", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_n_head(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_model_n_head_kv  (const struct llama_model * model);
@ctypes_function("llama_model_n_head_kv", [llama_model_p_ctypes], ctypes.c_int32)
def llama_model_n_head_kv(model: llama_model_p, /) -> int:
    ...


# // Get the model's RoPE frequency scaling factor
# LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);
@ctypes_function("llama_model_rope_freq_scale_train", [llama_model_p_ctypes], ctypes.c_float)
def llama_model_rope_freq_scale_train(model: llama_model_p, /) -> float:
    ...


# LLAMA_API enum llama_vocab_type   llama_vocab_type  (const struct llama_model * model);
@ctypes_function("llama_vocab_type", [llama_model_p_ctypes], ctypes.c_int)
def llama_vocab_type(model: llama_model_p, /) -> int:
    ...


# LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
@ctypes_function("llama_vocab_n_tokens", [llama_vocab_p_ctypes], ctypes.c_int32)
def llama_vocab_n_tokens(vocab: llama_vocab_p, /) -> int:
    ...


# // Functions to access the model's GGUF metadata scalar values
# // - The functions return the length of the string on success, or -1 on failure
# // - The output string is always null-terminated and cleared on failure
# // - When retrieving a string, an extra byte must be allocated to account for the null terminator
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


# // Get the default chat template. Returns nullptr if not available
# // If name is NULL, returns the default chat template
# LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
@ctypes_function("llama_model_chat_template", [llama_model_p_ctypes, ctypes.c_char_p], ctypes.c_char_p)
def llama_model_chat_template(model: llama_model_p, name: Optional[bytes], /) -> Optional[bytes]:
    """Get the default chat template. Returns None if not available
    If name is None, returns the default chat template"""
    ...


# // Returns the total number of parameters in the model
# LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);
@ctypes_function("llama_model_n_params", [llama_model_p_ctypes], ctypes.c_uint64)
def llama_model_n_params(model: llama_model_p, /) -> int:
    """Returns the total number of parameters in the model"""
    ...


# // Returns true if the model contains an encoder that requires llama_encode() call
# LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);
@ctypes_function("llama_model_has_encoder", [llama_model_p_ctypes], ctypes.c_bool)
def llama_model_has_encoder(model: llama_model_p, /) -> bool:
    """Returns true if the model contains an encoder that requires llama_encode() call"""
    ...


# // Returns true if the model contains a decoder that requires llama_decode() call
# LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);
@ctypes_function("llama_model_has_decoder", [llama_model_p_ctypes], ctypes.c_bool)
def llama_model_has_decoder(model: llama_model_p, /) -> bool:
    """Returns true if the model contains a decoder that requires llama_decode() call"""
    ...


# // For encoder-decoder models, this function returns id of the token that must be provided
# // to the decoder to start generating output sequence. For other models, it returns -1.
# LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);
@ctypes_function(
    "llama_model_decoder_start_token", [llama_model_p_ctypes], ctypes.c_int32
)
def llama_model_decoder_start_token(model: llama_model_p, /) -> int:
    """For encoder-decoder models, this function returns id of the token that must be provided
    to the decoder to start generating output sequence. For other models, it returns -1.
    """
    ...


# // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
# LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);
@ctypes_function("llama_model_is_recurrent", [llama_model_p_ctypes], ctypes.c_bool)
def llama_model_is_recurrent(model: llama_model_p, /) -> bool:
    """Returns true if the model is recurrent (like Mamba, RWKV, etc.)"""
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


# // Load a LoRA adapter from file
# LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
#         struct llama_model * model,
#         const char * path_lora);
@ctypes_function(
    "llama_adapter_lora_init",
    [llama_model_p_ctypes, ctypes.c_char_p],
    llama_adapter_lora_p_ctypes,
)
def llama_adapter_lora_init(
    model: llama_model_p, path_lora: bytes, /
) -> Optional[llama_adapter_lora_p]:
    ...


# // Manually free a LoRA adapter
# // Note: loaded adapters will be free when the associated model is deleted
# LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);
@ctypes_function(
    "llama_adapter_lora_free",
    [llama_adapter_lora_p_ctypes],
    None,
)
def llama_adapter_lora_free(adapter: llama_adapter_lora_p, /):
    ...


# // The following functions operate on a llama_context, hence the naming: llama_verb_...


# // Add a loaded LoRA adapter to given context
# // This will not modify model's weight
# LLAMA_API int32_t llama_set_adapter_lora(
#         struct llama_context * ctx,
#         struct llama_adapter_lora * adapter,
#         float scale);
@ctypes_function(
    "llama_set_adapter_lora",
    [llama_context_p_ctypes, llama_adapter_lora_p_ctypes, ctypes.c_float],
    ctypes.c_int32,
)
def llama_set_adapter_lora(
    ctx: llama_context_p, adapter: llama_adapter_lora_p, scale: float, /
) -> int:
    """Add a loaded LoRA adapter to given context
    This will not modify model's weight"""
    ...


# // Remove a specific LoRA adapter from given context
# // Return -1 if the adapter is not present in the context
# LLAMA_API int32_t llama_rm_adapter_lora(
#         struct llama_context * ctx,
#         struct llama_adapter_lora * adapter);
@ctypes_function(
    "llama_rm_adapter_lora",
    [llama_context_p_ctypes, llama_adapter_lora_p_ctypes],
    ctypes.c_int32,
)
def llama_rm_adapter_lora(
    ctx: llama_context_p, adapter: llama_adapter_lora_p, /
) -> int:
    """Remove a specific LoRA adapter from given context
    Return -1 if the adapter is not present in the context"""
    ...


# // Remove all LoRA adapters from given context
# LLAMA_API void llama_clear_adapter_lora(struct llama_context * ctx);
@ctypes_function(
    "llama_clear_adapter_lora",
    [llama_context_p_ctypes],
    None,
)
def llama_clear_adapter_lora(ctx: llama_context_p, /):
    """Remove all LoRA adapters from given context"""
    ...


# // Apply a loaded control vector to a llama_context, or if data is NULL, clear
# // the currently loaded vector.
# // n_embd should be the size of a single layer's control, and data should point
# // to an n_embd x n_layers buffer starting from layer 1.
# // il_start and il_end are the layer range the vector should apply to (both inclusive)
# // See llama_control_vector_load in common to load a control vector.
# LLAMA_API int32_t llama_apply_adapter_cvec(
#         struct llama_context * ctx,
#                  const float * data,
#                       size_t   len,
#                      int32_t   n_embd,
#                      int32_t   il_start,
#                      int32_t   il_end);
@ctypes_function(
    "llama_apply_adapter_cvec",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    ctypes.c_int32,
)
def llama_apply_adapter_cvec(
    ctx: llama_context_p,
    data: CtypesPointerOrRef[ctypes.c_float],
    len: int,
    n_embd: int,
    il_start: int,
    il_end: int,
    /,
) -> int:
    """Apply a loaded control vector to a llama_context, or if data is NULL, clear
    the currently loaded vector.
    n_embd should be the size of a single layer's control, and data should point
    to an n_embd x n_layers buffer starting from layer 1.
    il_start and il_end are the layer range the vector should apply to (both inclusive)
    See llama_control_vector_load in common to load a control vector."""
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
    """Information associated with an individual cell in the KV cache view.

    Attributes:
        pos (llama_pos): The position for this cell. Takes KV cache shifts into account.
            May be negative if the cell is not populated."""

    if TYPE_CHECKING:
        pos: llama_pos

    _fields_ = [("pos", llama_pos)]


# // An updateable view of the KV cache.
# struct llama_kv_cache_view {
#     // Number of KV cache cells. This will be the same as the context size.
#     int32_t n_cells;

#     // Maximum number of sequences that can exist in a cell. It's not an error
#     // if there are more sequences in a cell than this value, however they will
#     // not be visible in the view cells_sequences.
#     int32_t n_seq_max;

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


#     // The sequences for each cell. There will be n_seq_max items per cell.
#     llama_seq_id * cells_sequences;
# };
class llama_kv_cache_view(ctypes.Structure):
    if TYPE_CHECKING:
        n_cells: int
        n_max_seq: int
        token_count: int
        used_cells: int
        max_contiguous: int
        max_contiguous_idx: int
        cells: CtypesArray[llama_kv_cache_view_cell]
        cells_sequences: CtypesArray[llama_seq_id]

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
# LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);
@ctypes_function(
    "llama_kv_cache_view_init",
    [llama_context_p_ctypes, ctypes.c_int32],
    llama_kv_cache_view,
)
def llama_kv_cache_view_init(
    ctx: llama_context_p, n_seq_max: Union[ctypes.c_int32, int], /
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
# LLAMA_API int32_t llama_kv_self_n_tokens(const struct llama_context * ctx);
@ctypes_function(
    "llama_kv_self_n_tokens", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_kv_self_n_tokens(ctx: llama_context_p, /) -> int:
    """Returns the number of tokens in the KV cache (slow, use only for debug)
    If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    """
    ...


# DEPRECATED(LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx),
#         "use llama_kv_self_n_tokens instead");
@ctypes_function(
    "llama_get_kv_cache_token_count", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_get_kv_cache_token_count(ctx: llama_context_p, /) -> int:
    """Returns the number of tokens in the KV cache (slow, use only for debug)
    If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    """
    ...


# // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
# LLAMA_API int32_t llama_kv_self_used_cells(const struct llama_context * ctx);
@ctypes_function(
    "llama_kv_self_used_cells", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_kv_self_used_cells(ctx: llama_context_p, /) -> int:
    """Returns the number of used KV cells (i.e. have at least one sequence assigned to them)"""
    ...


# DEPRECATED(LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx),
#         "use llama_kv_self_used_cells instead");
@ctypes_function(
    "llama_get_kv_cache_used_cells", [llama_context_p_ctypes], ctypes.c_int32
)
def llama_get_kv_cache_used_cells(ctx: llama_context_p, /) -> int:
    """Returns the number of used KV cells (i.e. have at least one sequence assigned to them)"""
    ...


# // Clear the KV cache - both cell info is erased and KV data is zeroed
# LLAMA_API void llama_kv_self_clear(
#         struct llama_context * ctx);
@ctypes_function(
    "llama_kv_self_clear", [llama_context_p_ctypes], None
)
def llama_kv_self_clear(ctx: llama_context_p, /):
    """Clear the KV cache - both cell info is erased and KV data is zeroed"""
    ...

# NOTE: Deprecated
@ctypes_function("llama_kv_self_clear", [llama_context_p_ctypes], None)
def llama_kv_cache_clear(ctx: llama_context_p, /):
    """Clear the KV cache"""
    ...


# // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
# // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
# // seq_id < 0 : match any sequence
# // p0 < 0     : [0,  p1]
# // p1 < 0     : [p0, inf)
# LLAMA_API bool llama_kv_cache_seq_rm(
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
    ctypes.c_bool,
)
def llama_kv_cache_seq_rm(
    ctx: llama_context_p,
    seq_id: Union[llama_seq_id, int],
    p0: Union[llama_pos, int],
    p1: Union[llama_pos, int],
    /,
) -> bool:
    """Removes all tokens that belong to the specified sequence and have positions in [p0, p1)

    Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails

    seq_id < 0 : match any sequence
    p0 < 0     : [0,  p1]
    p1 < 0     : [p0, inf)"""
    ...


# // Copy all tokens that belong to the specified sequence to another sequence
# // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
# // p0 < 0 : [0,  p1]
# // p1 < 0 : [p0, inf)
# LLAMA_API void llama_kv_self_seq_cp(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id_src,
#                 llama_seq_id   seq_id_dst,
#                    llama_pos   p0,
#                    llama_pos   p1);
@ctypes_function(
    "llama_kv_self_seq_cp",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_seq_id,
        llama_pos,
        llama_pos,
    ],
    None,
)
def llama_kv_self_seq_cp(
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


# NOTE: Deprecated
@ctypes_function(
    "llama_kv_self_seq_cp",
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
# LLAMA_API void llama_kv_self_seq_keep(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id);
@ctypes_function(
    "llama_kv_self_seq_keep", [llama_context_p_ctypes, llama_seq_id], None
)
def llama_kv_self_seq_keep(ctx: llama_context_p, seq_id: Union[llama_seq_id, int], /):
    """Removes all tokens that do not belong to the specified sequence"""
    ...


# NOTE: Deprecated
@ctypes_function(
    "llama_kv_self_seq_keep", [llama_context_p_ctypes, llama_seq_id], None
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
    "llama_kv_self_seq_add",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_pos,
        llama_pos,
        llama_pos,
    ],
    None,
)
def llama_kv_self_seq_add(
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


# // NOTE: Deprecated
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
    "llama_kv_self_seq_add",
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
    "llama_kv_self_seq_div",
    [
        llama_context_p_ctypes,
        llama_seq_id,
        llama_pos,
        llama_pos,
        ctypes.c_int,
    ],
    None,
)
def llama_kv_self_seq_div(
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


# // NOTE: Deprecated
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
    "llama_kv_self_seq_div",
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


# // Returns the largest position present in the KV cache for the specified sequence
# LLAMA_API llama_pos llama_kv_self_seq_pos_max(
#         struct llama_context * ctx,
#                  llama_seq_id   seq_id);
@ctypes_function(
    "llama_kv_self_seq_pos_max", [llama_context_p_ctypes, llama_seq_id], llama_pos
)
def llama_kv_self_seq_pos_max(
    ctx: llama_context_p, seq_id: Union[llama_seq_id, int], /
) -> int:
    """Returns the largest position present in the KV cache for the specified sequence"""
    ...


# // Defragment the KV cache
# // This will be applied:
# //   - lazily on next llama_decode()
# //   - explicitly with llama_kv_self_update()
# LLAMA_API void llama_kv_self_defrag(struct llama_context * ctx);
@ctypes_function("llama_kv_self_defrag", [llama_context_p_ctypes], None)
def llama_kv_self_defrag(ctx: llama_context_p, /):
    """Defragment the KV cache
    This will be applied:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()"""
    ...


# NOTE: Deprecated
# // Defragment the KV cache
# // This will be applied:
# //   - lazily on next llama_decode()
# //   - explicitly with llama_kv_self_update()
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
@ctypes_function("llama_kv_self_update", [llama_context_p_ctypes], None)
def llama_kv_self_update(ctx: llama_context_p, /):
    """Apply the KV cache updates (such as K-shifts, defragmentation, etc.)"""
    ...

# // NOTE: Deprecated
# // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
# LLAMA_API void llama_kv_cache_update(struct llama_context * ctx);
@ctypes_function("llama_kv_self_update", [llama_context_p_ctypes], None)
def llama_kv_cache_update(ctx: llama_context_p, /):
    """Apply the KV cache updates (such as K-shifts, defragmentation, etc.)"""
    ...


# // Check if the context supports KV cache shifting
# LLAMA_API bool llama_kv_cache_can_shift(struct llama_context * ctx);
@ctypes_function("llama_kv_self_can_shift", [llama_context_p_ctypes], ctypes.c_bool)
def llama_kv_self_can_shift(ctx: llama_context_p, /) -> bool:
    """Check if the context supports KV cache shifting"""
    ...


# // NOTE: Deprecated
# // Check if the context supports KV cache shifting
# LLAMA_API bool llama_kv_cache_can_shift(struct llama_context * ctx);
@ctypes_function("llama_kv_self_can_shift", [llama_context_p_ctypes], ctypes.c_bool)
def llama_kv_cache_can_shift(ctx: llama_context_p, /) -> bool:
    """Check if the context supports KV cache shifting"""
    ...


# //
# // State / sessions
# //


# // Returns the *actual* size in bytes of the state
# // (logits, embedding and kv_cache)
# // Only use when saving the state, not when restoring it, otherwise the size may be too small.
# LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
@ctypes_function("llama_state_get_size", [llama_context_p_ctypes], ctypes.c_size_t)
def llama_state_get_size(ctx: llama_context_p, /) -> int:
    """Returns the *actual* size in bytes of the state (rng, logits, embedding and kv_cache) - will often be smaller after compacting tokens"""
    ...


# LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
#     "use llama_state_get_size instead");
@ctypes_function("llama_get_state_size", [llama_context_p_ctypes], ctypes.c_size_t)
def llama_get_state_size(ctx: llama_context_p, /) -> int:
    """Returns the maximum size in bytes of the state (rng, logits, embedding
    and kv_cache) - will often be smaller after compacting tokens"""
    ...


# // Copies the state to the specified destination address.
# // Destination needs to have allocated enough memory.
# // Returns the number of bytes copied
# LLAMA_API size_t llama_state_get_data(
#         struct llama_context * ctx,
#                      uint8_t * dst,
#                       size_t   size);
@ctypes_function(
    "llama_state_get_data",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ],
    ctypes.c_size_t,
)
def llama_state_get_data(
    ctx: llama_context_p,
    dst: CtypesArray[ctypes.c_uint8],
    size: Union[ctypes.c_size_t, int],
    /,
) -> int:
    """Copies the state to the specified destination address.
    Destination needs to have allocated enough memory.
    Returns the number of bytes copied"""
    ...


# LLAMA_API DEPRECATED(size_t llama_copy_state_data(
#         struct llama_context * ctx,
#                      uint8_t * dst),
#     "use llama_state_get_data instead");
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
# LLAMA_API size_t llama_state_set_data(
#         struct llama_context * ctx,
#                const uint8_t * src,
#                       size_t   size);
@ctypes_function(
    "llama_state_set_data",
    [llama_context_p_ctypes, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t],
    ctypes.c_size_t,
)
def llama_state_set_data(
    ctx: llama_context_p,
    src: CtypesArray[ctypes.c_uint8],
    size: Union[ctypes.c_size_t, int],
    /,
) -> int:
    """Set the state reading from the specified address
    Returns the number of bytes read"""
    ...


# LLAMA_API DEPRECATED(size_t llama_set_state_data(
#         struct llama_context * ctx,
#                const uint8_t * src),
#     "use llama_state_set_data instead");
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
# LLAMA_API bool llama_state_load_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#                  llama_token * tokens_out,
#                       size_t   n_token_capacity,
#                       size_t * n_token_count_out);
@ctypes_function(
    "llama_state_load_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_token_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    ctypes.c_bool,
)
def llama_state_load_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens_out: CtypesArray[llama_token],
    n_token_capacity: Union[ctypes.c_size_t, int],
    n_token_count_out: CtypesPointerOrRef[ctypes.c_size_t],
    /,
) -> bool:
    ...


# LLAMA_API DEPRECATED(bool llama_load_session_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#                  llama_token * tokens_out,
#                       size_t   n_token_capacity,
#                       size_t * n_token_count_out),
#     "use llama_state_load_file instead");
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


# LLAMA_API bool llama_state_save_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#            const llama_token * tokens,
#                       size_t   n_token_count);
@ctypes_function(
    "llama_state_save_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_token_p,
        ctypes.c_size_t,
    ],
    ctypes.c_bool,
)
def llama_state_save_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens: CtypesArray[llama_token],
    n_token_count: Union[ctypes.c_size_t, int],
    /,
) -> bool:
    ...


# LLAMA_API DEPRECATED(bool llama_save_session_file(
#         struct llama_context * ctx,
#                   const char * path_session,
#            const llama_token * tokens,
#                       size_t   n_token_count),
#     "use llama_state_save_file instead");
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


# // Get the exact size needed to copy the KV cache of a single sequence
# LLAMA_API size_t llama_state_seq_get_size(
#         struct llama_context * ctx,
#                 llama_seq_id   seq_id);
@ctypes_function(
    "llama_state_seq_get_size",
    [llama_context_p_ctypes, llama_seq_id],
    ctypes.c_size_t,
)
def llama_state_seq_get_size(ctx: llama_context_p, seq_id: llama_seq_id, /) -> int:
    """Get the exact size needed to copy the KV cache of a single sequence"""
    ...


# // Copy the KV cache of a single sequence into the specified buffer
# LLAMA_API size_t llama_state_seq_get_data(
#         struct llama_context * ctx,
#                      uint8_t * dst,
#                       size_t   size,
#                 llama_seq_id   seq_id);
@ctypes_function(
    "llama_state_seq_get_data",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        llama_seq_id,
    ],
    ctypes.c_size_t,
)
def llama_state_seq_get_data(
    ctx: llama_context_p,
    dst: CtypesArray[ctypes.c_uint8],
    size: Union[ctypes.c_size_t, int],
    seq_id: llama_seq_id,
    /,
) -> int:
    """Copy the KV cache of a single sequence into the specified buffer"""
    ...


# // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
# // Returns:
# //  - Positive: Ok
# //  - Zero: Failed to load
# LLAMA_API size_t llama_state_seq_set_data(
#         struct llama_context * ctx,
#                const uint8_t * src,
#                       size_t   size,
#                 llama_seq_id   dest_seq_id);
@ctypes_function(
    "llama_state_seq_set_data",
    [
        llama_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        llama_seq_id,
    ],
    ctypes.c_size_t,
)
def llama_state_seq_set_data(
    ctx: llama_context_p,
    src: CtypesArray[ctypes.c_uint8],
    size: Union[ctypes.c_size_t, int],
    dest_seq_id: llama_seq_id,
    /,
) -> int:
    """Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence"""
    ...


# LLAMA_API size_t llama_state_seq_save_file(
#         struct llama_context * ctx,
#                   const char * filepath,
#                 llama_seq_id   seq_id,
#            const llama_token * tokens,
#                       size_t   n_token_count);
@ctypes_function(
    "llama_state_seq_save_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_seq_id,
        llama_token_p,
        ctypes.c_size_t,
    ],
    ctypes.c_size_t,
)
def llama_state_seq_save_file(
    ctx: llama_context_p,
    filepath: bytes,
    seq_id: llama_seq_id,
    tokens: CtypesArray[llama_token],
    n_token_count: Union[ctypes.c_size_t, int],
    /,
) -> int:
    ...


# LLAMA_API size_t llama_state_seq_load_file(
#         struct llama_context * ctx,
#                   const char * filepath,
#                 llama_seq_id   dest_seq_id,
#                  llama_token * tokens_out,
#                       size_t   n_token_capacity,
#                       size_t * n_token_count_out);
@ctypes_function(
    "llama_state_seq_load_file",
    [
        llama_context_p_ctypes,
        ctypes.c_char_p,
        llama_seq_id,
        llama_token_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    ctypes.c_size_t,
)
def llama_state_seq_load_file(
    ctx: llama_context_p,
    filepath: bytes,
    dest_seq_id: llama_seq_id,
    tokens_out: CtypesArray[llama_token],
    n_token_capacity: Union[ctypes.c_size_t, int],
    n_token_count_out: CtypesPointerOrRef[ctypes.c_size_t],
    /,
) -> int:
    ...


# //
# // Decoding
# //


# // Return batch for single sequence of tokens
# // The sequence ID will be fixed to 0
# // The position of the tokens will be tracked automatically by llama_decode
# //
# // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
# //
# LLAMA_API struct llama_batch llama_batch_get_one(
#               llama_token * tokens,
#                   int32_t   n_tokens);
@ctypes_function(
    "llama_batch_get_one",
    [
        llama_token_p,
        ctypes.c_int32,
    ],
    llama_batch,
)
def llama_batch_get_one(
    tokens: CtypesArray[llama_token],
    n_tokens: Union[ctypes.c_int, int],
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


# // Processes a batch of tokens with the ecoder part of the encoder-decoder model.
# // Stores the encoder output internally for later use by the decoder cross-attention layers.
# //   0 - success
# // < 0 - error
# LLAMA_API int32_t llama_encode(
#         struct llama_context * ctx,
#           struct llama_batch   batch);
@ctypes_function("llama_encode", [llama_context_p_ctypes, llama_batch], ctypes.c_int32)
def llama_encode(ctx: llama_context_p, batch: llama_batch, /) -> int:
    """Processes a batch of tokens with the ecoder part of the encoder-decoder model.
    Stores the encoder output internally for later use by the decoder cross-attention layers.
    0 - success
    < 0 - error"""
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
# LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);
@ctypes_function(
    "llama_set_n_threads",
    [
        llama_context_p_ctypes,
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    None,
)
def llama_set_n_threads(
    ctx: llama_context_p,
    n_threads: Union[ctypes.c_int32, int],
    n_threads_batch: Union[ctypes.c_int32, int],
    /,
):
    """Set the number of threads used for decoding
    n_threads is the number of threads used for generation (single token)
    n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    """
    ...


# // Get the number of threads used for generation of a single token.
# LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);
@ctypes_function("llama_n_threads", [llama_context_p_ctypes], ctypes.c_int32)
def llama_n_threads(ctx: llama_context_p, /) -> int:
    """Get the number of threads used for generation of a single token"""
    ...


# // Get the number of threads used for prompt and batch processing (multiple token).
# LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);
@ctypes_function("llama_n_threads_batch", [llama_context_p_ctypes], ctypes.c_int32)
def llama_n_threads_batch(ctx: llama_context_p, /) -> int:
    """Get the number of threads used for prompt and batch processing (multiple token)"""
    ...


# // Set whether the model is in embeddings mode or not
# // If true, embeddings will be returned but logits will not
# LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
@ctypes_function("llama_set_embeddings", [llama_context_p_ctypes, ctypes.c_bool], None)
def llama_set_embeddings(ctx: llama_context_p, embeddings: bool, /):
    """Set whether the model is in embeddings model or not
    If true, embeddings will be returned but logits will not"""
    ...


# // Set whether to use causal attention or not
# // If set to true, the model will only attend to the past tokens
# LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);
@ctypes_function("llama_set_causal_attn", [llama_context_p_ctypes, ctypes.c_bool], None)
def llama_set_causal_attn(ctx: llama_context_p, causal_attn: bool, /):
    """Set whether to use causal attention or not
    If set to true, the model will only attend to the past tokens"""
    ...


# // Set whether the model is in warmup mode or not
# // If true, all model tensors are activated during llama_decode() to load and cache their weights.
# LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);
@ctypes_function("llama_set_warmup", [llama_context_p_ctypes, ctypes.c_bool], None)
def llama_set_warmup(ctx: llama_context_p, warmup: bool, /):
    """Set whether the model is in warmup mode or not
    If true, all model tensors are activated during llama_decode() to load and cache their weights."""
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


# // Wait until all computations are finished
# // This is automatically done when using one of the functions below to obtain the computation results
# // and is not necessary to call it explicitly in most cases
# LLAMA_API void llama_synchronize(struct llama_context * ctx);
@ctypes_function("llama_synchronize", [llama_context_p_ctypes], None)
def llama_synchronize(ctx: llama_context_p, /):
    """Wait until all computations are finished
    This is automatically done when using one of the functions below to obtain the computation results
    and is not necessary to call it explicitly in most cases"""
    ...


# // Token logits obtained from the last call to llama_decode()
# // The logits for which llama_batch.logits[i] != 0 are stored contiguously
# // in the order they have appeared in the batch.
# // Rows: number of tokens for which llama_batch.logits[i] != 0
# // Cols: n_vocab
# LLAMA_API float * llama_get_logits(struct llama_context * ctx);
@ctypes_function(
    "llama_get_logits", [llama_context_p_ctypes], ctypes.POINTER(ctypes.c_float)
)
def llama_get_logits(ctx: llama_context_p, /) -> CtypesArray[ctypes.c_float]:
    """Token logits obtained from the last call to llama_decode()
    The logits for which llama_batch.logits[i] != 0 are stored contiguously
    in the order they have appeared in the batch.
    Rows: number of tokens for which llama_batch.logits[i] != 0
    Cols: n_vocab

    Returns:
        Pointer to the logits buffer of shape (n_tokens, n_vocab)"""
    ...


# // Logits for the ith token. For positive indices, Equivalent to:
# // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
# // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
# // returns NULL for invalid ids.
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


# // Get all output token embeddings.
# // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
# // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
# // in the order they have appeared in the batch.
# // shape: [n_outputs*n_embd]
# // Otherwise, returns NULL.
# LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
@ctypes_function(
    "llama_get_embeddings", [llama_context_p_ctypes], ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings(ctx: llama_context_p, /) -> CtypesArray[ctypes.c_float]:
    """Get the embeddings for the input
    shape: [n_embd] (1-dimensional)"""
    ...


# // Get the embeddings for the ith token. For positive indices, Equivalent to:
# // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
# // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
# // shape: [n_embd] (1-dimensional)
# // returns NULL for invalid ids.
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


# // Get the embeddings for a sequence id
# // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
# // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
# // otherwise: float[n_embd] (1-dimensional)
# LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);
@ctypes_function(
    "llama_get_embeddings_seq",
    [llama_context_p_ctypes, llama_seq_id],
    ctypes.POINTER(ctypes.c_float),
)
def llama_get_embeddings_seq(
    ctx: llama_context_p, seq_id: Union[llama_seq_id, int], /
) -> CtypesArray[ctypes.c_float]:
    """Get the embeddings for a sequence id
    Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    shape: [n_embd] (1-dimensional)"""
    ...


# //
# // Vocab
# //


# LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
@ctypes_function(
    "llama_vocab_get_text", [llama_vocab_p_ctypes, llama_token], ctypes.c_char_p
)
def llama_vocab_get_text(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> bytes:
    ...


# LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);
@ctypes_function(
    "llama_vocab_get_score", [llama_vocab_p_ctypes, llama_token], ctypes.c_float
)
def llama_vocab_get_score(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> float:
    ...


# LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);
@ctypes_function(
    "llama_vocab_get_attr", [llama_vocab_p_ctypes, llama_token], ctypes.c_int
)
def llama_vocab_get_attr(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> int:
    ...


# // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
# LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
@ctypes_function(
    "llama_vocab_is_eog", [llama_vocab_p_ctypes, llama_token], ctypes.c_bool
)
def llama_vocab_is_eog(vocab: llama_vocab_p, token: Union[llama_token, int], /) -> bool:
    """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
    ...


# // Identify if Token Id is a control token or a render-able token
# LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);
@ctypes_function(
    "llama_vocab_is_control", [llama_vocab_p_ctypes, llama_token], ctypes.c_bool
)
def llama_vocab_is_control(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> bool:
    """Identify if Token Id is a control token or a render-able token"""
    ...


# // Special tokens


# LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
@ctypes_function("llama_vocab_bos", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_bos(vocab: llama_vocab_p, /) -> llama_token:
    """beginning-of-sentence"""
    ...


# LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
@ctypes_function("llama_vocab_eos", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_eos(vocab: llama_vocab_p, /) -> llama_token:
    """end-of-sentence"""
    ...


# LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
@ctypes_function("llama_vocab_eot", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_eot(vocab: llama_vocab_p, /) -> llama_token:
    """end-of-turn"""
    ...


# LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
@ctypes_function("llama_vocab_sep", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_sep(vocab: llama_vocab_p, /) -> llama_token:
    """sentence separator"""
    ...


# LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
@ctypes_function("llama_vocab_nl", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_nl(vocab: llama_vocab_p, /) -> llama_token:
    """next-line"""
    ...


# LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
@ctypes_function("llama_vocab_pad", [llama_vocab_p_ctypes], llama_token)
def llama_vocab_pad(vocab: llama_vocab_p, /) -> llama_token:
    """padding"""
    ...

# LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_get_add_bos",
    [llama_vocab_p_ctypes],
    ctypes.c_bool,
)
def llama_vocab_get_add_bos(vocab: llama_vocab_p, /) -> bool:
    ...


# LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_get_add_eos",
    [llama_vocab_p_ctypes],
    ctypes.c_bool,
)
def llama_vocab_get_add_eos(vocab: llama_vocab_p, /) -> bool:
    ...


# LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_pre",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_pre(vocab: llama_vocab_p, /) -> llama_token:
    ...


# LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_suf",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_suf(vocab: llama_vocab_p, /) -> llama_token:
    ...


# LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_mid",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_mid(vocab: llama_vocab_p, /) -> llama_token:
    ...


# LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_pad",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_pad(vocab: llama_vocab_p, /) -> llama_token:
    ...


# LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_rep",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_rep(vocab: llama_vocab_p, /) -> llama_token:
    ...


# LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_vocab_fim_sep",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_fim_sep(vocab: llama_vocab_p, /) -> llama_token:
    ...



# DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
@ctypes_function(
    "llama_token_get_text",
    [llama_vocab_p_ctypes, llama_token],
    ctypes.c_char_p,
)
def llama_token_get_text(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> bytes:
    ...


# DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
@ctypes_function(
    "llama_token_get_score",
    [llama_vocab_p_ctypes, llama_token],
    ctypes.c_float,
)
def llama_token_get_score(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> float:
    ...

# DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
@ctypes_function(
    "llama_token_get_attr",
    [llama_vocab_p_ctypes, llama_token],
    ctypes.c_int,
)
def llama_token_get_attr(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> int:
    ...

# DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
@ctypes_function(
    "llama_token_is_eog",
    [llama_vocab_p_ctypes, llama_token],
    ctypes.c_bool,
)
def llama_token_is_eog(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> bool:
    ...

# DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
@ctypes_function(
    "llama_token_is_control",
    [llama_vocab_p_ctypes, llama_token],
    ctypes.c_bool,
)
def llama_token_is_control(
    vocab: llama_vocab_p, token: Union[llama_token, int], /
) -> bool:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
@ctypes_function(
    "llama_token_bos",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_bos(vocab: llama_vocab_p, /) -> int:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
@ctypes_function(
    "llama_token_eos",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_eos(vocab: llama_vocab_p, /) -> int:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
@ctypes_function(
    "llama_token_eot",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_eot(vocab: llama_vocab_p, /) -> int:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
@ctypes_function(
    "llama_token_cls",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_cls(vocab: llama_vocab_p, /) -> int:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
@ctypes_function(
    "llama_token_sep",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_sep(vocab: llama_vocab_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
@ctypes_function(
    "llama_token_nl",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_nl(vocab: llama_vocab_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
@ctypes_function(
    "llama_token_pad",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_pad(vocab: llama_vocab_p, /) -> int:
    ...


# DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
@ctypes_function(
    "llama_add_bos_token",
    [llama_vocab_p_ctypes],
    ctypes.c_bool,
)
def llama_add_bos_token(vocab: llama_vocab_p, /) -> bool:
    ...

# DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
@ctypes_function(
    "llama_add_eos_token",
    [llama_vocab_p_ctypes],
    ctypes.c_bool,
)
def llama_add_eos_token(vocab: llama_vocab_p, /) -> bool:
    ...


# DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
@ctypes_function(
    "llama_token_fim_pre",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_pre(vocab: llama_vocab_p, /) -> llama_token:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
@ctypes_function(
    "llama_token_fim_suf",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_suf(vocab: llama_vocab_p, /) -> llama_token:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
@ctypes_function(
    "llama_token_fim_mid",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_mid(vocab: llama_vocab_p, /) -> llama_token:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
@ctypes_function(
    "llama_token_fim_pad",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_pad(vocab: llama_vocab_p, /) -> llama_token:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
@ctypes_function(
    "llama_token_fim_rep",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_rep(vocab: llama_vocab_p, /) -> llama_token:
    ...

# DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");
@ctypes_function(
    "llama_token_fim_sep",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_token_fim_sep(vocab: llama_vocab_p, /) -> llama_token:
    ...

# // CLS is equivalent to BOS
# DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
#         "use llama_vocab_bos instead");
@ctypes_function(
    "llama_vocab_cls",
    [llama_vocab_p_ctypes],
    llama_token,
)
def llama_vocab_cls(vocab: llama_vocab_p, /) -> llama_token:
    ...


# //
# // Tokenization
# //
# // The API is thread-safe.
# //


# /// @details Convert the provided text into tokens.
# /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
# /// @return Returns the number of tokens on success, no more than n_tokens_max
# /// @return Returns a negative number on failure - the number of tokens that would have been returned
# /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
# /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
# ///                      as plaintext. Does not insert a leading space.
# LLAMA_API int32_t llama_tokenize(
#     const struct llama_vocab * vocab,
#                   const char * text,
#                      int32_t   text_len,
#                  llama_token * tokens,
#                      int32_t   n_tokens_max,
#                         bool   add_special,
#                         bool   parse_special);
@ctypes_function(
    "llama_tokenize",
    [
        llama_vocab_p_ctypes,
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
    vocab: llama_vocab_p,
    text: bytes,
    text_len: Union[ctypes.c_int, int],
    tokens: CtypesArray[llama_token],
    n_tokens_max: Union[ctypes.c_int, int],
    add_special: Union[ctypes.c_bool, bool],
    parse_special: Union[ctypes.c_bool, bool],
    /,
) -> int:
    """Convert the provided text into tokens.

    Args:
        vocab: The vocabulary to use for tokenization.
        text: The text to tokenize.
        text_len: The length of the text.
        tokens: The tokens pointer must be large enough to hold the resulting tokens.
        n_max_tokens: The maximum number of tokens to return.
        add_special: Allow adding special tokenns if the model is configured to do so.
        parse_special: Allow parsing special tokens.

    Returns:
        Returns the number of tokens on success, no more than n_tokens_max
        Returns a negative number on failure - the number of tokens that would have been returned
    """
    ...


# // Token Id -> Piece.
# // Uses the vocabulary in the provided context.
# // Does not write null terminator to the buffer.
# // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
# // @param special If true, special tokens are rendered in the output.
# LLAMA_API int32_t llama_token_to_piece(
#           const struct llama_vocab * vocab,
#                        llama_token   token,
#                               char * buf,
#                            int32_t   length,
#                            int32_t   lstrip,
#                               bool   special);
@ctypes_function(
    "llama_token_to_piece",
    [
        llama_vocab_p_ctypes,
        llama_token,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ],
    ctypes.c_int32,
)
def llama_token_to_piece(
    vocab: llama_vocab_p,
    token: Union[llama_token, int],
    buf: Union[ctypes.c_char_p, bytes, CtypesArray[ctypes.c_char]],
    length: Union[ctypes.c_int, int],
    lstrip: Union[ctypes.c_int, int],
    special: Union[ctypes.c_bool, bool],
    /,
) -> int:
    """Token Id -> Piece.
    Uses the vocabulary in the provided context.
    Does not write null terminator to the buffer.
    User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.

    Args:
        vocab: The vocabulary to use for tokenization.
        token: The token to convert.
        buf: The buffer to write the token to.
        length: The length of the buffer.
        lstrip: The number of leading spaces to skip.
        special: If true, special tokens are rendered in the output."""
    ...


# # // check if token0 is contained as a prefix in token1
# # LLAMA_API bool llama_token_is_prefix(
# #           const struct llama_model * model,
# #                        llama_token   token0,
# #                        llama_token   token1);
# @ctypes_function(
#     "llama_token_is_prefix",
#     [llama_model_p_ctypes, llama_token, llama_token],
#     ctypes.c_bool,
# )
# def llama_token_is_prefix(
#     model: llama_model_p, token0: Union[llama_token, int], token1: Union[llama_token, int], /
# ) -> bool:
#     """Check if token0 is contained as a prefix in token1"""
#     ...


# /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
# /// @param text The char pointer must be large enough to hold the resulting text.
# /// @return Returns the number of chars/bytes on success, no more than text_len_max.
# /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
# /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
# /// @param unparse_special If true, special tokens are rendered in the output.
# LLAMA_API int32_t llama_detokenize(
#     const struct llama_model * model,
#            const llama_token * tokens,
#                      int32_t   n_tokens,
#                         char * text,
#                      int32_t   text_len_max,
#                         bool   remove_special,
#                         bool   unparse_special);
@ctypes_function(
    "llama_detokenize",
    [
        llama_model_p_ctypes,
        ctypes.POINTER(llama_token),
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ],
    ctypes.c_int32,
)
def llama_detokenize(
    model: llama_model_p,
    tokens: CtypesArray[llama_token],
    n_tokens: Union[ctypes.c_int, int],
    text: bytes,
    text_len_max: Union[ctypes.c_int, int],
    remove_special: Union[ctypes.c_bool, bool],
    unparse_special: Union[ctypes.c_bool, bool],
    /,
) -> int:
    """Convert the provided tokens into text (inverse of llama_tokenize()).

    Args:
        model: The model to use for tokenization.
        tokens: The tokens to convert.
        n_tokens: The number of tokens.
        text: The buffer to write the text to.
        text_len_max: The length of the buffer.
        remove_special: Allow to remove BOS and EOS tokens if model is configured to do so.
        unparse_special: If true, special tokens are rendered in the output."""
    ...


# //
# // Chat templates
# //


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
#                         const char * tmpl,
#    const struct llama_chat_message * chat,
#                             size_t   n_msg,
#                               bool   add_ass,
#                               char * buf,
#                            int32_t   length);
@ctypes_function(
    "llama_chat_apply_template",
    [
        ctypes.c_char_p,  # tmpl
        ctypes.POINTER(llama_chat_message),  # chat
        ctypes.c_size_t,  # n_msg
        ctypes.c_bool,    # add_ass (added)
        ctypes.c_char_p,  # buf
        ctypes.c_int32,   # length
    ],
    ctypes.c_int32,
)
def llama_chat_apply_template(
    tmpl: bytes,
    chat: CtypesArray[llama_chat_message],
    n_msg: int,
    add_ass: bool,  # Added parameter
    buf: bytes,
    length: int,
    /,
) -> int:
    """Apply chat template.

    Args:
        tmpl: Template to use. If None, uses model's default
        chat: Array of chat messages
        n_msg: Number of messages
        add_ass: Whether to end prompt with assistant token
        buf: Output buffer
        length: Buffer length

    Returns:
        Number of bytes written, or needed if buffer too small
    """
    ...


# // Get list of built-in chat templates
# LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);
@ctypes_function(
    "llama_chat_builtin_templates",
    [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_chat_builtin_templates(
    output: CtypesArray[bytes],
    len: Union[ctypes.c_size_t, int],
    /,
) -> int:
    """Get list of built-in chat templates.

    Args:
        output: Output buffer to store template names.
        len: Length of the output buffer.

    Returns:
        Number of templates available.
        Returns a negative number on error.
    """
    ...


# //
# // Sampling API
# //
# // Sample usage:
# //
# //    // prepare the sampling chain at the start
# //    auto sparams = llama_sampler_chain_default_params();
# //
# //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
# //
# //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
# //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
# //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
# //
# //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
# //    // this sampler will be responsible to select the actual token
# //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
# //
# //    ...
# //
# //    // decoding loop:
# //    while (...) {
# //        ...
# //
# //        llama_decode(ctx, batch);
# //
# //        // sample from the logits of the last token in the batch
# //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
# //
# //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
# //        llama_sampler_accept(smpl, id);
# //        ...
# //    }
# //
# //    llama_sampler_free(smpl);
# //
# // TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
# //

# typedef void * llama_sampler_context_t;
llama_sampler_context_t = ctypes.c_void_p


# // user code can implement the interface below in order to create custom llama_sampler
# struct llama_sampler_i {
#     const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
#     void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
#     void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
#     void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
#     struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
#     void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
#
#     // TODO: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
#     //void (*apply_ggml) (struct llama_sampler * smpl, ...);
# };
class llama_sampler_i(ctypes.Structure):
    ...


# struct llama_sampler {
#     const struct llama_sampler_i  * iface;
#     llama_sampler_context_t         ctx;
# };
class llama_sampler(ctypes.Structure):
    _fields_ = [
        ("iface", ctypes.POINTER(llama_sampler_i)),
        ("ctx", llama_sampler_context_t),
    ]


if TYPE_CHECKING:
    llama_sampler_p = CtypesPointer[llama_sampler]

llama_sampler_p_ctypes = ctypes.POINTER(llama_sampler)

llama_sampler_i_name = ctypes.CFUNCTYPE(ctypes.c_char_p, llama_sampler_p_ctypes)
llama_sampler_i_accept = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes, llama_token)
llama_sampler_i_apply = ctypes.CFUNCTYPE(
    None, llama_sampler_p_ctypes, llama_token_data_array_p
)
llama_sampler_i_reset = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes)
llama_sampler_i_clone = ctypes.CFUNCTYPE(llama_sampler_p_ctypes, llama_sampler_p_ctypes)
llama_sampler_i_free = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes)

llama_sampler_i._fields_ = [
    ("name", llama_sampler_i_name),
    ("accept", llama_sampler_i_accept),
    ("apply", llama_sampler_i_apply),
    ("reset", llama_sampler_i_reset),
    ("clone", llama_sampler_i_clone),
    ("free", llama_sampler_i_free),
]


# // mirror of llama_sampler_i:
# LLAMA_API struct llama_sampler * llama_sampler_init  (const struct llama_sampler_i * iface, llama_sampler_context_t ctx);
@ctypes_function(
    "llama_sampler_init",
    [ctypes.POINTER(llama_sampler_i), llama_sampler_context_t],
    llama_sampler_p_ctypes,
)
def llama_sampler_init(
    iface: ctypes.POINTER(llama_sampler_i), ctx: llama_sampler_context_t, /
) -> llama_sampler_p:
    ...


# LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_name",
    [llama_sampler_p_ctypes],
    ctypes.c_char_p,
)
def llama_sampler_name(smpl: llama_sampler_p, /) -> bytes:
    ...


# LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
@ctypes_function(
    "llama_sampler_accept",
    [llama_sampler_p_ctypes, llama_token],
    None,
)
def llama_sampler_accept(smpl: llama_sampler_p, token: Union[llama_token, int], /):
    ...


# LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
@ctypes_function(
    "llama_sampler_apply",
    [llama_sampler_p_ctypes, llama_token_data_array_p],
    None,
)
def llama_sampler_apply(
    smpl: llama_sampler_p, cur_p: CtypesArray[llama_token_data_array], /
):
    ...


# LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_reset",
    [llama_sampler_p_ctypes],
    None,
)
def llama_sampler_reset(smpl: llama_sampler_p, /):
    ...


# LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_clone",
    [llama_sampler_p_ctypes],
    llama_sampler_p_ctypes,
)
def llama_sampler_clone(smpl: llama_sampler_p, /) -> llama_sampler_p:
    ...


# // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
# LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_free",
    [llama_sampler_p_ctypes],
    None,
)
def llama_sampler_free(smpl: llama_sampler_p, /):
    ...


# // llama_sampler_chain
# // a type of llama_sampler that can chain multiple samplers one after another
#
# LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
@ctypes_function(
    "llama_sampler_chain_init",
    [llama_sampler_chain_params],
    llama_sampler_p_ctypes,
)
def llama_sampler_chain_init(params: llama_sampler_chain_params, /) -> llama_sampler_p:
    ...


# // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
# LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_chain_add",
    [llama_sampler_p_ctypes, llama_sampler_p_ctypes],
    None,
)
def llama_sampler_chain_add(chain: llama_sampler_p, smpl: llama_sampler_p, /):
    ...


# LLAMA_API struct llama_sampler * llama_sampler_chain_get(const struct llama_sampler * chain, int32_t i);
@ctypes_function(
    "llama_sampler_chain_get",
    [llama_sampler_p_ctypes, ctypes.c_int32],
    llama_sampler_p_ctypes,
)
def llama_sampler_chain_get(
    chain: llama_sampler_p, i: Union[ctypes.c_int32, int], /
) -> llama_sampler_p:
    ...


# LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);
@ctypes_function(
    "llama_sampler_chain_n",
    [llama_sampler_p_ctypes],
    ctypes.c_int,
)
def llama_sampler_chain_n(chain: llama_sampler_p, /) -> int:
    ...


# // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
# LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);
@ctypes_function(
    "llama_sampler_chain_remove",
    [llama_sampler_p_ctypes, ctypes.c_int32],
    llama_sampler_p_ctypes,
)
def llama_sampler_chain_remove(
    chain: llama_sampler_p, i: Union[ctypes.c_int32, int], /
) -> llama_sampler_p:
    ...


# // available samplers:
#
# LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);
@ctypes_function("llama_sampler_init_greedy", [], llama_sampler_p_ctypes)
def llama_sampler_init_greedy() -> llama_sampler_p:
    ...


# LLAMA_API struct llama_sampler * llama_sampler_init_dist  (uint32_t seed);
@ctypes_function("llama_sampler_init_dist", [ctypes.c_uint32], llama_sampler_p_ctypes)
def llama_sampler_init_dist(seed: int) -> llama_sampler_p:
    ...


# /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
# /// NOTE: Avoid using on the full vocabulary as the sorting can become slow. For example, apply top-k or top-p sampling first.
# DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_softmax    (void),
#     "will be removed in the future (see https://github.com/ggerganov/llama.cpp/pull/9896#discussion_r1800920915)");
@ctypes_function("llama_sampler_init_softmax", [], llama_sampler_p_ctypes)
def llama_sampler_init_softmax() -> llama_sampler_p:
    ...


# /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# /// Setting k <= 0 makes this a noop
# LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);
@ctypes_function("llama_sampler_init_top_k", [ctypes.c_int32], llama_sampler_p_ctypes)
def llama_sampler_init_top_k(k: int) -> llama_sampler_p:
    ...


# /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);
@ctypes_function(
    "llama_sampler_init_top_p",
    [ctypes.c_float, ctypes.c_size_t],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_top_p(p: float, min_keep: int) -> llama_sampler_p:
    ...


# /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
# LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);
@ctypes_function(
    "llama_sampler_init_min_p",
    [ctypes.c_float, ctypes.c_size_t],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_min_p(p: float, min_keep: int) -> llama_sampler_p:
    ...


# /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
# LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);
@ctypes_function(
    "llama_sampler_init_typical",
    [ctypes.c_float, ctypes.c_size_t],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_typical(p: float, min_keep: int) -> llama_sampler_p:
    ...


# LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);
@ctypes_function("llama_sampler_init_temp", [ctypes.c_float], llama_sampler_p_ctypes)
def llama_sampler_init_temp(t: float) -> llama_sampler_p:
    ...


# /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
# LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);
@ctypes_function(
    "llama_sampler_init_temp_ext",
    [ctypes.c_float, ctypes.c_float, ctypes.c_float],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_temp_ext(
    t: float, delta: float, exponent: float
) -> llama_sampler_p:
    ...


# /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
# LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);
@ctypes_function(
    "llama_sampler_init_xtc",
    [ctypes.c_float, ctypes.c_float, ctypes.c_size_t, ctypes.c_uint32],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_xtc(
    p: float, t: float, min_keep: int, seed: int, /
) -> llama_sampler_p:
    ...


# /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
# LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);
@ctypes_function(
    "llama_sampler_init_top_n_sigma",
    [ctypes.c_float],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_top_n_sigma(n: float, /) -> llama_sampler_p:
    ...


# /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
#                          int32_t   n_vocab,
#                         uint32_t   seed,
#                            float   tau,
#                            float   eta,
#                          int32_t   m);
@ctypes_function(
    "llama_sampler_init_mirostat",
    [ctypes.c_int32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_int32],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_mirostat(
    n_vocab: int, seed: int, tau: float, eta: float, m: int, /
) -> llama_sampler_p:
    ...


# /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
#                         uint32_t   seed,
#                            float   tau,
#                            float   eta);
@ctypes_function(
    "llama_sampler_init_mirostat_v2",
    [ctypes.c_uint32, ctypes.c_float, ctypes.c_float],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_mirostat_v2(
    seed: int, tau: float, eta: float, /
) -> llama_sampler_p:
    ...


# /// @details Intializes a GBNF grammar, see grammars/README.md for details.
# /// @param vocab The vocabulary that this grammar will be used with.
# /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
# /// @param grammar_root The name of the start symbol for the grammar.
# LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
#         const struct llama_vocab * vocab,
#                       const char * grammar_str,
#                       const char * grammar_root);
@ctypes_function(
    "llama_sampler_init_grammar",
    [llama_vocab_p_ctypes, ctypes.c_char_p, ctypes.c_char_p],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_grammar(
    vocab: llama_vocab_p, grammar_str: bytes, grammar_root: bytes, /
) -> llama_sampler_p:
    ...


# /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
# /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
# /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
# LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
#     const struct llama_vocab * vocab,
#                   const char * grammar_str,
#                   const char * grammar_root,
#                  const char ** trigger_patterns,
#                         size_t num_trigger_patterns,
#            const llama_token * trigger_tokens,
#                         size_t num_trigger_tokens);
@ctypes_function(
    "llama_sampler_init_grammar_lazy_patterns",
    [
        llama_vocab_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(llama_token),
        ctypes.c_size_t,
    ],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_grammar_lazy_patterns(
    vocab: llama_vocab_p,
    grammar_str: bytes,
    grammar_root: bytes,
    trigger_patterns: CtypesArray[bytes],
    num_trigger_patterns: int,
    trigger_tokens: CtypesArray[llama_token],
    num_trigger_tokens: int,
    /,
) -> llama_sampler_p:
    ...


# /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
# LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
#                          int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
#                            float   penalty_repeat,   // 1.0 = disabled
#                            float   penalty_freq,     // 0.0 = disabled
#                            float   penalty_present); // 0.0 = disabled
@ctypes_function(
    "llama_sampler_init_penalties",
    [ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_penalties(
    penalty_last_n: int,
    penalty_repeat: float,
    penalty_freq: float,
    penalty_present: float,
    /,
) -> llama_sampler_p:
    ...


# ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
# LLAMA_API struct llama_sampler *    llama_sampler_init_dry(
#         const struct llama_vocab *  vocab,
#                          int32_t    n_ctx_train,
#                            float    dry_multiplier,
#                            float    dry_base,
#                          int32_t    dry_allowed_length,
#                          int32_t    dry_penalty_last_n,
#                       const char ** seq_breakers,
#                           size_t    num_breakers);
@ctypes_function(
    "llama_sampler_init_dry",
    [
        llama_vocab_p_ctypes,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
    ],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_dry(
    vocab: llama_vocab_p,
    n_ctx_train: int,
    dry_multiplier: float,
    dry_base: float,
    dry_allowed_length: int,
    dry_penalty_last_n: int,
    seq_breakers,
    num_breakers: int,
    /,
) -> llama_sampler_p:
    ...


# LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
#                          int32_t   n_vocab,
#                          int32_t   n_logit_bias,
#           const llama_logit_bias * logit_bias);
@ctypes_function(
    "llama_sampler_init_logit_bias",
    [ctypes.c_int32, ctypes.c_int32, llama_logit_bias_p],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_logit_bias(
    n_vocab: int, n_logit_bias: int, logit_bias: CtypesArray[llama_logit_bias], /
) -> llama_sampler_p:
    ...


# // this sampler is meant to be used for fill-in-the-middle infilling
# // it's supposed to be used after top_k + top_p sampling
# //
# // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
# // 2. combine probs of tokens that have the same prefix
# //
# // example:
# //
# // - before:
# //   "hel":   0.5
# //   "hell":  0.2
# //   "hello": 0.1
# //   "dummy": 0.1
# //
# // - after:
# //   "hel":   0.8
# //   "dummy": 0.1
# //
# // 3. discard non-EOG tokens with low prob
# // 4. if no tokens are left -> pick EOT
# //
# LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_sampler_init_infill",
    [llama_vocab_p_ctypes],
    llama_sampler_p_ctypes,
)
def llama_sampler_init_infill(vocab: llama_vocab_p, /) -> llama_sampler_p:
    ...


# // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
# LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);
@ctypes_function(
    "llama_sampler_get_seed",
    [llama_sampler_p_ctypes],
    ctypes.c_uint32,
)
def llama_sampler_get_seed(smpl: llama_sampler_p, /) -> int:
    ...


# /// @details Sample and accept a token from the idx-th output of the last evaluation
# //
# // Shorthand for:
# //    const auto * logits = llama_get_logits_ith(ctx, idx);
# //    llama_token_data_array cur_p = { ... init from logits ... };
# //    llama_sampler_apply(smpl, &cur_p);
# //    auto token = cur_p.data[cur_p.selected].id;
# //    llama_sampler_accept(smpl, token);
# //    return token;
# // Returns the sampled token
# LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
@ctypes_function(
    "llama_sampler_sample",
    [llama_sampler_p_ctypes, llama_context_p_ctypes, ctypes.c_int32],
    llama_token,
)
def llama_sampler_sample(
    smpl: llama_sampler_p, ctx: llama_context_p, idx: int, /
) -> int:
    ...


# //
# // Model split
# //


# /// @details Build a split GGUF final path for this chunk.
# ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
# //  Returns the split_path length.
# LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);
@ctypes_function(
    "llama_split_path",
    [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)
def llama_split_path(
    split_path: bytes,
    maxlen: Union[ctypes.c_size_t, int],
    path_prefix: bytes,
    split_no: Union[ctypes.c_int, int],
    split_count: Union[ctypes.c_int, int],
    /,
) -> int:
    """Build a split GGUF final path for this chunk."""
    ...


# /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
# ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
# //  Returns the split_prefix length.
# LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);
@ctypes_function(
    "llama_split_prefix",
    [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)
def llama_split_prefix(
    split_prefix: bytes,
    maxlen: Union[ctypes.c_size_t, int],
    split_path: bytes,
    split_no: Union[ctypes.c_int, int],
    split_count: Union[ctypes.c_int, int],
    /,
) -> int:
    """Extract the path prefix from the split_path if and only if the split_no and split_count match."""
    ...


# // Print system information
# LLAMA_API const char * llama_print_system_info(void);
@ctypes_function("llama_print_system_info", [], ctypes.c_char_p)
def llama_print_system_info() -> bytes:
    ...


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


# //
# // Performance utils
# //
# // NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
# //


# struct llama_perf_context_data {
#     double t_start_ms;
#     double t_load_ms;
#     double t_p_eval_ms;
#     double t_eval_ms;
#
#     int32_t n_p_eval;
#     int32_t n_eval;
# };
class llama_perf_context_data(ctypes.Structure):
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
    ]


# struct llama_perf_sampler_data {
#     double t_sample_ms;
#
#     int32_t n_sample;
# };
class llama_perf_sampler_data(ctypes.Structure):
    _fields_ = [
        ("t_sample_ms", ctypes.c_double),
        ("n_sample", ctypes.c_int32),
    ]


# LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
@ctypes_function(
    "llama_perf_context",
    [llama_context_p_ctypes],
    llama_perf_context_data,
)
def llama_perf_context(ctx: llama_context_p, /) -> llama_perf_context_data:
    ...


# LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
@ctypes_function(
    "llama_perf_context_print",
    [llama_context_p_ctypes],
    None,
)
def llama_perf_context_print(ctx: llama_context_p, /):
    ...


# LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);
@ctypes_function(
    "llama_perf_context_reset",
    [llama_context_p_ctypes],
    None,
)
def llama_perf_context_reset(ctx: llama_context_p, /):
    ...


# // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
# LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
@ctypes_function(
    "llama_perf_sampler",
    [llama_sampler_p_ctypes],
    llama_perf_sampler_data,
)
def llama_perf_sampler(chain: llama_sampler_p, /) -> llama_perf_sampler_data:
    ...


# LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
@ctypes_function(
    "llama_perf_sampler_print",
    [llama_sampler_p_ctypes],
    None,
)
def llama_perf_sampler_print(chain: llama_sampler_p, /):
    ...


# LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);
@ctypes_function(
    "llama_perf_sampler_reset",
    [llama_sampler_p_ctypes],
    None,
)
def llama_perf_sampler_reset(chain: llama_sampler_p, /):
    ...


