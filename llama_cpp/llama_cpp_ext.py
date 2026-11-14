"""Compatibility imports for formerly non-public llama.cpp APIs.

These functions are now exposed by llama_cpp.llama_cpp.
This module remains experimental; new callers should use the public bindings.
"""

from .llama_cpp import (
    llama_get_ctx_other as llama_get_ctx_other,
    llama_get_embeddings_nextn as llama_get_embeddings_nextn,
    llama_get_embeddings_nextn_ith as llama_get_embeddings_nextn_ith,
    llama_set_embeddings_nextn as llama_set_embeddings_nextn,
    llama_set_nextn_layer_offset as llama_set_nextn_layer_offset,
)
