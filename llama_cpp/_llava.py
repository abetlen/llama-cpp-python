from __future__ import annotations

import os
import ctypes
import typing
import contextlib

import numpy as np

import llama_cpp
import llama_cpp.llava_cpp as llava_cpp


class LlavaEmbedding:
    def __init__(self, embedding: ctypes._Pointer[llava_cpp.llava_image_embed]):
        self._embedding = embedding
        self._exit_stack = contextlib.ExitStack()

        def llava_image_embed_free():
            llava_cpp.llava_image_embed_free(self._embedding)

        self._exit_stack.callback(llava_image_embed_free)

    @property
    def n_image_pos(self) -> int:
        return self._embedding.contents.n_image_pos

    def embed(
        self, llama_ctx: llama_cpp.llama_context_p, n_tokens: int, n_batch: int
    ) -> int:
        n_past = ctypes.c_int(n_tokens)
        n_past_p = ctypes.pointer(n_past)
        llava_cpp.llava_eval_image_embed(
            llama_ctx,
            self._embedding,
            n_batch,
            n_past_p,
        )
        return n_past.value

    def numpy_view(self, shape: typing.Tuple[int, int]) -> np.ndarray:
        return np.ctypeslib.as_array(
            self._embedding.contents.embed, shape=shape
        )


class LlavaModel:
    def __init__(self, path: str, n_threads: int = 1):
        self._path = path
        self._n_threads = n_threads
        self._exit_stack = contextlib.ExitStack()

        if not os.path.exists(self._path):
            raise ValueError(f"Clip model path does not exist: {self._path}")

        clip_ctx = llava_cpp.clip_model_load(self._path.encode(), 0)

        if clip_ctx is None:
            raise ValueError(f"Failed to load clip model: {self._path}")

        self._clip_ctx = clip_ctx

        def clip_free():
            llava_cpp.clip_free(self._clip_ctx)
            print("Clip model freed")

        self._exit_stack.callback(clip_free)

    def embed_bytes(self, image_bytes: bytes):
        embed = llava_cpp.llava_image_embed_make_with_bytes(
            self._clip_ctx,
            self._n_threads,
            (ctypes.c_uint8 * len(image_bytes)).from_buffer(bytearray(image_bytes)),
            len(image_bytes),
        )
        return LlavaEmbedding(embed)

