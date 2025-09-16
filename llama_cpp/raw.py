"""Minimal transparent bindings over llama.cpp: no hidden logic, no token/header munging.

This module provides LlamaRaw – a deliberately tiny, explicit wrapper around the
ctypes surface. It aims to (a) expose only direct llama.cpp operations and (b)
perform zero implicit mutation of user-provided data other than what the
underlying C API itself performs.

Design principles:
 1. 1:1 parameter mapping to underlying llama.cpp defaults where possible.
 2. No automatic BOS/EOS insertion, space-prefix hacks, infill re‑ordering,
    caching, seed advancement, repetition penalties, grammar, or processor chains.
 3. No Python‑side logits mutation. You receive raw logits exactly as produced.
 4. Resource ownership explicit: model + context + (optional) batch. User frees.
 5. Small, easy to audit code (< ~200 lines) and self‑documenting.

Motivation: Allow advanced users / downstream frameworks to build their own
policies (prompt construction, sampling, caching, grammar) without fighting the
high‑level convenience layer.

NOTE: High‑level `Llama` class remains unchanged and continues to provide its
richer feature set. This raw layer is opt‑in.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from typing import Iterable, List, Sequence, Optional

import numpy as np

import llama_cpp.llama_cpp as llama_cpp
import llama_cpp._internals as internals
from ._utils import suppress_stdout_stderr

_backend_initialized = False


def _ensure_backend_initialized():
    global _backend_initialized
    if not _backend_initialized:
        with suppress_stdout_stderr(disable=True):  # silent like upstream
            llama_cpp.llama_backend_init()
        _backend_initialized = True


@dataclass
class RawModelResources:
    model: internals.LlamaModel
    ctx: internals.LlamaContext
    batch: internals.LlamaBatch

    def close(self):  # explicit free
        # contextlib.closing already handles destructors, but offer explicit hook
        self.batch.close()
        self.ctx.close()
        self.model.close()


class LlamaRaw:
    """Ultra‑thin wrapper exposing only primitive llama.cpp operations.

    Public methods intentionally mirror underlying C semantics.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 512,
        n_batch: int = 512,
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[Sequence[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[dict] = None,
        logits_all: bool = False,
        embedding: bool = False,
        verbose: bool = False,
    ) -> None:
        _ensure_backend_initialized()

        self.verbose = verbose
        self._seed = seed
        self._logits_all = logits_all
        self._n_batch = min(n_ctx, n_batch)

        # --- model params (direct copy, no extra logic) ---
        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = 0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers
        mparams.split_mode = split_mode
        mparams.main_gpu = main_gpu
        if tensor_split is not None:
            if len(tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:
                raise ValueError("tensor_split length exceeds LLAMA_MAX_DEVICES")
            FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES
            self._c_tensor_split = FloatArray(*tensor_split)  # keep ref
            mparams.tensor_split = self._c_tensor_split
        mparams.vocab_only = vocab_only
        mparams.use_mmap = use_mmap
        mparams.use_mlock = use_mlock
        self._kv_overrides = None
        if kv_overrides:
            # Direct translation – identical to high level but without abstraction.
            arr_len = len(kv_overrides) + 1
            overrides = (llama_cpp.llama_model_kv_override * arr_len)()
            for i, (k, v) in enumerate(kv_overrides.items()):
                overrides[i].key = k.encode("utf-8")
                if isinstance(v, bool):
                    overrides[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL
                    overrides[i].value.val_bool = v
                elif isinstance(v, int):
                    overrides[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT
                    overrides[i].value.val_i64 = v
                elif isinstance(v, float):
                    overrides[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT
                    overrides[i].value.val_f64 = v
                elif isinstance(v, str):
                    vb = v.encode("utf-8")
                    if len(vb) > 128:
                        raise ValueError("kv_override str too long (max 128)")
                    vb = vb.ljust(128, b"\0")
                    overrides[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR
                    # Directly copy into the union's byte array region
                    addr = ctypes.addressof(overrides[i].value)
                    ctypes.memmove(addr, vb, 128)
                else:
                    raise TypeError(f"Unsupported kv_override type for {k}: {type(v)}")
            overrides[-1].key = b"\0"
            mparams.kv_overrides = overrides
            self._kv_overrides = overrides  # keep ref

        # --- context params ---
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = self._n_batch
        cparams.n_ubatch = self._n_batch
        cparams.embeddings = embedding
        cparams.seed = seed  # if supported in version, else ignored silently
        if logits_all:
            cparams.logits_all = 1  # some versions may ignore; harmless

        # Load model + context
        self._model = internals.LlamaModel(
            path_model=model_path, params=mparams, verbose=verbose
        )
        self._ctx = internals.LlamaContext(
            model=self._model, params=cparams, verbose=verbose
        )
        self._batch = internals.LlamaBatch(
            n_tokens=self._n_batch, embd=0, n_seq_max=cparams.n_ctx, verbose=verbose
        )

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()

    # ---------------------- primitive queries ----------------------
    def n_vocab(self) -> int:  # direct
        return self._model.n_vocab()

    def n_ctx(self) -> int:  # direct
        return self._ctx.n_ctx()

    def n_embd(self) -> int:
        return self._model.n_embd()

    # ---------------------- tokenization (raw) ----------------------
    def tokenize(
        self, text: bytes, add_bos: bool = True, parse_special: bool = False
    ) -> List[int]:
        # No prefix space, no BOS/EOS stripping beyond requested add_bos.
        # We ask model for max potential tokens (heuristic: len(text)+8) like upstream guidance.
        max_tokens = len(text) + 8
        arr = (llama_cpp.llama_token * max_tokens)()
        # llama_tokenize expects a model pointer
        n = llama_cpp.llama_tokenize(
            self._model.model,
            text,
            len(text),
            arr,
            max_tokens,
            add_bos,
            parse_special,
        )
        if n < 0:
            raise RuntimeError("Tokenization failed (buffer too small?)")
        return [arr[i] for i in range(n)]

    def detokenize(self, tokens: Sequence[int]) -> bytes:
        # Straight concatenation of token pieces (no space heuristics)
        parts: List[bytes] = []
        vocab = self._model.vocab
        for t in tokens:
            parts.append(llama_cpp.llama_token_get_text(vocab, t))
        return b"".join(parts)

    # ---------------------- evaluation ----------------------
    def eval(self, tokens: Sequence[int]):
        # Evaluate tokens sequentially in mini-batches; no cache prefix tricks.
        for i in range(0, len(tokens), self._n_batch):
            batch_slice = tokens[i : i + self._n_batch]
            self._batch.set_batch(batch_slice, n_past=self._ctx.n_tokens(), logits_all=self._logits_all)  # type: ignore[arg-type]
            self._ctx.decode(self._batch)

    def get_logits(self) -> np.ndarray:
        # Return view (copy to ensure immutability by caller)
        shape = (self._n_vocab,)
        logits = np.ctypeslib.as_array(self._ctx.get_logits(), shape=shape)
        return logits.copy()

    # ---------------------- sampling ----------------------
    def sample_greedy(self) -> int:
        # Pure greedy: argmax over last logits
        logits = self.get_logits()
        return int(int(np.argmax(logits)))

    # (Optional) user can implement their own top-k/p externally using returned logits.

    # ---------------------- embeddings ----------------------
    def get_embeddings(self):
        if not self._ctx.ctx:  # safety
            raise RuntimeError("Context not initialized")
        ptr = llama_cpp.llama_get_embeddings(self._ctx.ctx)
        if ptr is None:
            raise RuntimeError("Embeddings not available (model not in embedding mode)")
        n_embd = self.n_embd()
        return ptr[:n_embd]

    # ---------------------- state I/O ----------------------
    def save_state_bytes(self) -> bytes:
        size = llama_cpp.llama_get_state_size(self._ctx.ctx)
        buf = (ctypes.c_uint8 * size)()
        wrote = llama_cpp.llama_copy_state_data(self._ctx.ctx, buf)
        if wrote != size:
            raise RuntimeError("State copy size mismatch")
        return bytes(buf)

    def load_state_bytes(self, data: bytes):
        size = len(data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(data)
        wrote = llama_cpp.llama_set_state_data(self._ctx.ctx, buf)
        if wrote != size:
            raise RuntimeError("State restore size mismatch")

    # ---------------------- RNG seed ----------------------
    def set_seed(self, seed: int):
        # Only updates internal tracking; llama.cpp global sampler seeds are separate when using custom sampling.
        self._seed = seed

    # ---------------------- teardown ----------------------
    def close(self):
        self._batch.close()
        self._ctx.close()
        self._model.close()

    # Support context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):  # safety – explicit free
        try:
            self.close()
        except Exception:
            pass
