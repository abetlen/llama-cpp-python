from __future__ import annotations

import os
import ctypes

from typing import (
    List,
    Optional,
    Sequence,
)
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .llama_types import *
from .llama_grammar import LlamaGrammar

import llama_cpp.llama_cpp as llama_cpp


# Python wrappers over llama.h structs


class _LlamaModel:
    """Intermediate Python wrapper for a llama.cpp llama_model.
    NOTE: For stability it's recommended you use the Llama class instead."""

    _llama_free_model = None
    # NOTE: this must be "saved" here to avoid exceptions when calling __del__

    def __init__(
        self,
        *,
        path_model: str,
        params: llama_cpp.llama_model_params,
        verbose: bool = True,
    ):
        self.path_model = path_model
        self.params = params
        self.verbose = verbose

        self._llama_free_model = llama_cpp._lib.llama_free_model  # type: ignore

        self.model = None

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        self.model = llama_cpp.llama_load_model_from_file(
            self.path_model.encode("utf-8"), self.params
        )

        if self.model is None:
            raise ValueError(f"Failed to load model from file: {path_model}")

    def __del__(self):
        if self.model is not None and self._llama_free_model is not None:
            self._llama_free_model(self.model)
            self.model = None

    def vocab_type(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_n_vocab(self.model)

    def n_ctx_train(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_n_ctx_train(self.model)

    def n_embd(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        assert self.model is not None
        return llama_cpp.llama_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        assert self.model is not None
        buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.value.decode("utf-8")

    def size(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_model_size(self.model)

    def n_params(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_model_n_params(self.model)

    def get_tensor(self, name: str) -> ctypes.c_void_p:
        assert self.model is not None
        return llama_cpp.llama_get_model_tensor(self.model, name.encode("utf-8"))

    def apply_lora_from_file(
        self,
        lora_path: str,
        scale: float,
        path_base_model: Optional[str],
        n_threads: int,
    ):
        assert self.model is not None
        return llama_cpp.llama_model_apply_lora_from_file(
            self.model,
            lora_path.encode("utf-8"),
            scale,
            path_base_model.encode("utf-8")
            if path_base_model is not None
            else ctypes.c_char_p(0),
            n_threads,
        )

    # Vocab

    def token_get_text(self, token: int) -> str:
        # TODO: Fix
        assert self.model is not None
        return llama_cpp.llama_token_get_text(self.model, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        assert self.model is not None
        return llama_cpp.llama_token_get_score(self.model, token)

    def token_get_type(self, token: int) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_get_type(self.model, token)

    # Special tokens

    def token_bos(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_bos(self.model)

    def token_eos(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_eos(self.model)

    def token_nl(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_nl(self.model)

    def token_prefix(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_prefix(self.model)

    def token_middle(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_middle(self.model)

    def token_suffix(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_suffix(self.model)

    def token_eot(self) -> int:
        assert self.model is not None
        return llama_cpp.llama_token_eot(self.model)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool):
        assert self.model is not None
        n_ctx = self.n_ctx_train()
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            self.model, text, len(text), tokens, n_ctx, add_bos, special
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.model, text, len(text), tokens, n_tokens, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )
        return list(tokens[:n_tokens])

    def token_to_piece(self, token: int) -> bytes:
        assert self.model is not None
        buf = ctypes.create_string_buffer(32)
        llama_cpp.llama_token_to_piece(self.model, token, buf, 32)
        return bytes(buf)

    def detokenize(self, tokens: List[int]) -> bytes:
        assert self.model is not None
        output = b""
        size = 32
        buffer = (ctypes.c_char * size)()
        for token in tokens:
            n = llama_cpp.llama_token_to_piece(
                self.model, llama_cpp.llama_token(token), buffer, size
            )
            assert n <= size
            output += bytes(buffer[:n])
        # NOTE: Llama1 models automatically added a space at the start of the prompt
        # this line removes a leading space if the first token is a beginning of sentence token
        return (
            output[1:] if len(tokens) > 0 and tokens[0] == self.token_bos() else output
        )

    # Extra
    def metadata(self) -> Dict[str, str]:
        assert self.model is not None
        metadata: Dict[str, str] = {}
        buffer_size = 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        # zero the buffer
        buffer.value = b'\0' * buffer_size
        # iterate over model keys
        for i in range(llama_cpp.llama_model_meta_count(self.model)):
            nbytes = llama_cpp.llama_model_meta_key_by_index(self.model, i, buffer, buffer_size)
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = ctypes.create_string_buffer(buffer_size)
                nbytes = llama_cpp.llama_model_meta_key_by_index(self.model, i, buffer, buffer_size)
            key = buffer.value.decode("utf-8")
            nbytes = llama_cpp.llama_model_meta_val_str_by_index(self.model, i, buffer, buffer_size)
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = ctypes.create_string_buffer(buffer_size)
                nbytes = llama_cpp.llama_model_meta_val_str_by_index(self.model, i, buffer, buffer_size)
            value = buffer.value.decode("utf-8")
            metadata[key] = value
        return metadata

    @staticmethod
    def default_params():
        """Get the default llama_model_params."""
        return llama_cpp.llama_model_default_params()


class _LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context.
    NOTE: For stability it's recommended you use the Llama class instead."""

    _llama_free = None

    def __init__(
        self,
        *,
        model: _LlamaModel,
        params: llama_cpp.llama_context_params,
        verbose: bool = True,
    ):
        self.model = model
        self.params = params
        self.verbose = verbose

        self._llama_free = llama_cpp._lib.llama_free  # type: ignore
        self.ctx = None

        assert self.model.model is not None

        self.ctx = llama_cpp.llama_new_context_with_model(
            self.model.model, self.params
        )

        if self.ctx is None:
            raise ValueError("Failed to create llama_context")

    def __del__(self):
        if self.ctx is not None and self._llama_free is not None:
            self._llama_free(self.ctx)
            self.ctx = None

    def n_ctx(self) -> int:
        assert self.ctx is not None
        return llama_cpp.llama_n_ctx(self.ctx)

    def kv_cache_clear(self):
        assert self.ctx is not None
        llama_cpp.llama_kv_cache_clear(self.ctx)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        assert self.ctx is not None
        llama_cpp.llama_kv_cache_seq_rm(self.ctx, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        assert self.ctx is not None
        llama_cpp.llama_kv_cache_seq_cp(self.ctx, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        assert self.ctx is not None
        llama_cpp.llama_kv_cache_seq_keep(self.ctx, seq_id)

    def kv_cache_seq_shift(self, seq_id: int, p0: int, p1: int, shift: int):
        assert self.ctx is not None
        llama_cpp.llama_kv_cache_seq_add(self.ctx, seq_id, p0, p1, shift)

    def get_state_size(self) -> int:
        assert self.ctx is not None
        return llama_cpp.llama_get_state_size(self.ctx)

    # TODO: copy_state_data

    # TODO: set_state_data

    # TODO: llama_load_session_file

    # TODO: llama_save_session_file

    def decode(self, batch: "_LlamaBatch"):
        assert self.ctx is not None
        assert batch.batch is not None
        return_code = llama_cpp.llama_decode(
            self.ctx,
            batch.batch,
        )
        if return_code != 0:
            raise RuntimeError(f"llama_decode returned {return_code}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        assert self.ctx is not None
        llama_cpp.llama_set_n_threads(self.ctx, n_threads, n_threads_batch)

    def get_logits(self):
        assert self.ctx is not None
        return llama_cpp.llama_get_logits(self.ctx)

    def get_logits_ith(self, i: int):
        assert self.ctx is not None
        return llama_cpp.llama_get_logits_ith(self.ctx, i)

    def get_embeddings(self):
        assert self.ctx is not None
        return llama_cpp.llama_get_embeddings(self.ctx)

    # Sampling functions

    def set_rng_seed(self, seed: int):
        assert self.ctx is not None
        llama_cpp.llama_set_rng_seed(self.ctx, seed)

    def sample_repetition_penalties(
        self,
        candidates: "_LlamaTokenDataArray",
        last_tokens_data: "llama_cpp.Array[llama_cpp.llama_token]",
        penalty_last_n: int,
        penalty_repeat: float,
        penalty_freq: float,
        penalty_present: float,
    ):
        assert self.ctx is not None
        llama_cpp.llama_sample_repetition_penalties(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
            last_tokens_data,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        )

    def sample_softmax(self, candidates: "_LlamaTokenDataArray"):
        assert self.ctx is not None
        llama_cpp.llama_sample_softmax(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
        )

    def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
        assert self.ctx is not None
        llama_cpp.llama_sample_top_k(
            self.ctx, llama_cpp.byref(candidates.candidates), k, min_keep
        )

    def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        assert self.ctx is not None
        llama_cpp.llama_sample_top_p(
            self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        )

    def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        assert self.ctx is not None
        llama_cpp.llama_sample_min_p(
            self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        )

    def sample_tail_free(
        self, candidates: "_LlamaTokenDataArray", z: float, min_keep: int
    ):
        assert self.ctx is not None
        llama_cpp.llama_sample_tail_free(
            self.ctx, llama_cpp.byref(candidates.candidates), z, min_keep
        )

    def sample_typical(
        self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    ):
        assert self.ctx is not None
        llama_cpp.llama_sample_typical(
            self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        )

    def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
        assert self.ctx is not None
        llama_cpp.llama_sample_temp(
            self.ctx, llama_cpp.byref(candidates.candidates), temp
        )

    def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
        assert self.ctx is not None
        assert grammar.grammar is not None
        llama_cpp.llama_sample_grammar(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
            grammar.grammar,
        )

    def sample_token_mirostat(
        self,
        candidates: "_LlamaTokenDataArray",
        tau: float,
        eta: float,
        m: int,
        mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    ) -> int:
        assert self.ctx is not None
        return llama_cpp.llama_sample_token_mirostat(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
            tau,
            eta,
            m,
            mu,
        )

    def sample_token_mirostat_v2(
        self, candidates: "_LlamaTokenDataArray", tau: float, eta: float, mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float]
    ) -> int:
        assert self.ctx is not None
        return llama_cpp.llama_sample_token_mirostat_v2(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
            tau,
            eta,
            mu,
        )

    def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
        assert self.ctx is not None
        return llama_cpp.llama_sample_token_greedy(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
        )

    def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
        assert self.ctx is not None
        return llama_cpp.llama_sample_token(
            self.ctx,
            llama_cpp.byref(candidates.candidates),
        )

    # Grammar
    def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
        assert self.ctx is not None
        assert grammar.grammar is not None
        llama_cpp.llama_grammar_accept_token(self.ctx, grammar.grammar, token)

    def reset_timings(self):
        assert self.ctx is not None
        llama_cpp.llama_reset_timings(self.ctx)

    def print_timings(self):
        assert self.ctx is not None
        llama_cpp.llama_print_timings(self.ctx)

    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return llama_cpp.llama_context_default_params()


class _LlamaBatch:
    _llama_batch_free = None

    def __init__(
        self, *, n_tokens: int, embd: int, n_seq_max: int, verbose: bool = True
    ):
        self._n_tokens = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose

        self._llama_batch_free = llama_cpp._lib.llama_batch_free  # type: ignore

        self.batch = None
        self.batch = llama_cpp.llama_batch_init(
            self._n_tokens, self.embd, self.n_seq_max
        )

    def __del__(self):
        if self.batch is not None and self._llama_batch_free is not None:
            self._llama_batch_free(self.batch)
            self.batch = None

    def n_tokens(self) -> int:
        assert self.batch is not None
        return self.batch.n_tokens

    def reset(self):
        assert self.batch is not None
        self.batch.n_tokens = 0

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        assert self.batch is not None
        n_tokens = len(batch)
        self.batch.n_tokens = n_tokens
        for i in range(n_tokens):
            self.batch.token[i] = batch[i]
            self.batch.pos[i] = n_past + i
            self.batch.seq_id[i][0] = 0
            self.batch.n_seq_id[i] = 1
            self.batch.logits[i] = logits_all
        self.batch.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        assert self.batch is not None
        n_tokens = len(batch)
        n_tokens0 = self.batch.n_tokens
        self.batch.n_tokens += n_tokens
        for i in range(n_tokens):
            j = n_tokens0 + i
            self.batch.token[j] = batch[i]
            self.batch.pos[j] = i
            self.batch.seq_id[j][0] = seq_id
            self.batch.n_seq_id[j] = 1
            self.batch.logits[j] = logits_all
        self.batch.logits[n_tokens - 1] = True


class _LlamaTokenDataArray:
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab
        self.candidates_data = np.array(
            [],
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self.candidates_data.resize(3, self.n_vocab, refcheck=False)
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            sorted=False,
        )
        self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc) # type: ignore
        self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

    def copy_logits(self, logits: npt.NDArray[np.single]):
        self.candidates_data["id"][:] = self.default_candidates_data_id
        self.candidates_data["logit"][:] = logits
        self.candidates_data["p"][:] = self.default_candidates_data_p
        self.candidates.data = self.candidates_data.ctypes.data_as(
            llama_cpp.llama_token_data_p
        )
        self.candidates.sorted = ctypes.c_bool(False)
        self.candidates.size = ctypes.c_size_t(self.n_vocab)


# Python wrappers over common/common
def _tokenize(model: _LlamaModel, text: str, add_bos: bool, special: bool) -> list[int]:
    assert model.model is not None
    n_tokens = len(text) + 1 if add_bos else len(text)
    result = (llama_cpp.llama_token * n_tokens)()
    n_tokens = llama_cpp.llama_tokenize(
        model.model,
        text.encode("utf-8"),
        len(text),
        result,
        n_tokens,
        add_bos,
        special,
    )
    if n_tokens < 0:
        result = (llama_cpp.llama_token * -n_tokens)()
        check = llama_cpp.llama_tokenize(
            model.model,
            text.encode("utf-8"),
            len(text),
            result,
            len(result),
            add_bos,
            special,
        )
        if check != -n_tokens:
            raise RuntimeError(f'Failed to tokenize: text="{text}" n_tokens={n_tokens}')
    else:
        result = result[:n_tokens]
    return list(result)


def _token_to_piece(model: _LlamaModel, token: int) -> str:
    assert model.model is not None
    result = (ctypes.c_char * 8)(0)
    n_tokens = llama_cpp.llama_token_to_piece(model.model, token, result, len(result))
    if n_tokens < 0:
        result = (ctypes.c_char * -n_tokens)(0)
        check = llama_cpp.llama_token_to_piece(model.model, token, result, len(result))
        if check != -n_tokens:
            raise RuntimeError(f"Failed to get piece: token={token}")
    else:
        result = result[:n_tokens]
    return bytes(result).decode("utf-8")


def _detokenize_spm(model: _LlamaModel, tokens: List[int]) -> str:
    bos_id = model.token_bos()
    result = ""
    for i, token in enumerate(tokens):
        piece = _token_to_piece(model, token)
        if (
            (tokens[0] == bos_id and i == 1) or (tokens[0] != bos_id and i == 0)
        ) and piece[0] == " ":
            piece = piece[1:]
        result += piece
    return result


def _detokenize_bpe(model: _LlamaModel, tokens: List[int]) -> str:
    result = ""
    for token in tokens:
        piece = _token_to_piece(model, token)
        result += piece
    return result


def _should_add_bos(model: _LlamaModel) -> bool:
    assert model.model is not None
    add_bos = llama_cpp.llama_add_bos_token(model.model)
    if add_bos != -1:
        return add_bos != 0
    else:
        return llama_cpp.llama_vocab_type(model.model) == llama_cpp.LLAMA_VOCAB_TYPE_SPM


# Python wrappers over common/sampling structs


@dataclass
class _LlamaSamplingParams:
    n_prev: int = 64
    n_probs: int = 0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    tfs_z: float = 1.00
    typical_p: float = 1.00
    temp: float = 0.80
    penalty_last_n: int = 64
    penalty_repeat: float = 1.10
    penalty_freq: float = 0.00
    penalty_present: float = 0.00
    mirostat: int = 0
    mirostat_tau: float = 5.00
    mirostat_eta: float = 0.10
    penalize_nl: bool = True

    grammar: str = ""

    cfg_negative_prompt: str = ""
    cfg_scale: float = 1.00

    logit_bias: dict[int, float] = field(default_factory=dict)


@dataclass
class _LlamaSamplingContext:
    params: _LlamaSamplingParams = field(default_factory=_LlamaSamplingParams)
    mirostat_mu: ctypes.c_float = field(default_factory=ctypes.c_float)
    grammar: Optional[LlamaGrammar] = None
    # NOTE: Missing parsed_grammar
    prev: list[int] = field(default_factory=list)
    cur: list[llama_cpp.llama_token_data] = field(default_factory=list)

    def reset(self):
        self.prev = []
        self.cur = []
        if self.grammar is not None:
            self.grammar.reset()

    def cp(self):
        return _LlamaSamplingContext(
            params=self.params,
            mirostat_mu=self.mirostat_mu,
            grammar=self.grammar,
            prev=self.prev.copy(),
            cur=self.cur.copy(),
        )

    def last(self) -> Optional[int]:
        if len(self.prev) > 0:
            return self.prev[-1]
        else:
            return None

    def prev_str(self, ctx_main: _LlamaContext, n: int) -> str:
        return ctx_main.model.detokenize(self.prev[-n:]).decode("utf-8")

    def sample(
        self, ctx_main: _LlamaContext, idx: int = 0, logits_array: Optional[npt.NDArray[np.single]] = None
    ):
        n_vocab = ctx_main.model.n_vocab()
        id: int = 0

        if logits_array is None:
            logits = ctx_main.get_logits_ith(idx)
            logits_array = np.array(
                ctypes.cast(logits, ctypes.POINTER(ctypes.c_float * n_vocab)).contents,
                dtype=np.single,
            )

        # apply logit_bias
        for token, logit_bias in self.params.logit_bias.items():
            logits_array[token] += logit_bias

        token_data_array = _LlamaTokenDataArray(
            n_vocab=n_vocab
        )  # TODO: Only create this once
        token_data_array.copy_logits(logits_array)

        # apply penalties
        if len(self.prev) > 0:
            nl_token = ctx_main.model.token_nl()
            nl_logit = logits_array[nl_token]
            if self.params.penalty_last_n > 0:
                ctx_main.sample_repetition_penalties(
                    token_data_array,
                    # TODO: Only create this once
                    (llama_cpp.llama_token * len(self.prev))(*self.prev),
                    self.params.penalty_last_n,
                    self.params.penalty_repeat,
                    self.params.penalty_freq,
                    self.params.penalty_present,
                )
            if not self.params.penalize_nl:
                token_data_array.candidates_data["logit"][nl_token] = nl_logit

        if self.grammar is not None:
            ctx_main.sample_grammar(token_data_array, self.grammar)

        if self.params.temp < 0:
            ctx_main.sample_softmax(token_data_array)
            id = token_data_array.candidates_data["id"][0]
        elif self.params.temp == 0:
            id = ctx_main.sample_token_greedy(token_data_array)
        else:
            if self.params.mirostat == 1:
                mirostat_m = 100
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token_mirostat(
                    token_data_array,
                    self.params.mirostat_tau,
                    self.params.mirostat_eta,
                    mirostat_m,
                    ctypes.pointer(self.mirostat_mu),
                )
            elif self.params.mirostat == 2:
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token_mirostat_v2(
                    token_data_array,
                    self.params.mirostat_tau,
                    self.params.mirostat_eta,
                    ctypes.pointer(self.mirostat_mu),
                )
            else:
                min_keep = max(1, self.params.n_probs)
                ctx_main.sample_top_k(
                    token_data_array, self.params.top_k, min_keep=min_keep
                )
                ctx_main.sample_tail_free(
                    token_data_array, self.params.tfs_z, min_keep=min_keep
                )
                ctx_main.sample_typical(
                    token_data_array, self.params.typical_p, min_keep=min_keep
                )
                ctx_main.sample_top_p(
                    token_data_array, self.params.top_p, min_keep=min_keep
                )
                ctx_main.sample_min_p(
                    token_data_array, self.params.min_p, min_keep=min_keep
                )
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token(token_data_array)
        return id

    def accept(self, ctx_main: _LlamaContext, id: int, apply_grammar: bool):
        if apply_grammar and self.grammar is not None:
            ctx_main.grammar_accept_token(self.grammar, id)
        self.prev.append(id)