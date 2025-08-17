from __future__ import annotations

import os
import ctypes

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Sequence,
    Callable,
    Union,
)
from dataclasses import dataclass, field
from contextlib import ExitStack

import numpy as np
import numpy.typing as npt

from .llama_types import *
from .llama_grammar import LlamaGrammar
from ._utils import suppress_stdout_stderr

import llama_cpp.llama_cpp as llama_cpp


# Python wrappers over llama.h structs


class LlamaModel:
    """Intermediate Python wrapper for a llama.cpp llama_model.
    NOTE: For stability it's recommended you use the Llama class instead."""

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
        self._exit_stack = ExitStack()

        model = None

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        with suppress_stdout_stderr(disable=verbose):
            model = llama_cpp.llama_model_load_from_file(
                self.path_model.encode("utf-8"), self.params
            )

        if model is None:
            raise ValueError(f"Failed to load model from file: {path_model}")

        vocab = llama_cpp.llama_model_get_vocab(model)

        if vocab is None:
            raise ValueError(f"Failed to get vocab from model: {path_model}")

        self.model = model
        self.vocab = vocab
        self.sampler = None  # LlamaModel doesn't use samplers, but some cleanup code expects this attribute

        def free_model():
            if self.model is None:
                return
            llama_cpp.llama_model_free(self.model)
            self.model = None

        self._exit_stack.callback(free_model)

    def close(self):
        if self.sampler is not None:
            # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
            for i, _ in reversed(self.custom_samplers):
                llama_cpp.llama_sampler_chain_remove(self.sampler, i)
            self.custom_samplers.clear()
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def vocab_type(self) -> int:
        return llama_cpp.llama_vocab_type(self.vocab)

    def n_vocab(self) -> int:
        return llama_cpp.llama_vocab_n_tokens(self.vocab)

    def n_ctx_train(self) -> int:
        return llama_cpp.llama_model_n_ctx_train(self.model)

    def n_embd(self) -> int:
        return llama_cpp.llama_model_n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        return llama_cpp.llama_model_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.value.decode("utf-8")

    def size(self) -> int:
        return llama_cpp.llama_model_size(self.model)

    def n_params(self) -> int:
        return llama_cpp.llama_model_n_params(self.model)

    def get_tensor(self, name: str) -> ctypes.c_void_p:
        raise NotImplementedError("get_tensor is not implemented in llama.cpp")

    # Vocab

    def token_get_text(self, token: int) -> str:
        return llama_cpp.llama_vocab_get_text(self.vocab, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        return llama_cpp.llama_vocab_get_score(self.vocab, token)

    def token_get_attr(self, token: int) -> int:
        return llama_cpp.llama_vocab_get_attr(self.vocab, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama_cpp.llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_cpp.llama_vocab_eos(self.vocab)

    def token_cls(self) -> int:
        return llama_cpp.llama_vocab_cls(self.vocab)

    def token_sep(self) -> int:
        return llama_cpp.llama_vocab_sep(self.vocab)

    def token_nl(self) -> int:
        return llama_cpp.llama_vocab_nl(self.vocab)

    def token_prefix(self) -> int:
        return llama_cpp.llama_vocab_fim_pre(self.vocab)

    def token_middle(self) -> int:
        return llama_cpp.llama_vocab_fim_mid(self.vocab)

    def token_suffix(self) -> int:
        return llama_cpp.llama_vocab_fim_suf(self.vocab)

    def token_eot(self) -> int:
        return llama_cpp.llama_vocab_eot(self.vocab)

    def add_bos_token(self) -> bool:
        return llama_cpp.llama_vocab_get_add_bos(self.vocab)

    def add_eos_token(self) -> bool:
        return llama_cpp.llama_vocab_get_add_eos(self.vocab)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool):
        n_ctx = self.n_ctx_train()
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            self.vocab, text, len(text), tokens, n_ctx, add_bos, special
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.vocab, text, len(text), tokens, n_tokens, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )
        return list(tokens[:n_tokens])

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        buf = ctypes.create_string_buffer(32)
        llama_cpp.llama_token_to_piece(self.vocab, token, buf, 32, 0, special)
        return bytes(buf)

    def detokenize(self, tokens: List[int], special: bool = False) -> bytes:
        output = b""
        size = 32
        buffer = (ctypes.c_char * size)()
        for token in tokens:
            n = llama_cpp.llama_token_to_piece(
                self.vocab, llama_cpp.llama_token(token), buffer, size, 0, special
            )
            assert n <= size
            output += bytes(buffer[:n])
        # NOTE: Llama1 models automatically added a space at the start of the prompt
        # this line removes a leading space if the first token is a beginning of sentence token
        return (
            output[1:]
            if len(tokens) > 0 and tokens[0] == self.token_bos() and output[0:1] == b" "
            else output
        )

    # Extra
    def metadata(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        buffer_size = 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        # zero the buffer
        buffer.value = b"\0" * buffer_size
        # iterate over model keys
        for i in range(llama_cpp.llama_model_meta_count(self.model)):
            nbytes = llama_cpp.llama_model_meta_key_by_index(
                self.model, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = ctypes.create_string_buffer(buffer_size)
                nbytes = llama_cpp.llama_model_meta_key_by_index(
                    self.model, i, buffer, buffer_size
                )
            key = buffer.value.decode("utf-8")
            nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                self.model, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = ctypes.create_string_buffer(buffer_size)
                nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                    self.model, i, buffer, buffer_size
                )
            value = buffer.value.decode("utf-8")
            metadata[key] = value
        return metadata

    @staticmethod
    def default_params():
        """Get the default llama_model_params."""
        return llama_cpp.llama_model_default_params()


class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context.
    NOTE: For stability it's recommended you use the Llama class instead."""

    def __init__(
        self,
        *,
        model: LlamaModel,
        params: llama_cpp.llama_context_params,
        verbose: bool = True,
    ):
        self.model = model
        self.params = params
        self.verbose = verbose
        self._exit_stack = ExitStack()

        ctx = llama_cpp.llama_init_from_model(self.model.model, self.params)

        if ctx is None:
            raise ValueError("Failed to create llama_context")

        self.ctx = ctx
        self.memory = llama_cpp.llama_get_memory(self.ctx)
        self.sampler = None  # LlamaContext doesn't manage samplers directly, but some cleanup code expects this attribute

        def free_ctx():
            if self.ctx is None:
                return
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self.ctx)

    def pooling_type(self) -> int:
        return llama_cpp.llama_pooling_type(self.ctx)

    def kv_cache_clear(self):
        assert self.memory is not None, "Memory is not initialized"
        llama_cpp.llama_memory_clear(self.memory, True)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        assert self.memory is not None, "Memory is not initialized"
        seq_id = seq_id if seq_id >= 0 else 0
        llama_cpp.llama_memory_seq_rm(self.memory, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        assert self.memory is not None, "Memory is not initialized"
        llama_cpp.llama_memory_seq_cp(self.memory, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        assert self.memory is not None, "Memory is not initialized"
        llama_cpp.llama_memory_seq_keep(self.memory, seq_id)

    def kv_cache_seq_shift(self, seq_id: int, p0: int, p1: int, shift: int):
        assert self.memory is not None, "Memory is not initialized"
        llama_cpp.llama_memory_seq_add(self.memory, seq_id, p0, p1, shift)

    def get_state_size(self) -> int:
        return llama_cpp.llama_state_get_size(self.ctx)

    # TODO: copy_state_data

    # TODO: set_state_data

    # TODO: llama_load_session_file

    # TODO: llama_save_session_file

    def decode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_decode(
            self.ctx,
            batch.batch,
        )
        if return_code != 0:
            raise RuntimeError(f"llama_decode returned {return_code}")

    def encode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_encode(
            self.ctx,
            batch.batch,
        )
        if return_code != 0:
            raise RuntimeError(f"llama_encode returned {return_code}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        llama_cpp.llama_set_n_threads(self.ctx, n_threads, n_threads_batch)

    def get_logits(self):
        return llama_cpp.llama_get_logits(self.ctx)

    def get_logits_ith(self, i: int):
        return llama_cpp.llama_get_logits_ith(self.ctx, i)

    def get_embeddings(self):
        return llama_cpp.llama_get_embeddings(self.ctx)

    def get_embeddings_ith(self, i: int):
        return llama_cpp.llama_get_embeddings_ith(self.ctx, i)

    def get_embeddings_seq(self, seq_id: int):
        return llama_cpp.llama_get_embeddings_seq(self.ctx, seq_id)

    # Sampling functions - deprecated, use LlamaSampler instead

    def set_rng_seed(self, seed: int):
        raise NotImplementedError("set_rng_seed is deprecated, use LlamaSampler instead")

    def sample_repetition_penalties(
        self,
        candidates: "_LlamaTokenDataArray",
        last_tokens_data: "llama_cpp.Array[llama_cpp.llama_token]",
        penalty_last_n: int,
        penalty_repeat: float,
        penalty_freq: float,
        penalty_present: float,
    ):
        raise NotImplementedError("sample_repetition_penalties is deprecated, use LlamaSampler instead")

    def sample_softmax(self, candidates: "_LlamaTokenDataArray"):
        raise NotImplementedError("sample_softmax is deprecated, use LlamaSampler instead")

    def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
        raise NotImplementedError("sample_top_k is deprecated, use LlamaSampler instead")

    def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        raise NotImplementedError("sample_top_p is deprecated, use LlamaSampler instead")

    def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        raise NotImplementedError("sample_min_p is deprecated, use LlamaSampler instead")

    def sample_typical(
        self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    ):
        raise NotImplementedError("sample_typical is deprecated, use LlamaSampler instead")

    def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
        raise NotImplementedError("sample_temp is deprecated, use LlamaSampler instead")

    def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
        raise NotImplementedError("sample_grammar is deprecated, use LlamaSampler instead")

    def sample_token_mirostat(
        self,
        candidates: "_LlamaTokenDataArray",
        tau: float,
        eta: float,
        m: int,
        mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    ) -> int:
        raise NotImplementedError("sample_token_mirostat is deprecated, use LlamaSampler instead")

    def sample_token_mirostat_v2(
        self,
        candidates: "_LlamaTokenDataArray",
        tau: float,
        eta: float,
        mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    ) -> int:
        raise NotImplementedError("sample_token_mirostat_v2 is deprecated, use LlamaSampler instead")

    def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
        raise NotImplementedError("sample_token_greedy is deprecated, use LlamaSampler instead")

    def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
        raise NotImplementedError("sample_token is deprecated, use LlamaSampler instead")

    # Grammar
    def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
        raise NotImplementedError("grammar_accept_token is deprecated, use LlamaSampler instead")

    def reset_timings(self):
        llama_cpp.llama_perf_context_reset(self.ctx)

    def print_timings(self):
        llama_cpp.llama_perf_context_print(self.ctx)

    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return llama_cpp.llama_context_default_params()


class LlamaBatch:
    def __init__(
        self, *, n_tokens: int, embd: int, n_seq_max: int, verbose: bool = True
    ):
        self._n_tokens = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self._exit_stack = ExitStack()

        batch = llama_cpp.llama_batch_init(self._n_tokens, self.embd, self.n_seq_max)

        if batch is None:
            raise ValueError("Failed to create llama_batch")

        self.batch = batch
        self.sampler = None  # LlamaBatch doesn't use samplers, but some cleanup code expects this attribute

        def free_batch():
            if self.batch is None:
                return
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None

        self._exit_stack.callback(free_batch)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_tokens(self) -> int:
        return self.batch.n_tokens

    def reset(self):
        self.batch.n_tokens = 0

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
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


class LlamaTokenDataArray:
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab
        self.candidates_data = np.recarray(
            (self.n_vocab,),
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            sorted=False,
        )
        self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
        self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)
        self.sampler = None  # LlamaTokenDataArray doesn't use samplers, but some cleanup code expects this attribute

    def copy_logits(self, logits: npt.NDArray[np.single]):
        self.candidates_data.id[:] = self.default_candidates_data_id
        self.candidates_data.logit[:] = logits
        self.candidates_data.p[:] = self.default_candidates_data_p
        self.candidates.sorted = False
        self.candidates.size = self.n_vocab


# Embedding functions


def normalize_embedding(embedding):
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return [v / norm for v in embedding]


# Python wrappers over common/sampling structs


@dataclass
class LlamaSamplingParams:
    n_prev: int = 64
    n_probs: int = 0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    tfs_z: float = 1.00
    typical_p: float = 1.00
    temp: float = 0.80
    penalty_last_n: int = 64
    penalty_repeat: float = 1.0
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
class LlamaSamplingContext:
    params: LlamaSamplingParams = field(default_factory=LlamaSamplingParams)
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
        return LlamaSamplingContext(
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

    def prev_str(self, ctx_main: LlamaContext, n: int) -> str:
        return ctx_main.model.detokenize(self.prev[-n:]).decode("utf-8")

    def sample(
        self,
        ctx_main: LlamaContext,
        idx: int = 0,
        logits_array: Optional[npt.NDArray[np.single]] = None,
    ):
        # This method is deprecated in favor of using LlamaSampler directly
        raise NotImplementedError("LlamaSamplingContext.sample is deprecated, use LlamaSampler instead")

    def accept(self, ctx_main: LlamaContext, id: int, apply_grammar: bool):
        self.prev.append(id)


class CustomSampler:
    def __init__(
        self, apply_func: Callable[[llama_cpp.llama_token_data_array], None]
    ):
        self.apply_func = apply_func

        def apply_wrapper(
            sampler: llama_cpp.llama_sampler_p,
            cur_p: llama_cpp.llama_token_data_array_p,
        ):
            self.apply_func(cur_p)

        def free_wrapper(sampler: llama_cpp.llama_sampler_p):
            pass

        sampler_i = llama_cpp.llama_sampler_i()
        sampler_i.apply = llama_cpp.llama_sampler_i_apply(apply_wrapper)
        self._apply_wrapper_ref = apply_wrapper

        sampler_i.name = llama_cpp.llama_sampler_i_name(0)
        sampler_i.accept = llama_cpp.llama_sampler_i_accept(0)
        sampler_i.reset = llama_cpp.llama_sampler_i_reset(0)
        sampler_i.clone = llama_cpp.llama_sampler_i_clone(0)
        sampler_i.free = llama_cpp.llama_sampler_i_free(0)

        self.sampler = llama_cpp.llama_sampler()
        self.sampler.iface = ctypes.pointer(sampler_i)
        self.sampler.ctx = None

    def get_sampler(self) -> llama_cpp.llama_sampler_p:
        return ctypes.pointer(self.sampler)


class LlamaSampler:
    def __init__(self):
        params = llama_cpp.llama_sampler_chain_default_params()
        self.sampler = llama_cpp.llama_sampler_chain_init(params)
        self.custom_samplers: List[Tuple[int, CustomSampler]] = []
        self._exit_stack = ExitStack()

        def free_sampler():
            if self.sampler is not None:
                # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
                for i, _ in reversed(self.custom_samplers):
                    llama_cpp.llama_sampler_chain_remove(self.sampler, i)
                llama_cpp.llama_sampler_free(self.sampler)
                self.sampler = None

        self._exit_stack.callback(free_sampler)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def add_greedy(self):
        sampler = llama_cpp.llama_sampler_init_greedy()
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_dist(self, seed: int):
        sampler = llama_cpp.llama_sampler_init_dist(seed)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_softmax(self):
        sampler = llama_cpp.llama_sampler_init_softmax()
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_top_k(self, k: int):
        sampler = llama_cpp.llama_sampler_init_top_k(k)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_top_p(self, p: float, min_keep: int = 1):
        sampler = llama_cpp.llama_sampler_init_top_p(p, min_keep)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_min_p(self, p: float, min_keep: int = 1):
        sampler = llama_cpp.llama_sampler_init_min_p(p, min_keep)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_typical(self, p: float, min_keep: int = 1):
        sampler = llama_cpp.llama_sampler_init_typical(p, min_keep)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_temp(self, temp: float):
        sampler = llama_cpp.llama_sampler_init_temp(temp)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_temp_ext(self, t: float, delta: float, exponent: float):
        sampler = llama_cpp.llama_sampler_init_temp_ext(t, delta, exponent)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_xtc(self, p: float, t: float, min_keep: int, seed: int):
        sampler = llama_cpp.llama_sampler_init_xtc(p, t, min_keep, seed)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_top_n_sigma(self, n: float):
        sampler = llama_cpp.llama_sampler_init_top_n_sigma(n)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_mirostat(self, n_vocab: int, seed: int, tau: float, eta: float, m: int):
        sampler = llama_cpp.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_mirostat_v2(self, seed: int, tau: float, eta: float):
        sampler = llama_cpp.llama_sampler_init_mirostat_v2(seed, tau, eta)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_grammar(self, model: LlamaModel, grammar: LlamaGrammar):
        sampler = llama_cpp.llama_sampler_init_grammar(
            model.vocab, grammar._grammar.encode("utf-8"), grammar._root.encode("utf-8")
        )
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_grammar_lazy_patterns(
        self, 
        model: LlamaModel, 
        grammar: LlamaGrammar,
        trigger_patterns: List[str],
        trigger_tokens: List[int]
    ):
        # Convert patterns to C array
        pattern_ptrs = (ctypes.c_char_p * len(trigger_patterns))()
        for i, pattern in enumerate(trigger_patterns):
            pattern_ptrs[i] = pattern.encode("utf-8")
        
        # Convert tokens to C array
        token_array = (llama_cpp.llama_token * len(trigger_tokens))(*trigger_tokens)
        
        sampler = llama_cpp.llama_sampler_init_grammar_lazy_patterns(
            model.vocab,
            grammar._grammar.encode("utf-8"),
            grammar._root.encode("utf-8"),
            pattern_ptrs,
            len(trigger_patterns),
            token_array,
            len(trigger_tokens)
        )
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_penalties(
        self,
        penalty_last_n: int,
        penalty_repeat: float,
        penalty_freq: float,
        penalty_present: float,
    ):
        sampler = llama_cpp.llama_sampler_init_penalties(
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        )
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_dry(
        self,
        model: LlamaModel,
        n_ctx_train: int,
        dry_multiplier: float,
        dry_base: float,
        dry_allowed_length: int,
        dry_penalty_last_n: int,
        seq_breakers: List[str]
    ):
        # Convert seq_breakers to C array
        breaker_ptrs = (ctypes.c_char_p * len(seq_breakers))()
        for i, breaker in enumerate(seq_breakers):
            breaker_ptrs[i] = breaker.encode("utf-8")
        
        sampler = llama_cpp.llama_sampler_init_dry(
            model.vocab,
            n_ctx_train,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            breaker_ptrs,
            len(seq_breakers)
        )
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_logit_bias(
        self, 
        n_vocab: int, 
        logit_bias: Dict[int, float]
    ):
        # Convert logit_bias dict to C array
        bias_array = (llama_cpp.llama_logit_bias * len(logit_bias))()
        for i, (token, bias) in enumerate(logit_bias.items()):
            bias_array[i].token = token
            bias_array[i].bias = bias
        
        sampler = llama_cpp.llama_sampler_init_logit_bias(
            n_vocab,
            len(logit_bias),
            bias_array
        )
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_infill(self, model: LlamaModel):
        sampler = llama_cpp.llama_sampler_init_infill(model.vocab)
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)

    def add_custom(
        self, apply_func: Callable[[llama_cpp.llama_token_data_array], None]
    ):
        custom_sampler = CustomSampler(apply_func)
        sampler = custom_sampler.get_sampler()
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)
        # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
        self.custom_samplers.append(
            (llama_cpp.llama_sampler_chain_n(self.sampler) - 1, custom_sampler)
        )

    def get_seed(self) -> int:
        return llama_cpp.llama_sampler_get_seed(self.sampler)

    def sample(self, ctx: LlamaContext, idx: int = -1) -> int:
        return llama_cpp.llama_sampler_sample(self.sampler, ctx.ctx, idx)

    def accept(self, token: int):
        llama_cpp.llama_sampler_accept(self.sampler, token)

    def reset(self):
        llama_cpp.llama_sampler_reset(self.sampler)

    def clone(self):
        # NOTE: Custom samplers cannot be cloned due to Python callback limitations
        if self.custom_samplers:
            raise NotImplementedError("Cannot clone LlamaSampler that contains custom samplers")
        
        cloned_sampler = llama_cpp.llama_sampler_clone(self.sampler)
        # Create a new wrapper around the cloned sampler
        new_sampler = LlamaSampler.__new__(LlamaSampler)
        new_sampler.sampler = cloned_sampler
        new_sampler.custom_samplers = []
        new_sampler._exit_stack = ExitStack()
        
        def free_sampler():
            if new_sampler.sampler is not None:
                llama_cpp.llama_sampler_free(new_sampler.sampler)
                new_sampler.sampler = None

        new_sampler._exit_stack.callback(free_sampler)
        return new_sampler
