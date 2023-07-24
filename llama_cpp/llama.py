import os
import sys
import uuid
import time
import math
import multiprocessing
from abc import ABC, abstractmethod
from typing import (
    List,
    Optional,
    Union,
    Generator,
    Sequence,
    Iterator,
    Deque,
    Tuple,
    Callable,
)
from collections import deque, OrderedDict

import diskcache
import ctypes

from . import llama_cpp
from .llama_types import *

import numpy as np
import numpy.typing as npt

class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    @abstractmethod
    def __getitem__(self, key: Sequence[int]) -> "LlamaState":
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Sequence[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: Sequence[int], value: "LlamaState") -> None:
        raise NotImplementedError


class LlamaRAMCache(BaseLlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[Tuple[int, ...], "LlamaState"] = OrderedDict()

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key = None
        keys = (
            (k, Llama.longest_token_prefix(k, key)) for k in self.cache_state.keys()
        )
        for k, prefix_len in keys:
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]
        self.cache_state[key] = value
        while self.cache_size > self.capacity_bytes and len(self.cache_state) > 0:
            self.cache_state.popitem(last=False)


# Alias for backwards compatibility
LlamaCache = LlamaRAMCache


class LlamaDiskCache(BaseLlamaCache):
    """Cache for a llama.cpp model using disk."""

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        super().__init__(capacity_bytes)
        self.cache = diskcache.Cache(cache_dir)

    @property
    def cache_size(self):
        return int(self.cache.volume())  # type: ignore

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        for k in self.cache.iterkeys():  # type: ignore
            prefix_len = Llama.longest_token_prefix(k, key)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k  # type: ignore
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value: "LlamaState" = self.cache.pop(_key)  # type: ignore
        # NOTE: This puts an integer as key in cache, which breaks,
        # Llama.longest_token_prefix(k, key) above since k is not a tuple of ints/tokens
        # self.cache.push(_key, side="front")  # type: ignore
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
        key = tuple(key)
        if key in self.cache:
            print("LlamaDiskCache.__setitem__: delete", file=sys.stderr)
            del self.cache[key]
        self.cache[key] = value
        print("LlamaDiskCache.__setitem__: set", file=sys.stderr)
        while self.cache_size > self.capacity_bytes and len(self.cache) > 0:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        print("LlamaDiskCache.__setitem__: trim", file=sys.stderr)


class LlamaState:
    def __init__(
        self,
        input_ids: npt.NDArray[np.intc],
        scores: npt.NDArray[np.single],
        n_tokens: int,
        llama_state: bytes,
        llama_state_size: int,
    ):
        self.input_ids = input_ids
        self.scores = scores
        self.n_tokens = n_tokens
        self.llama_state = llama_state
        self.llama_state_size = llama_state_size


LogitsProcessor = Callable[[List[int], List[float]], List[float]]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


StoppingCriteria = Callable[[List[int], List[float]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(self, input_ids: List[int], logits: List[float]) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
        self,
        model_path: str,
        # NOTE: These parameters are likely to change in the future.
        n_ctx: int = 512,
        n_parts: int = -1,
        n_gpu_layers: int = 0,
        seed: int = 1337,
        f16_kv: bool = True,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        last_n_tokens_size: int = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        low_vram: bool = False,
        tensor_split: Optional[List[float]] = None,
        rope_freq_base: float = 10000.0,
        rope_freq_scale: float = 1.0,
        n_gqa: Optional[int] = None,  # (TEMPORARY) must be 8 for llama2 70b
        rms_norm_eps: Optional[float] = None, # (TEMPORARY)
        verbose: bool = True,
    ):
        """Load a llama.cpp model from `model_path`.

        Args:
            model_path: Path to the model.
            n_ctx: Maximum context size.
            n_parts: Number of parts to split the model into. If -1, the number of parts is automatically determined.
            seed: Random seed. -1 for random.
            f16_kv: Use half-precision for key/value cache.
            logits_all: Return logits for all tokens, not just the last token.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            tensor_split: List of floats to split the model across multiple GPUs. If None, the model is not split.
            rope_freq_base: Base frequency for rope sampling.
            rope_freq_scale: Scale factor for rope sampling.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """

        self.verbose = verbose
        self.model_path = model_path

        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = n_ctx
        self.params.n_gpu_layers = n_gpu_layers
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap if lora_path is None else False
        self.params.use_mlock = use_mlock
        self.params.embedding = embedding
        self.params.low_vram = low_vram

        self.tensor_split = tensor_split
        self._c_tensor_split = None

        if self.tensor_split is not None:
            #Type conversion and expand the list to the length of LLAMA_MAX_DEVICES
            FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES.value
            self._c_tensor_split = FloatArray(*tensor_split) # keep a reference to the array so it is not gc'd
            self.params.tensor_split = self._c_tensor_split

        self.params.rope_freq_base = rope_freq_base
        self.params.rope_freq_scale = rope_freq_scale

        if n_gqa is not None:
            self.params.n_gqa = n_gqa

        if rms_norm_eps is not None:
            self.params.rms_norm_eps = rms_norm_eps

        self.last_n_tokens_size = last_n_tokens_size
        self.n_batch = min(n_ctx, n_batch)

        self.cache: Optional[BaseLlamaCache] = None

        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)

        self.lora_base = lora_base
        self.lora_path = lora_path

        ### DEPRECATED ###
        self.n_parts = n_parts
        ### DEPRECATED ###

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.model = llama_cpp.llama_load_model_from_file(
            self.model_path.encode("utf-8"), self.params
        )
        assert self.model is not None

        self.ctx = llama_cpp.llama_new_context_with_model(self.model, self.params)

        assert self.ctx is not None

        if self.lora_path:
            if llama_cpp.llama_model_apply_lora_from_file(
                self.model,
                llama_cpp.c_char_p(self.lora_path.encode("utf-8")),
                llama_cpp.c_char_p(self.lora_base.encode("utf-8"))
                if self.lora_base is not None
                else llama_cpp.c_char_p(0),
                llama_cpp.c_int(self.n_threads),
            ):
                raise RuntimeError(
                    f"Failed to apply LoRA from lora path: {self.lora_path} to base path: {self.lora_base}"
                )

        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()
        size = llama_cpp.c_size_t(self._n_vocab)
        sorted = llama_cpp.c_bool(False)
        self._candidates_data = np.array(
            [],
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self._candidates_data.resize(3, self._n_vocab, refcheck=False)
        candidates = llama_cpp.llama_token_data_array(
            data=self._candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=size,
            sorted=sorted,
        )
        self._candidates = candidates
        self._token_nl = Llama.token_nl()
        self._token_eos = Llama.token_eos()
        self._candidates_data_id = np.arange(self._n_vocab, dtype=np.intc)  # type: ignore
        self._candidates_data_p = np.zeros(self._n_vocab, dtype=np.single)

        self.n_tokens = 0
        self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)
        self.scores: npt.NDArray[np.single] = np.ndarray(
            (n_ctx, self._n_vocab), dtype=np.single
        )

    @property
    def _input_ids(self) -> npt.NDArray[np.intc]:
        return self.input_ids[: self.n_tokens]

    @property
    def _scores(self) -> npt.NDArray[np.single]:
        return self.scores[: self.n_tokens, :]

    @property
    def eval_tokens(self) -> Deque[int]:
        return deque(self.input_ids[: self.n_tokens].tolist(), maxlen=self._n_ctx)

    @property
    def eval_logits(self) -> Deque[List[float]]:
        return deque(
            self.scores[: self.n_tokens, :].tolist(),
            maxlen=self._n_ctx if self.params.logits_all else 1,
        )

    def tokenize(self, text: bytes, add_bos: bool = True) -> List[int]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        assert self.ctx is not None
        n_ctx = self._n_ctx
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            self.ctx,
            text,
            tokens,
            llama_cpp.c_int(n_ctx),
            llama_cpp.c_bool(add_bos),
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.ctx,
                text,
                tokens,
                llama_cpp.c_int(n_tokens),
                llama_cpp.c_bool(add_bos),
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )
        return list(tokens[:n_tokens])

    def detokenize(self, tokens: List[int]) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        assert self.ctx is not None
        output = b""
        for token in tokens:
            output += llama_cpp.llama_token_to_str(
                self.ctx, llama_cpp.llama_token(token)
            )
        return output

    def set_cache(self, cache: Optional[BaseLlamaCache]):
        """Set the cache.

        Args:
            cache: The cache to set.
        """
        self.cache = cache

    def reset(self):
        """Reset the model state."""
        self.n_tokens = 0

    def eval(self, tokens: Sequence[int]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        assert self.ctx is not None
        n_ctx = self._n_ctx
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = min(n_ctx - len(batch), len(self._input_ids))
            n_tokens = len(batch)
            return_code = llama_cpp.llama_eval(
                ctx=self.ctx,
                tokens=(llama_cpp.llama_token * len(batch))(*batch),
                n_tokens=llama_cpp.c_int(n_tokens),
                n_past=llama_cpp.c_int(n_past),
                n_threads=llama_cpp.c_int(self.n_threads),
            )
            if return_code != 0:
                raise RuntimeError(f"llama_eval returned {return_code}")
            # Save tokens
            self.input_ids[self.n_tokens : self.n_tokens + n_tokens] = batch
            # Save logits
            rows = n_tokens if self.params.logits_all else 1
            cols = self._n_vocab
            offset = (
                0 if self.params.logits_all else n_tokens - 1
            )  # NOTE: Only save the last token logits if logits_all is False
            self.scores[self.n_tokens + offset : self.n_tokens + n_tokens, :].reshape(
                -1
            )[:] = llama_cpp.llama_get_logits(self.ctx)[: rows * cols]
            # Update n_tokens
            self.n_tokens += n_tokens

    def _sample(
        self,
        last_n_tokens_data,  # type: llama_cpp.Array[llama_cpp.llama_token]
        last_n_tokens_size: llama_cpp.c_int,
        top_k: llama_cpp.c_int,
        top_p: llama_cpp.c_float,
        temp: llama_cpp.c_float,
        tfs_z: llama_cpp.c_float,
        repeat_penalty: llama_cpp.c_float,
        frequency_penalty: llama_cpp.c_float,
        presence_penalty: llama_cpp.c_float,
        mirostat_mode: llama_cpp.c_int,
        mirostat_tau: llama_cpp.c_float,
        mirostat_eta: llama_cpp.c_float,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
    ):
        assert self.ctx is not None
        assert self.n_tokens > 0
        n_vocab = self._n_vocab
        n_ctx = self._n_ctx
        top_k = llama_cpp.c_int(n_vocab) if top_k.value <= 0 else top_k
        last_n_tokens_size = (
            llama_cpp.c_int(n_ctx)
            if last_n_tokens_size.value < 0
            else last_n_tokens_size
        )
        logits: npt.NDArray[np.single] = self._scores[-1, :]

        if logits_processor is not None:
            logits = np.array(
                logits_processor(self._input_ids.tolist(), logits.tolist()),
                dtype=np.single,
            )
            self._scores[-1, :] = logits

        nl_logit = logits[self._token_nl]
        candidates = self._candidates
        candidates_data = self._candidates_data
        candidates_data["id"][:] = self._candidates_data_id  # type: ignore
        candidates_data["logit"][:] = logits
        candidates_data["p"][:] = self._candidates_data_p  # type: ignore
        candidates.data = candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p)
        candidates.sorted = llama_cpp.c_bool(False)
        candidates.size = llama_cpp.c_size_t(n_vocab)
        llama_cpp.llama_sample_repetition_penalty(
            ctx=self.ctx,
            last_tokens_data=last_n_tokens_data,
            last_tokens_size=last_n_tokens_size,
            candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            penalty=repeat_penalty,
        )
        llama_cpp.llama_sample_frequency_and_presence_penalties(
            ctx=self.ctx,
            candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            last_tokens_data=last_n_tokens_data,
            last_tokens_size=last_n_tokens_size,
            alpha_frequency=frequency_penalty,
            alpha_presence=presence_penalty,
        )
        if not penalize_nl:
            candidates.data[self._token_nl].logit = llama_cpp.c_float(nl_logit)
        if temp.value == 0.0:
            return llama_cpp.llama_sample_token_greedy(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            )
        elif mirostat_mode.value == 1:
            mirostat_mu = llama_cpp.c_float(2.0 * mirostat_tau.value)
            mirostat_m = llama_cpp.c_int(100)
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                temp=temp,
            )
            return llama_cpp.llama_sample_token_mirostat(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=llama_cpp.ctypes.byref(mirostat_mu),  # type: ignore
                m=mirostat_m,
            )
        elif mirostat_mode.value == 2:
            mirostat_mu = llama_cpp.c_float(2.0 * mirostat_tau.value)
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                temp=temp,
            )
            return llama_cpp.llama_sample_token_mirostat_v2(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=llama_cpp.ctypes.byref(mirostat_mu),  # type: ignore
            )
        else:
            llama_cpp.llama_sample_top_k(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                k=top_k,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_tail_free(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                z=tfs_z,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_typical(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                p=llama_cpp.c_float(1.0),
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_top_p(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                p=top_p,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                temp=temp,
            )
            return llama_cpp.llama_sample_token(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            )

    def sample(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.80,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
    ):
        """Sample a token from the model.

        Args:
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.

        Returns:
            The sampled token.
        """
        assert self.ctx is not None
        last_n_tokens_data = [llama_cpp.llama_token(0)] * max(
            0, self.last_n_tokens_size - len(self._input_ids)
        ) + self._input_ids[-self.last_n_tokens_size :].tolist()
        return self._sample(
            last_n_tokens_data=(llama_cpp.llama_token * self.last_n_tokens_size)(
                *last_n_tokens_data
            ),
            last_n_tokens_size=llama_cpp.c_int(self.last_n_tokens_size),
            top_k=llama_cpp.c_int(top_k),
            top_p=llama_cpp.c_float(top_p),
            temp=llama_cpp.c_float(temp),
            tfs_z=llama_cpp.c_float(tfs_z),
            repeat_penalty=llama_cpp.c_float(repeat_penalty),
            frequency_penalty=llama_cpp.c_float(frequency_penalty),
            presence_penalty=llama_cpp.c_float(presence_penalty),
            mirostat_mode=llama_cpp.c_int(mirostat_mode),
            mirostat_tau=llama_cpp.c_float(mirostat_tau),
            mirostat_eta=llama_cpp.c_float(mirostat_eta),
            penalize_nl=penalize_nl,
            logits_processor=logits_processor,
        )

    def generate(
        self,
        tokens: Sequence[int],
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.80,
        repeat_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        assert self.ctx is not None

        if reset and len(self._input_ids) > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                if self.verbose:
                    print("Llama.generate: prefix-match hit", file=sys.stderr)
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix

        if reset:
            self.reset()

        while True:
            self.eval(tokens)
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                logits_processor=logits_processor,
            )
            if stopping_criteria is not None and stopping_criteria(
                self._input_ids.tolist(), self._scores[-1, :].tolist()
            ):
                return
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

    def create_embedding(
        self, input: Union[str, List[str]], model: Optional[str] = None
    ) -> Embedding:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            An embedding object.
        """
        assert self.ctx is not None
        model_name: str = model if model is not None else self.model_path

        if self.params.embedding == False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        data: List[EmbeddingData] = []
        total_tokens = 0
        for index, input in enumerate(inputs):
            tokens = self.tokenize(input.encode("utf-8"))
            self.reset()
            self.eval(tokens)
            n_tokens = len(tokens)
            total_tokens += n_tokens
            embedding = llama_cpp.llama_get_embeddings(self.ctx)[
                : llama_cpp.llama_n_embd(self.ctx)
            ]

            data.append(
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": index,
                }
            )
        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    def embed(self, input: str) -> List[float]:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            A list of embeddings
        """
        return list(map(float, self.create_embedding(input)["data"][0]["embedding"]))

    def _create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:
        assert self.ctx is not None

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        completion_tokens: List[int] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[int] = self.tokenize(b" " + prompt.encode("utf-8"))
        text: bytes = b""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else self.model_path

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        if len(prompt_tokens) >= llama_cpp.llama_n_ctx(self.ctx):
            raise ValueError(
                f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if max_tokens <= 0:
            # Unlimited, depending on n_ctx.
            max_tokens = llama_cpp.llama_n_ctx(self.ctx) - len(prompt_tokens)

        # Truncate max_tokens if requested tokens would exceed the context window
        max_tokens = (
            max_tokens
            if max_tokens + len(prompt_tokens) < self._n_ctx
            else (self._n_ctx - len(prompt_tokens))
        )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self.params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache:
            try:
                cache_item = self.cache[prompt_tokens]
                cache_prefix_len = Llama.longest_token_prefix(
                    cache_item.input_ids.tolist(), prompt_tokens
                )
                eval_prefix_len = Llama.longest_token_prefix(
                    self._input_ids.tolist(), prompt_tokens
                )
                if cache_prefix_len > eval_prefix_len:
                    self.load_state(cache_item)
                    if self.verbose:
                        print("Llama._create_completion: cache hit", file=sys.stderr)
            except KeyError:
                if self.verbose:
                    print("Llama._create_completion: cache miss", file=sys.stderr)

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        ):
            if token == self._token_eos:
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens)

            # Contains multi-byte UTF8
            for k, char in enumerate(all_text[-3:]):
                k = 3 - k
                for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                    # Bitwise AND check
                    if num > k and pattern & char == pattern:
                        multibyte_fix = num - k

            # Stop incomplete bytes from passing
            if multibyte_fix > 0:
                multibyte_fix -= 1
                continue

            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = self.detokenize(remaining_tokens)
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0
                for token in remaining_tokens:
                    token_end_position += len(self.detokenize([token]))
                    # Check if stop sequence is in the token
                    if token_end_position >= (
                        remaining_length - first_stop_position
                    ):
                        break
                    logprobs_or_none: Optional[CompletionLogprobs] = None
                    if logprobs is not None:
                        token_str = self.detokenize([token]).decode(
                            "utf-8", errors="ignore"
                        )
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens])
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :].tolist()
                        current_logprobs = Llama.logits_to_logprobs(logits)
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]).decode(
                                "utf-8", errors="ignore"
                            ): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob.update({token_str: current_logprobs[int(token)]})
                        logprobs_or_none = {
                            "tokens": [
                                self.detokenize([token]).decode(
                                    "utf-8", errors="ignore"
                                )
                            ],
                            "text_offset": [text_offset],
                            "token_logprobs": [current_logprobs[int(token)]],
                            "top_logprobs": [top_logprob],
                        }
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": self.detokenize([token]).decode(
                                    "utf-8", errors="ignore"
                                ),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if stopping_criteria is not None and stopping_criteria(
            self._input_ids.tolist(), self._scores[-1, :].tolist()
        ):
            text = self.detokenize(completion_tokens)
            finish_reason = "stop"

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            all_text = self.detokenize(remaining_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                end = min(all_text.index(stop) for stop in any_stop)
            else:
                end = len(all_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(self.detokenize([token]))

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    token_str = self.detokenize([token]).decode(
                        "utf-8", errors="ignore"
                    )
                    text_offset = len(prompt) + len(
                        self.detokenize(completion_tokens[:returned_tokens])
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :].tolist()
                    current_logprobs = Llama.logits_to_logprobs(logits)
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob.update({token_str: current_logprobs[int(token)]})
                    logprobs_or_none = {
                        "tokens": [
                            self.detokenize([token]).decode("utf-8", errors="ignore")
                        ],
                        "text_offset": [text_offset],
                        "token_logprobs": [current_logprobs[int(token)]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = self.detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ].decode("utf-8", errors="ignore"),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": self.detokenize([token]).decode(
                                "utf-8", errors="ignore"
                            ),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": None,
                        }
                    ],
                }
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            if self.cache:
                if self.verbose:
                    print("Llama._create_completion: cache save", file=sys.stderr)
                self.cache[prompt_tokens + completion_tokens] = self.save_state()
                print("Llama._create_completion: cache saved", file=sys.stderr)
            return

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token
                all_tokens = prompt_tokens[1:] + completion_tokens
            else:
                all_tokens = completion_tokens

            all_token_strs = [
                self.detokenize([token]).decode("utf-8", errors="ignore")
                for token in all_tokens
            ]
            all_logprobs = [
                Llama.logits_to_logprobs(row.tolist()) for row in self._scores
            ][token_offset:]
            for token, token_str, logprobs_token in zip(
                all_tokens, all_token_strs, all_logprobs
            ):
                text_offsets.append(text_offset)
                text_offset += len(token_str)
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(logprobs_token[int(token)])
                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)
            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        return self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )

    def _convert_text_completion_to_chat(
        self, completion: Completion
    ) -> ChatCompletion:
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"],
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }

    def _convert_text_completion_chunks_to_chat(
        self,
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield {
                    "id": "chat" + chunk["id"],
                    "model": chunk["model"],
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {},
                        "finish_reason": chunk["choices"][0]["finish_reason"],
                    }
                ],
            }

    def create_chat_completion(
        self,
        messages: List[ChatCompletionMessage],
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        max_tokens: int = 256,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Generate a chat completion from a list of messages.

        Args:
            messages: A list of messages to generate a response for.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.
            stop: A list of strings to stop generation when encountered.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            repeat_penalty: The penalty to apply to repeated tokens.

        Returns:
            Generated chat completion or a stream of chat completion chunks.
        """
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        chat_history = "".join(
            f'### {"Human" if message["role"] == "user" else "Assistant"}:{message["content"]}'
            for message in messages
        )
        PROMPT = chat_history + "### Assistant:"
        PROMPT_STOP = ["### Assistant:", "### Human:"]
        completion_or_chunks = self(
            prompt=PROMPT,
            stop=PROMPT_STOP + stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks  # type: ignore
            return self._convert_text_completion_chunks_to_chat(chunks)
        else:
            completion: Completion = completion_or_chunks  # type: ignore
            return self._convert_text_completion_to_chat(completion)

    def __del__(self):
        if self.model is not None:
            llama_cpp.llama_free_model(self.model)
            self.model = None
        if self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

    def __getstate__(self):
        return dict(
            verbose=self.verbose,
            model_path=self.model_path,
            n_ctx=self.params.n_ctx,
            n_gpu_layers=self.params.n_gpu_layers,
            seed=self.params.seed,
            f16_kv=self.params.f16_kv,
            logits_all=self.params.logits_all,
            vocab_only=self.params.vocab_only,
            use_mmap=self.params.use_mmap,
            use_mlock=self.params.use_mlock,
            embedding=self.params.embedding,
            low_vram=self.params.low_vram,
            last_n_tokens_size=self.last_n_tokens_size,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            lora_base=self.lora_base,
            lora_path=self.lora_path,
            tensor_split=self.tensor_split,
            ### TEMPORARY ###
            n_gqa=self.params.n_gqa,
            rms_norm_eps=self.params.rms_norm_eps,
            ### TEMPORARY ###
            ### DEPRECATED ###
            n_parts=self.n_parts,
            ### DEPRECATED ###
        )

    def __setstate__(self, state):
        self.__init__(
            model_path=state["model_path"],
            n_ctx=state["n_ctx"],
            n_gpu_layers=state["n_gpu_layers"],
            seed=state["seed"],
            f16_kv=state["f16_kv"],
            logits_all=state["logits_all"],
            vocab_only=state["vocab_only"],
            use_mmap=state["use_mmap"],
            use_mlock=state["use_mlock"],
            embedding=state["embedding"],
            low_vram=state["low_vram"],
            n_threads=state["n_threads"],
            n_batch=state["n_batch"],
            last_n_tokens_size=state["last_n_tokens_size"],
            lora_base=state["lora_base"],
            lora_path=state["lora_path"],
            tensor_split=state["tensor_split"],
            verbose=state["verbose"],
            ### TEMPORARY ###
            n_gqa=state["n_gqa"],
            rms_norm_eps=state["rms_norm_eps"],
            ### TEMPORARY ###
            ### DEPRECATED ###
            n_parts=state["n_parts"],
            ### DEPRECATED ###
        )

    def save_state(self) -> LlamaState:
        assert self.ctx is not None
        if self.verbose:
            print("Llama.save_state: saving llama state", file=sys.stderr)
        state_size = llama_cpp.llama_get_state_size(self.ctx)
        if self.verbose:
            print(f"Llama.save_state: got state size: {state_size}", file=sys.stderr)
        llama_state = (llama_cpp.c_uint8 * int(state_size))()
        if self.verbose:
            print("Llama.save_state: allocated state", file=sys.stderr)
        n_bytes = llama_cpp.llama_copy_state_data(self.ctx, llama_state)
        if self.verbose:
            print(f"Llama.save_state: copied llama state: {n_bytes}", file=sys.stderr)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        llama_state_compact = (llama_cpp.c_uint8 * int(n_bytes))()
        llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))
        if self.verbose:
            print(
                f"Llama.save_state: saving {n_bytes} bytes of llama state",
                file=sys.stderr,
            )
        return LlamaState(
            scores=self.scores.copy(),
            input_ids=self.input_ids.copy(),
            n_tokens=self.n_tokens,
            llama_state=bytes(llama_state_compact),
            llama_state_size=n_bytes,
        )

    def load_state(self, state: LlamaState) -> None:
        assert self.ctx is not None
        self.scores = state.scores.copy()
        self.input_ids = state.input_ids.copy()
        self.n_tokens = state.n_tokens
        state_size = state.llama_state_size
        LLamaStateArrayType = llama_cpp.c_uint8 * state_size
        llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)

        if llama_cpp.llama_set_state_data(self.ctx, llama_state) != state_size:
            raise RuntimeError("Failed to set llama state data")

    def n_ctx(self) -> int:
        """Return the context window size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_ctx(self.ctx)

    def n_embd(self) -> int:
        """Return the embedding size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_embd(self.ctx)

    def n_vocab(self) -> int:
        """Return the vocabulary size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_vocab(self.ctx)

    def tokenizer(self) -> "LlamaTokenizer":
        """Return the tokenizer for this model."""
        assert self.ctx is not None
        return LlamaTokenizer(self)

    @staticmethod
    def token_eos() -> int:
        """Return the end-of-sequence token."""
        return llama_cpp.llama_token_eos()

    @staticmethod
    def token_bos() -> int:
        """Return the beginning-of-sequence token."""
        return llama_cpp.llama_token_bos()

    @staticmethod
    def token_nl() -> int:
        """Return the newline token."""
        return llama_cpp.llama_token_nl()

    @staticmethod
    def logits_to_logprobs(logits: List[float]) -> List[float]:
        exps = [math.exp(float(x)) for x in logits]
        sum_exps = sum(exps)
        return [math.log(x / sum_exps) for x in exps]

    @staticmethod
    def longest_token_prefix(a: Sequence[int], b: Sequence[int]):
        longest_prefix = 0
        for _a, _b in zip(a, b):
            if _a == _b:
                longest_prefix += 1
            else:
                break
        return longest_prefix


class LlamaTokenizer:
    def __init__(self, llama: Llama):
        self.llama = llama

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        return self.llama.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos
        )

    def decode(self, tokens: List[int]) -> str:
        return self.llama.detokenize(tokens).decode("utf-8", errors="ignore")

    @classmethod
    def from_ggml_file(cls, path: str) -> "LlamaTokenizer":
        return cls(Llama(model_path=path, vocab_only=True))
