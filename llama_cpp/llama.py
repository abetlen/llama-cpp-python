from __future__ import annotations

import os
import sys
import uuid
import time
import json
import ctypes
import typing
import random
import fnmatch
import warnings
import contextlib
import multiprocessing

from typing import (
    Any,
    List,
    Literal,
    Optional,
    Union,
    Generator,
    Sequence,
    Iterator,
    Deque,
    Callable,
    Dict,
)
from collections import deque
from pathlib import Path


from .llama_types import *
from .llama_grammar import LlamaGrammar
from .llama_cache import (
    BaseLlamaCache,
    LlamaCache,  # type: ignore
    LlamaDiskCache,  # type: ignore
    LlamaRAMCache,  # type: ignore
)
from .llama_tokenizer import BaseLlamaTokenizer, LlamaTokenizer
import llama_cpp.llama_cpp as llama_cpp
import llama_cpp.llama_chat_format as llama_chat_format

from llama_cpp.llama_speculative import LlamaDraftModel

import numpy as np
import numpy.typing as npt

import llama_cpp._internals as internals
from ._logger import set_verbose
from ._utils import suppress_stdout_stderr


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    __backend_initialized = False

    def __init__(
        self,
        model_path: str,
        *,
        # Model Params
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,
        # Context Params
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[
            int
        ] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        yarn_ext_factor: float = -1.0,
        yarn_attn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_orig_ctx: int = 0,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        op_offload: Optional[bool] = None,
        swa_full: Optional[bool] = None,
        # Sampling Params
        no_perf: bool = False,
        last_n_tokens_size: int = 64,
        # LoRA Params
        lora_base: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_path: Optional[str] = None,
        # Backend Params
        numa: Union[bool, int] = False,
        # Chat Format Params
        chat_format: Optional[str] = None,
        chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
        # Speculative Decoding
        draft_model: Optional[LlamaDraftModel] = None,
        # Tokenizer Override
        tokenizer: Optional[BaseLlamaTokenizer] = None,
        # KV cache quantization
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        # Misc
        spm_infill: bool = False,
        verbose: bool = True,
        # Extra Params
        **kwargs,  # type: ignore
    ):
        """Load a llama.cpp model from `model_path`.

        Examples:
            Basic usage

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ... )
            >>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])
            the lazy dog

            Loading a chat model

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ...     chat_format="llama-2",
            ... )
            >>> print(model.create_chat_completion(
            ...     messages=[{
            ...         "role": "user",
            ...         "content": "what is the meaning of life?"
            ...     }]
            ... ))

        Args:
            model_path: Path to the model.
            n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            split_mode: How to split the model across GPUs. See llama_cpp.LLAMA_SPLIT_* for options.
            main_gpu: main_gpu interpretation depends on split_mode: LLAMA_SPLIT_MODE_NONE: the GPU that is used for the entire model. LLAMA_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results. LLAMA_SPLIT_MODE_LAYER: ignored
            tensor_split: How split tensors should be distributed across GPUs. If None, the model is not split.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            kv_overrides: Key-value overrides for the model.
            seed: RNG seed, -1 for random
            n_ctx: Text context, 0 = from model
            n_batch: Prompt processing maximum batch size
            n_ubatch: Physical batch size
            n_threads: Number of threads to use for generation
            n_threads_batch: Number of threads to use for batch processing
            rope_scaling_type: RoPE scaling type, from `enum llama_rope_scaling_type`. ref: https://github.com/ggerganov/llama.cpp/pull/2054
            pooling_type: Pooling type, from `enum llama_pooling_type`.
            rope_freq_base: RoPE base frequency, 0 = from model
            rope_freq_scale: RoPE frequency scaling factor, 0 = from model
            yarn_ext_factor: YaRN extrapolation mix factor, negative = from model
            yarn_attn_factor: YaRN magnitude scaling factor
            yarn_beta_fast: YaRN low correction dim
            yarn_beta_slow: YaRN high correction dim
            yarn_orig_ctx: YaRN original context size
            logits_all: Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.
            embedding: Embedding mode only.
            offload_kqv: Offload K, Q, V to GPU.
            flash_attn: Use flash attention.
            op_offload: offload host tensor operations to device
            swa_full: use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
            no_perf: Measure performance timings.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            numa: numa policy
            chat_format: String specifying the chat format to use when calling create_chat_completion.
            chat_handler: Optional chat handler to use when calling create_chat_completion.
            draft_model: Optional draft model to use for speculative decoding.
            tokenizer: Optional tokenizer to override the default tokenizer from llama.cpp.
            verbose: Print verbose output to stderr.
            type_k: KV cache data type for K (default: f16)
            type_v: KV cache data type for V (default: f16)
            spm_infill: Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.verbose = verbose
        self._stack = contextlib.ExitStack()

        set_verbose(verbose)

        if not Llama.__backend_initialized:
            with suppress_stdout_stderr(disable=verbose):
                llama_cpp.llama_backend_init()
            Llama.__backend_initialized = True

        if isinstance(numa, bool):
            self.numa = (
                llama_cpp.GGML_NUMA_STRATEGY_DISTRIBUTE
                if numa
                else llama_cpp.GGML_NUMA_STRATEGY_DISABLED
            )
        else:
            self.numa = numa

        if self.numa != llama_cpp.GGML_NUMA_STRATEGY_DISABLED:
            with suppress_stdout_stderr(disable=verbose):
                llama_cpp.llama_numa_init(self.numa)

        self.model_path = model_path

        # Model Params
        self.model_params = llama_cpp.llama_model_default_params()
        self.model_params.n_gpu_layers = (
            0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers
        )  # 0x7FFFFFFF is INT32 max, will be auto set to all layers
        self.model_params.split_mode = split_mode
        self.model_params.main_gpu = main_gpu
        self.tensor_split = tensor_split
        self._c_tensor_split = None
        if self.tensor_split is not None:
            if len(self.tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:
                raise ValueError(
                    f"Attempt to split tensors that exceed maximum supported devices. Current LLAMA_MAX_DEVICES={llama_cpp.LLAMA_MAX_DEVICES}"
                )
            # Type conversion and expand the list to the length of LLAMA_MAX_DEVICES
            FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES
            self._c_tensor_split = FloatArray(
                *tensor_split  # type: ignore
            )  # keep a reference to the array so it is not gc'd
            self.model_params.tensor_split = self._c_tensor_split
        self.model_params.vocab_only = vocab_only
        self.model_params.use_mmap = use_mmap if lora_path is None else False
        self.model_params.use_mlock = use_mlock

        # kv_overrides is the original python dict
        self.kv_overrides = kv_overrides
        if kv_overrides is not None:
            # _kv_overrides_array is a ctypes.Array of llama_model_kv_override Structs
            kvo_array_len = len(kv_overrides) + 1  # for sentinel element
            self._kv_overrides_array = (
                llama_cpp.llama_model_kv_override * kvo_array_len
            )()

            for i, (k, v) in enumerate(kv_overrides.items()):
                self._kv_overrides_array[i].key = k.encode("utf-8")
                if isinstance(v, bool):
                    self._kv_overrides_array[
                        i
                    ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL
                    self._kv_overrides_array[i].value.val_bool = v
                elif isinstance(v, int):
                    self._kv_overrides_array[
                        i
                    ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT
                    self._kv_overrides_array[i].value.val_i64 = v
                elif isinstance(v, float):
                    self._kv_overrides_array[
                        i
                    ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT
                    self._kv_overrides_array[i].value.val_f64 = v
                elif isinstance(v, str):  # type: ignore
                    v_bytes = v.encode("utf-8")
                    if len(v_bytes) > 128:  # TODO: Make this a constant
                        raise ValueError(f"Value for {k} is too long: {v}")
                    v_bytes = v_bytes.ljust(128, b"\0")
                    self._kv_overrides_array[
                        i
                    ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR
                    # copy min(v_bytes, 128) to str_value
                    address = typing.cast(
                        int,
                        ctypes.addressof(self._kv_overrides_array[i].value)
                        + llama_cpp.llama_model_kv_override_value.val_str.offset,
                    )
                    buffer_start = ctypes.cast(address, ctypes.POINTER(ctypes.c_char))
                    ctypes.memmove(
                        buffer_start,
                        v_bytes,
                        128,
                    )
                else:
                    raise ValueError(f"Unknown value type for {k}: {v}")

            self._kv_overrides_array[
                -1
            ].key = b"\0"  # ensure sentinel element is zeroed
            self.model_params.kv_overrides = self._kv_overrides_array

        self.n_batch = min(n_ctx, n_batch)  # ???
        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)
        self.n_threads_batch = n_threads_batch or multiprocessing.cpu_count()

        # Used by the sampler
        self._seed = seed or llama_cpp.LLAMA_DEFAULT_SEED

        # Context Params
        self.context_params = llama_cpp.llama_context_default_params()
        self.context_params.n_ctx = n_ctx
        self.context_params.n_batch = self.n_batch
        self.context_params.n_ubatch = min(self.n_batch, n_ubatch)
        self.context_params.n_threads = self.n_threads
        self.context_params.n_threads_batch = self.n_threads_batch
        self.context_params.rope_scaling_type = (
            rope_scaling_type
            if rope_scaling_type is not None
            else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
        )
        self.context_params.pooling_type = pooling_type
        self.context_params.rope_freq_base = (
            rope_freq_base if rope_freq_base != 0.0 else 0
        )
        self.context_params.rope_freq_scale = (
            rope_freq_scale if rope_freq_scale != 0.0 else 0
        )
        self.context_params.yarn_ext_factor = (
            yarn_ext_factor if yarn_ext_factor != 0.0 else 0
        )
        self.context_params.yarn_attn_factor = (
            yarn_attn_factor if yarn_attn_factor != 0.0 else 0
        )
        self.context_params.yarn_beta_fast = (
            yarn_beta_fast if yarn_beta_fast != 0.0 else 0
        )
        self.context_params.yarn_beta_slow = (
            yarn_beta_slow if yarn_beta_slow != 0.0 else 0
        )
        self.context_params.yarn_orig_ctx = yarn_orig_ctx if yarn_orig_ctx != 0 else 0
        self._logits_all = logits_all if draft_model is None else True
        self.context_params.embeddings = embedding  # TODO: Rename to embeddings
        self.context_params.offload_kqv = offload_kqv
        self.context_params.flash_attn = flash_attn

        if op_offload is not None:
            self.context_params.op_offload = op_offload

        if swa_full is not None:
            self.context_params.swa_full = swa_full

        #  KV cache quantization
        if type_k is not None:
            self.context_params.type_k = type_k
        if type_v is not None:
            self.context_params.type_v = type_v
        # Sampling Params
        self.context_params.no_perf = no_perf
        self.last_n_tokens_size = last_n_tokens_size

        self.cache: Optional[BaseLlamaCache] = None

        self.lora_base = lora_base
        self.lora_scale = lora_scale
        self.lora_path = lora_path

        self.spm_infill = spm_infill

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self._model = self._stack.enter_context(
            contextlib.closing(
                internals.LlamaModel(
                    path_model=self.model_path,
                    params=self.model_params,
                    verbose=self.verbose,
                )
            )
        )

        # Override tokenizer
        self.tokenizer_ = tokenizer or LlamaTokenizer(self)

        # Set the default value for the context and correct the batch
        if n_ctx == 0:
            n_ctx = self._model.n_ctx_train()
            self.n_batch = min(n_ctx, n_batch)
            self.context_params.n_ctx = self._model.n_ctx_train()
            self.context_params.n_batch = self.n_batch
            self.context_params.n_ubatch = min(self.n_batch, n_ubatch)

        self._ctx = self._stack.enter_context(
            contextlib.closing(
                internals.LlamaContext(
                    model=self._model,
                    params=self.context_params,
                    verbose=self.verbose,
                )
            )
        )

        self._batch = self._stack.enter_context(
            contextlib.closing(
                internals.LlamaBatch(
                    n_tokens=self.n_batch,
                    embd=0,
                    n_seq_max=self.context_params.n_ctx,
                    verbose=self.verbose,
                )
            )
        )

        self._lora_adapter: Optional[llama_cpp.llama_adapter_lora_p] = None

        if self.lora_path:
            self._lora_adapter = llama_cpp.llama_adapter_lora_init(
                self._model.model,
                self.lora_path.encode("utf-8"),
            )
            if self._lora_adapter is None:
                raise RuntimeError(
                    f"Failed to initialize LoRA adapter from lora path: {self.lora_path}"
                )

            def free_lora_adapter():
                if self._lora_adapter is None:
                    return
                llama_cpp.llama_adapter_lora_free(self._lora_adapter)
                self._lora_adapter = None

            self._stack.callback(free_lora_adapter)

            if llama_cpp.llama_set_adapter_lora(
                self._ctx.ctx, self._lora_adapter, self.lora_scale
            ):
                raise RuntimeError(
                    f"Failed to set LoRA adapter from lora path: {self.lora_path}"
                )

        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

        self.chat_format = chat_format
        self.chat_handler = chat_handler
        self._chat_handlers: Dict[
            str, llama_chat_format.LlamaChatCompletionHandler
        ] = {}

        self.draft_model = draft_model

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()

        self._token_nl = self.token_nl()
        self._token_eos = self.token_eos()

        self._candidates = internals.LlamaTokenDataArray(n_vocab=self._n_vocab)

        self.n_tokens = 0
        self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)
        self.scores: npt.NDArray[np.single] = np.ndarray(
            (n_ctx if logits_all == True else n_batch, self._n_vocab), dtype=np.single
        )

        self._mirostat_mu = ctypes.c_float(
            2.0 * 5.0
        )  # TODO: Move this to sampling context

        try:
            self.metadata = self._model.metadata()
        except Exception as e:
            self.metadata = {}
            if self.verbose:
                print(f"Failed to load metadata: {e}", file=sys.stderr)

        if self.verbose:
            print(f"Model metadata: {self.metadata}", file=sys.stderr)

        eos_token_id = self.token_eos()
        bos_token_id = self.token_bos()

        eos_token = (
            self._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
        )
        bos_token = (
            self._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
        )

        # Unfortunately the llama.cpp API does not return metadata arrays, so we can't get template names from tokenizer.chat_templates
        template_choices = dict(
            (name[10:], template)
            for name, template in self.metadata.items()
            if name.startswith("tokenizer.chat_template.")
        )

        if "tokenizer.chat_template" in self.metadata:
            template_choices["chat_template.default"] = self.metadata[
                "tokenizer.chat_template"
            ]

        if self.verbose and template_choices:
            print(
                f"Available chat formats from metadata: {', '.join(template_choices.keys())}",
                file=sys.stderr,
            )

        for name, template in template_choices.items():
            self._chat_handlers[name] = llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
                stop_token_ids=[eos_token_id],
            ).to_chat_handler()

        if (
            self.chat_format is None
            and self.chat_handler is None
            and "chat_template.default" in template_choices
        ):
            chat_format = llama_chat_format.guess_chat_format_from_gguf_metadata(
                self.metadata
            )

            if chat_format is not None:
                self.chat_format = chat_format
                if self.verbose:
                    print(f"Guessed chat format: {chat_format}", file=sys.stderr)
            else:
                if self.verbose:
                    print(
                        f"Using gguf chat template: {template_choices['chat_template.default']}",
                        file=sys.stderr,
                    )
                    print(f"Using chat eos_token: {eos_token}", file=sys.stderr)
                    print(f"Using chat bos_token: {bos_token}", file=sys.stderr)

                self.chat_format = "chat_template.default"

        if self.chat_format is None and self.chat_handler is None:
            self.chat_format = "llama-2"
            if self.verbose:
                print(
                    f"Using fallback chat format: {self.chat_format}", file=sys.stderr
                )

        self._sampler = None

    @property
    def ctx(self) -> llama_cpp.llama_context_p:
        return self._ctx.ctx

    @property
    def model(self) -> llama_cpp.llama_model_p:
        return self._model.model

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
            maxlen=self._n_ctx if self._logits_all else 1,
        )

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = False
    ) -> List[int]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.
            add_bos: Whether to add a beginning of sequence token.
            special: Whether to tokenize special tokens.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        return self.tokenizer_.tokenize(text, add_bos, special)

    def detokenize(
        self,
        tokens: List[int],
        prev_tokens: Optional[List[int]] = None,
        special: bool = False,
    ) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.
            prev_tokens: The list of previous tokens. Offset mapping will be performed if provided.
            special: Whether to detokenize special tokens.

        Returns:
            The detokenized string.
        """
        return self.tokenizer_.detokenize(
            tokens, prev_tokens=prev_tokens, special=special
        )

    def set_cache(self, cache: Optional[BaseLlamaCache]):
        """Set the cache.

        Args:
            cache: The cache to set.
        """
        self.cache = cache

    def set_seed(self, seed: int):
        """Set the random seed.

        Args:
            seed: The random seed.
        """
        self._seed = seed

    def reset(self):
        """Reset the model state."""
        self.n_tokens = 0

    def eval(self, tokens: Sequence[int]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = self.n_tokens
            n_tokens = len(batch)
            self._batch.set_batch(
                batch=batch, n_past=n_past, logits_all=self._logits_all
            )
            self._ctx.decode(self._batch)
            # Save tokens
            self.input_ids[n_past : n_past + n_tokens] = batch
            # Save logits
            if self._logits_all:
                rows = n_tokens
                cols = self._n_vocab
                logits = np.ctypeslib.as_array(
                    self._ctx.get_logits(), shape=(rows * cols,)
                )
                self.scores[n_past : n_past + n_tokens, :].reshape(-1)[::] = logits
            else:
                # rows = 1
                # cols = self._n_vocab
                # logits = np.ctypeslib.as_array(
                #     self._ctx.get_logits(), shape=(rows * cols,)
                # )
                # self.scores[n_past + n_tokens - 1, :].reshape(-1)[::] = logits
                # NOTE: Now that sampling is done inside the sampler, logits are only needed for logprobs which requires logits_all
                pass
            # Update n_tokens
            self.n_tokens += n_tokens

    def _init_sampler(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ):
        sampler = internals.LlamaSampler()

        if logits_processor is not None:
            # Create and add a custom sampler
            def apply_func(token_data_array: llama_cpp.llama_token_data_array_p):
                size = token_data_array.contents.size
                data_soa = token_data_array.contents.data
                data_soa_address = ctypes.addressof(data_soa.contents)
                # NOTE: This is probably broken
                recarray = np.recarray(
                    shape=(size,),
                    dtype=np.dtype(
                        [("id", np.intc), ("logit", np.single), ("p", np.single)],
                        align=True,
                    ),
                    buf=(llama_cpp.llama_token_data * size).from_address(
                        data_soa_address
                    ),
                )
                for logit_processor in logits_processor:
                    recarray.logit[:] = logit_processor(self._input_ids, recarray.logit)

            sampler.add_custom(apply_func)

        sampler.add_penalties(
            # n_vocab=self._n_vocab,
            # special_eos_id=self._token_eos,
            # linefeed_id=self._token_nl,
            penalty_last_n=self.last_n_tokens_size,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty,
            # penalize_nl=penalize_nl,
            # ignore_eos=False,
        )

        if grammar is not None:
            sampler.add_grammar(self._model, grammar)

        if temp < 0.0:
            sampler.add_softmax()
            sampler.add_dist(self._seed)
        elif temp == 0.0:
            sampler.add_greedy()
        else:
            if mirostat_mode == 1:
                mirostat_m = 100
                sampler.add_mirostat(
                    self._n_vocab,
                    self._seed,
                    mirostat_tau,
                    mirostat_eta,
                    mirostat_m,
                )
            elif mirostat_mode == 2:
                sampler.add_mirostat_v2(
                    self._seed,
                    mirostat_tau,
                    mirostat_eta,
                )
            else:
                n_probs = 0
                min_keep = max(1, n_probs)
                sampler.add_top_k(top_k)
                sampler.add_typical(typical_p, min_keep)
                sampler.add_top_p(top_p, min_keep)
                sampler.add_min_p(min_p, min_keep)
                sampler.add_temp(temp)
                sampler.add_dist(self._seed)
        return sampler

    def sample(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        idx: Optional[int] = None,
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
        assert self.n_tokens > 0

        tmp_sampler = False

        if self._sampler is None:
            tmp_sampler = True
            self._sampler = self._init_sampler(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                penalize_nl=penalize_nl,
                logits_processor=logits_processor,
                grammar=grammar,
            )

        ridx = idx - self.n_tokens if idx is not None else -1

        assert self.ctx is not None
        token = self._sampler.sample(self._ctx, ridx)
        if tmp_sampler:
            self._sampler = None
        return token

    def generate(
        self,
        tokens: Sequence[int],
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
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
        # Reset mirostat sampling
        self._mirostat_mu = ctypes.c_float(2.0 * mirostat_tau)
        self._sampler = self._init_sampler(
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            penalize_nl=penalize_nl,
            logits_processor=logits_processor,
            grammar=grammar,
        )

        # Check for kv cache prefix match
        if reset and self.n_tokens > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix
                if self.verbose:
                    print(
                        f"Llama.generate: {longest_prefix} prefix-match hit, "
                        f"remaining {len(tokens)} prompt tokens to eval",
                        file=sys.stderr,
                    )

        # Reset the model state
        if reset:
            self.reset()

        # # Reset the grammar
        # if grammar is not None:
        #     grammar.reset()

        sample_idx = self.n_tokens + len(tokens) - 1
        tokens = list(tokens)

        # Eval and sample
        while True:
            self.eval(tokens)
            while sample_idx < self.n_tokens:
                token = self.sample(
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    typical_p=typical_p,
                    temp=temp,
                    repeat_penalty=repeat_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    tfs_z=tfs_z,
                    mirostat_mode=mirostat_mode,
                    mirostat_tau=mirostat_tau,
                    mirostat_eta=mirostat_eta,
                    logits_processor=logits_processor,
                    grammar=grammar,
                    penalize_nl=penalize_nl,
                    idx=sample_idx,
                )

                sample_idx += 1
                if stopping_criteria is not None and stopping_criteria(
                    self._input_ids[: sample_idx], self._scores[sample_idx - self.n_tokens, :]
                ):
                    return
                tokens_or_none = yield token
                tokens.clear()
                tokens.append(token)
                if tokens_or_none is not None:
                    tokens.extend(tokens_or_none)

                if sample_idx < self.n_tokens and token != self._input_ids[sample_idx]:
                    self.n_tokens = sample_idx
                    self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
                    break

            if self.draft_model is not None:
                self.input_ids[self.n_tokens : self.n_tokens + len(tokens)] = tokens
                draft_tokens = self.draft_model(
                    self.input_ids[: self.n_tokens + len(tokens)]
                )
                tokens.extend(
                    draft_tokens.astype(int)[
                        : self._n_ctx - self.n_tokens - len(tokens)
                    ]
                )

    def create_embedding(
        self, input: Union[str, List[str]], model: Optional[str] = None
    ) -> CreateEmbeddingResponse:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            An embedding object.
        """
        model_name: str = model if model is not None else self.model_path

        input = input if isinstance(input, list) else [input]

        # get numeric embeddings
        embeds: Union[List[List[float]], List[List[List[float]]]]
        total_tokens: int
        embeds, total_tokens = self.embed(input, return_count=True)  # type: ignore

        # convert to CreateEmbeddingResponse
        data: List[Embedding] = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx,
            }
            for idx, emb in enumerate(embeds)
        ]

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    def embed(
        self,
        input: Union[str, List[str]],
        normalize: bool = False,
        truncate: bool = True,
        return_count: bool = False,
    ):
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            A list of embeddings
        """
        n_embd = self.n_embd()
        n_batch = self.n_batch

        # get pooling information
        pooling_type = self.pooling_type()
        logits_all = pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE

        if self.context_params.embeddings is False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_perf_context_reset(self._ctx.ctx)

        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        # reset batch
        self._batch.reset()

        # decode and fetch embeddings
        data: Union[List[List[float]], List[List[List[float]]]] = []

        def decode_batch(seq_sizes: List[int]):
            llama_cpp.llama_kv_self_clear(self._ctx.ctx)
            self._ctx.decode(self._batch)
            self._batch.reset()

            # store embeddings
            if pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE:
                pos: int = 0
                for i, size in enumerate(seq_sizes):
                    ptr = llama_cpp.llama_get_embeddings(self._ctx.ctx)
                    embedding: List[List[float]] = [
                        ptr[pos + j * n_embd : pos + (j + 1) * n_embd]
                        for j in range(size)
                    ]
                    if normalize:
                        embedding = [
                            internals.normalize_embedding(e) for e in embedding
                        ]
                    data.append(embedding)
                    pos += size
            else:
                for i in range(len(seq_sizes)):
                    ptr = llama_cpp.llama_get_embeddings_seq(self._ctx.ctx, i)
                    embedding: List[float] = ptr[:n_embd]
                    if normalize:
                        embedding = internals.normalize_embedding(embedding)
                    data.append(embedding)

        # init state
        total_tokens = 0
        s_batch = []
        t_batch = 0
        p_batch = 0

        # accumulate batches and encode
        for text in inputs:
            tokens = self.tokenize(text.encode("utf-8"))
            if truncate:
                tokens = tokens[:n_batch]

            n_tokens = len(tokens)
            total_tokens += n_tokens

            # check for overrun
            if n_tokens > n_batch:
                raise ValueError(
                    f"Requested tokens ({n_tokens}) exceed batch size of {n_batch}"
                )

            # time to eval batch
            if t_batch + n_tokens > n_batch:
                decode_batch(s_batch)
                s_batch = []
                t_batch = 0
                p_batch = 0

            # add to batch
            self._batch.add_sequence(tokens, p_batch, logits_all)

            # update batch stats
            s_batch.append(n_tokens)
            t_batch += n_tokens
            p_batch += 1

        # hanlde last batch
        decode_batch(s_batch)

        if self.verbose:
            llama_cpp.llama_perf_context_print(self._ctx.ctx)

        output = data[0] if isinstance(input, str) else data

        llama_cpp.llama_kv_self_clear(self._ctx.ctx)
        self.reset()

        if return_count:
            return output, total_tokens
        else:
            return output

    def _create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Union[
        Iterator[CreateCompletionResponse], Iterator[CreateCompletionStreamResponse]
    ]:
        assert suffix is None or suffix.__class__ is str

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        bos_token_id: int = self.token_bos()
        cls_token_id: int = self._model.token_cls()
        sep_token_id: int = self._model.token_sep()
        prefix_token_id: int = 0 # self._model.token_prefix() # TODO: Fix
        middle_token_id: int = 0 # self._model.token_middle() # TODO: Fix
        suffix_token_id: int = 0 # self._model.token_suffix() # TODO: Fix
        add_space_prefix: bool = (
            self.metadata.get("tokenizer.ggml.add_space_prefix", "true") == "true"
        )
        bos_tokens: List[int] = [cls_token_id if cls_token_id != -1 else bos_token_id]
        eos_tokens: List[int] = [
            sep_token_id if sep_token_id != -1 else self.token_eos()
        ]

        if (
            (isinstance(prompt, list) and suffix is None)
            or not self._model.add_bos_token()
            or bos_tokens[:1] == [-1]
        ):
            bos_tokens = []

        if (isinstance(prompt, list) and suffix is None) or (
            not self._model.add_eos_token() and sep_token_id == -1
        ):
            eos_tokens = []

        suffix_space_prefix: int = 0
        # Tokenizer hack to remove leading space
        if add_space_prefix and suffix_token_id >= 0 and suffix:
            suffix = "☺" + suffix
            suffix_space_prefix = 2

        # If prompt is empty, initialize completion with BOS token to avoid
        # detokenization including a space at the beginning of the completion
        completion_tokens: List[int] = [] if len(prompt) > 0 else [bos_token_id]
        # Add blank space to start of prompt to match OG llama tokenizer
        prefix_tokens: List[int] = (
            [prefix_token_id] if prefix_token_id >= 0 and suffix is not None else []
        ) + (
            (
                self.tokenize(
                    prompt.encode("utf-8"),
                    add_bos=False,
                    special=(prefix_token_id < 0 or suffix is None),
                )
                if prompt != ""
                else []
            )
            if isinstance(prompt, str)
            else prompt
        )
        suffix_tokens: List[int] = (
            (
                [suffix_token_id]
                + (
                    self.tokenize(suffix.encode("utf-8"), add_bos=False, special=False)[
                        suffix_space_prefix:
                    ]
                    if suffix
                    else []
                )
            )
            if suffix_token_id >= 0 and suffix is not None
            else []
        )
        middle_tokens: List[int] = (
            [middle_token_id] if middle_token_id >= 0 and suffix is not None else []
        )
        prompt_tokens: List[int] = (
            bos_tokens
            + (
                (suffix_tokens + prefix_tokens + middle_tokens)
                if self.spm_infill
                else (prefix_tokens + suffix_tokens + middle_tokens)
            )
            + eos_tokens
        )
        text: bytes = b""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else self.model_path

        if prompt_tokens[:2] == [self.token_bos()] * 2:
            warnings.warn(
                f'Detected duplicate leading "{self._model.token_get_text(self.token_bos())}" in prompt, this will likely reduce response quality, consider removing it...',
                RuntimeWarning,
            )

        # NOTE: This likely doesn't work correctly for the first token in the prompt
        # because of the extra space added to the start of the prompt_tokens
        if logit_bias is not None:
            logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

            def logit_bias_processor(
                input_ids: npt.NDArray[np.intc],
                scores: npt.NDArray[np.single],
            ) -> npt.NDArray[np.single]:
                new_scores = np.copy(
                    scores
                )  # Does it make sense to copy the whole array or can we just overwrite the original one?
                for input_id, score in logit_bias_map.items():
                    new_scores[input_id] = score + scores[input_id]
                return new_scores

            _logit_bias_processor = LogitsProcessorList([logit_bias_processor])
            if logits_processor is None:
                logits_processor = _logit_bias_processor
            else:
                logits_processor = logits_processor.extend(_logit_bias_processor)

        if self.verbose:
            self._ctx.reset_timings()

        if len(prompt_tokens) >= self._n_ctx:
            raise ValueError(
                f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if max_tokens is None or max_tokens <= 0:
            # Unlimited, depending on n_ctx.
            max_tokens = self._n_ctx - len(prompt_tokens)

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

        if logprobs is not None and self._logits_all is False:
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

        if seed is not None:
            self.set_seed(seed)
        else:
            self.set_seed(random.Random(self._seed).randint(0, 2 ** 32))

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
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
            grammar=grammar,
        ):
            if llama_cpp.llama_token_is_eog(self._model.vocab, token):
                text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)

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
                remaining_text = self.detokenize(
                    remaining_tokens,
                    prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
                )
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

                if logprobs is not None:
                    # not sure how to handle this branch when dealing
                    # with CJK output, so keep it unchanged
                    for token in remaining_tokens:
                        if token == bos_token_id:
                            continue
                        token_end_position += len(
                            self.detokenize(
                                [token],
                                prev_tokens=prompt_tokens
                                + completion_tokens[:returned_tokens],
                            )
                        )
                        # Check if stop sequence is in the token
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        token_str = self.detokenize(
                            [token],
                            prev_tokens=prompt_tokens
                            + completion_tokens[:returned_tokens],
                        ).decode("utf-8", errors="ignore")
                        text_offset = len(prompt) + len(
                            self.detokenize(
                                completion_tokens[:returned_tokens],
                                prev_tokens=prompt_tokens
                                + completion_tokens[:returned_tokens],
                            ).decode("utf-8", errors="ignore")
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :]
                        current_logprobs = Llama.logits_to_logprobs(logits).tolist()
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
                                self.detokenize(
                                    [token],
                                    prev_tokens=prompt_tokens
                                    + completion_tokens[:returned_tokens],
                                ).decode("utf-8", errors="ignore")
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
                                    "text": self.detokenize(
                                        [token],
                                        prev_tokens=prompt_tokens
                                        + completion_tokens[:returned_tokens],
                                    ).decode("utf-8", errors="ignore"),
                                    "index": 0,
                                    "logprobs": logprobs_or_none,
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    while len(remaining_tokens) > 0:
                        decode_success = False
                        for i in range(1, len(remaining_tokens) + 1):
                            try:
                                bs = self.detokenize(
                                    remaining_tokens[:i],
                                    prev_tokens=prompt_tokens
                                    + completion_tokens[:returned_tokens],
                                )
                                ts = bs.decode("utf-8")
                                decode_success = True
                                break
                            except UnicodeError:
                                pass
                        else:
                            break
                        if not decode_success:
                            # all remaining tokens cannot be decoded to a UTF-8 character
                            break
                        token_end_position += len(bs)
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        remaining_tokens = remaining_tokens[i:]
                        returned_tokens += i

                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": ts,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "length"
                break

        if stopping_criteria is not None and stopping_criteria(
            self._input_ids, self._scores[-1, :]
        ):
            text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
            finish_reason = "stop"

        if self.verbose:
            self._ctx.print_timings()

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            remaining_text = self.detokenize(
                remaining_tokens,
                prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
            )
            any_stop = [s for s in stop_sequences if s in remaining_text]
            if len(any_stop) > 0:
                end = min(remaining_text.index(stop) for stop in any_stop)
            else:
                end = len(remaining_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(
                    self.detokenize(
                        [token],
                        prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
                    )
                )

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    if token == bos_token_id:
                        continue
                    token_str = self.detokenize([token]).decode(
                        "utf-8", errors="ignore"
                    )
                    text_offset = len(prompt) + len(
                        self.detokenize(
                            completion_tokens[:returned_tokens],
                            prev_tokens=prompt_tokens
                            + completion_tokens[:returned_tokens],
                        )
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :]
                    current_logprobs = Llama.logits_to_logprobs(logits).tolist()
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
                if self.verbose:
                    print("Llama._create_completion: cache saved", file=sys.stderr)
            return

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix_token_id < 0 and suffix is not None:
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
                # Remove leading BOS token if exists
                all_tokens = (
                    prompt_tokens[1 if prompt_tokens[0] == self.token_bos() else 0 :]
                    + completion_tokens
                )
            else:
                all_tokens = completion_tokens

            all_token_strs = [
                self.detokenize([token], prev_tokens=all_tokens[:i]).decode(
                    "utf-8", errors="ignore"
                )
                for i, token in enumerate(all_tokens)
            ]
            all_logprobs = Llama.logits_to_logprobs(self._scores)[token_offset:]
            # TODO: may be able to change this loop to use np.take_along_dim
            for idx, (token, token_str, logprobs_token) in enumerate(
                zip(all_tokens, all_token_strs, all_logprobs)
            ):
                if token == bos_token_id:
                    continue
                text_offsets.append(
                    text_offset
                    + len(
                        self.detokenize(all_tokens[:idx]).decode(
                            "utf-8", errors="ignore"
                        )
                    )
                )
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(logprobs_token[int(token)])
                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i], prev_tokens=all_tokens[:idx]).decode(
                        "utf-8", errors="ignore"
                    ): logprob
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
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.
            logit_bias: A logit bias to use.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=-1 if max_tokens is None else max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            seed=seed,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        if stream:
            chunks: Iterator[CreateCompletionStreamResponse] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.
            logit_bias: A logit bias to use.

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
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            seed=seed,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )

    def create_chat_completion(
        self,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        """Generate a chat completion from a list of messages.

        Args:
            messages: A list of messages to generate a response for.
            functions: A list of functions to use for the chat completion.
            function_call: A function call to use for the chat completion.
            tools: A list of tools to use for the chat completion.
            tool_choice: A tool choice to use for the chat completion.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            stream: Whether to stream the results.
            stop: A list of strings to stop generation when encountered.
            seed: The seed to use for sampling.
            response_format: The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            tfs_z: The tail-free sampling parameter.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The mirostat sampling tau parameter.
            mirostat_eta: The mirostat sampling eta parameter.
            model: The name to use for the model in the completion object.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use.
            logit_bias: A logit bias to use.

        Returns:
            Generated chat completion or a stream of chat completion chunks.
        """
        handler = (
            self.chat_handler
            or self._chat_handlers.get(self.chat_format)
            or llama_chat_format.get_chat_completion_handler(self.chat_format)
        )
        return handler(
            llama=self,
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )

    def create_chat_completion_openai_v1(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        """Generate a chat completion with return type based on the the OpenAI v1 API.

        OpenAI python package is required to use this method.

        You can install it with `pip install openai`.

        Args:
            *args: Positional arguments to pass to create_chat_completion.
            **kwargs: Keyword arguments to pass to create_chat_completion.

        Returns:
            Generated chat completion or a stream of chat completion chunks.
        """
        try:
            from openai.types.chat import ChatCompletion, ChatCompletionChunk

            stream = kwargs.get("stream", False)  # type: ignore
            assert isinstance(stream, bool)
            if stream:
                return (ChatCompletionChunk(**chunk) for chunk in self.create_chat_completion(*args, **kwargs))  # type: ignore
            else:
                return ChatCompletion(**self.create_chat_completion(*args, **kwargs))  # type: ignore
        except ImportError:
            raise ImportError(
                "To use create_chat_completion_openai_v1, you must install the openai package."
                "You can install it with `pip install openai`."
            )

    def __getstate__(self):
        return dict(
            model_path=self.model_path,
            # Model Params
            n_gpu_layers=self.model_params.n_gpu_layers,
            split_mode=self.model_params.split_mode,
            main_gpu=self.model_params.main_gpu,
            tensor_split=self.tensor_split,
            vocab_only=self.model_params.vocab_only,
            use_mmap=self.model_params.use_mmap,
            use_mlock=self.model_params.use_mlock,
            kv_overrides=self.kv_overrides,
            # Context Params
            seed=self._seed,
            n_ctx=self.context_params.n_ctx,
            n_batch=self.n_batch,
            n_ubatch=self.context_params.n_ubatch,
            n_threads=self.context_params.n_threads,
            n_threads_batch=self.context_params.n_threads_batch,
            rope_scaling_type=self.context_params.rope_scaling_type,
            pooling_type=self.context_params.pooling_type,
            rope_freq_base=self.context_params.rope_freq_base,
            rope_freq_scale=self.context_params.rope_freq_scale,
            yarn_ext_factor=self.context_params.yarn_ext_factor,
            yarn_attn_factor=self.context_params.yarn_attn_factor,
            yarn_beta_fast=self.context_params.yarn_beta_fast,
            yarn_beta_slow=self.context_params.yarn_beta_slow,
            yarn_orig_ctx=self.context_params.yarn_orig_ctx,
            logits_all=self._logits_all,
            embedding=self.context_params.embeddings,
            offload_kqv=self.context_params.offload_kqv,
            flash_attn=self.context_params.flash_attn,
            op_offload=self.context_params.op_offload,
            swa_full=self.context_params.swa_full,
            # Sampling Params
            no_perf=self.context_params.no_perf,
            last_n_tokens_size=self.last_n_tokens_size,
            # LoRA Params
            lora_base=self.lora_base,
            lora_scale=self.lora_scale,
            lora_path=self.lora_path,
            # Backend Params
            numa=self.numa,
            # Chat Format Params
            chat_format=self.chat_format,
            chat_handler=self.chat_handler,
            # Speculative Decidng
            draft_model=self.draft_model,
            # KV cache quantization
            type_k=self.context_params.type_k,
            type_v=self.context_params.type_v,
            # Misc
            spm_infill=self.spm_infill,
            verbose=self.verbose,
        )

    def __setstate__(self, state):
        self.__init__(**state)

    def save_state(self) -> LlamaState:
        if self.verbose:
            print("Llama.save_state: saving llama state", file=sys.stderr)
        state_size = llama_cpp.llama_get_state_size(self._ctx.ctx)
        if self.verbose:
            print(f"Llama.save_state: got state size: {state_size}", file=sys.stderr)
        llama_state = (ctypes.c_uint8 * int(state_size))()
        if self.verbose:
            print("Llama.save_state: allocated state", file=sys.stderr)
        n_bytes = llama_cpp.llama_copy_state_data(self._ctx.ctx, llama_state)
        if self.verbose:
            print(f"Llama.save_state: copied llama state: {n_bytes}", file=sys.stderr)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        llama_state_compact = (ctypes.c_uint8 * int(n_bytes))()
        llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))
        if self.verbose:
            print(
                f"Llama.save_state: saving {n_bytes} bytes of llama state",
                file=sys.stderr,
            )
        return LlamaState(
            scores=self._scores.copy(),
            input_ids=self.input_ids.copy(),
            n_tokens=self.n_tokens,
            llama_state=bytes(llama_state_compact),
            llama_state_size=n_bytes,
            seed=self._seed,
        )

    def load_state(self, state: LlamaState) -> None:
        # Only filling in up to `n_tokens` and then zero-ing out the rest
        self.scores[: state.n_tokens, :] = state.scores.copy()
        rest = self.scores[state.n_tokens :, :]
        rest[rest > 0] = 0.0
        self.input_ids = state.input_ids.copy()
        self.n_tokens = state.n_tokens
        self._seed = state.seed
        state_size = state.llama_state_size
        LLamaStateArrayType = ctypes.c_uint8 * state_size
        llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)

        if llama_cpp.llama_set_state_data(self._ctx.ctx, llama_state) != state_size:
            raise RuntimeError("Failed to set llama state data")

    def n_ctx(self) -> int:
        """Return the context window size."""
        return self._ctx.n_ctx()

    def n_embd(self) -> int:
        """Return the embedding size."""
        return self._model.n_embd()

    def n_vocab(self) -> int:
        """Return the vocabulary size."""
        return self._model.n_vocab()

    def tokenizer(self) -> LlamaTokenizer:
        """Return the llama tokenizer for this model."""
        return LlamaTokenizer(self)

    def token_eos(self) -> int:
        """Return the end-of-sequence token."""
        return self._model.token_eos()

    def token_bos(self) -> int:
        """Return the beginning-of-sequence token."""
        return self._model.token_bos()

    def token_nl(self) -> int:
        """Return the newline token."""
        return self._model.token_nl()

    def pooling_type(self) -> str:
        """Return the pooling type."""
        return self._ctx.pooling_type()

    def close(self) -> None:
        """Explicitly free the model from memory."""
        self._stack.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def logits_to_logprobs(
        logits: Union[npt.NDArray[np.single], List], axis: int = -1
    ) -> npt.NDArray[np.single]:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html
        logits_maxs: np.ndarray = np.amax(logits, axis=axis, keepdims=True)
        if logits_maxs.ndim > 0:
            logits_maxs[~np.isfinite(logits_maxs)] = 0
        elif not np.isfinite(logits_maxs):
            logits_maxs = 0
        subtract_maxs = np.subtract(logits, logits_maxs, dtype=np.single)
        exp = np.exp(subtract_maxs)
        # Suppress warnings about log of zero
        with np.errstate(divide="ignore"):
            summed = np.sum(exp, axis=axis, keepdims=True)
            out = np.log(summed)
        return subtract_maxs - out

    @staticmethod
    def longest_token_prefix(a: Sequence[int], b: Sequence[int]):
        longest_prefix = 0
        for _a, _b in zip(a, b):
            if _a == _b:
                longest_prefix += 1
            else:
                break
        return longest_prefix

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str],
        additional_files: Optional[List] = None,
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        **kwargs: Any,
    ) -> "Llama":
        """Create a Llama model from a pretrained model name or path.
        This method requires the huggingface-hub package.
        You can install it with `pip install huggingface-hub`.

        Args:
            repo_id: The model repo id.
            filename: A filename or glob pattern to match the model file in the repo.
            additional_files: A list of filenames or glob patterns to match additional model files in the repo.
            local_dir: The local directory to save the model to.
            local_dir_use_symlinks: Whether to use symlinks when downloading the model.
            **kwargs: Additional keyword arguments to pass to the Llama constructor.

        Returns:
            A Llama model."""
        try:
            from huggingface_hub import hf_hub_download, HfFileSystem
            from huggingface_hub.utils import validate_repo_id
        except ImportError:
            raise ImportError(
                "Llama.from_pretrained requires the huggingface-hub package. "
                "You can install it with `pip install huggingface-hub`."
            )

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id, recursive=True)
        ]

        # split each file into repo_id, subfolder, filename
        file_list: List[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        # find the only/first shard file:
        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(files)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
        )

        if additional_files:
            for additonal_file_name in additional_files:
                # find the additional shard file:
                matching_additional_files = [file for file in file_list if fnmatch.fnmatch(file, additonal_file_name)]

                if len(matching_additional_files) == 0:
                    raise ValueError(
                        f"No file found in {repo_id} that match {additonal_file_name}\n\n"
                        f"Available Files:\n{json.dumps(file_list)}"
                    )

                if len(matching_additional_files) > 1:
                    raise ValueError(
                        f"Multiple files found in {repo_id} matching {additonal_file_name}\n\n"
                        f"Available Files:\n{json.dumps(files)}"
                    )

                (matching_additional_file,) = matching_additional_files

                # download the additional file
                hf_hub_download(
                    repo_id=repo_id,
                    filename=matching_additional_file,
                    subfolder=subfolder,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                    cache_dir=cache_dir,
                )

        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, filename)

        # loading the first file of a sharded GGUF loads all remaining shard files in the subfolder
        return cls(
            model_path=model_path,
            **kwargs,
        )


class LlamaState:
    def __init__(
        self,
        input_ids: npt.NDArray[np.intc],
        scores: npt.NDArray[np.single],
        n_tokens: int,
        llama_state: bytes,
        llama_state_size: int,
        seed: int,
    ):
        self.input_ids = input_ids
        self.scores = scores
        self.n_tokens = n_tokens
        self.llama_state = llama_state
        self.llama_state_size = llama_state_size
        self.seed = seed


LogitsProcessor = Callable[
    [npt.NDArray[np.intc], npt.NDArray[np.single]], npt.NDArray[np.single]
]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


StoppingCriteria = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
    ) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])


class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, min_tokens: int, token_eos: int):
        self.min_tokens = min_tokens
        self.token_eos = token_eos
        self.prompt_tokens = None

    def __call__(
        self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        if self.prompt_tokens is None:
            self.prompt_tokens = len(input_ids)
        if len(input_ids) - self.prompt_tokens < self.min_tokens:
            scores[self.token_eos] = -np.inf
        return scores
