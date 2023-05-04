import os
import sys
import uuid
import time
import math
import multiprocessing
from typing import List, Optional, Union, Generator, Sequence, Iterator, Deque, Tuple
from collections import deque

from . import llama_cpp
from .llama_types import *


class LlamaCache:
    """Cache for a llama.cpp model."""

    def __init__(self):
        self.cache_state: Dict[Tuple[llama_cpp.llama_token, ...], "LlamaState"] = dict()

    def _sorted_keys(self) -> List[Tuple[llama_cpp.llama_token, ...]]:
        return [
            key
            for _, key in sorted(
                ((len(key), key) for key in self.cache_state.keys()), reverse=True
            )
        ]

    def _find_key(
        self, key: Tuple[llama_cpp.llama_token, ...]
    ) -> Optional[Tuple[llama_cpp.llama_token, ...]]:
        for k in self._sorted_keys():
            if key[: len(k)] == k:
                return k
        return None

    def __getitem__(
        self, key: Sequence[llama_cpp.llama_token]
    ) -> Optional["LlamaState"]:
        _key = self._find_key(tuple(key))
        if _key is None:
            return None
        return self.cache_state[_key]

    def __contains__(self, key: Sequence[llama_cpp.llama_token]) -> bool:
        return self._find_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[llama_cpp.llama_token], value: "LlamaState"):
        self.cache_state = dict()  # NOTE: Currently limit to one cache entry.
        self.cache_state[tuple(key)] = value


class LlamaState:
    def __init__(
        self,
        eval_tokens: Deque[llama_cpp.llama_token],
        eval_logits: Deque[List[llama_cpp.c_float]],
        llama_state,
        llama_state_size: llama_cpp.c_size_t,
    ):
        self.eval_tokens = eval_tokens
        self.eval_logits = eval_logits
        self.llama_state = llama_state
        self.llama_state_size = llama_state_size


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
        self,
        model_path: str,
        # NOTE: These parameters are likely to change in the future.
        n_ctx: int = 512,
        n_parts: int = -1,
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
        verbose: bool = True,
    ):
        """Load a llama.cpp model from `model_path`.

        Args:
            model_path: Path to the model.
            n_ctx: Maximum context size.
            n_parts: Number of parts to split the model into. If -1, the number of parts is automatically determined.
            seed: Random seed. 0 for random.
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
        self.params.n_parts = n_parts
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap if lora_path is None else False
        self.params.use_mlock = use_mlock
        self.params.embedding = embedding

        self.last_n_tokens_size = last_n_tokens_size
        self.n_batch = min(n_ctx, n_batch)
        self.eval_tokens: Deque[llama_cpp.llama_token] = deque(maxlen=n_ctx)
        self.eval_logits: Deque[List[llama_cpp.c_float]] = deque(
            maxlen=n_ctx if logits_all else 1
        )

        self.cache: Optional[LlamaCache] = None

        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)

        self.lora_base = lora_base
        self.lora_path = lora_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.ctx = llama_cpp.llama_init_from_file(
            self.model_path.encode("utf-8"), self.params
        )

        assert self.ctx is not None

        if self.lora_path:
            if llama_cpp.llama_apply_lora_from_file(
                self.ctx,
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

    def tokenize(self, text: bytes) -> List[llama_cpp.llama_token]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        assert self.ctx is not None
        n_ctx = llama_cpp.llama_n_ctx(self.ctx)
        tokens = (llama_cpp.llama_token * int(n_ctx))()
        n_tokens = llama_cpp.llama_tokenize(
            self.ctx,
            text,
            tokens,
            n_ctx,
            llama_cpp.c_bool(True),
        )
        if int(n_tokens) < 0:
            raise RuntimeError(f'Failed to tokenize: text="{text}" n_tokens={n_tokens}')
        return list(tokens[:n_tokens])

    def detokenize(self, tokens: List[llama_cpp.llama_token]) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        assert self.ctx is not None
        output = b""
        for token in tokens:
            output += llama_cpp.llama_token_to_str(self.ctx, token)
        return output

    def set_cache(self, cache: Optional[LlamaCache]):
        """Set the cache.

        Args:
            cache: The cache to set.
        """
        self.cache = cache

    def reset(self):
        """Reset the model state."""
        self.eval_tokens.clear()
        self.eval_logits.clear()

    def eval(self, tokens: Sequence[llama_cpp.llama_token]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        assert self.ctx is not None
        n_ctx = int(llama_cpp.llama_n_ctx(self.ctx))
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = min(n_ctx - len(batch), len(self.eval_tokens))
            n_tokens = len(batch)
            return_code = llama_cpp.llama_eval(
                ctx=self.ctx,
                tokens=(llama_cpp.llama_token * len(batch))(*batch),
                n_tokens=llama_cpp.c_int(n_tokens),
                n_past=llama_cpp.c_int(n_past),
                n_threads=llama_cpp.c_int(self.n_threads),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"llama_eval returned {return_code}")
            # Save tokens
            self.eval_tokens.extend(batch)
            # Save logits
            rows = n_tokens if self.params.logits_all else 1
            n_vocab = llama_cpp.llama_n_vocab(self.ctx)
            cols = int(n_vocab)
            logits_view = llama_cpp.llama_get_logits(self.ctx)
            logits: List[List[llama_cpp.c_float]] = [
                [logits_view[i * cols + j] for j in range(cols)] for i in range(rows)
            ]
            self.eval_logits.extend(logits)

    def _sample_top_p_top_k(
        self,
        last_n_tokens_data,  # type: llama_cpp.Array[llama_cpp.llama_token]
        last_n_tokens_size: llama_cpp.c_int,
        top_k: llama_cpp.c_int,
        top_p: llama_cpp.c_float,
        temp: llama_cpp.c_float,
        repeat_penalty: llama_cpp.c_float,
    ):
        assert self.ctx is not None
        assert len(self.eval_logits) > 0
        n_vocab = int(llama_cpp.llama_n_vocab(self.ctx))
        logits = self.eval_logits[-1]
        data = (llama_cpp.llama_token_data * n_vocab)(
            *[
                llama_cpp.llama_token_data(
                    id=llama_cpp.llama_token(i),
                    logit=logits[i],
                    p=llama_cpp.c_float(0.0),
                )
                for i in range(n_vocab)
            ]
        )
        size = llama_cpp.c_size_t(n_vocab)
        sorted = False
        candidates = llama_cpp.llama_token_data_array(
            data=data,
            size=size,
            sorted=sorted,
        )
        llama_cpp.llama_sample_repetition_penalty(
            ctx=self.ctx,
            last_tokens_data=last_n_tokens_data,
            last_tokens_size=last_n_tokens_size,
            candidates=llama_cpp.ctypes.pointer(candidates),
            penalty=repeat_penalty,
        )
        if temp == 0.0:
            return llama_cpp.llama_sample_token_greedy(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
            )
        else:
            llama_cpp.llama_sample_top_k(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                k=top_k,
            )
            llama_cpp.llama_sample_tail_free(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                z=llama_cpp.c_float(1.0),
            )
            llama_cpp.llama_sample_typical(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                p=llama_cpp.c_float(1.0),
            )
            llama_cpp.llama_sample_top_p(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                p=top_p,
            )
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                temp=temp,
            )
            return llama_cpp.llama_sample_token(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
            )

    def sample(
        self,
        top_k: int,
        top_p: float,
        temp: float,
        repeat_penalty: float,
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
            0, self.last_n_tokens_size - len(self.eval_tokens)
        ) + list(self.eval_tokens)[-self.last_n_tokens_size :]
        return self._sample_top_p_top_k(
            last_n_tokens_data=(llama_cpp.llama_token * self.last_n_tokens_size)(
                *last_n_tokens_data
            ),
            last_n_tokens_size=llama_cpp.c_int(self.last_n_tokens_size),
            top_k=llama_cpp.c_int(top_k),
            top_p=llama_cpp.c_float(top_p),
            temp=llama_cpp.c_float(temp),
            repeat_penalty=llama_cpp.c_float(repeat_penalty),
        )

    def generate(
        self,
        tokens: Sequence[llama_cpp.llama_token],
        top_k: int,
        top_p: float,
        temp: float,
        repeat_penalty: float,
        reset: bool = True,
    ) -> Generator[
        llama_cpp.llama_token, Optional[Sequence[llama_cpp.llama_token]], None
    ]:
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

        if (
            reset
            and len(self.eval_tokens) > 0
            and tuple(self.eval_tokens) == tuple(tokens[: len(self.eval_tokens)])
        ):
            if self.verbose:
                print("Llama.generate: cache hit", file=sys.stderr)
            reset = False
            tokens = tokens[len(self.eval_tokens) :]

        if reset:
            self.reset()
        while True:
            self.eval(tokens)
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
            )
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

    def create_embedding(self, input: str) -> Embedding:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            An embedding object.
        """
        assert self.ctx is not None

        if self.params.embedding == False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        tokens = self.tokenize(input.encode("utf-8"))
        self.reset()
        self.eval(tokens)
        n_tokens = len(tokens)
        embedding = llama_cpp.llama_get_embeddings(self.ctx)[
            : llama_cpp.llama_n_embd(self.ctx)
        ]

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": 0,
                }
            ],
            "model": self.model_path,
            "usage": {
                "prompt_tokens": n_tokens,
                "total_tokens": n_tokens,
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
        stop: Optional[List[str]] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:
        assert self.ctx is not None
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        completion_tokens: List[llama_cpp.llama_token] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[llama_cpp.llama_token] = self.tokenize(
            b" " + prompt.encode("utf-8")
        )
        text: bytes = b""
        returned_characters: int = 0
        stop = stop if stop is not None else []

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        if len(prompt_tokens) + max_tokens > int(llama_cpp.llama_n_ctx(self.ctx)):
            raise ValueError(
                f"Requested tokens exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self.params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache and prompt_tokens in self.cache:
            if self.verbose:
                print("Llama._create_completion: cache hit", file=sys.stderr)
            self.load_state(self.cache[prompt_tokens])

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty,
        ):
            if token == llama_cpp.llama_token_eos():
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            if self.cache and len(completion_tokens) == 0:
                if prompt_tokens not in self.cache:
                    if self.verbose:
                        print("Llama._create_completion: cache miss", file=sys.stderr)
                    self.cache[prompt_tokens] = self.save_state()

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
                start = returned_characters
                longest = 0
                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                for s in stop_sequences:
                    for i in range(len(s), 0, -1):
                        if all_text.endswith(s[:i]):
                            if i > longest:
                                longest = i
                            break
                text = all_text[: len(all_text) - longest]
                returned_characters += len(text[start:])
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_path,
                    "choices": [
                        {
                            "text": text[start:].decode("utf-8", errors="ignore"),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if stream:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": text[returned_characters:].decode(
                            "utf-8", errors="ignore"
                        ),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            return

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0
            text_offsets: List[int] = []
            token_logprobs: List[float] = []
            tokens: List[str] = []
            top_logprobs: List[Dict[str, float]] = []

            all_tokens = prompt_tokens + completion_tokens
            all_token_strs = [
                self.detokenize([token]).decode("utf-8", errors="ignore")
                for token in all_tokens
            ]
            all_logprobs = [Llama.logits_to_logprobs(list(map(float, row))) for row in self.eval_logits]
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
                token_logprobs.append(sorted_logprobs[int(token)][0])
                top_logprob = {
                    self.detokenize([llama_cpp.llama_token(i)]).decode(
                        "utf-8", errors="ignore"
                    ): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: sorted_logprobs[int(token)][0]})
                top_logprobs.append(top_logprob)
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_path,
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
        stop: Optional[List[str]] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
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
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
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
        stop: Optional[List[str]] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
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
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
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
                        },
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
        stop: Optional[List[str]] = [],
        max_tokens: int = 256,
        repeat_penalty: float = 1.1,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Generate a chat completion from a list of messages.

        Args:
            messages: A list of messages to generate a response for.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.
            stop: A list of strings to stop generation when encountered.
            max_tokens: The maximum number of tokens to generate.
            repeat_penalty: The penalty to apply to repeated tokens.

        Returns:
            Generated chat completion or a stream of chat completion chunks.
        """
        stop = stop if stop is not None else []
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
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks  # type: ignore
            return self._convert_text_completion_chunks_to_chat(chunks)
        else:
            completion: Completion = completion_or_chunks  # type: ignore
            return self._convert_text_completion_to_chat(completion)

    def __del__(self):
        if self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

    def __getstate__(self):
        return dict(
            verbose=self.verbose,
            model_path=self.model_path,
            n_ctx=self.params.n_ctx,
            n_parts=self.params.n_parts,
            seed=self.params.seed,
            f16_kv=self.params.f16_kv,
            logits_all=self.params.logits_all,
            vocab_only=self.params.vocab_only,
            use_mmap=self.params.use_mmap,
            use_mlock=self.params.use_mlock,
            embedding=self.params.embedding,
            last_n_tokens_size=self.last_n_tokens_size,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            lora_base=self.lora_base,
            lora_path=self.lora_path,
        )

    def __setstate__(self, state):
        self.__init__(
            model_path=state["model_path"],
            n_ctx=state["n_ctx"],
            n_parts=state["n_parts"],
            seed=state["seed"],
            f16_kv=state["f16_kv"],
            logits_all=state["logits_all"],
            vocab_only=state["vocab_only"],
            use_mmap=state["use_mmap"],
            use_mlock=state["use_mlock"],
            embedding=state["embedding"],
            n_threads=state["n_threads"],
            n_batch=state["n_batch"],
            last_n_tokens_size=state["last_n_tokens_size"],
            lora_base=state["lora_base"],
            lora_path=state["lora_path"],
            verbose=state["verbose"],
        )

    def save_state(self) -> LlamaState:
        assert self.ctx is not None
        state_size = llama_cpp.llama_get_state_size(self.ctx)
        llama_state = (llama_cpp.c_uint8 * int(state_size))()
        n_bytes = llama_cpp.llama_copy_state_data(self.ctx, llama_state)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        llama_state_compact = (llama_cpp.c_uint8 * int(n_bytes))()
        llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))
        if self.verbose:
            print(f"Llama.save_state: saving {n_bytes} bytes of llama state", file=sys.stderr)
        return LlamaState(
            eval_tokens=self.eval_tokens.copy(),
            eval_logits=self.eval_logits.copy(),
            llama_state=llama_state_compact,
            llama_state_size=n_bytes,
        )

    def load_state(self, state: LlamaState) -> None:
        assert self.ctx is not None
        self.eval_tokens = state.eval_tokens.copy()
        self.eval_logits = state.eval_logits.copy()
        state_size = state.llama_state_size
        if llama_cpp.llama_set_state_data(self.ctx, state.llama_state) != state_size:
            raise RuntimeError("Failed to set llama state data")

    @staticmethod
    def token_eos() -> llama_cpp.llama_token:
        """Return the end-of-sequence token."""
        return llama_cpp.llama_token_eos()

    @staticmethod
    def token_bos() -> llama_cpp.llama_token:
        """Return the beginning-of-sequence token."""
        return llama_cpp.llama_token_bos()

    @staticmethod
    def logits_to_logprobs(logits: List[float]) -> List[float]:
        exps = [math.exp(float(x)) for x in logits]
        sum_exps = sum(exps)
        return [math.log(x / sum_exps) for x in exps]
