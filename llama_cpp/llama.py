import os
import uuid
import time
import multiprocessing
from typing import List, Optional, Union, Generator, Sequence
from collections import deque

from . import llama_cpp
from .llama_types import *


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
        self,
        model_path: str,
        # NOTE: These parameters are likely to change in the future.
        n_ctx: int = 512,
        n_parts: int = -1,
        seed: int = 1337,
        f16_kv: bool = False,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads: Optional[int] = None,
        n_batch: int = 8,
        last_n_tokens_size: int = 64,
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
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.model_path = model_path

        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = n_ctx
        self.params.n_parts = n_parts
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only
        self.params.use_mlock = use_mlock
        self.params.embedding = embedding

        self.last_n_tokens_size = last_n_tokens_size
        self.n_batch = n_batch

        self.n_threads = n_threads or multiprocessing.cpu_count()

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.ctx = llama_cpp.llama_init_from_file(
            self.model_path.encode("utf-8"), self.params
        )

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

    def generate(
        self,
        tokens: Sequence[llama_cpp.llama_token],
        top_k: int,
        top_p: float,
        temp: float,
        repeat_penalty: float,
    ) -> Generator[
        llama_cpp.llama_token, Optional[Sequence[llama_cpp.llama_token]], None
    ]:
        # Temporary workaround for https://github.com/ggerganov/llama.cpp/issues/684
        if temp == 0.0:
            temp = 1.0
            top_p = 0.0
            top_k = 1
        assert self.ctx is not None
        n_ctx = int(llama_cpp.llama_n_ctx(self.ctx))
        n_tokens = 0
        last_n_tokens = deque(
            [llama_cpp.llama_token(0)] * self.last_n_tokens_size,
            maxlen=self.last_n_tokens_size,
        )
        while True:
            for i in range(0, len(tokens), self.n_batch):
                batch = tokens[i : min(len(tokens), i + self.n_batch)]
                n_past = min(n_ctx - len(batch), n_tokens)
                return_code = llama_cpp.llama_eval(
                    ctx=self.ctx,
                    tokens=(llama_cpp.llama_token * len(batch))(*batch),
                    n_tokens=llama_cpp.c_int(len(batch)),
                    n_past=llama_cpp.c_int(n_past),
                    n_threads=llama_cpp.c_int(self.n_threads),
                )
                if int(return_code) != 0:
                    raise RuntimeError(f"llama_eval returned {return_code}")
                last_n_tokens.extend(batch)
                n_tokens += len(batch)
            token = llama_cpp.llama_sample_top_p_top_k(
                ctx=self.ctx,
                last_n_tokens_data=(llama_cpp.llama_token * self.last_n_tokens_size)(
                    *last_n_tokens
                ),
                last_n_tokens_size=llama_cpp.c_int(self.last_n_tokens_size),
                top_k=llama_cpp.c_int(top_k),
                top_p=llama_cpp.c_float(top_p),
                temp=llama_cpp.c_float(temp),
                repeat_penalty=llama_cpp.c_float(repeat_penalty),
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
        tokens = self.tokenize(input.encode("utf-8"))
        next(self.generate(tokens, top_k=0, top_p=0.0, temp=1.0, repeat_penalty=1.0))
        n_tokens = len(tokens)
        embedding = llama_cpp.llama_get_embeddings(self.ctx)[
            : llama_cpp.llama_n_embd(self.ctx)
        ]
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

    def _create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: List[str] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[
        Generator[Completion, None, None],
        Generator[CompletionChunk, None, None],
    ]:
        assert self.ctx is not None
        completion_id = f"cmpl-{str(uuid.uuid4())}"
        created = int(time.time())
        completion_tokens: List[llama_cpp.llama_token] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens = self.tokenize(b" " + prompt.encode("utf-8"))
        text = b""

        if len(prompt_tokens) + max_tokens > int(llama_cpp.llama_n_ctx(self.ctx)):
            raise ValueError(
                f"Requested tokens exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if stop != []:
            stop_bytes = [s.encode("utf-8") for s in stop]
        else:
            stop_bytes = []

        finish_reason = None
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty,
        ):
            if token == llama_cpp.llama_token_eos():
                finish_reason = "stop"
                break
            completion_tokens.append(token)

            text = self.detokenize(completion_tokens)
            any_stop = [s for s in stop_bytes if s in text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = text[: text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                start = len(self.detokenize(completion_tokens[:-1]))
                longest = 0
                # TODO: Clean up this mess
                for s in stop_bytes:
                    for i in range(len(s), 0, -1):
                        if s[-i:] == text[-i:]:
                            if i > longest:
                                longest = i
                            break
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_path,
                    "choices": [
                        {
                            "text": text[start : len(text) - longest].decode("utf-8"),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }

        if finish_reason is None:
            finish_reason = "length"

        if stream:
            if finish_reason == "stop":
                start = len(self.detokenize(completion_tokens[:-1]))
                text = text[start:].decode("utf-8")
            else:
                text = ""
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            return

        text = text.decode("utf-8")

        if echo:
            text = prompt + text

        if suffix is not None:
            text = text + suffix

        if logprobs is not None:
            raise NotImplementedError("logprobs not implemented")

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_path,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
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
        stop: List[str] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[Completion, Generator[CompletionChunk, None, None]]:
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
            chunks: Generator[CompletionChunk, None, None] = completion_or_chunks
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
        stop: List[str] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ):
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

    def __del__(self):
        if self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None
