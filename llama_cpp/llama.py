import os
import uuid
import time
import multiprocessing
from typing import List, Optional

from . import llama_cpp


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_parts: int = -1,
        seed: int = 1337,
        f16_kv: bool = False,
        logits_all: bool = False,
        vocab_only: bool = False,
        n_threads: Optional[int] = None,
    ) -> "Llama":
        """Load a llama.cpp model from `model_path`.

        Args:
            model_path: Path to the model directory.
            n_ctx: Number of tokens to keep in memory.
            n_parts: Number of parts to split the model into. If -1, the number of parts is automatically determined.
            seed: Random seed.
            f16_kv: Use half-precision for key/value matrices.
            logits_all: Return logits for all tokens, not just the vocabulary.
            vocab_only: Only use tokens in the vocabulary.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.model_path = model_path

        self.last_n = 64
        self.max_chunk_size = 32

        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = n_ctx
        self.params.n_parts = n_parts
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only

        self.n_threads = n_threads or multiprocessing.cpu_count()

        self.tokens = (llama_cpp.llama_token * self.params.n_ctx)()

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.ctx = llama_cpp.llama_init_from_file(
            self.model_path.encode("utf-8"), self.params
        )

    def __call__(
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

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        text = b""
        finish_reason = "length"
        completion_tokens = 0

        if stop is not None:
            stop = [s.encode("utf-8") for s in stop]

        prompt_tokens = llama_cpp.llama_tokenize(
            self.ctx,
            prompt.encode("utf-8"),
            self.tokens,
            llama_cpp.llama_n_ctx(self.ctx),
            True,
        )
        if prompt_tokens < 0:
            raise RuntimeError(f"Failed to tokenize prompt: {prompt_tokens}")

        if prompt_tokens + max_tokens > self.params.n_ctx:
            raise ValueError(
                f"Requested tokens exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        # Process prompt in chunks to avoid running out of memory
        for i in range(0, prompt_tokens, self.max_chunk_size):
            chunk = self.tokens[i : min(prompt_tokens, i + self.max_chunk_size)]
            rc = llama_cpp.llama_eval(
                self.ctx,
                (llama_cpp.llama_token * len(chunk))(*chunk),
                len(chunk),
                max(0, i - 1),
                self.n_threads,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to evaluate prompt: {rc}")

        for i in range(max_tokens):
            tokens_seen = prompt_tokens + completion_tokens
            last_n_tokens = [0] * max(0, self.last_n - tokens_seen) + [
                self.tokens[j]
                for j in range(max(tokens_seen - self.last_n, 0), tokens_seen)
            ]

            token = llama_cpp.llama_sample_top_p_top_k(
                self.ctx,
                (llama_cpp.llama_token * len(last_n_tokens))(*last_n_tokens),
                len(last_n_tokens),
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty,
            )
            if token == llama_cpp.llama_token_eos():
                finish_reason = "stop"
                break
            text += llama_cpp.llama_token_to_str(self.ctx, token)
            self.tokens[prompt_tokens + i] = token
            completion_tokens += 1

            any_stop = [s for s in stop if s in text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = text[: text.index(first_stop)]
                finish_reason = "stop"
                break

            rc = llama_cpp.llama_eval(
                self.ctx,
                (llama_cpp.llama_token * 1)(self.tokens[prompt_tokens + i]),
                1,
                prompt_tokens + completion_tokens,
                self.n_threads,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to evaluate next token: {rc}")

        text = text.decode("utf-8")

        if echo:
            text = prompt + text

        if suffix is not None:
            text = text + suffix

        if logprobs is not None:
            logprobs = llama_cpp.llama_get_logits(
                self.ctx,
            )[:logprobs]

        return {
            "id": f"cmpl-{str(uuid.uuid4())}",  # Likely to change
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": logprobs,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def __del__(self):
        llama_cpp.llama_free(self.ctx)
