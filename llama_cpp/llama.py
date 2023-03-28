import os
import uuid
import time
import multiprocessing
from typing import List, Optional
from collections import deque

from . import llama_cpp


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
        self,
        model_path: str,
        # NOTE: The following parameters are likely to change in the future.
        n_ctx: int = 512,
        n_parts: int = -1,
        seed: int = 1337,
        f16_kv: bool = False,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads: Optional[int] = None,
    ) -> "Llama":
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

        self.last_n = 64
        self.max_chunk_size = n_ctx

        self.n_threads = n_threads or multiprocessing.cpu_count()

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.ctx = llama_cpp.llama_init_from_file(
            self.model_path.encode("utf-8"), self.params
        )

    def tokenize(self, text: bytes) -> List[int]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.

        Returns:
            A list of tokens.
        """
        n_ctx = llama_cpp.llama_n_ctx(self.ctx)
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            self.ctx,
            text,
            tokens,
            n_ctx,
            True,
        )
        if n_tokens < 0:
            raise RuntimeError(f"Failed to tokenize: text=\"{text}\" n_tokens={n_tokens}")
        return list(tokens[:n_tokens])

    def detokenize(self, tokens: List[int]) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        output = b""
        for token in tokens:
            output += llama_cpp.llama_token_to_str(self.ctx, token)
        return output


    def _eval(self, tokens: List[int], n_past):
        rc = llama_cpp.llama_eval(
            self.ctx,
            (llama_cpp.llama_token * len(tokens))(*tokens),
            len(tokens),
            n_past,
            self.n_threads,
        )
        if rc != 0:
            raise RuntimeError(f"Failed to evaluate: {rc}")

    def _sample(self, last_n_tokens, top_p, top_k, temp, repeat_penalty):
        return llama_cpp.llama_sample_top_p_top_k(
            self.ctx,
            (llama_cpp.llama_token * len(last_n_tokens))(*last_n_tokens),
            len(last_n_tokens),
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        )

    def _generate(self, past_tokens, max_tokens, top_p, top_k, temp, repeat_penalty):
        last_n_tokens = deque([0] * self.last_n, maxlen=self.last_n)
        last_n_tokens.extend(past_tokens)
        for i in range(max_tokens):
            token = self._sample(
                last_n_tokens,
                top_p=top_p,
                top_k=top_k,
                temp=temp,
                repeat_penalty=repeat_penalty
            )
            yield token
            self._eval([token], len(past_tokens) + i)

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
        completion_id = f"cmpl-{str(uuid.uuid4())}"
        created= int(time.time())
        text = b""
        completion_tokens = []
        last_n_tokens = deque([0] * self.last_n, maxlen=self.last_n)

        prompt_tokens = self.tokenize(prompt.encode("utf-8"))

        if len(prompt_tokens) + max_tokens > llama_cpp.llama_n_ctx(self.ctx):
            raise ValueError(
                f"Requested tokens exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        # Process prompt in chunks to avoid running out of memory
        for i in range(0, len(prompt_tokens), self.max_chunk_size):
            chunk = prompt_tokens[i : min(len(prompt_tokens), i + self.max_chunk_size)]
            self._eval(chunk, n_past=i)

        if stop is not None:
            stop = [s.encode("utf-8") for s in stop]

        finish_reason = None
        for token in self._generate(prompt_tokens, max_tokens, top_p, top_k, temperature, repeat_penalty):
            if token == llama_cpp.llama_token_eos():
                finish_reason = "stop"
                break
            text += self.detokenize([token])
            last_n_tokens.append(token)
            completion_tokens.append(token)

            any_stop = [s for s in stop if s in text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = text[: text.index(first_stop)]
                finish_reason = "stop"
                break

        if finish_reason is None:
            finish_reason = "length"

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
            "id": completion_id,
            "object": "text_completion",
            "created": created,
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
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def __del__(self):
        llama_cpp.llama_free(self.ctx)
