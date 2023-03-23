import uuid
import time
import multiprocessing
from typing import List, Optional

from . import llama_cpp

class Llama:
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
        model_name: Optional[str]=None,
    ):
        self.model_path = model_path
        self.model = model_name or model_path

        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = n_ctx
        self.params.n_parts = n_parts
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only

        self.n_threads = n_threads or multiprocessing.cpu_count()

        self.tokens = (llama_cpp.llama_token * self.params.n_ctx)()

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
        text = b""
        finish_reason = "length"
        completion_tokens = 0

        if stop is not None:
            stop = [s.encode("utf-8") for s in stop]

        prompt_tokens = llama_cpp.llama_tokenize(
            self.ctx, prompt.encode("utf-8"), self.tokens, self.params.n_ctx, True
        )

        if prompt_tokens + max_tokens > self.params.n_ctx:
            raise ValueError(
                f"Requested tokens exceed context window of {self.params.n_ctx}"
            )

        for i in range(prompt_tokens):
            llama_cpp.llama_eval(
                self.ctx, (llama_cpp.c_int * 1)(self.tokens[i]), 1, i, self.n_threads
            )

        for i in range(max_tokens):
            token = llama_cpp.llama_sample_top_p_top_k(
                self.ctx,
                self.tokens,
                prompt_tokens + completion_tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty,
            )
            if token == llama_cpp.llama_token_eos():
                finish_reason = "stop"
                break
            # text += llama_cpp.llama_token_to_str(self.ctx, token).decode("utf-8")
            text += llama_cpp.llama_token_to_str(self.ctx, token)
            self.tokens[prompt_tokens + i] = token
            completion_tokens += 1

            any_stop = [s for s in stop if s in text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = text[: text.index(first_stop)]
                finish_reason = "stop"
                break

            llama_cpp.llama_eval(
                self.ctx,
                (llama_cpp.c_int * 1)(self.tokens[prompt_tokens + i]),
                1,
                prompt_tokens + completion_tokens,
                self.n_threads,
            )

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
            "id": f"cmpl-{str(uuid.uuid4())}", # Likely to change
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model, # Likely to change
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


