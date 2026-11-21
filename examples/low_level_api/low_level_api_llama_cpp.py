import codecs
import ctypes
import os
from contextlib import ExitStack

import llama_cpp

PROMPT = "\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
N_PREDICT = 20


def decode(ctx, batch, tokens, start):
    """Decode one batch from sequence zero and request logits for its last token."""
    batch.n_tokens = len(tokens)
    for i, token in enumerate(tokens):
        batch.token[i] = token
        batch.pos[i] = start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = i == len(tokens) - 1
    status = llama_cpp.llama_decode(ctx, batch)
    if status != 0:
        raise RuntimeError(f"llama_decode returned {status} at position {start}")


def token_piece(vocab, token):
    buf = ctypes.create_string_buffer(32)
    n = llama_cpp.llama_token_to_piece(vocab, token, buf, len(buf), 0, False)
    if n < 0:
        buf = ctypes.create_string_buffer(-n)
        n = llama_cpp.llama_token_to_piece(vocab, token, buf, len(buf), 0, False)
    if n < 0:
        raise RuntimeError(f"Could not convert token {token} to bytes")
    return buf.raw[:n]


def generate(model_path, prompt, n_predict):
    with ExitStack() as stack:
        llama_cpp.llama_backend_init()
        stack.callback(llama_cpp.llama_backend_free)

        model = llama_cpp.llama_model_load_from_file(
            os.fsencode(model_path), llama_cpp.llama_model_default_params()
        )
        if not model:
            raise RuntimeError(f"Could not load model: {model_path}")
        stack.callback(llama_cpp.llama_model_free, model)
        vocab = llama_cpp.llama_model_get_vocab(model)

        params = llama_cpp.llama_context_default_params()
        params.n_ctx = 512
        params.n_batch = 24
        params.no_perf = False
        ctx = llama_cpp.llama_init_from_model(model, params)
        if not ctx:
            raise RuntimeError("Could not create the model context")
        stack.callback(llama_cpp.llama_free, ctx)

        n_ctx = llama_cpp.llama_n_ctx(ctx)
        prompt_bytes = prompt.encode("utf-8")
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            vocab, prompt_bytes, len(prompt_bytes), tokens, n_ctx, True, False
        )
        if n_tokens < 0 or n_tokens >= n_ctx:
            raise ValueError(
                f"The {abs(n_tokens)}-token prompt must leave room for generation "
                f"in the {n_ctx}-token context"
            )
        if n_tokens == 0:
            raise ValueError("The prompt produced no tokens")

        n_batch = llama_cpp.llama_n_batch(ctx)
        batch = llama_cpp.llama_batch_init(n_batch, 0, 1)
        stack.callback(llama_cpp.llama_batch_free, batch)
        for start in range(0, n_tokens, n_batch):
            decode(ctx, batch, tokens[start : min(start + n_batch, n_tokens)], start)

        sampler = llama_cpp.llama_sampler_chain_init(
            llama_cpp.llama_sampler_chain_default_params()
        )
        stack.callback(llama_cpp.llama_sampler_free, sampler)
        llama_cpp.llama_sampler_chain_add(
            sampler, llama_cpp.llama_sampler_init_top_k(40)
        )
        llama_cpp.llama_sampler_chain_add(
            sampler, llama_cpp.llama_sampler_init_top_p(0.8, 1)
        )
        llama_cpp.llama_sampler_chain_add(
            sampler, llama_cpp.llama_sampler_init_temp(0.2)
        )
        llama_cpp.llama_sampler_chain_add(
            sampler, llama_cpp.llama_sampler_init_dist(llama_cpp.LLAMA_DEFAULT_SEED)
        )

        print(prompt, end="", flush=True)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        end = min(n_ctx, n_tokens + n_predict)
        for pos in range(n_tokens, end):
            token = llama_cpp.llama_sampler_sample(sampler, ctx, -1)
            if llama_cpp.llama_vocab_is_eog(vocab, token):
                break
            print(decoder.decode(token_piece(vocab, token)), end="", flush=True)
            if pos + 1 < end:
                decode(ctx, batch, [token], pos)
        print(decoder.decode(b"", final=True))
        llama_cpp.llama_perf_context_print(ctx)


if __name__ == "__main__":
    generate(os.environ.get("MODEL", "models/model.gguf"), PROMPT, N_PREDICT)
