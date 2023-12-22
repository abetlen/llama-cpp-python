import ctypes
import os
import multiprocessing

import llama_cpp

llama_cpp.llama_backend_init(numa=False)

N_THREADS = multiprocessing.cpu_count()
MODEL_PATH = os.environ.get('MODEL', "../models/7B/ggml-model.bin")

prompt = b"\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"

lparams = llama_cpp.llama_model_default_params()
cparams = llama_cpp.llama_context_default_params()
model = llama_cpp.llama_load_model_from_file(MODEL_PATH.encode('utf-8'), lparams)
ctx = llama_cpp.llama_new_context_with_model(model, cparams)

# determine the required inference memory per token:
tmp = [0, 1, 2, 3]
llama_cpp.llama_eval(
            ctx = ctx, 
            tokens=(llama_cpp.c_int * len(tmp))(*tmp),
            n_tokens=len(tmp),
            n_past=0
        )# Deprecated

n_past = 0

prompt = b" " + prompt

embd_inp = (llama_cpp.llama_token * (len(prompt) + 1))()
n_of_tok = llama_cpp.llama_tokenize(
    model=model,
    text=bytes(str(prompt),'utf-8'),
    text_len=len(embd_inp), 
    tokens=embd_inp,
    n_max_tokens=len(embd_inp),
    add_bos=False,
    special=False
)
embd_inp = embd_inp[:n_of_tok]

n_ctx = llama_cpp.llama_n_ctx(ctx)

n_predict = 20
n_predict = min(n_predict, n_ctx - len(embd_inp))

input_consumed = 0
input_noecho = False

remaining_tokens = n_predict

embd = []
last_n_size = 64
last_n_tokens_data = [0] * last_n_size
n_batch = 24
last_n_repeat = 64
repeat_penalty = 1
frequency_penalty = 0.0
presence_penalty = 0.0

while remaining_tokens > 0:
    if len(embd) > 0:
        llama_cpp.llama_eval(
            ctx = ctx, 
            tokens=(llama_cpp.c_int * len(embd))(*embd),
            n_tokens=len(embd),
            n_past=n_past
        )# Deprecated

    n_past += len(embd)
    embd = []
    if len(embd_inp) <= input_consumed:
        logits = llama_cpp.llama_get_logits(ctx)
        n_vocab = llama_cpp.llama_n_vocab(model)

        _arr = (llama_cpp.llama_token_data * n_vocab)(*[
            llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
            for token_id in range(n_vocab)
        ])
        candidates_p = llama_cpp.ctypes.pointer(
            llama_cpp.llama_token_data_array(_arr, len(_arr), False))

        _arr = (llama_cpp.c_int * len(last_n_tokens_data))(*last_n_tokens_data)
        llama_cpp.llama_sample_repetition_penalties(ctx, candidates_p,
            _arr,
            penalty_last_n=last_n_repeat,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty)

        llama_cpp.llama_sample_top_k(ctx, candidates_p, k=40, min_keep=1)
        llama_cpp.llama_sample_top_p(ctx, candidates_p, p=0.8, min_keep=1)
        llama_cpp.llama_sample_temperature(ctx, candidates_p, temp=0.2)
        id = llama_cpp.llama_sample_token(ctx, candidates_p)

        last_n_tokens_data = last_n_tokens_data[1:] + [id]
        embd.append(id)
        input_noecho = False
        remaining_tokens -= 1
    else:
        while len(embd_inp) > input_consumed:
            embd.append(embd_inp[input_consumed])
            last_n_tokens_data = last_n_tokens_data[1:] + [embd_inp[input_consumed]]
            input_consumed += 1
            if len(embd) >= n_batch:
                break
    if not input_noecho:
        for id in embd:
            size = 32
            buffer = (ctypes.c_char * size)()
            n = llama_cpp.llama_token_to_piece(
                model, llama_cpp.llama_token(id), buffer, size)
            assert n <= size
            print(
                buffer[:n].decode('utf-8'),
                end="",
                flush=True,
            )

    if len(embd) > 0 and embd[-1] == llama_cpp.llama_token_eos(ctx):
        break

print()

llama_cpp.llama_print_timings(ctx)

llama_cpp.llama_free(ctx)
