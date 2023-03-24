""" Example of low-level API inference based on https://github.com/ggerganov/llama.cpp/issues/384#issuecomment-1480129622 
"""
import multiprocessing

import llama_cpp

N_THREADS = multiprocessing.cpu_count()

prompt = b"\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"

lparams = llama_cpp.llama_context_default_params()

ctx = llama_cpp.llama_init_from_file(b"models/ggml-alpaca-7b-q4.bin", lparams)
tmp = [0, 1, 2, 3]
llama_cpp.llama_eval(ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, N_THREADS)

embd_inp = (llama_cpp.llama_token * (len(prompt) + 1))()
n_of_tok = llama_cpp.llama_tokenize(ctx, prompt, embd_inp, len(embd_inp), True)
embd_inp = embd_inp[:n_of_tok]

for i in range(len(embd_inp)):
    llama_cpp.llama_eval(ctx, (llama_cpp.c_int * 1)(embd_inp[i]), 1, i, N_THREADS)

prediction = b""
embd = embd_inp

n = 8

for i in range(n):
    id = llama_cpp.llama_sample_top_p_top_k(ctx, (llama_cpp.c_int * len(embd))(*embd), n_of_tok + i, 40, 0.8, 0.2, 1.0/0.85)

    embd.append(id)

    prediction += llama_cpp.llama_token_to_str(ctx, id)

    llama_cpp.llama_eval(ctx, (llama_cpp.c_int * 1)(embd[-1]), 1, len(embd), N_THREADS)


llama_cpp.llama_free(ctx)

print(prediction.decode("utf-8"))