import multiprocessing
import numpy as np
from huggingface_hub import hf_hub_download
import llama_cpp
import llama_cpp._internals as internals

# Diagnostic comparison between low-level and high-level first token sampling.
#
# Creates two low-level contexts (n_ctx=16,32) and mirrors the sampler chain used in
# tests/test_llama.py::test_real_model, then compares first sampled token and
# full 4-token continuation against the high-level API with identical parameters.
#
# Run:
#   python scripts/diagnostics_compare.py
# Optional env:
#   LLAMA_CPP_DEBUG_DET=1  (already used by high-level path)

PROMPT = b"The quick brown fox jumps"
SEED = 1337
REPO_ID = "Qwen/Qwen2-0.5B-Instruct-GGUF"
FILENAME = "qwen2-0_5b-instruct-q8_0.gguf"

model_path = hf_hub_download(REPO_ID, FILENAME)

params = llama_cpp.llama_model_default_params()
params.use_mmap = llama_cpp.llama_supports_mmap()
params.use_mlock = llama_cpp.llama_supports_mlock()
params.check_tensors = False
model = internals.LlamaModel(path_model=model_path, params=params)

print("Low-level comparisons:")
low_level_logits = {}
for ctx_n in (16, 32):
    cparams = llama_cpp.llama_context_default_params()
    cparams.n_ctx = ctx_n
    cparams.n_batch = ctx_n
    cparams.n_ubatch = ctx_n
    cparams.n_threads = multiprocessing.cpu_count()
    cparams.n_threads_batch = multiprocessing.cpu_count()
    cparams.logits_all = False
    cparams.flash_attn = True

    context = internals.LlamaContext(model=model, params=cparams)
    tokens = model.tokenize(PROMPT, add_bos=True, special=True)
    batch = internals.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

    sampler = internals.LlamaSampler()
    sampler.add_top_k(50)
    sampler.add_top_p(0.9, 1)
    sampler.add_temp(0.8)
    sampler.add_dist(SEED)

    toks = tokens.copy()
    n_eval = 0
    result = tokens.copy()
    first_tok = None

    # Step 0: evaluate prompt only, capture logits for last prompt token
    batch.set_batch(toks, n_past=n_eval, logits_all=False)
    context.decode(batch)
    n_eval += len(toks)
    ll_ptr = context.get_logits_ith(-1)
    if ll_ptr is not None:
        low_level_logits[ctx_n] = np.ctypeslib.as_array(ll_ptr, shape=(model.n_vocab(),)).copy()
    # Sample first token
    token_id = sampler.sample(context, -1)
    first_tok = token_id
    toks = [token_id]
    result.append(token_id)

    # Steps 1-3: sample remaining tokens (total 4 sampled tokens)
    for _ in range(3):
        batch.set_batch(toks, n_past=n_eval, logits_all=False)
        context.decode(batch)
        n_eval += len(toks)
        token_id = sampler.sample(context, -1)
        toks = [token_id]
        result.append(token_id)

    out_text = model.detokenize(result[len(tokens):], special=True).decode('utf-8', errors='ignore')
    print(f"  n_ctx={ctx_n} first_token={first_tok} continuation='{out_text}'")

print("\nHigh-level comparison (n_ctx=32):")
ll = llama_cpp.Llama(model_path,
                     n_ctx=32,
                     n_batch=32,
                     n_ubatch=32,
                     logits_all=False,
                     flash_attn=True,
                     verbose=False)

# Reproduce low-level prompt eval inside high-level object
prompt_tokens_hl = ll.tokenize(PROMPT, add_bos=True, special=True)
ll.reset()
ll.eval(prompt_tokens_hl)
hl_logits_ptr = ll._ctx.get_logits_ith(-1)  # noqa: SLF001 access internal for diagnostics
if hl_logits_ptr is not None:
    hl_logits = np.ctypeslib.as_array(hl_logits_ptr, shape=(ll._n_vocab,)).copy()  # noqa: SLF001
    # Compare with low-level n_ctx=32 logits
    ref = low_level_logits.get(32)
    if ref is not None:
        diff = np.max(np.abs(ref - hl_logits))
        print(f"  Logits diff (low-level n_ctx=32 vs high-level) max_abs={diff:.6f}")
        # Show top 5 indices where difference is largest
        delta = np.abs(ref - hl_logits)
        top_idx = np.argsort(delta)[-5:][::-1]
        print("  Top differing token ids:", [(int(i), float(delta[i])) for i in top_idx])
        # Show probabilities for target tokens 31 and 916 after softmax for sanity
        def softmax(x):
            m = np.max(x)
            e = np.exp(x - m)
            return e / np.sum(e)
        ref_p = softmax(ref)
        hl_p = softmax(hl_logits)
        for tid in (31, 916):
            print(f"    token {tid}: ref_logit={ref[tid]:.4f} hl_logit={hl_logits[tid]:.4f} ref_p={ref_p[tid]:.6f} hl_p={hl_p[tid]:.6f}")

resp = ll.create_completion("The quick brown fox jumps", max_tokens=4, top_k=50, top_p=0.9, temperature=0.8, seed=SEED, stream=False)
if isinstance(resp, dict):
    print("  high_level_text=", resp['choices'][0]['text'])
else:
    # If somehow streaming iterator was returned, consume first
    collected = None
    for part in resp:
        collected = part
    if collected:
        print("  high_level_text=", collected['choices'][0]['text'])
print("Done.")
