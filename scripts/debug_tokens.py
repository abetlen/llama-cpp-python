import os, multiprocessing
from huggingface_hub import hf_hub_download
import llama_cpp, llama_cpp._internals as internals

repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
filename = "qwen2-0_5b-instruct-q8_0.gguf"
model_path = hf_hub_download(repo_id, filename)

PROMPT = b"The quick brown fox jumps"
seed = 1337

# Low-level setup (mirrors test_real_model)
params = llama_cpp.llama_model_default_params()
params.use_mmap = llama_cpp.llama_supports_mmap()
params.use_mlock = False
params.check_tensors = False
model = internals.LlamaModel(path_model=model_path, params=params)

cparams = llama_cpp.llama_context_default_params()
cparams.n_ctx = 32
cparams.n_batch = 32
cparams.n_ubatch = 32
cparams.n_threads = multiprocessing.cpu_count()
cparams.n_threads_batch = multiprocessing.cpu_count()
cparams.logits_all = False
cparams.flash_attn = True
context = internals.LlamaContext(model=model, params=cparams)

low_tokens = model.tokenize(PROMPT, add_bos=True, special=True)
print("low_level_tokens", low_tokens)

sampler = internals.LlamaSampler()
sampler.add_top_k(50)
sampler.add_top_p(0.9, 1)
sampler.add_temp(0.8)
sampler.add_dist(seed)

# Evaluate prompt once (like test_real_model loop first iteration)
batch = internals.LlamaBatch(n_tokens=len(low_tokens), embd=0, n_seq_max=1)
batch.set_batch(low_tokens, n_past=0, logits_all=False)
context.decode(batch)

# Grab logits pointer for last token of prompt (not exposed since logits_all False, so we re-run with True)
# Re-run with logits_all True to capture distribution for last prompt token
cparams2 = llama_cpp.llama_context_default_params()
for attr in ("n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch"):
    setattr(cparams2, attr, getattr(cparams, attr))
cparams2.logits_all = True
context2 = internals.LlamaContext(model=model, params=cparams2)
batch2 = internals.LlamaBatch(n_tokens=len(low_tokens), embd=0, n_seq_max=1)
batch2.set_batch(low_tokens, n_past=0, logits_all=True)
context2.decode(batch2)
import numpy as np

rows = len(low_tokens)
cols = model.n_vocab()
raw_logits = context2.get_logits()
logits = np.ctypeslib.as_array(raw_logits, shape=(rows * cols,))
last_logits = logits[(rows - 1) * cols : rows * cols]
# Extract logits for tokens 31 and 916
print(
    "low_level_last_logits_selected",
    {31: float(last_logits[31]), 916: float(last_logits[916])},
)
# Top10 tokens
top_indices = np.argsort(last_logits)[-10:][::-1]
print("low_level_top10", list(map(int, top_indices)))

# Now high-level API path
llama = llama_cpp.Llama(
    model_path, n_ctx=32, n_batch=32, n_ubatch=32, logits_all=True, flash_attn=True
)
# Simulate _create_completion prompt tokenization logic
add_bos_flag = llama._model.add_bos_token()
hi_tokens_add = llama.tokenize(PROMPT, add_bos=True, special=True)
hi_tokens_no = llama.tokenize(PROMPT, add_bos=False, special=True)
print("high_level_add_bos_tokens", hi_tokens_add)
print("high_level_no_bos_tokens", hi_tokens_no)
constructed_prompt_tokens = (
    [] if (not add_bos_flag) else [llama.token_bos()]
) + hi_tokens_no
print("constructed_prompt_tokens", constructed_prompt_tokens)

# Evaluate using llama.eval directly to capture logits for prompt
llama.eval(constructed_prompt_tokens)
if llama._logits_all:
    prompt_last_logits = llama._scores[llama.n_tokens - 1, :]
    print(
        "high_level_prompt_last_logits_selected",
        {31: float(prompt_last_logits[31]), 916: float(prompt_last_logits[916])},
    )
    top_hi = np.argsort(prompt_last_logits)[-10:][::-1]
    print("high_level_prompt_top10", list(map(int, top_hi)))
else:
    print("ERROR: logits_all not enabled for high-level eval")

print("DONE")
