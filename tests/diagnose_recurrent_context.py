import hashlib
import multiprocessing
import os

from huggingface_hub import hf_hub_download

import llama_cpp


model_path = hf_hub_download(
    repo_id="QuantFactory/mamba-130m-hf-GGUF",
    filename="mamba-130m-hf.Q2_K.gguf",
)

with open(model_path, "rb") as model_file:
    model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()

print(f"model_path={model_path}", flush=True)
print(f"model_sha256={model_sha256}", flush=True)
print(llama_cpp.llama_print_system_info().decode(), flush=True)

verbose = os.environ.get("VERBOSE") == "1"
print(f"verbose={verbose}", flush=True)

model = llama_cpp.Llama(
    model_path,
    n_ctx=32,
    n_batch=32,
    n_ubatch=32,
    n_threads=multiprocessing.cpu_count(),
    n_threads_batch=multiprocessing.cpu_count(),
    logits_all=False,
    verbose=verbose,
)
print("context construction succeeded", flush=True)
model.close()
