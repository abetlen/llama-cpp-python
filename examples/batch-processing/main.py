import llama_cpp

from contextlib import ExitStack

from huggingface_hub import hf_hub_download # type: ignore

repo_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
filename = "qwen1_5-0_5b-chat-q8_0.gguf"

hf_hub_download(repo_id=repo_id, filename=filename)
model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)

with ExitStack() as es:
    llama_cpp.llama_backend_init()
    es.callback(llama_cpp.llama_backend_free)

    model_params = llama_cpp.llama_model_default_params()

    model_params.use_lock = llama_cpp.llama_supports_mlock()
    model_params.use_mmap = llama_cpp.llama_supports_mmap()

    model = llama_cpp.llama_load_model_from_file(
        model_path.encode('utf-8'),
        model_params
    )
    assert model is not None
    es.callback(lambda: llama_cpp.llama_free_model(model))

    context_params = llama_cpp.llama_context_default_params()
    context = llama_cpp.llama_new_context_with_model(model, context_params)
