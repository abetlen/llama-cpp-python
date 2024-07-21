"""llama-cpp-python server from scratch in a single file.
"""

# import llama_cpp

# path = b"../../models/Qwen1.5-0.5B-Chat-GGUF/qwen1_5-0_5b-chat-q8_0.gguf"

# model_params = llama_cpp.llama_model_default_params()
# model = llama_cpp.llama_load_model_from_file(path, model_params)

# if model is None:
#     raise RuntimeError(f"Failed to load model from file: {path}")


# ctx_params = llama_cpp.llama_context_default_params()
# ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

# if ctx is None:
#     raise RuntimeError("Failed to create context")


from fastapi import FastAPI

app = FastAPI()

import openai.types.chat as types


@app.post("/v1/chat/completions")
def create_chat_completions():
    return {"message": "Hello World"}
