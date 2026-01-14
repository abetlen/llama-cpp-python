from llama_cpp import Llama
from llama_cpp.llama_chat_format import SmolVLMChatHandler


chat_handler = SmolVLMChatHandler.from_pretrained(
    repo_id="ggml-org/SmolVLM-256M-Instruct-GGUF",
    filename="mmproj*Q8_0*",
)
llama = Llama.from_pretrained(
    repo_id="ggml-org/SmolVLM-256M-Instruct-GGUF",
    filename="SmolVLM*Q8_0*",
    chat_handler=chat_handler,
    n_ctx=8192,
    n_gpu_layers=-1,
)
response = llama.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "https://huggingface.co/spaces/ibm-granite/granite-docling-258m-demo/resolve/main/data/images/lake-zurich-switzerland-view-nature-landscapes-7bbda4-1024.jpg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
    stream=True,
    special=True,
)

for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" not in delta:
        continue
    print(delta["content"], end="", flush=True)

print()
