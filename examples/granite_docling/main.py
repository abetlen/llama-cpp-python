from llama_cpp import Llama
from llama_cpp.llama_chat_format import GraniteDoclingChatHandler


chat_handler = GraniteDoclingChatHandler.from_pretrained(
    repo_id="ggml-org/granite-docling-258M-GGUF",
    filename="mmproj*Q8_0*",
)
llama = Llama.from_pretrained(
    repo_id="ggml-org/granite-docling-258M-GGUF",
    filename="granite*Q8_0*",
    chat_handler=chat_handler,
    n_ctx=8192,
)
response = llama.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "https://huggingface.co/spaces/ibm-granite/granite-docling-258m-demo/resolve/main/data/images/new_arxiv.png"},
                {"type": "text", "text": "Convert this page to docling."},
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
