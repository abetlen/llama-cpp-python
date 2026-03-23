import llama_cpp
import llama_cpp.llama_tokenizer


llama = llama_cpp.Llama.from_pretrained(
    repo_id="lmstudio-community/Qwen3.5-0.8B-GGUF",
    filename="*Q8_0.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B"
    ),
    verbose=False,
)

response = llama.create_chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"},
            },
            "required": ["country", "capital"],
        },
    },
    stream=True,
)

for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" not in delta:
        continue
    print(delta["content"], end="", flush=True)

print()
