# OpenAI Compatible Server

`llama-cpp-python` offers an OpenAI API compatible web server.

This web server can be used to serve local models and easily connect them to existing clients.

## Setup

### Installation

The server can be installed by running the following command:

```bash
pip install llama-cpp-python[server]
```

### Running the server

The server can then be started by running the following command:

```bash
python3 -m llama_cpp.server --model <model_path>
```

### Server options

For a full list of options, run:

```bash
python3 -m llama_cpp.server --help
```

NOTE: All server options are also available as environment variables. For example, `--model` can be set by setting the `MODEL` environment variable.

## Guides

### Multi-modal Models

`llama-cpp-python` supports the llava1.5 family of multi-modal models which allow the language model to
read information from both text and images.

You'll first need to download one of the available multi-modal models in GGUF format:

- [llava1.5 7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
- [llava1.5 13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)

Then when you run the server you'll need to also specify the path to the clip model used for image embedding and the `llava-1-5` chat_format

```bash
python3 -m llama_cpp.server --model <model_path> --clip-model-path <clip_model_path> --chat-format llava-1-5
```

Then you can just use the OpenAI API as normal

```python3
from openai import OpenAI

client = OpenAI(base_url="http://<host>:<port>/v1", api_key="sk-xxx")
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "<image_url>"
                    },
                },
                {"type": "text", "text": "What does the image say"},
            ],
        }
    ],
)
print(response)
```