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

### Code Completion

`llama-cpp-python` supports code completion via GitHub Copilot.

*NOTE*: Without GPU acceleration this is unlikely to be fast enough to be usable.

You'll first need to download one of the available code completion models in GGUF format:

- [replit-code-v1_5-GGUF](https://huggingface.co/abetlen/replit-code-v1_5-3b-GGUF)

Then you'll need to run the OpenAI compatible web server with a increased context size substantially for GitHub Copilot requests:

```bash
python3 -m llama_cpp.server --model <model_path> --n_ctx 16192
```

Then just update your settings in `.vscode/settings.json` to point to your code completion server:

```json
{
    // ...
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "http://<host>:<port>",
        "debug.overrideProxyUrl": "http://<host>:<port>"
    }
    // ...
}
```

### Function Calling

`llama-cpp-python` supports structured function calling based on a JSON schema.
Function calling is completely compatible with the OpenAI function calling API and can be used by connecting with the official OpenAI Python client.

You'll first need to download one of the available function calling models in GGUF format:

- [functionary-7b-v1](https://huggingface.co/abetlen/functionary-7b-v1-GGUF)

Then when you run the server you'll need to also specify the `functionary` chat_format

```bash
python3 -m llama_cpp.server --model <model_path> --chat_format functionary
```

Check out this [example notebook](https://github.com/abetlen/llama-cpp-python/blob/main/examples/notebooks/Functions.ipynb) for a walkthrough of some interesting use cases for function calling.

### Multimodal Models

`llama-cpp-python` supports the llava1.5 family of multi-modal models which allow the language model to
read information from both text and images.

You'll first need to download one of the available multi-modal models in GGUF format:

- [llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
- [llava-v1.5-13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [bakllava-1-7b](https://huggingface.co/mys/ggml_bakllava-1)

Then when you run the server you'll need to also specify the path to the clip model used for image embedding and the `llava-1-5` chat_format

```bash
python3 -m llama_cpp.server --model <model_path> --clip_model_path <clip_model_path> --chat_format llava-1-5
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