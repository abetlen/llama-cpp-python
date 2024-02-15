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

Check out the server config reference below settings for more information on the available options.
CLI arguments and environment variables are available for all of the fields defined in [`ServerSettings`](#llama_cpp.server.settings.ServerSettings) and [`ModelSettings`](#llama_cpp.server.settings.ModelSettings) 

Additionally the server supports configuration check out the [configuration section](#configuration-and-multi-model-support) for more information and examples.


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

- [functionary](https://huggingface.co/meetkai)

Then when you run the server you'll need to also specify either `functionary-v1` or `functionary-v2` chat_format.

Note that since functionary requires a HF Tokenizer due to discrepancies between llama.cpp and HuggingFace's tokenizers as mentioned [here](https://github.com/abetlen/llama-cpp-python/blob/main?tab=readme-ov-file#function-calling), you will need to pass in the path to the tokenizer too. The tokenizer files are already included in the respective HF repositories hosting the gguf files.

```bash
python3 -m llama_cpp.server --model <model_path_to_functionary_v2_model> --chat_format functionary-v2 --hf_pretrained_model_name_or_path <model_path_to_functionary_v2_tokenizer>
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

## Configuration and Multi-Model Support

The server supports configuration via a JSON config file that can be passed using the `--config_file` parameter or the `CONFIG_FILE` environment variable.

```bash
python3 -m llama_cpp.server --config_file <config_file>
```

Config files support all of the server and model options supported by the cli and environment variables however instead of only a single model the config file can specify multiple models.

The server supports routing requests to multiple models based on the `model` parameter in the request which matches against the `model_alias` in the config file.

At the moment only a single model is loaded into memory at, the server will automatically load and unload models as needed.

```json
{
    "host": "0.0.0.0",
    "port": 8080,
    "models": [
        {
            "model": "models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_alias": "gpt-3.5-turbo",
            "chat_format": "chatml",
            "n_gpu_layers": -1,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_alias": "gpt-4",
            "chat_format": "chatml",
            "n_gpu_layers": -1,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/ggml_llava-v1.5-7b/ggml-model-q4_k.gguf",
            "model_alias": "gpt-4-vision-preview",
            "chat_format": "llava-1-5",
            "clip_model_path": "models/ggml_llava-v1.5-7b/mmproj-model-f16.gguf",
            "n_gpu_layers": -1,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/mistral-7b-v0.1-GGUF/ggml-model-Q4_K.gguf",
            "model_alias": "text-davinci-003",
            "n_gpu_layers": -1,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/replit-code-v1_5-3b-GGUF/replit-code-v1_5-3b.Q4_0.gguf",
            "model_alias": "copilot-codex",
            "n_gpu_layers": -1,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 1024,
            "n_ctx": 9216
        }
    ]
}
```

The config file format is defined by the [`ConfigFileSettings`](#llama_cpp.server.settings.ConfigFileSettings) class.

## Server Options Reference

::: llama_cpp.server.settings.ConfigFileSettings
    options:
        show_if_no_docstring: true

::: llama_cpp.server.settings.ServerSettings
    options:
        show_if_no_docstring: true

::: llama_cpp.server.settings.ModelSettings
    options:
        show_if_no_docstring: true
