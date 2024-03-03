# 🦙 Python Bindings for [`llama.cpp`](https://github.com/ggerganov/llama.cpp)

[![Documentation Status](https://readthedocs.org/projects/llama-cpp-python/badge/?version=latest)](https://llama-cpp-python.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/abetlen/llama-cpp-python/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/abetlen/llama-cpp-python/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/llama-cpp-python)](https://pypi.org/project/llama-cpp-python/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llama-cpp-python)](https://pypi.org/project/llama-cpp-python/)
[![PyPI - License](https://img.shields.io/pypi/l/llama-cpp-python)](https://pypi.org/project/llama-cpp-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-cpp-python)](https://pypi.org/project/llama-cpp-python/)

Simple Python bindings for **@ggerganov's** [`llama.cpp`](https://github.com/ggerganov/llama.cpp) library.
This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for text completion
    - OpenAI-like API
    - [LangChain compatibility](https://python.langchain.com/docs/integrations/llms/llamacpp)
    - [LlamaIndex compatibility](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html)
- OpenAI compatible web server
    - [Local Copilot replacement](https://llama-cpp-python.readthedocs.io/en/latest/server/#code-completion)
    - [Function Calling support](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)
    - [Vision API support](https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models)
    - [Multiple Models](https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support)

Documentation is available at [https://llama-cpp-python.readthedocs.io/en/latest](https://llama-cpp-python.readthedocs.io/en/latest).

## Installation

Requirements:

  - Python 3.8+
  - C compiler
      - Linux: gcc or clang
      - Windows: Visual Studio or MinGW
      - MacOS: Xcode

To install the package, run:

```bash
pip install llama-cpp-python
```

This will also build `llama.cpp` from source and install it alongside this python package.

If this fails, add `--verbose` to the `pip install` see the full cmake build log.

### Installation Configuration

`llama.cpp` supports a number of hardware acceleration backends to speed up inference as well as backend specific options. See the [llama.cpp README](https://github.com/ggerganov/llama.cpp#build) for a full list.

All `llama.cpp` cmake build options can be set via the `CMAKE_ARGS` environment variable or via the `--config-settings / -C` cli flag during installation.

<details open>
<summary>Environment Variables</summary>

```bash
# Linux and Mac
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python
```

```powershell
# Windows
$env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python
```
</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install llama-cpp-python \
  -C cmake.args="-DLLAMA_BLAS=ON;-DLLAMA_BLAS_VENDOR=OpenBLAS"
```

```txt
# requirements.txt

llama-cpp-python -C cmake.args="-DLLAMA_BLAS=ON;-DLLAMA_BLAS_VENDOR=OpenBLAS"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<details open>
<summary>OpenBLAS (CPU)</summary>

To install with OpenBLAS, set the `LLAMA_BLAS` and `LLAMA_BLAS_VENDOR` environment variables before installing:

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```
</details>

<details>
<summary>cuBLAS (CUDA)</summary>

To install with cuBLAS, set the `LLAMA_CUBLAS=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

</details>

<details>
<summary>Metal</summary>

To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

</details>
<details>

<summary>CLBlast (OpenCL)</summary>

To install with CLBlast, set the `LLAMA_CLBLAST=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python
```

</details>

<details>
<summary>hipBLAS (ROCm)</summary>

To install with hipBLAS / ROCm support for AMD cards, set the `LLAMA_HIPBLAS=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

</details>

<details>
<summary>Vulkan</summary>

To install with Vulkan support, set the `LLAMA_VULKAN=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python
```

</details>

<details>
<summary>Kompute</summary>

To install with Kompute support, set the `LLAMA_KOMPUTE=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_KOMPUTE=on" pip install llama-cpp-python
```
</details>

<details>
<summary>SYCL</summary>

To install with SYCL support, set the `LLAMA_SYCL=on` environment variable before installing:

```bash
source /opt/intel/oneapi/setvars.sh   
CMAKE_ARGS="-DLLAMA_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install llama-cpp-python
```
</details>


### Windows Notes

<details>
<summary>Error: Can't find 'nmake' or 'CMAKE_C_COMPILER'</summary>

If you run into issues where it complains it can't find `'nmake'` `'?'` or CMAKE_C_COMPILER, you can extract w64devkit as [mentioned in llama.cpp repo](https://github.com/ggerganov/llama.cpp#openblas) and add those manually to CMAKE_ARGS before running `pip` install:

```ps
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
```

See the above instructions and set `CMAKE_ARGS` to the BLAS backend you want to use.
</details>

### MacOS Notes

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

<details>
<summary>M1 Mac Performance Issue</summary>

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Otherwise, while installing it will build the llama.cpp x86 version which will be 10x slower on Apple Silicon (M1) Mac.
</details>

<details>
<summary>M Series Mac Error: `(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))`</summary>

Try installing with

```bash
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```
</details>

### Upgrading and Reinstalling

To upgrade and rebuild `llama-cpp-python` add `--upgrade --force-reinstall --no-cache-dir` flags to the `pip install` command to ensure the package is rebuilt from source.

## High-level API

[API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api)

The high-level API provides a simple managed interface through the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

Below is a short example demonstrating how to use the high-level API to for basic text completion:

```python
>>> from llama_cpp import Llama
>>> llm = Llama(
      model_path="./models/7B/llama-model.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
>>> output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
>>> print(output)
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "./models/7B/llama-model.gguf",
  "choices": [
    {
      "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
      "index": 0,
      "logprobs": None,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 28,
    "total_tokens": 42
  }
}
```

Text completion is available through the [`__call__`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__) and [`create_completion`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion) methods of the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

### Pulling models from Hugging Face Hub

You can download `Llama` models in `gguf` format directly from Hugging Face using the [`from_pretrained`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.from_pretrained) method.
You'll need to install the `huggingface-hub` package to use this feature (`pip install huggingface-hub`).

```python
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)
```

By default [`from_pretrained`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.from_pretrained) will download the model to the huggingface cache directory, you can then manage installed model files with the [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) tool.

### Chat Completion

The high-level API also provides a simple interface for chat completion.

Chat completion requires that the model know how to format the messages into a single prompt.
The `Llama` class does this using pre-registered chat formats (ie. `chatml`, `llama-2`, `gemma`, etc) or by providing a custom chat handler object.

The model will will format the messages into a single prompt using the following order of precedence:
  - Use the `chat_handler` if provided
  - Use the `chat_format` if provided
  - Use the `tokenizer.chat_template` from the `gguf` model's metadata (should work for most new models, older models may not have this)
  - else, fallback to the `llama-2` chat format

Set `verbose=True` to see the selected chat format.

```python
>>> from llama_cpp import Llama
>>> llm = Llama(
      model_path="path/to/llama-2/llama-model.gguf",
      chat_format="llama-2"
)
>>> llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "Describe this image in detail please."
          }
      ]
)
```

Chat completion is available through the [`create_chat_completion`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion) method of the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

For OpenAI API v1 compatibility, you use the [`create_chat_completion_openai_v1`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion_openai_v1) method which will return pydantic models instead of dicts.


### JSON and JSON Schema Mode

To constrain chat responses to only valid JSON or a specific JSON Schema use the `response_format` argument in [`create_chat_completion`](http://localhost:8000/api-reference/#llama_cpp.Llama.create_chat_completion).

#### JSON Mode

The following example will constrain the response to valid JSON strings only.

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="path/to/model.gguf", chat_format="chatml")
>>> llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
    },
    temperature=0.7,
)
```

#### JSON Schema Mode

To constrain the response further to a specific JSON Schema add the schema to the `schema` property of the `response_format` argument.

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="path/to/model.gguf", chat_format="chatml")
>>> llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"team_name": {"type": "string"}},
            "required": ["team_name"],
        },
    },
    temperature=0.7,
)
```

### Function Calling

The high-level API supports OpenAI compatible function and tool calling. This is possible through the `functionary` pre-trained models chat format or through the generic `chatml-function-calling` chat format.

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="path/to/chatml/llama-model.gguf", chat_format="chatml-function-calling")
>>> llm.create_chat_completion(
      messages = [
        {
          "role": "system",
          "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

        },
        {
          "role": "user",
          "content": "Extract Jason is 25 years old"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
      tool_choice=[{
        "type": "function",
        "function": {
          "name": "UserDetail"
        }
      }]
)
```

<details>
<summary>Functionary v2</summary>

The various gguf-converted files for this set of models can be found [here](https://huggingface.co/meetkai). Functionary is able to intelligently call functions and also analyze any provided function outputs to generate coherent responses. All v2 models of functionary supports **parallel function calling**. You can provide either `functionary-v1` or `functionary-v2` for the `chat_format` when initializing the Llama class.

Due to discrepancies between llama.cpp and HuggingFace's tokenizers, it is required to provide HF Tokenizer for functionary. The `LlamaHFTokenizer` class can be initialized and passed into the Llama class. This will override the default llama.cpp tokenizer used in Llama class. The tokenizer files are already included in the respective HF repositories hosting the gguf files.

```python
>>> from llama_cpp import Llama
>>> from llama_cpp.llama_tokenizer import LlamaHFTokenizer
>>> llm = Llama.from_pretrained(
  repo_id="meetkai/functionary-small-v2.2-GGUF",
  filename="functionary-small-v2.2.q4_0.gguf",
  chat_format="functionary-v2",
  tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.2-GGUF")
)
```
</details>

### Multi-modal Models

`llama-cpp-python` supports the llava1.5 family of multi-modal models which allow the language model to
read information from both text and images.

You'll first need to download one of the available multi-modal models in GGUF format:

- [llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
- [llava-v1.5-13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [bakllava-1-7b](https://huggingface.co/mys/ggml_bakllava-1)

Then you'll need to use a custom chat handler to load the clip model and process the chat messages and images.

```python
>>> from llama_cpp import Llama
>>> from llama_cpp.llama_chat_format import Llava15ChatHandler
>>> chat_handler = Llava15ChatHandler(clip_model_path="path/to/llava/mmproj.bin")
>>> llm = Llama(
  model_path="./path/to/llava/llama-model.gguf",
  chat_handler=chat_handler,
  n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
  logits_all=True,# needed to make llava work
)
>>> llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://.../image.png"}},
                {"type" : "text", "text": "Describe this image in detail please."}
            ]
        }
    ]
)
```

<details>
<summary>Loading a Local Image</summary>

Images can be passed as base64 encoded data URIs. The following example demonstrates how to do this.

```python
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
file_path = 'file_path.png'
data_uri = image_to_base64_data_uri(file_path)

messages = [
    {"role": "system", "content": "You are an assistant who perfectly describes images."},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri }},
            {"type" : "text", "text": "Describe this image in detail please."}
        ]
    }
]

```

</details>

### Speculative Decoding

`llama-cpp-python` supports speculative decoding which allows the model to generate completions based on a draft model.

The fastest way to use speculative decoding is through the `LlamaPromptLookupDecoding` class.

Just pass this as a draft model to the `Llama` class during initialization.

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

llama = Llama(
    model_path="path/to/model.gguf",
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10) # num_pred_tokens is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines.
)
```

### Embeddings

To generate text embeddings use [`create_embedding`](http://localhost:8000/api-reference/#llama_cpp.Llama.create_embedding).

```python
import llama_cpp

llm = llama_cpp.Llama(model_path="path/to/model.gguf", embedding=True)

embeddings = llm.create_embedding("Hello, world!")

# or create multiple embeddings at once

embeddings = llm.create_embedding(["Hello, world!", "Goodbye, world!"])
```

### Adjusting the Context Window

The context window of the Llama models determines the maximum number of tokens that can be processed at once. By default, this is set to 512 tokens, but can be adjusted based on your requirements.

For instance, if you want to work with larger contexts, you can expand the context window by setting the n_ctx parameter when initializing the Llama object:

```python
llm = Llama(model_path="./models/7B/llama-model.gguf", n_ctx=2048)
```

## OpenAI Compatible Web Server

`llama-cpp-python` offers a web server which aims to act as a drop-in replacement for the OpenAI API.
This allows you to use llama.cpp compatible models with any OpenAI compatible client (language libraries, services, etc).

To install the server package and get started:

```bash
pip install 'llama-cpp-python[server]'
python3 -m llama_cpp.server --model models/7B/llama-model.gguf
```

Similar to Hardware Acceleration section above, you can also install with GPU (cuBLAS) support like this:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install 'llama-cpp-python[server]'
python3 -m llama_cpp.server --model models/7B/llama-model.gguf --n_gpu_layers 35
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to see the OpenAPI documentation.

To bind to `0.0.0.0` to enable remote connections, use `python3 -m llama_cpp.server --host 0.0.0.0`.
Similarly, to change the port (default is 8000), use `--port`.

You probably also want to set the prompt format. For chatml, use

```bash
python3 -m llama_cpp.server --model models/7B/llama-model.gguf --chat_format chatml
```

That will format the prompt according to how model expects it. You can find the prompt format in the model card.
For possible options, see [llama_cpp/llama_chat_format.py](llama_cpp/llama_chat_format.py) and look for lines starting with "@register_chat_format".

If you have `huggingface-hub` installed, you can also use the `--hf_model_repo_id` flag to load a model from the Hugging Face Hub.

```bash
python3 -m llama_cpp.server --hf_model_repo_id Qwen/Qwen1.5-0.5B-Chat-GGUF --model '*q8_0.gguf'
```

### Web Server Features

- [Local Copilot replacement](https://llama-cpp-python.readthedocs.io/en/latest/server/#code-completion)
- [Function Calling support](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)
- [Vision API support](https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models)
- [Multiple Models](https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support)

## Docker image

A Docker image is available on [GHCR](https://ghcr.io/abetlen/llama-cpp-python). To run the server:

```bash
docker run --rm -it -p 8000:8000 -v /path/to/models:/models -e MODEL=/models/llama-model.gguf ghcr.io/abetlen/llama-cpp-python:latest
```

[Docker on termux (requires root)](https://gist.github.com/FreddieOliveira/efe850df7ff3951cb62d74bd770dce27) is currently the only known way to run this on phones, see [termux support issue](https://github.com/abetlen/llama-cpp-python/issues/389)

## Low-level API

[API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#low-level-api)

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `llama.cpp`.
The entire low-level API can be found in [llama_cpp/llama_cpp.py](https://github.com/abetlen/llama-cpp-python/blob/master/llama_cpp/llama_cpp.py) and directly mirrors the C API in [llama.h](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).

Below is a short example demonstrating how to use the low-level API to tokenize a prompt:

```python
>>> import llama_cpp
>>> import ctypes
>>> llama_cpp.llama_backend_init(False) # Must be called once at the start of each program
>>> params = llama_cpp.llama_context_default_params()
# use bytes for char * params
>>> model = llama_cpp.llama_load_model_from_file(b"./models/7b/llama-model.gguf", params)
>>> ctx = llama_cpp.llama_new_context_with_model(model, params)
>>> max_tokens = params.n_ctx
# use ctypes arrays for array params
>>> tokens = (llama_cpp.llama_token * int(max_tokens))()
>>> n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, llama_cpp.c_bool(True))
>>> llama_cpp.llama_free(ctx)
```

Check out the [examples folder](examples/low_level_api) for more examples of using the low-level API.

## Documentation

Documentation is available via [https://llama-cpp-python.readthedocs.io/](https://llama-cpp-python.readthedocs.io/).
If you find any issues with the documentation, please open an issue or submit a PR.

## Development

This package is under active development and I welcome any contributions.

To get started, clone the repository and install the package in editable / development mode:

```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python

# Upgrade pip (required for editable mode)
pip install --upgrade pip

# Install with pip
pip install -e .

# if you want to use the fastapi / openapi server
pip install -e .[server]

# to install all optional dependencies
pip install -e .[all]

# to clear the local build cache
make clean
```

You can also test out specific commits of `lama.cpp` by checking out the desired commit in the `vendor/llama.cpp` submodule and then running `make clean` and `pip install -e .` again. Any changes in the `llama.h` API will require
changes to the `llama_cpp/llama_cpp.py` file to match the new API (additional changes may be required elsewhere).

## FAQ

### Are there pre-built binaries / binary wheels available?

The recommended installation method is to install from source as described above.
The reason for this is that `llama.cpp` is built with compiler optimizations that are specific to your system.
Using pre-built binaries would require disabling these optimizations or supporting a large number of pre-built binaries for each platform.

That being said there are some pre-built binaries available through the Releases as well as some community provided wheels.

In the future, I would like to provide pre-built binaries and wheels for common platforms and I'm happy to accept any useful contributions in this area.
This is currently being tracked in [#741](https://github.com/abetlen/llama-cpp-python/issues/741)

### How does this compare to other Python bindings of `llama.cpp`?

I originally wrote this package for my own use with two goals in mind:

- Provide a simple process to install `llama.cpp` and access the full C API in `llama.h` from Python
- Provide a high-level Python API that can be used as a drop-in replacement for the OpenAI API so existing apps can be easily ported to use `llama.cpp`

Any contributions and changes to this package will be made with these goals in mind.

## License

This project is licensed under the terms of the MIT license.
