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
- OpenAI compatible web server
    - [Local Copilot replacement](https://llama-cpp-python.readthedocs.io/en/latest/server/#code-completion)
    - [Function Calling support](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)
    - [Vision API support](https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models)
    - [Multiple Models](https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support)

Documentation is available at [https://llama-cpp-python.readthedocs.io/en/latest](https://llama-cpp-python.readthedocs.io/en/latest).




## Installation

`llama-cpp-python` can be installed directly from PyPI as a source distribution by running:

```bash
pip install llama-cpp-python
```

This will build `llama.cpp` from source using cmake and your system's c compiler (required) and install the library alongside this python package.

If you run into issues during installation add the `--verbose` flag to the `pip install` command to see the full cmake build log.


### Installation with Specific Hardware Acceleration (BLAS, CUDA, Metal, etc)

The default pip install behaviour is to build `llama.cpp` for CPU only on Linux and Windows and use Metal on MacOS.

`llama.cpp` supports a number of hardware acceleration backends depending including OpenBLAS, cuBLAS, CLBlast, HIPBLAS, and Metal.
See the [llama.cpp README](https://github.com/ggerganov/llama.cpp#build) for a full list of supported backends.

All of these backends are supported by `llama-cpp-python` and can be enabled by setting the `CMAKE_ARGS` environment variable before installing.

On Linux and Mac you set the `CMAKE_ARGS` like this:

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

On Windows you can set the `CMAKE_ARGS` like this:

```ps
$env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python
```

#### OpenBLAS

To install with OpenBLAS, set the `LLAMA_BLAS and LLAMA_BLAS_VENDOR` environment variables before installing:

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

#### cuBLAS

To install with cuBLAS, set the `LLAMA_CUBLAS=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

#### Metal

To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

#### CLBlast

To install with CLBlast, set the `LLAMA_CLBLAST=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python
```

#### hipBLAS

To install with hipBLAS / ROCm support for AMD cards, set the `LLAMA_HIPBLAS=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

### Windows Notes

If you run into issues where it complains it can't find `'nmake'` `'?'` or CMAKE_C_COMPILER, you can extract w64devkit as [mentioned in llama.cpp repo](https://github.com/ggerganov/llama.cpp#openblas) and add those manually to CMAKE_ARGS before running `pip` install:
```ps
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe" 
```

See the above instructions and set `CMAKE_ARGS` to the BLAS backend you want to use.

### MacOS Notes

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```
Otherwise, while installing it will build the llama.cpp x86 version which will be 10x slower on Apple Silicon (M1) Mac.

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

### Upgrading and Reinstalling

To upgrade or rebuild `llama-cpp-python` add the following flags to ensure that the package is rebuilt correctly:

```bash
pip install llama-cpp-python  --upgrade --force-reinstall --no-cache-dir
```

This will ensure that all source files are re-built with the most recently set `CMAKE_ARGS` flags.

## High-level API

[API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api)

The high-level API provides a simple managed interface through the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

Below is a short example demonstrating how to use the high-level API to for basic text completion:

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/llama-model.gguf")
>>> output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
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

### Chat Completion

The high-level API also provides a simple interface for chat completion.

Note that `chat_format` option must be set for the particular model you are using.

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="path/to/llama-2/llama-model.gguf", chat_format="llama-2")
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

### Function Calling

The high-level API also provides a simple interface for function calling.

Note that the only model that supports full function calling at this time is "functionary".
The gguf-converted files for this model can be found here: [functionary-7b-v1](https://huggingface.co/abetlen/functionary-7b-v1-GGUF)


```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="path/to/functionary/llama-model.gguf", chat_format="functionary")
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
pip install llama-cpp-python[server]
python3 -m llama_cpp.server --model models/7B/llama-model.gguf
```

Similar to Hardware Acceleration section above, you can also install with GPU (cuBLAS) support like this:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]
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
>>> llama_cpp.llama_backend_init(numa=False) # Must be called once at the start of each program
>>> params = llama_cpp.llama_context_default_params()
# use bytes for char * params
>>> model = llama_cpp.llama_load_model_from_file(b"./models/7b/llama-model.gguf", params)
>>> ctx = llama_cpp.llama_new_context_with_model(model, params)
>>> max_tokens = params.n_ctx
# use ctypes arrays for array params
>>> tokens = (llama_cpp.llama_token * int(max_tokens))()
>>> n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, add_bos=llama_cpp.c_bool(True))
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
