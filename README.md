# 🦙 Python Bindings for `llama.cpp`

[![Documentation](https://img.shields.io/badge/docs-passing-green.svg)](https://abetlen.github.io/llama-cpp-python)
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
  - LangChain compatibility

## Installation from PyPI (recommended)

Install from PyPI (requires a c compiler):

```bash
pip install llama-cpp-python
```

The above command will attempt to install the package and build build `llama.cpp` from source.
This is the recommended installation method as it ensures that `llama.cpp` is built with the available optimizations for your system.


### Installation with OpenBLAS / cuBLAS / CLBlast

`llama.cpp` supports multiple BLAS backends for faster processing.
Use the `FORCE_CMAKE=1` environment variable to force the use of `cmake` and install the pip package for the desired BLAS backend.

To install with OpenBLAS, set the `LLAMA_OPENBLAS=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with cuBLAS, set the `LLAMA_CUBLAS=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with CLBlast, set the `LLAMA_CLBLAST=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python
```


## High-level API

The high-level API provides a simple managed interface through the `Llama` class.

Below is a short example demonstrating how to use the high-level API to generate text:

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/ggml-model.bin")
>>> output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
>>> print(output)
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "./models/7B/ggml-model.bin",
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

## Web Server

`llama-cpp-python` offers a web server which aims to act as a drop-in replacement for the OpenAI API.
This allows you to use llama.cpp compatible models with any OpenAI compatible client (language libraries, services, etc).

To install the server package and get started:

```bash
pip install llama-cpp-python[server]
python3 -m llama_cpp.server --model models/7B/ggml-model.bin
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to see the OpenAPI documentation.

## Docker image

A Docker image is available on [GHCR](https://ghcr.io/abetlen/llama-cpp-python). To run the server:

```bash
docker run --rm -it -p8000:8000 -v /path/to/models:/models -eMODEL=/models/ggml-model-name.bin ghcr.io/abetlen/llama-cpp-python:latest
```

## Low-level API

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `llama.cpp`.
The entire lowe-level API can be found in [llama_cpp/llama_cpp.py](https://github.com/abetlen/llama-cpp-python/blob/master/llama_cpp/llama_cpp.py) and directly mirrors the C API in [llama.h](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).

Below is a short example demonstrating how to use the low-level API to tokenize a prompt:

```python
>>> import llama_cpp
>>> import ctypes
>>> params = llama_cpp.llama_context_default_params()
# use bytes for char * params
>>> ctx = llama_cpp.llama_init_from_file(b"./models/7b/ggml-model.bin", params)
>>> max_tokens = params.n_ctx
# use ctypes arrays for array params
>>> tokens = (llama_cppp.llama_token * int(max_tokens))()
>>> n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, add_bos=llama_cpp.c_bool(True))
>>> llama_cpp.llama_free(ctx)
```

Check out the [examples folder](examples/low_level_api) for more examples of using the low-level API.


# Documentation

Documentation is available at [https://abetlen.github.io/llama-cpp-python](https://abetlen.github.io/llama-cpp-python).
If you find any issues with the documentation, please open an issue or submit a PR.

# Development

This package is under active development and I welcome any contributions.

To get started, clone the repository and install the package in development mode:

```bash
git clone --recurse-submodules git@github.com:abetlen/llama-cpp-python.git
# Will need to be re-run any time vendor/llama.cpp is updated
python3 setup.py develop
```

# How does this compare to other Python bindings of `llama.cpp`?

I originally wrote this package for my own use with two goals in mind:

- Provide a simple process to install `llama.cpp` and access the full C API in `llama.h` from Python
- Provide a high-level Python API that can be used as a drop-in replacement for the OpenAI API so existing apps can be easily ported to use `llama.cpp`

Any contributions and changes to this package will be made with these goals in mind.

# License

This project is licensed under the terms of the MIT license.
