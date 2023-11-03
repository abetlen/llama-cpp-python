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
  - LangChain compatibility

Documentation is available at [https://llama-cpp-python.readthedocs.io/en/latest](https://llama-cpp-python.readthedocs.io/en/latest).

> [!WARNING]  
> Starting with version 0.1.79 the model format has changed from `ggmlv3` to `gguf`. Old model files can be converted using the `convert-llama-ggmlv3-to-gguf.py` script in [`llama.cpp`](https://github.com/ggerganov/llama.cpp)


## Installation from PyPI

Install from PyPI (requires a c compiler):

```bash
pip install llama-cpp-python
```

The above command will attempt to install the package and build `llama.cpp` from source.
This is the recommended installation method as it ensures that `llama.cpp` is built with the available optimizations for your system.

If you have previously installed `llama-cpp-python` through pip and want to upgrade your version or rebuild the package with different  compiler options, please add the following flags to ensure that the package is rebuilt correctly:

```bash
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```
Otherwise, while installing it will build the llama.ccp x86 version which will be 10x slower on Apple Silicon (M1) Mac.

### Installation with Hardware Acceleration

`llama.cpp` supports multiple BLAS backends for faster processing.

To install with OpenBLAS, set the `LLAMA_BLAS and LLAMA_BLAS_VENDOR` environment variables before installing:

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

To install with cuBLAS, set the `LLAMA_CUBLAS=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

To install with CLBlast, set the `LLAMA_CLBLAST=1` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python
```

To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

To install with hipBLAS / ROCm support for AMD cards, set the `LLAMA_HIPBLAS=on` environment variable before installing:

```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

#### Windows remarks

To set the variables `CMAKE_ARGS`in PowerShell, follow the next steps (Example using, OpenBLAS):

```ps
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on"
```

Then, call `pip` after setting the variables:
```
pip install llama-cpp-python
```

If you run into issues where it complains it can't find `'nmake'` `'?'` or CMAKE_C_COMPILER, you can extract w64devkit as [mentioned in llama.cpp repo](https://github.com/ggerganov/llama.cpp#openblas) and add those manually to CMAKE_ARGS before running `pip` install:
```ps
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe" 
```

See the above instructions and set `CMAKE_ARGS` to the BLAS backend you want to use.

#### MacOS remarks

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](docs/install/macos.md)

## High-level API

The high-level API provides a simple managed interface through the `Llama` class.

Below is a short example demonstrating how to use the high-level API to generate text:

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/llama-model.gguf")
>>> output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
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

### Adjusting the Context Window
The context window of the Llama models determines the maximum number of tokens that can be processed at once. By default, this is set to 512 tokens, but can be adjusted based on your requirements.

For instance, if you want to work with larger contexts, you can expand the context window by setting the n_ctx parameter when initializing the Llama object:

```python
llm = Llama(model_path="./models/7B/llama-model.gguf", n_ctx=2048)
```

### Loading llama-2 70b

Llama2 70b must set the `n_gqa` parameter (grouped-query attention factor) to 8 when loading:

```python
llm = Llama(model_path="./models/70B/llama-model.gguf", n_gqa=8)
```

## Web Server

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


## Docker image

A Docker image is available on [GHCR](https://ghcr.io/abetlen/llama-cpp-python). To run the server:

```bash
docker run --rm -it -p 8000:8000 -v /path/to/models:/models -e MODEL=/models/llama-model.gguf ghcr.io/abetlen/llama-cpp-python:latest
```
[Docker on termux (requires root)](https://gist.github.com/FreddieOliveira/efe850df7ff3951cb62d74bd770dce27) is currently the only known way to run this on phones, see [termux support issue](https://github.com/abetlen/llama-cpp-python/issues/389) 

## Low-level API

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


# Documentation

Documentation is available at [https://abetlen.github.io/llama-cpp-python](https://abetlen.github.io/llama-cpp-python).
If you find any issues with the documentation, please open an issue or submit a PR.

# Development

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

# How does this compare to other Python bindings of `llama.cpp`?

I originally wrote this package for my own use with two goals in mind:

- Provide a simple process to install `llama.cpp` and access the full C API in `llama.h` from Python
- Provide a high-level Python API that can be used as a drop-in replacement for the OpenAI API so existing apps can be easily ported to use `llama.cpp`

Any contributions and changes to this package will be made with these goals in mind.

# License

This project is licensed under the terms of the MIT license.
