# Getting Started

## 🦙 Python Bindings for `llama.cpp`

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

## Installation

Install from PyPI:

```bash
pip install llama-cpp-python
```

## High-level API

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
export MODEL=./models/7B/ggml-model.bin
python3 -m llama_cpp.server
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to see the OpenAPI documentation.

## Low-level API

The low-level API is a direct `ctypes` binding to the C API provided by `llama.cpp`.
The entire API can be found in [llama_cpp/llama_cpp.py](https://github.com/abetlen/llama-cpp-python/blob/master/llama_cpp/llama_cpp.py) and should mirror [llama.h](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).


## Development

This package is under active development and I welcome any contributions.

To get started, clone the repository and install the package in development mode:

```bash
git clone git@github.com:abetlen/llama-cpp-python.git
git submodule update --init --recursive
# Will need to be re-run any time vendor/llama.cpp is updated
python3 setup.py develop
```

## API Reference

::: llama_cpp.Llama
    options:
        members:
            - __init__
            - tokenize
            - detokenize
            - reset
            - eval
            - sample
            - generate
            - create_embedding
            - embed
            - create_completion
            - __call__
            - create_chat_completion
            - set_cache
            - save_state
            - load_state
            - token_bos
            - token_eos
        show_root_heading: true

::: llama_cpp.LlamaCache

::: llama_cpp.LlamaState

::: llama_cpp.llama_cpp
    options:
        show_if_no_docstring: true

## License

This project is licensed under the terms of the MIT license.