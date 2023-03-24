# 🦙 Python Bindings for `llama.cpp`

Simple Python bindings for **@ggerganov's** [`llama.cpp`](https://github.com/ggerganov/llama.cpp) library.
This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for text completion
  - OpenAI-like API
  - LangChain compatibility


## API Reference

::: llama_cpp.Llama
    options:
        members:
            - __init__
            - __call__
        show_root_heading: true

::: llama_cpp.llama_cpp
    options:
        show_if_no_docstring: true