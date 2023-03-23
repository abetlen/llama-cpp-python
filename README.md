# `llama.cpp` Python Bindings

Simple Python bindings for @ggerganov's [`llama.cpp`](https://github.com/ggerganov/llama.cpp) library.

These bindings expose the low-level `llama.cpp` C API through a complete `ctypes` interface.
This module also exposes a high-level Python API that is more convenient to use and follows a familiar format.

# Install

```bash
pip install llama-cpp-python
```

# Usage

```python
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="models/7B/...")
>>> output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
>>> print(output)
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "models/7B/...",
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
