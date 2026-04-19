# llama-cpp-python – Project Guide

## 1. Project Overview

**llama-cpp-python** is a Python binding for [`llama.cpp`](https://github.com/ggerganov/llama.cpp), enabling efficient local inference of large language models (LLMs) in GGUF format directly from Python.

### Key Technologies

| Layer | Technology |
|---|---|
| Core inference engine | `llama.cpp` (C/C++, vendored as a git submodule in `vendor/llama.cpp`) |
| Python–C bridge | `ctypes` (no Cython / pybind11 required) |
| Build system | CMake + [scikit-build-core](https://scikit-build-core.readthedocs.io/) |
| Web server | FastAPI + Uvicorn (OpenAI-compatible REST API) |
| Testing | pytest |
| Linting / formatting | [Ruff](https://docs.astral.sh/ruff/) |
| Documentation | MkDocs + mkdocstrings |
| Python versions | 3.8 – 3.13 |

### High-level Architecture

```
llama-cpp-python
├── vendor/llama.cpp       ← upstream C++ inference engine (git submodule)
├── CMakeLists.txt         ← builds llama.cpp shared libraries
├── llama_cpp/             ← Python package
│   ├── llama_cpp.py       ← low-level ctypes bindings (mirrors llama.h)
│   ├── llama.py           ← high-level Llama class
│   ├── llama_chat_format.py ← chat template handling
│   ├── llama_grammar.py   ← grammar / constrained generation
│   ├── llama_cache.py     ← KV-cache helpers
│   ├── llama_speculative.py ← speculative decoding
│   ├── llama_tokenizer.py ← HuggingFace tokenizer bridge
│   ├── llava_cpp.py       ← LLaVA multimodal C bindings
│   └── server/            ← OpenAI-compatible HTTP server
└── tests/                 ← pytest test suite
```

---

## 2. Getting Started

### Prerequisites

- **Python 3.8+**
- A C/C++ compiler:
  - Linux: `gcc` or `clang`
  - Windows: Visual Studio or MinGW (`w64devkit`)
  - macOS: Xcode command-line tools
- **CMake ≥ 3.21** (installed automatically by scikit-build-core if needed)
- *(Optional)* CUDA toolkit, ROCm, or Xcode for GPU-accelerated builds

### Installation

```bash
# Basic CPU install (builds llama.cpp from source)
pip install llama-cpp-python

# Pre-built CPU wheel (faster, no compiler needed)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# GPU-accelerated builds (set CMAKE_ARGS before installing)
CMAKE_ARGS="-DGGML_CUDA=on"    pip install llama-cpp-python  # CUDA
CMAKE_ARGS="-DGGML_METAL=on"   pip install llama-cpp-python  # macOS Metal
CMAKE_ARGS="-DGGML_VULKAN=on"  pip install llama-cpp-python  # Vulkan

# Install with the optional HTTP server extras
pip install "llama-cpp-python[server]"

# Install all extras (server + dev + test)
pip install "llama-cpp-python[all]"
```

> **Reinstalling after a change** – add `--upgrade --force-reinstall --no-cache-dir` to rebuild from scratch.

### Basic Usage

```python
from llama_cpp import Llama

# Load a GGUF model
llm = Llama(model_path="./models/llama-model.gguf")

# Text completion
output = llm("Q: What is 2+2? A:", max_tokens=16, stop=["\n"])
print(output["choices"][0]["text"])

# Pull a model directly from Hugging Face Hub
llm = Llama.from_pretrained(
    repo_id="lmstudio-community/Qwen3.5-0.8B-GGUF",
    filename="*Q8_0.gguf",
)

# Chat completion (OpenAI-style)
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["choices"][0]["message"]["content"])
```

### Running the HTTP Server

```bash
# Start the OpenAI-compatible server
python3 -m llama_cpp.server --model path/to/model.gguf

# With explicit chat format
python3 -m llama_cpp.server --model path/to/model.gguf --chat_format chatml

# All options
python3 -m llama_cpp.server --help
```

The server exposes standard OpenAI endpoints (e.g. `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`) and an interactive Swagger UI at `http://localhost:8000/docs`.

### Running Tests

```bash
# Install test dependencies first
pip install "llama-cpp-python[test]"

# Run the full test suite
make test
# or directly
python3 -m pytest --full-trace -v
```

---

## 3. Project Structure

```
llama-cpp-python/
├── .continue/rules/       ← Continue AI project rules (this file)
├── .github/               ← CI workflows, issue / PR templates
├── CMakeLists.txt         ← Top-level CMake build for llama.cpp shared libs
├── Makefile               ← Developer convenience targets
├── pyproject.toml         ← Package metadata, build config, tool settings
├── mkdocs.yml             ← Documentation site config
├── docs/                  ← MkDocs markdown sources
│   ├── index.md
│   ├── server.md          ← Server usage & configuration guide
│   ├── api-reference.md
│   └── install/macos.md
├── docker/                ← Dockerfile examples
├── examples/              ← Usage examples
│   ├── high_level_api/
│   ├── low_level_api/
│   ├── gradio_chat/
│   ├── batch-processing/
│   ├── hf_pull/
│   ├── ray/               ← Distributed inference with Ray
│   └── notebooks/
├── llama_cpp/             ← Main Python package
│   ├── __init__.py        ← Public API surface + version string
│   ├── llama_cpp.py       ← Low-level ctypes C-API bindings
│   ├── llava_cpp.py       ← LLaVA / vision C-API bindings
│   ├── mtmd_cpp.py        ← Multimodal C-API bindings
│   ├── llama.py           ← High-level Llama class
│   ├── llama_cache.py     ← DiskCache / RAM KV-cache wrappers
│   ├── llama_chat_format.py ← Chat format registry & handlers
│   ├── llama_grammar.py   ← GBNF grammar & JSON schema support
│   ├── llama_speculative.py ← Speculative decoding helpers
│   ├── llama_tokenizer.py ← HF tokenizer integration
│   ├── llama_types.py     ← Pydantic / TypedDict response types
│   ├── _ctypes_extensions.py ← ctypes helpers
│   ├── _ggml.py           ← GGML tensor type constants
│   ├── _internals.py      ← Internal C-struct wrappers
│   ├── _logger.py         ← Logging configuration
│   ├── _utils.py          ← Shared utilities
│   └── server/            ← OpenAI-compatible HTTP server
│       ├── __main__.py    ← Entry point (`python -m llama_cpp.server`)
│       ├── app.py         ← FastAPI app factory
│       ├── cli.py         ← CLI argument parsing
│       ├── model.py       ← Per-request model management
│       ├── settings.py    ← Pydantic settings (ServerSettings, ModelSettings)
│       ├── types.py       ← OpenAI request/response Pydantic models
│       └── errors.py      ← HTTP error handlers
├── scripts/               ← Helper scripts (release, etc.)
├── tests/                 ← pytest test suite
│   ├── test_llama.py
│   ├── test_llama_chat_format.py
│   ├── test_llama_grammar.py
│   └── test_llama_speculative.py
└── vendor/llama.cpp/      ← llama.cpp C++ source (git submodule)
```

### Key Configuration Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, scikit-build-core options, Ruff & pytest config |
| `CMakeLists.txt` | Builds `libllama` and related shared libraries from `vendor/llama.cpp` |
| `Makefile` | Convenience targets: `build`, `test`, `lint`, `format`, `docker` |
| `.gitmodules` | Points `vendor/llama.cpp` to upstream `ggerganov/llama.cpp` |
| `mkdocs.yml` | Documentation site structure |

---

## 4. Development Workflow

### Setting Up a Dev Environment

```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
pip install ".[all]"          # installs in editable mode with all extras
# or use the Makefile shortcut:
make deps
make build
```

### Coding Standards

- **Linter / formatter**: [Ruff](https://docs.astral.sh/ruff/) (target Python 3.8).
- Line length is **88 characters**.
- Run checks before committing:

```bash
make lint     # check for errors
make format   # auto-fix and reformat
```

### Testing

```bash
make test
# Tests live in tests/; pytest config is in pyproject.toml [tool.pytest.ini_options]
```

Most tests require an actual GGUF model file at a path provided via environment variables; see `tests/test_llama.py` for the expected variables.

### Build Variants

```bash
make build              # standard CPU build (editable install)
make build.debug        # debug symbols, no optimisation
make build.cuda         # CUDA GPU build
make build.metal        # macOS Metal build
make build.openblas     # OpenBLAS CPU BLAS build
make build.vulkan       # Vulkan build
```

### Updating the vendored llama.cpp

```bash
make update.vendor      # pulls latest master of llama.cpp
git add vendor/llama.cpp
git commit -m "chore: bump llama.cpp to <new-sha>"
```

### Documentation

```bash
mkdocs serve            # live-reload local preview
make deploy.gh-docs     # build & push to GitHub Pages
```

### Release / Publishing

```bash
make build.sdist        # create source distribution
make deploy.pypi        # upload to PyPI with twine
```

---

## 5. Key Concepts

### GGUF Format
Models must be in the [GGUF](https://huggingface.co/docs/hub/gguf) file format — the successor to GGML. GGUF encodes weights, tokenizer, and metadata in a single file and supports various quantisation levels (e.g. Q4_0, Q8_0, F16).

### ctypes Bindings (`llama_cpp.py` / `llava_cpp.py`)
The low-level layer wraps `libllama` with pure Python `ctypes`. Every exported C function is declared with its argument types and return type. Consumers of this layer work directly with C structs and pointers.

### High-level `Llama` Class (`llama.py`)
Manages model loading, context creation, sampling, and streaming. Exposes OpenAI-compatible methods:
- `__call__` / `create_completion` – text completion
- `create_chat_completion` / `create_chat_completion_openai_v1` – chat
- `create_embedding` – embeddings
- `from_pretrained` – download + load from Hugging Face Hub

### Chat Formats (`llama_chat_format.py`)
A registry of named chat templates (`chatml`, `llama-2`, `gemma`, `mistral`, `functionary-v2`, etc.). Each format converts a list of messages into a single prompt string and handles stop tokens. Custom formats can be registered with `@register_chat_format`.

### Grammar / Constrained Generation (`llama_grammar.py`)
Supports GBNF (grammar-based constrained generation) to enforce structured output (e.g. valid JSON, JSON Schema). Pass `grammar` or `response_format` to `create_chat_completion`.

### KV Cache (`llama_cache.py`)
Optional caching of key-value pairs across calls. Supports in-memory (`LlamaRAMCache`) and disk-persistent (`LlamaDiskCache`) backends.

### Speculative Decoding (`llama_speculative.py`)
Accelerates generation using a smaller draft model to propose tokens that the target model verifies in parallel.

### OpenAI-Compatible Server (`llama_cpp/server/`)
A FastAPI application that implements the OpenAI REST API, allowing drop-in replacement for OpenAI clients. Supports multi-model configurations via a YAML/JSON config file.

---

## 6. Common Tasks

### Load a Model with GPU Acceleration

```python
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1,   # offload all layers to GPU
    n_ctx=4096,        # context window size
)
```

### Stream a Chat Completion

```python
for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
):
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

### Constrain Output to a JSON Schema

```python
llm.create_chat_completion(
    messages=[{"role": "user", "content": "Give me a user object"}],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
    },
)
```

### Use the Low-level C API

```python
from llama_cpp import llama_cpp   # ctypes bindings module

# Example: list available model metadata
ctx_params = llama_cpp.llama_context_default_params()
```

### Run Multi-model Server with Config File

Create a `config.yaml`:

```yaml
models:
  - model: /path/to/model1.gguf
    model_alias: "llama3"
    chat_format: chatml
  - model: /path/to/model2.gguf
    model_alias: "mistral"
    chat_format: mistral
```

```bash
python3 -m llama_cpp.server --config_file config.yaml
```

### Add a Custom Chat Format

```python
from llama_cpp import llama_chat_format

@llama_chat_format.register_chat_format("my-format")
def my_format(messages, **kwargs):
    # Build and return a ChatFormatterResponse
    ...
```

---

## 7. Troubleshooting

### Build fails: `Can't find 'nmake'` or `CMAKE_C_COMPILER` (Windows)

Add MinGW to the path and set the generator:

```powershell
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
pip install llama-cpp-python
```

### macOS: `incompatible architecture (have 'x86_64', need 'arm64')`

Force an arm64 build:

```bash
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_METAL=on" \
  pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

### Slow inference on Apple Silicon

Ensure you are using an `arm64` Python interpreter (e.g. via Miniforge/Mambaforge) and that Metal is enabled.

### Debugging verbose build output

```bash
pip install --verbose llama-cpp-python
# or use the Makefile target:
make build.debug
```

### `libllama.so` / `.dylib` not found at runtime

Run `make clean` then reinstall:

```bash
make clean
pip install --force-reinstall -e .
```

### Tests fail: model file not found

Most tests expect a GGUF model file. Check the environment variables in `tests/test_llama.py` and set the appropriate path before running.

---

## 8. References

| Resource | URL |
|---|---|
| Official documentation | https://llama-cpp-python.readthedocs.io/en/latest/ |
| API Reference | https://llama-cpp-python.readthedocs.io/en/latest/api-reference/ |
| Server guide | https://llama-cpp-python.readthedocs.io/en/latest/server/ |
| macOS install guide | https://llama-cpp-python.readthedocs.io/en/latest/install/macos/ |
| Changelog | https://llama-cpp-python.readthedocs.io/en/latest/changelog/ |
| PyPI package | https://pypi.org/project/llama-cpp-python/ |
| GitHub repository | https://github.com/abetlen/llama-cpp-python |
| upstream llama.cpp | https://github.com/ggerganov/llama.cpp |
| GGUF model format | https://huggingface.co/docs/hub/gguf |
| Hugging Face Hub | https://huggingface.co/models?library=gguf |
| LangChain integration | https://python.langchain.com/docs/integrations/llms/llamacpp |
| LlamaIndex integration | https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html |
