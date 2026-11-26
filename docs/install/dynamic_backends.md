# Explicit dynamic backend loading (internal API)

Custom builds using `GGML_BACKEND_DL=ON` place GGML backend modules alongside the package libraries.
The native default discovery paths may not include that directory when Python runs elsewhere.
Existing native `GGML_BACKEND_PATH` can name an out-of-tree backend module; the internal `llama_cpp._ggml` module also exposes explicit registration functions.
These are low-level, private interfaces and may change with the bundled GGML version.
They are not exported as a stable top-level `llama_cpp` API.
Importing `_ggml` does not register backends or change the existing automatic selection behavior.

Load a known module before constructing models:

```python
import os
from contextlib import closing
from pathlib import Path

from llama_cpp import Llama, _ggml

# Example for a custom Unix GGML_BACKEND_DL CPU build.
# Choose the actual module filename produced by your build on other platforms.
plugin = Path(_ggml.libggml_base_path) / "libggml-cpu.so"
backend = _ggml.ggml_backend_load(os.fsencode(plugin))
if backend is None:
    raise RuntimeError(f"Could not load backend: {plugin}")

try:
    with closing(Llama(model_path="model.gguf", n_gpu_layers=0)) as model:
        result = model.create_completion("Hello", max_tokens=8)
finally:
    _ggml.ggml_backend_unload(backend)
```

The returned handle is opaque and native-owned; keep it until every model, context and other resource using that backend has closed.
Unload it exactly once, and never use the handle afterward.
Do not unload a backend that another thread or model is using.
The functions do not manage model lifetime or synchronize other users.

To request native discovery in a particular directory, use:

```python
_ggml.ggml_backend_load_all_from_path(os.fsencode(_ggml.libggml_base_path))
```

`ggml_backend_load_all()` and `ggml_backend_load_all_from_path(None)` request the existing native default search paths.
Both bulk functions return `None`; they do not return unload handles.
Paths use `bytes` (for example, `os.fsencode`), and the individual loader returns `None` on failure.
These operations require backend modules compatible with the loaded GGML library; they do not make a statically linked build dynamic.

The generic loading contract has been exercised with real dynamic modules and a tiny model on macOS arm64.
The macOS build may also contain Metal and BLAS modules; `n_gpu_layers=0` is used for the CPU model controls.
This does not establish Linux x86 CPU-variant selection, cross-platform backend compatibility or a performance improvement.

For opt-in dynamic integration tests, set `LLAMA_CPP_TEST_BACKEND` to the absolute CPU module path and `LLAMA_CPP_TEST_MODEL` to an existing small GGUF, then run `python -m pytest tests/test_ggml_backend.py`.
The tests isolate registry changes in subprocesses and do not download models.
