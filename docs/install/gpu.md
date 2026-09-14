# GPU Backends & Build Options

`llama-cpp-python` builds the bundled `llama.cpp` from source by default, which produces a
CPU-only build. To use a GPU (or a tuned CPU BLAS backend) you either pass the matching
`llama.cpp` cmake option through `CMAKE_ARGS`, or install a pre-built wheel for your
platform.

This page consolidates the build flags, the available pre-built wheels, and how to verify
that GPU offload is actually active. For the CLI/`pip` invocation syntax (and the
`-C cmake.args=...` alternative) see the [Getting Started](../index.md) page.

## CMAKE_ARGS mapping table

All `llama.cpp` cmake build options can be set via the `CMAKE_ARGS` environment variable
before installing, e.g.:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

| Backend | `CMAKE_ARGS` | Requirements |
| --- | --- | --- |
| CPU (default) | *(none)* | Any supported platform |
| OpenBLAS (CPU) | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` | OpenBLAS installed |
| CUDA | `-DGGML_CUDA=on` | NVIDIA GPU + CUDA toolkit |
| Metal | `-DGGML_METAL=on` | macOS 11.0+ on Apple Silicon / supported GPU |
| HIP (ROCm) | `-DGGML_HIP=on` | AMD GPU + ROCm/HIP toolkit |
| Vulkan | `-DGGML_VULKAN=on` | Vulkan-capable GPU + SDK |
| SYCL | `-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx` | Intel oneAPI (`source /opt/intel/oneapi/setvars.sh` first) |
| RPC | `-DGGML_RPC=on` | Distributed inference over RPC |

!!! note
    SYCL builds require the Intel oneAPI environment to be sourced in the same shell before
    installing:

    ```bash
    source /opt/intel/oneapi/setvars.sh
    CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install llama-cpp-python
    ```

## Pre-built wheels

For some backends you can skip the source build entirely by adding the matching
`--extra-index-url`. Pre-built wheels are currently published for Python 3.10, 3.11 and
3.12.

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<tag>
```

| Backend | `<tag>` | Notes |
| --- | --- | --- |
| CUDA 11.8 | `cu118` | Compute capability 6.0–8.9 |
| CUDA 12.1 | `cu121` | Compute capability 6.0+ |
| CUDA 12.2 | `cu122` | Compute capability 6.0+ |
| CUDA 12.3 | `cu123` | Compute capability 6.0+ |
| CUDA 12.4 | `cu124` | Compute capability 6.0+ |
| CUDA 12.5 | `cu125` | Compute capability 6.0+ |
| CUDA 13.0 | `cu130` | Compute capability 7.5+ |
| CUDA 13.2 | `cu132` | Compute capability 7.5+ |
| Metal | `metal` | macOS 11.0+ |
| ROCm (Linux) | `rocm72` | AMD GPU on Linux |
| HIP Radeon (Windows) | `hip-radeon` | AMD Radeon on Windows |
| Vulkan | `vulkan` | Linux or Windows |

For example, to install the CUDA 12.1 wheel:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

!!! warning "Match the wheel to your driver"
    A CUDA wheel must match a CUDA toolkit/driver that supports your GPU's compute
    capability. CUDA 11.8 wheels cover compute capability 6.0–8.9; CUDA 12 wheels cover
    6.0 and newer; CUDA 13 wheels require 7.5 and newer. If no wheel matches your setup,
    fall back to a source build with the relevant `CMAKE_ARGS` above.

## Verifying GPU offload is active

A build can silently fall back to CPU (for example, if the GPU backend wasn't compiled in,
or the wheel didn't match your toolkit). Two quick checks:

**1. Does this build support GPU offload at all?**

```python
import llama_cpp

print("GPU offload supported:", llama_cpp.llama_supports_gpu_offload())
print(llama_cpp.llama_print_system_info().decode())
```

If `llama_supports_gpu_offload()` returns `False`, the installed build is CPU-only —
re-install with the appropriate backend above.

**2. Are layers actually placed on the GPU?**

Load a model with `n_gpu_layers=-1` (offload all layers) and `verbose=True`. `llama.cpp`
logs how many layers were assigned to the GPU and the per-device memory usage:

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", n_gpu_layers=-1, verbose=True)
```

Look for log lines such as `offloaded N/N layers to GPU` and a non-CPU buffer in the model
load summary. If you see `0` layers offloaded despite requesting `-1`, the backend isn't
being used.

## Platform-specific notes

- **macOS (Metal):** see the [macOS (Metal)](macos.md) guide for detailed Apple Silicon
  build/runtime instructions.
- **Windows:** if cmake can't find `nmake` or `CMAKE_C_COMPILER`, install
  [w64devkit](https://github.com/ggml-org/llama.cpp) and point `CMAKE_ARGS` at its
  compilers:

  ```ps
  $env:CMAKE_GENERATOR = "MinGW Makefiles"
  $env:CMAKE_ARGS = "-DGGML_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
  ```