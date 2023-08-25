# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.79]

### Added

- GGUF Support (breaking change requiring new model format)

## [0.1.78]

### Added

- Grammar based sampling via LlamaGrammar which can be passed to completions
- Make n_gpu_layers == -1 offload all layers

## [0.1.77]

- (llama.cpp) Update llama.cpp add support for LLaMa 2 70B
- (server) Add temporary n_gqa and rms_norm_eps parameters required for LLaMa 2 70B

## [0.1.76]

- (llama.cpp) Update llama.cpp add support for LLaMa 2 70B

## [0.1.75]

- Update llama.cpp

## [0.1.74]

### Added

- (server) OpenAI style error responses

## [0.1.73]

### Added

- (server) Add rope parameters to server settings

## [0.1.72]

### Added

- (llama.cpp) Update llama.cpp added custom_rope for extended context lengths

## [0.1.71]

### Added

- (llama.cpp) Update llama.cpp

### Fixed

- (server) Fix several pydantic v2 migration bugs

## [0.1.70]

### Fixed

- (Llama.create_completion) Revert change so that `max_tokens` is not truncated to `context_size` in `create_completion`
- (server) Fixed changed settings field names from pydantic v2 migration

## [0.1.69]

### Added

- (server) Streaming requests can are now interrupted pre-maturely when a concurrent request is made. Can be controlled with the `interrupt_requests` setting.
- (server) Moved to fastapi v0.100.0 and pydantic v2
- (docker) Added a new "simple" image that builds llama.cpp from source when started.

## Fixed

- (server) performance improvements by avoiding unnecessary memory allocations during sampling

## [0.1.68]

### Added

- (llama.cpp) Update llama.cpp

## [0.1.67]

### Fixed

- Fix performance bug in Llama model by pre-allocating memory tokens and logits.
- Fix bug in Llama model where the model was not free'd after use.

## [0.1.66]

### Added

- (llama.cpp) New model API

### Fixed

- Performance issue during eval caused by looped np.concatenate call
- State pickling issue when saving cache to disk

## [0.1.65]

### Added

- (llama.cpp) Fix struct misalignment bug

## [0.1.64]

### Added

- (llama.cpp) Update llama.cpp
- Fix docs for seed. Set -1 for random.

## [0.1.63]

### Added

- (llama.cpp) Add full gpu utilisation in CUDA
- (llama.cpp) Add get_vocab
- (llama.cpp) Add low_vram parameter
- (server) Add logit_bias parameter

## [0.1.62]

### Fixed

- Metal support working
- Cache re-enabled

## [0.1.61]

### Fixed

- Fix broken pip installation

## [0.1.60]

### NOTE

- This release was deleted due to a bug  with the packaging system that caused pip installations to fail.

### Fixed

- Truncate max_tokens in create_completion so requested tokens doesn't exceed context size.
- Temporarily disable cache for completion requests

## [v0.1.59]

### Added

- (llama.cpp) k-quants support
- (server) mirostat sampling parameters to server

### Fixed

- Support both `.so` and `.dylib` for `libllama` on MacOS

## [v0.1.58]

### Added

- (llama.cpp) Metal Silicon support

## [v0.1.57]

### Added

- (llama.cpp) OpenLlama 3B support

## [v0.1.56]

### Added

- (misc) Added first version of the changelog
- (server) Use async routes
- (python-api) Use numpy for internal buffers to reduce memory usage and improve performance.

### Fixed

- (python-api) Performance bug in stop sequence check slowing down streaming.