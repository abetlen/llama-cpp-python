# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.11]

- Fix bug in `llama_model_params` object has no attribute `logits_all` by @abetlen in d696251fbe40015e8616ea7a7d7ad5257fd1b896 

## [0.2.10]

- Fix bug 'llama_model_params' object has no attribute 'embedding' by @abetlen in 42bb721d64d744242f9f980f2b89d5a6e335b5e4

## [0.2.9]

- Fix critical bug in pip installation of v0.2.8 due to `.git` directory in ac853e01e1a217a578080a4e1b851d2d08450adf

## [0.2.8]

- Update llama.cpp to ggerganov/llama.cpp@40e07a60f9ce06e79f3ccd4c903eba300fb31b5e
- Add configurable chat formats by @abetlen in #711
- Fix rope scaling bug by @Josh-XT in #767
- Fix missing numa parameter in server by @abetlen in d9bce17794d0dd6f7962d10aad768fedecf3ab89

## [0.2.7]

- Update llama.cpp to ggerganov/llama.cpp@a98b1633d5a94d0aa84c7c16e1f8df5ac21fc850
- Install required runtime dlls to package directory on windows by @abetlen in 8d75016549e2ff62a511b1119d966ffc0df5c77b
- Add openai-processing-ms to server response header by @Tradunsky in #748
- Bump minimum version of scikit-build-core to 0.5.1 to fix msvc cmake issue by @abetlen in 1ed0f3ebe16993a0f961155aa4b2c85f1c68f668
- Update `llama_types.py` to better match the openai api, old names are aliased to new ones by @abetlen in dbca136feaaf7f8b1182c4c3c90c32918b1d0bb3

## [0.2.6]

- Update llama.cpp to 80291a1d02a07f7f66666fb576c5b1e75aa48b46

## [0.2.5]

- Fix docker images missing starlette-context dependency by @abetlen in 22917989003c5e67623d54ab45affa1e0e475410
- Fix loading dll in Windows Isolation Containers by @abetlen in 847466562573191efa655753d9252f308c4fbdb0
- Fix build issue on m1 macs by @abetlen in dbd3a6d1ed8416a8fd800127251e730153afa305
- Update docs to gguf and add hw acceleration docs for server by @jasonacox in #688

## [0.2.4]

- Add NUMA support. **NOTE** low level api users must call llama_backend_init at the start of their programs by abetlen in f4090a0bb2a2a25acfe28d31c82cc1aa273bedee
- Fix tensor_split server cli argument by @abetlen in c4c440ba2dc86d9de728a751311fdd1c8e3756fa
- Made all `Llama` init parameters into keyword-only parameters by @abetlen in c8f9b8a734b5b040379bbd93995ba177affab1fe
- Added server params for `low_vram`, `main_gpu`, `lora_base`, and `lora_path` by @abetlen in 2920c4bf7ee1412d6bba7846e0e1b7ef6d34043b
- Removed server params for `rms_norm_eps` and `n_gqa` by @abetlen in 2920c4bf7ee1412d6bba7846e0e1b7ef6d34043b
- Fix boolean cli options by @abetlen in c999325e8e4507f6c6249dd2fb8de7f8bf57f71e and 0449d29b9f940e437231a07b9d56550226558bac
- Silence Pydantic Settings warnings about `model_alias` setting by @earonesty in #705

## [0.2.3]

- Update llama.cpp to ggerganov/llama.cpp@71ca2fad7d6c0ef95ef9944fb3a1a843e481f314
- Add X-Request-ID request header for mirroring custom IDs by @devrimcavusoglu in #703
- Add pyproject extra for scikit-build-core to ensure compatible pathspec version by @abetlen in 6cfc54284b99ef1bff8193e2d5e483dbd89ada02
- Fix issue with Literal and Optional cli arguments not working by @abetlen in #702

## [0.2.2]

- Fix bug in pip install of v0.2.1 due to scikit-build-core removing all `.metal` files in the source distribution (see #701)

## [0.2.1]

- Fix bug in pip install of v0.2.0 due to .git folder being included in the source distribution (see #701)

## [0.2.0]

- Migrated to scikit-build-core build system by @abetlen in #499
- Use `numpy` views for `LogitsProcessor` and `StoppingCriteria` instead of python lists by @abetlen in #499
- Drop support for end-of-life Python3.7 by @abetlen in #499
- Convert low level `llama.cpp` constants to use basic python types instead of `ctypes` types by @abetlen in #499

## [0.1.85]

- Add `llama_cpp.__version__` attribute by @janvdp in #684
- Fix low level api examples by @jbochi in #680

## [0.1.84]

- Update llama.cpp

## [0.1.83]

- Update llama.cpp

## [0.1.82]

- Update llama.cpp

## [0.1.81]

- Update llama.cpp

## [0.1.80]

- Update llama.cpp

## [0.1.79]

- GGUF Support (breaking change requiring new model format)

## [0.1.78]

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

- (server) OpenAI style error responses

## [0.1.73]

- (server) Add rope parameters to server settings

## [0.1.72]

- (llama.cpp) Update llama.cpp added custom_rope for extended context lengths

## [0.1.71]

- (llama.cpp) Update llama.cpp

- (server) Fix several pydantic v2 migration bugs

## [0.1.70]

- (Llama.create_completion) Revert change so that `max_tokens` is not truncated to `context_size` in `create_completion`
- (server) Fixed changed settings field names from pydantic v2 migration

## [0.1.69]

- (server) Streaming requests can are now interrupted pre-maturely when a concurrent request is made. Can be controlled with the `interrupt_requests` setting.
- (server) Moved to fastapi v0.100.0 and pydantic v2
- (docker) Added a new "simple" image that builds llama.cpp from source when started.
- (server) performance improvements by avoiding unnecessary memory allocations during sampling

## [0.1.68]

- (llama.cpp) Update llama.cpp

## [0.1.67]

- Fix performance bug in Llama model by pre-allocating memory tokens and logits.
- Fix bug in Llama model where the model was not free'd after use.

## [0.1.66]

- (llama.cpp) New model API

- Performance issue during eval caused by looped np.concatenate call
- State pickling issue when saving cache to disk

## [0.1.65]

- (llama.cpp) Fix struct misalignment bug

## [0.1.64]

- (llama.cpp) Update llama.cpp
- Fix docs for seed. Set -1 for random.

## [0.1.63]

- (llama.cpp) Add full gpu utilisation in CUDA
- (llama.cpp) Add get_vocab
- (llama.cpp) Add low_vram parameter
- (server) Add logit_bias parameter

## [0.1.62]

- Metal support working
- Cache re-enabled

## [0.1.61]

- Fix broken pip installation

## [0.1.60]

NOTE: This release was deleted due to a bug  with the packaging system that caused pip installations to fail.

- Truncate max_tokens in create_completion so requested tokens doesn't exceed context size.
- Temporarily disable cache for completion requests

## [v0.1.59]

- (llama.cpp) k-quants support
- (server) mirostat sampling parameters to server
- Support both `.so` and `.dylib` for `libllama` on MacOS

## [v0.1.58]

- (llama.cpp) Metal Silicon support

## [v0.1.57]

- (llama.cpp) OpenLlama 3B support

## [v0.1.56]

- (misc) Added first version of the changelog
- (server) Use async routes
- (python-api) Use numpy for internal buffers to reduce memory usage and improve performance.
- (python-api) Performance bug in stop sequence check slowing down streaming.