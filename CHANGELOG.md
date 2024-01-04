# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.27]

- feat: Update llama.cpp to ggerganov/llama.cpp@b3a7c20b5c035250257d2b62851c379b159c899a
- feat: Add `saiga` chat format by @femoiseev in #1050
- feat: Added `chatglm3` chat format by @xaviviro in #1059
- fix: Correct typo in README.md by @qeleb in (#1058) 

## [0.2.26]

- feat: Update llama.cpp to ggerganov/llama.cpp@f6793491b5af6da75edad34d6f503ef86d31b09f

## [0.2.25]

- feat(server): Multi model support by @D4ve-R in #931
- feat(server): Support none defaulting to infinity for completions by @swg in #111
- feat(server): Implement openai api compatible authentication by @docmeth2 in #1010
- fix: text_offset of multi-token characters by @twaka in #1037
- fix: ctypes bindings for kv override by @phiharri in #1011
- fix: ctypes definitions of llama_kv_cache_view_update and llama_kv_cache_view_free. by @e-c-d in #1028

## [0.2.24]

- feat: Update llama.cpp to ggerganov/llama.cpp@0e18b2e7d0b5c0a509ea40098def234b8d4a938a
- feat: Add offload_kqv option to llama and server by @abetlen in 095c65000642a3cf73055d7428232fb18b73c6f3
- feat: n_ctx=0 now uses the n_ctx_train of the model by @DanieleMorotti in #1015
- feat: logits_to_logprobs supports both 2-D and 3-D logits arrays by @kddubey in #1002
- fix: Remove f16_kv, add offload_kqv fields in low level and llama apis by @brandonrobertz in #1019
- perf: Don't convert logprobs arrays to lists by @kddubey in #1021
- docs: Fix README.md functionary demo typo by @evelynmitchell in #996
- examples: Update low_level_api_llama_cpp.py to match current API by @jsoma in #1023

## [0.2.23]

- Update llama.cpp to ggerganov/llama.cpp@948ff137ec37f1ec74c02905917fa0afc9b97514
- Add qwen chat format by @yhfgyyf in #1005
- Add support for running the server with SSL by @rgerganov in #994
- Replace logits_to_logprobs implementation with numpy equivalent to llama.cpp by @player1537 in #991
- Fix UnsupportedOperation: fileno in suppress_stdout_stderr by @zocainViken in #961
- Add Pygmalion chat format by @chiensen in #986
- README.md multimodal params fix by @zocainViken in #967
- Fix minor typo in README by @aniketmaurya in #958

## [0.2.22]

- Update llama.cpp to ggerganov/llama.cpp@8a7b2fa528f130631a5f43648481596ab320ed5a
- Fix conflict with transformers library by kddubey in #952

## [0.2.21]

- Update llama.cpp to ggerganov/llama.cpp@64e64aa2557d97490b2fe1262b313e2f4a1607e3
- Make building llava optional by setting `CMAKE_ARGS="-DLLAVA_BUILD=OFF"` and using `LLAVA_CPP_LIB` to specify alternative path to shared library by @abetlen in e3941d9c674dbd9891dc3ceda390daeb21f05fd1

## [0.2.20]

- Update llama.cpp to ggerganov/llama.cpp@b38a16dfcff88d547f78f52d1bea31b84a05aff7
- Add `zephyr` chat format by @fakerybakery in #938
- Add `baichuan` chat format by @caiyesd in #938
- Add `baichuan-2` chat format by @caiyesd in #936
- Improve documentation for server chat formats by @jooray in #934
- Fix typo in README by @antonvice in 940
- Fix typo in the Open Orca chat format by @gardner in #947

## [0.2.19]

- Update llama.cpp to ggerganov/llama.cpp@0b871f1a04ef60e114bbe43004fd9c21114e802d
- Fix #569: stop parameter in chat completion api should accept str by @abetlen in 128dc4731fa846ead7e684a137ca57d8931b8899
- Document server host and port parameters by @jamesbraza in #768
- Do not set grammar to None when initializing LlamaGrammar by @mthuurne in #834
- Add mistrallite, intel, and openchat formats by @fakerybakery in #927
- Add support for min_p parameter by @tk-master in #921
- Fix #929: tokenizer adding leading space when generating from empty prompt by @abetlen in a34d48014192771d2e308a76c22f33bc0318d983
- Fix low level api example by @zocainViken in #925
- Fix missing package in openblas docker image by @ZisisTsatsas in #920

## [0.2.18]

- Update llama.cpp to ggerganov/llama.cpp@6bb4908a17150b49373b5f977685b2e180a04f6f

## [0.2.17]

- Update llama.cpp to ggerganov/llama.cpp@df9d1293defe783f42bc83af732d3c670552c541
- Hotfix: Set `CUDA_ARCHITECTURES=OFF` for `llava_shared` target on Windows by @abetlen in 4388f3341413110217b98c4f097ac5c590bdf40b

## [0.2.16]

- Update llama.cpp to ggerganov/llama.cp@a75fa576abba9d37f463580c379e4bbf1e1ad03c
- Add `set_seed` to `Llama` class by @abetlen in fd41ed3a908761d286102a019a34c2938a15118d
- Fix server doc arguments by @kjunggithub in #892
- Fix response_format handler in llava chat handler by @abetlen in b62c44983921197ed10a7d29dc4ba920e9979380
- Fix default max_tokens, chat completion is now unlimited (to context length) and completion is 16 tokens to match OpenAI defaults by @abetlen in e7962d2c733cbbeec5a37392c81f64185a9a39e8
- Fix json_schema_to_gbnf helper so that it takes a json schema string as input instead by @abetlen in faeae181b1e868643c0dc28fcf039f077baf0829
- Add support for $ref and $def in json_schema_to_gbnf to handle more complex function schemas by @abetlen in 770df344369c0630df1be14be9f9e301e7c56d24
- Update functionary chat handler for new OpenAI api by abetlen in 1b376c62b775b401653facf25a519d116aafe99a
- Fix add default stop sequence to chatml chat format by @abetlen in b84d76a844149216d511cfd8cdb9827148a1853c
- Fix sampling bug when logits_all=False by @abetlen in 6f0b0b1b840af846938ed74d0e8170a91c40e617

## [0.2.15]

- Update llama.cpp to ggerganov/llama.cpp@0a7c980b6f94a049cb804573df2d8092a34df8e4
- Add support for Llava1.5 multimodal models by @damian0815 and @abetlen in #821
- Update OpenAI API compatibility to match dev day update by @abetlen in #821
- Add seed parameter to completion and chat_completion functions of Llama class by @abetlen in 86aeb9f3a14808575d2bb0076e6acb4a30907e6a
- Add JSON mode support to constrain chat completion to JSON objects by @abetlen in b30b9c338bf9af316d497ea501d39f5c246900db

## [0.2.14]

- Update llama.cpp to ggerganov/llama.cpp@f0b30ef7dc1360922ccbea0a8cd3918ecf15eaa7
- Add support for Huggingface Autotokenizer Chat Formats by @bioshazard and @abetlen in #790 and bbffdaebaa7bb04b543dbf683a07276087251f86
- Fix llama-2 chat format by @earonesty in #869
- Add support for functionary chat format by @abetlen in #784
- Migrate inference from deprecated `llama_eval`API to `llama_batch` and `llama_decode` by @abetlen in #795

## [0.2.13]

- Update llama.cpp to ggerganov/llama.cpp@51b2fc11f7f605fff49725a4540e9a6ef7b51b70
- Fix name 'open' is not defined exception when deleting model by @abetlen in 011b95d7f34cbfc528af75a892757bd9a20838ab
- Fix tokenization of special characters by @antoine-lizee in #850

## [0.2.12]

- Update llama.cpp to ggerganov/llama.cpp@50337961a678fce4081554b24e56e86b67660163
- Fix missing `n_seq_id` in `llama_batch` by @NickAlgra in #842
- Fix for shared libraries on Windows that start with `lib` prefix by @sujeendran in #848
- Fix exception raised in `__del__` when freeing models by @cebtenzzre in #846
- Performance improvement for logit bias by @zolastro in #851
- Fix suffix check arbitrary code execution bug by @mtasic85 in #854
- Fix typo in `function_call` parameter in `llama_types.py` by @akatora28 in #849
- Fix streaming not returning `finish_reason` by @gmcgoldr in #798
- Fix `n_gpu_layers` check to allow values less than 1 for server by @hxy9243 in #826
- Supppress stdout and stderr when freeing model by @paschembri in #803
- Fix `llama2` chat format by @delock in #808
- Add validation for tensor_split size by @eric1932 #820
- Print stack trace on server error by @abetlen in d6a130a052db3a50975a719088a9226abfebb266
- Update docs for gguf by @johnccshen in #783
- Add `chatml` chat format by @abetlen in 305482bd4156c70802fc054044119054806f4126

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