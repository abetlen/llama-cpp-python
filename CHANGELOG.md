# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.90]

- feat: Update llama.cpp to ggerganov/llama.cpp@1d1ccce67613674c75c9c7e3fa4c1e24e428ba48
- feat: Add support for `MiniCPMv26ChatHandler` and `minicpm-v-26` in server by @abetlen in f70df824985d875226793b94dacc0c302a4256b2

## [0.2.89]

- feat: Update llama.cpp to ggerganov/llama.cpp@cfac111e2b3953cdb6b0126e67a2487687646971
- fix: Llama.close didn't free lora adapter by @jkawamoto in #1679
- fix: missing dependencies for test by @jkawamoto in #1680

## [0.2.88]

- feat: Update llama.cpp to ggerganov/llama.cpp@fc4ca27b25464a11b3b86c9dbb5b6ed6065965c2
- fix: only print 'cache saved' in verbose mode by @lsorber in #1668 
- fix: Added back from_file method to LlamaGrammar by @ExtReMLapin in #1673
- fix: grammar prints on each call by @abetlen in 0998ea0deea076a547d54bd598d6b413b588ee2b
- feat: Enable recursive search of HFFS.ls when using from_pretrained by @benHeidabetlen in #1656
- feat: Add more detailed log for prefix-match by @xu-song in #1659

## [0.2.87]

- feat: Update llama.cpp to ggerganov/llama.cpp@be55695eff44784a141a863f273661a6bce63dfc
- fix: Include all llama.cpp source files and subdirectories by @abetlen in 9cad5714ae6e7c250af8d0bbb179f631368c928b
- feat(ci): Re-build wheel index automatically when releases are created by @abetlen in 198f47dc1bd202fd2b71b29e041a9f33fe40bfad

## [0.2.86]

- feat: Update llama.cpp to ggerganov/llama.cpp@398ede5efeb07b9adf9fbda7ea63f630d476a792
- feat: Ported back new grammar changes from C++ to Python implementation by @ExtReMLapin in (#1637)
- fix: llama_grammar_accept_token arg order by @tc-wolf in (#1649)

## [0.2.85]

- feat: Update llama.cpp to ggerganov/llama.cpp@398ede5efeb07b9adf9fbda7ea63f630d476a792
- fix: Missing LoRA adapter after API change by @shamitv in #1630
- fix(docker): Update Dockerfile BLAS options by @olivierdebauche in #1632
- fix(docker): Fix GGML_CUDA param by @olivierdebauche in #1633
- fix(docker): Update Dockerfile build options from `LLAMA_` to `GGML_` by @olivierdebauche in #1634
- feat: FreeBSD compatibility by @yurivict in #1635

## [0.2.84]

- feat: Update llama.cpp to ggerganov/llama.cpp@4730faca618ff9cee0780580145e3cbe86f24876
- fix: fix: Correcting run.sh filepath in Simple Docker implementation by @mashuk999 in #1626

## [0.2.83]

- feat: Update llama.cpp to ggerganov/llama.cpp@081fe431aa8fb6307145c4feb3eed4f48cab19f8
- feat: Add 'required' literal to ChatCompletionToolChoiceOption by @mjschock in #1597
- fix: Change repeat_penalty to 1.0 to match llama.cpp defaults by @ddh0 in #1590
- fix(docs): Update README.md typo by @ericcurtin in #1589
- fix(server): Use split_mode from model settings by @grider-withourai in #1594
- feat(ci): Dockerfile update base images and post-install cleanup by @Smartappli in #1530

## [0.2.82]

- feat: Update llama.cpp to ggerganov/llama.cpp@7fdb6f73e35605c8dbc39e9f19cd9ed84dbc87f2

## [0.2.81]

- feat: Update llama.cpp to ggerganov/llama.cpp@968967376dc2c018d29f897c4883d335bbf384fb
- fix(ci): Fix CUDA wheels, use LLAMA_CUDA instead of removed LLAMA_CUBLAS by @abetlen in 4fb6fc12a02a68884c25dd9f6a421cacec7604c6
- fix(ci): Fix MacOS release, use macos-12 image instead of removed macos-11 by @abetlen in 3a551eb5263fdbd24b36d7770856374c04e92788

## [0.2.80]

- feat: Update llama.cpp to ggerganov/llama.cpp@023b8807e10bc3ade24a255f01c1ad2a01bb4228
- fix(server): Fix bug in FastAPI streaming response where dependency was released before request completes causing SEGFAULT by @abetlen in 296304b60bb83689659883c9cc24f4c074dd88ff
- fix(server): Update default config value for embeddings to False to fix error in text generation where logits were not allocated by llama.cpp by @abetlen in bf5e0bb4b151f4ca2f5a21af68eb832a96a79d75
- fix(ci): Fix the CUDA workflow by @oobabooga in #1551
- docs: Update readme examples to use newer Qwen2 model by @jncraton in #1544

## [0.2.79]

- feat: Update llama.cpp to ggerganov/llama.cpp@9c77ec1d74874ee22bdef8f110e8e8d41389abf2
- feat(ci): Update workflows and pre-built wheels by @Smartappli in #1416
- feat: Add .close() method to Llama class to explicitly free model from memory by @jkawamoto in #1513
- feat: Support SPM infill by @CISC in #1492

## [0.2.78]

- feat: Update llama.cpp to ggerganov/llama.cpp@fd5ea0f897ecb3659d6c269ef6f3d833e865ead7
- fix: Avoid duplicate special tokens in chat formats by @CISC in #1439
- fix: fix logprobs when BOS is not present by @ghorbani in #1471
- feat: adding rpc_servers parameter to Llama class by @chraac in #1477

## [0.2.77]

- feat: Update llama.cpp to ggerganov/llama.cpp@bde7cd3cd949c1a85d3a199498ac98e78039d46f
- fix: string value kv_overrides by @abetlen in df45a4b3fe46e72664bda87301b318210c6d4782
- fix: Fix typo in Llama3VisionAlphaChatHandler by @abetlen in 165b4dc6c188f8fda2fc616154e111f710484eba
- fix: Use numpy recarray for candidates data, fixes bug with temp < 0 by @abetlen in af3ed503e9ce60fe6b5365031abad4176a3536b3
fix: Disable Windows+CUDA workaround when compiling for HIPBLAS by Engininja2 in #1493

## [0.2.76]

- feat: Update llama.cpp to ggerganov/llama.cpp@0df0aa8e43c3378975269a51f9b876c8692e70da
- feat: Improve Llama.eval performance by avoiding list conversion by @thoughtp0lice in #1476
- example: LLM inference with Ray Serve by @rgerganov in #1465

## [0.2.75]

- feat: Update llama.cpp to ggerganov/llama.cpp@13ad16af1231ab2d245d35df3295bcfa23de1305
- fix: segfault for models without eos / bos tokens by @abetlen in d99a6ba607a4885fb00e63e967964aa41bdbbbcb
- feat: add MinTokensLogitProcessor and min_tokens argument to server by @twaka in #1333
- misc: Remove unnecessary metadata lookups by @CISC in #1448

## [0.2.74]

- feat: Update llama.cpp to ggerganov/llama.cpp@b228aba91ac2cd9eb90e9d423ba1d0d20e0117e2
- fix: Enable CUDA backend for llava by @abetlen in 7f59856fa6f3e23f07e12fc15aeb9359dc6c3bb4
- docs: Fix typo in README.md by @yupbank in #1444

## [0.2.73]

- feat: Update llama.cpp to ggerganov/llama.cpp@25c6e82e7a1ad25a42b0894e87d9b5c557409516
- fix: Clear kv cache at beginning of image chat formats to avoid bug when image is evaluated first by @abetlen in ac55d0a175115d1e719672ce1cb1bec776c738b1

## [0.2.72]

- fix(security): Remote Code Execution by Server-Side Template Injection in Model Metadata by @retr0reg in b454f40a9a1787b2b5659cd2cb00819d983185df
- fix(security): Update remaining jinja chat templates to use immutable sandbox by @CISC in #1441

## [0.2.71]

- feat: Update llama.cpp to ggerganov/llama.cpp@911b3900dded9a1cfe0f0e41b82c7a29baf3a217
- fix: Make leading bos_token optional for image chat formats, fix nanollava system message by @abetlen in 77122638b4153e31d9f277b3d905c2900b536632
- fix: free last image embed in llava chat handler by @abetlen in 3757328b703b2cd32dcbd5853271e3a8c8599fe7

## [0.2.70]

- feat: Update llama.cpp to ggerganov/llama.cpp@c0e6fbf8c380718102bd25fcb8d2e55f8f9480d1
- feat: fill-in-middle support by @CISC in #1386
- fix: adding missing args in create_completion for functionary chat handler by @skalade in #1430
- docs: update README.md @eltociear in #1432
- fix: chat_format log where auto-detected format prints None by @balvisio in #1434
- feat(server): Add support for setting root_path by @abetlen in 0318702cdc860999ee70f277425edbbfe0e60419
- feat(ci): Add docker checks and check deps more frequently by @Smartappli in #1426
- fix: detokenization case where first token does not start with a leading space by @noamgat in #1375
- feat: Implement streaming for Functionary v2 + Bug fixes by @jeffrey-fong in #1419
- fix: Use memmove to copy str_value kv_override by @abetlen in 9f7a85571ae80d3b6ddbd3e1bae407b9f1e3448a
- feat(server): Remove temperature bounds checks for server by @abetlen in 0a454bebe67d12a446981eb16028c168ca5faa81
- fix(server): Propagate flash_attn to model load by @dthuerck in #1424

## [0.2.69]

- feat: Update llama.cpp to ggerganov/llama.cpp@6ecf3189e00a1e8e737a78b6d10e1d7006e050a2
- feat: Add llama-3-vision-alpha chat format by @abetlen in 31b1d95a6c19f5b615a3286069f181a415f872e8
- fix: Change default verbose value of verbose in image chat format handlers to True to match Llama by @abetlen in 4f01c452b6c738dc56eacac3758119b12c57ea94
- fix: Suppress all logs when verbose=False, use hardcoded fileno's to work in colab notebooks by @abetlen in f116175a5a7c84569c88cad231855c1e6e59ff6e
- fix: UTF-8 handling with grammars by @jsoma in #1415

## [0.2.68]

- feat: Update llama.cpp to ggerganov/llama.cpp@77e15bec6217a39be59b9cc83d6b9afb6b0d8167
- feat: Add option to enable flash_attn to Lllama params and ModelSettings by @abetlen in 22d77eefd2edaf0148f53374d0cac74d0e25d06e
- fix(ci): Fix build-and-release.yaml by @Smartappli in #1413

## [0.2.67]

- fix: Ensure image renders before text in chat formats regardless of message content order by @abetlen in 3489ef09d3775f4a87fb7114f619e8ba9cb6b656
- fix(ci): Fix bug in use of upload-artifact failing to merge multiple artifacts into a single release by @abetlen in d03f15bb73a1d520970357b702a9e7d4cc2a7a62

## [0.2.66]

- feat: Update llama.cpp to ggerganov/llama.cpp@8843a98c2ba97a25e93319a104f9ddfaf83ce4c4
- feat: Generic Chat Formats, Tool Calling, and Huggingface Pull Support for Multimodal Models (Obsidian, LLaVA1.6, Moondream) by @abetlen in #1147
- ci(fix): Workflow actions updates and fix arm64 wheels not included in release by @Smartappli in #1392
- ci: Add support for pre-built cuda 12.4.1 wheels by @Smartappli in #1388
- feat: Add support for str type kv_overrides by @abetlen in a411612b385cef100d76145da1fbd02a7b7cc894
- fix: Functionary bug fixes by @jeffrey-fong in #1385
- examples: fix quantize example by @iyubondyrev in #1387
- ci: Update dependabot.yml by @Smartappli in #1391

## [0.2.65]

- feat: Update llama.cpp to ggerganov/llama.cpp@46e12c4692a37bdd31a0432fc5153d7d22bc7f72
- feat: Allow for possibly non-pooled embeddings by @iamlemec in #1380

## [0.2.64]

- feat: Update llama.cpp to ggerganov/llama.cpp@4e96a812b3ce7322a29a3008db2ed73d9087b176
- feat: Add `llama-3` chat format by @andreabak in #1371
- feat: Use new llama_token_is_eog in create_completions by @abetlen in d40a250ef3cfaa8224d12c83776a2f1de96ae3d1
- feat(server): Provide ability to dynamically allocate all threads if desired using -1 by @sean-bailey in #1364
- ci: Build arm64 wheels by @gaby in 611781f5319719a3d05fefccbbf0cc321742a026
- fix: Update scikit-build-core build dependency avoid bug in 0.9.1 by @evelkey in #1370

## [0.2.63]

- feat: Update llama.cpp to ggerganov/llama.cpp@0e4802b2ecbaab04b4f829fde4a3096ca19c84b5
- feat: Add stopping_criteria to ChatFormatter, allow stopping on arbitrary token ids, fixes llama3 instruct by @abetlen in cc81afebf04d26ca1ac3cf72f23f18da6ab58588

## [0.2.62]

- feat: Update llama.cpp to ggerganov/llama.cpp@3b8f1ec4b18770531d0b1d792f3edf08254e4f0c
- feat: update grammar schema converter to match llama.cpp by @themrzmaster in #1353
- feat: add disable_ping_events flag by @khimaros in #1257
- feat: Make saved state more compact on-disk by @tc-wolf in #1296
- feat: Use all available CPUs for batch processing by @ddh0 in #1345

## [0.2.61]

- feat: Update llama.cpp to ggerganov/llama.cpp@ba5e134e073ec6837078c874aba44a702944a676
- fix: pass correct type to chat handlers for chat completion logprobs by @abetlen in bb65b4d76411112c6fb0bf759efd746f99ef3c6b
- feat: Add support for yaml based server configs by @abetlen in 060bfa64d529ade2af9b1f4e207a3937bbc4138f
- feat: Add typechecking for ctypes structure attributes by @abetlen in 1347e1d050fc5a9a32ffe0bb3e22858da28003bd

## [0.2.60]

- feat: Update llama.cpp to ggerganov/llama.cpp@75cd4c77292034ecec587ecb401366f57338f7c0
- fix: Always embed metal library by @abetlen in b3bfea6dbfb6ed9ce18f9a2723e0a9e4bd1da7ad
- fix: missing logprobs in response, incorrect response type for functionary by @abetlen in 1ae3abbcc3af7f4a25a3ffc40b246f18039565e8
- fix(docs): incorrect tool_choice example by @CISC in #1330

## [0.2.59]

- feat: Update llama.cpp to ggerganov/llama.cpp@ba0c7c70ab5b15f1f2be7fb0dfbe0366dda30d6c
- feat: Binary wheels for CPU, CUDA (12.1 - 12.3), Metal by @abetlen, @jllllll, and @oobabooga in #1247
- fix: segfault when logits_all=False by @abetlen in 8649d7671bd1a7c0d9cc6a5ad91c6ca286512ab3
- fix: last tokens passing to sample_repetition_penalties function by @ymikhailov in #1295

## [0.2.58]

- feat: Update llama.cpp to ggerganov/llama.cpp@ba0c7c70ab5b15f1f2be7fb0dfbe0366dda30d6c
- feat: add support for KV cache quantization options by @Limour-dev in #1307
- feat: Add logprobs support to chat completions by @windspirit95 in #1311
- fix: set LLAMA_METAL_EMBED_LIBRARY=on on MacOS arm64 by @bretello in #1289
- feat: Add tools/functions variables to Jinja2ChatFormatter, add function response formatting for all simple chat formats by @CISC in #1273
- fix: Changed local API doc references to hosted by by @lawfordp2017 in #1317

## [0.2.57]

- feat: Update llama.cpp to ggerganov/llama.cpp@ac9ee6a4ad740bc1ee484ede43e9f92b5af244c1
- fix: set default embedding pooling type to unspecified by @abetlen in 4084aabe867b8ec2aba1b22659e59c9318b0d1f3
- fix: Fix and optimize functionary chat handler by @jeffrey-fong in #1282
- fix: json mode for basic chat formats by @abetlen in 20e6815252d0efd9f015f7adbf108faaf36e3f3c

## [0.2.56]

- feat: Update llama.cpp to ggerganov/llama.cpp@c2101a2e909ac7c08976d414e64e96c90ee5fa9e
- feat(server): Add endpoints for tokenize, detokenize and count tokens by @felipelo in #1136
- feat: Switch embed to llama_get_embeddings_seq by @iamlemec in #1263
- fix: Fixed json strings grammar by blacklisting character control set by @ExtReMLapin in d02a9cf16ff88ad011e2eb1ce29f4d9400f13cd1
- fix: Check for existence of clip model path by @kejcao in #1264

## [0.2.55]

- feat: Update llama.cpp to ggerganov/llama.cpp@9731134296af3a6839cd682e51d9c2109a871de5
- docs: fix small typo in README: 'model know how' -> 'model knows how' by @boegel in #1244

## [0.2.54]

- feat: Update llama.cpp to ggerganov/llama.cpp@cb49e0f8c906e5da49e9f6d64a57742a9a241c6a
- docs: fix typo in README.md embeddings example by @iamlemec in #1232

## [0.2.53]

- feat: Update llama.cpp to ggerganov/llama.cpp@cb49e0f8c906e5da49e9f6d64a57742a9a241c6a
- fix: eos/bos_token set correctly for Jinja2ChatFormatter and automatic chat formatter by @CISC in #1230

## [0.2.52]

- feat: Update llama.cpp to ggerganov/llama.cpp@a33e6a0d2a66104ea9a906bdbf8a94d050189d91
- fix: Llava15ChatHandler (this function takes at least 4 arguments) by @abetlen in 8383a9e5620f5df5a88f62da16813eac200dd706

## [0.2.51]

- feat: Update llama.cpp to ggerganov/llama.cpp@c39373398803c669056304090050fe3f44b41bf9
- fix: Restore type hints for low-level api by @abetlen in 19234aa0dbd0c3c87656e65dd2b064665371925b

## [0.2.50]

- docs: Update Functionary OpenAI Server Readme by @jeffrey-fong in #1193
- fix: LlamaHFTokenizer now receives pre_tokens by @abetlen in 47bad30dd716443652275099fa3851811168ff4a

## [0.2.49]

- fix: module 'llama_cpp.llama_cpp' has no attribute 'c_uint8' in Llama.save_state by @abetlen in db776a885cd4c20811f22f8bd1a27ecc71dba927
- feat: Auto detect Mixtral's slightly different format by @lukestanley in #1214

## [0.2.48]

- feat: Update llama.cpp to ggerganov/llama.cpp@15499eb94227401bdc8875da6eb85c15d37068f7
- feat: Add Google's Gemma formatting via chat_format="gemma" by @alvarobartt in #1210
- feat: support minItems/maxItems in JSON grammar converter by @nopperl in 3921e10770996d95a9eb22c8248bacef39f69365
- fix: Update from_pretrained defaults to match hf_hub_download and pull to local cache folder by @abetlen in e6d6260a91b7831733f7d1f73c7af46a3e8185ed
- fix: Raise exceptions when llama model or context fails to load by @abetlen in dd22010e85265ae840c76ec835d67a29ed852722
- docs: Update README.md to fix pip install llama cpp server by @audip in #1187

## [0.2.47]

- feat: Update llama.cpp to ggerganov/llama.cpp@973053d8b0d04809836b3339a50f68d9c842de90

## [0.2.46]

- feat: Update llama.cpp to ggerganov/llama.cpp@ba2135ccae7462470b3865c6e41d2e1d734eac05
- feat: Pull models directly from huggingface by @abetlen in #1206
- feat(low-level-api): Improve API static type-safety and performance. Low level api functions are positional args only now. by @abetlen in #1205

## [0.2.45]

- feat: Update llama.cpp to ggerganov/llama.cpp@89febfed9322c8849520dc63c93ee4f5fd72556e

## [0.2.44]

- feat: Update llama.cpp to ggerganov/llama.cpp@4524290e87b8e107cc2b56e1251751546f4b9051
- fix: create_embedding broken response for input type str by @abetlen in 0ce66bc080fe537590b05b24bf442480bf2dd045
- fix: Use '\n' seperator for EventSourceResponse by @khimaros in #1188
- fix: Incorporate embedding pooling layer fixes by @iamlemec in #1194

## [0.2.43]

- feat: Update llama.cpp to ggerganov/llama.cpp@8084d554406b767d36b3250b3b787462d5dd626f
- feat: Support batch embeddings by @iamlemec in #1186
- fix: submodule kompute is not included in sdist by @abetlen in 7dbbfdecadebe7750be650d9409959640ff9a460
- fix: fix: Update openbuddy prompt format by @abetlen in 07a783779a62a4aac0b11161c7e0eb983ff215f8

## [0.2.42]

- feat: Update llama.cpp to ggerganov/llama.cpp@ea9c8e11436ad50719987fa23a289c74b7b40d40
- fix: sample idx off-by-one error for logit_processors by @lapp0 in #1179
- fix: chat formatting bugs in `chatml-function-calling` by @abetlen in 4b0e3320bd8c2c209e29978d0b21e2e471cc9ee3 and 68fb71b6a26a1e57331868f959b47ab4b87851e1

## [0.2.41]

- feat: Update llama.cpp to ggerganov/llama.cpp@895407f31b358e3d9335e847d13f033491ec8a5b
- fix: Don't change order of json schema object properties in generated grammar unless prop_order is passed by @abetlen in d1822fed6b706f38bd1ff0de4dec5baaa3cf84fa

## [0.2.40]

- feat: Update llama.cpp to ggerganov/llama.cpp@3bdc4cd0f595a6096cca4a64aa75ffa8a3503465
- feat: Generic chatml Function Calling using chat_format="chatml-function-calling"` by @abetlen in #957
- fix: Circular dependancy preventing early Llama object free by @notwa in #1176
- docs: Set the correct command for compiling with syscl support by @akarshanbiswas in #1172
- feat: use gpu backend for clip if available by @iamlemec in #1175

## [0.2.39]

- feat: Update llama.cpp to ggerganov/llama.cpp@b08f22c882a1443e6b97081f3ce718a4d1a741f8
- fix: Fix destructor logging bugs by using llama_log_callback to avoid suppress_stdout_stderr by @abetlen in 59760c85eddc72dfcc1839f43760ef72c23d6874

## [0.2.38]

- feat: Update llama.cpp to ggerganov/llama.cpp@1cfb5372cf5707c8ec6dde7c874f4a44a6c4c915
- feat: Add speculative decoding by @abetlen in #1120
- fix: Pass raise_exception and add_generation_prompt to jinja2 chat template by @abetlen in 078cca0361bf5a94d2cf52ed04980d20e32d6f95

## [0.2.37]

- feat: Update llama.cpp to ggerganov/llama.cpp@fea4fd4ba7f6b754ac795387b275e1a014a77bde
- feat: Automatically set chat format from gguf by @abetlen in #1110

## [0.2.36]

- feat: Update llama.cpp to ggerganov/llama.cpp@2aed77eb06a329f0d82bb1c467f4244904d4073f
- feat: Add mistral instruct chat format as "mistral-instruct" by @Rafaelblsilva in #799

## [0.2.35]

- feat: Update llama.cpp to ggerganov/llama.cpp@d2f650cb5b04ee2726663e79b47da5efe196ce00

## [0.2.34]

- feat: Update llama.cpp to ggerganov/llama.cpp@6db2b41a76ee78d5efdd5c3cddd5d7ad3f646855
- feat: Add json schema mode by @abetlen in #1122

## [0.2.33]

- feat: Update llama.cpp to ggerganov/llama.cpp@faa3526a1eba458120987ed8269e5616385a76f4
- feat(server): include llama-cpp-python version in openapi spec by @abetlen in cde7514c3d28e6d52f272614e9957208c344dde5
- fix: use both eos and bos tokens as stop sequences for hf-tokenizer-config chat format. by @abetlen in 5b982d0f8c6f35242c8862ffdce00e17cea0b44f
- fix: GGUF metadata KV overrides, re #1011 by @phiharri in #1116
- fix: llama_log_set should be able to accept null pointer by @abetlen in c970d41a85381fd55235136f123422df0bf0c7e7

## [0.2.32]

- feat: Update llama.cpp to ggerganov/llama.cpp@504dc37be8446fb09b1ede70300250ad41be32a2
- fix: from_json_schema oneof/anyof bug by @jndiogo in d3f5528ca8bcb9d69d4f27e21631e911f1fb9bfe
- fix: pass chat handler not chat formatter for huggingface autotokenizer and tokenizer_config formats by @abetlen in 24f39454e91cf5dddbc4b6041aead4accc7c7a2d
- feat: Add add_generation_prompt option for jinja2chatformatter by @abetlen in 7f3209b1eb4ad3260ba063801fab80a8c25a2f4c
- feat: Add Jinja2ChatFormatter by @abetlen in be09318c26add8674ce494ae7cc480cce72a4146
- feat: Expose gguf model metadata in metadata property by @abetlen in 5a34c57e5479e50c99aba9b38218cc48e6560b81

## [0.2.31]

- feat: Update llama.cpp to ggerganov/llama.cpp@a5cacb22b2114fd9adf61c00cbb237384d86bced
- fix: Mirostat sampling now passes correct type to ctypes and tracks state during generation by @abetlen in 3babe3512cb95743108f2b595210c38ed6f1b904
- fix: Python3.8 support in server by @abetlen in 141293a75b564a8699e0acba1da24d9aa1cf0ab1

## [0.2.30]

- feat: Update llama.cpp to ggerganov/llama.cpp@57e2a7a52a819883f40dada8a2edc24ecf48186b
- feat(server): Add ability to load chat format from huggingface autotokenizer or tokenizer_config.json files by @abetlen in b8fc1c7d83ad4a9207c707ba1d954fe580286a01
- feat: Integration of Jinja2 Templating for chat formats by @teleprint-me in #875
- fix: Offload KQV by default by @abetlen in 48c3b77e6f558a9899de0e1155c7dc0c7958d8e8
- fix: Support Accept text/event-stream in chat and completion endpoints, resolves #1083 by @aniljava in #1088
- fix(cli): allow passing n_ctx=0 to openAI API server args to use model n_ctx_train field per #1015 by @K-Mistele in #1093

## [0.2.29]

- feat: Update llama.cpp to ggerganov/llama.cpp@4483396751c79dea540808b9cb9238245d06da2b
- feat: Add split_mode option by @abetlen in 84615adbc6855c8384807c42f0130f9a1763f99d
- feat: Implement GGUF metadata KV overrides by @phiharri in #1011
- fix: Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor by @yieldthought in #1012
- fix: Fix low_level_api_chat_cpp example to match current API by @aniljava in #1086
- fix: Fix Pydantic model parsing by @DeNeutoy in #1087

## [0.2.28]

- feat: Update llama.cpp to ggerganov/llama.cpp@6efb8eb30e7025b168f3fda3ff83b9b386428ad6
- feat: Add ability to pass in penalize_nl param by @shankinson in #1068
- fix: print_grammar to stderr by @turian in #1052

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

NOTE: This release was deleted due to a bug with the packaging system that caused pip installations to fail.

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
