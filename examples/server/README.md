# Server Example

This example is an experimental OpenAI-compatible server built directly on llama.cpp through `llama-cpp-python`.
It is intended for development of batching, memory reuse, multimodal prompts, and newer API compatibility before those pieces are promoted into a smaller public server surface.

## Running

- Start the default Qwen3.5 vision-language configuration with `python server.py -C config.json`.
- The default `config.json` listens on `0.0.0.0:8000`.
- The default model alias is `gpt-4o-mini` so OpenAI-compatible clients can target a multimodal-capable model name.
- The default model is loaded from `lmstudio-community/Qwen3.5-0.8B-GGUF`.
- The default multimodal projector is loaded from the same Hugging Face repository.

## OpenAI-Compatible APIs

- `POST /v1/completions` supports text completions, streaming, multiple non-streaming prompts, stop sequences, logprobs, penalties, seeds, and grammar-backed response formats.
- `POST /v1/chat/completions` supports chat messages, streaming chunks, tools, forced tool choice, legacy function-call request fields, reasoning content parsing, and structured response parsing.
- `POST /v1/responses` supports the OpenAI Responses-style request shape, response items, streaming events, tools, and stateless HTTP requests.
- `WS /v1/responses` supports a stateful websocket transport with per-connection `previous_response_id` replay.
- `GET /v1/models` returns the configured model alias.
- `GET /healthz` returns a typed health check response.
- `GET /metrics` exposes Prometheus text metrics.

## Scheduling And Batching

- Requests are admitted by a memory policy before they enter decode scheduling.
- The scheduler batches prompt prefill, generation decode, and draft verification work into llama.cpp batches.
- Text-only requests keep the normal batched path even when a multimodal-capable model is loaded.
- Partitioned, unified, and checkpoint-style memory policies are supported for different llama.cpp memory layouts.
- Prompt prefixes are tracked with a radix trie and sequence history so resident llama.cpp sequence state can be reused.

## Multimodal Support

- `model.mtmd` enables llama.cpp MTMD projector loading.
- Image and audio inputs are accepted from OpenAI-style chat content parts.
- Text and media prompt segments are represented in one prompt plan so text-only ranges can still use prefix reuse.
- Media embeddings can be cached with `model.mtmd.embedding_cache`.
- Remote media loading can be restricted with `allowed_media_domains`.
- Local `file://` media loading can be restricted with `allowed_local_media_path`.
- Image and audio byte limits are configured with `image_max_bytes` and `audio_max_bytes`.

## Model Features

- Models can be loaded from a local `path` or from Hugging Face with `from_pretrained`.
- LoRA adapters can be loaded at startup from local paths or Hugging Face with `model.loras`.
- MTP draft decoding is supported for compatible Qwen3.5 text-only requests.
- Prompt lookup decoding is also available as a simpler draft provider.
- Disk sequence caching can store serialized llama.cpp sequence state for repeated prompt prefixes.

## Configuration Notes

- `n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`, `threads`, and `threads_batch` are the main throughput and memory tuning knobs.
- `kv_unified` should match the intended llama.cpp memory layout for the model and workload.
- `use_mmap` and `use_mlock` control model memory mapping and page locking.
- `response_schema` and `chat_template` define how model text is rendered and parsed into OpenAI-compatible fields.
- Keep additional local benchmark configs separate from `config.json` so the checked-in config remains a runnable default.

## Status

- This example is intentionally self-contained while the server design is still changing.
- Prefer small changes that preserve the current endpoints and default config behavior.
- Do not depend on `llama_cpp_ext.py` outside this example because it exposes experimental non-public llama.cpp APIs.
