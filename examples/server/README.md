# Server Example

This example is an updated OpenAI-compatible web server that depends only on the low-level C bindings.
It supports batched inference, prompt caching, response parsing, `/v1/responses`, `/v1/embeddings`, disk sequence caching, MTP, LoRA, and multimodal image/audio inputs.

## Setup

The server is a [`uv` inline script](https://docs.astral.sh/uv/guides/scripts/), so `uv` can create the script environment and install the Python dependencies automatically.

```bash
cd examples/server
uv run --script server.py -C configs/qwen3.5-0.8b.json
```

Use `uv run --extra-index-url` to pull a pre-built `llama-cpp-python` binary wheel instead of building from source.

```bash
cd examples/server
uv run \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  --script server.py -C configs/qwen3.5-0.8b.json
```

Pick the wheel index that matches the backend you want.

| Backend | Wheel index |
| --- | --- |
| CPU | `https://abetlen.github.io/llama-cpp-python/whl/cpu` |
| CUDA 11.8 | `https://abetlen.github.io/llama-cpp-python/whl/cu118` |
| CUDA 12.1 | `https://abetlen.github.io/llama-cpp-python/whl/cu121` |
| CUDA 12.2 | `https://abetlen.github.io/llama-cpp-python/whl/cu122` |
| CUDA 12.3 | `https://abetlen.github.io/llama-cpp-python/whl/cu123` |
| CUDA 12.4 | `https://abetlen.github.io/llama-cpp-python/whl/cu124` |
| CUDA 12.5 | `https://abetlen.github.io/llama-cpp-python/whl/cu125` |
| CUDA 13.0 | `https://abetlen.github.io/llama-cpp-python/whl/cu130` |
| CUDA 13.2 | `https://abetlen.github.io/llama-cpp-python/whl/cu132` |
| Metal | `https://abetlen.github.io/llama-cpp-python/whl/metal` |
| ROCm | `https://abetlen.github.io/llama-cpp-python/whl/rocm72` |
| Vulkan | `https://abetlen.github.io/llama-cpp-python/whl/vulkan` |

See the repository installation section for the full [pre-built wheel requirements](../../README.md#supported-backends).

## Model Configs

The smallest checked-in example uses Qwen3.5 0.8B so the server can be started on a normal development machine.

| Config | Model | Notes |
| --- | --- | --- |
| [`configs/bge-small-en-v1.5.json`](configs/bge-small-en-v1.5.json) | [`CompendiumLabs/bge-small-en-v1.5-gguf`](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf) | Small embedding model config for `/v1/embeddings`. |
| [`configs/qwen3.5-0.8b.json`](configs/qwen3.5-0.8b.json) | [`lmstudio-community/Qwen3.5-0.8B-GGUF`](https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF) | Default small multimodal example. |
| [`configs/gemma-4-12b-it-qat.json`](configs/gemma-4-12b-it-qat.json) | [`unsloth/gemma-4-12B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | Larger Gemma 4 QAT multimodal config with projector. |
| [`configs/qwen3.6-27b.json`](configs/qwen3.6-27b.json) | [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | Larger Qwen3.6 multimodal config. |
| [`configs/qwen3.6-35b-a3b.json`](configs/qwen3.6-35b-a3b.json) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Larger Qwen3.6 MoE multimodal config. |
| [`configs/gpt-oss-120b.json`](configs/gpt-oss-120b.json) | [`ggml-org/gpt-oss-120b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) | Large text-only split-GGUF config. |

The larger model configs default to `n_gpu_layers: -1` and `flash_attn: true`.

## Client Examples

Point an OpenAI-compatible client at the local `/v1` base URL.

### Chat Completions

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

response = client.chat.completions.create(
    model="qwen3.5-0.8b-vl",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

### Responses API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

response = client.responses.create(
    model="qwen3.5-0.8b-vl",
    input="Write one sentence about why prefix caching helps batched inference.",
)
print(response.output_text)
```

### Embeddings

Start the server with an embedding config before calling `/v1/embeddings`.

```bash
cd examples/server
uv run --script server.py -C configs/bge-small-en-v1.5.json
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

response = client.embeddings.create(
    model="bge-small-en-v1.5",
    input=["The food was delicious.", "The meal was excellent."],
)
print(len(response.data[0].embedding))
```

## API Surface

| Endpoint | Purpose | Reference |
| --- | --- | --- |
| `POST /v1/completions` | Legacy text completions with streaming, stop sequences, logprobs, penalties, seeds, and grammar-backed JSON output. | [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions) |
| `POST /v1/embeddings` | OpenAI-compatible embeddings for embedding-mode GGUF models, including string inputs, token inputs, base64 output, and dimensions truncation. | [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings) |
| `POST /v1/chat/completions` | Chat completions with streaming, tools, forced tool choice, reasoning parsing, multimodal content parts, and structured response parsing. | [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat) |
| `POST /v1/responses` | Stateless Responses API compatibility for clients that use response items and response events. | [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) |
| `WS /v1/responses` | Stateful websocket Responses transport with per-connection `previous_response_id` replay. | [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) |
| `GET /v1/models` | Returns the configured model alias. | [OpenAI Models API](https://platform.openai.com/docs/api-reference/models) |
| `GET /healthz` | Returns a simple typed health response. | |
| `GET /metrics` | Exposes scheduler, cache, draft, and model metrics in Prometheus text format. | [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) |

## Config Overview

Config files have three top-level sections.

```json
{
  "server": {},
  "model": {},
  "disk_cache": {}
}
```

| Section | Required | Purpose |
| --- | --- | --- |
| `server` | No | Uvicorn host and port settings. |
| `model` | Yes | Model source, llama.cpp runtime settings, chat formatting, LoRA, MTMD, draft decoding, and output parsing. |
| `disk_cache` | No | Optional serialized sequence cache for repeated prompt prefixes. |

## `server`

Use `server.host` and `server.port` to choose the bind address.

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

| Field | Default | Notes |
| --- | --- | --- |
| `host` | `127.0.0.1` | Use `0.0.0.0` to expose the server on the network. |
| `port` | `8000` | Passed directly to `uvicorn.run()`. |

## `model` Source

Load a local GGUF with `path` or download a GGUF from Hugging Face with `from_pretrained`.

```json
{
  "model": {
    "alias": "qwen3.5-0.8b-vl",
    "from_pretrained": {
      "repo_id": "lmstudio-community/Qwen3.5-0.8B-GGUF",
      "filename": "Qwen3.5-0.8B-Q8_0.gguf"
    }
  }
}
```

| Field | Notes |
| --- | --- |
| `path` | Local GGUF path. |
| `from_pretrained.repo_id` | Hugging Face model repository. |
| `from_pretrained.filename` | File name or glob pattern for the GGUF. |
| `from_pretrained.additional_files` | Extra files to download from the same repository. |
| `from_pretrained.cache_dir` | Optional Hugging Face cache directory. |
| `alias` | Model id returned by `/v1/models` and used by OpenAI-compatible clients. |

See the [Hugging Face Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for cache behavior and repository file resolution.

## llama.cpp Runtime Settings

Most model runtime fields map to `llama_model_params` or `llama_context_params` in [`llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h).

```json
{
  "model": {
    "n_ctx": 32768,
    "n_seq_max": 64,
    "n_batch": 128,
    "n_ubatch": 128,
    "threads": 4,
    "threads_batch": 8,
    "kv_unified": true,
    "use_mmap": true,
    "use_mlock": true
  }
}
```

| Field | Purpose |
| --- | --- |
| `n_ctx` | Total context size. |
| `n_seq_max` | Maximum number of concurrent llama.cpp sequence ids. |
| `n_batch` | Logical batch capacity. |
| `n_ubatch` | Physical microbatch capacity. |
| `threads` | Decode thread count. |
| `threads_batch` | Prefill and batch thread count. |
| `kv_unified` | Selects unified or per-sequence memory layout. |
| `embedding` | Overrides embedding mode; omit to auto-detect pooled embedding GGUFs from model metadata. |
| `pooling_type` | Overrides pooled embedding behavior for embedding models, such as `1` for mean pooling. |
| `store_logits` | Keeps logits after decode when needed by sampling or diagnostics. |
| `use_mmap` | Memory maps model weights. |
| `use_mlock` | Attempts to lock model pages into RAM. |

GPU and backend-related fields are passed through to llama.cpp when set.

```json
{
  "model": {
    "n_gpu_layers": -1,
    "split_mode": 1,
    "main_gpu": 0,
    "tensor_split": [1.0],
    "flash_attn": true,
    "offload_kqv": true,
    "op_offload": true
  }
}
```

## Chat Template

`model.chat_template` is a Jinja chat template compatible with the style used by [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating).

```json
{
  "model": {
    "chat_template": "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}"
  }
}
```

Use an array of strings when the template is too large to read or edit as one JSON string.

```json
{
  "model": {
    "chat_template": [
      "{{ bos_token }}",
      "{{ messages[0].content }}",
      "{{ eos_token }}"
    ]
  }
}
```

The checked-in [`configs/qwen3.5-0.8b.json`](configs/qwen3.5-0.8b.json) includes a Qwen3.5 template with reasoning text, tool calls, forced tool choice, image markers, and video markers.

## Response Parsing

`model.response_schema` parses generated text into OpenAI-compatible fields with JSON Schema plus the Hugging Face `x-regex` extensions.

```json
{
  "model": {
    "response_schema": {
      "type": "object",
      "properties": {
        "role": {"const": "assistant"},
        "content": {
          "type": "string",
          "x-regex": "^(.*)$"
        }
      },
      "required": ["role"]
    }
  }
}
```

Use `x-regex-iterator` and `x-regex-key-value` to parse repeated tool-call blocks.

See [Hugging Face response parsing](https://huggingface.co/docs/transformers/chat_response_parsing) and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/reference) for the underlying schema concepts.

## Multimodal `model.mtmd`

`model.mtmd` loads a llama.cpp multimodal projector and enables OpenAI-style image, audio, and video content parts.

```json
{
  "model": {
    "mtmd": {
      "mmproj_from_pretrained": {
        "repo_id": "lmstudio-community/Qwen3.5-0.8B-GGUF",
        "filename": "mmproj-Qwen3.5-0.8B-BF16.gguf"
      },
      "embedding_cache": {
        "path": ".cache/mtmd-embeddings",
        "max_bytes": 1073741824
      },
      "batch_max_tokens": 1024,
      "image_max_bytes": 20971520,
      "audio_max_bytes": 104857600,
      "video_max_bytes": 536870912,
      "image_timeout_seconds": 10.0
    }
  }
}
```

| Field | Purpose |
| --- | --- |
| `mmproj_path` | Local multimodal projector path. |
| `mmproj_from_pretrained` | Hugging Face projector source. |
| `embedding_cache.path` | Directory for cached image, audio, and video embeddings. |
| `embedding_cache.max_bytes` | Maximum embedding cache size. |
| `batch_max_tokens` | Maximum number of media output tokens per MTMD projector-side encode batch. |
| `image_max_bytes` | Maximum image payload size. |
| `audio_max_bytes` | Maximum audio payload size. |
| `video_max_bytes` | Maximum video payload size. |
| `image_timeout_seconds` | Timeout for remote image, audio, and video URL fetches. |

Send image inputs with OpenAI chat content parts.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ]
}
```

Send audio inputs as a URL or as base64 `input_audio` content.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": "...base64...", "format": "wav"}}
      ]
    }
  ]
}
```

## Media Loading Policy

Remote `http:` and `https:` media URLs are unrestricted unless `allowed_media_domains` is set.

```json
{
  "model": {
    "mtmd": {
      "allowed_media_domains": ["example.com", "static.example.com"]
    }
  }
}
```

Local `file:` media URLs are disabled unless `allowed_local_media_path` is set.

```json
{
  "model": {
    "mtmd": {
      "allowed_local_media_path": "/srv/llama-cpp-python/media"
    }
  }
}
```

`allowed_media_domains` matches exact hostnames and does not allow wildcard patterns.

## LoRA `model.loras`

Load LoRA adapters once at startup from local files or Hugging Face.

```json
{
  "model": {
    "loras": [
      {
        "from_pretrained": {
          "repo_id": "example/qwen-lora-gguf",
          "filename": "adapter.gguf"
        },
        "scale": 1.0
      }
    ]
  }
}
```

The current implementation does not hot-swap LoRAs per request.

## Draft Decoding

Set `model.draft_model` to enable speculative draft providers.

```json
{
  "model": {
    "draft_model": "prompt-lookup-decoding",
    "draft_model_num_pred_tokens": 8,
    "draft_model_max_ngram_size": 4
  }
}
```

### Multi-Token Prediction (MTP)

Use MTP when the loaded model and llama.cpp build expose the required draft state.

```json
{
  "model": {
    "draft_model": "draft-mtp",
    "draft_model_num_pred_tokens": 2,
    "draft_model_threads": 4,
    "draft_model_threads_batch": 8
  }
}
```

By default `draft-mtp` creates the MTP context from the target model.
Set `draft_model_path` or `draft_model_from_pretrained` when the model uses a separate assistant GGUF.

```json
{
  "model": {
    "draft_model": "draft-mtp",
    "draft_model_num_pred_tokens": 2,
    "draft_model_from_pretrained": {
      "repo_id": "example/gemma-assistant-GGUF",
      "filename": "assistant.gguf"
    }
  }
}
```

MTP currently applies to text-only requests.

## Disk Sequence Cache

`disk_cache` stores serialized llama.cpp sequence state for repeated prompt prefixes.

```json
{
  "disk_cache": {
    "path": ".cache/sequences",
    "max_bytes": 1073741824,
    "min_tokens": 128
  }
}
```

| Field | Purpose |
| --- | --- |
| `path` | Directory for cached sequence files. |
| `max_bytes` | Maximum cache size before background cleanup removes entries. |
| `min_tokens` | Minimum prefix length that is worth saving. |

The cache is versioned by model and context compatibility data and should be treated as ephemeral.
