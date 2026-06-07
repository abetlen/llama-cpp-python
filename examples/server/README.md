# Server Example

This example is an updated OpenAI-compatible web server that depends only on the low-level C bindings.
It supports batched inference, prompt caching, response parsing, `/v1/responses`, disk sequence caching, MTP, LoRA, and multimodal image/audio inputs.

## Setup

The server is a [`uv` inline script](https://docs.astral.sh/uv/guides/scripts/), so `uv` can create the script environment and install the Python dependencies automatically.

```bash
cd examples/server
uv run --script server.py -C config.json
```

Use the local checkout instead of the published `llama-cpp-python` package when testing changes in this repository.

```bash
cd examples/server
uv run --with-editable ../.. --script server.py -C config.json
```

Check that the server is running.

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

Use an OpenAI-compatible client by pointing it at the local server.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

## API Surface

| Endpoint | Purpose | Reference |
| --- | --- | --- |
| `POST /v1/completions` | Legacy text completions with streaming, stop sequences, logprobs, penalties, seeds, and grammar-backed JSON output. | [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions) |
| `POST /v1/chat/completions` | Chat completions with streaming, tools, forced tool choice, reasoning parsing, multimodal content parts, and structured response parsing. | [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat) |
| `POST /v1/responses` | Stateless Responses API compatibility for clients that use response items and response events. | [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) |
| `WS /v1/responses` | Stateful websocket Responses transport with per-connection `previous_response_id` replay. | [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) |
| `GET /v1/models` | Returns the configured model alias. | [OpenAI Models API](https://platform.openai.com/docs/api-reference/models) |
| `GET /healthz` | Returns a simple typed health response. | |
| `GET /metrics` | Exposes scheduler, cache, draft, and model metrics in Prometheus text format. | [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) |

## Config Overview

`config.json` has three top-level sections.

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
    "alias": "gpt-4o-mini",
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

The checked-in `config.json` includes a Qwen3.5 template with reasoning text, tool calls, forced tool choice, image markers, and video markers.

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

`model.mtmd` loads a llama.cpp multimodal projector and enables OpenAI-style image and audio content parts.

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
      "image_max_bytes": 20971520,
      "audio_max_bytes": 104857600,
      "image_timeout_seconds": 10.0
    }
  }
}
```

| Field | Purpose |
| --- | --- |
| `mmproj_path` | Local multimodal projector path. |
| `mmproj_from_pretrained` | Hugging Face projector source. |
| `embedding_cache.path` | Directory for cached image and audio embeddings. |
| `embedding_cache.max_bytes` | Maximum embedding cache size. |
| `image_max_bytes` | Maximum image payload size. |
| `audio_max_bytes` | Maximum audio payload size. |
| `image_timeout_seconds` | Timeout for remote image and audio URL fetches. |

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

Use Qwen3.5 MTP when the loaded model and llama.cpp build expose the required draft state.

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

## Minimal Text-Only Config

Use this as a starting point when multimodal support is not needed.

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "model": {
    "alias": "local-model",
    "path": "/path/to/model.gguf",
    "n_ctx": 8192,
    "n_seq_max": 8,
    "n_batch": 512,
    "n_ubatch": 512,
    "threads": 8,
    "threads_batch": 8,
    "kv_unified": true
  }
}
```

## Notes

This example is self-contained while the server design is still changing.
Keep checked-in `config.json` runnable and put machine-specific experiments in separate local config files.
Do not depend on `llama_cpp_ext.py` outside this example because it exposes experimental non-public llama.cpp APIs.
