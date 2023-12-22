from __future__ import annotations

import multiprocessing

from typing import Optional, List, Literal
from pydantic import Field
from pydantic_settings import BaseSettings

import llama_cpp

# Disable warning for model and model_alias settings
BaseSettings.model_config["protected_namespaces"] = ()


class ModelSettings(BaseSettings):
    """Model settings used to load a Llama model."""

    model: str = Field(
        description="The path to the model to use for generating completions."
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="The alias of the model to use for generating completions.",
    )
    # Model Params
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.",
    )
    main_gpu: int = Field(
        default=0,
        ge=0,
        description="Main GPU to use.",
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description="Split layers across multiple GPUs in proportion.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_mmap_supported(),
        description="Use mmap.",
    )
    use_mlock: bool = Field(
        default=llama_cpp.llama_mlock_supported(),
        description="Use mlock.",
    )
    # Context Params
    seed: int = Field(
        default=llama_cpp.LLAMA_DEFAULT_SEED, description="Random seed. -1 for random."
    )
    n_ctx: int = Field(default=2048, ge=1, description="The context size.")
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use.",
    )
    n_threads_batch: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=0,
        description="The number of threads to use when batch processing.",
    )
    rope_scaling_type: int = Field(default=llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED)
    rope_freq_base: float = Field(default=0.0, description="RoPE base frequency")
    rope_freq_scale: float = Field(
        default=0.0, description="RoPE frequency scaling factor"
    )
    yarn_ext_factor: float = Field(default=-1.0)
    yarn_attn_factor: float = Field(default=1.0)
    yarn_beta_fast: float = Field(default=32.0)
    yarn_beta_slow: float = Field(default=1.0)
    yarn_orig_ctx: int = Field(default=0)
    mul_mat_q: bool = Field(
        default=True, description="if true, use experimental mul_mat_q kernels"
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    offload_kqv: bool = Field(
        default=False, description="Whether to offload kqv to the GPU."
    )
    # Sampling Params
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    # LoRA Params
    lora_base: Optional[str] = Field(
        default=None,
        description="Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.",
    )
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to a LoRA file to apply to the model.",
    )
    # Backend Params
    numa: bool = Field(
        default=False,
        description="Enable NUMA support.",
    )
    # Chat Format Params
    chat_format: str = Field(
        default="llama-2",
        description="Chat format to use.",
    )
    clip_model_path: Optional[str] = Field(
        default=None,
        description="Path to a CLIP model to use for multi-modal chat completion.",
    )
    # Cache Params
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="The type of cache to use. Only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    # Misc
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )


class ServerSettings(BaseSettings):
    """Server settings used to configure the FastAPI and Uvicorn server."""

    # Uvicorn Settings
    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8000, description="Listen port")
    ssl_keyfile: Optional[str] = Field(
        default=None, description="SSL key file for HTTPS"
    )
    ssl_certfile: Optional[str] = Field(
        default=None, description="SSL certificate file for HTTPS"
    )
    # FastAPI Settings
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If set all requests need to be authenticated.",
    )
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )


class Settings(ServerSettings, ModelSettings):
    pass


class ConfigFileSettings(ServerSettings):
    """Configuration file format settings."""

    models: List[ModelSettings] = Field(
        default=[], description="Model configs"
    )
