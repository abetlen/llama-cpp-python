from __future__ import annotations

from typing import Optional, Union, List

import llama_cpp

from llama_cpp.server.settings import ModelSettings


class LlamaProxy:
    def __init__(self, models: List[ModelSettings]) -> None:
        assert len(models) > 0, "No models provided!"

        self._model_settings_dict: dict[str, ModelSettings] = {}
        for model in models:
            if not model.model_alias:
                model.model_alias = model.model
            self._model_settings_dict[model.model_alias] = model

        self._current_model: Optional[llama_cpp.Llama] = None
        self._current_model_alias: Optional[str] = None

        self._default_model_settings: ModelSettings = models[0]
        self._default_model_alias: str = self._default_model_settings.model_alias  # type: ignore

        # Load default model
        self._current_model = self.load_llama_from_model_settings(
            self._default_model_settings
        )
        self._current_model_alias = self._default_model_alias

    def __call__(self, model: Optional[str] = None) -> llama_cpp.Llama:
        if model is None:
            model = self._default_model_alias

        if model not in self._model_settings_dict:
            model = self._default_model_alias

        if model == self._current_model_alias:
            if self._current_model is not None:
                return self._current_model

        self._current_model = None

        settings = self._model_settings_dict[model]
        self._current_model = self.load_llama_from_model_settings(settings)
        self._current_model_alias = model
        return self._current_model

    def __getitem__(self, model: str):
        return self._model_settings_dict[model].model_dump()

    def __setitem__(self, model: str, settings: Union[ModelSettings, str, bytes]):
        if isinstance(settings, (bytes, str)):
            settings = ModelSettings.model_validate_json(settings)
        self._model_settings_dict[model] = settings

    def __iter__(self):
        for model in self._model_settings_dict:
            yield model

    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def load_llama_from_model_settings(settings: ModelSettings) -> llama_cpp.Llama:
        chat_handler = None
        if settings.chat_format == "llava-1-5":
            assert settings.clip_model_path is not None, "clip model not found"
            chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )

        _model = llama_cpp.Llama(
            model_path=settings.model,
            # Model Params
            n_gpu_layers=settings.n_gpu_layers,
            main_gpu=settings.main_gpu,
            tensor_split=settings.tensor_split,
            vocab_only=settings.vocab_only,
            use_mmap=settings.use_mmap,
            use_mlock=settings.use_mlock,
            # Context Params
            seed=settings.seed,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_threads=settings.n_threads,
            n_threads_batch=settings.n_threads_batch,
            rope_scaling_type=settings.rope_scaling_type,
            rope_freq_base=settings.rope_freq_base,
            rope_freq_scale=settings.rope_freq_scale,
            yarn_ext_factor=settings.yarn_ext_factor,
            yarn_attn_factor=settings.yarn_attn_factor,
            yarn_beta_fast=settings.yarn_beta_fast,
            yarn_beta_slow=settings.yarn_beta_slow,
            yarn_orig_ctx=settings.yarn_orig_ctx,
            mul_mat_q=settings.mul_mat_q,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            offload_kqv=settings.offload_kqv,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            chat_handler=chat_handler,
            # Misc
            verbose=settings.verbose,
        )
        if settings.cache:
            if settings.cache_type == "disk":
                if settings.verbose:
                    print(f"Using disk cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                if settings.verbose:
                    print(f"Using ram cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
            _model.set_cache(cache)
        return _model

