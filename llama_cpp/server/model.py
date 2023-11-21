import os
from typing import Any, Optional
from threading import Lock
import llama_cpp
from llama_cpp.server.settings import Settings, get_settings

FILE_EXT = ".gguf"
MODEL_ENV_ARG = "MODEL"
DEFAULT_MODEL_DIR = "/models"

def models_root_dir(path = None):
    path = os.path.abspath(path or os.environ.get(MODEL_ENV_ARG, DEFAULT_MODEL_DIR))
    if os.path.isdir(path): return path
    return os.path.dirname(path)

class MultiLlama:
    _model: Optional[llama_cpp.Llama] = None
    _models = {}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model_root = models_root_dir(settings.model)
        for filename in os.listdir(model_root):
            if filename.endswith(FILE_EXT):
                self._models[filename.split(FILE_EXT)[0]] = os.path.join(model_root, filename)

    def __call__(self, model: str, **kwargs: Any) -> llama_cpp.Llama:
        try:
            model_path = self._models[model]
        except KeyError:
            # TODO server raises 500 ?
            raise Exception(404, f"Model file for {model} NOT found")
        
        if self._model:
            if self._model.model_path == model_path:
                return self._model
            del self._model

        settings = self._settings
        self._model = llama_cpp.Llama(
            model_path=model_path,
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
            f16_kv=settings.f16_kv,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            clip_model_path=settings.clip_model_path,
            # Cache
            cache=settings.cache,
            cache_type=settings.cache_type,
            cache_size=settings.cache_size,
            # Misc
            verbose=settings.verbose,
            **kwargs
        )
        return self._model

    def __getitem__(self, model):
        return self._models[model]
    
    def __setitem__(self, model, path):
        self._models[model] = path
    
    def __iter__(self):
        for model in self._models:
            yield model
    
LLAMA: Optional[MultiLlama] = None

def _set_llama(settings: Optional[Settings] = None):
    global LLAMA
    LLAMA = MultiLlama(settings or next(get_settings()))

llama_outer_lock = Lock()
llama_inner_lock = Lock()

def get_llama():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            if not LLAMA:
                _set_llama()
            llama_outer_lock.release()
            release_outer_lock = False
            yield LLAMA
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()
