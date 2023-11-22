import os
from typing import Any, Optional
from threading import Lock
import logging
import llama_cpp
from llama_cpp.server.settings import Settings, ModelSettings, get_settings

FILE_EXT = ".gguf"
MODEL_ENV_ARG = "MODEL"
DEFAULT_MODEL_DIR = "/models"

logger = logging.getLogger("uvicorn")

def models_root_dir(path = None):
    path = os.path.abspath(path or os.environ.get(MODEL_ENV_ARG, DEFAULT_MODEL_DIR))
    if os.path.isdir(path): return path
    return os.path.dirname(path)

class MultiLlama:
    _model: Optional[llama_cpp.Llama] = None
    _models: dict[str,str] = {}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model_root = models_root_dir(settings.model)
        for filename in os.listdir(model_root):
            if filename.endswith(FILE_EXT):
                self._models[filename.split(FILE_EXT)[0]] = os.path.join(model_root, filename)
        if os.path.isfile(settings.model):
            self(settings.model.split(os.path.sep)[-1].split(FILE_EXT)[0])

    def __call__(self, model: Optional[str] = None) -> llama_cpp.Llama:
        # handle backward compatibility, model param optional
        try:
            model_path = self._models[model]
        except KeyError:
            if self._model:
                if self._settings.verbose: logger.warn(f"Model file for {model} NOT found! Using preloaded")
                return self._model
            else: raise Exception(404, f"Model file for {model} NOT found")
        
        if self._model:
            if self._model.model_path == model_path:
                return self._model
            del self._model

        settings_path = os.path.join(os.path.dirname(model_path), 
                                     model_path.split(os.path.sep)[-1].split(FILE_EXT)[0] + ".json")
        try:
            with open(settings_path, 'rb') as f:
                settings = ModelSettings.model_validate_json(f.read())
        except:
            if self._settings.verbose: logger.warn(f"Loading settings for {model} FAILED! Using default")
            settings = self._settings

        self._model = llama_cpp.Llama(
            model_path=model_path,
            **(settings.model_dump(exclude={"model",}))
        )
        return self._model

    def __getitem__(self, model: str) -> str:
        return self._models[model]
    
    def __setitem__(self, model: str, path: str):
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
