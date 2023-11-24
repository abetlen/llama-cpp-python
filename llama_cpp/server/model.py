import os
import logging
from typing import Optional, Union
from threading import Lock
import llama_cpp
from llama_cpp.server.settings import Settings, ModelSettings

FILE_EXT = ".gguf"
MODEL_ENV_ARG = "MODEL"
DEFAULT_MODEL_DIR = os.path.join(os.getcwd(), "/models")

logger = logging.getLogger("uvicorn")

def models_root_dir(path: Optional[str] = None):
    path = os.path.abspath(path or os.environ.get(MODEL_ENV_ARG, DEFAULT_MODEL_DIR))
    if os.path.isdir(path): return path
    return os.path.dirname(path)

def model_alias(path: str) -> str:
    return path.split(os.path.sep)[-1].split(FILE_EXT)[0]

class LlamaProxy:
    _model: Optional[llama_cpp.Llama] = None
    _models: dict[str,ModelSettings] = {}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        for model in settings.models:
            if not model.model_alias:
                model.model_alias = model_alias(model.model)
            self._models[model.model_alias] = model

        model_root = models_root_dir(settings.model)
        for filename in os.listdir(model_root):
            if filename.endswith(FILE_EXT):
                alias = model_alias(filename)
                if alias in self._models: continue
                exclude={'model', 'model_alias', 'models', 'host', 'port', 'interrupt_requests', 'config'}
                default_settings = settings.model_dump(exclude=exclude)
                self._models[alias] = ModelSettings(model=os.path.join(model_root, filename), 
                                                    model_alias=alias, **default_settings)

        if os.path.isfile(settings.model):
            alias = settings.model_alias
            if alias is None: alias = model_alias(settings.model)
            if alias not in self._models:
                self._models[alias] = settings
            self(alias)

    def __call__(self, model: Optional[str] = None) -> llama_cpp.Llama:
        # handle backward compatibility, model param optional
        try:
            model_alias = self._models[model].model_alias
        except KeyError:
            if self._model:
                if self._settings.verbose: logger.warn(f"Model {model} NOT found! Using {self._model.alias}")
                return self._model
            else: raise Exception(404, f"Model {model} NOT found!")
        
        if self._model:
            if self._model.alias == model_alias:
                return self._model
            del self._model

        settings = self._models[model]

        chat_handler = None
        if settings.chat_format == "llava-1-5":
            assert settings.clip_model_path is not None, "clip model not found"
            chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                clip_model_path=settings.clip_model_path,
                verbose=settings.verbose
            )

        self._model = llama_cpp.Llama(
            model_path=settings.model,
            **(settings.model_dump(exclude={'model', 'models'})),
            chat_handler=chat_handler
        )
        self._model.alias = model
        return self._model

    def __getitem__(self, model: str):
        return self._models[model].model_dump()

    def __setitem__(self, model: str, settings: Union[ModelSettings, str, bytes]):
        if isinstance(settings, (bytes, str)):
            settings = ModelSettings.model_validate_json(settings)
        self._models[model] = settings
    
    def __iter__(self):
        for model in self._models:
            yield model

    def free(self):
        if self._model: del self._model
    
LLAMA: Optional[LlamaProxy] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()

def set_llama(settings: Settings):
    global LLAMA
    LLAMA = LlamaProxy(settings)

def get_llama():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield LLAMA
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()
