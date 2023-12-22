from __future__ import annotations

import os
import json

from threading import Lock
from functools import partial
from typing import Iterator, List, Optional, Union, Dict

import llama_cpp

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import (
    Depends,
    FastAPI,
    APIRouter,
    Request,
    HTTPException,
    status,
)
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from sse_starlette.sse import EventSourceResponse
from starlette_context.plugins import RequestIdPlugin  # type: ignore
from starlette_context.middleware import RawContextMiddleware

from llama_cpp.server.model import (
    LlamaProxy,
)
from llama_cpp.server.settings import (
    ConfigFileSettings,
    Settings,
    ModelSettings,
    ServerSettings,
)
from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    CreateChatCompletionRequest,
    ModelList,
)
from llama_cpp.server.errors import RouteErrorHandler


router = APIRouter(route_class=RouteErrorHandler)

_settings: Optional[ServerSettings] = None


def set_settings(settings: ServerSettings):
    global _settings
    _settings = settings


def get_settings():
    yield _settings


LLAMA: Optional[LlamaProxy] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()


def set_llama(models: List[ModelSettings]):
    global LLAMA
    LLAMA = LlamaProxy(models=models)


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


def create_app(
    settings: Settings | None = None,
    server_settings: ServerSettings | None = None,
    model_settings: List[ModelSettings] | None = None,
):
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is not None:
        if not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} not found!")
        with open(config_file, "rb") as f:
            config_file_settings = ConfigFileSettings.model_validate_json(f.read())
            server_settings = ServerSettings(
                **{
                    k: v
                    for k, v in config_file_settings.model_dump().items()
                    if k in ServerSettings.model_fields
                }
            )
            model_settings = config_file_settings.models

    if server_settings is None and model_settings is None:
        if settings is None:
            settings = Settings()
        server_settings = ServerSettings(
            **{
                k: v
                for k, v in settings.model_dump().items()
                if k in ServerSettings.model_fields
            }
        )
        model_settings = [
            ModelSettings(
                **{
                    k: v
                    for k, v in settings.model_dump().items()
                    if k in ModelSettings.model_fields
                }
            )
        ]

    assert (
        server_settings is not None and model_settings is not None
    ), "server_settings and model_settings must be provided together"

    set_settings(server_settings)
    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="🦙 llama.cpp Python API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    assert model_settings is not None
    set_llama(models=model_settings)

    return app


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                if (
                    next(get_settings()).interrupt_requests
                    and llama_outer_lock.locked()
                ):
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e


def _logit_bias_tokens_to_input_ids(
    llama: llama_cpp.Llama,
    logit_bias: Dict[str, float],
) -> Dict[str, float]:
    to_bias: Dict[str, float] = {}
    for token, score in logit_bias.items():
        token = token.encode("utf-8")
        for input_id in llama.tokenize(token, add_bos=False, special=True):
            to_bias[str(input_id)] = score
    return to_bias


# Setup Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    settings: Settings = Depends(get_settings),
    authorization: Optional[str] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


@router.post("/v1/completions", summary="Completion")
@router.post("/v1/engines/copilot-codex/completions", include_in_schema=False)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
    authenticated: str = Depends(authenticate),
) -> llama_cpp.Completion:
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    llama = llama(body.model)

    exclude = {
        "n",
        "best_of",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    iterator_or_completion: Union[
        llama_cpp.CreateCompletionResponse,
        Iterator[llama_cpp.CreateCompletionStreamResponse],
    ] = await run_in_threadpool(llama, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


@router.post("/v1/embeddings", summary="Embedding")
async def create_embedding(
    request: CreateEmbeddingRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
    authenticated: str = Depends(authenticate),
):
    return await run_in_threadpool(
        llama(request.model).create_embedding, **request.model_dump(exclude={"user"})
    )


@router.post("/v1/chat/completions", summary="Chat")
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
    settings: Settings = Depends(get_settings),
    authenticated: str = Depends(authenticate),
) -> llama_cpp.ChatCompletion:
    exclude = {
        "n",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)
    llama = llama(body.model)
    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    iterator_or_completion: Union[
        llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]
    ] = await run_in_threadpool(llama.create_chat_completion, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


@router.get("/v1/models", summary="Models")
async def get_models(
    settings: Settings = Depends(get_settings),
    authenticated: str = Depends(authenticate),
    llama: llama_cpp.Llama = Depends(get_llama),
) -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": model_alias,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_alias in llama
        ],
    }
