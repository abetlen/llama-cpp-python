from __future__ import annotations

import collections
import os
import json
from typing import Annotated, Callable
import asyncio
from functools import partial
from typing import Iterator, List, Optional, Union, Dict

import llama_cpp
import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, HTTPException, status, Body
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse
from starlette_context.plugins import RequestIdPlugin  # type: ignore
from starlette_context.middleware import RawContextMiddleware

from llama_cpp.server.settings import read_config
from llama_cpp.server.model import (
    LlamaProxy,
)
from llama_cpp.server.settings import (
    Settings,
    ModelSettings,
    ServerSettings,
)
from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    CreateChatCompletionRequest,
    ModelList,
    TokenizeInputRequest,
    TokenizeInputResponse,
    TokenizeInputCountResponse,
    DetokenizeInputRequest,
    DetokenizeInputResponse,
)
from llama_cpp.server.errors import RouteErrorHandler

router = APIRouter(route_class=RouteErrorHandler)

_server_settings: Optional[ServerSettings] = None


def set_server_settings(server_settings: ServerSettings):
    global _server_settings
    _server_settings = server_settings


def get_server_settings():
    yield _server_settings


_llama_proxy_context_manager: Optional[LlamaProxyContextManager] = None


def set_llama_proxy_context_manage(model_settings: List[ModelSettings]):
    global _llama_proxy_context_manager
    _llama_proxy_context_manager = LlamaProxyContextManager(model_settings)


def get_llama_proxy_context_manager():
    return _llama_proxy_context_manager


class LlamaProxyContextManager:
    _llama_proxy: LlamaProxy
    _lock = asyncio.Lock()

    def __init__(self, model_settings: List[ModelSettings]):
        self._llama_proxy = LlamaProxy(models=model_settings)

    async def __aenter__(self) -> LlamaProxy:
        await self._lock.acquire()
        return self._llama_proxy

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()


_ping_message_factory = None


def set_ping_message_factory(factory):
    global _ping_message_factory
    _ping_message_factory = factory


def create_app(
        settings: Settings | None = None,
        server_settings: ServerSettings | None = None,
        model_settings: List[ModelSettings] | None = None,
):
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is not None:
        server_settings, model_settings = read_config(config_file)

    if server_settings is None and model_settings is None:
        if settings is None:
            settings = Settings()
        server_settings = ServerSettings.model_validate(settings)
        model_settings = [ModelSettings.model_validate(settings)]

    assert (
            server_settings is not None and model_settings is not None
    ), "server_settings and model_settings must be provided together"

    set_server_settings(server_settings)
    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="🦙 llama.cpp Python API",
        version=llama_cpp.__version__,
        root_path=server_settings.root_path,
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
    set_llama_proxy_context_manage(model_settings=model_settings)

    if server_settings.disable_ping_events:
        set_ping_message_factory(lambda: bytes())

    return app


async def get_event_publisher(
        request: Request,
        inner_send_chan: MemoryObjectSendStream,
        iterator: collections.AsyncIterable,
):
    async with inner_send_chan:
        try:
            async for chunk in iterator:
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
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
        settings: Settings = Depends(get_server_settings),
        authorization: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)] = None
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


openai_v1_tag = "OpenAI V1"


async def completion_async_generator(req: Request, request_body: CreateCompletionRequest,
                                     llama_proxy_context_manager: LlamaProxyContextManager, model_name: str,
                                     exclude: set[str], method: Callable) -> collections.AsyncIterable:
    kwargs = request_body.model_dump(exclude=exclude)
    if await req.is_disconnected():
        raise anyio.get_cancelled_exc_class()()
    async with llama_proxy_context_manager as llama_proxy:
        llama = llama_proxy(model_name)

        if request_body.logit_bias is not None:
            kwargs["logit_bias"] = (
                _logit_bias_tokens_to_input_ids(llama, request_body.logit_bias)
                if request_body.logit_bias_type == "tokens"
                else request_body.logit_bias
            )

        if request_body.grammar is not None:
            kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(request_body.grammar)

        if request_body.min_tokens > 0:
            _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
                [llama_cpp.MinTokensLogitsProcessor(request_body.min_tokens, llama.token_eos())]
            )
            if "logits_processor" not in kwargs:
                kwargs["logits_processor"] = _min_tokens_logits_processor
            else:
                kwargs["logits_processor"].extend(_min_tokens_logits_processor)
        if await req.is_disconnected():
            raise anyio.get_cancelled_exc_class()()
        iterator_or_completion: Union[
            llama_cpp.CreateCompletionResponse |
            Iterator[llama_cpp.CreateCompletionStreamResponse],
        ] = await run_in_threadpool(method, llama, **kwargs)
        if await req.is_disconnected():
            raise anyio.get_cancelled_exc_class()()
        if isinstance(iterator_or_completion, Iterator):
            async for chunk in iterate_in_threadpool(iterator_or_completion):
                if await req.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                yield False, chunk
        else:
            yield True, iterator_or_completion


async def handle_completion_request(request: Request, request_body: CreateCompletionRequest,
                                    llama_proxy_context_manager: LlamaProxyContextManager, model_name: str,
                                    exclude: set[str], method: Callable):
    completion_iter = await run_in_threadpool(completion_async_generator, request, request_body,
                                              llama_proxy_context_manager, model_name,
                                              exclude, method)

    first_response = None
    complete_response = False
    async for response in completion_iter:
        complete_response, first_response = response
        break

    if complete_response:
        return first_response

    async def response_async_generator():
        yield first_response
        async for cr, item in completion_iter:
            yield item

    send_chan, recv_chan = anyio.create_memory_object_stream(10)

    return EventSourceResponse(
        recv_chan,
        data_sender_callable=partial(  # type: ignore
            get_event_publisher,
            request=request,
            inner_send_chan=send_chan,
            iterator=response_async_generator(),
        ),
        sep="\n",
        ping_message_factory=_ping_message_factory,
    )


@router.post(
    "/v1/completions",
    summary="Completion",
    dependencies=[Depends(authenticate)],
    response_model=Union[
        llama_cpp.CreateCompletionResponse,
        str,
    ],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/CreateCompletionResponse"}
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True. "
                                 + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server"
                                   "-sent_events/Using_server-sent_events#Event_stream_format",
                        # noqa: E501
                        "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: 
                        [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
@router.post(
    "/v1/engines/copilot-codex/completions",
    include_in_schema=False,
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_completion(
        request: Request,
        body: CreateCompletionRequest,
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> llama_cpp.Completion:
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    exclude = {
        "n",
        "best_of",
        "logit_bias_type",
        "user",
        "min_tokens",
    }

    model_name = body.model if request.url.path != "/v1/engines/copilot-codex/completions" else "copilot-codex"

    method = llama_cpp.Llama.create_completion

    return await handle_completion_request(request, body,
                                           llama_proxy_context_manager, model_name,
                                           exclude, method)


@router.post(
    "/v1/embeddings",
    summary="Embedding",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_embedding(
        request: CreateEmbeddingRequest,
        llama_proxy_context: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
):
    async with llama_proxy_context as llama_proxy:
        return await run_in_threadpool(
            llama_proxy(request.model).create_embedding,
            **request.model_dump(exclude={"user"}),
        )


@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[llama_cpp.ChatCompletion, str],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                                 + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server"
                                   "-sent_events/Using_server-sent_events#Event_stream_format",
                        # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... 
                        data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
async def create_chat_completion(
        request: Request,
        body: CreateChatCompletionRequest = Body(
            openapi_examples={
                "normal": {
                    "summary": "Chat Completion",
                    "value": {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is the capital of France?"},
                        ],
                    },
                },
                "json_mode": {
                    "summary": "JSON Mode",
                    "value": {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Who won the world series in 2020"},
                        ],
                        "response_format": {"type": "json_object"}
                    },
                },
                "tool_calling": {
                    "summary": "Tool Calling",
                    "value": {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Extract Jason is 30 years old."},
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "User",
                                    "description": "User record",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "age": {"type": "number"},
                                        },
                                        "required": ["name", "age"],
                                    },
                                }
                            }
                        ],
                        "tool_choice": {
                            "type": "function",
                            "function": {
                                "name": "User",
                            }
                        }
                    },
                },
                "logprobs": {
                    "summary": "Logprobs",
                    "value": {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is the capital of France?"},
                        ],
                        "logprobs": True,
                        "top_logprobs": 10
                    },
                },
            }
        ),
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> llama_cpp.ChatCompletion:
    exclude = {
        "n",
        "logit_bias_type",
        "user",
        "min_tokens",
    }

    model_name = body.model

    method = llama_cpp.Llama.create_chat_completion

    return await handle_completion_request(request, body,
                                           llama_proxy_context_manager, model_name,
                                           exclude, method)


@router.get(
    "/v1/models",
    summary="Models",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def get_models(
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> ModelList:
    async with llama_proxy_context_manager as llama_proxy:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_alias,
                    "object": "model",
                    "owned_by": "me",
                    "permissions": [],
                }
                for model_alias in llama_proxy
            ],
        }


extras_tag = "Extras"


@router.post(
    "/extras/tokenize",
    summary="Tokenize",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def tokenize(
        body: TokenizeInputRequest,
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> TokenizeInputResponse:
    async with llama_proxy_context_manager as llama_proxy:
        tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)
    return TokenizeInputResponse(tokens=tokens)


@router.post(
    "/extras/tokenize/count",
    summary="Tokenize Count",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def count_query_tokens(
        body: TokenizeInputRequest,
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> TokenizeInputCountResponse:
    async with llama_proxy_context_manager as llama_proxy:
        tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)
    return TokenizeInputCountResponse(count=len(tokens))


@router.post(
    "/extras/detokenize",
    summary="Detokenize",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def detokenize(
        body: DetokenizeInputRequest,
        llama_proxy_context_manager: LlamaProxyContextManager = Depends(get_llama_proxy_context_manager),
) -> DetokenizeInputResponse:
    async with llama_proxy_context_manager as llama_proxy:
        text = llama_proxy(body.model).detokenize(body.tokens).decode("utf-8")
    return DetokenizeInputResponse(text=text)
