import sys
import json
import multiprocessing
import time
from re import compile, Match, Pattern
from threading import Lock
from functools import partial
from typing import Callable, Coroutine, Iterator, List, Optional, Tuple, Union, Dict
from typing_extensions import TypedDict, Literal

import llama_cpp

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, Response
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse
from starlette_context import plugins
from starlette_context.middleware import RawContextMiddleware

import numpy as np
import numpy.typing as npt


# Disable warning for model and model_alias settings
BaseSettings.model_config['protected_namespaces'] = ()


class Settings(BaseSettings):
    model: str = Field(
        description="The path to the model to use for generating completions."
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="The alias of the model to use for generating completions.",
    )
    seed: int = Field(default=llama_cpp.LLAMA_DEFAULT_SEED, description="Random seed. -1 for random.")
    n_ctx: int = Field(default=2048, ge=1, description="The context size.")
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="The number of layers to put on the GPU. The rest will be on the CPU.",
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
    rope_freq_base: float = Field(
        default=0.0, description="RoPE base frequency"
    )
    rope_freq_scale: float = Field(
        default=0.0, description="RoPE frequency scaling factor"
    )
    mul_mat_q: bool = Field(
        default=True, description="if true, use experimental mul_mat_q kernels"
    )
    f16_kv: bool = Field(default=True, description="Whether to use f16 key/value.")
    logits_all: bool = Field(default=True, description="Whether to return logits.")
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
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use.",
    )
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    lora_base: Optional[str] = Field(
        default=None,
        description="Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model."
    )
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to a LoRA file to apply to the model.",
    )
    numa: bool = Field(
        default=False,
        description="Enable NUMA support.",
    )
    chat_format: str = Field(
        default="llama-2",
        description="Chat format to use.",
    )
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
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )
    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8000, description="Listen port")
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )


class ErrorResponse(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class ErrorResponseFormatters:
    """Collection of formatters for error responses.

    Args:
        request (Union[CreateCompletionRequest, CreateChatCompletionRequest]):
            Request body
        match (Match[str]): Match object from regex pattern

    Returns:
        Tuple[int, ErrorResponse]: Status code and error response
    """

    @staticmethod
    def context_length_exceeded(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for context length exceeded error"""

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_tokens
        if hasattr(request, "messages"):
            # Chat completion
            message = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens "
                "({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            # Text completion
            message = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens "
                "({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=message.format(
                context_window,
                completion_tokens + prompt_tokens,
                prompt_tokens,
                completion_tokens,
            ),
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )

    @staticmethod
    def model_not_found(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for model_not_found error"""

        model_path = str(match.group(1))
        message = f"The model `{model_path}` does not exist"
        return 400, ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=None,
            code="model_not_found",
        )


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    # key: regex pattern for original error message from llama_cpp
    # value: formatter function
    pattern_and_formatters: Dict[
        "Pattern",
        Callable[
            [
                Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
                "Match[str]",
            ],
            Tuple[int, ErrorResponse],
        ],
    ] = {
        compile(
            r"Requested tokens \((\d+)\) exceed context window of (\d+)"
        ): ErrorResponseFormatters.context_length_exceeded,
        compile(
            r"Model path does not exist: (.+)"
        ): ErrorResponseFormatters.model_not_found,
    }

    def error_message_wrapper(
        self,
        error: Exception,
        body: Optional[
            Union[
                "CreateChatCompletionRequest",
                "CreateCompletionRequest",
                "CreateEmbeddingRequest",
            ]
        ] = None,
    ) -> Tuple[int, ErrorResponse]:
        """Wraps error message in OpenAI style error response"""
        print(f"Exception: {str(error)}", file=sys.stderr)
        if body is not None and isinstance(
            body,
            (
                CreateCompletionRequest,
                CreateChatCompletionRequest,
            ),
        ):
            # When text completion or chat completion
            for pattern, callback in self.pattern_and_formatters.items():
                match = pattern.search(str(error))
                if match is not None:
                    return callback(body, match)

        # Wrap other errors as internal server error
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=None,
            code=None,
        )

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Defines custom route handler that catches exceptions and formats
        in OpenAI style error response"""

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                start_sec = time.perf_counter()
                response = await original_route_handler(request)
                elapsed_time_ms = int((time.perf_counter() - start_sec) * 1000)
                response.headers["openai-processing-ms"] = f"{elapsed_time_ms}"
                return response
            except Exception as exc:
                json_body = await request.json()
                try:
                    if "messages" in json_body:
                        # Chat completion
                        body: Optional[
                            Union[
                                CreateChatCompletionRequest,
                                CreateCompletionRequest,
                                CreateEmbeddingRequest,
                            ]
                        ] = CreateChatCompletionRequest(**json_body)
                    elif "prompt" in json_body:
                        # Text completion
                        body = CreateCompletionRequest(**json_body)
                    else:
                        # Embedding
                        body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    # Invalid request body
                    body = None

                # Get proper error message from the exception
                (
                    status_code,
                    error_message,
                ) = self.error_message_wrapper(error=exc, body=body)
                return JSONResponse(
                    {"error": error_message},
                    status_code=status_code,
                )

        return custom_route_handler


router = APIRouter(route_class=RouteErrorHandler)

settings: Optional[Settings] = None
llama: Optional[llama_cpp.Llama] = None


def create_app(settings: Optional[Settings] = None):
    if settings is None:
        settings = Settings()

    middleware = [
        Middleware(RawContextMiddleware, plugins=(plugins.RequestIdPlugin(),))
    ]
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
    global llama
    llama = llama_cpp.Llama(
        model_path=settings.model,
        seed=settings.seed,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        n_gpu_layers=settings.n_gpu_layers,
        main_gpu=settings.main_gpu,
        tensor_split=settings.tensor_split,
        rope_freq_base=settings.rope_freq_base,
        rope_freq_scale=settings.rope_freq_scale,
        mul_mat_q=settings.mul_mat_q,
        f16_kv=settings.f16_kv,
        logits_all=settings.logits_all,
        vocab_only=settings.vocab_only,
        use_mmap=settings.use_mmap,
        use_mlock=settings.use_mlock,
        embedding=settings.embedding,
        n_threads=settings.n_threads,
        last_n_tokens_size=settings.last_n_tokens_size,
        lora_base=settings.lora_base,
        lora_path=settings.lora_path,
        numa=settings.numa,
        chat_format=settings.chat_format,
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

        cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
        llama.set_cache(cache)

    def set_settings(_settings: Settings):
        global settings
        settings = _settings

    set_settings(settings)
    return app


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
            llama_outer_lock.release()
            release_outer_lock = False
            yield llama
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


def get_settings():
    yield settings


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
                if settings.interrupt_requests and llama_outer_lock.locked():
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e


model_field = Field(
    description="The model to use for generating completions.", default=None
)

max_tokens_field = Field(
    default=16, ge=1, description="The maximum number of tokens to generate."
)

temperature_field = Field(
    default=0.8,
    ge=0.0,
    le=2.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p_field = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

stop_field = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream_field = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k_field = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    logprobs: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(default=None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                    "stop": ["\n", "###"],
                }
            ]
        }
    }


def make_logit_bias_processor(
    llama: llama_cpp.Llama,
    logit_bias: Dict[str, float],
    logit_bias_type: Optional[Literal["input_ids", "tokens"]],
):
    if logit_bias_type is None:
        logit_bias_type = "input_ids"

    to_bias: Dict[int, float] = {}
    if logit_bias_type == "input_ids":
        for input_id, score in logit_bias.items():
            input_id = int(input_id)
            to_bias[input_id] = score

    elif logit_bias_type == "tokens":
        for token, score in logit_bias.items():
            token = token.encode("utf-8")
            for input_id in llama.tokenize(token, add_bos=False):
                to_bias[input_id] = score

    def logit_bias_processor(
        input_ids: npt.NDArray[np.intc],
        scores: npt.NDArray[np.single],
    ) -> npt.NDArray[np.single]:
        new_scores = [None] * len(scores)
        for input_id, score in enumerate(scores):
            new_scores[input_id] = score + to_bias.get(input_id, 0.0)

        return new_scores

    return logit_bias_processor


@router.post(
    "/v1/completions",
)
@router.post("/v1/engines/copilot-codex/completions")
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
) -> llama_cpp.Completion:
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    exclude = {
        "n",
        "best_of",
        "logit_bias",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
            [
                make_logit_bias_processor(llama, body.logit_bias, body.logit_bias_type),
            ]
        )

    iterator_or_completion: Union[
        llama_cpp.Completion, Iterator[llama_cpp.CompletionChunk]
    ] = await run_in_threadpool(llama, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CompletionChunk]:
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


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = model_field
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


@router.post(
    "/v1/embeddings",
)
async def create_embedding(
    request: CreateEmbeddingRequest, llama: llama_cpp.Llama = Depends(get_llama)
):
    return await run_in_threadpool(
        llama.create_embedding, **request.model_dump(exclude={"user"})
    )


class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    functions: Optional[List[llama_cpp.ChatCompletionFunction]] = Field(
        default=None,
        description="A list of functions to apply to the generated completions.",
    )
    function_call: Optional[Union[str, llama_cpp.ChatCompletionFunctionCall]] = Field(
        default=None,
        description="A function to apply to the generated completions.",
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    stop: Optional[List[str]] = stop_field
    stream: bool = stream_field
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ).model_dump(),
                        ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ).model_dump(),
                    ]
                }
            ]
        }
    }


@router.post(
    "/v1/chat/completions",
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
    settings: Settings = Depends(get_settings),
) -> llama_cpp.ChatCompletion:
    exclude = {
        "n",
        "logit_bias",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
            [
                make_logit_bias_processor(llama, body.logit_bias, body.logit_bias_type),
            ]
        )

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


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


@router.get("/v1/models")
async def get_models(
    settings: Settings = Depends(get_settings),
) -> ModelList:
    assert llama is not None
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_alias
                if settings.model_alias is not None
                else llama.model_path,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }
