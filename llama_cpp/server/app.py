import json
import multiprocessing
from threading import Lock
from typing import List, Optional, Union, Iterator, Dict
from typing_extensions import TypedDict, Literal

import llama_cpp

from fastapi import Depends, FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model: str = Field(
        description="The path to the model to use for generating completions."
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
    f16_kv: bool = Field(default=True, description="Whether to use f16 key/value.")
    use_mlock: bool = Field(
        default=llama_cpp.llama_mlock_supported(),
        description="Use mlock.",
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_mmap_supported(),
        description="Use mmap.",
    )
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )


router = APIRouter()

llama: Optional[llama_cpp.Llama] = None


def create_app(settings: Optional[Settings] = None):
    if settings is None:
        settings = Settings()
    app = FastAPI(
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
        f16_kv=settings.f16_kv,
        use_mlock=settings.use_mlock,
        use_mmap=settings.use_mmap,
        embedding=settings.embedding,
        logits_all=settings.logits_all,
        n_threads=settings.n_threads,
        n_batch=settings.n_batch,
        n_ctx=settings.n_ctx,
        last_n_tokens_size=settings.last_n_tokens_size,
        vocab_only=settings.vocab_only,
        verbose=settings.verbose,
    )
    if settings.cache:
        cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
        llama.set_cache(cache)
    return app


llama_lock = Lock()


def get_llama():
    with llama_lock:
        yield llama


model_field = Field(description="The model to use for generating completions.")

max_tokens_field = Field(
    default=16, ge=1, le=2048, description="The maximum number of tokens to generate."
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
    default=1.0,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)


class CreateCompletionRequest(BaseModel):
    prompt: Optional[str] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[List[str]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    logprobs: Optional[int] = Field(None)
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field

    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


CreateCompletionResponse = create_model_from_typeddict(llama_cpp.Completion)


@router.post(
    "/v1/completions",
    response_model=CreateCompletionResponse,
)
def create_completion(
    request: CreateCompletionRequest, llama: llama_cpp.Llama = Depends(get_llama)
):
    completion_or_chunks = llama(
        **request.dict(
            exclude={
                "model",
                "n",
                "frequency_penalty",
                "presence_penalty",
                "best_of",
                "logit_bias",
                "user",
            }
        )
    )
    if request.stream:
        chunks: Iterator[llama_cpp.CompletionChunk] = completion_or_chunks  # type: ignore
        return EventSourceResponse(dict(data=json.dumps(chunk)) for chunk in chunks)
    completion: llama_cpp.Completion = completion_or_chunks  # type: ignore
    return completion


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = model_field
    input: str = Field(description="The input to embed.")
    user: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "input": "The food was delicious and the waiter...",
            }
        }


CreateEmbeddingResponse = create_model_from_typeddict(llama_cpp.Embedding)


@router.post(
    "/v1/embeddings",
    response_model=CreateEmbeddingResponse,
)
def create_embedding(
    request: CreateEmbeddingRequest, llama: llama_cpp.Llama = Depends(get_llama)
):
    return llama.create_embedding(**request.dict(exclude={"model", "user"}))


class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    stop: Optional[List[str]] = stop_field
    stream: bool = stream_field

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    ChatCompletionRequestMessage(
                        role="system", content="You are a helpful assistant."
                    ),
                    ChatCompletionRequestMessage(
                        role="user", content="What is the capital of France?"
                    ),
                ]
            }
        }


CreateChatCompletionResponse = create_model_from_typeddict(llama_cpp.ChatCompletion)


@router.post(
    "/v1/chat/completions",
    response_model=CreateChatCompletionResponse,
)
def create_chat_completion(
    request: CreateChatCompletionRequest,
    llama: llama_cpp.Llama = Depends(get_llama),
) -> Union[llama_cpp.ChatCompletion, EventSourceResponse]:
    completion_or_chunks = llama.create_chat_completion(
        **request.dict(
            exclude={
                "model",
                "n",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
            }
        ),
    )

    if request.stream:

        async def server_sent_events(
            chat_chunks: Iterator[llama_cpp.ChatCompletionChunk],
        ):
            for chat_chunk in chat_chunks:
                yield dict(data=json.dumps(chat_chunk))
            yield dict(data="[DONE]")

        chunks: Iterator[llama_cpp.ChatCompletionChunk] = completion_or_chunks  # type: ignore

        return EventSourceResponse(
            server_sent_events(chunks),
        )
    completion: llama_cpp.ChatCompletion = completion_or_chunks  # type: ignore
    return completion


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


GetModelResponse = create_model_from_typeddict(ModelList)


@router.get("/v1/models", response_model=GetModelResponse)
def get_models(
    llama: llama_cpp.Llama = Depends(get_llama),
) -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": llama.model_path,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }
