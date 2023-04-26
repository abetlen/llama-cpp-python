"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi uvicorn sse-starlette
export MODEL=../models/7B/...
uvicorn fastapi_server_chat:app --reload
```

Then visit http://localhost:8000/docs to see the interactive API docs.

"""
import os
import json
from threading import Lock
from typing import List, Optional, Literal, Union, Iterator, Dict
from typing_extensions import TypedDict

import llama_cpp

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model: str
    n_ctx: int = int(os.getenv('N_CTX', '2048'))
    n_batch: int = int(os.getenv('N_BATCH', '512'))
    n_threads: int = int(os.getenv('N_THREADS', '4'))
    f16_kv: bool = bool(os.getenv('F16_KV', 'True'))
    use_mlock: bool = bool(os.getenv('USE_MLOCK', 'False'))
    use_mmap: bool = bool(os.getenv('USE_MMAP', 'True'))
    embedding: bool = bool(os.getenv('EMBEDDING', 'True'))
    last_n_tokens_size: int = int(os.getenv('LAST_N_TOKENS_SIZE', '64'))
    logits_all: bool = bool(os.getenv('LOGITS_ALL', 'False'))
    cache: bool = bool(os.getenv('CACHE', 'False'))

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
settings = Settings()
llama = llama_cpp.Llama(
    settings.model,
    f16_kv=settings.f16_kv,
    use_mlock=settings.use_mlock,
    use_mmap=settings.use_mmap,
    embedding=settings.embedding,
    logits_all=settings.logits_all,
    n_threads=settings.n_threads,
    n_batch=settings.n_batch,
    n_ctx=settings.n_ctx,
    last_n_tokens_size=settings.last_n_tokens_size,
)
if settings.cache:
    cache = llama_cpp.LlamaCache()
    llama.set_cache(cache)
llama_lock = Lock()


def get_llama():
    with llama_lock:
        yield llama


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]]
    suffix: Optional[str] = Field(None)
    max_tokens: int = int(os.getenv('MAX_TOKENS', '16'))
    temperature: float = float(os.getenv('TEMPERATURE', '0.8'))
    top_p: float = float(os.getenv('TOP_P', '0.95'))
    echo: bool = bool(os.getenv('ECHO', 'False'))
    stop: Optional[List[str]] = []
    stream: bool = False

    # ignored or currently unsupported
    model: Optional[str] = Field(None)
    n: Optional[int] = 1
    logprobs: Optional[int] = Field(None)
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = int(os.getenv('TOP_K', '40'))
    repeat_penalty: float = float(os.getenv('REPEAT_PENALTY', '1.1'))

    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


CreateCompletionResponse = create_model_from_typeddict(llama_cpp.Completion)


@app.post(
    "/v1/completions",
    response_model=CreateCompletionResponse,
)
def create_completion(
    request: CreateCompletionRequest, llama: llama_cpp.Llama = Depends(get_llama)
):
    if isinstance(request.prompt, list):
        request.prompt = "".join(request.prompt)

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
    model: Optional[str]
    input: str
    user: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "input": "The food was delicious and the waiter...",
            }
        }


CreateEmbeddingResponse = create_model_from_typeddict(llama_cpp.Embedding)


@app.post(
    "/v1/embeddings",
    response_model=CreateEmbeddingResponse,
)
def create_embedding(
    request: CreateEmbeddingRequest, llama: llama_cpp.Llama = Depends(get_llama)
):
    return llama.create_embedding(**request.dict(exclude={"model", "user"}))


class ChatCompletionRequestMessage(BaseModel):
    role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
    content: str
    user: Optional[str] = None


class CreateChatCompletionRequest(BaseModel):
    model: Optional[str]
    messages: List[ChatCompletionRequestMessage]
    temperature: float = float(os.getenv('TEMPERATURE', '0.8'))
    top_p: float = float(os.getenv('TOP_P', '0.95'))
    stream: bool = bool(os.getenv('STREAM', 'False'))
    stop: Optional[List[str]] = []
    max_tokens: int = int(os.getenv('MAX_TOKENS', '128'))

    # ignored or currently unsupported
    model: Optional[str] = Field(None)
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    repeat_penalty: float = float(os.getenv('REPEAT_PENALTY', '1.1'))

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


@app.post(
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


@app.get("/v1/models", response_model=GetModelResponse)
def get_models() -> ModelList:
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


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8000))
    )
