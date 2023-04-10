"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi uvicorn sse-starlette
export MODEL=../models/7B/ggml-model.bin
uvicorn fastapi_server_chat:app --reload
```

Then visit http://localhost:8000/docs to see the interactive API docs.

"""
import os
import json
from typing import List, Optional, Literal, Union, Iterator, Dict
from typing_extensions import TypedDict

import llama_cpp

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model: str
    n_ctx: int = 2048
    n_batch: int = 8
    n_threads: int = int(os.cpu_count() / 2) or 1
    f16_kv: bool = True
    use_mlock: bool = False     # This causes a silent failure on platforms that don't support mlock (e.g. Windows) took forever to figure out...
    embedding: bool = True
    last_n_tokens_size: int = 64


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
    embedding=settings.embedding,
    n_threads=settings.n_threads,
    n_batch=settings.n_batch,
    n_ctx=settings.n_ctx,
    last_n_tokens_size=settings.last_n_tokens_size,
)


class CreateCompletionRequest(BaseModel):
    prompt: str
    suffix: Optional[str] = Field(None)
    max_tokens: int = 16
    temperature: float = 0.8
    top_p: float = 0.95
    echo: bool = False
    stop: List[str] = []
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
    top_k: int = 40
    repeat_penalty: float = 1.1

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
def create_completion(request: CreateCompletionRequest):
    if request.stream:
        chunks: Iterator[llama_cpp.CompletionChunk] = llama(**request.dict())  # type: ignore
        return EventSourceResponse(dict(data=json.dumps(chunk)) for chunk in chunks)
    return llama(
        **request.dict(
            exclude={
                "model",
                "n",
                "logprobs",
                "frequency_penalty",
                "presence_penalty",
                "best_of",
                "logit_bias",
                "user",
            }
        )
    )


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
def create_embedding(request: CreateEmbeddingRequest):
    return llama.create_embedding(**request.dict(exclude={"model", "user"}))


class ChatCompletionRequestMessage(BaseModel):
    role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
    content: str
    user: Optional[str] = None


class CreateChatCompletionRequest(BaseModel):
    model: Optional[str]
    messages: List[ChatCompletionRequestMessage]
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False
    stop: List[str] = []
    max_tokens: int = 128

    # ignored or currently unsupported
    model: Optional[str] = Field(None)
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    repeat_penalty: float = 1.1

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
async def create_chat_completion(
    request: CreateChatCompletionRequest,
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

    uvicorn.run(app, host=os.getenv("HOST", "localhost"), port=os.getenv("PORT", 8000))
