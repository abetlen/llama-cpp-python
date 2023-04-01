"""Example FastAPI server for llama.cpp.
"""
import json
from typing import List, Optional, Iterator

import llama_cpp

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model: str


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
    f16_kv=True,
    use_mlock=True,
    embedding=True,
    n_threads=6,
    n_batch=2048,
)


class CreateCompletionRequest(BaseModel):
    prompt: str
    suffix: Optional[str] = Field(None)
    max_tokens: int = 16
    temperature: float = 0.8
    top_p: float = 0.95
    logprobs: Optional[int] = Field(None)
    echo: bool = False
    stop: List[str] = []
    repeat_penalty: float = 1.1
    top_k: int = 40
    stream: bool = False

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
    return llama(**request.dict())


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
    return llama.create_embedding(request.input)
