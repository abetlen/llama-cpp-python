"""Example FastAPI server for llama.cpp.
"""
from typing import List, Optional

from llama_cpp import Llama

from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings, Field


class Settings(BaseSettings):
    model: str


app = FastAPI(
    title="🦙 llama.cpp Python API",
    version="0.0.1",
)
settings = Settings()
llama = Llama(settings.model)


class CompletionRequest(BaseModel):
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

    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    return llama(**request.dict())
