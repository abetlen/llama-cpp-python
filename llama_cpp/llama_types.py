"""Types and request signatrues for OpenAI compatibility

Based on the OpenAI OpenAPI specification:
https://github.com/openai/openai-openapi/blob/master/openapi.yaml

"""
from typing import Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class Embedding(TypedDict):
    index: int
    object: str
    embedding: List[float]


EmbeddingData = Embedding


class CreateEmbeddingResponse(TypedDict):
    object: Literal["list"]
    model: str
    data: List[Embedding]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateCompletionStreamResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]


CompletionChunk = CreateCompletionStreamResponse


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


Completion = CreateCompletionResponse


class ChatCompletionFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionResponseMessage(TypedDict):
    role: Literal["assistant", "user", "system", "function"]
    content: Optional[str]
    user: NotRequired[str]
    function_call: NotRequired[ChatCompletionFunctionCall]


ChatCompletionMessage = ChatCompletionResponseMessage


class ChatCompletionResponseFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, Any]  # TODO: make this more specific


ChatCompletionFunction = ChatCompletionResponseFunction


class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


ChatCompletionChoice = ChatCompletionResponseChoice


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


ChatCompletion = CreateChatCompletionResponse


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty


class ChatCompletionStreamResponseDelta(TypedDict):
    role: NotRequired[Literal["assistant"]]
    content: NotRequired[str]
    function_call: NotRequired[ChatCompletionFunctionCall]


ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union[ChatCompletionChunkDelta, ChatCompletionChunkDeltaEmpty]
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice


class ChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionChunkChoice]


ChatCompletionChunk = ChatCompletionStreamResponse

JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionFunctionCallOption(TypedDict):
    name: str


class ChatCompletionRequestMessage(TypedDict):
    role: Literal["assistant", "user", "system", "function"]
    content: Optional[str]
    name: NotRequired[str]
    funcion_call: NotRequired[ChatCompletionFunctionCall]
