"""Types and request signatures for OpenAI compatibility

NOTE: These types may change to match the OpenAI OpenAPI specification.

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


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionFunctionCall(TypedDict):
    name: str
    arguments: str


class _ChatCompletionTextContent(TypedDict):
    type: Literal["text"]
    text: str


class _ChatCompletionImageUrlContentUrl(TypedDict):
    url: str


class _ChatCompletionImageUrlContent(TypedDict):
    type: Literal["image_url"]
    image_url: _ChatCompletionImageUrlContentUrl


class ChatCompletionResponseMessage(TypedDict):
    role: Literal["assistant", "user", "system", "function"]
    content: Optional[
        Union[str, _ChatCompletionTextContent, _ChatCompletionImageUrlContent]
    ]
    user: NotRequired[str]
    function_call: NotRequired[ChatCompletionFunctionCall]


class ChatCompletionFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, Any]  # TODO: make this more specific


class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: "ChatCompletionMessage"
    finish_reason: Optional[str]


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List["ChatCompletionChoice"]
    usage: CompletionUsage


class ChatCompletionMessageToolCallChunk(TypedDict):
    index: int
    id: NotRequired[str]
    type: Literal["function"]
    function: ChatCompletionFunctionCall


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[str]
    function_call: NotRequired[ChatCompletionFunctionCall]
    tool_calls: NotRequired[List[ChatCompletionMessageToolCallChunk]]
    role: NotRequired[Literal["system", "user", "assistant", "tool"]]


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union["ChatCompletionChunkDelta", "ChatCompletionChunkDeltaEmpty"]
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class ChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List["ChatCompletionChunkChoice"]


JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionFunctionCallOption(TypedDict):
    name: str


class ChatCompletionRequestMessageContentPartText(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionRequestMessageContentPartImageImageUrl(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ChatCompletionRequestMessageContentPartImage(TypedDict):
    type: Literal["image_url"]
    image_url: ChatCompletionRequestMessageContentPartImageImageUrl


ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
]


class ChatCompletionRequestSystemMessage(TypedDict):
    role: Literal["system"]
    content: Optional[str]


class ChatCompletionRequestUserMessage(TypedDict):
    role: Literal["user"]
    content: Optional[Union[str, List[ChatCompletionRequestMessageContentPart]]]


class ChatCompletionMessageToolCallFunction(TypedDict):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction


ChatCompletionMessageToolCalls = List[ChatCompletionMessageToolCall]


class ChatCompletionRequestAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: Optional[str]
    tool_calls: NotRequired[ChatCompletionMessageToolCalls]
    function_call: NotRequired[ChatCompletionFunctionCall]  # DEPRECATED


class ChatCompletionRequestToolMessage(TypedDict):
    role: Literal["tool"]
    content: Optional[str]
    tool_call_id: str


class ChatCompletionRequestFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[str]
    name: str


ChatCompletionRequestMessage = Union[
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
]

# NOTE: The following type names are not part of the OpenAI OpenAPI specification
# and will be removed in a future major release.

EmbeddingData = Embedding
CompletionChunk = CreateCompletionStreamResponse
Completion = CreateCompletionResponse
ChatCompletionMessage = ChatCompletionResponseMessage
ChatCompletionChoice = ChatCompletionResponseChoice
ChatCompletion = CreateChatCompletionResponse
ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty
ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice
ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta
ChatCompletionChunk = ChatCompletionStreamResponse
ChatCompletionResponseFunction = ChatCompletionFunction
