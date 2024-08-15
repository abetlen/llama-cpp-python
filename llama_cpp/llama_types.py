"""Types and request signatures for OpenAI compatibility

NOTE: These types may change to match the OpenAI OpenAPI specification.

Based on the OpenAI OpenAPI specification:
https://github.com/openai/openai-openapi/blob/master/openapi.yaml

"""

from typing import Any, Dict, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

# NOTE: Defining this correctly using annotations seems to break pydantic validation.
#       This is a workaround until we can figure out how to do this correctly
# JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]
JsonType = Union[None, int, str, bool, list[Any], dict[str, Any]]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class Embedding(TypedDict):
    index: int
    object: str
    embedding: list[float] | list[list[float]]


class CreateEmbeddingResponse(TypedDict):
    object: Literal["list"]
    model: str
    data: list[Embedding]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: list[int]
    token_logprobs: list[float | None]
    tokens: list[str]
    top_logprobs: list[dict[str, float] | None]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: CompletionLogprobs | None
    finish_reason: Literal["stop", "length"] | None


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: NotRequired[CompletionUsage]


class ChatCompletionResponseFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionResponseMessage(TypedDict):
    content: str | None
    tool_calls: NotRequired["ChatCompletionMessageToolCalls"]
    role: Literal["assistant", "function"]  # NOTE: "function" may be incorrect here
    function_call: NotRequired[ChatCompletionResponseFunctionCall]  # DEPRECATED


class ChatCompletionFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: "ChatCompletionResponseMessage"
    logprobs: CompletionLogprobs | None
    finish_reason: str | None


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list["ChatCompletionResponseChoice"]
    usage: CompletionUsage


class ChatCompletionMessageToolCallChunkFunction(TypedDict):
    name: str | None
    arguments: str


class ChatCompletionMessageToolCallChunk(TypedDict):
    index: int
    id: NotRequired[str]
    type: Literal["function"]
    function: ChatCompletionMessageToolCallChunkFunction


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


class ChatCompletionStreamResponseDeltaFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[str | None]
    function_call: NotRequired[
        ChatCompletionStreamResponseDeltaFunctionCall | None
    ]  # DEPRECATED
    tool_calls: NotRequired[list[ChatCompletionMessageToolCallChunk] | None]
    role: NotRequired[Literal["system", "user", "assistant", "tool"] | None]


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: ChatCompletionStreamResponseDelta | ChatCompletionStreamResponseDeltaEmpty
    finish_reason: Literal["stop", "length", "tool_calls", "function_call"] | None
    logprobs: NotRequired[CompletionLogprobs | None]


class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: list[ChatCompletionStreamResponseChoice]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionFunctionCallOption(TypedDict):
    name: str


class ChatCompletionRequestResponseFormat(TypedDict):
    type: Literal["text", "json_object"]
    schema: NotRequired[
        JsonType
    ]  # https://docs.endpoints.anyscale.com/guides/json_mode/


class ChatCompletionRequestMessageContentPartText(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionRequestMessageContentPartImageImageUrl(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ChatCompletionRequestMessageContentPartImage(TypedDict):
    type: Literal["image_url"]
    image_url: str | ChatCompletionRequestMessageContentPartImageImageUrl


ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
]


class ChatCompletionRequestSystemMessage(TypedDict):
    role: Literal["system"]
    content: str | None


class ChatCompletionRequestUserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[ChatCompletionRequestMessageContentPart] | None


class ChatCompletionMessageToolCallFunction(TypedDict):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction


ChatCompletionMessageToolCalls = list[ChatCompletionMessageToolCall]


class ChatCompletionRequestAssistantMessageFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionRequestAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[ChatCompletionMessageToolCalls]
    function_call: NotRequired[
        ChatCompletionRequestAssistantMessageFunctionCall
    ]  # DEPRECATED


class ChatCompletionRequestToolMessage(TypedDict):
    role: Literal["tool"]
    content: str | None
    tool_call_id: str


class ChatCompletionRequestFunctionMessage(TypedDict):
    role: Literal["function"]
    content: str | None
    name: str


ChatCompletionRequestMessage = Union[
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
]


class ChatCompletionRequestFunctionCallOption(TypedDict):
    name: str


ChatCompletionRequestFunctionCall = Union[
    Literal["none", "auto"], ChatCompletionRequestFunctionCallOption,
]

ChatCompletionFunctionParameters = dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: ChatCompletionFunctionParameters


class ChatCompletionTool(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolFunction


class ChatCompletionNamedToolChoiceFunction(TypedDict):
    name: str


class ChatCompletionNamedToolChoice(TypedDict):
    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto", "required"], ChatCompletionNamedToolChoice,
]


# NOTE: The following type names are not part of the OpenAI OpenAPI specification
# and will be removed in a future major release.

EmbeddingData = Embedding
CompletionChunk = CreateCompletionResponse
Completion = CreateCompletionResponse
CreateCompletionStreamResponse = CreateCompletionResponse
ChatCompletionMessage = ChatCompletionResponseMessage
ChatCompletionChoice = ChatCompletionResponseChoice
ChatCompletion = CreateChatCompletionResponse
ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty
ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice
ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta
ChatCompletionChunk = CreateChatCompletionStreamResponse
ChatCompletionStreamResponse = CreateChatCompletionStreamResponse
ChatCompletionResponseFunction = ChatCompletionFunction
ChatCompletionFunctionCall = ChatCompletionResponseFunctionCall
