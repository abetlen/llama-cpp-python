# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator (autorest: 3.9.5, generator: @autorest/python@6.4.15)
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._enums import (ChatCompletionMessageRole, ChatCompletionObject,
                     ChatCompletionRequestMessageRole, CompletionObject,
                     EmbeddingObject, ModelDataObject, ModelListObject)
from ._models import (ChatCompletion, ChatCompletionChoice,
                      ChatCompletionMessage, ChatCompletionRequestMessage,
                      Completion, CompletionChoice, CompletionLogprobs,
                      CompletionUsage, CreateChatCompletionRequest,
                      CreateCompletionRequest, CreateCompletionRequestStop,
                      CreateEmbeddingRequest, Embedding, EmbeddingData,
                      EmbeddingUsage, HTTPValidationError, ModelData,
                      ModelList, ValidationError, ValidationErrorLocItem)
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import __all__ as _patch_all
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionMessage",
    "ChatCompletionRequestMessage",
    "Completion",
    "CompletionChoice",
    "CompletionLogprobs",
    "CompletionUsage",
    "CreateChatCompletionRequest",
    "CreateCompletionRequest",
    "CreateCompletionRequestStop",
    "CreateEmbeddingRequest",
    "Embedding",
    "EmbeddingData",
    "EmbeddingUsage",
    "HTTPValidationError",
    "ModelData",
    "ModelList",
    "ValidationError",
    "ValidationErrorLocItem",
    "ChatCompletionMessageRole",
    "ChatCompletionObject",
    "ChatCompletionRequestMessageRole",
    "CompletionObject",
    "EmbeddingObject",
    "ModelDataObject",
    "ModelListObject",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
