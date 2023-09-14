from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union
from llama_cpp.llama import Llama, LogitsProcessorList
from llama_cpp.llama_grammar import LlamaGrammar

from llama_cpp.llama_types import (
    ChatCompletionFunction,
    ChatCompletionFunctionCall,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)


class ChatCompletionFormat(ABC):
    """Base class for chat completion templates."""

    @abstractmethod
    def create_chat_completion(
        self,
        llama: Llama,
        messages: List[ChatCompletionMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[Union[str, ChatCompletionFunctionCall]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        max_tokens: int = 256,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        raise NotImplementedError


class DefaultChatCompletionFormat(ABC):
    """Base class for chat completion templates."""

    def create_chat_completion(
        self,
        llama: Llama,
        messages: List[ChatCompletionMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[Union[str, ChatCompletionFunctionCall]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        max_tokens: int = 256,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        chat_history = "".join(
            f'### {"Human" if message["role"] == "user" else "Assistant"}:{message["content"]}'
            for message in messages
        )
        PROMPT = chat_history + "### Assistant:"
        PROMPT_STOP = ["### Assistant:", "### Human:"]
        return llama.create_completion(
            prompt=PROMPT,
            stop=PROMPT_STOP + stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
