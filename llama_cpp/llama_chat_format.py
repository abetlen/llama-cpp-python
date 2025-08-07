from __future__ import annotations

import os
import sys
import json
import ctypes
import dataclasses
import random
import string

from datetime import datetime
from contextlib import ExitStack
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Protocol,
    cast,
)

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

import numpy as np
import numpy.typing as npt

import llama_cpp.llama_cpp as llama_cpp
import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar

from ._logger import logger
from ._utils import suppress_stdout_stderr, Singleton

### Common Chat Templates and Special Tokens ###

# Source: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json
CHATML_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
CHATML_BOS_TOKEN = "<s>"
CHATML_EOS_TOKEN = "<|im_end|>"

# Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
MISTRAL_INSTRUCT_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
MISTRAL_INSTRUCT_BOS_TOKEN = "<s>"
MISTRAL_INSTRUCT_EOS_TOKEN = "</s>"

# Source: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/tokenizer_config.json
MIXTRAL_INSTRUCT_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

# Source: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
LLAMA3_INSTRUCT_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

### Chat Completion Handler ###


class LlamaChatCompletionHandler(Protocol):
    """Base Protocol for a llama chat completion handler.

    Very generic protocol that can be used to implement any chat format.
    The only hard requirement is that it must return a ChatCompletion when
    stream=False and an iterator of ChatCompletionChunks when stream=True."""

    def __call__(
        self,
        *,
        # llama.cpp instance
        llama: llama.Llama,
        # openai api parameters
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        model: Optional[str] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        # llama.cpp parameters
        min_p: float = 0.05,
        typical_p: float = 1.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]: ...


class LlamaChatCompletionHandlerNotFoundException(Exception):
    pass


class LlamaChatCompletionHandlerRegistry(Singleton):
    _chat_handlers: Dict[str, LlamaChatCompletionHandler] = {}

    def register_chat_completion_handler(
        self,
        name: str,
        chat_handler: LlamaChatCompletionHandler,
        overwrite: bool = False,
    ):
        if not overwrite and name in self._chat_handlers:
            raise ValueError(
                f"Formatter with name '{name}' is already registered. Use `overwrite=True` to overwrite it."
            )
        self._chat_handlers[name] = chat_handler

    def unregister_chat_handler(self, name: str):
        if name in self._chat_handlers:
            del self._chat_handlers[name]
        else:
            raise ValueError(f"No formatter registered under the name '{name}'.")

    def get_chat_completion_handler_by_name(
        self, name: str
    ) -> LlamaChatCompletionHandler:
        try:
            chat_handler = self._chat_handlers[name]
            return chat_handler
        except KeyError:
            raise LlamaChatCompletionHandlerNotFoundException(
                f"Invalid chat handler: {name} (valid formats: {list(self._chat_handlers.keys())})"
            )


def get_chat_completion_handler(name: str) -> LlamaChatCompletionHandler:
    return LlamaChatCompletionHandlerRegistry().get_chat_completion_handler_by_name(
        name
    )


def register_chat_completion_handler(name: str):
    def decorator(f: LlamaChatCompletionHandler):
        LlamaChatCompletionHandlerRegistry().register_chat_completion_handler(name, f)
        return f

    return decorator


### Chat Formatter ###


@dataclasses.dataclass
class ChatFormatterResponse:
    """Dataclass that stores completion parameters for a given chat format and
    create_chat_completion request.

    prompt contains the formatted prompt generated from the chat format and messages.
    stop contains the stop token or list of stop tokens to use for the chat format."""

    prompt: str
    stop: Optional[Union[str, List[str]]] = None
    stopping_criteria: Optional[llama.StoppingCriteriaList] = None
    added_special: bool = False


class ChatFormatter(Protocol):
    """Base Protocol for a chat formatter. A chat formatter is a function that
    takes a list of messages and returns a chat format response which can be used
    to generate a completion. The response can also include a stop token or list
    of stop tokens to use for the completion."""

    def __call__(
        self,
        *,
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse: ...


class Jinja2ChatFormatter(ChatFormatter):
    def __init__(
        self,
        template: str,
        eos_token: str,
        bos_token: str,
        add_generation_prompt: bool = True,
        stop_token_ids: Optional[List[int]] = None,
    ):
        """A chat formatter that uses jinja2 templates to format the prompt."""
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.add_generation_prompt = add_generation_prompt
        self.stop_token_ids = (
            set(stop_token_ids) if stop_token_ids is not None else None
        )

        self._environment = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.template)

    @staticmethod
    def strftime_now(f: str) -> str:
        return datetime.now().strftime(f)

    def __call__(
        self,
        *,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        def raise_exception(message: str):
            raise ValueError(message)

        prompt = self._environment.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=self.add_generation_prompt,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            strftime_now=self.strftime_now,
        )

        stopping_criteria = None
        if self.stop_token_ids is not None:

            def stop_on_last_token(
                tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
            ) -> bool:
                return tokens[-1] in self.stop_token_ids

            stopping_criteria = llama.StoppingCriteriaList([stop_on_last_token])

        return ChatFormatterResponse(
            prompt=prompt,
            stop=[self.eos_token],
            stopping_criteria=stopping_criteria,
            added_special=True,
        )

    def to_chat_handler(self) -> LlamaChatCompletionHandler:
        return chat_formatter_to_chat_completion_handler(self)


def _convert_text_completion_logprobs_to_chat(
    logprobs: Optional[llama_types.CompletionLogprobs],
) -> llama_types.ChatCompletionLogprobs:
    if logprobs is None:
        return None

    return {
        "content": [
            {
                "token": token,
                "bytes": None,
                "logprob": logprob,
                "top_logprobs": [
                    {
                        "token": top_token,
                        "logprob": top_logprob,
                        "bytes": None,
                    }
                    for top_token, top_logprob in top_logprobs.items()
                ],
            } for (token, logprob, top_logprobs) in zip(logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"])
        ],
        "refusal": None,
    }

def _convert_text_completion_to_chat(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "logprobs": _convert_text_completion_logprobs_to_chat(completion["choices"][0]["logprobs"]),
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {}
                    ),
                    "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }


def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)


def _convert_completion_to_chat_function(
    tool_name: str,
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool,
):
    if not stream:
        completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
        assert "usage" in completion
        tool_id = "call_" + "_0_" + tool_name + "_" + completion["id"]
        # TODO: Fix for legacy function calls
        chat_completion: llama_types.CreateChatCompletionResponse = {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": tool_name,
                            "arguments": completion["choices"][0]["text"],
                        },
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                        ],
                    },
                    "logprobs": _convert_text_completion_logprobs_to_chat(completion["choices"][0]["logprobs"]),
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": completion["usage"],
        }
        return chat_completion
    else:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore

        def _stream_response_to_function_stream(
            chunks: Iterator[llama_types.CreateCompletionStreamResponse],
        ) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
            # blank first message
            first = True
            id_ = None
            created = None
            model = None
            tool_id = None
            for chunk in chunks:
                if first:
                    id_ = "chat" + chunk["id"]
                    created = chunk["created"]
                    model = chunk["model"]
                    tool_id = "call_" + "_0_" + tool_name + "_" + chunk["id"]
                    yield {
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": None,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "function_call": None,
                                    "tool_calls": None,
                                },
                            }
                        ],
                    }
                    yield {
                        "id": "chat" + chunk["id"],
                        "object": "chat.completion.chunk",
                        "created": chunk["created"],
                        "model": chunk["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                "delta": {
                                    "role": None,
                                    "content": None,
                                    "function_call": {
                                        "name": tool_name,
                                        "arguments": chunk["choices"][0]["text"],
                                    },
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": chunk["choices"][0][
                                                    "text"
                                                ],
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                    first = False
                    continue
                assert tool_id is not None
                yield {
                    "id": "chat" + chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": {
                                    "name": tool_name,
                                    "arguments": chunk["choices"][0]["text"],
                                },
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": chunk["choices"][0]["text"],
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }

            if id_ is not None and created is not None and model is not None:
                yield {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "logprobs": None,
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": None,
                                "tool_calls": None,
                            },
                        }
                    ],
                }

        return _stream_response_to_function_stream(chunks)


def chat_formatter_to_chat_completion_handler(
    chat_formatter: ChatFormatter,
) -> LlamaChatCompletionHandler:
    def chat_completion_handler(
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        result = chat_formatter(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
        )
        prompt = llama.tokenize(
            result.prompt.encode("utf-8"),
            add_bos=not result.added_special,
            special=True,
        )
        if result.stop is not None:
            stop = [] if stop is None else [stop] if isinstance(stop, str) else stop
            rstop = result.stop if isinstance(result.stop, list) else [result.stop]
            stop = stop + rstop

        stopping_criteria = None
        if result.stopping_criteria is not None:
            stopping_criteria = result.stopping_criteria

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(
                response_format, verbose=llama.verbose
            )

        # Convert legacy functions to tools
        if functions is not None:
            tools = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                # create grammar from json schema
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama.verbose
                )
            except Exception as e:
                if llama.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        if tool is not None:
            tool_name = tool["function"]["name"]
            return _convert_completion_to_chat_function(
                tool_name, completion_or_chunks, stream
            )
        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

    return chat_completion_handler


def hf_autotokenizer_to_chat_formatter(
    pretrained_model_name_or_path: Union[str, os.PathLike[str]]
) -> ChatFormatter:
    # https://huggingface.co/docs/transformers/main/chat_templating
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)  # type: ignore

    def format_autotokenizer(
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        tokenizer.use_default_system_prompt = False  # type: ignore
        prompt: str = tokenizer.apply_chat_template(messages, tokenize=False)  # type: ignore
        assert isinstance(prompt, str)
        # Return formatted prompt and eos token by default
        return ChatFormatterResponse(
            prompt=prompt, stop=tokenizer.eos_token, added_special=True
        )

    return format_autotokenizer


def hf_autotokenizer_to_chat_completion_handler(
    pretrained_model_name_or_path: Union[str, os.PathLike[str]]
) -> LlamaChatCompletionHandler:
    chat_formatter = hf_autotokenizer_to_chat_formatter(pretrained_model_name_or_path)
    return chat_formatter_to_chat_completion_handler(chat_formatter)


def hf_tokenizer_config_to_chat_formatter(
    tokenizer_config: Dict[str, Any],
    add_generation_prompt: bool = True,
) -> ChatFormatter:
    assert isinstance(tokenizer_config, dict)

    assert "chat_template" in tokenizer_config
    assert isinstance(tokenizer_config["chat_template"], str)
    chat_template = tokenizer_config["chat_template"]

    assert "bos_token" in tokenizer_config
    assert isinstance(tokenizer_config["bos_token"], str)
    bos_token = tokenizer_config["bos_token"]

    assert "eos_token" in tokenizer_config
    assert isinstance(tokenizer_config["eos_token"], str)
    eos_token = tokenizer_config["eos_token"]

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    ).from_string(chat_template)

    def format_tokenizer_config(
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        # TODO: veryify this is correct
        # Add a blank assistant message to the end of the messages to prompt the model to generate a response
        if add_generation_prompt:
            messages = [
                *messages,
                llama_types.ChatCompletionRequestAssistantMessage(
                    role="assistant", content=""
                ),
            ]
        prompt = env.render(
            messages=messages,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        return ChatFormatterResponse(
            prompt=prompt, stop=[eos_token, bos_token], added_special=True
        )

    return format_tokenizer_config


def hf_tokenizer_config_to_chat_completion_handler(
    tokenizer_config: Dict[str, Any],
    add_generation_prompt: bool = True,
) -> LlamaChatCompletionHandler:
    chat_formatter = hf_tokenizer_config_to_chat_formatter(
        tokenizer_config, add_generation_prompt=add_generation_prompt
    )
    return chat_formatter_to_chat_completion_handler(chat_formatter)


def guess_chat_format_from_gguf_metadata(metadata: Dict[str, str]) -> Optional[str]:
    if "tokenizer.chat_template" not in metadata:
        return None

    if metadata["tokenizer.chat_template"] == CHATML_CHAT_TEMPLATE:
        return "chatml"

    if (
        metadata["tokenizer.chat_template"] == MISTRAL_INSTRUCT_CHAT_TEMPLATE
        or metadata["tokenizer.chat_template"] == MIXTRAL_INSTRUCT_CHAT_TEMPLATE
    ):
        return "mistral-instruct"

    if metadata["tokenizer.chat_template"] == LLAMA3_INSTRUCT_CHAT_TEMPLATE:
        return "llama-3"

    return None


### Utility functions for formatting chat prompts ###
# TODO: Replace these with jinja2 templates


def _get_system_message(
    messages: List[llama_types.ChatCompletionRequestMessage],
) -> str:
    """Get the first system message."""
    for message in messages:
        if message["role"] == "system":
            return message["content"] or ""
    return ""


def _map_roles(
    messages: List[llama_types.ChatCompletionRequestMessage],
    role_map: Dict[str, str],
) -> List[Tuple[str, Optional[str]]]:
    """Map the message roles."""
    output: List[Tuple[str, Optional[str]]] = []
    for message in messages:
        role = message["role"]
        if role in role_map:
            content: str | None = (
                message["content"] if isinstance(message["content"], str) else None
            )
            output.append((role_map[role], content))
    return output


def _format_llama2(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str, sep2: str
) -> str:
    """Format the prompt with the llama2 style."""
    seps = [sep, sep2]
    ret = system_message + sep
    for i, (role, message) in enumerate(messages):
        if system_message and i == 0:
            m = message or ""
            ret += m + seps[i % 2]
        elif message:
            ret += role + message + " " + seps[i % 2]
        else:
            ret += role + " "
    return ret


def _format_add_colon_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the add-colon-single style."""
    ret = system_message + sep
    for role, message in messages:
        if message:
            ret += role + ": " + message + sep
        else:
            ret += role + ":"
    return ret


def _format_add_colon_two(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str, sep2: str
) -> str:
    """Format the prompt with the add-colon-two style."""
    seps = [sep, sep2]
    ret = system_message + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            ret += role + ": " + message + seps[i % 2]
        else:
            ret += role + ":"
    return ret


def _format_no_colon_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the no-colon-single style."""
    ret = system_message
    for role, message in messages:
        if message:
            ret += role + message + sep
        else:
            ret += role
    return ret


def _format_add_colon_space_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the add-colon-space-single style."""
    ret = system_message + sep
    for role, message in messages:
        if message:
            ret += role + ": " + message + sep
        else:
            ret += role + ": "  # must be end with a space
    return ret


def _format_chatml(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the chatml style."""
    ret = "" if system_message == "" else system_message + sep + "\n"
    for role, message in messages:
        if message:
            ret += role + "\n" + message + sep + "\n"
        else:
            ret += role + "\n"
    return ret


def _format_chatglm3(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the chatglm3 style."""
    ret = ""
    if system_message:
        ret += system_message
    for role, message in messages:
        if message:
            ret += role + "\n" + " " + message
        else:
            ret += role
    return ret


def _grammar_for_json(verbose: bool = False):
    return llama_grammar.LlamaGrammar.from_string(
        llama_grammar.JSON_GBNF, verbose=verbose
    )


def _grammar_for_json_schema(
    schema: str, verbose: bool = False, fallback_to_json: bool = True
):
    try:
        return llama_grammar.LlamaGrammar.from_json_schema(schema, verbose=verbose)
    except Exception as e:
        if fallback_to_json:
            return _grammar_for_json(verbose=verbose)
        else:
            raise e


def _grammar_for_response_format(
    response_format: llama_types.ChatCompletionRequestResponseFormat,
    verbose: bool = False,
):
    if response_format["type"] != "json_object":
        return None

    if "schema" in response_format:
        return _grammar_for_json_schema(
            json.dumps(response_format["schema"]), verbose=verbose
        )
    else:
        return _grammar_for_json(verbose=verbose)


### Chat Formats ###


def register_chat_format(name: str):
    def decorator(f: ChatFormatter):
        chat_completion_handler = chat_formatter_to_chat_completion_handler(f)
        LlamaChatCompletionHandlerRegistry().register_chat_completion_handler(
            name, chat_completion_handler
        )
        return f

    return decorator


# see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py
# system prompt is "embedded" in the first message
@register_chat_format("llama-2")
def format_llama2(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>"
    _roles = dict(user="<s>[INST]", assistant="[/INST]")
    _messages = _map_roles(messages, _roles)
    system_message = _get_system_message(messages)
    if system_message:
        system_message = _system_template.format(system_message=system_message)
    _prompt = _format_llama2(system_message, _messages, " ", "</s>") + "[/INST]"
    return ChatFormatterResponse(prompt=_prompt)


# Chat format for Llama-3 models, see more details at:
# https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202-L229
@register_chat_format("llama-3")
def format_llama3(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(
        system="<|start_header_id|>system<|end_header_id|>\n\n",
        user="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    _sep = "<|eot_id|>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single("", _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


@register_chat_format("alpaca")
def format_alpaca(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="### Instruction", assistant="### Response")
    _sep = "\n\n"
    _sep2 = "</s>"
    system_message = _get_system_message(messages)
    _messages = _map_roles(messages, _roles)
    _prompt = _format_add_colon_two(system_message, _messages, _sep, _sep2)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("qwen")
def format_qwen(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="<|im_start|>user", assistant="<|im_start|>assistant")
    system_message = _get_system_message(messages) or "You are a helpful assistant."
    system_template = "<|im_start|>system\n{system_message}"
    system_message = system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _sep = "<|im_end|>"
    _prompt = _format_chatml(system_message, _messages, _sep)
    _sep2 = "<|endoftext|>"
    return ChatFormatterResponse(prompt=_prompt, stop=_sep2)


@register_chat_format("vicuna")
def format(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    _roles = dict(user="USER", assistant="ASSISTANT")
    _sep = " "
    _sep2 = "</s>"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_two(system_message, _messages, _sep, _sep2)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("oasst_llama")
def format_oasst_llama(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    _roles = dict(user="<|prompter|>", assistant="<|assistant|>")
    _sep = "</s>"
    system_message = _get_system_message(messages)
    system_message = _system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("baichuan-2")
def format_baichuan2(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "{system_message}"
    _roles = dict(user="<reserved_106>", assistant="<reserved_107>")
    _sep = ""
    system_message = _get_system_message(messages)
    system_message = _system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("baichuan")
def format_baichuan(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "{system_message}"
    _roles = dict(user="<reserved_102>", assistant="<reserved_103>")
    _sep = ""
    system_message = _get_system_message(messages)
    system_message = _system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("openbuddy")
def format_openbuddy(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = """You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You can speak fluently in many languages, for example: English, Chinese.
You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.
You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.

"""
    _roles = dict(user="User", assistant="Assistant")
    _sep = "\n"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("redpajama-incite")
def format_redpajama_incite(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = _get_system_message(messages)
    _roles = dict(user="<human>", assistant="<bot>")
    _sep = "\n"
    _stop = "<human>"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_stop)


@register_chat_format("snoozy")
def format_snoozy(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = "### Instruction:\n{system_message}"
    default_system_message = "The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response."
    _system_message = _get_system_message(messages)
    _system_message = (
        _system_message if _system_message != "" else default_system_message
    )
    system_message = system_template.format(system_message=_system_message)
    _roles = dict(user="### Prompt", assistant="### Response")
    _sep = "\n"
    _stop = "###"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_stop)


@register_chat_format("phind")
def format_phind(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="### User Message", assistant="### Assistant")
    _sep = "\n\n"
    _system_message = "### System Prompt\nYou are an intelligent programming assistant."
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(_system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("intel")
def format_intel(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="### User:", assistant="### Assistant:")
    _sep = "\n"
    _system_message = "### System:\n{system_message}"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(_system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("open-orca")
def format_open_orca(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = "{system_message}"
    system_message = (
        "You are a helpful assistant. Please answer truthfully and write out your "
        "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
        "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
        "aren't sure about something, say so clearly. You will act as a professional logician, mathematician, "
        "and physicist. You will also act as the most appropriate type of expert to answer any particular "
        "question or solve the relevant problem; state which expert type your are, if so. Also think of "
        "any particular named expert that would be ideal to answer the relevant question or solve the "
        "relevant problem; name and act as them, if appropriate."
    )
    roles = ("User", "Assistant")
    sep = "<|end_of_turn|>\n"
    # stop_token_ids=[32000, 32001],  # "<|end_of_turn|>"
    stop_str = "User"
    system_message = system_template.format(system_message=system_message)
    _messages = _map_roles(messages, dict(zip(roles, roles)))
    _messages.append((roles[1], None))
    _prompt = _format_add_colon_space_single(system_message, _messages, sep)
    return ChatFormatterResponse(prompt=_prompt, stop=stop_str)


@register_chat_format("mistrallite")
def format_mistrallite(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="<|prompter|>", assistant="</s>\n<|assistant|>")
    _sep = " "
    system_template = """<|system|>{system_message}</s>"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("zephyr")
def format_zephyr(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = """<|system|>
{system_message}"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(user="<|user|>\n", assistant="<|assistant|>\n")
    _sep = "</s>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatml(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


@register_chat_format("pygmalion")
def format_pygmalion(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = """<|system|>{system_message}"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(user="<|user|>", assistant="<|model|>")
    _sep = "\n"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatml(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


@register_chat_format("chatml")
def format_chatml(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = """<|im_start|>system
{system_message}"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(user="<|im_start|>user", assistant="<|im_start|>assistant")
    _sep = "<|im_end|>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatml(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


@register_chat_format("mistral-instruct")
def format_mistral_instruct(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    eos = "</s>"
    stop = eos
    prompt = ""
    for message in messages:
        if (
            message["role"] == "user"
            and message["content"] is not None
            and isinstance(message["content"], str)
        ):
            prompt += "[INST] " + message["content"]
        elif message["role"] == "assistant" and message["content"] is not None:
            prompt += " [/INST]" + message["content"] + eos
    prompt += " [/INST]"
    return ChatFormatterResponse(prompt=prompt, stop=stop)


@register_chat_format("chatglm3")
def format_chatglm3(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = """<|system|>
{system_message}"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(user="<|user|>", assistant="<|assistant|>")
    _sep = "</s>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatglm3(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


@register_chat_format("openchat")
def format_openchat(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = "{system_message}<|end_of_turn|>"
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(
        user="GPT4 Correct User: ", assistant="<|end_of_turn|>GPT4 Correct Assistant: "
    )
    _sep = "<|end_of_turn|>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatml(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


# Chat format for Saiga models, see more details and available models:
# https://huggingface.co/collections/IlyaGusev/saiga2-saigamistral-6505d4ccc3d1e53166b636cd
@register_chat_format("saiga")
def format_saiga(
    messages: list[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _message_template = "<s>{role}\n{content}</s>"
    _roles = dict(user="user", bot="bot", system="system")
    _messages = _map_roles(messages, _roles)

    _prompt = ""
    for role, content in _messages:
        if content:
            _prompt += _message_template.format(role=role, content=content)
        else:
            _prompt += f"<s>{role}\n"
    # Response template
    _prompt += "<s>bot"
    return ChatFormatterResponse(prompt=_prompt.strip())


# Chat format for Google's Gemma models, see more details and available models:
# https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b
@register_chat_format("gemma")
def format_gemma(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_message = _get_system_message(messages)
    if system_message != "":
        logger.debug(
            "`role='system'` messages are not allowed on Google's Gemma models."
        )
    _roles = dict(user="<start_of_turn>user\n", assistant="<start_of_turn>model\n")
    _sep = "<end_of_turn>\n"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message="", messages=_messages, sep=_sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


# Tricky chat formats that require custom chat handlers


@register_chat_completion_handler("functionary")
def functionary_chat_handler(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    **kwargs,  # type: ignore
) -> Union[llama_types.ChatCompletion, Iterator[llama_types.ChatCompletionChunk]]:
    SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""

    def generate_type_definition(
        param: Dict[str, llama_types.JsonType], indent_level: int, shared_defs
    ) -> str:
        indent = "  " * indent_level
        if "$ref" in param:
            # Reference to a shared definition
            ref_name = param["$ref"].split("/")[
                -1
            ]  # Extract the type name from the reference
            return ref_name
        elif param.get("type") == "array":
            items = param.get("items", {})
            item_type = generate_type_definition(items, indent_level + 1, shared_defs)
            return f"Array<{item_type}>"
        elif param.get("type") == "object":
            properties = param.get("properties", {})
            nested_schema = "{\n"
            for nested_param_name, nested_param in properties.items():
                nested_param_type = generate_type_definition(
                    nested_param, indent_level + 1, shared_defs
                )
                nested_schema += (
                    f"{indent}  {nested_param_name}: {nested_param_type},\n"
                )
            nested_schema += indent + "}"
            return nested_schema
        elif "enum" in param:
            # Enum type
            return " | ".join([f'"{enum_value}"' for enum_value in param["enum"]])
        else:
            # Simple type
            return param.get("type", "any")

    def generate_shared_definitions(shared_defs, indent_level: int) -> str:
        indent = "  " * indent_level
        shared_definitions = ""
        for def_name, def_properties in shared_defs.items():
            shared_definitions += f"{indent}type {def_name} = "
            if def_properties.get("type") == "object":
                shared_definitions += generate_type_definition(
                    def_properties, indent_level, shared_defs
                )
            elif "enum" in def_properties:
                # Enum type
                shared_definitions += " | ".join(
                    [f'"{enum_value}"' for enum_value in def_properties["enum"]]
                )
            shared_definitions += ";\n"
        return shared_definitions

    def generate_schema_from_functions(functions, namespace="functions") -> str:
        schema = (
            "// Supported function definitions that should be called when necessary.\n"
        )
        schema += f"namespace {namespace} {{\n\n"

        # Generate shared definitions
        shared_definitions = {}
        for function in functions:
            parameters = function.get("parameters", {})
            shared_definitions.update(parameters.get("$defs", {}))

        schema += generate_shared_definitions(shared_definitions, 1)

        for function in functions:
            function_name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            required_params = parameters.get("required", [])

            schema += f"  // {description}\n"
            schema += f"  type {function_name} = (_: {{\n"

            for param_name, param in parameters.get("properties", {}).items():
                param_description = param.get("description", "")
                param_type = generate_type_definition(param, 2, shared_definitions)
                optional_indicator = "" if param_name in required_params else "?"
                schema += f"    // {param_description}\n"
                schema += f"    {param_name}{optional_indicator}: {param_type},\n"
            schema += "  }) => any;\n\n"

        schema += "}} // namespace {}\n".format(namespace)
        return schema

    def prepare_messages_for_inference(
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunctions]] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    ):
        all_messages: List[llama_types.ChatCompletionRequestMessage] = []
        if functions is not None:
            all_messages.append(
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system", content=generate_schema_from_functions(functions)
                )
            )

        if tools is not None:
            all_messages.append(
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system",
                    content=generate_schema_from_functions(
                        [
                            tool["function"]
                            for tool in tools
                            if tool["type"] == "function"
                        ]
                    ),
                )
            )

        all_messages.append(
            llama_types.ChatCompletionRequestSystemMessage(
                role="system", content=SYSTEM_MESSAGE
            )
        )

        for message in messages:
            # Function call responses
            if message["role"] == "function" and "name" in message:
                message["name"] = f"functions.{message['name']}"
            # Function call requests by assistant
            if "function_call" in message:
                message["function_call"][
                    "name"
                ] = f"functions.{message['function_call']['name']}"
            all_messages.append(message)

        all_messages.append(
            llama_types.ChatCompletionRequestAssistantMessage(
                role="assistant", content=None
            )
        )

        def message_to_str(msg: llama_types.ChatCompletionRequestMessage):
            if msg["role"] == "system":
                return f"system:\n{msg['content']}\n"

            elif msg["role"] == "function" and "name" in msg:
                return f"function name={msg['name']}:\n{msg['content']}\n"
            elif msg["role"] == "function" and "function_call" in msg:
                return f"function name={msg['function_call']['name']}:\n{msg['function_call']['arguments']}\n"
            elif msg["role"] == "tool":
                if msg["content"] is not None:
                    return f"function name={msg['tool_call_id']}:\n{msg['content']}\n"
                else:
                    return f"function name={msg['tool_call_id']}\n"
            elif msg["role"] == "user":
                if msg["content"] is None:
                    return "user:\n</s></s>\n"
                else:
                    return f"user:\n</s>{msg['content']}</s>\n"
            elif msg["role"] == "assistant":
                if msg["content"] is not None and "function_call" in msg:
                    return f"assistant:\n{msg['content']}\nassistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>\n"
                elif "function_call" in msg:
                    return f"assistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>\n"
                elif "tool_calls" in msg and len(msg["tool_calls"]) > 0:
                    for tool_call in msg[
                        "tool_calls"
                    ]:  # NOTE: probably doesn't work with the functionary model
                        return f"assistant to={tool_call['id']}:\n{tool_call['function']['arguments']}</s>\n"
                elif msg["content"] is None:
                    return "assistant"
                else:
                    return f"assistant:\n{msg['content']}\n"
            else:
                raise ValueError(f"Unsupported role: {msg['role']}")

        return "".join([message_to_str(msg) for msg in all_messages])

    if tools is not None:
        functions = [tool["function"] for tool in tools if tool["type"] == "function"]

    if tool_choice is not None:
        function_call = (
            tool_choice if isinstance(tool_choice, str) else tool_choice["function"]
        )

    prompt = prepare_messages_for_inference(messages, functions, tools)

    if function_call is None and (functions is None or len(functions) == 0):
        completion_or_completion_chunks = llama.create_completion(
            prompt=prompt + ":\n",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=["user:", "</s>"],
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat(completion_or_completion_chunks, stream=stream)  # type: ignore

    if function_call is None or (
        isinstance(function_call, str) and function_call == "auto"
    ):
        stop = "\n"
        completion: llama_types.Completion = llama.create_completion(
            prompt=prompt, stop=stop, stream=False
        )  # type: ignore
        completion_text = completion["choices"][0]["text"]
        # strip " to=functions." and ending ":"
        function_call = completion_text.split(".")[-1][:-1]
        new_prompt = prompt + completion_text + stop
    elif isinstance(function_call, str) and function_call != "none":
        new_prompt = prompt + ":\n"
    elif isinstance(function_call, dict):
        new_prompt = prompt + f" to=functions.{function_call['name']}:\n"
        function_call = function_call["name"]
    else:
        new_prompt = prompt + ":\n"

    function_body = None
    for function in functions or []:
        if function["name"] == function_call:
            function_body = function["parameters"]
            break
    for tool in tools or []:
        if tool["type"] == "function" and tool["function"]["name"] == function_call:
            function_body = tool["function"]["parameters"]
            break

    if function_body is not None:
        try:
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar_text = llama_grammar.json_schema_to_gbnf(
                    json.dumps(function_body)
                )
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.json_schema_to_gbnf(json.dumps(function_body)),
                    verbose=llama.verbose,
                )
                print(grammar_text)
        except Exception as e:
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF,
                    verbose=llama.verbose,
                )
    else:
        with suppress_stdout_stderr(disable=llama.verbose):
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )

    completion: llama_types.Completion = llama.create_completion(
        prompt=new_prompt,
        stop=["user:", "</s>"],
        stream=False,
        grammar=grammar,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
    )  # type: ignore

    assert "usage" in completion
    assert isinstance(function_call, str)
    assert stream is False  # TODO: support stream mode

    if llama.verbose:
        print(new_prompt)
        print(completion["choices"][0]["text"])

    # TODO: support stream mode
    return llama_types.CreateChatCompletionResponse(
        id="chat" + completion["id"],
        object="chat.completion",
        created=completion["created"],
        model=completion["model"],
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_call,
                        "arguments": completion["choices"][0]["text"],
                    },
                    "tool_calls": [
                        {
                            "id": function_call,
                            "type": "function",
                            "function": {
                                "name": function_call,
                                "arguments": completion["choices"][0]["text"],
                            },
                        }
                    ],
                },
                "logprobs": _convert_text_completion_logprobs_to_chat(completion["choices"][0]["logprobs"]),
                "finish_reason": "tool_calls",
            }
        ],
        usage=completion["usage"],
    )


@register_chat_completion_handler("functionary-v1")
@register_chat_completion_handler("functionary-v2")
def functionary_v1_v2_chat_handler(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    **kwargs,  # type: ignore
) -> Union[llama_types.ChatCompletion, Iterator[llama_types.ChatCompletionChunk]]:
    SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""

    tokenizer = llama.tokenizer_
    assert hasattr(
        tokenizer, "hf_tokenizer"
    ), "Please provide a valid hf_tokenizer_path from https://huggingface.co/meetkai when initializing the Llama class"
    from transformers import AutoTokenizer

    if "<|START_OF_FUNCTION_CALL|>" in tokenizer.hf_tokenizer.additional_special_tokens:
        version = "v1"
        END_SYSTEM_TOKEN = "<|END_OF_SYSTEM|>"
        END_USER_TOKEN = "<|END_OF_USER|>"
        END_ASSISTANT_TOKEN = "<|END_OF_ASSISTANT|>"
        END_FUNCTION_RESULT_TOKEN = "<|END_OF_FUNCTION_RESULT|>"
        START_FUNCTION_CALL_TOKEN = "<|START_OF_FUNCTION_CALL|>"
        END_FUNCTION_CALL_TOKEN = "<|END_OF_FUNCTION_CALL|>"
    else:
        version = "v2"
        RECIPIENT_TOKEN = "<|recipient|>"
        FROM_TOKEN = "<|from|>"
        STOP_TOKEN = "<|stop|>"
        CONTENT_TOKEN = "<|content|>"

    def generate_type_definition(
        param: Dict[str, llama_types.JsonType], indent_level: int, shared_defs
    ) -> str:
        indent = "  " * indent_level
        if "$ref" in param:
            # Reference to a shared definition
            ref_name = param["$ref"].split("/")[
                -1
            ]  # Extract the type name from the reference
            return ref_name
        elif param.get("type") == "array":
            items = param.get("items", {})
            item_type = generate_type_definition(items, indent_level + 1, shared_defs)
            return f"Array<{item_type}>"
        elif param.get("type") == "object":
            properties = param.get("properties", {})
            nested_schema = "{\n"
            for nested_param_name, nested_param in properties.items():
                nested_param_type = generate_type_definition(
                    nested_param, indent_level + 1, shared_defs
                )
                nested_schema += (
                    f"{indent}  {nested_param_name}: {nested_param_type},\n"
                )
            nested_schema += indent + "}"
            return nested_schema
        elif "enum" in param:
            # Enum type
            return " | ".join([f'"{enum_value}"' for enum_value in param["enum"]])
        else:
            # Simple type
            return param.get("type", "any")

    def generate_shared_definitions(shared_defs, indent_level: int) -> str:
        indent = "  " * indent_level
        shared_definitions = ""
        for def_name, def_properties in shared_defs.items():
            shared_definitions += f"{indent}type {def_name} = "
            if def_properties.get("type") == "object":
                shared_definitions += generate_type_definition(
                    def_properties, indent_level, shared_defs
                )
            elif "enum" in def_properties:
                # Enum type
                shared_definitions += " | ".join(
                    [f'"{enum_value}"' for enum_value in def_properties["enum"]]
                )
            shared_definitions += ";\n"
        return shared_definitions

    def generate_schema_from_functions(functions, namespace="functions") -> str:
        schema = (
            "// Supported function definitions that should be called when necessary.\n"
        )
        schema += f"namespace {namespace} {{\n\n"

        # Generate shared definitions
        shared_definitions = {}
        for function in functions:
            parameters = function.get("parameters", {})
            shared_definitions.update(parameters.get("$defs", {}))

        schema += generate_shared_definitions(shared_definitions, 1)

        for function in functions:
            function_name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            required_params = parameters.get("required", [])

            schema += f"// {description}\n"
            schema += f"type {function_name} = (_: {{\n"

            for param_name, param in parameters.get("properties", {}).items():
                param_description = param.get("description", "")
                param_type = generate_type_definition(param, 2, shared_definitions)
                optional_indicator = "" if param_name in required_params else "?"
                schema += f"// {param_description}\n"
                schema += f"{param_name}{optional_indicator}: {param_type},\n"
            schema += "}) => any;\n\n"

        schema += "}} // namespace {}".format(namespace)
        return schema

    def prepare_messages_for_inference(
        messages: List[llama_types.ChatCompletionRequestMessage],
        tokenizer: AutoTokenizer,
        version: Literal["v1", "v2"],
        functions: Optional[List[llama_types.ChatCompletionFunctions]] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Union[Dict, str] = "auto",
    ):
        all_messages: List[llama_types.ChatCompletionRequestMessage] = []
        if tool_choice == "none":
            all_messages.append(
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system", content=generate_schema_from_functions([])
                )
            )
        else:
            if functions is not None:
                all_messages.append(
                    llama_types.ChatCompletionRequestSystemMessage(
                        role="system", content=generate_schema_from_functions(functions)
                    )
                )
            elif tools is not None and tool_choice != "none":
                all_messages.append(
                    llama_types.ChatCompletionRequestSystemMessage(
                        role="system",
                        content=generate_schema_from_functions(
                            [
                                tool["function"]
                                for tool in tools
                                if tool["type"] == "function"
                            ]
                        ),
                    )
                )

        all_messages.append(
            llama_types.ChatCompletionRequestSystemMessage(
                role="system", content=SYSTEM_MESSAGE
            )
        )

        for message in messages:
            # Function call responses
            if message["role"] == "function" and "name" in message:
                message["name"] = f"functions.{message['name']}"
            # Function call requests by assistant
            if "function_call" in message:
                message["function_call"][
                    "name"
                ] = f"functions.{message['function_call']['name']}"
            all_messages.append(message)

        if version == "v1":
            suffix = "assistant:\n"
        else:
            suffix = "<|from|>assistant\n<|recipient|>"

        return (
            tokenizer.hf_tokenizer.apply_chat_template(all_messages, tokenize=False)
            + suffix
        )

    if tools is not None:
        functions = [tool["function"] for tool in tools if tool["type"] == "function"]

    if tool_choice is not None:
        function_call = (
            tool_choice if isinstance(tool_choice, str) else tool_choice["function"]
        )
    elif function_call is not None:
        pass
    else:
        function_call = "auto"

    prompt = prepare_messages_for_inference(
        messages, tokenizer, version, functions, tools, function_call
    )

    # If no tools/functions are provided
    if function_call == "none" or functions is None or len(functions) == 0:
        if version == "v1":
            stop = END_ASSISTANT_TOKEN
        else:
            stop = STOP_TOKEN
            prompt += "all\n<|content|>"

        completion_or_completion_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        if stream is False:
            completion_or_completion_chunks["choices"][0]["text"] = (
                completion_or_completion_chunks["choices"][0]["text"].lstrip()
            )
        return _convert_completion_to_chat(completion_or_completion_chunks, stream=stream)  # type: ignore

    def get_grammar(function_call):
        function_body = None
        for function in functions or []:
            if function["name"] == function_call:
                function_body = function["parameters"]
                break
        for tool in tools or []:
            if tool["type"] == "function" and tool["function"]["name"] == function_call:
                function_body = tool["function"]["parameters"]
                break

        try:
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar_text = llama_grammar.json_schema_to_gbnf(
                    json.dumps(function_body)
                )
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.json_schema_to_gbnf(json.dumps(function_body))
                )
                print(grammar_text)
        except Exception as e:
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
            with suppress_stdout_stderr(disable=llama.verbose):
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        return grammar

    def create_completion(prompt, stop, grammar):
        completion = cast(
            llama_types.Completion,
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
        )

        return completion

    content = ""
    function_calls, function_bodies = [], []
    completion_tokens = 0

    def generate_streaming(tools, functions, function_call, prompt):
        assert version == "v2", "Streaming for v1 is not supported"

        chunk_id, chunk_created = None, None

        # If tool_choice/function_call is provided
        if isinstance(function_call, dict):
            prompt += f"{function_call['name']}\n{CONTENT_TOKEN}"
            grammar = get_grammar(function_call["name"])
            stops = [STOP_TOKEN, FROM_TOKEN]
            tool_id = "".join(
                [random.choice(string.ascii_letters + string.digits) for _ in range(24)]
            )
            completion = create_completion(prompt=prompt, stop=stops, grammar=grammar)
            completion_text = ""
            first = True
            for chunk in completion:
                # Yield the tool/function name first
                if first:
                    if tools is not None:
                        func_call_dict = {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_" + tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": function_call["name"],
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    else:
                        func_call_dict = {
                            "function_call": {
                                "name": function_call["name"],
                                "arguments": "",
                            }
                        }
                    yield llama_types.CreateChatCompletionStreamResponse(
                        id="chat" + chunk["id"],
                        object="chat.completion.chunk",
                        created=chunk["created"],
                        model=chunk["model"],
                        choices=[
                            {
                                "index": 0,
                                "logprobs": None,
                                "delta": {
                                    "role": None,
                                    "content": None,
                                    **func_call_dict,
                                },
                            }
                        ],
                    )
                    first = False
                if tools is not None:
                    func_call_dict = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_" + tool_id,
                                "type": "function",
                                "function": {
                                    "name": None,
                                    "arguments": chunk["choices"][0]["text"].rstrip(),
                                },
                            }
                        ]
                    }
                else:
                    func_call_dict = {
                        "function_call": {
                            "name": None,
                            "arguments": chunk["choices"][0]["text"].rstrip(),
                        }
                    }
                if len(chunk["choices"][0]["text"].rstrip()) > 0:
                    yield llama_types.CreateChatCompletionStreamResponse(
                        id="chat" + chunk["id"],
                        object="chat.completion.chunk",
                        created=chunk["created"],
                        model=chunk["model"],
                        choices=[
                            {
                                "index": 0,
                                "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                "delta": {
                                    "role": None,
                                    "content": None,
                                    **func_call_dict,
                                },
                            }
                        ],
                    )
            # Yield tool_call/function_call stop message
            yield llama_types.CreateChatCompletionStreamResponse(
                id="chat" + chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                model=chunk["model"],
                choices=[
                    {
                        "index": 0,
                        "finish_reason": (
                            "tool_calls" if tools is not None else "function_call"
                        ),
                        "logprobs": None,
                        "delta": {
                            "role": None,
                            "content": None,
                            "function_call": None,
                            "tool_calls": None,
                        },
                    }
                ],
            )
        # If "auto" or no tool_choice/function_call
        elif isinstance(function_call, str) and function_call == "auto":
            tool_index = 0
            while True:
                # Generate function name first
                grammar = None
                stops = CONTENT_TOKEN
                completion = create_completion(
                    prompt=prompt, stop=stops, grammar=grammar
                )
                completion_text = ""
                for chunk in completion:
                    completion_text += chunk["choices"][0]["text"]
                if chunk_id is None:
                    chunk_id = chunk["id"]
                if chunk_created is None:
                    chunk_created = chunk["created"]
                function_name = completion_text.strip()
                if function_name == "all":
                    prompt += "all\n<|content|>"
                    # Yield the first empty message for content
                    yield llama_types.CreateChatCompletionStreamResponse(
                        id="chat" + chunk_id,
                        model=chunk["model"],
                        created=chunk_created,
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    )
                else:
                    prompt += f"{function_name}\n<|content|>"
                    grammar = get_grammar(function_name)
                    tool_id = "".join(
                        [
                            random.choice(string.ascii_letters + string.digits)
                            for _ in range(24)
                        ]
                    )
                    if tools is not None:
                        func_call_dict = {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "id": "call_" + tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    else:
                        func_call_dict = {
                            "function_call": {"name": function_name, "arguments": ""}
                        }
                    # Stream function name
                    yield llama_types.CreateChatCompletionStreamResponse(
                        id="chat" + chunk_id,
                        object="chat.completion.chunk",
                        created=chunk_created,
                        model=chunk["model"],
                        choices=[
                            {
                                "index": 0,
                                "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    **func_call_dict,
                                },
                            }
                        ],
                    )
                # Generate content
                stops = [RECIPIENT_TOKEN, STOP_TOKEN]
                completion = create_completion(
                    prompt=prompt, stop=stops, grammar=grammar
                )
                if function_name == "all":
                    completion_text = ""
                    stop_sequence, buffer, is_end = (
                        "\n<|from|>assistant\n<|recipient|>",
                        [],
                        False,
                    )
                    for i, chunk in enumerate(completion):
                        completion_text += chunk["choices"][0]["text"]
                        if is_end:
                            buffer.append(chunk["choices"][0]["text"].strip(" "))
                            if stop_sequence.startswith("".join(buffer)):
                                continue
                            else:
                                buffer.pop()
                                while len(buffer) > 0:
                                    yield llama_types.CreateChatCompletionStreamResponse(
                                        id="chat" + chunk_id,
                                        object="chat.completion.chunk",
                                        created=chunk_created,
                                        model=chunk["model"],
                                        choices=[
                                            {
                                                "index": 0,
                                                "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                                "delta": {
                                                    "role": "assistant",
                                                    "content": buffer.pop(0),
                                                },
                                            }
                                        ],
                                    )
                                is_end = False
                        elif chunk["choices"][0]["text"] == "\n":
                            is_end = True
                            buffer.append(chunk["choices"][0]["text"].strip(" "))
                            continue

                        if len(buffer) == 0 and len(chunk["choices"][0]["text"]) > 0:
                            yield llama_types.CreateChatCompletionStreamResponse(
                                id="chat" + chunk_id,
                                object="chat.completion.chunk",
                                created=chunk_created,
                                model=chunk["model"],
                                choices=[
                                    {
                                        "index": 0,
                                        "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                        "delta": {
                                            "role": "assistant",
                                            "content": (
                                                chunk["choices"][0]["text"]
                                                if i > 0
                                                else chunk["choices"][0][
                                                    "text"
                                                ].lstrip()
                                            ),
                                        },
                                    }
                                ],
                            )
                    # Check whether the model wants to generate another turn
                    if (
                        "<|from|> assistant" in completion_text
                        or "<|from|>assistant" in completion_text
                    ):
                        if completion_text.endswith("\n<|from|>assistant\n"):
                            cleaned_completion_text = completion_text[
                                : -len("\n<|from|>assistant\n")
                            ].strip()
                        elif completion_text.endswith("\n<|from|> assistant\n"):
                            cleaned_completion_text = completion_text[
                                : -len("\n<|from|> assistant\n")
                            ].strip()
                        else:
                            cleaned_completion_text = completion_text.strip()
                        prompt += f"{cleaned_completion_text}\n<|from|>assistant\n<|recipient|>"
                    else:
                        # Yield stop message
                        yield llama_types.CreateChatCompletionStreamResponse(
                            id="chat" + chunk_id,
                            model=chunk["model"],
                            created=chunk_created,
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {},
                                    "logprobs": None,
                                    "finish_reason": "stop",
                                }
                            ],
                        )
                        break
                else:
                    # Check whether the model wants to generate another turn
                    completion_text = ""
                    for chunk in completion:
                        completion_text += chunk["choices"][0]["text"]
                        if len(chunk["choices"][0]["text"].rstrip()) > 0:
                            if tools is not None:
                                func_call_dict = {
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "id": "call_" + tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": None,
                                                "arguments": chunk["choices"][0][
                                                    "text"
                                                ].rstrip(),
                                            },
                                        }
                                    ]
                                }
                            else:
                                func_call_dict = {
                                    "function_call": {
                                        "name": None,
                                        "arguments": chunk["choices"][0][
                                            "text"
                                        ].rstrip(),
                                    }
                                }
                            yield llama_types.CreateChatCompletionStreamResponse(
                                id="chat" + chunk_id,
                                object="chat.completion.chunk",
                                created=chunk_created,
                                model=chunk["model"],
                                choices=[
                                    {
                                        "index": 0,
                                        "logprobs": _convert_text_completion_logprobs_to_chat(chunk["choices"][0]["logprobs"]),
                                        "delta": {
                                            "role": None,
                                            "content": None,
                                            **func_call_dict,
                                        },
                                    }
                                ],
                            )
                    prompt += completion_text.strip()
                    grammar = None
                    completion = create_completion(
                        prompt=prompt, stop=stops, grammar=grammar
                    )
                    completion_text += "".join(
                        [chunk["choices"][0]["text"] for chunk in completion]
                    )
                    if (
                        "<|from|> assistant" in completion_text
                        or "<|from|>assistant" in completion_text
                    ) and tools is not None:
                        prompt += "\n<|from|>assistant\n<|recipient|>"
                        tool_index += 1
                    else:
                        # Yield tool_call/function_call stop message
                        yield llama_types.CreateChatCompletionStreamResponse(
                            id="chat" + chunk_id,
                            object="chat.completion.chunk",
                            created=chunk_created,
                            model=chunk["model"],
                            choices=[
                                {
                                    "index": 0,
                                    "finish_reason": (
                                        "tool_calls"
                                        if tools is not None
                                        else "function_call"
                                    ),
                                    "logprobs": None,
                                    "delta": {
                                        "role": None,
                                        "content": None,
                                        "function_call": None,
                                        "tool_calls": None,
                                    },
                                }
                            ],
                        )
                        break

    if stream is not False:
        return generate_streaming(
            tools=tools, functions=functions, function_call=function_call, prompt=prompt
        )
    else:
        if version == "v1":
            # If no or "auto" tool_choice/function_call
            if isinstance(function_call, str) and function_call == "auto":
                stops = ["\n", END_ASSISTANT_TOKEN]
            # If tool_choice/function_call is provided
            elif isinstance(function_call, dict):
                prompt += f"{START_FUNCTION_CALL_TOKEN}{function_call['name']}:\n"
                stops = END_FUNCTION_CALL_TOKEN
                function_call = function_call["name"]
                function_calls.append(function_call)
                grammar = get_grammar(function_call)
            else:
                prompt = prompt
                stops = ["\n", END_ASSISTANT_TOKEN]

            completion = create_completion(prompt=prompt, stop=stops, grammar=grammar)
            completion_text = completion["choices"][0]["text"]
            completion_tokens += completion["usage"]["completion_tokens"]

            # If the generation does not involve a function call
            if (
                START_FUNCTION_CALL_TOKEN not in prompt
                and START_FUNCTION_CALL_TOKEN not in completion_text
            ):
                completion["usage"]["completion_tokens"] = completion_tokens
                return _convert_completion_to_chat(completion, stream=stream)  # type: ignore
            # If the generation involves a function call in completion, generate the parameters
            elif (
                START_FUNCTION_CALL_TOKEN not in prompt
                and START_FUNCTION_CALL_TOKEN in completion_text
            ):
                prompt += (
                    completion_text.replace(
                        f"{START_FUNCTION_CALL_TOKEN} ", START_FUNCTION_CALL_TOKEN
                    )
                    + "\n"
                )
                function_calls.append(
                    completion_text.split(START_FUNCTION_CALL_TOKEN)[-1][:-1].strip()
                )
                grammar = get_grammar(function_calls[-1])
                completion = create_completion(
                    prompt=prompt, stop=END_FUNCTION_CALL_TOKEN, grammar=grammar
                )
                completion_tokens += completion["usage"]["completion_tokens"]
                function_bodies.append(completion["choices"][0]["text"].strip())
            # If the prompt involves a function call, just append generated parameters to function_bodies
            else:
                function_bodies.append(completion_text.strip())
        else:
            # If tool_choice/function_call is provided
            if isinstance(function_call, dict):
                prompt += f"{function_call['name']}\n{CONTENT_TOKEN}"
                function_call = function_call["name"]
                function_calls.append(function_call)
                grammar = get_grammar(function_call)
                stops = [STOP_TOKEN, FROM_TOKEN]
                completion = create_completion(
                    prompt=prompt, stop=stops, grammar=grammar
                )
                completion_text = completion["choices"][0]["text"]
                completion_tokens += completion["usage"]["completion_tokens"]
                function_bodies.append(completion_text.strip())
            # If "auto" or no tool_choice/function_call
            elif isinstance(function_call, str) and function_call == "auto":
                while True:
                    # Generate function name first
                    grammar = None
                    stops = CONTENT_TOKEN
                    completion = create_completion(
                        prompt=prompt, stop=stops, grammar=grammar
                    )
                    completion_text = completion["choices"][0]["text"]
                    completion_tokens += completion["usage"]["completion_tokens"]
                    function_name = completion_text.strip()
                    if function_name == "all":
                        prompt += "all\n<|content|>"
                    else:
                        function_call = completion_text.strip()
                        prompt += f"{function_call}\n<|content|>"
                        function_calls.append(function_call)
                        grammar = get_grammar(function_call)
                    # Generate content
                    stops = [RECIPIENT_TOKEN, STOP_TOKEN]
                    completion = create_completion(
                        prompt=prompt, stop=stops, grammar=grammar
                    )
                    completion_text = completion["choices"][0]["text"]
                    completion_tokens += completion["usage"]["completion_tokens"]
                    if function_name == "all":
                        if completion_text.endswith("\n<|from|>assistant\n"):
                            content += completion_text[: -len("\n<|from|>assistant\n")]
                        if completion_text.endswith("\n<|from|> assistant\n"):
                            content += completion_text[-len("\n<|from|> assistant\n")]
                        else:
                            content += completion_text
                        content = content.lstrip()
                        # Check whether the model wants to generate another turn
                        if (
                            "<|from|> assistant" in completion_text
                            or "<|from|>assistant" in completion_text
                        ):
                            if completion_text.endswith("\n<|from|>assistant\n"):
                                cleaned_completion_text = completion_text[
                                    : -len("\n<|from|>assistant\n")
                                ].strip()
                            elif completion_text.endswith("\n<|from|> assistant\n"):
                                cleaned_completion_text = completion_text[
                                    -len("\n<|from|> assistant\n")
                                ].strip()
                            else:
                                cleaned_completion_text = completion_text.strip()
                            prompt += f"{cleaned_completion_text}\n<|from|>assistant\n<|recipient|>"
                        else:
                            break
                    else:
                        function_bodies.append(completion_text.strip())
                        # Check whether the model wants to generate another turn
                        prompt += completion_text.strip()
                        grammar = None
                        completion = create_completion(
                            prompt=prompt, stop=stops, grammar=grammar
                        )
                        completion_tokens += completion["usage"]["completion_tokens"]
                        if (
                            "<|from|> assistant" in completion["choices"][0]["text"]
                            or "<|from|>assistant" in completion["choices"][0]["text"]
                        ):
                            prompt += "\n<|from|>assistant\n<|recipient|>"
                        else:
                            break

        assert "usage" in completion
        assert len(function_calls) == len(function_bodies)

        tool_calls: List[llama_types.ChatCompletionMessageToolCall] = []
        for function_call, function_body in zip(function_calls, function_bodies):
            tool_calls.append(
                {
                    "id": "call_"
                    + "".join(
                        [
                            random.choice(string.ascii_letters + string.digits)
                            for _ in range(24)
                        ]
                    ),
                    "type": "function",
                    "function": {
                        "name": function_call,
                        "arguments": function_body,
                    },
                }
            )

        # TODO: support stream mode
        function_call_dict: Union[
            Dict[str, str],
            Dict[
                Literal["function_call"],
                llama_types.ChatCompletionRequestAssistantMessageFunctionCall,
            ],
        ] = {}
        if len(tool_calls) > 0:
            if tools is not None:
                function_call_dict["tool_calls"] = tool_calls
            else:
                function_call_dict["function_call"] = {
                    "name": tool_calls[0]["function"]["name"],
                    "arguments": tool_calls[0]["function"]["arguments"],
                }
        completion["usage"]["completion_tokens"] = completion_tokens
        return llama_types.CreateChatCompletionResponse(
            id="chat" + completion["id"],
            object="chat.completion",
            created=completion["created"],
            model=completion["model"],
            choices=[
                {
                    "index": 0,
                    "logprobs": _convert_text_completion_logprobs_to_chat(completion["choices"][0]["logprobs"]),
                    "message": {
                        "role": "assistant",
                        "content": None if content == "" else content,
                        **function_call_dict,
                    },
                    "finish_reason": "tool_calls" if len(tool_calls) > 0 else "stop",
                }
            ],
            usage=completion["usage"],
        )


class Llava15ChatHandler:
    DEFAULT_SYSTEM_MESSAGE: Optional[str] = (
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    )

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "{% if message.content is string %}"
        "\nUSER: {{ message.content }}"
        "{% endif %}"
        "{% if message.content is iterable %}"
        "\nUSER: "
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{% endif %}"
        "{% if message.role == 'assistant' and message.content is not none %}"
        "\nASSISTANT: {{ message.content }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "\nASSISTANT: "
        "{% endif %}"
    )

    def __init__(self, clip_model_path: str, verbose: bool = True):
        import llama_cpp.mtmd_cpp as mtmd_cpp

        self.clip_model_path = clip_model_path
        self.verbose = verbose
        self._mtmd_cpp = mtmd_cpp
        self._exit_stack = ExitStack()
        self.mtmd_ctx: Optional[mtmd_cpp.mtmd_context_p] = None

        if not os.path.exists(clip_model_path):
            raise ValueError(f"Clip model path does not exist: {clip_model_path}")

    def _init_mtmd_context(self, llama_model: llama.Llama):
        """Initialize mtmd context with the llama model."""
        if self.mtmd_ctx is not None:
            return  # Already initialized

        with suppress_stdout_stderr(disable=self.verbose):
            # Get default parameters
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = True # TODO: Make this configurable
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.verbosity = 2 if self.verbose else 0  # GGML_LOG_LEVEL_INFO = 2

            # Initialize mtmd context
            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(),
                llama_model.model,
                ctx_params
            )

            if self.mtmd_ctx is None:
                raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")

            # Check if vision is supported
            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")

            def mtmd_free():
                with suppress_stdout_stderr(disable=self.verbose):
                    if self.mtmd_ctx is not None:
                        self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                        self.mtmd_ctx = None

            self._exit_stack.callback(mtmd_free)

    def load_image(self, image_url: str) -> bytes:
        return self._load_image(image_url)

    def _create_bitmap_from_bytes(self, image_bytes: bytes):
        """Create mtmd_bitmap from image bytes."""
        if self.mtmd_ctx is None:
            raise ValueError("mtmd context not initialized")

        with suppress_stdout_stderr(disable=self.verbose):
            # Create bitmap from buffer using helper function
            bitmap = self._mtmd_cpp.mtmd_helper_bitmap_init_from_buf(
                self.mtmd_ctx,
                (ctypes.c_uint8 * len(image_bytes)).from_buffer(bytearray(image_bytes)),
                len(image_bytes)
            )
            
            if bitmap is None:
                raise ValueError("Failed to create bitmap from image bytes")
            
            return bitmap

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        # Initialize mtmd context
        self._init_mtmd_context(llama)
        assert self.mtmd_ctx is not None

        system_prompt = _get_system_message(messages)
        if system_prompt == "" and self.DEFAULT_SYSTEM_MESSAGE is not None:
            messages = [
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system", content=self.DEFAULT_SYSTEM_MESSAGE
                )
            ] + messages

        image_urls = self.get_image_urls(messages)
        template = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.CHAT_FORMAT)
        
        # Get the default media marker
        media_marker = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')
        
        # Replace image URLs with media markers in the template
        text = template.render(
            messages=messages,
            add_generation_prompt=True,
            eos_token=llama.detokenize([llama.token_eos()]),
            bos_token=llama.detokenize([llama.token_bos()]),
        )
        
        # Replace image URLs in text with media markers
        for image_url in image_urls:
            text = text.replace(image_url, media_marker)

        if self.verbose:
            print(text, file=sys.stderr)

        # Create bitmaps from images
        bitmaps = []
        bitmap_cleanup = []
        try:
            for image_url in image_urls:
                image_bytes = self.load_image(image_url)
                bitmap = self._create_bitmap_from_bytes(image_bytes)
                bitmaps.append(bitmap)
                bitmap_cleanup.append(bitmap)

            # Create input text structure
            input_text = self._mtmd_cpp.mtmd_input_text()
            input_text.text = text.encode('utf-8')
            input_text.add_special = True
            input_text.parse_special = True

            # Create input chunks
            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to create input chunks")

            try:
                # Tokenize text and images together
                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps)
                )

                if result != 0:
                    raise ValueError(f"Failed to tokenize input: error code {result}")

                # Reset llama context
                llama.reset()
                llama._ctx.kv_cache_clear()

                # Process each chunk
                n_past = llama_cpp.llama_pos(0)
                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)
                
                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None:
                        continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                    
                    if chunk_type == self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        # Handle text chunk
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                            chunk, ctypes.byref(n_tokens_out)
                        )
                        
                        if tokens_ptr and n_tokens_out.value > 0:
                            # Convert ctypes array to Python list
                            tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                            
                            if llama.n_tokens + len(tokens) > llama.n_ctx():
                                raise ValueError(
                                    f"Prompt exceeds n_ctx: {llama.n_tokens + len(tokens)} > {llama.n_ctx()}"
                                )
                            llama.eval(tokens)
                    
                    elif chunk_type in [self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE, self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                        # Handle image/audio chunk using helper
                        chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                        
                        if llama.n_tokens + chunk_n_tokens > llama.n_ctx():
                            raise ValueError(
                                f"Prompt exceeds n_ctx: {llama.n_tokens + chunk_n_tokens} > {llama.n_ctx()}"
                            )
                        
                        new_n_past = llama_cpp.llama_pos(0)
                        result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                            self.mtmd_ctx,
                            llama._ctx.ctx,
                            chunk,
                            llama_cpp.llama_pos(llama.n_tokens),
                            llama_cpp.llama_seq_id(0),
                            llama.n_batch,
                            False,  # logits_last
                            ctypes.byref(new_n_past)
                        )
                        
                        if result != 0:
                            raise ValueError(f"Failed to evaluate chunk: error code {result}")
                        
                        # Update llama's token count
                        llama.n_tokens = new_n_past.value

                # Get prompt tokens to avoid a cache miss
                prompt = llama.input_ids[: llama.n_tokens].tolist()

            finally:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)

        finally:
            # Cleanup bitmaps
            for bitmap in bitmap_cleanup:
                self._mtmd_cpp.mtmd_bitmap_free(bitmap)

        # Handle response format and tools (same as before)
        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        # Convert legacy functions to tools
        if functions is not None:
            tools = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                # create grammar from json schema
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama.verbose
                )
            except Exception as e:
                if llama.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        
        if tool is not None:
            tool_name = tool["function"]["name"]
            return _convert_completion_to_chat_function(
                tool_name, completion_or_chunks, stream
            )
        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

    @staticmethod
    def _load_image(image_url: str) -> bytes:
        # TODO: Add Pillow support for other image formats beyond (jpg, png)
        if image_url.startswith("data:"):
            import base64
            image_bytes = base64.b64decode(image_url.split(",")[1])
            return image_bytes
        else:
            import urllib.request
            with urllib.request.urlopen(image_url) as f:
                image_bytes = f.read()
                return image_bytes

    @staticmethod
    def get_image_urls(messages: List[llama_types.ChatCompletionRequestMessage]):
        image_urls: List[str] = []
        for message in messages:
            if message["role"] == "user":
                if message["content"] is None:
                    continue
                for content in message["content"]:
                    if isinstance(content, dict) and "type" in content:
                        if content["type"] == "image_url":
                            if (
                                isinstance(content["image_url"], dict)
                                and "url" in content["image_url"]
                            ):
                                image_urls.append(content["image_url"]["url"])
                            else:
                                image_urls.append(content["image_url"])
        return image_urls

    @staticmethod
    def split_text_on_image_urls(text: str, image_urls: List[str]):
        """This method is no longer used in the new implementation."""
        def find_first(s: str, substrs: List[str]):
            for i, substr in enumerate(substrs):
                pos = s.find(substr)
                if pos != -1:
                    return pos, i
            return None, None

        split_text: List[Tuple[Literal["text", "image_url"], str]] = []
        remaining = text
        while remaining:
            # Find first image_url
            pos, i = find_first(remaining, image_urls)
            if pos is not None and i is not None:
                if pos > 0:
                    split_text.append(("text", remaining[:pos]))
                split_text.append(("image_url", image_urls[i]))
                remaining = remaining[pos + len(image_urls[i]) :]
            else:
                split_text.append(("text", remaining))
                remaining = ""
        return split_text

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str],
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        **kwargs: Any,
    ) -> "Llava15ChatHandler":
        import fnmatch
        from pathlib import Path

        try:
            from huggingface_hub import hf_hub_download, HfFileSystem  # type: ignore
            from huggingface_hub.utils import validate_repo_id  # type: ignore
        except ImportError:
            raise ImportError(
                "Llama.from_pretrained requires the huggingface-hub package. "
                "You can install it with `pip install huggingface-hub`."
            )

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id)  # type: ignore
        ]

        # split each file into repo_id, subfolder, filename
        file_list: List[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(files)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=cast(Union[str, Path, None], local_dir),
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cast(Union[str, Path, None], cache_dir),
        )

        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cast(Union[str, Path, None], cache_dir),
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, filename)

        return cls(
            clip_model_path=model_path,
            **kwargs,
        )


class ObsidianChatHandler(Llava15ChatHandler):
    # Prompt Format
    # The model followed ChatML format. However, with ### as the seperator

    # <|im_start|>user
    # What is this sign about?\n<image>
    # ###
    # <|im_start|>assistant
    # The sign is about bullying, and it is placed on a black background with a red background.
    # ###

    CHAT_FORMAT = (
        "{% for message in messages %}"
        # System message
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n"
        "{{ message.content }}\n"
        "###\n"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "###\n"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{{ message.content }}"
        "###\n"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class MoondreamChatHandler(Llava15ChatHandler):
    # Chat Format:
    # f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}\n\n"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}\n\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "Question: {{ content.text }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "Question: {{ message.content }}\n\n"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "Answer:{{ message.content }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "Answer:"
        "{% endif %}"
    )


class Llava16ChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "

    # Example prompt
    # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}\n"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "Answer:"
        "{% endif %}"
    )


class NanoLlavaChatHandler(Llava15ChatHandler):
    # Prompt Format
    # The model follow the ChatML standard, however, without \n at the end of <|im_end|>:

    # <|im_start|>system
    # Answer the question<|im_end|><|im_start|>user
    # <image>
    # What is the picture about?<|im_end|><|im_start|>assistant
    DEFAULT_SYSTEM_MESSAGE = "Answer the question"

    CHAT_FORMAT = (
        "{% for message in messages %}"
        # System message
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n"
        "{{ message.content }}"
        "<|im_end|>"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<|im_end|>"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{{ message.content }}"
        "<|im_end|>"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class Llama3VisionAlphaChatHandler(Llava15ChatHandler):
    # question = "<image>" + q

    # prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    DEFAULT_SYSTEM_MESSAGE = None

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "<|start_header_id|>"
        "{% if message.role == 'user' %}"
        "user<|end_header_id|>\n\n"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "assistant<|end_header_id|>\n\n"
        "{{ message.content }}"
        "{% endif %}"
        "<|eot_id|>"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    )


# alias
Llama3VisionAlpha = Llama3VisionAlphaChatHandler


class MiniCPMv26ChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is iterable %}"
        "{% for content in message['content'] %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"

        "{% for content in message['content'] %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class Qwen25VLChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

    CHAT_FORMAT = (
        #"{% set image_count = namespace(value=0) %}"
        #"{% set video_count = namespace(value=0) %}"
        "{% for message in messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
        "<|im_start|>system\n"
        "{{ self.DEFAULT_SYSTEM_MESSAGE }}<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}"
        "{% else %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        #"{% set image_count.value = image_count.value + 1 %}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "<|im_start|>assistant\n"
    )

    def __call__(self, **kwargs):
        llama = kwargs['llama']

        # Clear state for multiple runs
        llama.reset()
        llama._ctx.kv_cache_clear()
        llama.n_tokens = 0

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        # Clear any handler state
        if hasattr(self, '_last_image_embed'):
            self._last_image_embed = None
            self._last_image_hash = None

        if self.verbose:
            messages = kwargs.get('messages', [])
            image_count = len(self.get_image_urls(messages))
            print(f"Minimal - Cleared state, processing {image_count} images", file=sys.stderr)

        # Use parent implementation
        return super().__call__(**kwargs)


@register_chat_completion_handler("chatml-function-calling")
def chatml_function_calling(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    function_calling_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        # System message
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% if tool_calls %}"
        "\n\nYou have access to the following functions:\n"
        "{% for tool in tools %}"
        "\nfunctions.{{ tool.function.name }}:\n"
        "{{ tool.function.parameters | tojson }}"
        "\n{% endfor %}"
        "\n\nYou can respond to users messages with either a single message or one or more function calls."
        "\n\nTo respond with a message begin the message with 'message:', use the following format:"
        "\n\nmessage:"
        "\n<message>"
        "\n\nTo respond with one or more function calls begin the message with 'functions.<function_name>:', use the following format:"
        "\n\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "{% endif %}"
        "<|im_end|>\n"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        ## Reglar message
        "{% if message.content and message.content | length > 0 %}"
        "{% if tool_calls %}"
        "message:\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        ## Function calls
        "{% if 'tool_calls' in message %}"
        "{% for tool_call in message.tool_calls %}"
        "functions.{{ tool_call.function.name }}:\n"
        "{{ tool_call.function.arguments }}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    template_renderer = ImmutableSandboxedEnvironment(
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = (
        [stop, "<|im_end|>"]
        if isinstance(stop, str)
        else stop + ["<|im_end|>"] if stop else ["<|im_end|>"]
    )

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2: Tool choice by user
    if isinstance(tool_choice, dict):
        tool_name = tool_choice["function"]["name"]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == tool_name), None
        )
        if tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in tools")
        prompt = template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=True,
            add_generation_prompt=True,
        )
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat_function(
            tool_name, completion_or_chunks, stream
        )

    # Case 3: Automatic tool choice
    assert isinstance(tool_choice, str) and tool_choice == "auto"
    function_names = " | ".join(
        [f'''"functions.{tool['function']['name']}:"''' for tool in tools]
    )
    initial_gbnf_tool_grammar = (
        """root   ::= functions | "message:"\n"""
        f"""functions ::= {function_names}\n"""
    )
    follow_up_gbnf_tool_grammar = (
        """root   ::= functions | "<|im_end|>"\n"""
        f"""functions ::= {function_names}\n"""
    )
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )
    completion_or_chunks = llama.create_completion(
        prompt=prompt,
        temperature=0,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=[":"],
        max_tokens=None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=llama_grammar.LlamaGrammar.from_string(
            initial_gbnf_tool_grammar, verbose=llama.verbose
        ),
    )
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    text = completion["choices"][0]["text"]
    if "message" in text:
        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt + "message:\n",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=["<|im_end|>"],
                logprobs=top_logprobs if logprobs else None,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=llama_grammar.LlamaGrammar.from_string(
                    follow_up_gbnf_tool_grammar, verbose=llama.verbose
                ),
            ),
            stream=stream,
        )

    # One or more function calls
    tool_name = text[len("functions.") :]
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    if not stream:
        completions: List[llama_types.CreateCompletionResponse] = []
        completions_tool_name: List[str] = []
        while tool is not None:
            prompt += f"functions.{tool_name}:\n"
            try:
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
                )
            except Exception as e:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )
                if llama.verbose:
                    print(
                        "Failed to parse function body as JSON schema, falling back to default grammar"
                    )
                    print(e)
            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            )
            completion_or_chunks = cast(
                llama_types.CreateCompletionResponse, completion_or_chunks
            )
            completions.append(completion_or_chunks)
            completions_tool_name.append(tool_name)
            prompt += completion_or_chunks["choices"][0]["text"]
            prompt += "\n"

            response = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=llama_grammar.LlamaGrammar.from_string(
                    follow_up_gbnf_tool_grammar, verbose=llama.verbose
                ),
            )
            response = cast(llama_types.CreateCompletionResponse, response)

            tool_name = response["choices"][0]["text"][len("functions.") :]
            tool = next(
                (tool for tool in tools if tool["function"]["name"] == tool_name), None
            )

        # Merge completions
        function_call_dict: Union[
            Dict[str, str],
            Dict[
                Literal["function_call"],
                llama_types.ChatCompletionRequestAssistantMessageFunctionCall,
            ],
        ] = (
            {
                "function_call": {
                    "name": tool_name,
                    "arguments": completions[0]["choices"][0]["text"],
                }
            }
            if len(completions) == 1
            else {}
        )
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": _convert_text_completion_logprobs_to_chat(completion["choices"][0]["logprobs"]),
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_"
                                + f"_{i}_"
                                + tool_name
                                + "_"
                                + completion["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                            for i, (tool_name, completion) in enumerate(
                                zip(completions_tool_name, completions)
                            )
                        ],
                        **function_call_dict,
                    },
                }
            ],
            "usage": {
                "completion_tokens": sum(
                    (
                        completion["usage"]["completion_tokens"]
                        if "usage" in completion
                        else 0
                    )
                    for completion in completions
                ),
                "prompt_tokens": sum(
                    completion["usage"]["prompt_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
                "total_tokens": sum(
                    completion["usage"]["total_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
            },
        }

    raise ValueError("Automatic streaming tool choice is not supported")
