import json
import inspect

import jinja2

import llama_cpp
from llama_cpp import ChatCompletionRequestUserMessage
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_chat_format as llama_chat_format
import llama_cpp.server.types as server_types

from llama_cpp.llama_chat_format import hf_tokenizer_config_to_chat_formatter


def test_mistral_instruct():
    chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    chat_formatter = jinja2.Template(chat_template)
    messages = [
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="Instruction"
        ),
        llama_types.ChatCompletionRequestAssistantMessage(
            role="assistant", content="Model answer"
        ),
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="Follow-up instruction"
        ),
    ]
    response = llama_chat_format.format_mistral_instruct(
        messages=messages,
    )
    prompt = ("" if response.added_special else "<s>") + response.prompt
    reference = chat_formatter.render(
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
    )
    assert prompt == reference


mistral_7b_tokenizer_config = """{
  "add_bos_token": true,
  "add_eos_token": false,
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "additional_special_tokens": [],
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "</s>",
  "legacy": true,
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": null,
  "sp_model_kwargs": {},
  "spaces_between_special_tokens": false,
  "tokenizer_class": "LlamaTokenizer",
  "unk_token": "<unk>",
  "use_default_system_prompt": false,
  "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
}"""


def test_hf_tokenizer_config_str_to_chat_formatter():
    tokenizer_config = json.loads(mistral_7b_tokenizer_config)
    chat_formatter = hf_tokenizer_config_to_chat_formatter(tokenizer_config)
    chat_formatter_respoonse = chat_formatter(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ]
    )

    assert chat_formatter_respoonse.prompt == ("<s>[INST] Hello, world! [/INST]</s>")


def test_jinja2_chat_formatter_passes_template_kwargs():
    chat_formatter = llama_chat_format.Jinja2ChatFormatter(
        template="{{ reasoning_effort | default('unset') }} {{ messages[0]['content'] }}",
        bos_token="<s>",
        eos_token="</s>",
    )
    response = chat_formatter(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ],
        reasoning_effort="low",
    )

    assert response.prompt == "low Hello, world!"


def test_hf_tokenizer_config_chat_formatter_passes_template_kwargs():
    tokenizer_config = {
        "chat_template": "{{ bos_token }}{{ reasoning_effort | default('unset') }} {{ messages[0]['content'] }}",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    chat_formatter = hf_tokenizer_config_to_chat_formatter(
        tokenizer_config, add_generation_prompt=False
    )
    response = chat_formatter(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ],
        reasoning_effort="medium",
    )

    assert response.prompt == "<s>medium Hello, world!"


def test_chat_completion_handler_passes_template_kwargs():
    captured = {}

    def chat_formatter(*, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return llama_chat_format.ChatFormatterResponse(prompt="Hello")

    handler = llama_chat_format.chat_formatter_to_chat_completion_handler(
        chat_formatter
    )

    class DummyLlama:
        verbose = False

        def tokenize(self, data, add_bos, special):
            return [1]

        def create_completion(self, **kwargs):
            return {
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 0,
                "model": "dummy",
                "choices": [
                    {
                        "text": "world",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

    response = handler(
        llama=DummyLlama(),
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ],
        reasoning_effort="high",
    )

    assert response["choices"][0]["message"]["content"] == "world"
    assert captured["kwargs"]["reasoning_effort"] == "high"


def test_create_chat_completion_exposes_reasoning_effort_parameter():
    parameter = inspect.signature(llama_cpp.Llama.create_chat_completion).parameters[
        "reasoning_effort"
    ]

    assert parameter.default is None


def test_server_chat_completion_request_accepts_reasoning_effort():
    request = server_types.CreateChatCompletionRequest(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!")
        ],
        reasoning_effort="minimal",
    )

    assert request.reasoning_effort == "minimal"
