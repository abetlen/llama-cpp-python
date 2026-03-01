import json
import sys
import logging
import ctypes
from unittest.mock import MagicMock

import jinja2

# Stub the native C library and dependent modules so tests can run
# without compiling llama.cpp
_mock_llama_cpp = MagicMock()
_mock_llama_cpp.llama_log_callback = lambda f: f  # decorator passthrough
_mock_llama_cpp.llama_log_set = MagicMock()
sys.modules.setdefault("llama_cpp.llama_cpp", _mock_llama_cpp)

_mock_llama = MagicMock()
_mock_llama.StoppingCriteriaList = list
_mock_llama.LogitsProcessorList = list
_mock_llama.LlamaGrammar = MagicMock
sys.modules.setdefault("llama_cpp.llama", _mock_llama)

import llama_cpp.llama_types as llama_types
import llama_cpp.llama_chat_format as llama_chat_format

from llama_cpp.llama_chat_format import (
    hf_tokenizer_config_to_chat_formatter,
    guess_chat_format_from_gguf_metadata,
    DEEPSEEK_R1_CHAT_TEMPLATE,
)

ChatCompletionRequestUserMessage = llama_types.ChatCompletionRequestUserMessage

def test_mistral_instruct():
    chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    chat_formatter = jinja2.Template(chat_template)
    messages = [
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Instruction"),
        llama_types.ChatCompletionRequestAssistantMessage(role="assistant", content="Model answer"),
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Follow-up instruction"),
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
    chat_formatter = hf_tokenizer_config_to_chat_formatter(
        tokenizer_config
    )
    chat_formatter_respoonse = chat_formatter(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ]
    )

    assert chat_formatter_respoonse.prompt == ("<s>[INST] Hello, world! [/INST]</s>" "")


def test_deepseek_r1_single_turn():
    """Test DeepSeek R1 format with a single user message."""
    messages = [
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Hello"),
    ]
    response = llama_chat_format.format_deepseek_r1(messages=messages)

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    user_tag = "<\uff5cUser\uff5c>"
    assistant_tag = "<\uff5cAssistant\uff5c>"

    expected = f"{bos}{user_tag}Hello{assistant_tag}"
    assert response.prompt == expected
    assert response.stop == eos
    assert response.added_special is True


def test_deepseek_r1_with_system_message():
    """Test DeepSeek R1 format with a system message."""
    messages = [
        llama_types.ChatCompletionRequestSystemMessage(role="system", content="You are a helpful assistant."),
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Hi"),
    ]
    response = llama_chat_format.format_deepseek_r1(messages=messages)

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    user_tag = "<\uff5cUser\uff5c>"
    assistant_tag = "<\uff5cAssistant\uff5c>"

    expected = f"{bos}You are a helpful assistant.{user_tag}Hi{assistant_tag}"
    assert response.prompt == expected


def test_deepseek_r1_multi_turn():
    """Test DeepSeek R1 format with multi-turn conversation."""
    messages = [
        llama_types.ChatCompletionRequestUserMessage(role="user", content="What is 2+2?"),
        llama_types.ChatCompletionRequestAssistantMessage(role="assistant", content="4"),
        llama_types.ChatCompletionRequestUserMessage(role="user", content="And 3+3?"),
    ]
    response = llama_chat_format.format_deepseek_r1(messages=messages)

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    user_tag = "<\uff5cUser\uff5c>"
    assistant_tag = "<\uff5cAssistant\uff5c>"

    expected = (
        f"{bos}"
        f"{user_tag}What is 2+2?"
        f"{assistant_tag}4{eos}"
        f"{user_tag}And 3+3?"
        f"{assistant_tag}"
    )
    assert response.prompt == expected


def test_deepseek_r1_think_stripping():
    """Test that </think> reasoning content is stripped from assistant messages in multi-turn."""
    messages = [
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Solve x+1=3"),
        llama_types.ChatCompletionRequestAssistantMessage(
            role="assistant",
            content="<think>Let me solve this step by step. x+1=3, so x=2.</think>x = 2",
        ),
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Are you sure?"),
    ]
    response = llama_chat_format.format_deepseek_r1(messages=messages)

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    user_tag = "<\uff5cUser\uff5c>"
    assistant_tag = "<\uff5cAssistant\uff5c>"

    # The thinking content should be stripped, only "x = 2" remains
    expected = (
        f"{bos}"
        f"{user_tag}Solve x+1=3"
        f"{assistant_tag}x = 2{eos}"
        f"{user_tag}Are you sure?"
        f"{assistant_tag}"
    )
    assert response.prompt == expected


def test_deepseek_r1_distill_aliases():
    """Test that distilled model aliases produce the same output as the base format."""
    messages = [
        llama_types.ChatCompletionRequestUserMessage(role="user", content="Hello"),
    ]
    base = llama_chat_format.format_deepseek_r1(messages=messages)
    qwen = llama_chat_format.format_deepseek_r1_distill_qwen(messages=messages)
    llama_variant = llama_chat_format.format_deepseek_r1_distill_llama(messages=messages)

    assert base.prompt == qwen.prompt
    assert base.prompt == llama_variant.prompt
    assert base.stop == qwen.stop == llama_variant.stop
    assert base.added_special == qwen.added_special == llama_variant.added_special


def test_guess_chat_format_deepseek_r1_exact_match():
    """Test auto-detection via exact template match."""
    metadata = {"tokenizer.chat_template": DEEPSEEK_R1_CHAT_TEMPLATE}
    assert guess_chat_format_from_gguf_metadata(metadata) == "deepseek-r1"


def test_guess_chat_format_deepseek_r1_heuristic():
    """Test auto-detection via heuristic token presence."""
    # A template that contains the DeepSeek tokens but isn't an exact match
    fake_template = "some preamble <\uff5cUser\uff5c> stuff <\uff5cAssistant\uff5c> more stuff"
    metadata = {"tokenizer.chat_template": fake_template}
    assert guess_chat_format_from_gguf_metadata(metadata) == "deepseek-r1"


def test_guess_chat_format_no_match():
    """Test that unrecognized templates return None."""
    metadata = {"tokenizer.chat_template": "some unknown template"}
    assert guess_chat_format_from_gguf_metadata(metadata) is None


def test_guess_chat_format_no_template():
    """Test that missing chat_template returns None."""
    metadata = {}
    assert guess_chat_format_from_gguf_metadata(metadata) is None
