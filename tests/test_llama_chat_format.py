import json

import jinja2
import pytest

from llama_cpp import (
    ChatCompletionRequestUserMessage,
)
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_chat_format as llama_chat_format

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


@pytest.mark.parametrize(
    "formatter",
    [
        llama_chat_format.format_llama2,
        llama_chat_format.format_llama3,
        llama_chat_format.format_chatml,
        llama_chat_format.format_zephyr,
        llama_chat_format.format_mistral_instruct,
        llama_chat_format.format_gemma,
    ],
)
def test_chat_formatters_keep_text_of_content_parts(formatter):
    text_messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    part_messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "there"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "user", "content": [{"type": "text", "text": "Bye"}]},
    ]
    prompt = formatter(messages=part_messages).prompt
    assert "Hello there" in prompt
    assert prompt == formatter(messages=text_messages).prompt


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
