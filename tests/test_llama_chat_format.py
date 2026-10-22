import json

import jinja2

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


def test_format_llama3():
    messages = [
        llama_types.ChatCompletionRequestSystemMessage(
            role="system", content="You are a helpful assistant."
        ),
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="Hello Llama 3"
        ),
    ]
    response = llama_chat_format.format_llama3(messages=messages)
    expected_prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Hello Llama 3<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert response.prompt == expected_prompt
    assert response.stop == "<|eot_id|>"


def test_format_alpaca():
    messages = [
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="Summarize this text."
        ),
    ]
    response = llama_chat_format.format_alpaca(messages=messages)
    assert "### Instruction: Summarize this text." in response.prompt


def test_format_qwen():
    messages = [
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="Hello Qwen"
        ),
    ]
    response = llama_chat_format.format_qwen(messages=messages)
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in response.prompt
    assert "<|im_start|>user\nHello Qwen<|im_end|>" in response.prompt
    assert "<|im_start|>assistant\n" in response.prompt
    assert response.stop == "<|endoftext|>"


def test_format_chatml():
    messages = [
        llama_types.ChatCompletionRequestSystemMessage(
            role="system", content="System instruction"
        ),
        llama_types.ChatCompletionRequestUserMessage(
            role="user", content="User message"
        ),
    ]
    response = llama_chat_format.format_chatml(messages=messages)
    assert "<|im_start|>system\nSystem instruction<|im_end|>" in response.prompt
    assert "<|im_start|>user\nUser message<|im_end|>" in response.prompt
    assert "<|im_start|>assistant\n" in response.prompt
    assert response.stop == "<|im_end|>"

