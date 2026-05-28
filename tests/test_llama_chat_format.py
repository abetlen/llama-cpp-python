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


def _parse(text):
    return llama_chat_format._parse_gemma4_native_tool_calls(text)


def test_gemma4_parse_string_args():
    text = (
        '<|tool_call>call:write_file{'
        'content:<|"|>print("hello")<|"|>,'
        'file_path:<|"|>hello.py<|"|>'
        '}<tool_call|>'
    )
    content, tool_calls = _parse(text)
    assert content is None
    assert tool_calls is not None and len(tool_calls) == 1
    fn = tool_calls[0]["function"]
    assert fn["name"] == "write_file"
    assert json.loads(fn["arguments"]) == {
        "content": 'print("hello")',
        "file_path": "hello.py",
    }


def test_gemma4_parse_primitive_args():
    text = (
        '<|tool_call>call:do_thing{'
        'timeout:30,temperature:0.5,background:false,note:null'
        '}<tool_call|>'
    )
    _, tool_calls = _parse(text)
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "timeout": 30,
        "temperature": 0.5,
        "background": False,
        "note": None,
    }


def test_gemma4_parse_list_of_strings():
    text = (
        '<|tool_call>call:read_files{'
        'files:[<|"|>a.py<|"|>,<|"|>b.py<|"|>]'
        '}<tool_call|>'
    )
    _, tool_calls = _parse(text)
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "files": ["a.py", "b.py"]
    }


def test_gemma4_strips_thought_block():
    text = (
        '<|channel>thought\nLet me call the function.\n<channel|>'
        '<|tool_call>call:f{x:1}<tool_call|>'
    )
    _, tool_calls = _parse(text)
    assert tool_calls and json.loads(tool_calls[0]["function"]["arguments"]) == {"x": 1}


def test_gemma4_plain_text_passthrough():
    text = "Just a normal reply with no tool call."
    content, tool_calls = _parse(text)
    assert tool_calls is None
    assert content == text


def test_gemma4_multiple_tool_calls():
    text = (
        '<|tool_call>call:a{x:1}<tool_call|>'
        '<|tool_call>call:b{y:<|"|>two<|"|>}<tool_call|>'
    )
    _, tool_calls = _parse(text)
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "a"
    assert tool_calls[1]["function"]["name"] == "b"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"y": "two"}
    # IDs must be unique across calls.
    assert tool_calls[0]["id"] != tool_calls[1]["id"]


def test_gemma4_surrounding_plain_text():
    text = "Sure, I will help.\n<|tool_call>call:f{x:1}<tool_call|>"
    content, tool_calls = _parse(text)
    assert tool_calls is not None
    assert content == "Sure, I will help."


def test_gemma4_string_with_embedded_quotes():
    # Delimiter is the 3-char sequence <|"|>, so literal " inside is fine.
    text = '<|tool_call>call:say{msg:<|"|>hello, "world"!<|"|>}<tool_call|>'
    _, tool_calls = _parse(text)
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "msg": 'hello, "world"!'
    }
