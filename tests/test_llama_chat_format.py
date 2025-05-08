import json
import os
import platform
from collections.abc import Iterator
from typing import cast

import pytest
import jinja2
from typeguard import ForwardRefPolicy, check_type

from llama_cpp import (
    ChatCompletionRequestUserMessage,
    Llama,
    llama_chat_format,
    llama_supports_gpu_offload,
    llama_types
)
from llama_cpp.llama_chat_format import hf_tokenizer_config_to_chat_formatter
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    ChatCompletionTool,
    ChatCompletionToolChoiceOption,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


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


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(True, id="stream=True"),
        pytest.param(False, id="stream=False"),
    ],
)
@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("none", id="tool_choice=none"),
        pytest.param("auto", id="tool_choice=auto"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}}, id="tool_choice=fixed"
        ),
    ],
)
@pytest.mark.parametrize(
    "user_prompt_expected_tool_calls",
    [
        pytest.param(
            ("Is 7 a prime number?", 0),
            id="expected_tool_calls=0",
        ),
        pytest.param(
            ("What's the weather like in Paris today?", 1),
            id="expected_tool_calls=1",
        ),
        pytest.param(
            ("What's the weather like in Paris today? What about New York?", 2),
            id="expected_tool_calls=2",
        ),
    ],
)
@pytest.mark.parametrize(
    "llm_repo_id",
    [
        pytest.param("bartowski/Llama-3.2-3B-Instruct-GGUF", id="llama_3.2_3B"),
        pytest.param(
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            id="llama_3.1_8B",
            marks=pytest.mark.skipif(
                not is_accelerator_available(), reason="Accelerator not available"
            ),
        ),
    ],
)
@pytest.mark.skipif(
    platform.system() == "Darwin" and (os.cpu_count() or 1) < 8,
    reason="Insufficient resources on macOS",
)
def test_llama_cpp_python_tool_use(
    llm_repo_id: str,
    user_prompt_expected_tool_calls: tuple[str, int],
    tool_choice: ChatCompletionToolChoiceOption,
    stream: bool,
) -> None:
    """Test the upgraded chatml-function-calling llama-cpp-python chat handler."""
    user_prompt, expected_tool_calls = user_prompt_expected_tool_calls
    if isinstance(tool_choice, dict) and expected_tool_calls == 0:
        pytest.skip("Nonsensical")
    llm = Llama.from_pretrained(
        repo_id=llm_repo_id,
        filename="*Q4_K_M.gguf",
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="chatml-function-calling",
    )
    messages: list[ChatCompletionRequestMessage] = [{"role": "user", "content": user_prompt}]
    tools: list[ChatCompletionTool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "A city name."}},
                },
            },
        }
    ]
    response = llm.create_chat_completion(
        messages=messages, tools=tools, tool_choice=tool_choice, stream=stream
    )
    if stream:
        response = cast(Iterator[CreateChatCompletionStreamResponse], response)
        num_tool_calls = 0
        for chunk in response:
            check_type(chunk, CreateChatCompletionStreamResponse)
            tool_calls = chunk["choices"][0]["delta"].get("tool_calls")
            if isinstance(tool_calls, list):
                num_tool_calls = max(tool_call["index"] for tool_call in tool_calls) + 1
        assert num_tool_calls == (expected_tool_calls if tool_choice != "none" else 0)
    else:
        response = cast(CreateChatCompletionResponse, response)
        check_type(
            response, CreateChatCompletionResponse, forward_ref_policy=ForwardRefPolicy.IGNORE
        )
        if expected_tool_calls == 0 or tool_choice == "none":
            assert response["choices"][0]["message"].get("tool_calls") is None
        else:
            assert len(response["choices"][0]["message"]["tool_calls"]) == expected_tool_calls
            assert all(
                tool_call["function"]["name"] == tools[0]["function"]["name"]
                for tool_call in response["choices"][0]["message"]["tool_calls"]
            )
