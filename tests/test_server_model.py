from unittest.mock import Mock

import pytest

import llama_cpp
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import ModelSettings


@pytest.mark.parametrize(
    "value",
    [
        "user: ",
        "{{ messages[0]['role'] == 'user' }}",
        "user: name=value",
        "",
        ":",
        "=",
        "用户: name=value",
    ],
)
def test_server_kv_overrides_preserve_string_values(monkeypatch, value):
    create_model = Mock()
    monkeypatch.setattr(llama_cpp, "Llama", create_model)
    settings = ModelSettings(
        model="unused.gguf",
        kv_overrides=[f"tokenizer.chat_template=str:{value}"],
    )

    result = LlamaProxy.load_llama_from_model_settings(settings)

    assert result is create_model.return_value
    assert create_model.call_args.kwargs["kv_overrides"] == {
        "tokenizer.chat_template": value
    }


def test_server_kv_overrides_preserve_numeric_and_bool_values(monkeypatch):
    create_model = Mock()
    monkeypatch.setattr(llama_cpp, "Llama", create_model)
    settings = ModelSettings(
        model="unused.gguf",
        kv_overrides=[
            "tokenizer.ggml.add_bos_token=bool:true",
            "tokenizer.ggml.add_eos_token=bool:false",
            "llama.context_length=int:2048",
            "llama.rope.freq_base=float:10000.5",
        ],
    )

    LlamaProxy.load_llama_from_model_settings(settings)

    assert create_model.call_args.kwargs["kv_overrides"] == {
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": False,
        "llama.context_length": 2048,
        "llama.rope.freq_base": 10000.5,
    }
