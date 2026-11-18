import json
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llama_cpp
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import ModelSettings


@pytest.fixture
def loader(monkeypatch, tmp_path):
    path = tmp_path / "tokenizer_config.json"
    factory = Mock(return_value=object())
    model = Mock()
    monkeypatch.setattr(llama_cpp, "Llama", Mock(return_value=model), raising=False)
    monkeypatch.setattr(
        llama_cpp,
        "llama_chat_format",
        SimpleNamespace(hf_tokenizer_config_to_chat_completion_handler=factory),
        raising=False,
    )
    settings = ModelSettings(
        model="unused.gguf",
        chat_format="hf-tokenizer-config",
        hf_tokenizer_config_path=str(path),
    )
    return path, factory, model, settings


def test_invalid_json_closes_file_even_while_traceback_is_retained(loader):
    path, factory, _, settings = loader
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as error:
        LlamaProxy.load_llama_from_model_settings(settings)
    traceback = error.value.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__") == "json" and "fp" in frame.f_locals:
            assert frame.f_locals["fp"].closed
            break
        traceback = traceback.tb_next
    else:
        pytest.fail("json.load frame missing from the retained traceback")
    factory.assert_not_called()


def test_success_does_not_emit_unclosed_file_warning(loader):
    path, factory, model, settings = loader
    path.write_text('{"chat_template": "test"}', encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        assert LlamaProxy.load_llama_from_model_settings(settings) is model
    assert not [
        warning for warning in caught if issubclass(warning.category, ResourceWarning)
    ]
    factory.assert_called_once_with({"chat_template": "test"})


def test_handler_failure_preserves_error_without_unclosed_file_warning(loader):
    path, factory, _, settings = loader
    path.write_text("{}", encoding="utf-8")
    failure = ValueError("handler failed")
    factory.side_effect = failure
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(ValueError, match="handler failed") as error:
            LlamaProxy.load_llama_from_model_settings(settings)
    assert error.value is failure
    assert not [
        warning for warning in caught if issubclass(warning.category, ResourceWarning)
    ]


def test_missing_file_preserves_error_without_creating_model(loader):
    _, factory, _, settings = loader
    with pytest.raises(FileNotFoundError):
        LlamaProxy.load_llama_from_model_settings(settings)
    factory.assert_not_called()
    llama_cpp.Llama.assert_not_called()
