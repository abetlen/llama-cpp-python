from unittest.mock import Mock

import pytest

from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import ModelSettings


@pytest.fixture
def proxy_and_loader(monkeypatch):
    loader = Mock(side_effect=lambda settings: Mock())
    monkeypatch.setattr(LlamaProxy, "load_llama_from_model_settings", loader)
    proxy = LlamaProxy([ModelSettings(model="model.gguf", model_alias="default")])
    return proxy, loader


def test_free_can_be_called_twice(proxy_and_loader):
    proxy, _ = proxy_and_loader
    model = proxy()
    proxy.free()
    proxy.free()
    model.close.assert_called_once_with()


def test_free_allows_default_model_reload(proxy_and_loader):
    proxy, loader = proxy_and_loader
    original = proxy()
    proxy.free()
    replacement = proxy()
    assert replacement is not original
    assert loader.call_count == 2
    original.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_free_allows_switch_to_another_model(proxy_and_loader):
    proxy, loader = proxy_and_loader
    proxy["other"] = ModelSettings(model="other.gguf", model_alias="other")
    original = proxy()
    proxy.free()
    replacement = proxy("other")
    assert replacement is not original
    assert loader.call_count == 2
    assert loader.call_args.args[0].model == "other.gguf"
    original.close.assert_called_once_with()


def test_free_without_loaded_model(proxy_and_loader):
    proxy, _ = proxy_and_loader
    proxy._current_model = None
    proxy.free()
    proxy.free()


def test_free_preserves_model_when_close_raises(proxy_and_loader):
    proxy, _ = proxy_and_loader
    model = proxy()
    model.close.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        proxy.free()
    assert proxy._current_model is model
    model.close.side_effect = None
    proxy.free()
    proxy.free()
    assert model.close.call_count == 2
