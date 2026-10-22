import argparse
import pytest
from unittest.mock import MagicMock

from llama_cpp.server.settings import ServerSettings, ModelSettings
from llama_cpp.server.cli import add_args_from_model, parse_model_from_args
from llama_cpp.server.types import (
    ChatCompletionRequestMessage,
    DetokenizeInputRequest,
    TokenizeInputRequest,
)


def test_server_types():
    # Verify tool role is allowed in ChatCompletionRequestMessage
    msg = ChatCompletionRequestMessage(role="tool", content="tool output")
    assert msg.role == "tool"
    assert msg.content == "tool output"

    detok = DetokenizeInputRequest(tokens=[1, 2, 3])
    assert detok.tokens == [1, 2, 3]

    tok = TokenizeInputRequest(input="hello")
    assert tok.input == "hello"


def test_server_cli_args():
    parser = argparse.ArgumentParser()
    add_args_from_model(parser, ServerSettings)
    args = parser.parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--interrupt_requests",
            "false",
        ]
    )
    settings = parse_model_from_args(ServerSettings, args)
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.interrupt_requests is False


def test_llama_proxy_container():
    from llama_cpp.server.model import LlamaProxy

    mock_model = MagicMock()
    model_settings = [
        ModelSettings(model="model-a.gguf", model_alias="model-a"),
        ModelSettings(model="model-b.gguf", model_alias="model-b"),
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LlamaProxy, "load_llama_from_model_settings", lambda self, s: mock_model
        )
        proxy = LlamaProxy(models=model_settings)

        assert len(proxy) == 2
        assert "model-a" in proxy
        assert "model-b" in proxy
        assert "non-existent" not in proxy
        assert proxy.get("model-a") is not None
        assert proxy.get("non-existent") is None
        assert proxy.get("non-existent", "default") == "default"


def test_server_models_endpoints():
    pytest.importorskip("sse_starlette")
    pytest.importorskip("starlette_context")
    from fastapi.testclient import TestClient
    from llama_cpp.server.app import create_app
    from llama_cpp.server.model import LlamaProxy

    mock_model = MagicMock()
    model_settings = [
        ModelSettings(model="model-a.gguf", model_alias="model-a"),
    ]
    server_settings = ServerSettings(api_key=None)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LlamaProxy, "load_llama_from_model_settings", lambda self, s: mock_model
        )
        app = create_app(server_settings=server_settings, model_settings=model_settings)
        client = TestClient(app)

        # GET /v1/models
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "model-a"

        # GET /v1/models/{model} for existing model
        response = client.get("/v1/models/model-a")
        assert response.status_code == 200
        model_data = response.json()
        assert model_data["id"] == "model-a"
        assert model_data["object"] == "model"
        assert model_data["owned_by"] == "me"

        # GET /v1/models/{model} for non-existent model returns 404
        response = client.get("/v1/models/unknown-model")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_server_authentication():
    pytest.importorskip("sse_starlette")
    pytest.importorskip("starlette_context")
    from fastapi.testclient import TestClient
    from llama_cpp.server.app import create_app
    from llama_cpp.server.model import LlamaProxy

    mock_model = MagicMock()
    model_settings = [
        ModelSettings(model="model-a.gguf", model_alias="model-a"),
    ]
    server_settings = ServerSettings(api_key="secret-token-123")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LlamaProxy, "load_llama_from_model_settings", lambda self, s: mock_model
        )
        app = create_app(server_settings=server_settings, model_settings=model_settings)
        client = TestClient(app)

        # Missing token -> 401
        res = client.get("/v1/models")
        assert res.status_code == 401

        # Invalid token -> 401
        res = client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong-token"}
        )
        assert res.status_code == 401

        # Valid token -> 200
        res = client.get(
            "/v1/models", headers={"Authorization": "Bearer secret-token-123"}
        )
        assert res.status_code == 200


def test_server_route_error_handler_resilience():
    pytest.importorskip("sse_starlette")
    pytest.importorskip("starlette_context")
    from fastapi.testclient import TestClient
    from llama_cpp.server.app import create_app
    from llama_cpp.server.model import LlamaProxy

    mock_model = MagicMock()
    mock_model.tokenize.return_value = [1, 2, 3]
    mock_model.detokenize.return_value = b"hello"
    model_settings = [
        ModelSettings(model="model-a.gguf", model_alias="model-a"),
    ]
    server_settings = ServerSettings(api_key=None)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LlamaProxy, "load_llama_from_model_settings", lambda self, s: mock_model
        )
        app = create_app(server_settings=server_settings, model_settings=model_settings)
        client = TestClient(app)

        # Post non-json body to tokenize
        response = client.post(
            "/extras/tokenize",
            content=b"not-json-content",
            headers={"Content-Type": "application/json"},
        )
        # Should gracefully return 422 unprocessable entity, not 500 JSONDecodeError crash
        assert response.status_code in (400, 422)
