import argparse
import json
import os

import pytest
from fastapi.testclient import TestClient

from llama_cpp.server import app as server_app
from llama_cpp.server.cli import add_args_from_model, parse_model_from_args
from llama_cpp.server.settings import (
    ConfigFileSettings,
    ModelSettings,
    ServerSettings,
    Settings,
)
from llama_cpp.server.types import CreateChatCompletionRequest


GRAMMAR_A = 'root ::= "a"'
GRAMMAR_B = 'root ::= "b"'


def test_cli_default_grammar():
    parser = argparse.ArgumentParser()
    add_args_from_model(parser, Settings)
    args = parser.parse_args(["--model", "local.gguf", "--grammar", GRAMMAR_A])
    assert parse_model_from_args(ServerSettings, args).grammar == GRAMMAR_A


def test_environment_default_grammar(monkeypatch):
    monkeypatch.setenv("GRAMMAR", GRAMMAR_A)
    assert ServerSettings().grammar == GRAMMAR_A


def test_config_file_default_grammar():
    config = ConfigFileSettings.model_validate_json(
        json.dumps({"grammar": GRAMMAR_A, "models": [{"model": "local.gguf"}]})
    )
    assert ServerSettings.model_validate(config).grammar == GRAMMAR_A


@pytest.fixture(params=[None, GRAMMAR_A])
def native_server(request, monkeypatch):
    # Opt in with a small real GGUF; never download model weights in this suite.
    model = os.environ.get("LLAMA_CPP_TEST_MODEL")
    if not model:
        pytest.skip("Set LLAMA_CPP_TEST_MODEL to a local GGUF for native server tests")
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("GRAMMAR", raising=False)
    app = server_app.create_app(
        server_settings=ServerSettings(grammar=request.param),
        model_settings=[
            ModelSettings(
                model=model,
                n_gpu_layers=0,
                n_ctx=128,
                n_batch=128,
                n_threads=2,
                n_threads_batch=2,
                verbose=False,
            )
        ],
    )
    try:
        with TestClient(app) as client:
            yield client, request.param
    finally:
        server_app._llama_proxy.free()


def completion_request(**kwargs):
    return {"prompt": "Once upon a time", "max_tokens": 8, "temperature": 0, **kwargs}


def test_native_default_or_unconstrained_completion(native_server):
    client, default = native_server
    response = client.post("/v1/completions", json=completion_request())
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["completion_tokens"] > 0
    text = body["choices"][0]["text"]
    assert text == "a" if default is not None else len(text) > 0


def test_native_request_grammar_overrides_default(native_server):
    client, _ = native_server
    response = client.post(
        "/v1/completions", json=completion_request(grammar=GRAMMAR_B)
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "b"


def test_native_streaming_grammar(native_server):
    client, default = native_server
    kwargs = {} if default is not None else {"grammar": GRAMMAR_B}
    response = client.post(
        "/v1/completions", json=completion_request(stream=True, **kwargs)
    )
    assert response.status_code == 200
    events = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    text = "".join(json.loads(event)["choices"][0]["text"] for event in events[:-1])
    assert text == ("a" if default is not None else "b")


def test_default_grammar_does_not_apply_to_chat(native_server):
    request = CreateChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    kwargs = {}
    server_app.prepare_request_resources(request, server_app._llama_proxy, None, kwargs)
    assert "grammar" not in kwargs
