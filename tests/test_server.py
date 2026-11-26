import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import llama_cpp
from llama_cpp.server.app import get_llama_proxy, get_server_settings, router
from llama_cpp.server.settings import ServerSettings


MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


@pytest.fixture
def client():
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_llama_proxy] = lambda: lambda model=None: llama
    app.dependency_overrides[get_server_settings] = lambda: ServerSettings()
    return TestClient(app)


def test_detokenize_endpoint(client):
    response = client.post("/extras/detokenize", json={"tokens": [15043, 2787]})

    assert response.status_code == 200
    assert response.json() == {"text": " Hello World"}


def test_detokenize_endpoint_partial_utf8(client):
    # <0xE2> byte-fallback token: the first byte of a three-byte UTF-8 character.
    response = client.post("/extras/detokenize", json={"tokens": [0xE2 + 3]})

    assert response.status_code == 200
    assert response.json() == {"text": "\ufffd"}
