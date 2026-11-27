import argparse
import json
import os
from urllib.parse import quote

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


KEY = "local-test-key"


def test_static_directory_cli():
    parser = argparse.ArgumentParser()
    add_args_from_model(parser, Settings)
    args = parser.parse_args(["--model", "local.gguf", "--static_dir", "web"])
    assert parse_model_from_args(ServerSettings, args).static_dir == "web"


def test_static_directory_environment(monkeypatch):
    monkeypatch.setenv("STATIC_DIR", "web")
    assert ServerSettings().static_dir == "web"


def test_static_directory_config_file():
    config = ConfigFileSettings.model_validate_json(
        json.dumps({"static_dir": "web", "models": [{"model": "local.gguf"}]})
    )
    assert ServerSettings.model_validate(config).static_dir == "web"


def test_missing_directory_fails_before_model_load(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    with pytest.raises(RuntimeError, match="does not exist"):
        server_app.create_app(
            server_settings=ServerSettings(static_dir=str(tmp_path / "missing")),
            model_settings=[ModelSettings(model="unused.gguf")],
        )


@pytest.fixture(params=[False, True], ids=["disabled", "enabled"])
def native_server(request, tmp_path, monkeypatch):
    # Opt in with a small real GGUF; never download model weights in this suite.
    model = os.environ.get("LLAMA_CPP_TEST_MODEL")
    if not model:
        pytest.skip("Set LLAMA_CPP_TEST_MODEL to a local GGUF for native server tests")
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("STATIC_DIR", raising=False)
    public = tmp_path / "public"
    public.mkdir()
    (public / "index.html").write_text("<html>static fixture</html>")
    (public / "asset.txt").write_text("inside-asset")
    private = tmp_path / "private"
    private.mkdir()
    outside = private / "secret.txt"
    outside.write_text("outside-sentinel")
    (public / "escape.txt").symlink_to(outside)
    (public / "escape-dir").symlink_to(private, target_is_directory=True)
    (public / "inside-link.txt").symlink_to(public / "asset.txt")
    (public / "v1").mkdir()
    (public / "v1" / "models").write_text("must-not-shadow-api")
    (public / "docs").write_text("must-not-shadow-docs")
    app = server_app.create_app(
        server_settings=ServerSettings(
            api_key=KEY, static_dir=str(public) if request.param else None
        ),
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
            yield client, request.param, outside
    finally:
        server_app._llama_proxy.free()


def test_public_static_directory_is_opt_in(native_server):
    client, enabled, _ = native_server
    for path, text in [("/", "static fixture"), ("/asset.txt", "inside-asset")]:
        # Static resources are public, even when the API has a key.
        response = client.get(path)
        assert response.status_code == (200 if enabled else 404)
        if enabled:
            assert text in response.text


def test_static_directory_confinement(native_server):
    client, enabled, outside = native_server
    for path in [
        "/%2e%2e/private/secret.txt",
        "/" + quote(str(outside), safe=""),
        "/escape.txt",
        "/escape-dir/secret.txt",
    ]:
        response = client.get(path)
        assert response.status_code == 404
        assert "outside-sentinel" not in response.text
    response = client.get("/inside-link.txt")
    assert response.status_code == (200 if enabled else 404)
    if enabled:
        assert response.text == "inside-asset"


def test_api_and_docs_take_precedence(native_server):
    client, _, _ = native_server
    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200
    assert response.json()["data"]
    assert "must-not-shadow-api" not in response.text
    response = client.get("/docs")
    assert response.status_code == 200
    assert "must-not-shadow-docs" not in response.text


def test_native_completion_auth_is_unchanged(native_server):
    client, _, _ = native_server
    body = {"prompt": "Once upon a time", "max_tokens": 3, "temperature": 0}
    assert client.post("/v1/completions", json=body).status_code == 401
    response = client.post(
        "/v1/completions", json=body, headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] > 0
    assert response.json()["choices"][0]["text"]
