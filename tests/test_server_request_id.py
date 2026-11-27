import json
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from starlette_context import context

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings


@pytest.fixture(scope="module")
def client():
    model = os.environ.get("LLAMA_CPP_TEST_MODEL")
    if not model:
        pytest.skip("Set LLAMA_CPP_TEST_MODEL to a small GGUF model")
    app = create_app(
        Settings(
            model=model,
            n_gpu_layers=0,
            n_ctx=128,
            n_batch=128,
            n_threads=2,
            n_threads_batch=2,
            api_key="test-key",
            verbose=False,
        )
    )

    @app.get("/request-context")
    async def request_context():
        return {"request_id": context["X-Request-ID"]}

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "request_id",
    ["proxy-request-42", "request/42:abc", "123e4567-e89b-42d3-a456-426614174000"],
)
def test_request_id_is_preserved(client, request_id):
    response = client.get("/request-context", headers={"X-Request-ID": request_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert response.json()["request_id"] == request_id


@pytest.mark.parametrize("headers", [{}, {"X-Request-ID": ""}])
def test_missing_request_id_is_generated(client, headers):
    first = client.get("/request-context", headers=headers)
    second = client.get("/request-context", headers=headers)
    assert first.status_code == second.status_code == 200
    first_id = first.headers["X-Request-ID"]
    second_id = second.headers["X-Request-ID"]
    UUID(first_id)
    UUID(second_id)
    assert first_id != second_id
    assert first.json()["request_id"] == first_id
    assert second.json()["request_id"] == second_id


def test_request_ids_are_isolated(client):
    def request(i):
        request_id = f"proxy-request-{i}"
        response = client.get("/request-context", headers={"X-Request-ID": request_id})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == request_id
        assert response.json()["request_id"] == request_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(request, range(8)))


def test_request_id_keeps_api_authentication(client):
    headers = {"X-Request-ID": "proxy-request-42"}
    denied = client.get("/v1/models", headers=headers)
    assert denied.status_code == 401
    assert denied.headers["X-Request-ID"] == "proxy-request-42"
    headers["Authorization"] = "Bearer test-key"
    allowed = client.get("/v1/models", headers=headers)
    assert allowed.status_code == 200
    assert allowed.headers["X-Request-ID"] == "proxy-request-42"
    assert allowed.json()["data"]


@pytest.mark.parametrize("stream", [False, True])
def test_native_completion_keeps_request_id(client, stream):
    response = client.post(
        "/v1/completions",
        headers={
            "X-Request-ID": "proxy-request-42",
            "Authorization": "Bearer test-key",
        },
        json={
            "prompt": "Once upon a time",
            "max_tokens": 3,
            "temperature": 0,
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "proxy-request-42"
    if stream:
        assert "text/event-stream" in response.headers["content-type"]
        events = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks)
    else:
        assert response.json()["usage"]["completion_tokens"] > 0
        assert response.json()["choices"][0]["text"]
