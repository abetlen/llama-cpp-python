from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from llama_cpp.server import app as server_app
from llama_cpp.server.settings import Settings


@pytest.fixture
def completion_client(monkeypatch):
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    prompts = []

    def complete(self, *, prompt, stream=False, **kwargs):
        prompts.append(prompt)
        response = {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "text": "done",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return iter([response]) if stream else response

    class FakeLlama:
        __call__ = complete

    proxy = Mock(return_value=FakeLlama())
    monkeypatch.setattr(server_app, "LlamaProxy", lambda models: proxy)
    monkeypatch.setattr(server_app, "_llama_proxy", None)
    monkeypatch.setattr(server_app, "_server_settings", None)
    monkeypatch.setattr(server_app.llama_cpp.Llama, "__call__", complete)
    app = server_app.create_app(Settings(model="unused.gguf", api_key=None))

    with TestClient(app) as client:
        yield client, proxy, prompts


@pytest.mark.parametrize(
    "path", ["/v1/completions", "/v1/engines/copilot-codex/completions"]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt", [["first", "second"], ["", ""], ["a", "b", "c"]])
def test_completion_rejects_multiple_prompts(completion_client, path, stream, prompt):
    client, proxy, prompts = completion_client
    response = client.post(path, json={"prompt": prompt, "stream": stream})

    assert response.status_code == 400
    assert response.json() == {"detail": "Only a single prompt is supported."}
    proxy.assert_not_called()
    assert prompts == []


@pytest.mark.parametrize(
    "path", ["/v1/completions", "/v1/engines/copilot-codex/completions"]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("prompt", "expected_prompt"),
    [("hello", "hello"), ("", ""), (["hello"], "hello"), ([], "")],
)
def test_completion_supported_prompts(
    completion_client, path, stream, prompt, expected_prompt
):
    client, proxy, prompts = completion_client
    response = client.post(path, json={"prompt": prompt, "stream": stream})

    assert response.status_code == 200
    assert prompts == [expected_prompt]
    proxy.assert_called_once_with(
        "copilot-codex" if path.startswith("/v1/engines/") else None
    )
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"text": "done"' in response.text
        assert "data: [DONE]" in response.text
    else:
        assert response.json()["choices"][0]["text"] == "done"
