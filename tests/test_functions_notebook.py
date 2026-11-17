"""Exercise the independent documentation lookup cells without network or inference."""

import copy
from email.message import Message
import json
from pathlib import Path
from urllib.error import URLError

import pytest


@pytest.fixture
def example():
    notebook = (
        Path(__file__).resolve().parents[1] / "examples/notebooks/Functions.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith(("import json\nimport os\n", "LOOKUP_TOOL =")):
            exec(compile(source, str(notebook), "exec"), namespace)

    def no_network(*args, **kwargs):
        raise AssertionError("Fixture checks must not contact external services")

    namespace["urlopen"] = no_network
    return namespace


class LocalModel:
    def __init__(self, *messages):
        self.messages = iter(messages)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return {"choices": [{"message": next(self.messages)}]}


def local_call(name="lookup_python_docs", arguments="{}"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "local-call-17",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_default_and_no_tool_do_not_retrieve(example):
    message = {"role": "assistant", "content": "A local answer"}
    model = LocalModel(message)
    result = example["run_documentation_lookup"](model)
    assert result == {"status": "disabled", "message": message}
    assert "tools" not in model.calls[0]
    assert "tool_choice" not in model.calls[0]

    model = LocalModel(message)
    result = example["run_documentation_lookup"](model, enable_web_search=True)
    assert result == {"status": "no_tool", "message": message}
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "call",
    [local_call("web_fetch"), local_call(arguments='{"query":"private"}')],
)
def test_unexpected_call_does_not_retrieve(example, call):
    with pytest.raises(ValueError, match="Unexpected local function call"):
        example["run_documentation_lookup"](LocalModel(call), enable_web_search=True)


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError(), ValueError()])
def test_unavailable_does_not_generate_answer(example, error):
    def unavailable():
        raise error

    example["lookup_python_docs"] = unavailable
    model = LocalModel(local_call())
    result = example["run_documentation_lookup"](model, enable_web_search=True)
    assert result["status"] == "unavailable"
    assert len(model.calls) == 1


def test_empty_does_not_generate_answer(example):
    example["lookup_python_docs"] = lambda: []
    model = LocalModel(local_call())
    result = example["run_documentation_lookup"](model, enable_web_search=True)
    assert result == {"status": "empty", "sources": []}
    assert len(model.calls) == 1


@pytest.mark.parametrize("matching_citation", [True, False])
def test_local_id_evidence_and_original_citation(example, matching_citation):
    sources = [
        {"url": "https://docs.python.org/example", "excerpts": ["Exact excerpt"]}
    ]
    example["lookup_python_docs"] = lambda: sources
    source = sources[0]["url"] if matching_citation else "https://example.org/invented"
    model = LocalModel(
        local_call(),
        {
            "role": "assistant",
            "content": json.dumps({"answer": "Fixture answer", "source": source}),
        },
    )
    if matching_citation:
        result = example["run_documentation_lookup"](model, enable_web_search=True)
        assert result["answer"]["source"] == source
        assert result["tool_call_id"] == "local-call-17"
    else:
        with pytest.raises(ValueError, match="original returned source URL"):
            example["run_documentation_lookup"](model, enable_web_search=True)
    final = model.calls[-1]
    tool = next(m for m in final["messages"] if m["role"] == "tool")
    assert tool["tool_call_id"] == "local-call-17"
    assert json.loads(tool["content"]) == sources
    assert json.dumps(sources) in final["messages"][-1]["content"]
    assert final["tool_choice"] == "none"
    assert "tools" not in final
    assert final["response_format"]["schema"] == example["ANSWER_SCHEMA"]


@pytest.mark.parametrize("structured", [True, False])
def test_search_is_fixed_and_sources_are_bounded(example, structured):
    items = [
        {"url": "https://docs.python.org.evil.test/", "excerpts": ["bad"]},
        {"url": "http://docs.python.org/", "excerpts": ["bad"]},
        {"url": "https://name@docs.python.org/", "excerpts": ["bad"]},
        {"url": "https://docs.python.org/empty", "excerpts": []},
    ] + [
        {"url": "https://docs.python.org/" + str(i), "excerpts": ["x" * 3000] * 4}
        for i in range(3)
    ]
    data = {"results": items}
    result = (
        {"structuredContent": data}
        if structured
        else {"content": [{"type": "text", "text": json.dumps(data)}]}
    )
    requests = []

    def request(method, params, request_id=None):
        requests.append((method, params, request_id))
        return result if method == "tools/call" else {}

    example["mcp_request"] = request
    sources = example["lookup_python_docs"]()
    assert sources == [
        {"url": item["url"], "excerpts": ["x" * 2000] * 2} for item in items[4:6]
    ]
    assert [r[0] for r in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert requests[-1][1] == {
        "name": "web_search",
        "arguments": {
            "objective": "Find the Python documentation stating when tomllib was added to the standard library.",
            "search_queries": ["site:docs.python.org tomllib added version"],
        },
    }


@pytest.mark.parametrize("sse", [True, False])
def test_transport_headers_timeout_and_response(example, sse):
    reply = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}})
    body = (": keepalive\r\n\r\ndata: " + reply + "\r\n\r\n") if sse else reply

    class Response:
        headers = Message()
        headers["Content-Type"] = "text/event-stream" if sse else "application/json"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            assert limit == 1_048_577
            return body.encode()

    def open_request(request, timeout):
        assert timeout == 30
        assert request.full_url == "https://mcp.parallel.ai/mcp"
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["user-agent"] == "llama-cpp-python-functions-example/1.0"
        assert "authorization" not in headers
        assert "x-api-key" not in headers
        assert json.loads(request.data)["id"] == 7
        return Response()

    example["urlopen"] = open_request
    assert example["mcp_request"]("tools/list", {}, 7) == {"ok": True}
