import base64
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


def load_server_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "server" / "server.py"
    spec = importlib.util.spec_from_file_location("example_server", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


server = load_server_module()


def test_embedding_request_accepts_openai_input_shapes():
    assert server.CreateEmbeddingRequest(
        input="hello", model="model"
    ).normalized_input() == ["hello"]
    assert server.CreateEmbeddingRequest(
        input=["hello", "world"],
        model="model",
    ).normalized_input() == ["hello", "world"]
    assert server.CreateEmbeddingRequest(
        input=[1, 2, 3], model="model"
    ).normalized_input() == [[1, 2, 3]]
    assert server.CreateEmbeddingRequest(
        input=[[1, 2], [3, 4]],
        model="model",
    ).normalized_input() == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "input_value",
    [
        "",
        [],
        ["valid", ""],
        [1, "mixed"],
        [[]],
        ["x"] * 2049,
        [1] * 2049,
        [[1] * 2049],
    ],
)
def test_embedding_request_rejects_invalid_inputs(input_value):
    with pytest.raises(ValidationError):
        server.CreateEmbeddingRequest(input=input_value, model="model")


def test_embedding_response_supports_dimensions_for_float_output():
    response = server.CreateEmbeddingResponse.from_embeddings(
        model="model",
        embeddings=[[1.0, 2.0, 3.0]],
        total_tokens=3,
        encoding_format="float",
        dimensions=2,
    )

    assert response.model_dump(mode="json") == {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [1.0, 2.0],
                "index": 0,
            }
        ],
        "model": "model",
        "usage": {
            "prompt_tokens": 3,
            "total_tokens": 3,
        },
    }


def test_embedding_response_supports_base64_output():
    response = server.CreateEmbeddingResponse.from_embeddings(
        model="model",
        embeddings=[[1.0, 2.0, 3.0]],
        total_tokens=3,
        encoding_format="base64",
        dimensions=2,
    )

    payload = response.model_dump(mode="json")
    encoded = payload["data"][0]["embedding"]
    assert isinstance(encoded, str)
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    np.testing.assert_allclose(decoded, np.array([1.0, 2.0], dtype=np.float32))


def test_embedding_response_rejects_oversized_dimensions():
    with pytest.raises(server.CompletionRequestValidationError):
        server.CreateEmbeddingResponse.from_embeddings(
            model="model",
            embeddings=[[1.0, 2.0, 3.0]],
            total_tokens=3,
            encoding_format="float",
            dimensions=4,
        )
