import pytest
from pydantic import ValidationError

from llama_cpp.server.types import CreateChatCompletionRequest


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, None),
        ({"max_tokens": 3}, 3),
        ({"max_completion_tokens": 3}, 3),
        ({"max_tokens": None}, None),
        ({"max_completion_tokens": None}, None),
        ({"max_tokens": 7, "max_completion_tokens": 3}, 7),
        ({"max_tokens": None, "max_completion_tokens": 3}, None),
    ],
)
def test_chat_completion_token_limit(limits, expected):
    request = CreateChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}], **limits
    )

    assert request.max_tokens == expected
    payload = request.model_dump()
    assert payload["max_tokens"] == expected
    assert "max_completion_tokens" not in payload


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_chat_completion_token_limit_validation(field):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(**{field: "invalid"})
