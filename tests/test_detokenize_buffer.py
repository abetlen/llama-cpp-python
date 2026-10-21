import ctypes
from types import SimpleNamespace

import pytest

from llama_cpp import _internals as internals


@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 34, 128])
def test_detokenize_resizes_for_required_length(monkeypatch, length):
    payload = b"x" * length
    sizes = []

    def piece(vocab, token, buffer, size, lstrip, special):
        sizes.append(size)
        if size < len(payload):
            return -len(payload)
        ctypes.memmove(buffer, payload, len(payload))
        return len(payload)

    monkeypatch.setattr(internals.llama_cpp, "llama_token_to_piece", piece)
    model = SimpleNamespace(vocab=None, token_bos=lambda: -1)
    assert internals.LlamaModel.detokenize(model, [42]) == payload
    assert sizes == ([32] if length <= 32 else [32, length])


def test_detokenize_preserves_unicode_and_reuses_buffer(monkeypatch):
    payloads = ["বাংলা ভাষার পরীক্ষা".encode(), "১২৩৪".encode()]
    assert len(payloads[0]) > 32
    sizes = []

    def piece(vocab, token, buffer, size, lstrip, special):
        assert special is True
        assert lstrip == 0
        payload = payloads[token.value]
        sizes.append(size)
        if size < len(payload):
            return -len(payload)
        ctypes.memmove(buffer, payload, len(payload))
        return len(payload)

    monkeypatch.setattr(internals.llama_cpp, "llama_token_to_piece", piece)
    model = SimpleNamespace(vocab=None, token_bos=lambda: -1)
    assert internals.LlamaModel.detokenize(model, [0, 1], special=True) == b"".join(
        payloads
    )
    assert sizes == [32, len(payloads[0]), len(payloads[0])]


@pytest.mark.parametrize("retry_result", [-40, 41])
def test_detokenize_rejects_invalid_retry(monkeypatch, retry_result):
    responses = iter([-40, retry_result])
    monkeypatch.setattr(
        internals.llama_cpp, "llama_token_to_piece", lambda *args: next(responses)
    )
    model = SimpleNamespace(vocab=None, token_bos=lambda: -1)
    with pytest.raises(RuntimeError, match="after resizing buffer"):
        internals.LlamaModel.detokenize(model, [42])
