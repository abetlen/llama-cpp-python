import ctypes

import pytest

import llama_cpp

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama.gguf"


def test_llama_cpp_tokenization():
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    assert llama
    assert llama._ctx.ctx is not None

    text = b"Hello World"

    tokens = llama.tokenize(text)
    assert tokens[0] == llama.token_bos()
    assert tokens == [1, 15043, 2787]
    detokenized = llama.detokenize(tokens)
    assert detokenized == text

    tokens = llama.tokenize(text, add_bos=False)
    assert tokens[0] != llama.token_bos()
    assert tokens == [15043, 2787]

    detokenized = llama.detokenize(tokens)
    assert detokenized != text

    text = b"Hello World</s>"
    tokens = llama.tokenize(text)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [1, 15043, 2787, 829, 29879, 29958]

    tokens = llama.tokenize(text, special=True)
    assert tokens[-1] == llama.token_eos()
    assert tokens == [1, 15043, 2787, 2]

    text = b""
    tokens = llama.tokenize(text, add_bos=True, special=True)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [llama.token_bos()]
    assert text == llama.detokenize(tokens)


@pytest.fixture
def mock_llama(monkeypatch):
    def setup_mock(llama: llama_cpp.Llama, output_text: str):
        llama.reset()
        n_vocab = llama.n_vocab()
        output_tokens = llama.tokenize(
            output_text.encode("utf-8"), add_bos=True, special=True
        )
        n = 0
        last_n_tokens = 0

        def mock_decode(ctx: llama_cpp.llama_context_p, batch: llama_cpp.llama_batch):
            nonlocal n
            nonlocal last_n_tokens
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx
            assert llama.n_tokens == n
            assert batch.n_tokens > 0
            n += batch.n_tokens
            last_n_tokens = batch.n_tokens
            return 0

        def mock_get_logits(*args, **kwargs):
            nonlocal last_n_tokens
            size = n_vocab * last_n_tokens
            return (llama_cpp.c_float * size)()

        def mock_sample(*args, **kwargs):
            nonlocal n
            if n < len(output_tokens):
                return output_tokens[n]
            else:
                return llama.token_eos()

        monkeypatch.setattr("llama_cpp.llama_cpp.llama_decode", mock_decode)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_get_logits", mock_get_logits)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_sample_token", mock_sample)

    return setup_mock


def test_llama_patch(mock_llama):
    n_ctx = 128
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, n_ctx=n_ctx)
    n_vocab = llama_cpp.llama_n_vocab(llama._model.model)
    assert n_vocab == 32000

    text = "The quick brown fox"
    output_text = " jumps over the lazy dog."
    all_text = text + output_text

    ## Test basic completion from bos until eos
    mock_llama(llama, all_text)
    completion = llama.create_completion("", max_tokens=36)
    assert completion["choices"][0]["text"] == all_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until eos
    mock_llama(llama, all_text)
    completion = llama.create_completion(text, max_tokens=20)
    assert completion["choices"][0]["text"] == output_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until eos
    mock_llama(llama, all_text)
    chunks = list(llama.create_completion(text, max_tokens=20, stream=True))
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == output_text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until stop sequence
    mock_llama(llama, all_text)
    completion = llama.create_completion(text, max_tokens=20, stop=["lazy"])
    assert completion["choices"][0]["text"] == " jumps over the "
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until stop sequence
    mock_llama(llama, all_text)
    chunks = list(
        llama.create_completion(text, max_tokens=20, stream=True, stop=["lazy"])
    )
    assert (
        "".join(chunk["choices"][0]["text"] for chunk in chunks) == " jumps over the "
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until length
    mock_llama(llama, all_text)
    completion = llama.create_completion(text, max_tokens=2)
    assert completion["choices"][0]["text"] == " jumps"
    assert completion["choices"][0]["finish_reason"] == "length"

    ## Test streaming completion until length
    mock_llama(llama, all_text)
    chunks = list(llama.create_completion(text, max_tokens=2, stream=True))
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == " jumps"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_llama_pickle():
    import pickle
    import tempfile

    fp = tempfile.TemporaryFile()
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True)
    pickle.dump(llama, fp)
    fp.seek(0)
    llama = pickle.load(fp)

    assert llama
    assert llama.ctx is not None

    text = b"Hello World"

    assert llama.detokenize(llama.tokenize(text)) == text


def test_utf8(mock_llama, monkeypatch):
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, logits_all=True)
    n_ctx = llama.n_ctx()
    n_vocab = llama.n_vocab()

    output_text = "😀"
    output_tokens = llama.tokenize(
        output_text.encode("utf-8"), add_bos=True, special=True
    )
    token_eos = llama.token_eos()
    n = 0

    def reset():
        nonlocal n
        llama.reset()
        n = 0

    ## Set up mock function
    def mock_decode(ctx: llama_cpp.llama_context_p, batch: llama_cpp.llama_batch):
        nonlocal n
        assert batch.n_tokens > 0
        assert llama.n_tokens == n
        n += batch.n_tokens
        return 0

    def mock_get_logits(*args, **kwargs):
        size = n_vocab * n_ctx
        return (llama_cpp.c_float * size)()

    def mock_sample(*args, **kwargs):
        nonlocal n
        if n <= len(output_tokens):
            return output_tokens[n - 1]
        else:
            return token_eos

    monkeypatch.setattr("llama_cpp.llama_cpp.llama_decode", mock_decode)
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_get_logits", mock_get_logits)
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_sample_token", mock_sample)

    ## Test basic completion with utf8 multibyte
    # mock_llama(llama, output_text)
    reset()
    completion = llama.create_completion("", max_tokens=4)
    assert completion["choices"][0]["text"] == output_text

    ## Test basic completion with incomplete utf8 multibyte
    # mock_llama(llama, output_text)
    reset()
    completion = llama.create_completion("", max_tokens=1)
    assert completion["choices"][0]["text"] == ""


def test_llama_server():
    from fastapi.testclient import TestClient
    from llama_cpp.server.app import create_app, Settings

    settings = Settings(
        model=MODEL,
        vocab_only=True,
    )
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }


def test_llama_cpp_version():
    assert llama_cpp.__version__
