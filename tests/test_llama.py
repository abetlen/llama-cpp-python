import pytest
import llama_cpp

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama.gguf"


def test_llama_cpp_tokenization():
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    assert llama
    assert llama.ctx is not None

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


@pytest.mark.skip(reason="bug in tokenization where leading space is always inserted even if not after eos")
def test_llama_patch(monkeypatch):
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True)
    n_vocab = llama_cpp.llama_n_vocab(llama.ctx)

    ## Set up mock function
    def mock_eval(*args, **kwargs):
        return 0

    def mock_get_logits(*args, **kwargs):
        return (llama_cpp.c_float * n_vocab)(
            *[llama_cpp.c_float(0) for _ in range(n_vocab)]
        )

    monkeypatch.setattr("llama_cpp.llama_cpp.llama_eval", mock_eval)
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_get_logits", mock_get_logits)

    output_text = " jumps over the lazy dog."
    output_tokens = llama.tokenize(output_text.encode("utf-8"))
    token_eos = llama.token_eos()
    n = 0

    def mock_sample(*args, **kwargs):
        nonlocal n
        if n < len(output_tokens):
            n += 1
            return output_tokens[n - 1]
        else:
            return token_eos

    monkeypatch.setattr("llama_cpp.llama_cpp.llama_sample_token", mock_sample)

    text = "The quick brown fox"

    ## Test basic completion until eos
    n = 0  # reset
    completion = llama.create_completion(text, max_tokens=20)
    assert completion["choices"][0]["text"] == output_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until eos
    n = 0  # reset
    chunks = llama.create_completion(text, max_tokens=20, stream=True)
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == output_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until stop sequence
    n = 0  # reset
    completion = llama.create_completion(text, max_tokens=20, stop=["lazy"])
    assert completion["choices"][0]["text"] == " jumps over the "
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until stop sequence
    n = 0  # reset
    chunks = llama.create_completion(text, max_tokens=20, stream=True, stop=["lazy"])
    assert (
        "".join(chunk["choices"][0]["text"] for chunk in chunks) == " jumps over the "
    )
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until length
    n = 0  # reset
    completion = llama.create_completion(text, max_tokens=2)
    assert completion["choices"][0]["text"] == " j"
    assert completion["choices"][0]["finish_reason"] == "length"

    ## Test streaming completion until length
    n = 0  # reset
    chunks = llama.create_completion(text, max_tokens=2, stream=True)
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == " j"
    assert completion["choices"][0]["finish_reason"] == "length"


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


def test_utf8(monkeypatch):
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True)
    n_vocab = llama.n_vocab()

    ## Set up mock function
    def mock_eval(*args, **kwargs):
        return 0

    def mock_get_logits(*args, **kwargs):
        return (llama_cpp.c_float * n_vocab)(
            *[llama_cpp.c_float(0) for _ in range(n_vocab)]
        )

    monkeypatch.setattr("llama_cpp.llama_cpp.llama_eval", mock_eval)
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_get_logits", mock_get_logits)

    output_text = "😀"
    output_tokens = llama.tokenize(output_text.encode("utf-8"))
    token_eos = llama.token_eos()
    n = 0

    def mock_sample(*args, **kwargs):
        nonlocal n
        if n < len(output_tokens):
            n += 1
            return output_tokens[n - 1]
        else:
            return token_eos

    monkeypatch.setattr("llama_cpp.llama_cpp.llama_sample_token", mock_sample)

    ## Test basic completion with utf8 multibyte
    n = 0  # reset
    completion = llama.create_completion("", max_tokens=4)
    assert completion["choices"][0]["text"] == output_text

    ## Test basic completion with incomplete utf8 multibyte
    n = 0  # reset
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
