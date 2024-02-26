import ctypes

import numpy as np
import pytest
from scipy.special import log_softmax

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
        n_ctx = llama.n_ctx()
        n_vocab = llama.n_vocab()
        output_tokens = llama.tokenize(
            output_text.encode("utf-8"), add_bos=True, special=True
        )
        logits = (ctypes.c_float * (n_vocab * n_ctx))(-100.0)
        for i in range(n_ctx):
            output_idx = i + 1  # logits for first tokens predict second token
            if output_idx < len(output_tokens):
                logits[i * n_vocab + output_tokens[output_idx]] = 100.0
            else:
                logits[i * n_vocab + llama.token_eos()] = 100.0
        n = 0
        last_n_tokens = 0

        def mock_decode(ctx: llama_cpp.llama_context_p, batch: llama_cpp.llama_batch):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            assert batch.n_tokens > 0, "no tokens in batch"
            assert all(
                batch.n_seq_id[i] == 1 for i in range(batch.n_tokens)
            ), "n_seq >1 not supported by mock_llama"
            assert all(
                batch.seq_id[i][0] == 0 for i in range(batch.n_tokens)
            ), "n_seq >1 not supported by mock_llama"
            assert batch.logits[
                batch.n_tokens - 1
            ], "logits not allocated for last token"
            # Update the mock context state
            nonlocal n
            nonlocal last_n_tokens
            n = max(batch.pos[i] for i in range(batch.n_tokens)) + 1
            last_n_tokens = batch.n_tokens
            return 0

        def mock_get_logits(ctx: llama_cpp.llama_context_p):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            assert n > 0, "mock_llama_decode not called"
            assert last_n_tokens > 0, "mock_llama_decode not called"
            # Return view of logits for last_n_tokens
            return (ctypes.c_float * (last_n_tokens * n_vocab)).from_address(
                ctypes.addressof(logits)
                + (n - last_n_tokens) * n_vocab * ctypes.sizeof(ctypes.c_float)
            )

        monkeypatch.setattr("llama_cpp.llama_cpp.llama_decode", mock_decode)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_get_logits", mock_get_logits)

        def mock_kv_cache_clear(ctx: llama_cpp.llama_context_p):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            return

        def mock_kv_cache_seq_rm(
            ctx: llama_cpp.llama_context_p,
            seq_id: llama_cpp.llama_seq_id,
            pos0: llama_cpp.llama_pos,
            pos1: llama_cpp.llama_pos,
        ):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            return

        def mock_kv_cache_seq_cp(
            ctx: llama_cpp.llama_context_p,
            seq_id_src: llama_cpp.llama_seq_id,
            seq_id_dst: llama_cpp.llama_seq_id,
            pos0: llama_cpp.llama_pos,
            pos1: llama_cpp.llama_pos,
        ):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            return
    
        def mock_kv_cache_seq_keep(
            ctx: llama_cpp.llama_context_p,
            seq_id: llama_cpp.llama_seq_id,
        ):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            return

        def mock_kv_cache_seq_add(
            ctx: llama_cpp.llama_context_p,
            seq_id: llama_cpp.llama_seq_id,
            pos0: llama_cpp.llama_pos,
            pos1: llama_cpp.llama_pos,
        ):
            # Test some basic invariants of this mocking technique
            assert ctx == llama._ctx.ctx, "context does not match mock_llama"
            return

        monkeypatch.setattr("llama_cpp.llama_cpp.llama_kv_cache_clear", mock_kv_cache_clear)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_kv_cache_seq_rm", mock_kv_cache_seq_rm)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_kv_cache_seq_cp", mock_kv_cache_seq_cp)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_kv_cache_seq_keep", mock_kv_cache_seq_keep)
        monkeypatch.setattr("llama_cpp.llama_cpp.llama_kv_cache_seq_add", mock_kv_cache_seq_add)

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


def test_utf8(mock_llama):
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, logits_all=True)

    output_text = "😀"

    ## Test basic completion with utf8 multibyte
    mock_llama(llama, output_text)
    completion = llama.create_completion("", max_tokens=4)
    assert completion["choices"][0]["text"] == output_text

    ## Test basic completion with incomplete utf8 multibyte
    mock_llama(llama, output_text)
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


@pytest.mark.parametrize(
    "size_and_axis",
    [
        ((32_000,), -1),  # last token's next-token logits
        ((10, 32_000), -1),  # many tokens' next-token logits, or batch of last tokens
        ((4, 10, 32_000), -1),  # batch of texts
    ],
)
@pytest.mark.parametrize("convert_to_list", [True, False])
def test_logits_to_logprobs(size_and_axis, convert_to_list: bool, atol: float = 1e-7):
    size, axis = size_and_axis
    logits: np.ndarray = -np.random.uniform(low=0, high=60, size=size)
    logits = logits.astype(np.single)
    if convert_to_list:
        # Currently, logits are converted from arrays to lists. This may change soon
        logits = logits.tolist()
    log_probs = llama_cpp.Llama.logits_to_logprobs(logits, axis=axis)
    log_probs_correct = log_softmax(logits, axis=axis)
    assert log_probs.dtype == np.single
    assert log_probs.shape == size
    assert np.allclose(log_probs, log_probs_correct, atol=atol)


def test_llama_cpp_version():
    assert llama_cpp.__version__
