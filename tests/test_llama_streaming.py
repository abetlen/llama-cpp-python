from contextlib import ExitStack
from types import SimpleNamespace

import pytest

import llama_cpp
import llama_cpp.llama_cpp as bindings


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        ([b"\xc3", b"\xa9STOP"], "é"),
        ([b"helloSTOP"], "hello"),
        ([b"STOP"], ""),
    ],
)
def test_stream_flushes_utf8_before_stop(monkeypatch, pieces, expected):
    llama = llama_cpp.Llama.__new__(llama_cpp.Llama)
    llama._stack = ExitStack()
    llama._model = SimpleNamespace(
        vocab=None,
        token_cls=lambda: -1,
        token_sep=lambda: -1,
    )
    llama.metadata = {}
    llama.spm_infill = False
    llama.model_path = "synthetic"
    llama.verbose = False
    llama._n_ctx = 32
    llama.cache = None
    llama.token_bos = lambda: 1
    llama.token_eos = lambda: 2
    llama.set_seed = lambda seed: None
    token_ids = list(range(3, 3 + len(pieces)))
    token_bytes = dict(zip(token_ids, pieces))
    llama.generate = lambda *_args, **_kwargs: iter(token_ids)
    llama.detokenize = lambda tokens, prev_tokens=None: b"".join(
        token_bytes[token] for token in tokens
    )
    monkeypatch.setattr(bindings, "llama_vocab_is_eog", lambda _vocab, _token: False)

    chunks = list(llama.create_completion([9], stop=["STOP"], stream=True, seed=1))
    choices = [chunk["choices"][0] for chunk in chunks]

    assert "".join(choice["text"] for choice in choices) == expected
    assert choices[-1]["finish_reason"] == "stop"
    assert all(choice["finish_reason"] is None for choice in choices[:-1])
