import ctypes
from types import SimpleNamespace

import pytest

from llama_cpp import llama as llama_module


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_batch,n_seq_max", [(32, 4), (4, 4), (32, 1)])
def test_unpooled_embedding_offsets(monkeypatch, normalize, n_batch, n_seq_max):
    """Each sequence must read its own token rows from the flat native buffer."""
    sequences = []
    buffers = []
    tokens_by_text = {b"first": [1, 2], b"second": [3, 4], b"third": [5]}
    n_embd = 3

    def add_sequence(tokens, seq_id, logits_all):
        assert seq_id == len(sequences)
        assert logits_all
        sequences.append(tokens)

    def decode(batch):
        values = [
            float(token * n_embd + dim)
            for tokens in sequences
            for token in tokens
            for dim in range(n_embd)
        ]
        buffers[:] = [(ctypes.c_float * len(values))(*values)]

    model = SimpleNamespace(
        n_embd=lambda: n_embd,
        n_batch=n_batch,
        context_params=SimpleNamespace(n_seq_max=n_seq_max, embeddings=True),
        pooling_type=lambda: llama_module.llama_cpp.LLAMA_POOLING_TYPE_NONE,
        verbose=False,
        tokenize=lambda text: tokens_by_text[text],
        _batch=SimpleNamespace(reset=sequences.clear, add_sequence=add_sequence),
        _ctx=SimpleNamespace(ctx=None, kv_cache_clear=lambda: None, decode=decode),
        reset=lambda: None,
    )
    monkeypatch.setattr(
        llama_module.llama_cpp,
        "llama_get_embeddings",
        lambda ctx: ctypes.cast(buffers[0], ctypes.POINTER(ctypes.c_float)),
    )
    output, count = llama_module.Llama.embed(
        model, ["first", "second", "third"], normalize=normalize, return_count=True
    )
    assert count == 5
    assert [len(sequence) for sequence in output] == [2, 2, 1]
    for sequence, tokens in zip(output, tokens_by_text.values()):
        for row, token in zip(sequence, tokens):
            expected = [float(token * n_embd + dim) for dim in range(n_embd)]
            if normalize:
                norm = sum(value * value for value in expected) ** 0.5
                expected = [value / norm for value in expected]
            assert row == pytest.approx(expected)
