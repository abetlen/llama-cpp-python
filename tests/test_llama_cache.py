import numpy as np
import pytest

from llama_cpp import LlamaRAMCache, LlamaState


def test_ram_cache_counts_snapshot_buffers():
    state = LlamaState(
        input_ids=np.array([1, 2, 3, 4], dtype=np.intc),
        scores=np.zeros((4, 8), dtype=np.single),
        n_tokens=4,
        llama_state=b"state",
        llama_state_size=5,
        seed=0,
    )
    cache = LlamaRAMCache(capacity_bytes=149)
    cache[[1, 2, 3, 4]] = state

    # 5 native bytes, 16 token bytes, and 128 score bytes.
    assert cache.cache_size == 149
    assert cache[[1, 2, 3, 4]] is state


def _state(token, columns=8):
    return LlamaState(
        input_ids=np.array([token], dtype=np.intc),
        scores=np.zeros((1, columns), dtype=np.single),
        n_tokens=1,
        llama_state=b"state",
        llama_state_size=5,
        seed=0,
    )


@pytest.mark.parametrize("capacity, retained", [(40, False), (41, True), (42, True)])
def test_ram_cache_snapshot_capacity_boundary(capacity, retained):
    cache = LlamaRAMCache(capacity_bytes=capacity)
    cache[[1]] = _state(1)

    assert ([1] in cache) is retained
    assert cache.cache_size == (41 if retained else 0)


def test_ram_cache_evicts_least_recently_used_snapshot():
    cache = LlamaRAMCache(capacity_bytes=82)
    first, second, third = (_state(token) for token in (1, 2, 3))
    cache[[1]] = first
    cache[[2]] = second
    assert cache[[1]] is first

    cache[[3]] = third

    assert [2] not in cache
    assert cache[[1]] is first
    assert cache[[3]] is third
    assert cache.cache_size == 82


def test_ram_cache_replacement_accounts_for_new_buffers():
    cache = LlamaRAMCache(capacity_bytes=82)
    cache[[1]] = _state(1)
    cache[[2]] = _state(2)
    replacement = _state(1, columns=16)

    cache[[1]] = replacement

    assert [2] not in cache
    assert cache[[1]] is replacement
    assert cache.cache_size == 73


@pytest.mark.parametrize("tokens", [[1], [1, 2]])
def test_ram_cache_reuses_snapshot_within_capacity(tokens):
    cache = LlamaRAMCache(capacity_bytes=41)
    state = _state(1)
    cache[[1]] = state

    assert cache[tokens] is state
