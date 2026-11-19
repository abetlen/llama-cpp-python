import numpy as np
import pytest

from llama_cpp import LlamaState
from llama_cpp.llama_cache import LlamaDiskCache, LlamaRAMCache


@pytest.fixture(params=[LlamaRAMCache, LlamaDiskCache])
def cache(request, tmp_path):
    if request.param is LlamaDiskCache:
        cache = request.param(cache_dir=str(tmp_path / "cache"))
    else:
        cache = request.param()
    try:
        yield cache
    finally:
        if isinstance(cache, LlamaDiskCache):
            cache.cache.close()


@pytest.fixture
def state():
    return LlamaState(
        input_ids=np.array([1, 2, 3], dtype=np.intc),
        scores=np.zeros((3, 4), dtype=np.single),
        n_tokens=3,
        llama_state=b"saved model state",
        llama_state_size=17,
        seed=42,
    )


@pytest.mark.parametrize("query", [(1, 2, 3), (1, 2, 3, 4), (1, 2), (1, 2, 9)])
def test_cache_hits_remain_available(cache, state, query):
    cache[[1, 2, 3]] = state

    for _ in range(3):
        assert query in cache
        result = cache[query]
        np.testing.assert_array_equal(result.input_ids, state.input_ids)
        np.testing.assert_array_equal(result.scores, state.scores)
        assert result.n_tokens == state.n_tokens
        assert result.llama_state == state.llama_state
        assert result.llama_state_size == state.llama_state_size
        assert result.seed == state.seed
        assert [1, 2, 3] in cache


@pytest.mark.parametrize("query", [(), (9,)])
def test_cache_misses_leave_saved_states_available(cache, state, query):
    assert query not in cache
    with pytest.raises(KeyError):
        cache[query]

    cache[[1, 2, 3]] = state
    assert query not in cache
    with pytest.raises(KeyError):
        cache[query]
    assert [1, 2, 3] in cache


def test_disk_cache_hit_survives_reopening(tmp_path, state):
    directory = str(tmp_path / "cache")
    cache = LlamaDiskCache(cache_dir=directory)
    try:
        cache[[1, 2, 3]] = state
        assert cache[[1, 2, 3, 4]].llama_state == state.llama_state
    finally:
        cache.cache.close()

    reopened = LlamaDiskCache(cache_dir=directory)
    try:
        assert [1, 2, 3] in reopened
        assert reopened[[1, 2, 9]].llama_state == state.llama_state
        assert [1, 2, 3] in reopened
    finally:
        reopened.cache.close()
