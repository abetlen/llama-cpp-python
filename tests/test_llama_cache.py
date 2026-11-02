import diskcache
import numpy as np
import pytest

from llama_cpp import LlamaDiskCache, LlamaState


def make_state(tokens, size):
    return LlamaState(
        scores=np.empty((0, 0), dtype=np.single),
        input_ids=np.array(tokens, dtype=np.intc),
        n_tokens=len(tokens),
        llama_state=b"\0" * size,
        llama_state_size=size,
        seed=0,
    )


def test_disk_cache_honors_capacity_when_reopening(tmp_path):
    cache_dir = str(tmp_path / "cache")
    # A previously configured diskcache limit must not silently override the
    # capacity requested by LlamaDiskCache. Keep both limits small for the test.
    with diskcache.Cache(cache_dir, size_limit=64 << 10):
        pass

    cache = LlamaDiskCache(cache_dir, capacity_bytes=512 << 10)
    try:
        state = make_state([1, 2, 3], 128 << 10)
        cache[[1, 2, 3]] = state

        assert [1, 2, 3] in cache
        restored = cache[[1, 2, 3]]
        assert restored.llama_state == state.llama_state
        np.testing.assert_array_equal(restored.input_ids, state.input_ids)
        assert restored.llama_state_size == state.llama_state_size
    finally:
        cache.cache.close()


def test_disk_cache_still_rejects_states_larger_than_capacity(tmp_path):
    cache = LlamaDiskCache(str(tmp_path / "cache"), capacity_bytes=64 << 10)
    try:
        cache[[1, 2, 3]] = make_state([1, 2, 3], 128 << 10)

        assert [1, 2, 3] not in cache
        with pytest.raises(KeyError):
            cache[[1, 2, 3]]
    finally:
        cache.cache.close()


def test_disk_cache_retains_recent_states_when_full(tmp_path):
    cache = LlamaDiskCache(str(tmp_path / "cache"), capacity_bytes=256 << 10)
    try:
        for i in range(4):
            tokens = [i, i + 10]
            cache[tokens] = make_state(tokens, 64 << 10)

        # Three states fit. Evict only the oldest state rather than letting
        # diskcache's batch culling remove all four, including the newest.
        assert list(cache.cache) == [(1, 11), (2, 12), (3, 13)]
        assert cache.cache_size <= cache.capacity_bytes
        restored = cache[[3, 13]]
        np.testing.assert_array_equal(restored.input_ids, [3, 13])
    finally:
        cache.cache.close()
