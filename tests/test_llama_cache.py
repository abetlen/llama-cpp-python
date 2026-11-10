import random
from collections import OrderedDict

import numpy as np
import pytest

from llama_cpp import Llama, LlamaState
from llama_cpp.llama_cache import LlamaRAMCache


def state(tokens, size=1):
    return LlamaState(
        input_ids=np.array(tokens, dtype=np.intc),
        scores=np.empty((0, 0), dtype=np.single),
        n_tokens=len(tokens),
        llama_state=bytes(size),
        llama_state_size=size,
        seed=0,
    )


def scan(keys, query):
    best = None
    longest = 0
    for key in keys:
        length = Llama.longest_token_prefix(key, query)
        if length > longest:
            best, longest = key, length
    return best


@pytest.mark.parametrize(
    "keys,query,expected",
    [
        ([], (1,), None),
        ([()], (), None),
        ([(1, 2)], (), None),
        ([(1, 2)], (9,), None),
        ([(1, 2)], (1,), (1, 2)),
        ([(1, 2)], (1, 2, 3), (1, 2)),
        ([(1, 2, 3), (1, 2, 4)], (1, 2, 5), (1, 2, 3)),
        ([(1, 2), (1, 2, 3)], (1, 2, 4), (1, 2)),
        ([(1, 2, 3), (1, 2)], (1, 2, 4), (1, 2, 3)),
        ([(1, 2, 3), (1, 2, 4)], (1, 2, 4, 5), (1, 2, 4)),
        ([(1, 2, 3), (1, 2, 4)], (1, 9), (1, 2, 3)),
    ],
)
def test_ram_cache_longest_shared_prefix(keys, query, expected):
    cache = LlamaRAMCache()
    values = {}
    for key in keys:
        values[key] = state(key)
        cache[key] = values[key]
    assert cache._find_longest_prefix_key(query) == expected
    assert (list(query) in cache) == (expected is not None)
    if expected is None:
        with pytest.raises(KeyError):
            cache[list(query)]
    else:
        assert cache[np.array(query, dtype=np.intc)] is values[expected]


def test_ram_cache_ties_follow_lru_order():
    cache = LlamaRAMCache()
    a, b, c = (1, 2, 3), (1, 2, 4), (1, 5)
    for key in (a, b, c):
        cache[key] = state(key)
    assert cache._find_longest_prefix_key((1, 2)) == a
    cache[a]
    assert cache._find_longest_prefix_key((1, 2)) == b
    assert cache._find_longest_prefix_key((1,)) == b
    cache[b] = state(b)
    assert cache._find_longest_prefix_key((1,)) == c
    assert cache._find_longest_prefix_key((1, 2)) == a


def test_ram_cache_eviction_removes_obsolete_prefixes():
    cache = LlamaRAMCache(capacity_bytes=3)
    for key in [(1, 2, 3), (1, 2, 4), (1, 2), (9,)]:
        cache[key] = state(key)
    assert (1, 2, 3) not in cache.cache_state
    assert cache._find_longest_prefix_key((1, 2, 3)) == (1, 2, 4)
    cache[(8,)] = state((8,), size=2)
    assert list(cache.cache_state) == [(9,), (8,)]
    assert (1,) not in cache
    cache[(7,)] = state((7,), size=4)
    assert not cache.cache_state
    assert (7,) not in cache
    cache[(1,)] = state((1,))
    assert (1, 2) in cache


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 280])
def test_ram_cache_matches_scan_during_updates_and_evictions(seed):
    rng = random.Random(seed)
    cache = LlamaRAMCache(capacity_bytes=16)
    reference = OrderedDict()
    for _ in range(3000):
        key = tuple(rng.randrange(5) for _ in range(rng.randrange(12)))
        operation = rng.randrange(3)
        if operation == 0:
            value = state(key, size=rng.randrange(1, 5))
            if key in reference:
                del reference[key]
            reference[key] = value
            while sum(v.llama_state_size for v in reference.values()) > 16:
                reference.popitem(last=False)
            cache[key] = value
        else:
            expected = scan(reference, key)
            assert cache._find_longest_prefix_key(key) == expected
            assert (key in cache) == (expected is not None)
            if operation == 1 and expected is not None:
                assert cache[key] is reference[expected]
                reference.move_to_end(expected)
        assert list(cache.cache_state) == list(reference)
        assert cache.cache_size == sum(v.llama_state_size for v in reference.values())


def test_ram_cache_long_shared_edge_with_bounded_index():
    cache = LlamaRAMCache()
    prefix = tuple(range(10000))
    for suffix in range(32):
        key = prefix + (suffix,)
        cache[key] = state(key)
    assert cache._find_longest_prefix_key(prefix + (31, 99)) == prefix + (31,)


def test_ram_cache_nested_prefixes_survive_parent_eviction_and_replacement():
    cache = LlamaRAMCache(capacity_bytes=64)
    keys = [tuple(range(128 + n)) for n in range(64)]
    for key in keys:
        cache[key] = state(key)
    for key in keys[::2]:
        cache[key]
    for n in range(128):
        key = (999, n)
        cache[key] = state(key)
        for query in (keys[n % 64], (999,), (1234,)):
            assert cache._find_longest_prefix_key(query) == scan(
                cache.cache_state, query
            )

    cache._prefix_index.sync()
    assert len(cache._prefix_index.keys) == len(cache.cache_state)
    assert len(cache._prefix_index.ranks) == len(cache.cache_state)


def test_ram_cache_public_mapping_mutations_remain_visible():
    cache = LlamaRAMCache()
    a, b = (1, 2), (1, 3)
    cache[a] = state(a)
    cache[b] = state(b)
    assert cache._find_longest_prefix_key((1,)) == a
    cache.cache_state.move_to_end(a)
    assert cache._find_longest_prefix_key((1,)) == b
    cache.cache_state.move_to_end(a, last=False)
    assert cache._find_longest_prefix_key((1,)) == a
    cache.cache_state.pop(a)
    assert cache._find_longest_prefix_key((1,)) == b
    cache.cache_state.clear()
    assert (1,) not in cache
    with pytest.raises(KeyError):
        cache[(1,)]
    cache.cache_state = OrderedDict([(a, state(a))])
    assert cache[(1,)] is cache.cache_state[a]


def test_ram_cache_replacing_public_mapping_rebuilds_index():
    cache = LlamaRAMCache()
    for key in ((1, 2), (1, 3)):
        cache[key] = state(key)
    assert (1,) in cache
    a, b = (9, 3), (9, 2)
    cache.cache_state = OrderedDict([(a, state(a)), (b, state(b))])
    assert cache._find_longest_prefix_key((9,)) == a
    assert (1,) not in cache
    cache.cache_state.clear()
    assert (9,) not in cache
    assert not cache._prefix_index.order
    assert not cache._prefix_index.keys


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_ram_cache_numpy_token_dtype_boundary(dtype):
    cache = LlamaRAMCache()
    boundary = np.iinfo(dtype).max
    a, b = (int(boundary), 1), (int(boundary), 2)
    cache[a] = state(())
    cache[b] = state(())
    query = np.array([boundary, 3], dtype=dtype)
    assert cache._find_longest_prefix_key(tuple(query)) == a
    assert cache[query] is cache.cache_state[a]
    assert cache._find_longest_prefix_key(tuple(query)) == b
