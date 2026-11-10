from bisect import bisect_left
import sys
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Sequence,
    Tuple,
)
from collections import OrderedDict

import diskcache

import llama_cpp.llama

from .llama_types import *


class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    @abstractmethod
    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Sequence[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(
        self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"
    ) -> None:
        raise NotImplementedError


class _LlamaCachePrefixIndex:
    """Linear-storage sorted-key index with cache-order tie breaking."""

    def __init__(self, cache_state):
        self.cache_state = cache_state
        self.order = ()
        self.keys = []
        self.ranks = []
        self.clock = 0

    def sync(self):
        order = tuple(self.cache_state)
        if order != self.order:
            pairs = sorted((key, rank) for rank, key in enumerate(order) if key)
            self.keys = [key for key, rank in pairs]
            self.ranks = [rank for key, rank in pairs]
            self.order = order
            self.clock = len(order)

    def find(self, query):
        self.sync()
        # A lexicographic neighbor attains the maximum shared prefix.
        index = bisect_left(self.keys, query)
        best = 0
        for position in (index - 1, index):
            if 0 <= position < len(self.keys):
                length = 0
                for a, b in zip(self.keys[position], query):
                    if a != b:
                        break
                    length += 1
                best = max(best, length)
        if not best:
            return None
        prefix = query[:best]
        start = bisect_left(self.keys, prefix)
        stop = bisect_left(self.keys, prefix[:-1] + (int(prefix[-1]) + 1,))
        # All keys in this interval tie; preserve the original LRU winner.
        winner = min(range(start, stop), key=self.ranks.__getitem__)
        return self.keys[winner]

    def touch(self, key):
        position = bisect_left(self.keys, key)
        self.ranks[position] = self.clock
        self.clock += 1
        self.order = tuple(self.cache_state)


class LlamaRAMCache(BaseLlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[Tuple[int, ...], "llama_cpp.llama.LlamaState"] = (
            OrderedDict()
        )
        self._prefix_index = _LlamaCachePrefixIndex(self.cache_state)

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        if len(self.cache_state) <= 1:
            if (
                self._prefix_index.order
                or self._prefix_index.cache_state is not self.cache_state
            ):
                self._prefix_index = _LlamaCachePrefixIndex(self.cache_state)
            for candidate in self.cache_state:
                if llama_cpp.llama.Llama.longest_token_prefix(candidate, key):
                    return candidate
            return None
        if self._prefix_index.cache_state is not self.cache_state:
            self._prefix_index = _LlamaCachePrefixIndex(self.cache_state)
        return self._prefix_index.find(key)

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        if len(self.cache_state) > 1:
            self._prefix_index.touch(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]
        self.cache_state[key] = value
        while self.cache_size > self.capacity_bytes and len(self.cache_state) > 0:
            self.cache_state.popitem(last=False)


# Alias for backwards compatibility
LlamaCache = LlamaRAMCache


class LlamaDiskCache(BaseLlamaCache):
    """Cache for a llama.cpp model using disk."""

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        super().__init__(capacity_bytes)
        self.cache = diskcache.Cache(cache_dir)

    @property
    def cache_size(self):
        return int(self.cache.volume())  # type: ignore

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        for k in self.cache.iterkeys():  # type: ignore
            prefix_len = llama_cpp.llama.Llama.longest_token_prefix(k, key)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k  # type: ignore
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value: "llama_cpp.llama.LlamaState" = self.cache.pop(_key)  # type: ignore
        # NOTE: This puts an integer as key in cache, which breaks,
        # Llama.longest_token_prefix(k, key) above since k is not a tuple of ints/tokens
        # self.cache.push(_key, side="front")  # type: ignore
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
        key = tuple(key)
        if key in self.cache:
            print("LlamaDiskCache.__setitem__: delete", file=sys.stderr)
            del self.cache[key]
        self.cache[key] = value
        print("LlamaDiskCache.__setitem__: set", file=sys.stderr)
        while self.cache_size > self.capacity_bytes and len(self.cache) > 0:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        print("LlamaDiskCache.__setitem__: trim", file=sys.stderr)
