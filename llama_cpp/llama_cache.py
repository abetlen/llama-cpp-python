import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Optional,
    Sequence,
    Tuple,
)
from collections import OrderedDict

import diskcache

import llama_cpp.llama

from .llama_types import *

@dataclass(eq=True, frozen=True)
class LlamaCacheKey:
    """A key in a LlamaCache. Stores tokens to key by. Also stores
    information about active LoRA adapters, because we need different
    cached values for different active adapters, even for the same tokens."""
    active_lora_adapters: Tuple[Tuple[str, float], ...]
    tokens: Tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.tokens, tuple):
            raise ValueError("tokens must be a tuple")

class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    def _convert_to_cache_key(self, key: Union[Sequence[int], LlamaCacheKey]) -> LlamaCacheKey:
        """Convert raw tokens to a key if needed"""
        if type(key) == LlamaCacheKey:
            return key
        else:
            return LlamaCacheKey(active_lora_adapters=(), tokens=tuple(key))

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: LlamaCacheKey,
    ) -> Optional[LlamaCacheKey]:
        """Find the cached key with the longest matching token prefix. A match also requires that the active
        LoRA adapters match exactly.

        Args:
            key (LlamaCacheKey): The key to find a prefix match for.

        Returns:
            Optional[LlamaCacheKey]: The key with the longest matching prefix, or None if no match found.
        """
        pass

    @abstractmethod
    def __getitem__(self, key: Union[Sequence[int], LlamaCacheKey]) -> "llama_cpp.llama.LlamaState":
        """Retrieve a cached state by key, matching on the longest common token prefix. A match also requires
        that the active LoRA adapters match exactly.

        Args:
            key: Key to look up. Raw token sequences are supported for backwards compatibility
                 and assume no active LoRA adapters.

        Returns:
            llama_cpp.llama.LlamaState: The cached state for the entry sharing the longest token prefix.

        Raises:
            KeyError: If no prefix match is found.
        """
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Union[Sequence[int], LlamaCacheKey]) -> bool:
        """Check if any cached key shares a token prefix with the given key.

        Args:
            key: Key to look up. Raw token sequences are supported for backwards compatibility
                 and assume no active LoRA adapters.

        Returns:
            bool: True if any cached key shares a token prefix with this key.
        """
        raise NotImplementedError

    @abstractmethod
    def __setitem__(
        self, key: Union[Sequence[int], LlamaCacheKey], value: "llama_cpp.llama.LlamaState"
    ) -> None:
        """Store a state keyed on its tokens and information about active LoRA adapters.

        Args:
            key: Key to store. Raw token sequences are supported for backwards compatibility
                and assume no active LoRA adapters
            value: The state to cache
        """
        raise NotImplementedError

class LlamaRAMCache(BaseLlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[
            LlamaCacheKey, "llama_cpp.llama.LlamaState"
        ] = OrderedDict()

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    def _find_longest_prefix_key(
        self,
        key: LlamaCacheKey,
    ) -> Optional[LlamaCacheKey]:
        min_len = 0
        min_key: Optional[LlamaCacheKey] = None
        for k in self.cache_state.keys():
            if k.active_lora_adapters != key.active_lora_adapters: continue
            if len(k.tokens) < min_len: continue # Optimization
            prefix_len = llama_cpp.llama.Llama.longest_token_prefix(k.tokens, key.tokens)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Union[Sequence[int], LlamaCacheKey]) -> "llama_cpp.llama.LlamaState":
        key = self._convert_to_cache_key(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Union[Sequence[int], LlamaCacheKey]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Union[Sequence[int], LlamaCacheKey], value: "llama_cpp.llama.LlamaState"):
        key = self._convert_to_cache_key(key)
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
        key: LlamaCacheKey,
    ) -> Optional[LlamaCacheKey]:
        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        for k in self.cache.iterkeys():  # type: ignore
            if not isinstance(k, LlamaCacheKey):
                print("LlamaDiskCache: Disk cache keys must be LlamaCacheKey objects: skipping")
                continue
            if k.active_lora_adapters != key.active_lora_adapters: continue
            if len(k.tokens) < min_len: continue # Optimization
            prefix_len = llama_cpp.llama.Llama.longest_token_prefix(k.tokens, key.tokens)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Union[Sequence[int], LlamaCacheKey]) -> "llama_cpp.llama.LlamaState":
        key = self._convert_to_cache_key(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value: "llama_cpp.llama.LlamaState" = self.cache.pop(_key)  # type: ignore
        # NOTE: This puts an integer as key in cache, which breaks,
        # Llama.longest_token_prefix(k, key) above since k is not a tuple of ints/tokens
        # self.cache.push(_key, side="front")  # type: ignore
        return value

    def __contains__(self, key: Union[Sequence[int], LlamaCacheKey]) -> bool:
        return self._find_longest_prefix_key(self._convert_to_cache_key(key)) is not None

    def __setitem__(self, key: Union[Sequence[int], LlamaCacheKey], value: "llama_cpp.llama.LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
        key = self._convert_to_cache_key(key)
        if key in self.cache:
            print("LlamaDiskCache.__setitem__: delete", file=sys.stderr)
            del self.cache[key]
        self.cache[key] = value
        print("LlamaDiskCache.__setitem__: set", file=sys.stderr)
        while self.cache_size > self.capacity_bytes and len(self.cache) > 0:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        print("LlamaDiskCache.__setitem__: trim", file=sys.stderr)
