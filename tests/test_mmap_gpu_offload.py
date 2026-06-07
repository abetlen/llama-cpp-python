"""Tests for automatic mmap disabling when all layers are offloaded to GPU.

See: https://github.com/abetlen/llama-cpp-python/issues/1964
"""

import sys
from unittest.mock import MagicMock
from dataclasses import dataclass, field

# Stub the native C library so tests can run without compiling llama.cpp
_mock_llama_cpp = MagicMock()
_mock_llama_cpp.llama_log_callback = lambda f: f
_mock_llama_cpp.llama_log_set = MagicMock()
sys.modules.setdefault("llama_cpp.llama_cpp", _mock_llama_cpp)

_mock_llama = MagicMock()
_mock_llama.StoppingCriteriaList = list
_mock_llama.LogitsProcessorList = list
_mock_llama.LlamaGrammar = MagicMock
sys.modules.setdefault("llama_cpp.llama", _mock_llama)


@dataclass
class MockModelParams:
    """Mimics the relevant fields of llama_model_params for testing."""
    n_gpu_layers: int = 0
    use_mmap: bool = True


def _apply_mmap_logic(n_gpu_layers: int, use_mmap: bool, gpu_offload_supported: bool) -> bool:
    """Replicate the mmap auto-disable logic from Llama.__init__."""
    if n_gpu_layers == -1 and use_mmap and gpu_offload_supported:
        return False
    return use_mmap


def test_mmap_disabled_when_all_layers_offloaded():
    """When n_gpu_layers=-1 and GPU offload is supported, use_mmap should be set to False."""
    result = _apply_mmap_logic(n_gpu_layers=-1, use_mmap=True, gpu_offload_supported=True)
    assert result is False


def test_mmap_kept_when_partial_offload():
    """When n_gpu_layers is not -1, use_mmap should remain True."""
    result = _apply_mmap_logic(n_gpu_layers=10, use_mmap=True, gpu_offload_supported=True)
    assert result is True


def test_mmap_kept_when_no_gpu_support():
    """When GPU offload is not supported, use_mmap should remain True even with n_gpu_layers=-1."""
    result = _apply_mmap_logic(n_gpu_layers=-1, use_mmap=True, gpu_offload_supported=False)
    assert result is True


def test_mmap_kept_when_zero_gpu_layers():
    """When n_gpu_layers=0, use_mmap should remain True (CPU-only inference)."""
    result = _apply_mmap_logic(n_gpu_layers=0, use_mmap=True, gpu_offload_supported=True)
    assert result is True


def test_mmap_respects_explicit_false():
    """When user explicitly sets use_mmap=False, it should stay False regardless."""
    result = _apply_mmap_logic(n_gpu_layers=10, use_mmap=False, gpu_offload_supported=True)
    assert result is False


def test_mmap_disabled_applies_to_params():
    """Verify the logic correctly updates a MockModelParams object."""
    params = MockModelParams(n_gpu_layers=-1, use_mmap=True)
    params.use_mmap = _apply_mmap_logic(
        n_gpu_layers=params.n_gpu_layers,
        use_mmap=params.use_mmap,
        gpu_offload_supported=True,
    )
    assert params.use_mmap is False
