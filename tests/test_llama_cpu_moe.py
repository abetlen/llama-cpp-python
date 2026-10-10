import pytest
import llama_cpp
from llama_cpp.llama import _build_cpu_moe_patterns


def test_build_cpu_moe_patterns_disabled():
    patterns = _build_cpu_moe_patterns(cpu_moe=False, n_cpu_moe=0)
    assert patterns == []


def test_build_cpu_moe_patterns_all():
    patterns = _build_cpu_moe_patterns(cpu_moe=True, n_cpu_moe=0)
    assert len(patterns) == 1
    assert patterns[0] == llama_cpp.llama._LLM_FFN_EXPS_REGEX


def test_build_cpu_moe_patterns_n_layers():
    patterns = _build_cpu_moe_patterns(cpu_moe=False, n_cpu_moe=3)
    assert len(patterns) == 3
    assert patterns[0] == rb"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    assert patterns[1] == rb"blk\.1\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    assert patterns[2] == rb"blk\.2\.ffn_(up|down|gate|gate_up)_(ch|)exps"


def test_build_cpu_moe_patterns_cpu_moe_overrides_n():
    patterns = _build_cpu_moe_patterns(cpu_moe=True, n_cpu_moe=3)
    assert len(patterns) == 1
    assert patterns[0] == llama_cpp.llama._LLM_FFN_EXPS_REGEX


def test_llama_accepts_cpu_moe_params():
    model = llama_cpp.Llama(
        model_path="./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf",
        vocab_only=True,
        verbose=False,
        cpu_moe=False,
        n_cpu_moe=0,
    )
    assert model is not None
