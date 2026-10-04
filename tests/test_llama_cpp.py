import pytest

import llama_cpp

# ctypes function objects built without paramflags bind arguments
# positionally only; the trailing `/` on these bindings just keeps the
# type stub honest about that (#2371).


def test_llama_sampler_init_dist_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_dist(seed=0)


def test_llama_sampler_init_top_k_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_top_k(k=40)


def test_llama_sampler_init_top_p_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_top_p(0.9, min_keep=1)


def test_llama_sampler_init_min_p_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_min_p(0.1, min_keep=1)


def test_llama_sampler_init_typical_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_typical(1.0, min_keep=1)


def test_llama_sampler_init_temp_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_temp(t=0.8)


def test_llama_sampler_init_temp_ext_rejects_keyword_args():
    with pytest.raises(TypeError):
        llama_cpp.llama_sampler_init_temp_ext(0.8, 0.0, exponent=1.0)


def test_llama_sampler_init_top_p_silently_ignores_extra_keyword():
    # Known limitation, not fixed here: with all positional slots filled,
    # ctypes has no parameter names to match an extra keyword against, so
    # it's silently ignored instead of raising.
    sampler = llama_cpp.llama_sampler_init_top_p(0.9, 1, bogus=999)
    assert sampler
    llama_cpp.llama_sampler_free(sampler)
