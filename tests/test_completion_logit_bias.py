import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from llama_cpp import Llama, LogitsProcessorList


class GenerationBoundaryReached(Exception):
    pass


@pytest.fixture
def completion_boundary():
    """Exercise the public completion wrapper without native/model execution."""
    model = Llama.__new__(Llama)
    model._stack = contextlib.ExitStack()
    model._model = SimpleNamespace(
        token_cls=lambda: -1,
        token_sep=lambda: -1,
        add_bos_token=lambda: False,
        add_eos_token=lambda: False,
    )
    model.token_bos = lambda: 1
    model.token_eos = lambda: 2
    model.metadata = {}
    model.spm_infill = False
    model.model_path = "unused-model"
    model.verbose = False
    model._n_ctx = 16
    model.cache = None
    model._seed = 0
    captured = []

    def record_generation(tokens, **kwargs):
        captured.append(kwargs["logits_processor"])
        raise GenerationBoundaryReached

    model.generate = record_generation
    return model, captured


def capture_processors(boundary, processors, bias, stream=False):
    model, captured = boundary
    with pytest.raises(GenerationBoundaryReached):
        result = model.create_completion(
            [1, 3],
            logits_processor=processors,
            logit_bias=bias,
            max_tokens=1,
            stream=stream,
            seed=0,
        )
        if stream:
            next(result)
    return captured[-1]


@pytest.mark.parametrize("stream", [False, True])
def test_completion_combines_processors_and_logit_bias(completion_boundary, stream):
    calls = []

    def processor(tokens, scores):
        calls.append(tokens.tolist())
        return scores * 2

    processors = LogitsProcessorList([processor])
    combined = capture_processors(completion_boundary, processors, {1: 4.0}, stream)
    assert combined is not None
    result = combined(np.array([1, 3]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(result, [2.0, 8.0, 6.0])
    assert calls == [[1, 3]]
    assert processors == [processor]


def test_completion_does_not_accumulate_bias_in_reused_processors(completion_boundary):
    def processor(tokens, scores):
        return scores * 2

    processors = LogitsProcessorList([processor])
    first = capture_processors(completion_boundary, processors, {1: 4.0})
    second = capture_processors(completion_boundary, processors, {2: -3.0})
    assert first is not None and second is not None
    scores = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(first(np.array([1]), scores), [2.0, 8.0, 6.0])
    np.testing.assert_array_equal(second(np.array([1]), scores), [2.0, 4.0, 3.0])
    assert processors == [processor]


@pytest.mark.parametrize("bias", [{1: 4.0}, {}])
def test_completion_combines_empty_processor_list_with_bias(completion_boundary, bias):
    processors = LogitsProcessorList()
    combined = capture_processors(completion_boundary, processors, bias)
    assert combined is not None
    scores = np.array([1.0, 2.0, 3.0])
    expected = scores.copy()
    for token, value in bias.items():
        expected[token] += value
    np.testing.assert_array_equal(combined(np.array([1]), scores), expected)
    assert processors == []


def test_completion_processor_only_keeps_existing_behavior(completion_boundary):
    def processor(tokens, scores):
        return scores + 2

    processors = LogitsProcessorList([processor])
    forwarded = capture_processors(completion_boundary, processors, None)
    assert forwarded is processors
    np.testing.assert_array_equal(
        forwarded(np.array([1]), np.array([1.0, 2.0])), [3.0, 4.0]
    )


def test_completion_bias_only_keeps_existing_behavior(completion_boundary):
    forwarded = capture_processors(completion_boundary, None, {1: 4.0})
    scores = np.array([1.0, 2.0])
    np.testing.assert_array_equal(forwarded(np.array([1]), scores), [1.0, 6.0])
    np.testing.assert_array_equal(scores, [1.0, 2.0])


def test_completion_without_processors_or_bias(completion_boundary):
    assert capture_processors(completion_boundary, None, None) is None
