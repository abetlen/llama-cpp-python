"""Exercise native embedding batch limits in subprocesses (regressions can abort)."""

import contextlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download
import numpy as np
import pytest

import llama_cpp


@pytest.fixture(scope="module")
def long_embedding_model_path():
    return hf_hub_download(
        "lm-kit/bge-m3-gguf",
        "bge-m3-F16.gguf",
        revision="9379ce497e8814b200f2dc0d18eb4045426dcb8c",
    )


def _check_embedding_case(model_path, case, params):
    with contextlib.closing(
        llama_cpp.Llama(
            model_path,
            n_gpu_layers=0,
            n_threads=2,
            n_threads_batch=2,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,
            verbose=False,
            **params,
        )
    ) as model:
        assert not llama_cpp.llama_get_causal_attn(model._ctx.ctx)
        native_ubatch = llama_cpp.llama_n_ubatch(model._ctx.ctx)
        if case == "explicit-small":
            assert native_ubatch == 256
            text = "a " * 255
            assert len(model.tokenize(text.encode())) == 257
            with pytest.raises(ValueError, match="physical batch size"):
                model.embed(text)
            return
        if case == "explicit-small-multi":
            assert native_ubatch == 256
            inputs = ["a " * 126, "a " * 127]
            counts = [128, 129]
        elif case == "multi":
            assert native_ubatch == 513
            inputs = ["a " * 254, "a " * 255]
            counts = [256, 257]
        elif case == "short":
            assert native_ubatch == min(params["n_ctx"], params["n_batch"])
            inputs = ["a " * 3]
            counts = [5]
        else:
            assert native_ubatch == 513
            inputs = ["a " * 511]
            counts = [513]
        assert [len(model.tokenize(text.encode())) for text in inputs] == counts
        embeddings, count = model.embed(inputs, truncate=False, return_count=True)
        assert count == sum(counts)
        assert len(embeddings) == len(inputs)
        for embedding, tokens in zip(embeddings, counts):
            array = np.asarray(embedding)
            assert array.shape == (tokens, model.n_embd())
            assert np.isfinite(array).all()


@pytest.mark.parametrize(
    "case,params",
    [
        ("long", {"n_ctx": 1024, "n_batch": 513}),
        ("long", {"n_ctx": 0, "n_batch": 513}),
        ("long", {"n_ctx": 1024, "n_batch": 513, "n_ubatch": 513}),
        ("long", {"n_ctx": 1024, "n_batch": 513, "n_ubatch": 0}),
        ("multi", {"n_ctx": 1024, "n_batch": 513}),
        ("explicit-small", {"n_ctx": 1024, "n_batch": 513, "n_ubatch": 256}),
        ("explicit-small-multi", {"n_ctx": 1024, "n_batch": 513, "n_ubatch": 256}),
        ("short", {"n_ctx": 256, "n_batch": 513}),
        ("short", {"n_ctx": 512, "n_batch": 512}),
    ],
)
def test_native_embedding_batch_limits(long_embedding_model_path, case, params):
    code = (
        "import json, runpy, sys\n"
        "if sys.platform != 'win32':\n"
        "    import resource\n"
        "    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "runpy.run_path(sys.argv[1])['_check_embedding_case']("
        "sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            long_embedding_model_path,
            case,
            json.dumps(params),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("n_ubatch,expected", [(None, 512), (128, 128), (0, 513)])
@pytest.mark.parametrize("embedding", [False, True])
def test_causal_microbatch_defaults(n_ubatch, expected, embedding):
    model_path = hf_hub_download("ggml-org/models", "tinyllamas/stories15M-q4_0.gguf")
    params = {} if n_ubatch is None else {"n_ubatch": n_ubatch}
    if embedding and n_ubatch is None:
        expected = 513
    with contextlib.closing(
        llama_cpp.Llama(
            model_path,
            n_ctx=1024,
            n_batch=513,
            n_gpu_layers=0,
            n_threads=2,
            n_threads_batch=2,
            verbose=False,
            embedding=embedding,
            **params,
        )
    ) as model:
        assert llama_cpp.llama_get_causal_attn(model._ctx.ctx)
        assert llama_cpp.llama_n_ubatch(model._ctx.ctx) == expected
        result = model.create_completion(
            "Once upon a time", max_tokens=2, temperature=0
        )
        assert result["usage"]["completion_tokens"] == 2
        assert result["choices"][0]["text"]
