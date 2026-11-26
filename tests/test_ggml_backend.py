"""Native loader contracts; dynamic-model cases are opt-in for custom builds."""

import contextlib
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from llama_cpp import Llama, _ggml


def _registry():
    count = _ggml.libggml.ggml_backend_reg_count
    count.argtypes = []
    count.restype = ctypes.c_size_t
    get = _ggml.libggml.ggml_backend_reg_get
    get.argtypes = [ctypes.c_size_t]
    get.restype = ctypes.c_void_p
    name = _ggml.libggml.ggml_backend_reg_name
    name.argtypes = [ctypes.c_void_p]
    name.restype = ctypes.c_char_p
    return [name(get(i)).decode() for i in range(count())]


def _completion(model_path):
    with contextlib.closing(
        Llama(
            model_path=model_path,
            n_ctx=64,
            n_batch=64,
            n_threads=2,
            n_threads_batch=2,
            n_gpu_layers=0,
            verbose=False,
        )
    ) as model:
        completion = model.create_completion(
            "Once upon a time", max_tokens=2, temperature=0
        )
        assert completion["usage"]["completion_tokens"] == 2
        assert completion["choices"][0]["text"]
        return completion["choices"][0]["text"]


def _probe(mode, plugin, model_path):
    def no_network(*args, **kwargs):
        raise AssertionError("native loader test must not use the network")

    socket.create_connection = no_network
    socket.socket.connect = no_network
    if mode == "missing":
        missing = Path.cwd() / "missing-backend.so"
        assert _ggml.ggml_backend_load(os.fsencode(missing)) is None
        with pytest.raises(ctypes.ArgumentError):
            _ggml.ggml_backend_load(str(missing))
        assert _ggml.ggml_backend_load_all_from_path(os.fsencode(Path.cwd())) is None
        assert _ggml.ggml_backend_load_all_from_path(None) is None
        assert _ggml.ggml_backend_load_all() is None
        return {"mode": mode, "null_handle": True, "void_returns": True}

    assert Path(plugin).is_file()
    assert Path(model_path).is_file()
    assert _registry() == [], "use a fresh GGML_BACKEND_DL build for these cases"
    if mode == "load":
        with pytest.raises(ValueError, match="Failed to load model"):
            _completion(model_path)
        reg = _ggml.ggml_backend_load(os.fsencode(plugin))
        assert reg is not None
        try:
            assert _registry() == ["CPU"]
            text = _completion(model_path)
        finally:
            # _completion closes the model/context before the plugin is unloaded.
            assert _ggml.ggml_backend_unload(reg) is None
        assert _registry() == []
        with pytest.raises(ValueError, match="Failed to load model"):
            _completion(model_path)
        return {"mode": mode, "fail_pass_fail": True, "text": text}
    if mode == "from_path":
        assert (
            _ggml.ggml_backend_load_all_from_path(os.fsencode(Path(plugin).parent))
            is None
        )
    elif mode == "all":
        os.chdir(Path(plugin).parent)
        assert _ggml.ggml_backend_load_all() is None
    else:
        raise AssertionError(mode)
    assert "CPU" in _registry()
    return {"mode": mode, "registry": _registry(), "text": _completion(model_path)}


@pytest.mark.parametrize("mode", ["missing", "load", "from_path", "all"])
def test_native_backend_loaders(tmp_path, mode):
    plugin = os.environ.get("LLAMA_CPP_TEST_BACKEND", "")
    model_path = os.environ.get("LLAMA_CPP_TEST_MODEL", "")
    if mode != "missing" and not (plugin and model_path):
        pytest.skip(
            "set LLAMA_CPP_TEST_BACKEND and LLAMA_CPP_TEST_MODEL for dynamic tests"
        )
    env = os.environ.copy()
    # Test explicit registration, not the native environment-variable workaround.
    env.pop("GGML_BACKEND_PATH", None)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, plugin, model_path],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == mode
    print(result.stdout.strip())


if __name__ == "__main__":
    print(json.dumps(_probe(*sys.argv[1:])))
