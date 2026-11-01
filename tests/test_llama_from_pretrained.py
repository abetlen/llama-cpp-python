from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub import constants
from huggingface_hub.utils import LocalEntryNotFoundError

from llama_cpp import Llama


class LoadedPath(Llama):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.options = kwargs

    def close(self):
        pass


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def unexpected_listing(*args, **kwargs):
        pytest.fail("Offline loading must use the local snapshot")

    monkeypatch.setattr(huggingface_hub.HfFileSystem, "ls", unexpected_listing)
    repo = tmp_path / "models--test--model"
    revision = "a" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision)
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return tmp_path, snapshot


@pytest.mark.parametrize("filename", ["weights/model.gguf", "weights/*.gguf"])
def test_from_pretrained_uses_offline_snapshot(offline_cache, filename):
    cache_dir, snapshot = offline_cache
    model = snapshot / "weights" / "model.gguf"
    model.parent.mkdir()
    model.write_bytes(b"cached model")

    loaded = LoadedPath.from_pretrained(
        "test/model", filename, cache_dir=cache_dir, n_ctx=128
    )

    assert Path(loaded.model_path) == model
    assert loaded.options == {"n_ctx": 128}


@pytest.mark.parametrize("additional_file", ["model-00002.gguf", "*-00002.gguf"])
def test_from_pretrained_offline_additional_files(offline_cache, additional_file):
    cache_dir, snapshot = offline_cache
    for name in ["model-00001.gguf", "model-00002.gguf"]:
        (snapshot / name).write_bytes(b"cached shard")

    loaded = LoadedPath.from_pretrained(
        "test/model",
        "model-00001.gguf",
        additional_files=[additional_file],
        cache_dir=cache_dir,
    )

    assert Path(loaded.model_path) == snapshot / "model-00001.gguf"


@pytest.mark.parametrize(
    "filenames, pattern, message",
    [
        (["model.gguf"], "missing*.gguf", "No file found"),
        (["one.gguf", "two.gguf"], "*.gguf", "Multiple files found"),
    ],
)
def test_from_pretrained_offline_matching_errors(
    offline_cache, filenames, pattern, message
):
    cache_dir, snapshot = offline_cache
    for name in filenames:
        (snapshot / name).write_bytes(b"cached model")

    with pytest.raises(ValueError, match=message):
        LoadedPath.from_pretrained("test/model", pattern, cache_dir=cache_dir)


def test_from_pretrained_offline_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    with pytest.raises(LocalEntryNotFoundError):
        LoadedPath.from_pretrained("test/missing", "*.gguf", cache_dir=tmp_path)


def test_from_pretrained_offline_local_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    model = local_dir / "model.gguf"
    model.write_bytes(b"downloaded model")

    loaded = LoadedPath.from_pretrained(
        "test/model", "*.gguf", cache_dir=tmp_path / "cache", local_dir=local_dir
    )

    assert Path(loaded.model_path) == model


@pytest.mark.parametrize("existing_dir", [False, True])
def test_from_pretrained_offline_requires_files_in_local_dir(
    offline_cache, existing_dir
):
    cache_dir, snapshot = offline_cache
    (snapshot / "model.gguf").write_bytes(b"cached model")
    local_dir = cache_dir / "downloaded"
    if existing_dir:
        local_dir.mkdir()

    with pytest.raises(LocalEntryNotFoundError):
        LoadedPath.from_pretrained(
            "test/model", "*.gguf", cache_dir=cache_dir, local_dir=local_dir
        )


def test_from_pretrained_online_lists_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    calls = []

    def list_repository(self, repo_id, recursive):
        calls.append((repo_id, recursive))
        return [{"name": "test/model/model.gguf"}, "test/model/README.md"]

    def download(**kwargs):
        assert kwargs["repo_id"] == "test/model"
        assert kwargs["filename"] == "model.gguf"
        return str(tmp_path / "model.gguf")

    monkeypatch.setattr(huggingface_hub.HfFileSystem, "ls", list_repository)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    loaded = LoadedPath.from_pretrained("test/model", "*.gguf")

    assert Path(loaded.model_path) == tmp_path / "model.gguf"
    assert calls == [("test/model", True)]
