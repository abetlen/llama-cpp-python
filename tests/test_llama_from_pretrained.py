from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
from huggingface_hub import constants
import pytest

from llama_cpp import Llama


class RecordingLlama(Llama):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs

    def close(self):
        pass


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize(
    ("relative_path", "pattern"),
    [
        ("nested/model.gguf", "nested/model.gguf"),
        ("nested/model.gguf", "nested/*.gguf"),
        ("model.gguf", "model.gguf"),
    ],
)
def test_from_pretrained_local_dir(
    tmp_path, monkeypatch, offline, relative_path, pattern
):
    model_file = tmp_path / relative_path
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    listings = []

    def list_files(repo_id, recursive):
        listings.append((repo_id, recursive))
        return [{"name": f"{repo_id}/{relative_path}"}]

    monkeypatch.setattr(
        huggingface_hub, "HfFileSystem", lambda: SimpleNamespace(ls=list_files)
    )
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        assert kwargs["filename"] == model_file.name
        assert kwargs["subfolder"] == str(Path(relative_path).parent)
        assert kwargs["local_dir"] == tmp_path
        return str(model_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = RecordingLlama.from_pretrained(
        repo_id="owner/model", filename=pattern, local_dir=tmp_path, n_gpu_layers=0
    )

    assert Path(model.model_path) == model_file
    assert Path(model.model_path).is_file()
    assert model.kwargs == {"n_gpu_layers": 0}
    assert len(downloads) == 1
    assert listings == ([] if offline else [("owner/model", True)])


@pytest.mark.parametrize("pattern", ["nested/model.gguf", "nested/*.gguf"])
def test_from_pretrained_snapshot_cache(tmp_path, monkeypatch, pattern):
    model_file = tmp_path / "nested/model.gguf"
    model_file.parent.mkdir()
    model_file.touch()
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    snapshots = []

    def snapshot_download(**kwargs):
        snapshots.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        assert kwargs["filename"] == "model.gguf"
        assert kwargs["subfolder"] == "nested"
        assert kwargs["local_dir"] is None
        return str(model_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = RecordingLlama.from_pretrained(
        repo_id="owner/model", filename=pattern, cache_dir="cache", n_gpu_layers=0
    )

    assert Path(model.model_path) == model_file
    assert model.kwargs == {"n_gpu_layers": 0}
    assert snapshots == [
        {"repo_id": "owner/model", "cache_dir": "cache", "local_files_only": True}
    ]
    assert len(downloads) == 2
    assert "local_files_only" not in downloads[0]
    assert downloads[1]["local_files_only"] is True
