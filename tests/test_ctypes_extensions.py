import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llama_cpp._ctypes_extensions as ctypes_extensions


@pytest.fixture
def windows_loader(monkeypatch, tmp_path):
    handles = []

    def add_dll_directory(path):
        handle = SimpleNamespace(path=path, close=Mock())
        handles.append(handle)
        return handle

    environ = {"PATH": "original-path"}
    cdll = Mock(return_value=object())
    monkeypatch.setattr(
        ctypes_extensions,
        "sys",
        SimpleNamespace(platform="win32", version_info=(3, 12)),
    )
    monkeypatch.setattr(
        ctypes_extensions,
        "os",
        SimpleNamespace(
            environ=environ,
            path=os.path,
            pathsep=";",
            add_dll_directory=add_dll_directory,
        ),
    )
    monkeypatch.setattr(
        ctypes_extensions, "ctypes", SimpleNamespace(CDLL=cdll, RTLD_GLOBAL=0)
    )
    (tmp_path / "llama.dll").touch()
    return SimpleNamespace(
        base_path=tmp_path, environ=environ, handles=handles, cdll=cdll
    )


@pytest.mark.parametrize("sdk", [None, "CUDA_PATH", "HIP_PATH"])
def test_windows_failed_load_restores_search_paths(windows_loader, sdk):
    loader = windows_loader
    if sdk:
        loader.environ[sdk] = str(loader.base_path / "sdk")
    original_environ = loader.environ.copy()

    def fail_load(path, **kwargs):
        assert path == str(loader.base_path / "llama.dll")
        assert kwargs == {"winmode": 0}
        assert loader.environ["PATH"].split(";")[0] == str(loader.base_path)
        assert all(
            not handle.close.called for handle in loader.handles[-(3 if sdk else 1) :]
        )
        raise OSError("missing dependency")

    loader.cdll.side_effect = fail_load
    for _ in range(3):
        with pytest.raises(RuntimeError, match="missing dependency"):
            ctypes_extensions.load_shared_library("llama", loader.base_path)
        assert loader.environ == original_environ
        assert all(handle.close.call_count == 1 for handle in loader.handles)

    expected_directories = [str(loader.base_path)]
    if sdk:
        expected_directories += [
            os.path.join(loader.environ[sdk], "bin"),
            os.path.join(loader.environ[sdk], "lib"),
        ]
    assert [handle.path for handle in loader.handles] == expected_directories * 3


def test_windows_missing_library_restores_search_paths(windows_loader):
    loader = windows_loader
    (loader.base_path / "llama.dll").unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        ctypes_extensions.load_shared_library("llama", loader.base_path)
    assert loader.environ == {"PATH": "original-path"}
    assert all(handle.close.call_count == 1 for handle in loader.handles)
    loader.cdll.assert_not_called()


def test_windows_failed_load_preserves_previous_success(windows_loader):
    loader = windows_loader
    ctypes_extensions.load_shared_library("llama", loader.base_path)
    original_environ = loader.environ.copy()
    successful_handles = loader.handles.copy()
    loader.cdll.side_effect = OSError("missing dependency")

    for _ in range(3):
        with pytest.raises(RuntimeError, match="missing dependency"):
            ctypes_extensions.load_shared_library("llama", loader.base_path)
        assert loader.environ == original_environ
        assert all(not handle.close.called for handle in successful_handles)
        assert all(handle.close.call_count == 1 for handle in loader.handles[1:])


def test_windows_shared_sdk_directory_is_registered_once(windows_loader):
    loader = windows_loader
    sdk_path = str(loader.base_path / "sdk")
    loader.environ.update(CUDA_PATH=sdk_path, HIP_PATH=sdk_path)
    loader.cdll.side_effect = OSError("missing dependency")
    with pytest.raises(RuntimeError, match="missing dependency"):
        ctypes_extensions.load_shared_library("llama", loader.base_path)
    assert [handle.path for handle in loader.handles] == [
        str(loader.base_path),
        os.path.join(sdk_path, "bin"),
        os.path.join(sdk_path, "lib"),
    ]
    assert all(handle.close.call_count == 1 for handle in loader.handles)
    assert loader.environ == {
        "PATH": "original-path",
        "CUDA_PATH": sdk_path,
        "HIP_PATH": sdk_path,
    }


def test_windows_partial_setup_failure_restores_search_paths(
    windows_loader, monkeypatch
):
    loader = windows_loader
    loader.environ["CUDA_PATH"] = str(loader.base_path / "cuda")
    original_environ = loader.environ.copy()
    add_directory = ctypes_extensions.os.add_dll_directory

    def fail_second_directory(path):
        if path != str(loader.base_path):
            raise OSError("invalid SDK directory")
        return add_directory(path)

    monkeypatch.setattr(
        ctypes_extensions.os, "add_dll_directory", fail_second_directory
    )
    with pytest.raises(OSError, match="invalid SDK directory"):
        ctypes_extensions.load_shared_library("llama", loader.base_path)
    assert loader.environ == original_environ
    assert len(loader.handles) == 1
    loader.handles[0].close.assert_called_once_with()
    loader.cdll.assert_not_called()


@pytest.mark.parametrize(
    "original_path", [None, "", "original-path", "already-present"]
)
@pytest.mark.parametrize("succeeds", [False, True])
def test_windows_path_and_successful_load(windows_loader, original_path, succeeds):
    loader = windows_loader
    if original_path is None:
        loader.environ.pop("PATH")
    elif original_path == "already-present":
        loader.environ["PATH"] = "first;" + str(loader.base_path) + ";last"
    else:
        loader.environ["PATH"] = original_path
    original_environ = loader.environ.copy()
    if not succeeds:
        loader.cdll.side_effect = OSError("missing dependency")

    for _ in range(2):
        if succeeds:
            assert (
                ctypes_extensions.load_shared_library("llama", loader.base_path)
                is loader.cdll.return_value
            )
            assert loader.environ["PATH"].split(";").count(str(loader.base_path)) == 1
            assert all(not handle.close.called for handle in loader.handles)
        else:
            with pytest.raises(RuntimeError, match="missing dependency"):
                ctypes_extensions.load_shared_library("llama", loader.base_path)
            assert loader.environ == original_environ
            assert all(handle.close.call_count == 1 for handle in loader.handles)


@pytest.mark.parametrize("platform", ["linux", "freebsd14", "darwin"])
def test_non_windows_load_does_not_change_search_paths(
    windows_loader, monkeypatch, platform
):
    loader = windows_loader
    monkeypatch.setattr(ctypes_extensions, "sys", SimpleNamespace(platform=platform))
    (loader.base_path / "libllama.so").touch()
    assert (
        ctypes_extensions.load_shared_library("llama", loader.base_path)
        is loader.cdll.return_value
    )
    loader.cdll.assert_called_once_with(str(loader.base_path / "libllama.so"))
    assert loader.environ == {"PATH": "original-path"}
    assert not loader.handles
