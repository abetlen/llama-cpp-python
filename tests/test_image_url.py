import base64
import importlib.util
from pathlib import Path

import pytest

_UTILS = Path(__file__).resolve().parents[1] / "llama_cpp" / "_utils.py"
_spec = importlib.util.spec_from_file_location("llama_cpp_utils_under_test", _UTILS)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
_bytes_from_image_url = _utils._bytes_from_image_url


def test_base64_data_url_round_trip():
    payload = b"\x89PNG\r\n"
    url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    assert _bytes_from_image_url(url) == payload


def test_base64_data_url_is_case_insensitive():
    payload = b"hello"
    url = "data:image/png;BASE64," + base64.b64encode(payload).decode("ascii")
    assert _bytes_from_image_url(url) == payload


def test_data_url_without_comma_raises():
    with pytest.raises(ValueError, match="Invalid data URL"):
        _bytes_from_image_url("data:image/png;base64")


def test_plain_data_url_is_unquoted():
    assert _bytes_from_image_url("data:image/svg+xml,%3Csvg%3E") == b"<svg>"


def test_http_url(monkeypatch: pytest.MonkeyPatch):
    import urllib.request

    class FakeResp:
        def read(self):
            return b"img"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: FakeResp())
    assert _bytes_from_image_url("https://example.com/a.png") == b"img"
