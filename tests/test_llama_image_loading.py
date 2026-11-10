import base64
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llama_cpp.llama_chat_format import Llava15ChatHandler, MTMDChatHandler


@pytest.fixture(params=[Llava15ChatHandler, MTMDChatHandler])
def load_image(request):
    # Image downloading does not require a model or multimodal native context.
    return request.param._load_image


@pytest.fixture
def image_server():
    requests = []
    payload = b"\x89PNG\r\n\x1a\nimage-bytes"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("User-Agent")))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/image")
                self.end_headers()
                return
            if self.path == "/missing":
                self.send_error(404)
                return
            if self.headers.get("User-Agent") != "llama-cpp-python":
                self.send_error(403)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", payload, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_image_uses_package_user_agent(load_image, image_server):
    url, payload, requests = image_server
    assert load_image(url + "/image") == payload
    assert requests == [("/image", "llama-cpp-python")]


def test_image_redirect_preserves_user_agent(load_image, image_server):
    url, payload, requests = image_server
    assert load_image(url + "/redirect") == payload
    assert requests == [
        ("/redirect", "llama-cpp-python"),
        ("/image", "llama-cpp-python"),
    ]


def test_remote_image_http_errors_propagate(load_image, image_server):
    url, _, _ = image_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        load_image(url + "/missing")
    assert exc.value.code == 404


def test_data_image_does_not_open_url(load_image, monkeypatch):
    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("data images must not issue a request")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_urlopen)
    payload = b"\x89PNG\r\n\x1a\nimage-bytes"
    url = "data:image/png;base64," + base64.b64encode(payload).decode()
    assert load_image(url) == payload


def test_file_image_remains_supported(load_image, tmp_path):
    payload = b"local-image-bytes"
    path = tmp_path / "image.png"
    path.write_bytes(payload)
    assert load_image(path.as_uri()) == payload
