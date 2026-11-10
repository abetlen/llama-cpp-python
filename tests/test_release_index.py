"""Exercise the release-index generator without downloading release assets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from html.parser import HTMLParser

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "releases-to-pep-503.sh"
WHEEL_URL = "https://example.invalid/llama_cpp_python-0.3.36-py3-none-any.whl"
SHA256 = "0123456789abcdef" * 4


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs)["href"])


@pytest.fixture
def generate_index(tmp_path):
    for tool in ("bash", "jq"):
        if shutil.which(tool) is None:
            pytest.skip(f"release-index tests require {tool}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text('#!/bin/sh\ncat "$RELEASE_RESPONSE"\n')
    curl.chmod(0o755)
    response = tmp_path / "response.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RELEASE_RESPONSE": str(response),
    }

    def run(assets, releases="v0.3.36\n", pattern=None):
        response.write_text(json.dumps({"assets": assets}))
        (tmp_path / "all_releases.txt").write_text(releases)
        output = tmp_path / "index"
        command = ["bash", str(SCRIPT), str(output)]
        if pattern is not None:
            command.append(pattern)
        subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True)
        page = (output / "llama-cpp-python" / "index.html").read_text()
        parser = Links()
        parser.feed(page)
        return parser.hrefs, page

    return run


def wheel(**kwargs):
    return {
        "name": "llama_cpp_python-0.3.36-py3-none-any.whl",
        "browser_download_url": WHEEL_URL,
        **kwargs,
    }


@pytest.mark.parametrize("digest", [f"sha256:{SHA256}", f"sha256:{SHA256.upper()}"])
def test_available_sha256_digest(generate_index, digest):
    links, page = generate_index([wheel(digest=digest)])
    assert links == [f"{WHEEL_URL}#sha256={digest.split(':')[1]}"]
    assert f">{WHEEL_URL}</a>" in page


@pytest.mark.parametrize(
    "digest",
    [
        None,
        "",
        "sha256:abc",
        "sha256:" + "z" * 64,
        "sha512:" + "a" * 128,
        "sha256:" + SHA256 + "extra",
        123,
        False,
        {},
        [],
    ],
)
def test_unusable_digest_preserves_original_link(generate_index, digest):
    links, _ = generate_index([wheel(digest=digest)])
    assert links == [WHEEL_URL]


def test_historical_asset_without_digest(generate_index):
    links, _ = generate_index([wheel()])
    assert links == [WHEEL_URL]


def test_mixed_release_assets(generate_index):
    legacy_url = WHEEL_URL.replace("py3-none-any", "cp311-cp311-linux_x86_64")
    assets = [
        wheel(digest=f"sha256:{SHA256}"),
        {**wheel(), "browser_download_url": legacy_url},
        {
            "name": "source.tar.gz",
            "browser_download_url": "https://example.invalid/source.tar.gz",
            "digest": f"sha256:{SHA256}",
        },
    ]
    links, _ = generate_index(assets)
    assert links == [f"{WHEEL_URL}#sha256={SHA256}", legacy_url]


def test_backend_release_pattern(generate_index):
    links, page = generate_index(
        [wheel(digest=f"sha256:{SHA256}")],
        releases="v0.3.36\nv0.3.36-cu130\n",
        pattern=r"^[v]?[0-9]+\.[0-9]+\.[0-9]+-cu130$",
    )
    assert links == [f"{WHEEL_URL}#sha256={SHA256}"]
    assert "<h2>v0.3.36</h2>" in page


@pytest.mark.parametrize("assets,releases", [([], "v0.3.36\n"), ([wheel()], "")])
def test_empty_indexes_still_generate(generate_index, assets, releases):
    links, page = generate_index(assets, releases=releases)
    assert links == []
    assert page.rstrip().endswith("</html>")
