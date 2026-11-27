import struct
from pathlib import Path

import pytest

import llama_cpp


def _write_vocab_model(tmp_path, metadata):
    # This tracked GGUF contains vocabulary metadata only, with no tensor offsets.
    source = Path("vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf").read_bytes()
    magic, version, tensor_count, metadata_count = struct.unpack("<4sIQQ", source[:24])
    assert (magic, version, tensor_count) == (b"GGUF", 3, 0)

    def gguf_string(value):
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    entries = b"".join(
        gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)
        for key, value in metadata.items()
    )
    path = tmp_path / "metadata.gguf"
    path.write_bytes(
        source[:16]
        + struct.pack("<Q", metadata_count + len(metadata))
        + entries
        + source[24:]
    )
    return str(path)


@pytest.mark.parametrize("size", [1023, 1024, 1025])
def test_chat_template_at_metadata_buffer_boundary(tmp_path, size):
    prefix, suffix = "{{ messages[0]['content'] }}{#", "#}"
    template = prefix + "x" * (size - len(prefix) - len(suffix)) + suffix
    path = _write_vocab_model(tmp_path, {"tokenizer.chat_template": template})
    model = llama_cpp.Llama(model_path=path, vocab_only=True, verbose=False)
    try:
        assert model.metadata["tokenizer.chat_template"] == template
        assert model.chat_format == "chat_template.default"
    finally:
        model.close()


@pytest.mark.parametrize("size", [1023, 1024, 1025])
@pytest.mark.parametrize("field", ["key", "value"])
@pytest.mark.parametrize("ending", ["x", "é"], ids=["ascii", "utf8"])
def test_metadata_buffer_boundaries(tmp_path, size, field, ending):
    text = "x" * (size - len(ending.encode("utf-8"))) + ending
    key, value = (text, "value") if field == "key" else ("test.value", text)
    template = "{{ messages[0]['content'] }}"
    path = _write_vocab_model(
        tmp_path, {key: value, "tokenizer.chat_template": template}
    )
    model = llama_cpp.Llama(model_path=path, vocab_only=True, verbose=False)
    try:
        assert model.metadata[key] == value
        assert model.metadata["tokenizer.chat_template"] == template
        assert model.chat_format == "chat_template.default"
    finally:
        model.close()


@pytest.mark.parametrize("ending", ["x", "é"], ids=["ascii", "utf8"])
def test_metadata_value_at_resized_buffer_boundary(tmp_path, ending):
    # Reading this key grows the buffer to 1101 bytes before reading its value.
    key = "x" * 1100
    value = "y" * (1101 - len(ending.encode("utf-8"))) + ending
    path = _write_vocab_model(tmp_path, {key: value})
    model = llama_cpp.Llama(model_path=path, vocab_only=True, verbose=False)
    try:
        assert model.metadata[key] == value
    finally:
        model.close()
