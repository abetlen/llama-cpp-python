import io
import os
import sys

from llama_cpp._utils import suppress_stdout_stderr


class _FakeOutStream(io.TextIOBase):
    """Minimal stand-in for ipykernel's OutStream.

    ipykernel replaces sys.stdout / sys.stderr with OutStream objects that
    cache the original file descriptor in ``_original_stdstream_copy``.
    Writes go through that cached fd, which bypasses any ``os.dup2`` redirect
    performed against the real stdout/stderr fd numbers (1 / 2). The fix in
    ``suppress_stdout_stderr.__enter__`` temporarily re-points that copy at
    the real fd so the redirect actually takes effect; ``__exit__`` restores
    it.
    """

    def __init__(self, sentinel_fd: int):
        super().__init__()
        self._original_stdstream_copy = sentinel_fd
        self.written = []

    def writable(self) -> bool:
        return True

    def write(self, s):
        self.written.append(s)
        return len(s)


def test_suppress_stdout_stderr_repoints_ipykernel_fd_copy_and_restores_it():
    sentinel_stdout, sentinel_stderr = 4242, 4243
    fake_out = _FakeOutStream(sentinel_stdout)
    fake_err = _FakeOutStream(sentinel_stderr)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = fake_out, fake_err
    try:
        with suppress_stdout_stderr(disable=False):
            # Inside the context, the cached fd copies should have been
            # repointed at the real stdout/stderr fd numbers so that writes
            # going through them get caught by the os.dup2 redirect.
            assert fake_out._original_stdstream_copy == 1
            assert fake_err._original_stdstream_copy == 2

        # On exit, the original sentinel values must be restored, otherwise
        # subsequent ipykernel writes would land on fd 1 / 2 instead of the
        # frontend stream — silently leaking output to the terminal.
        assert fake_out._original_stdstream_copy == sentinel_stdout
        assert fake_err._original_stdstream_copy == sentinel_stderr
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr


def test_suppress_stdout_stderr_no_op_when_attribute_absent():
    # Plain sys.stdout / sys.stderr (no ``_original_stdstream_copy``) must
    # not gain that attribute as a side effect of entering the context.
    assert not hasattr(sys.stdout, "_original_stdstream_copy")
    assert not hasattr(sys.stderr, "_original_stdstream_copy")

    with suppress_stdout_stderr(disable=False):
        pass

    assert not hasattr(sys.stdout, "_original_stdstream_copy")
    assert not hasattr(sys.stderr, "_original_stdstream_copy")


def test_suppress_stdout_stderr_disable_skips_redirect():
    saved_stdout = sys.stdout
    with suppress_stdout_stderr(disable=True):
        assert sys.stdout is saved_stdout