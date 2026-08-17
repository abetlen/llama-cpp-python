"""Isolated test for Llama.close() chat handler cleanup logic.

Tests the fix for https://github.com/abetlen/llama-cpp-python/issues/2342
without needing to import the full llama_cpp package (which requires C extensions).

We extract the close() method from the source, compile it as a standalone
function, and call it with mocked 'self'.
"""

import ast
import textwrap
from unittest.mock import MagicMock
import pytest


class MockExitStack:
    """Stand-in for contextlib.ExitStack."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _extract_close_body():
    """Read the close() method body from llama_cpp/llama.py and return
    just the indented statements (ready to be called as a function)."""
    with open("llama_cpp/llama.py") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "close":
            if node.args.args and node.args.args[0].arg == "self":
                # Unparse the whole function, then exec to get the callable
                return ast.unparse(node)
    raise AssertionError("Could not find close(self) method in llama.py")


class TestCloseLogic:

    def _make_close_func(self):
        source = _extract_close_body()
        namespace = {}
        exec(compile(source, "<test>", "exec"), namespace)
        return namespace["close"]

    def test_close_cleans_up_exit_stack_handler(self):
        close_fn = self._make_close_func()

        handler = MagicMock()
        handler._exit_stack = MockExitStack()

        mock_self = MagicMock()
        mock_self.chat_handler = handler
        mock_self._stack = MockExitStack()

        close_fn(mock_self)

        assert handler._exit_stack.closed is True
        assert mock_self._stack.closed is True

    def test_close_cleans_up_close_method_handler(self):
        close_fn = self._make_close_func()

        class HandlerWithClose:
            def __init__(self):
                self.close_called = False

            def close(self):
                self.close_called = True

        handler = HandlerWithClose()
        mock_self = MagicMock()
        mock_self.chat_handler = handler
        mock_self._stack = MockExitStack()

        close_fn(mock_self)

        assert handler.close_called is True
        assert mock_self._stack.closed is True

    def test_close_handles_plain_handler(self):
        close_fn = self._make_close_func()

        class PlainHandler:
            pass

        handler = PlainHandler()
        mock_self = MagicMock()
        mock_self.chat_handler = handler
        mock_self._stack = MockExitStack()

        close_fn(mock_self)
        assert mock_self._stack.closed is True

    def test_close_handles_none_handler(self):
        close_fn = self._make_close_func()

        mock_self = MagicMock()
        mock_self.chat_handler = None
        mock_self._stack = MockExitStack()

        close_fn(mock_self)
        assert mock_self._stack.closed is True

    def test_close_prioritizes_exit_stack_over_close_method(self):
        """_exit_stack.close() takes priority over handler.close()."""
        close_fn = self._make_close_func()

        class DualHandler:
            def __init__(self):
                self.close_called = False

            def close(self):
                self.close_called = True

        handler = DualHandler()
        handler._exit_stack = MagicMock()
        handler._exit_stack.close = MagicMock()

        mock_self = MagicMock()
        mock_self.chat_handler = handler
        mock_self._stack = MockExitStack()

        close_fn(mock_self)

        handler._exit_stack.close.assert_called_once()
        assert handler.close_called is False

    def test_close_source_references_chat_handler(self):
        source = _extract_close_body()
        assert "chat_handler" in source
