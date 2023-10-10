from typing import List

import pytest

from llama_cpp import ChatCompletionMessage
from llama_cpp.llama_chat_format import Llama2Formatter


@pytest.fixture
def sequence_of_messages() -> List[ChatCompletionMessage]:
    return [
        ChatCompletionMessage(role="system", content="Welcome to CodeHelp Bot!"),
        ChatCompletionMessage(
            role="user", content="Hi there! I need some help with Python."
        ),
        ChatCompletionMessage(
            role="assistant", content="Of course! What do you need help with in Python?"
        ),
        ChatCompletionMessage(
            role="user",
            content="I'm trying to write a function to find the factorial of a number, but I'm stuck.",
        ),
        ChatCompletionMessage(
            role="assistant",
            content="I can help with that! Would you like a recursive or iterative solution?",
        ),
        ChatCompletionMessage(
            role="user", content="Let's go with a recursive solution."
        ),
    ]


def test_llama2_formatter(sequence_of_messages):
    prompt = """<<SYS>>Welcome to CodeHelp Bot!<</SYS>>\n[INST] Hi there! I need some help with Python. [/INST]\nOf course! What do you need help with in Python?\n[INST] I'm trying to write a function to find the factorial of a number, but I'm stuck. [/INST]\nI can help with that! Would you like a recursive or iterative solution?\n[INST] Let's go with a recursive solution. [/INST]"""
    llama2formatter = Llama2Formatter()
    assert prompt == llama2formatter._format_messages(sequence_of_messages)
