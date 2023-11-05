from typing import List

import pytest

from llama_cpp import ChatCompletionMessage
from llama_cpp.llama_jinja_format import Llama2Formatter


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
    expected_prompt = (
        "<<SYS>> Welcome to CodeHelp Bot! <</SYS>>\n"
        "[INST] Hi there! I need some help with Python. [/INST]\n"
        "Of course! What do you need help with in Python?\n"
        "[INST] I'm trying to write a function to find the factorial of a number, but I'm stuck. [/INST]\n"
        "I can help with that! Would you like a recursive or iterative solution?\n"
        "[INST] Let's go with a recursive solution. [/INST]\n"
    )

    llama2_formatter_instance = Llama2Formatter()
    formatter_response = llama2_formatter_instance(sequence_of_messages)
    assert (
        expected_prompt == formatter_response.prompt
    ), "The formatted prompt does not match the expected output."


# Optionally, include a test for the 'stop' if it's part of the functionality.
