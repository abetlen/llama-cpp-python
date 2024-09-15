import time
import uuid

import llama_cpp.llama_types as llama_types

from typing import Literal


class CreateCompletionResponseBuilder:
    """Helper class that transforms a stream of text and optional logprob information
    into OpenAI-API compatible completions or completion chunks."""

    def __init__(
        self,
        *,  # Enforce keyword arguments
        completion_id: str | None = None,
        created: int | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        suffix: str | None = None,
        echo: bool = False,
        prompt_tokens: list[str] | None = None,
        prompt_text_offsets: list[int] | None = None,
        prompt_logprobs: list[float | None] | None = None,
        prompt_top_logprobs: list[dict[str, float] | None] | None = None,
        stream: bool = False,
    ):
        self.completion_id = (
            completion_id if completion_id is not None else f"cmpl-{str(uuid.uuid4())}"
        )
        self.created = created if created is not None else int(time.time())
        self.model = model or ""
        self.max_tokens = max_tokens
        self.stop = stop or []
        self.logpbrobs = logprobs
        self.suffix = suffix
        self.echo = echo

        self.prompt_tokens = prompt_tokens or []

        self.prompt_text_offsets = prompt_text_offsets or []
        self.prompt_logprobs = prompt_logprobs or []
        self.prompt_top_logprobs = prompt_top_logprobs or []

        self.completion_tokens: list[str] = []

        self.completion_text_offsets: list[int] = []
        self.completion_logprobs: list[float | None] = []
        self.completion_top_logprobs: list[dict[str, float] | None] = []

        self.finish_reason: Literal["stop", "length"] | None = None

        self.text_offset = (
            self.prompt_text_offsets[-1]
            if self.prompt_text_offsets
            else len("".join(self.prompt_tokens))
        )
        self.stop_index: int | None = None

        self.stream = stream

    def add(
        self,
        text: str,
        *,  # Enforce keyword arguments
        text_offset: int | None = None,
        logprob: float | None = None,
        top_logprobs: dict[str, float] | None = None,
    ):
        """Add a text chunk to the completion. Optionally include logprob information for the given chunk."""
        # Don't add anything if we are already finished
        if self.finish_reason is not None:
            return

        # Add new token
        self.completion_tokens.append(text)

        # Add new logprob information
        if self.logpbrobs is not None:
            assert text_offset is not None, "text_offset must be provided"
            self.completion_text_offsets.append(text_offset)
            self.completion_logprobs.append(logprob)
            self.completion_top_logprobs.append(top_logprobs)

        # Check if we should finish

        # Check stop sequences
        completion_text = "".join(self.completion_tokens)
        stop_indexes = [completion_text.find(stop) for stop in self.stop]
        if any(stop_index != -1 for stop_index in stop_indexes):
            self.finish_reason = "stop"
            self.stop_index = min(
                stop_index for stop_index in stop_indexes if stop_index != -1
            )

        # Check max_tokens
        if (
            self.max_tokens is not None
            and len(self.prompt_tokens) + len(self.completion_tokens) >= self.max_tokens
        ):
            self.finish_reason = "length"

    def finish(self):
        """Mark the completion as finished with finish_reason "stop"."""
        if self.finish_reason is not None:
            return  # already finished
        self.finish_reason = "stop"

    def completion(self) -> llama_types.CreateCompletionResponse:
        """Return the completion as a dictionary."""
        assert self.finish_reason is not None, "completion() called before finished"

        completion_text = "".join(self.completion_tokens)

        if self.stop_index is not None:
            completion_text = completion_text[: self.stop_index]

        text = completion_text

        if self.echo:
            prompt_text = "".join(self.prompt_tokens)
            text = prompt_text + text

        if self.suffix:
            text += self.suffix

        logprobs_or_none: llama_types.CompletionLogprobs | None = None
        if self.logpbrobs is not None:
            text_offsets = (
                self.prompt_text_offsets + self.completion_text_offsets
                if self.echo
                else self.completion_text_offsets
            )
            token_logprobs = (
                self.prompt_logprobs + self.completion_logprobs
                if self.echo
                else self.completion_logprobs
            )
            tokens = (
                self.prompt_tokens + self.completion_tokens
                if self.echo
                else self.completion_tokens
            )
            top_logprobs = (
                self.prompt_top_logprobs + self.completion_top_logprobs
                if self.echo
                else self.completion_top_logprobs
            )
            logprobs_or_none = {
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "tokens": tokens,
                "top_logprobs": top_logprobs,
            }

        return {
            "id": self.completion_id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(self.prompt_tokens),
                "completion_tokens": len(self.completion_tokens),
                "total_tokens": len(self.prompt_tokens) + len(self.completion_tokens),
            },
        }


def split_on_space(text: str) -> list[str]:
    """Split text on space (preserving the space).

    >>> split_on_space("Hello world!")
    ["Hello ", "world!"]

    >>> split_on_space("Hello  world! ")
    ["Hello  ", "world! "]
    """
    current = ""
    output: list[str] = []
    in_space = False
    for char in text:
        if in_space:
            if char != " ":
                output.append(current)
                current = char
                in_space = False
            else:
                current += char
        else:
            current += char
            if char == " ":
                in_space = True
    if current:
        output.append(current)
    return output


def test_sanity():
    text = "The quick brown fox jumps over the lazy dog."
    assert split_on_space(text) == [
        "The ",
        "quick ",
        "brown ",
        "fox ",
        "jumps ",
        "over ",
        "the ",
        "lazy ",
        "dog.",
    ]


def test_openai_completion_manual_finish():
    cmpl = CreateCompletionResponseBuilder()
    cmpl.finish()
    assert cmpl.finish_reason == "stop"


def test_openai_completions():
    """Test OpenAICompletion class."""
    text = "The quick brown fox jumps over the lazy dog."
    tokens = split_on_space(text)
    prompt_tokens = tokens[:3]
    completion_tokens = tokens[3:]

    cmpl = CreateCompletionResponseBuilder(
        model="text-model",
        completion_id="cmpl-0",
        created=0,
        prompt_tokens=prompt_tokens,
    )

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()  # manually finish

    assert cmpl.finish_reason == "stop", "completion should be finished"
    assert cmpl.completion() == {
        "id": "cmpl-0",
        "object": "text_completion",
        "created": 0,
        "model": "text-model",
        "choices": [
            {
                "text": "fox jumps over the lazy dog.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 6,
            "total_tokens": 9,
        },
    }

    cmpl.add("foobar")
    assert (
        "foobar" not in cmpl.completion()["choices"][0]["text"]
    ), "foobar should not be in completion text"

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        max_tokens=6,
    )

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()

    assert (
        cmpl.finish_reason is "length"
    ), "completion should be finished due to max_tokens"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps over "

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        max_tokens=6,
        echo=True,
    )

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()

    assert cmpl.completion()["choices"][0]["text"] == "The quick brown fox jumps over "

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        echo=True,
        suffix="!",
    )

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()

    assert (
        cmpl.completion()["choices"][0]["text"]
        == "The quick brown fox jumps over the lazy dog.!"
    )

    cmpl = CreateCompletionResponseBuilder(prompt_tokens=prompt_tokens, stop=["og"])

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()

    assert cmpl.finish_reason is "stop"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps over the lazy d"

    cmpl = CreateCompletionResponseBuilder(prompt_tokens=prompt_tokens, stop=["r t", "og"])

    for token in completion_tokens:
        cmpl.add(token)
    cmpl.finish()

    assert cmpl.finish_reason is "stop"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps ove"

    text_offsets = [0, 4, 10, 16, 20, 26, 31, 35, 40]
    logprobs = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
    top_logprobs = [
        {"The ": -0.0, "A ": -1.0, "I ": -2.0},
        {"quick ": -0.0, "slow ": -1.0, "fast ": -2.0},
        {"brown ": -0.0, "red ": -1.0, "green ": -2.0},
        {"fox ": -0.0, "dog ": -1.0, "cat ": -2.0},
        {"jumps ": -0.0, "walks ": -1.0, "runs ": -2.0},
        {"over ": -0.0, "under ": -1.0, "around ": -2.0},
        {"the ": -0.0, "a ": -1.0, "an ": -2.0},
        {"lazy ": -0.0, "tired ": -1.0, "sleepy ": -2.0},
        {"dog.": -0.0, "cat.": -1.0, "mouse.": -2.0},
    ]

    prompt_text_offsets = text_offsets[:3]
    prompt_logprobs = logprobs[:3]
    prompt_top_logprobs = top_logprobs[:3]

    completion_text_offsets = text_offsets[3:]
    completion_logprobs = logprobs[3:]
    completion_top_logprobs = top_logprobs[3:]

    assert (
        len(prompt_tokens)
        == len(prompt_text_offsets)
        == len(prompt_logprobs)
        == len(prompt_top_logprobs)
    )
    assert (
        len(completion_tokens)
        == len(completion_text_offsets)
        == len(completion_logprobs)
        == len(completion_top_logprobs)
    )

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        prompt_text_offsets=prompt_text_offsets,
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top_logprobs,
        logprobs=3,
    )

    for token, text_offset, logprob, top_logprob in zip(
        completion_tokens,
        completion_text_offsets,
        completion_logprobs,
        completion_top_logprobs,
    ):
        cmpl.add(
            token, text_offset=text_offset, logprob=logprob, top_logprobs=top_logprob
        )

    cmpl.finish()

    assert cmpl.finish_reason is "stop"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps over the lazy dog."
    assert cmpl.completion()["choices"][0]["logprobs"] == {
        "text_offset": [16, 20, 26, 31, 35, 40],
        "token_logprobs": [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
        "tokens": ["fox ", "jumps ", "over ", "the ", "lazy ", "dog."],
        "top_logprobs": [
            {"fox ": -0.0, "dog ": -1.0, "cat ": -2.0},
            {"jumps ": -0.0, "walks ": -1.0, "runs ": -2.0},
            {"over ": -0.0, "under ": -1.0, "around ": -2.0},
            {"the ": -0.0, "a ": -1.0, "an ": -2.0},
            {"lazy ": -0.0, "tired ": -1.0, "sleepy ": -2.0},
            {"dog.": -0.0, "cat.": -1.0, "mouse.": -2.0},
        ],
    }

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        prompt_text_offsets=prompt_text_offsets,
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top_logprobs,
        logprobs=3,
        max_tokens=6,
    )

    for token, text_offset, logprob, top_logprob in zip(
        completion_tokens,
        completion_text_offsets,
        completion_logprobs,
        completion_top_logprobs,
    ):
        cmpl.add(
            token, text_offset=text_offset, logprob=logprob, top_logprobs=top_logprob
        )

    cmpl.finish()

    assert cmpl.finish_reason is "length"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps over "
    assert cmpl.completion()["choices"][0]["logprobs"] == {
        "text_offset": [16, 20, 26],
        "token_logprobs": [-0.0, -0.0, -0.0],
        "tokens": ["fox ", "jumps ", "over "],
        "top_logprobs": [
            {"fox ": -0.0, "dog ": -1.0, "cat ": -2.0},
            {"jumps ": -0.0, "walks ": -1.0, "runs ": -2.0},
            {"over ": -0.0, "under ": -1.0, "around ": -2.0},
        ],
    }

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        prompt_text_offsets=prompt_text_offsets,
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top_logprobs,
        logprobs=3,
        stop=["er"],
    )

    for token, text_offset, logprob, top_logprob in zip(
        completion_tokens,
        completion_text_offsets,
        completion_logprobs,
        completion_top_logprobs,
    ):
        cmpl.add(
            token, text_offset=text_offset, logprob=logprob, top_logprobs=top_logprob
        )

    cmpl.finish()

    assert cmpl.finish_reason is "stop"
    assert cmpl.completion()["choices"][0]["text"] == "fox jumps ov"
    assert cmpl.completion()["choices"][0]["logprobs"] == {
        "text_offset": [16, 20, 26],
        "token_logprobs": [-0.0, -0.0, -0.0],
        "tokens": ["fox ", "jumps ", "over "],
        "top_logprobs": [
            {"fox ": -0.0, "dog ": -1.0, "cat ": -2.0},
            {"jumps ": -0.0, "walks ": -1.0, "runs ": -2.0},
            {"over ": -0.0, "under ": -1.0, "around ": -2.0},
        ],
    }

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        prompt_text_offsets=prompt_text_offsets,
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top_logprobs,
        echo=True,
        logprobs=3,
    )

    for token, text_offset, logprob, top_logprob in zip(
        completion_tokens,
        completion_text_offsets,
        completion_logprobs,
        completion_top_logprobs,
    ):
        cmpl.add(
            token, text_offset=text_offset, logprob=logprob, top_logprobs=top_logprob
        )

    cmpl.finish()

    assert cmpl.finish_reason is "stop"
    assert (
        cmpl.completion()["choices"][0]["text"]
        == "The quick brown fox jumps over the lazy dog."
    )
    assert cmpl.completion()["choices"][0]["logprobs"] == {
        "text_offset": [0, 4, 10, 16, 20, 26, 31, 35, 40],
        "token_logprobs": [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
        "tokens": [
            "The ",
            "quick ",
            "brown ",
            "fox ",
            "jumps ",
            "over ",
            "the ",
            "lazy ",
            "dog.",
        ],
        "top_logprobs": [
            {"The ": -0.0, "A ": -1.0, "I ": -2.0},
            {"quick ": -0.0, "slow ": -1.0, "fast ": -2.0},
            {"brown ": -0.0, "red ": -1.0, "green ": -2.0},
            {"fox ": -0.0, "dog ": -1.0, "cat ": -2.0},
            {"jumps ": -0.0, "walks ": -1.0, "runs ": -2.0},
            {"over ": -0.0, "under ": -1.0, "around ": -2.0},
            {"the ": -0.0, "a ": -1.0, "an ": -2.0},
            {"lazy ": -0.0, "tired ": -1.0, "sleepy ": -2.0},
            {"dog.": -0.0, "cat.": -1.0, "mouse.": -2.0},
        ],
    }

    # TODO: first token logprob and top_one_logprob should be None when echo is True


def test_openai_completions_stream():
    """Test OpenAICompletion class."""
    prompt_tokens = ["The ", "quick ", "brown "]

    cmpl = CreateCompletionResponseBuilder(
        prompt_tokens=prompt_tokens,
        stream=True,
    )

    cmpl.add("fox ")
    cmpl.add("jumps ")
    cmpl.add("over ")

    cmpl.finish()  # manually finish
