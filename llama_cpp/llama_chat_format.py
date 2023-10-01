import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol
from . import llama_types


def _get_system_message(
    messages: List[llama_types.ChatCompletionRequestMessage],
) -> str:
    """Get the first system message."""
    for message in messages:
        if message["role"] == "system":
            return message["content"] or ""
    return ""


def _map_roles(
    messages: List[llama_types.ChatCompletionRequestMessage], role_map: Dict[str, str]
) -> List[Tuple[str, Optional[str]]]:
    """Map the message roles."""
    output: List[Tuple[str, Optional[str]]] = []
    for message in messages:
        role = message["role"]
        if role in role_map:
            output.append((role_map[role], message["content"]))
    return output


def _format_llama2(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the llama2 style."""
    ret = system_message + sep
    for role, message in messages:
        if message:
            ret += message + " "
        else:
            ret += role + " "
    return ret


def _format_add_colon_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the add-colon-single style."""
    ret = system_message + sep
    for role, message in messages:
        if message:
            ret += role + ": " + message + sep
        else:
            ret += role + ":"
    return ret


def _format_add_colon_two(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str, sep2: str
) -> str:
    """Format the prompt with the add-colon-two style."""
    seps = [sep, sep2]
    ret = system_message + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            ret += role + ": " + message + seps[i % 2]
        else:
            ret += role + ":"
    return ret


def _format_no_colon_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the no-colon-single style."""
    ret = system_message
    for role, message in messages:
        if message:
            ret += role + message + sep
        else:
            ret += role
    return ret


def _format_add_colon_space_single(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the add-colon-space-single style."""
    ret = system_message + sep
    for role, message in messages:
        if message:
            ret += role + ": " + message + sep
        else:
            ret += role + ": "  # must be end with a space
    return ret


def _format_chatml(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the chatml style."""
    ret = "" if system_message == "" else system_message + sep + "\n"
    for role, message in messages:
        if message:
            ret += role + "\n" + message + sep + "\n"
        else:
            ret += role + "\n"
    return ret


@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[Union[str, List[str]]] = None


class ChatFormatter(Protocol):
    def __call__(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        ...


_CHAT_FORMATS: Dict[str, ChatFormatter] = {}


def register_chat_format(name: str):
    def decorator(f: ChatFormatter):
        _CHAT_FORMATS[name] = f
        return f

    return decorator


def get_chat_format(name: str):
    try:
        return _CHAT_FORMATS[name]
    except KeyError:
        raise ValueError(
            f"Invalid chat format: {name} (valid formats: {list(_CHAT_FORMATS.keys())})"
        )


@register_chat_format("llama-2")
def format_llama2(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    _roles = dict(user="[INST]", assistant="[/INST]")
    _sep = "\n\n"
    system_message = _get_system_message(messages)
    system_message = _system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_llama2(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("alpaca")
def format_alpaca(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="### Instruction", assistant="### Response")
    _sep = "\n\n"
    _sep2 = "</s>"
    system_message = _get_system_message(messages)
    _messages = _map_roles(messages, _roles)
    _prompt = _format_add_colon_two(system_message, _messages, _sep, _sep2)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("vicuna")
def format(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    _roles = dict(user="USER", assistant="ASSISTANT")
    _sep = " "
    _sep2 = "</s>"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_two(system_message, _messages, _sep, _sep2)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("oasst_llama")
def format_oasst_llama(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    _roles = dict(user="<|prompter|>", assistant="<|assistant|>")
    _sep = "</s>"
    system_message = _get_system_message(messages)
    system_message = _system_template.format(system_message=system_message)
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("openbuddy")
def format_openbuddy(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = """Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?"""
    _roles = dict(user="User", assistant="Assistant")
    _sep = "\n"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("redpajama-incite")
def format_redpajama_incite(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _system_message = _get_system_message(messages)
    _roles = dict(user="<human>", assistant="<bot>")
    _sep = "\n"
    _stop = "<human>"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_stop)


@register_chat_format("snoozy")
def format_snoozy(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = "### Instruction:\n{system_message}"
    default_system_message = "The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response."
    _system_message = _get_system_message(messages)
    _system_message = (
        _system_message if _system_message != "" else default_system_message
    )
    system_message = system_template.format(system_message=_system_message)
    _roles = dict(user="### Prompt", assistant="### Response")
    _sep = "\n"
    _stop = "###"
    system_message = _system_message
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_stop)


@register_chat_format("phind")
def format_phind(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(user="### User Message", assistant="### Assistant")
    _sep = "\n\n"
    _system_message = "### System Prompt\nYou are an intelligent programming assistant."
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_add_colon_single(_system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)


@register_chat_format("open-orca")
def format_open_orca(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = "{system_message}"
    system_message = (
        "You are a helpful assistant. Please answer truthfully and write out your "
    )
    "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
    "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
    "aren't sure about something, say so clearly. You will act as a professional logician, mathematician, "
    "and physicist. You will also act as the most appropriate type of expert to answer any particular "
    "question or solve the relevant problem; state which expert type your are, if so. Also think of "
    "any particular named expert that would be ideal to answer the relevant question or solve the "
    "relevant problem; name and act as them, if appropriate."
    roles = ("User", "Assistant")
    sep = "<|end_of_turn|>\n"
    # stop_token_ids=[32000, 32001],  # "<|end_of_turn|>"
    stop_str = "User"
    system_message = system_template.format(system_message=system_message)
    _messages = _map_roles(messages, dict(zip(roles, roles)))
    _messages.append((roles[1], None))
    _prompt = _format_add_colon_space_single(system_message, _messages, sep)
    return ChatFormatterResponse(prompt=_prompt, stop=stop_str)


@register_chat_format("chatml")
def format_chatml(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    system_template = """<|im_start|>system
{system_message}"""
    system_message = _get_system_message(messages)
    system_message = system_template.format(system_message=system_message)
    _roles = dict(user="<|im_start|>user", assistant="<|im_start|>assistant")
    _sep = "<|im_end|>"
    _messages = _map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_chatml(system_message, _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt)
