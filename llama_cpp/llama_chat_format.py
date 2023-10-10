import dataclasses
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from . import llama_types

BASE_TEMPLATE = {
    "roles": {
        "system": {
            "prefix": "<<SYS>>",
            "postfix": "<</SYS>>",
            "format": None,
        },
        "user": {
            "prefix": "[INST] ",
            "postfix": " [/INST]",
            "format": None,
        },
        "assistant": {
            "prefix": "",
            "postfix": "",
            "format": None,
        },
    },
    "separators": {
        "after_system": "\n",
        "between_messages": "\n",
        "end_of_response": "",
    },
    "special_tokens": {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    },
    "default_termination": {
        "role": "assistant",
        "message": None,
    },
}


@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[Union[str, List[str]]] = None


class ChatFormatterTemplate(Protocol):
    def __init__(self, template: Dict[str, Any] = BASE_TEMPLATE):
        self.template = template

    # NOTE: Override private methods in inheriting classes as needed.
    def _get_system_message(
        self, messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]]
    ) -> str:
        """Get the first system message."""
        # NOTE: The system message is always the first element in a sequence,
        # any other order should be considered undefined.
        # If we always set the first element in the sequence to a system role,
        # it makes sense to simply check the first element and test to see if it is a system role.
        # This allows us to extract and return the system message from the list of messages
        # with a constant time complexity.
        try:
            if messages[0]["role"] == "system":
                # Retrieve role-specific formatting
                role_prefix = self.template["roles"]["system"]["prefix"]
                role_postfix = self.template["roles"]["system"]["postfix"]
                # Extract the role-based message content
                content = messages[0]["content"]
                # Format the message content with the role's prefix and postfix
                return role_prefix + content + role_postfix
            return ""
        except (IndexError, KeyError):
            return ""

    def _map_roles(
        self, messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]]
    ) -> List[Tuple[str, Optional[str]]]:
        """Map the message roles."""
        # Convert the messages into a list of (role, message) tuples
        mapped_sequence = []
        for message in messages:
            if message["role"] in ["user", "assistant"]:
                # Retrieve role-specific formatting
                role_prefix = self.template["roles"][message["role"]]["prefix"]
                role_postfix = self.template["roles"][message["role"]]["postfix"]
                # Format the message content with the role's prefix and postfix
                formatted_message = role_prefix + message["content"] + role_postfix
                # Map the formatted message to the sequence as a tuple
                mapped_sequence.append((message["role"], formatted_message))
        return mapped_sequence

    def _format_messages(
        self, messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]]
    ) -> str:
        """Transforms a list of messages into the appropriate format for the model."""
        ...

    def parse_response(
        self,
        messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]],
        **kwargs,
    ) -> ChatFormatterResponse:
        ...


class Llama2Formatter(ChatFormatterTemplate):
    def _format_messages(
        self, messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]]
    ) -> str:
        """Private method to format messages based on Llama2 template."""
        system_message = self._get_system_message(messages)
        mapped_messages = self._map_roles(messages)
        separator = self.template["separators"]["between_messages"]
        end_of_response = self.template["separators"]["end_of_response"]

        formatted_msg = separator.join([msg for role, msg in mapped_messages if msg])
        return system_message + separator + formatted_msg + end_of_response

    def parse_messages(
        self,
        messages: List[Dict[str, llama_types.ChatCompletionRequestMessage]],
        **kwargs,
    ) -> ChatFormatterResponse:
        """Parse messages and wrap in ChatFormatterResponse."""
        formatted_content = self._format_messages(messages)
        return ChatFormatterResponse(prompt=formatted_content)


class ChatFormatter:
    _chat_formatters: Dict[str, ChatFormatterTemplate] = {"llama-2": Llama2Formatter}

    def register_chat_format(self, cls, name: str):
        self._chat_formatters[name] = cls

    def get_chat_format(self, name: str):
        try:
            return self._chat_formatters[name]()
        except KeyError:
            valid_formats = list(self._chat_formatters.keys())
            raise ValueError(
                f"Invalid chat format: {name}. Valid formats: {valid_formats}"
            )

    def format(self, name: str, messages: List[dict]) -> str:
        formatter = self.get_chat_format(name)
        return formatter.format_messages(messages)

    def parse(self, name: str, raw_response: str) -> Tuple[str, List[dict]]:
        formatter = self.get_chat_format(name)
        return formatter.parse_response(raw_response)


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


# eg, export HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1
@register_chat_format("autotokenizer")
def format_autotokenizer(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    # https://huggingface.co/docs/transformers/main/chat_templating
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
    import os

    from transformers import AutoTokenizer

    huggingFaceModel = os.getenv("HF_MODEL")  # eg, mistralai/Mistral-7B-Instruct-v0.1
    print(huggingFaceModel)
    if not huggingFaceModel:
        raise Exception(
            "HF_MODEL needs to be set in env to use chat format 'autotokenizer'"
        )
    tokenizer = AutoTokenizer.from_pretrained(huggingFaceModel)
    tokenizer.use_default_system_prompt = False
    _prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    # Return formatted prompt and eos token by default
    return ChatFormatterResponse(prompt=_prompt, stop=tokenizer.eos_token)


@register_chat_format("functionary")
def format_functionary(
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunctions]] = None,
    **kwargs: Any,
) -> ChatFormatterResponse:
    SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""

    def generate_schema_from_functions(
        functions: List[llama_types.ChatCompletionFunctions],
        namespace: str = "functions",
    ):
        """
        Convert functions schema to a schema that language models can understand.
        """

        schema = (
            "// Supported function definitions that should be called when necessary.\n"
        )
        schema += f"namespace {namespace} {{\n\n"

        for function in functions:
            # Convert a Function object to dict, if necessary
            function_name = function["name"]
            description = function.get("description", "")
            schema += f"// {description}\n"
            schema += f"type {function_name}"

            parameters = function.get("parameters", None)
            schema += " = (_: {\n"
            required_params = parameters.get("required", [])
            for param_name, param in parameters.get("properties", {}).items():
                # Param Description
                description = param.get("description")
                if description is not None:
                    schema += f"// {description}\n"

                # Param Name
                schema += f"{param_name}"
                if param_name not in required_params:
                    schema += "?"

                # Param Type
                param_type = param.get("type", "any")
                if param_type == "integer":
                    param_type = "number"
                if "enum" in param:
                    param_type = " | ".join([f'"{v}"' for v in param["enum"]])
                schema += f": {param_type},\n"

            schema += "}) => any;\n\n"

        schema += f"}} // namespace {namespace}"

        return schema

    def prepare_messages_for_inference(
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunctions]] = None,
    ):
        all_messages: List[llama_types.ChatCompletionRequestMessage] = []
        if functions is not None:
            all_messages.append(
                llama_types.ChatCompletionRequestMessage(
                    role="system", content=generate_schema_from_functions(functions)
                )
            )

        all_messages.append(
            llama_types.ChatCompletionRequestMessage(
                role="system", content=SYSTEM_MESSAGE
            )
        )

        for message in messages:
            # Function call responses
            if message["role"] == "function" and "name" in message:
                message["name"] = f"functions.{message['name']}"
            # Function call requests by assistant
            if "function_call" in message:
                message["function_call"][
                    "name"
                ] = f"functions.{message['function_call']['name']}"
            all_messages.append(message)

        all_messages.append(
            llama_types.ChatCompletionRequestMessage(role="assistant", content=None)
        )

        def message_to_str(msg: llama_types.ChatCompletionRequestMessage):
            if msg["role"] == "system":
                return f"system:\n{msg['content']}\n"

            elif msg["role"] == "function" and "name" in msg:
                return f"function name={msg['name']}:\n{msg['content']}\n"
            elif msg["role"] == "user":
                if msg["content"] is None:
                    return "user:\n</s>"
                else:
                    return f"user:\n</s>{msg['content']}\n"
            elif msg["role"] == "assistant":
                if msg["content"] is not None and "function_call" in msg:
                    return f"assistant:\n{msg['content']}\nassistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>"
                elif "function_call" in msg:
                    return f"assistant to={msg['function_call']['name']}:\n{msg['function_call']['arguments']}</s>"
                elif msg["content"] is None:
                    return "assistant"
                else:
                    return f"assistant:\n{msg['content']}\n"
            else:
                raise ValueError(f"Unsupported role: {msg['role']}")

        return "".join([message_to_str(msg) for msg in all_messages])

    prompt = prepare_messages_for_inference(messages, functions)
    return ChatFormatterResponse(
        prompt=prompt,
        stop=["user:", "</s>"],
    )
