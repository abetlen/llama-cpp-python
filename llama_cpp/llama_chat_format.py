"""
llama_cpp/llama_chat_format.py

This module provides a chat formatting system that allows for custom templates and HuggingFace's jinja2-based chat templating.

To extend or customize, simply inherit from the ChatFormatter class and override the necessary methods. Registered formatters can be accessed using the ChatFormatterFactory.

NOTE: The system message is always assumed to be the first element in a sequence.

# Usage example:
# Registering a custom formatter
@ChatFormatterFactory.register_predefined_model("llama-2")
class Llama2Formatter(ChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)

# Obtaining a registered formatter
chat_formatter_factory = ChatFormatterFactory()
llama2_formatter = chat_formatter_factory.get_formatter_by_name("alpaca")

# Formatting messages
messages = [{"role": "user", "content": "Hello, World!"}]
response = llama2_formatter(messages)
print(response)
"""
import dataclasses
import os
from typing import Any, Dict, List, Optional, Protocol, Type, Union

from huggingface_hub import login
from transformers import AutoTokenizer

from . import llama_types

# NOTE: The default templates are defined here for reusability.
huggingface_template = {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "jinja": None,
    "tokenize": False,
}

common_template = {
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
    "include_prompt": False,
}

# Templates can be reused, modified, or overriden as needed on a model-by-model basis.
# This reduces noise in the code and ideally keeps the code base DRY.
llama2_template = {
    "roles": {
        "system": {
            "prefix": "<<SYS>>",
            "postfix": "<</SYS>>",
            "format": None,  # Optionally specify an custom format
        },
        "assistant": {
            "prefix": "",  # No prefix for assistant role by default
            "postfix": "",  # No postfix for assistant role by default
            "format": None,
        },
        "user": {
            "prefix": "[INST] ",
            "postfix": " [/INST]",  # Model starts generating from here
            "format": None,
        },
    }
}
# NOTE: The merge operator requires Python 3.9+
# Other options are to use `dict.update()` or to create a custom function that merges them.
# Source: https://docs.python.org/3/library/stdtypes.html?highlight=dict#dict
llama2_template |= common_template

# NOTE: If `include_prompt` is set to `True`, it will append the user prefix/postfix to the prompts output.
alpaca_template = {
    "roles": {
        "system": {
            "prefix": "",
            "postfix": "\n",
            "format": None,
        },
        "user": {
            "prefix": "### Instruction:\n",
            "postfix": "\n",
            "format": None,
        },
        "input": {
            "prefix": "### Input:\n",
            "postfix": "\n",
            "format": None,
        },
        "assistant": {
            "prefix": "### Response:\n",
            "postfix": "",  # Model starts generating from here
            "format": None,
        },
    }
}
alpaca_template |= common_template


@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[Union[str, List[str]]] = None


# Base Chat Formatter Protocol
class ChatFormatterInterface(Protocol):
    def __init__(self, template: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> ChatFormatterResponse:
        raise NotImplementedError


# Core Chat Formatter class
# NOTE: Methods can be overridden as needed on a model-by-model basis.
class ChatFormatter(ChatFormatterInterface):
    def __init__(self, template: Optional[Dict[str, Any]] = None):
        self.template = template or llama2_template

    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        formatted_messages = [
            self.format_message(msg["content"], msg["role"]) for msg in messages
        ]
        separator = self.format_separator("between_messages")
        formatted_sequence = separator.join(formatted_messages)
        # NOTE: Optionally include a prompt at the end
        if self.template["include_prompt"]:
            formatted_sequence += self.get_prompt()
        # NOTE: `stop` is handled within completion methods
        return ChatFormatterResponse(prompt=formatted_sequence)

    def format_message(self, message, role) -> str:
        """Format a message based on the specified role."""
        try:
            role_info = self.template["roles"][role]
        except KeyError:
            raise KeyError(
                f"The role '{role}' is not defined in the template. Please check your template configuration."
            )

        prefix = role_info.get("prefix", "")
        postfix = role_info.get("postfix", "")
        formatted_message = f"{prefix}{message}{postfix}"
        return formatted_message

    def format_separator(self, separator_type) -> str:
        """Format separators based on the specified type."""
        return self.template["separators"].get(separator_type, "")

    def format_special_token(self, token_type) -> str:
        """Format special tokens based on the specified type."""
        return self.template["special_tokens"].get(token_type, "")

    def get_prompt(self) -> str:
        # Implement logic to generate a prompt, if needed
        return self.template["roles"]["user"]["prefix"]


class TokenizerCache:
    _cache: Dict[str, AutoTokenizer] = {}

    @classmethod
    def get_tokenizer(cls, model_name: str) -> AutoTokenizer:
        if model_name not in cls._cache:
            cls._cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls._cache[model_name]


class AutoTokenizerFormatter(ChatFormatterInterface):
    def __init__(self, template: Optional[Dict[str, str]] = None):
        self.template = template or huggingface_template
        self.token = os.getenv("HF_TOKEN")
        if self.token is None:
            raise AttributeError(
                "Failed to login to huggingface. "
                "Did you forget to set the `HF_TOKEN` environment variable with your huggingface token?"
            )
        login(self.token)
        self.tokenizer = TokenizerCache.get_tokenizer(self.template["model"])

    def __call__(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs,
    ) -> ChatFormatterResponse:
        formatted_content = self.format_messages(messages)
        return ChatFormatterResponse(
            prompt=formatted_content, stop=[self.tokenizer.eos_token]
        )

    def format_messages(
        self, messages: List[llama_types.ChatCompletionRequestMessage]
    ) -> str:
        # If a custom template is provided, override the tokenizer's default template
        if self.template.get("jinja"):
            self.tokenizer.chat_template = self.template["jinja"]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=self.template["tokenize"]
        )


# NOTE: Template registration is currently a WIP (work in progress).
class FormatterNotFoundException(Exception):
    pass


# External developers can now use the `@ChatFormatter.register_predefined_model`
# method to register their own custom formatters.
class ChatFormatterFactory:
    _chat_formatters: Dict[str, ChatFormatterInterface] = {}

    @staticmethod
    def register_predefined_model(name: str):
        def decorator(cls: Type[ChatFormatterInterface]):
            ChatFormatterFactory._chat_formatters[name] = cls()
            return cls

        return decorator

    def register_custom_model(self, name: str, formatter: ChatFormatterInterface):
        self._chat_formatters[name] = formatter

    def get_formatter_by_name(self, name: str) -> ChatFormatterInterface:
        try:
            return self._chat_formatters[name]
        except KeyError:
            raise FormatterNotFoundException(
                f"Invalid chat format: {name} (valid formats: {list(self._chat_formatters.keys())})"
            )


# Define a chat format class and register it
@ChatFormatterFactory.register_predefined_model("llama-2")
class Llama2Formatter(ChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)


# Define a chat format class and register it
@ChatFormatterFactory.register_predefined_model("alpaca")
class AlpacaFormatter(ChatFormatter):
    def __init__(self):
        # Define the Alpaca template
        super().__init__(alpaca_template)


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
