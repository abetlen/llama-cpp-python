"""
llama_cpp/llama_chat_format.py

This module provides a chat formatting system that allows for custom templates and HuggingFace's jinja2-based chat templating.

To extend or customize, simply inherit from the ChatFormatter class and override the necessary methods. Registered formatters can be accessed using the ChatFormatterFactory.

NOTE: The system message is always assumed to be the first element in a sequence.

NOTE: Users should avoid tampering with special tokens to prevent model issues.

---

# IMPORTANT NOTES:

- The use of the merge operator (|) for dictionaries requires Python 3.9 or higher. Keep in mind that llama-cpp-python supports Python 3.8 and later versions. If you are working with an earlier Python version, consider alternatives such as `dict.update()` or creating a custom function to merge dictionaries. For Python 3.9 or higher, the merge operator simplifies dictionary merging.
Source: https://docs.python.org/3/library/stdtypes.html?highlight=dict#dict

- Special tokens are crucial for the model's underlying operations, impacting pre-training, fine-tuning, and low-level inference processes. Users should avoid modifying special tokens to prevent issues in the model's output during inference. These issues may manifest as token fixation, repetitive language patterns, contextual derailment, and hallucinations. Improper use of separators and templates can exacerbate these problems.

Example using the llama-2 model and its templating schema:

#  1  <<SYS>>My name is Llama and I am a helpful assistant.<</SYS>>$
#  2  [INST] Hello Llama, my name is User. What's your name? [/INST]$
#  3  Hello User, my name is Llama. Nice to meet you!$
#  4  [INST] What can you do? [/INST]$
#  5  I can assist you with various tasks, including providing structured output for certain queries.$
#  6  [INST] How can you assist me in my programming projects? [/INST]$
#  7  $

This initial example is a proper template format that the model understands. It results in proper output and does not confuse the model.

#  1  <<SYS>>My name is Llama and I am a helpful assistant.<</SYS>>$
#  2  <s>[INST] Hello Llama, my name is User. What's your name? [/INST]$
#  3  Hello User, my name is Llama. Nice to meet you!</s>$
#  4  <s>[INST] What can you do? [/INST]$
#  5  I can assist you with various tasks, including providing structured output for certain queries.</s>$
#  6  <s>[INST] How can you assist me in my programming projects? [/INST]$
#  7  $

This example includes the use of special tokens, and the model may or may not use these tokens as a result. The model is not expecting them during inference, which causes unexpected behavior.

#  1  <<SYS>>My name is Llama and I am a helpful assistant.<</SYS>>$
#  2  $
#  3  <s>[INST] Hello Llama, my name is User. What's your name? [/INST]$
#  4  Hello User, my name is Llama. Nice to meet you!</s>$
#  5  $
#  6  <s>[INST] What can you do? [/INST]$
#  7  I can assist you with various tasks, including providing structured output for certain queries.</s>$
#  8  $
#  9  <s>[INST] How can you assist me in my programming projects? [/INST]$
# 10  $

This example is improperly formatted and causes the model to become confused. The model begins to fixate on tokens, uses language repetition, and eventually derails.

---

# Usage example:
# Registering a custom formatter
@ChatFormatterFactory.register_predefined_model("llama-2")
class Llama2Formatter(ChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)

# Obtaining a registered formatter
chat_formatter_factory = ChatFormatterFactory()
llama2_formatter = chat_formatter_factory.get_formatter_by_name("llama-2")

# Formatting messages
messages = [{"role": "user", "content": "Hello, World!"}]
response = llama2_formatter(messages)
print(response)
"""
import dataclasses
import os
from typing import Any, Dict, List, Optional, Protocol, Type, Union

import huggingface_hub
from transformers import AutoTokenizer

from . import llama_types

# Default chat formatting templates for reusability.
# These templates can be reused or modified on a model-by-model basis.

# Template for HuggingFace-based models.
huggingface_template = {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "jinja": None,
    "tokenize": False,
}

# Common formatting settings applicable to all roles in chat models.
common_template: llama_types.CommonTemplate = {
    "separators": {
        "after_system": "\n",
        "between_messages": "\n",
        "end_of_response": "",
    },
    "default_termination": {
        "role": "assistant",  # Default role for termination
        "message": None,  # Default termination message (None for assistant)
    },
    "include_prompt": False,  # Whether to include user prefix/postfix in prompts
}

# Template for Llama-2 model.
llama2_template: llama_types.ChatMLTemplate = {
    "roles": {
        "system": {
            "prefix": "<<SYS>>",  # System message prefix
            "postfix": "<</SYS>>",  # System message postfix
            "format": None,  # Optionally specify a custom format
        },
        "user": {
            "prefix": "[INST] ",
            "postfix": " [/INST]",  # Model generates from here
            "format": None,
        },
        "assistant": {
            "prefix": "",  # No prefix for assistant role by default
            "postfix": "",  # No postfix for assistant role by default
            "format": None,  # Custom format for assistant (if needed)
        },
    }
}
# Merge common settings into the llama2_template to reduce code duplication.
llama2_template |= common_template

# Template for Alpaca model.
alpaca_template: llama_types.ChatMLTemplate = {
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
            "postfix": "",  # Model generates from here
            "format": None,
        },
    }
}
alpaca_template |= common_template

# Template for Vicuna model.
# NOTE: The v0 template differs from the v1.1, v1.3, and v1.5.
# This is the v1.5 Vicuna Template.
vicuna_template: llama_types.ChatMLTemplate = {
    "roles": {
        "system": {
            "prefix": "",
            "postfix": "\n",
            "format": None,
        },
        "user": {
            "prefix": "USER: ",
            "postfix": "",
            "format": None,
        },
        "assistant": {
            "prefix": "ASSISTANT: ",  # Model generates from here
            "postfix": "",
            "format": None,
        },
    }
}
vicuna_template |= common_template

# NOTE: Open Assistant uses multiple custom prompts.
# The oasst-llama hybrids utilize ChatML templates.
# The base template is defined here for convenience.
oasst_template: llama_types.ChatMLTemplate = {
    "roles": {
        "system": {
            "prefix": "<|system|>",
            "postfix": "<|endoftext|>",
            "format": None,
        },
        "user": {
            "prefix": "<|prompter|>",
            "postfix": "<|endoftext|>",
            "format": None,
        },
        "assistant": {
            "prefix": "<|assistant|>",  # Model generates from here
            "postfix": "<|endoftext|>",
            "format": None,
        },
    }
}
oasst_template |= common_template


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
        self.huggingface_login()
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

    def huggingface_login(self) -> None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise AttributeError(
                "Failed to login to huggingface. "
                "Did you forget to set the `HF_TOKEN` environment variable with your huggingface token?"
            )
        huggingface_hub.login(token)

    def format_messages(
        self, messages: List[llama_types.ChatCompletionRequestMessage]
    ) -> str:
        # If a custom template is provided, override the tokenizer's default template
        if self.template.get("jinja"):
            self.tokenizer.chat_template = self.template["jinja"]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=self.template.get("tokenize", False)
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


@ChatFormatterFactory.register_predefined_model("vicuna")
class VicunaFormatter(ChatFormatter):
    def __init__(self):
        # Define the Vicuna template
        super().__init__(vicuna_template)


# NOTE: Refer to `oasst_template` note for more information.
@ChatFormatterFactory.register_predefined_model("oasst")
class OpenAssistantFormatter(ChatFormatter):
    def __init__(self):
        # Define the Open Assistant template
        super().__init__(oasst_template)


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
