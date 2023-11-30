"""
llama_cpp/llama_jinja_format.py
"""
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import jinja2
from jinja2 import Template

# NOTE: We sacrifice readability for usability.
# It will fail to work as expected if we attempt to format it in a readable way.
llama2_template = """{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}\n{% elif message['role'] == 'system' %}<<SYS>> {{ message['content'] }} <</SYS>>\n{% endif %}{% endfor %}"""


class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()


@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[Union[str, List[str]]] = None


# Base Chat Formatter Protocol
class ChatFormatterInterface(Protocol):
    def __init__(self, template: Optional[object] = None):
        ...

    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> ChatFormatterResponse:
        ...

    @property
    def template(self) -> str:
        ...


class AutoChatFormatter(ChatFormatterInterface):
    def __init__(
        self,
        template: Optional[str] = None,
        template_class: Optional[Template] = None,
    ):
        if template is not None:
            self._template = template
        else:
            self._template = llama2_template  # default template

        self._environment = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(
            self._template,
            template_class=template_class,
        )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        formatted_sequence = self._environment.render(messages=messages, **kwargs)
        return ChatFormatterResponse(prompt=formatted_sequence)

    @property
    def template(self) -> str:
        return self._template


class FormatterNotFoundException(Exception):
    pass


class ChatFormatterFactory(Singleton):
    _chat_formatters: Dict[str, Callable[[], ChatFormatterInterface]] = {}

    def register_formatter(
        self,
        name: str,
        formatter_callable: Callable[[], ChatFormatterInterface],
        overwrite=False,
    ):
        if not overwrite and name in self._chat_formatters:
            raise ValueError(
                f"Formatter with name '{name}' is already registered. Use `overwrite=True` to overwrite it."
            )
        self._chat_formatters[name] = formatter_callable

    def unregister_formatter(self, name: str):
        if name in self._chat_formatters:
            del self._chat_formatters[name]
        else:
            raise ValueError(f"No formatter registered under the name '{name}'.")

    def get_formatter_by_name(self, name: str) -> ChatFormatterInterface:
        try:
            formatter_callable = self._chat_formatters[name]
            return formatter_callable()
        except KeyError:
            raise FormatterNotFoundException(
                f"Invalid chat format: {name} (valid formats: {list(self._chat_formatters.keys())})"
            )


# Define a chat format class
class Llama2Formatter(AutoChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)


# With the Singleton pattern applied, regardless of where or how many times
# ChatFormatterFactory() is called, it will always return the same instance
# of the factory, ensuring that the factory's state is consistent throughout
# the application.
ChatFormatterFactory().register_formatter("llama-2", Llama2Formatter)
