import dataclasses
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from . import llama_types

# NOTE: Custom Templates use Jinja2.
# If no template is given, then should default to hf's tokenizer template.
# We can define the model and template on a model-to-model basis,
# however, this should be allowed to be overridden for flexibility and extensibility.
# We only need 2 keys, the model name and the jinja2 template.
#
#   template = {"model": "meta-llama/Llama-2-7b-chat-hf", "template": None}
#
#   or
#
# chat_template = {
#     "model": "meta-llama/Llama-2-7b-chat-hf",
#     "jinja": "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '[ASST] '  + message ['content'] + ' [/ASST]' + eos_token }}{% endif %}{% endfor %}",
# }
#
# We can probably employ some kind of method for reading a template it in from a file in necessary.
#
# We leave template empty here because HuggingFace defined it already.
#
# Source: https://huggingface.co/docs/transformers/main/chat_templating
#
# Special Thanks and Credit goes to bioshazard for the idea and preliminary implementation.
# Source: https://github.com/abetlen/llama-cpp-python/pull/790


# NOTE: We can still use this for reverse compatibility with the currently employed API.
# This can be modified, if needed, in the future.
@dataclasses.dataclass
class ChatFormatterResponse:
    prompt: str
    stop: Optional[List[str]] = None


class TokenizerCache:
    _cache: Dict[str, AutoTokenizer] = {}

    @classmethod
    def get_tokenizer(cls, model_name: str) -> AutoTokenizer:
        if model_name not in cls._cache:
            cls._cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls._cache[model_name]


class ChatFormatterTemplate:
    def __init__(self, template: Optional[Dict[str, str]] = None):
        if template:
            self.template = template
        else:
            self.template = {
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "jinja": None,
                "tokenize": False,
            }
        self.tokenizer = TokenizerCache.get_tokenizer(self.template["model"])

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        # If a custom template is provided, override the tokenizer's default template
        if self.template.get("jinja"):
            self.tokenizer.chat_template = self.template["jinja"]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=self.template["tokenize"]
        )

    def parse_response(self, messages: List[Dict[str, str]]) -> ChatFormatterResponse:
        formatted_content = self._format_messages(messages)
        return ChatFormatterResponse(
            prompt=formatted_content, stop=[self.tokenizer.eos_token]
        )


class ChatFormatter:
    _chat_formatters: Dict[str, ChatFormatterTemplate] = {}

    def register_chat_format(
        self, model_name: str, template: Optional[Dict[str, str]] = None
    ):
        self._chat_formatters[model_name] = ChatFormatterTemplate(template)

    def get_chat_format(self, model_name: str) -> ChatFormatterTemplate:
        if model_name not in self._chat_formatters:
            raise ValueError(f"Model {model_name} is not registered.")

        return self._chat_formatters[model_name]

    def format(self, model_name: str, messages: List[Dict[str, str]]) -> str:
        formatter = self.get_chat_format(model_name)
        return formatter._format_messages(messages)

    def parse(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> ChatFormatterResponse:
        formatter = self.get_chat_format(model_name)
        return formatter.parse_response(messages)


# NOTE: Template registration is currently a WIP (work in progress)
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
