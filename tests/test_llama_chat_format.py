import json

from llama_cpp import (
    ChatCompletionRequestUserMessage,
)
from llama_cpp.llama_chat_format import hf_tokenizer_config_to_chat_formatter


mistral_7b_tokenizer_config = """{
  "add_bos_token": true,
  "add_eos_token": false,
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "additional_special_tokens": [],
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "</s>",
  "legacy": true,
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": null,
  "sp_model_kwargs": {},
  "spaces_between_special_tokens": false,
  "tokenizer_class": "LlamaTokenizer",
  "unk_token": "<unk>",
  "use_default_system_prompt": false,
  "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
}"""


def test_hf_tokenizer_config_str_to_chat_formatter():
    tokenizer_config = json.loads(mistral_7b_tokenizer_config)
    chat_formatter = hf_tokenizer_config_to_chat_formatter(
        tokenizer_config
    )
    chat_formatter_respoonse = chat_formatter(
        messages=[
            ChatCompletionRequestUserMessage(role="user", content="Hello, world!"),
        ]
    )

    assert chat_formatter_respoonse.prompt == ("<s>[INST] Hello, world! [/INST]</s>" "")

import uuid
from jinja2 import Template
from typing import List, Tuple, Any

def render_and_strip_images(template_string: str, **template_args: Any) -> Tuple[str, List[int]]:
    # Placeholder for images
    image_placeholder = uuid.uuid4().hex

    # Render the template
    template = Template(template_string)
    rendered = template.render(image=image_placeholder, **template_args)

    # Find positions of image placeholders and remove them
    positions: List[int] = []
    current_pos = 0

    while True:
        pos = rendered.find(image_placeholder, current_pos)
        if pos == -1:
            break
        # Adjust position for previously removed placeholders
        adjusted_pos = pos - len(image_placeholder) * len(positions)
        positions.append(adjusted_pos)
        current_pos = pos + len(image_placeholder)

    # Clean the rendered string
    clean_rendered = rendered.replace(image_placeholder, '')

    return clean_rendered, positions
  

def test_render_and_strip_images():
    template_string = "Hello, {{name}}! {{image}} How are you? {{image}}"
    template_args = {"name": "world"}

    # Sanity test
    assert Template(template_string).render({ "name": "world", "image": ""}) == "Hello, world!  How are you? "

    rendered, positions = render_and_strip_images(template_string, **template_args)

    assert rendered == "Hello, world!  How are you? "
    assert positions == [14, 28]