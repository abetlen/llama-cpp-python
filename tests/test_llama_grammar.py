import llama_cpp
import json
import ctypes
from pathlib import Path

import pytest

from llama_cpp.llama_grammar import json_schema_to_gbnf

tree = """
leaf ::= "."
node ::= leaf | "(" node node ")"
root ::= node
"""


def test_grammar_from_string():
    grammar = llama_cpp.LlamaGrammar.from_string(tree)
    # assert grammar._n_rules == 3
    # assert grammar._start_rule_index == 2
    # assert grammar.grammar is not None


def test_composed_pydantic_grammar():
    """
    from pydantic import BaseModel

    class A(BaseModel):
        a: int

    class B(BaseModel):
        a: A
        b: int
    """

    # This schema corresponds to the grammar in the comment above.
    # We don't use the pydantic models directly to avoid the dependency.
    schema = {
        "$defs": {
            "A": {
                "properties": {"a": {"title": "A", "type": "integer"}},
                "required": ["a"],
                "title": "A",
                "type": "object",
            }
        },
        "properties": {
            "a": {"$ref": "#/$defs/A"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "B",
        "type": "object",
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))

    # assert grammar.grammar is not None


def test_grammar_anyof():
    sch = {
        "properties": {
            "temperature": {
                "description": "The temperature mentioned",
                "type": "number",
            },
            "unit": {
                "anyOf": [
                    {
                        "description": "Unit for temperature",
                        "enum": ["celsius", "fahrenheit"],
                        "type": "string",
                    },
                    {"type": "null"},
                ],
            },
        },
        "type": "object",
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(sch))

    # assert grammar.grammar is not None


@pytest.fixture(scope="module")
def grammar_accepts():
    llama_cpp.llama_backend_init()
    params = llama_cpp.llama_model_default_params()
    params.vocab_only = True
    path = (
        Path(__file__).resolve().parents[1]
        / "vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"
    )
    model = llama_cpp.llama_model_load_from_file(str(path).encode(), params)
    assert model
    try:
        vocab = llama_cpp.llama_model_get_vocab(model)
        pieces = {}
        buffer = ctypes.create_string_buffer(32)
        for token in range(llama_cpp.llama_vocab_n_tokens(vocab)):
            size = llama_cpp.llama_token_to_piece(
                vocab, token, buffer, len(buffer), 0, False
            )
            if size == 1:
                pieces[buffer.raw[:size]] = token

        def accepts(schema, text):
            grammar = json_schema_to_gbnf(json.dumps(schema))
            sampler = llama_cpp.llama_sampler_init_grammar(
                vocab, grammar.encode(), b"root"
            )
            assert sampler
            try:
                tokens = [pieces[bytes([byte])] for byte in text.encode()]
                tokens.append(llama_cpp.llama_vocab_eos(vocab))
                for token in tokens:
                    candidate = (llama_cpp.llama_token_data * 1)(
                        llama_cpp.llama_token_data(token, 0.0, 0.0)
                    )
                    candidates = llama_cpp.llama_token_data_array(
                        candidate, 1, -1, False
                    )
                    llama_cpp.llama_sampler_apply(sampler, ctypes.byref(candidates))
                    if candidate[0].logit == float("-inf"):
                        return False
                    llama_cpp.llama_sampler_accept(sampler, token)
                return True
            finally:
                llama_cpp.llama_sampler_free(sampler)

        yield accepts
    finally:
        llama_cpp.llama_model_free(model)


@pytest.mark.parametrize("type_form", ["single", "nullable", "null-first"])
@pytest.mark.parametrize(
    ("schema", "valid", "invalid"),
    [
        (
            {"type": "string", "minLength": 2, "maxLength": 3},
            ['"ab"', '"abc"'],
            ['"a"', '"abcd"'],
        ),
        (
            {"type": "string", "pattern": "^[ab]+$"},
            ['"a"', '"abba"'],
            ['""', '"c"'],
        ),
        (
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 2,
            },
            ["[1]", "[1,2]"],
            ["[]", "[1,2,3]", '["a"]'],
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            ['{"value":1}'],
            ["{}", '{"value":"a"}', '{"value":1,"extra":2}'],
        ),
        (
            {"type": "string", "format": "date"},
            ['"2026-11-25"'],
            ['"2026-13-25"', '"abc"'],
        ),
        (
            {"type": "array", "prefixItems": [{"type": "integer"}]},
            ["[1]"],
            ["[]", '["a"]', "[1,2]"],
        ),
    ],
)
def test_type_union_preserves_constraints(
    grammar_accepts, type_form, schema, valid, invalid
):
    if type_form != "single":
        types = [schema["type"], "null"]
        schema = {**schema, "type": types if type_form == "nullable" else types[::-1]}
    for text in valid:
        assert grammar_accepts(schema, text), text
    for text in invalid:
        assert not grammar_accepts(schema, text), text
    assert grammar_accepts(schema, "null") == (type_form != "single")


def test_type_union_applies_constraints_only_to_matching_type(grammar_accepts):
    schema = {"type": ["string", "integer"], "minLength": 2, "maxLength": 3}
    assert grammar_accepts(schema, '"ab"')
    assert grammar_accepts(schema, "1")
    assert not grammar_accepts(schema, '"a"')
    assert not grammar_accepts(schema, "null")


def test_nested_type_union_preserves_constraints(grammar_accepts):
    schema = {
        "$defs": {"name": {"type": ["string", "null"], "pattern": "^[ab]+$"}},
        "type": "array",
        "items": {"$ref": "#/$defs/name"},
    }
    assert grammar_accepts(schema, '[null,"abba"]')
    assert not grammar_accepts(schema, '["c"]')
