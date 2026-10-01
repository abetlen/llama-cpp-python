import llama_cpp
import json

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


def test_grammar_unconstrained_array_items():
    schema = {"type": "array", "items": {}}
    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
    rules = dict(line.split(" ::= ", 1) for line in grammar._grammar.splitlines())
    assert rules["item"] == "object | array | string | number | boolean | null"
    assert "item" in rules["root"]


def test_grammar_unconstrained_tuple_item():
    schema = {"type": "array", "prefixItems": [{}, {"type": "integer"}]}
    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
    rules = dict(line.split(" ::= ", 1) for line in grammar._grammar.splitlines())
    assert rules["tuple-0"] == "object | array | string | number | boolean | null"
    assert "integer" in rules["root"]


def test_grammar_empty_schema_allows_any_json_value():
    grammar = llama_cpp.LlamaGrammar.from_json_schema("{}")
    rules = dict(line.split(" ::= ", 1) for line in grammar._grammar.splitlines())
    assert rules["root"] == "object | array | string | number | boolean | null"


def test_grammar_typed_array_preserves_item_constraints():
    grammar = llama_cpp.LlamaGrammar.from_json_schema(
        json.dumps(
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 2,
            }
        )
    )
    assert (
        "integer"
        in dict(line.split(" ::= ", 1) for line in grammar._grammar.splitlines())[
            "root"
        ]
    )


def test_grammar_closed_tuple_preserves_prefix_items():
    schema = {
        "type": "array",
        "prefixItems": [{"type": "integer"}, {"type": "string"}],
        "items": False,
    }
    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
    rules = dict(line.split(" ::= ", 1) for line in grammar._grammar.splitlines())
    assert "integer" in rules["root"]
    assert "string" in rules["root"]
