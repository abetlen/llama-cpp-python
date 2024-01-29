import llama_cpp
import json

tree = """
leaf ::= "."
node ::= leaf | "(" node node ")"
root ::= node
"""


def test_grammar_from_string():
    grammar = llama_cpp.LlamaGrammar.from_string(tree)
    assert grammar._n_rules == 3
    assert grammar._start_rule_index == 2
    assert grammar.grammar is not None


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

    assert grammar.grammar is not None


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

    assert grammar.grammar is not None