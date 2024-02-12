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
        "required": ["a"],
        "title": "B",
        "type": "object",
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))

    assert grammar.grammar is not None

    assert (
        llama_cpp.llama_grammar.json_schema_to_gbnf(
            json.dumps(schema), treat_optional_as_nullable=False
        )
        == r"""space ::= " "?
integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) space
a-A ::= "{" space "\"a\"" space ":" space integer space "}" space
root ::= "{" space "\"a\"" space ":" space a-A ("," space "\"b\"" space ":" space integer)? space "}" space"""
    )

    assert (
        llama_cpp.llama_grammar.json_schema_to_gbnf(
            json.dumps(schema), treat_optional_as_nullable=True
        )
        == r"""space ::= " "?
integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) space
a-A ::= "{" space "\"a\"" space ":" space integer space "}" space
integer-or-null ::= (("-"? ([0-9] | [1-9] [0-9]*)) | "null") space
root ::= "{" space "\"a\"" space ":" space a-A "," space "\"b\"" space ":" space integer-or-null space "}" space"""
    )


def test_grammar_anyof():
    schema = {
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

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))

    assert grammar.grammar is not None

    assert (
        llama_cpp.llama_grammar.json_schema_to_gbnf(json.dumps(schema), None)
        == r"""space ::= " "?
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
unit-0 ::= "\"celsius\"" | "\"fahrenheit\""
null ::= "null" space
unit ::= unit-0 | null
root ::= "{" space "\"temperature\"" space ":" space number "," space "\"unit\"" space ":" space unit space "}" space"""
    )


def test_grammar_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "test": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"other": {"type": "string"}},
                "required": [],
            },
        },
        "required": ["test"],
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))

    assert grammar.grammar is not None

    assert (
        llama_cpp.llama_grammar.json_schema_to_gbnf(
            json.dumps(schema), treat_optional_as_nullable=False
        )
        == r"""space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\""  space
nested ::= "{" space "\"other\"" space ":" space string space "}" space
root ::= "{" space ("\"nested\"" space ":" space nested "," space)? "\"test\"" space ":" space string space "}" space"""
    )

    assert (
        llama_cpp.llama_grammar.json_schema_to_gbnf(
            json.dumps(schema), treat_optional_as_nullable=True
        )
        == r"""space ::= " "?
string-or-null ::= ( "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\""  | "null") space
nested-or-null ::= ("{" space "\"other\"" space ":" space string-or-null space "}" | "null") space
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\""  space
root ::= "{" space "\"nested\"" space ":" space nested-or-null "," space "\"test\"" space ":" space string space "}" space"""
    )
