import json
import re

import pytest

from llama_cpp.llama_grammar import SchemaConverter, json_schema_to_gbnf


def _rules(schema):
    grammar = json_schema_to_gbnf(json.dumps(schema))
    return dict(line.split(" ::= ", 1) for line in grammar.splitlines())


def _resolve_alias(rules, name):
    seen = set()
    while re.fullmatch(r"[a-zA-Z0-9-]+", rules[name]):
        assert name not in seen
        seen.add(name)
        name = rules[name]
    return rules[name]


def _assert_defined_rules(rules):
    for name, body in rules.items():
        assert re.fullmatch(r"[a-zA-Z0-9-]+", name)
        assert body
        body = re.sub(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]', "", body)
        assert set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", body)) <= rules.keys()


@pytest.mark.parametrize("names", [("Item", "Item"), ("a_b", "a-b"), ("", "Item")])
def test_grammar_distinct_references_with_colliding_names(names):
    left, right = names
    schema = {
        "anyOf": [
            {"$ref": f"#/$defs/left/$defs/{left}"},
            {"$ref": f"#/$defs/right/$defs/{right}"},
        ],
        "$defs": {
            "left": {"$defs": {left: {"const": "left"}}},
            "right": {"$defs": {right: {"const": "right"}}},
        },
    }
    rules = _rules(schema)
    assert rules["root"] == "alternative-0 | alternative-1"
    assert _resolve_alias(rules, "alternative-0") == '"\\"left\\""'
    assert _resolve_alias(rules, "alternative-1") == '"\\"right\\""'
    _assert_defined_rules(rules)


@pytest.mark.parametrize("name", ["string", "integer", "space"])
def test_grammar_references_do_not_reuse_primitive_rules(name):
    first = {"const": 0} if name == "space" else {"type": name}
    schema = {
        "type": "object",
        "properties": {
            "primitive": first,
            "constant": {"$ref": f"#/$defs/{name}"},
        },
        "required": ["primitive", "constant"],
        "$defs": {name: {"const": 42}},
    }
    rules = _rules(schema)
    assert _resolve_alias(rules, "constant") == '"42"'
    if name != "space":
        assert rules["primitive-kv"].endswith(" " + name)
    _assert_defined_rules(rules)


def test_grammar_reuses_identical_references():
    schema = {
        "anyOf": [{"$ref": "#/$defs/Item"}, {"$ref": "#/$defs/Item"}],
        "$defs": {"Item": {"const": 42}},
    }
    rules = _rules(schema)
    assert rules["alternative-0"] == rules["alternative-1"]
    assert sum(body == '"42"' for body in rules.values()) == 1
    _assert_defined_rules(rules)


@pytest.mark.parametrize("name", ["Node", "node_name", "integer"])
def test_grammar_recursive_references_have_valid_names(name):
    ref = f"#/$defs/{name}"
    schema = {
        "anyOf": [{"$ref": ref}],
        "$defs": {name: {"type": "object", "properties": {"next": {"$ref": ref}}}},
    }
    _assert_defined_rules(_rules(schema))


def test_grammar_recursive_references_keep_distinct_identities():
    left_ref = "#/$defs/left/$defs/Node"
    right_ref = "#/$defs/right/$defs/Node"
    schema = {
        "anyOf": [{"$ref": left_ref}],
        "$defs": {
            "left": {
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "tag": {"const": "left"},
                            "next": {"$ref": right_ref},
                        },
                    }
                }
            },
            "right": {
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "tag": {"const": "right"},
                            "next": {"$ref": left_ref},
                        },
                    }
                }
            },
        },
    }
    converter = SchemaConverter(
        prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False
    )
    converter.resolve_refs(schema, "stdin")
    left_name = converter._resolve_ref("stdin" + left_ref)
    right_name = converter._resolve_ref("stdin" + right_ref)
    assert left_name != right_name
    assert converter._resolve_ref("stdin" + left_ref) == left_name
    assert converter._resolve_ref("stdin" + right_ref) == right_name
    assert converter._rules[left_name + "-next"] == right_name
    assert converter._rules[right_name + "-next"] == left_name
    _assert_defined_rules(converter._rules)
