#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "fastapi",
#   "jinja2",
#   "llama-cpp-python",
#   "numpy",
#   "openai",
#   "pydantic",
#   "uvicorn",
#   "websockets",
# ]
# ///

from __future__ import annotations

import abc
import os
import re
import json
import math
import time
import uuid
import queue
import ctypes
import fnmatch
import asyncio
import argparse
import threading
import multiprocessing
import copy
import shutil

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from collections import OrderedDict, deque
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import (
    CompletionChoice,
    Logprobs as CompletionLogprobs,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice as ChatCompletionChoice,
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChatCompletionChunkChoice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCallFunction,
    ChoiceDeltaToolCall,
    ChoiceLogprobs as ChatCompletionChunkChoiceLogprobs,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall as ChatCompletionMessageFunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ChatCompletionMessageToolCallFunction,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Deque,
    Literal,
    Iterator,
    TypedDict,
    cast,
    AsyncIterator,
)

import jinja2
import uvicorn
import numpy as np

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from jinja2.sandbox import ImmutableSandboxedEnvironment

from pydantic_core import from_json
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llama_cpp import llama_cpp  # noqa: E402


JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
"""


class JsonSchemaConverter:
    class BuiltinRule:
        def __init__(self, content: str, deps: Optional[List[str]] = None):
            self.content = content
            self.deps = deps or []

    SPACE_RULE = '" "?'
    INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
    GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
    GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}
    DOTALL = "[\\U00000000-\\U0010FFFF]"
    DOT = "[^\\x0A\\x0D]"
    NON_LITERAL_SET = set("|.()[]{}*+?")
    ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set("[]()|{}*+?")

    PRIMITIVE_RULES: Optional[Dict[str, "JsonSchemaConverter.BuiltinRule"]] = None
    STRING_FORMAT_RULES: Optional[Dict[str, "JsonSchemaConverter.BuiltinRule"]] = None
    RESERVED_NAMES: Optional[set[str]] = None

    @staticmethod
    def _build_repetition(
        item_rule: str,
        min_items: int,
        max_items: Optional[int],
        separator_rule: Optional[str] = None,
        item_rule_is_literal: bool = False,
    ) -> str:
        if not separator_rule:
            if min_items == 0 and max_items == 1:
                return f"{item_rule}?"
            if min_items == 1 and max_items is None:
                return f"{item_rule}+"

        result = ""

        if min_items > 0:
            if item_rule_is_literal and separator_rule is None:
                result = '"' + (item_rule[1:-1] * min_items) + '"'
            else:
                result = (f" {separator_rule} " if separator_rule else " ").join(
                    [item_rule] * min_items
                )

        def opt_repetitions(up_to_n: int, prefix_with_sep: bool = False) -> str:
            content = (
                f"{separator_rule} {item_rule}"
                if prefix_with_sep and separator_rule
                else item_rule
            )
            if up_to_n == 0:
                return ""
            if up_to_n == 1:
                return f"({content})?"
            if separator_rule and not prefix_with_sep:
                return f"({content} {opt_repetitions(up_to_n - 1, prefix_with_sep=True)})?"
            return (f"({content} " * up_to_n).rstrip() + (")?" * up_to_n)

        if min_items > 0 and max_items != min_items:
            result += " "

        if max_items is not None:
            result += opt_repetitions(max_items - min_items, prefix_with_sep=min_items > 0)
        else:
            item_operator = f"({separator_rule + ' ' if separator_rule else ''}{item_rule})"
            if min_items == 0 and separator_rule:
                result = f"({item_rule} {item_operator}*)?"
            else:
                result += f"{item_operator}*"

        return result

    @classmethod
    def _primitive_rules(cls) -> Dict[str, "JsonSchemaConverter.BuiltinRule"]:
        if cls.PRIMITIVE_RULES is None:
            up_to_15_digits = cls._build_repetition("[0-9]", 0, 15)
            cls.PRIMITIVE_RULES = {
                "boolean": cls.BuiltinRule('("true" | "false") space', []),
                "decimal-part": cls.BuiltinRule("[0-9] " + up_to_15_digits, []),
                "integral-part": cls.BuiltinRule(
                    "[0-9] | [1-9] " + up_to_15_digits,
                    [],
                ),
                "number": cls.BuiltinRule(
                    '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
                    ["integral-part", "decimal-part"],
                ),
                "integer": cls.BuiltinRule('("-"? integral-part) space', ["integral-part"]),
                "value": cls.BuiltinRule(
                    "object | array | string | number | boolean | null",
                    ["object", "array", "string", "number", "boolean", "null"],
                ),
                "object": cls.BuiltinRule(
                    '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
                    ["string", "value"],
                ),
                "array": cls.BuiltinRule(
                    '"[" space ( value ("," space value)* )? "]" space',
                    ["value"],
                ),
                "uuid": cls.BuiltinRule(
                    r'"\"" '
                    + ' "-" '.join("[0-9a-fA-F]" * n for n in [8, 4, 4, 4, 12])
                    + r' "\"" space',
                    [],
                ),
                "char": cls.BuiltinRule(
                    r'[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
                    [],
                ),
                "string": cls.BuiltinRule(r'"\"" char* "\"" space', ["char"]),
                "null": cls.BuiltinRule('"null" space', []),
            }
        return cls.PRIMITIVE_RULES

    @classmethod
    def _string_format_rules(cls) -> Dict[str, "JsonSchemaConverter.BuiltinRule"]:
        if cls.STRING_FORMAT_RULES is None:
            cls.STRING_FORMAT_RULES = {
                "date": cls.BuiltinRule(
                    '[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
                    [],
                ),
                "time": cls.BuiltinRule(
                    '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
                    [],
                ),
                "date-time": cls.BuiltinRule('date "T" time', ["date", "time"]),
                "date-string": cls.BuiltinRule('"\\"" date "\\"" space', ["date"]),
                "time-string": cls.BuiltinRule('"\\"" time "\\"" space', ["time"]),
                "date-time-string": cls.BuiltinRule(
                    '"\\"" date-time "\\"" space',
                    ["date-time"],
                ),
            }
        return cls.STRING_FORMAT_RULES

    @classmethod
    def _reserved_names(cls) -> set[str]:
        if cls.RESERVED_NAMES is None:
            cls.RESERVED_NAMES = set(
                ["root", "dot", *cls._primitive_rules().keys(), *cls._string_format_rules().keys()]
            )
        return cls.RESERVED_NAMES

    def __init__(
        self,
        *,
        prop_order: Dict[str, int],
        allow_fetch: bool,
        dotall: bool,
        raw_pattern: bool,
    ):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules: Dict[str, str] = {"space": self.SPACE_RULE}
        self._refs: Dict[str, Any] = {}
        self._refs_being_resolved: set[str] = set()

    def _format_literal(self, literal: str) -> str:
        escaped = self.GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda match: self.GRAMMAR_LITERAL_ESCAPES[match.group(0)],
            literal,
        )
        return f'"{escaped}"'

    def _add_rule(self, name: str, rule: str) -> str:
        escaped_name = self.INVALID_RULE_CHARS_RE.sub("-", name)
        if escaped_name not in self._rules or self._rules[escaped_name] == rule:
            key = escaped_name
        else:
            suffix = 0
            while (
                f"{escaped_name}{suffix}" in self._rules
                and self._rules[f"{escaped_name}{suffix}"] != rule
            ):
                suffix += 1
            key = f"{escaped_name}{suffix}"
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: Dict[str, Any], url: str) -> Dict[str, Any]:
        def visit(node: Any) -> Any:
            if isinstance(node, list):
                return [visit(child) for child in node]
            if isinstance(node, dict):
                ref = node.get("$ref")
                if ref is not None and ref not in self._refs:
                    if ref.startswith("https://"):
                        raise ValueError("remote schema fetch is not allowed")
                    elif ref.startswith("#/"):
                        target = schema
                        ref = f"{url}{ref}"
                        node["$ref"] = ref
                    else:
                        raise ValueError(f"Unsupported ref {ref}")

                    for selector in ref.split("#")[-1].split("/")[1:]:
                        assert target is not None and selector in target, (
                            f"Error resolving ref {ref}: {selector} not in {target}"
                        )
                        target = target[selector]
                    self._refs[ref] = target
                else:
                    for value in node.values():
                        visit(value)
            return node

        return cast(Dict[str, Any], visit(schema))

    def _generate_union_rule(self, name: str, alt_schemas: List[Dict[str, Any]]) -> str:
        return " | ".join(
            self.visit(alt_schema, f"{name}{'-' if name else 'alternative-'}{index}")
            for index, alt_schema in enumerate(alt_schemas)
        )

    def _visit_pattern(self, pattern: str, name: str) -> str:
        assert pattern.startswith("^") and pattern.endswith("$"), (
            'Pattern must start with "^" and end with "$"'
        )
        pattern = pattern[1:-1]
        sub_rule_ids: Dict[str, str] = {}
        index = 0
        length = len(pattern)

        def to_rule(item: Tuple[str, bool]) -> str:
            text, is_literal = item
            return f'"{text}"' if is_literal else text

        def transform() -> Tuple[str, bool]:
            nonlocal index
            start = index
            sequence: List[Tuple[str, bool]] = []

            def get_dot() -> str:
                rule = self.DOTALL if self._dotall else self.DOT
                return self._add_rule("dot", rule)

            def join_sequence() -> Tuple[str, bool]:
                if len(sequence) == 1:
                    return sequence[0]
                return (" ".join(to_rule(item) for item in sequence), False)

            while index < length:
                char = pattern[index]
                if char == ".":
                    sequence.append((get_dot(), False))
                    index += 1
                elif char == "(":
                    index += 1
                    if index < length:
                        assert pattern[index] != "?", (
                            f'Unsupported pattern syntax "{pattern[index]}" at index {index} of /{pattern}/'
                        )
                    sequence.append((f"({to_rule(transform())})", False))
                elif char == ")":
                    index += 1
                    assert start > 0 and pattern[start - 1] == "(", (
                        f"Unbalanced parentheses; start = {start}, index = {index}, pattern = {pattern}"
                    )
                    return join_sequence()
                elif char == "[":
                    square_brackets = char
                    index += 1
                    while index < length and pattern[index] != "]":
                        if pattern[index] == "\\":
                            square_brackets += pattern[index : index + 2]
                            index += 2
                        else:
                            square_brackets += pattern[index]
                            index += 1
                    assert index < length, (
                        f"Unbalanced square brackets; start = {start}, index = {index}, pattern = {pattern}"
                    )
                    square_brackets += "]"
                    index += 1
                    sequence.append((square_brackets, False))
                elif char == "|":
                    sequence.append(("|", False))
                    index += 1
                elif char in ("*", "+", "?"):
                    sequence[-1] = (to_rule(sequence[-1]) + char, False)
                    index += 1
                elif char == "{":
                    curly_brackets = char
                    index += 1
                    while index < length and pattern[index] != "}":
                        curly_brackets += pattern[index]
                        index += 1
                    assert index < length, (
                        f"Unbalanced curly brackets; start = {start}, index = {index}, pattern = {pattern}"
                    )
                    curly_brackets += "}"
                    index += 1
                    numbers = [part.strip() for part in curly_brackets[1:-1].split(",")]
                    min_times = 0
                    max_times: Optional[int] = None
                    try:
                        if len(numbers) == 1:
                            min_times = int(numbers[0])
                            max_times = min_times
                        else:
                            assert len(numbers) == 2
                            min_times = int(numbers[0]) if numbers[0] else 0
                            max_times = int(numbers[1]) if numbers[1] else None
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid quantifier {curly_brackets} in /{pattern}/"
                        ) from exc

                    sub, sub_is_literal = sequence[-1]
                    if not sub_is_literal:
                        rule_id = sub_rule_ids.get(sub)
                        if rule_id is None:
                            rule_id = self._add_rule(f"{name}-{len(sub_rule_ids) + 1}", sub)
                            sub_rule_ids[sub] = rule_id
                        sub = rule_id

                    sequence[-1] = (
                        self._build_repetition(
                            f'"{sub}"' if sub_is_literal else sub,
                            min_times,
                            max_times,
                            item_rule_is_literal=sub_is_literal,
                        ),
                        False,
                    )
                else:
                    literal = ""
                    while index < length:
                        if pattern[index] == "\\" and index < length - 1:
                            next_char = pattern[index + 1]
                            if next_char in self.ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                                index += 1
                                literal += pattern[index]
                                index += 1
                            else:
                                literal += pattern[index : index + 2]
                                index += 2
                        elif pattern[index] == '"' and not self._raw_pattern:
                            literal += '\\"'
                            index += 1
                        elif pattern[index] not in self.NON_LITERAL_SET and (
                            index == length - 1
                            or literal == ""
                            or pattern[index + 1] == "."
                            or pattern[index + 1] not in self.NON_LITERAL_SET
                        ):
                            literal += pattern[index]
                            index += 1
                        else:
                            break
                    if literal:
                        sequence.append((literal, True))

            return join_sequence()

        return self._add_rule(
            name,
            (
                to_rule(transform())
                if self._raw_pattern
                else '"\\"" ' + to_rule(transform()) + ' "\\"" space'
            ),
        )

    def _resolve_ref(self, ref: str) -> str:
        ref_name = ref.split("/")[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name

    def _generate_constant_rule(self, value: Any) -> str:
        return self._format_literal(json.dumps(value))

    def visit(self, schema: Dict[str, Any], name: str) -> str:
        schema_type = schema.get("type")
        schema_format = schema.get("format")
        rule_name = name + "-" if name in self._reserved_names() else name or "root"

        ref = schema.get("$ref")
        if ref is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        if "oneOf" in schema or "anyOf" in schema:
            return self._add_rule(
                rule_name,
                self._generate_union_rule(name, cast(List[Dict[str, Any]], schema.get("oneOf") or schema["anyOf"])),
            )

        if isinstance(schema_type, list):
            return self._add_rule(
                rule_name,
                self._generate_union_rule(name, [{"type": entry} for entry in schema_type]),
            )

        if "const" in schema:
            return self._add_rule(rule_name, self._generate_constant_rule(schema["const"]))

        if "enum" in schema:
            rule = " | ".join(self._generate_constant_rule(value) for value in schema["enum"])
            return self._add_rule(rule_name, rule)

        if schema_type in (None, "object") and (
            "properties" in schema
            or ("additionalProperties" in schema and schema["additionalProperties"] is not True)
        ):
            required_props = set(schema.get("required", []))
            property_items = list(cast(Dict[str, Any], schema.get("properties", {})).items())
            return self._add_rule(
                rule_name,
                self._build_object_rule(
                    property_items, required_props, name, schema.get("additionalProperties")
                ),
            )

        if schema_type in (None, "object") and "allOf" in schema:
            allof_required_props: set[str] = set()
            allof_property_items: List[Tuple[str, Any]] = []

            def add_component(component_schema: Dict[str, Any], is_required: bool) -> None:
                component_ref = component_schema.get("$ref")
                if component_ref is not None:
                    component_schema = cast(Dict[str, Any], self._refs[component_ref])
                if "properties" in component_schema:
                    for prop_name, prop_schema in cast(Dict[str, Any], component_schema["properties"]).items():
                        allof_property_items.append((prop_name, prop_schema))
                        if is_required:
                            allof_required_props.add(prop_name)

            for entry in cast(List[Dict[str, Any]], schema["allOf"]):
                if "anyOf" in entry:
                    for alt in cast(List[Dict[str, Any]], entry["anyOf"]):
                        add_component(alt, is_required=False)
                else:
                    add_component(entry, is_required=True)

            return self._add_rule(
                rule_name,
                self._build_object_rule(
                    allof_property_items,
                    allof_required_props,
                    name,
                    additional_properties=[],
                ),
            )

        if schema_type in (None, "array") and ("items" in schema or "prefixItems" in schema):
            items = schema.get("items") or schema["prefixItems"]
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space '
                    + ' "," space '.join(
                        self.visit(item, f"{name}{'-' if name else ''}tuple-{index}")
                        for index, item in enumerate(items)
                    )
                    + ' "]" space',
                )
            item_rule_name = self.visit(cast(Dict[str, Any], items), f"{name}{'-' if name else ''}item")
            min_items = int(schema.get("minItems", 0))
            max_items = cast(Optional[int], schema.get("maxItems"))
            return self._add_rule(
                rule_name,
                '"[" space '
                + self._build_repetition(
                    item_rule_name,
                    min_items,
                    max_items,
                    separator_rule='"," space',
                )
                + ' "]" space',
            )

        if schema_type in (None, "string") and "pattern" in schema:
            return self._visit_pattern(cast(str, schema["pattern"]), rule_name)

        if schema_type in (None, "string") and re.match(r"^uuid[1-5]?$", schema_format or ""):
            return self._add_primitive(
                "root" if rule_name == "root" else cast(str, schema_format),
                self._primitive_rules()["uuid"],
            )

        if schema_type in (None, "string") and f"{schema_format}-string" in self._string_format_rules():
            primitive_name = f"{schema_format}-string"
            return self._add_rule(
                rule_name,
                self._add_primitive(primitive_name, self._string_format_rules()[primitive_name]),
            )

        if schema_type == "string" and ("minLength" in schema or "maxLength" in schema):
            char_rule = self._add_primitive("char", self._primitive_rules()["char"])
            min_len = int(schema.get("minLength", 0))
            max_len = cast(Optional[int], schema.get("maxLength"))
            return self._add_rule(
                rule_name,
                r'"\"" '
                + self._build_repetition(char_rule, min_len, max_len)
                + r' "\"" space',
            )

        if schema_type == "object" or len(schema) == 0:
            return self._add_rule(
                rule_name,
                self._add_primitive("object", self._primitive_rules()["object"]),
            )

        primitive_rules = self._primitive_rules()
        assert schema_type in primitive_rules, f"Unrecognized schema: {schema}"
        return self._add_primitive(
            "root" if rule_name == "root" else cast(str, schema_type),
            primitive_rules[cast(str, schema_type)],
        )

    def _add_primitive(self, name: str, rule: "JsonSchemaConverter.BuiltinRule") -> str:
        rule_name = self._add_rule(name, rule.content)
        primitive_rules = self._primitive_rules()
        string_format_rules = self._string_format_rules()
        for dependency in rule.deps:
            dependency_rule = primitive_rules.get(dependency) or string_format_rules.get(
                dependency
            )
            assert dependency_rule is not None, f"Rule {dependency} not known"
            if dependency not in self._rules:
                self._add_primitive(dependency, dependency_rule)
        return rule_name

    def _build_object_rule(
        self,
        properties: List[Tuple[str, Any]],
        required: set[str],
        name: str,
        additional_properties: Union[bool, Any],
    ) -> str:
        prop_order = self._prop_order
        sorted_props = [
            key
            for _, (key, _) in sorted(
                enumerate(properties),
                key=lambda indexed_key: (
                    prop_order.get(indexed_key[1][0], len(prop_order)),
                    indexed_key[0],
                ),
            )
        ]

        property_kv_rule_names: Dict[str, str] = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(
                cast(Dict[str, Any], prop_schema),
                f"{name}{'-' if name else ''}{prop_name}",
            )
            property_kv_rule_names[prop_name] = self._add_rule(
                f"{name}{'-' if name else ''}{prop_name}-kv",
                rf'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}',
            )

        required_props = [key for key in sorted_props if key in required]
        optional_props = [key for key in sorted_props if key not in required]

        if additional_properties is True or isinstance(additional_properties, dict):
            sub_name = f"{name}{'-' if name else ''}additional"
            value_rule = self.visit(
                {} if additional_properties is True else cast(Dict[str, Any], additional_properties),
                f"{sub_name}-value",
            )
            property_kv_rule_names["*"] = self._add_rule(
                f"{sub_name}-kv",
                self._add_primitive("string", self._primitive_rules()["string"])
                + f' ":" space {value_rule}',
            )
            optional_props.append("*")

        rule = '"{" space '
        rule += ' "," space '.join(property_kv_rule_names[key] for key in required_props)

        if optional_props:
            if required_props:
                rule += ' ( "," space ( '
            else:
                rule += "( "

            def get_recursive_refs(keys: List[str], first_is_optional: bool) -> str:
                head, *rest = keys
                kv_rule_name = property_kv_rule_names[head]
                result = ""
                if head == "*":
                    if first_is_optional:
                        result = f"({kv_rule_name})?"
                    else:
                        result = kv_rule_name
                elif first_is_optional:
                    result = f'( "," space {kv_rule_name} )?'
                else:
                    result = kv_rule_name
                if rest:
                    result += " " + self._add_rule(
                        f"{name}{'-' if name else ''}{head}-rest",
                        get_recursive_refs(rest, first_is_optional=True),
                    )
                return result

            rule += " | ".join(
                get_recursive_refs(optional_props[index:], first_is_optional=False)
                for index in range(len(optional_props))
            )
            if required_props:
                rule += " )"
            rule += " )?"

        rule += ' "}" space'
        return rule

    def format_grammar(self) -> str:
        return "\n".join(
            f"{name} ::= {rule}"
            for name, rule in sorted(self._rules.items(), key=lambda item: item[0])
        )

    @classmethod
    def to_gbnf(cls, schema: str, prop_order: Optional[List[str]] = None) -> str:
        property_order = prop_order or []
        loaded_schema = json.loads(schema)
        order_index = {name: index for index, name in enumerate(property_order)}
        converter = cls(
            prop_order=order_index,
            allow_fetch=False,
            dotall=False,
            raw_pattern=False,
        )
        resolved_schema = converter.resolve_refs(loaded_schema, "stdin")
        converter.visit(resolved_schema, "")
        return converter.format_grammar()


class RadixTrie:
    __slots__ = ("root", "sequences", "sequence_lengths")

    @dataclass
    class Node:
        label: Tuple[int, ...] = ()
        parent: Optional["RadixTrie.Node"] = None
        children: Dict[int, "RadixTrie.Node"] = field(default_factory=dict)
        sequences: set[int] = field(default_factory=set)
        tails: set[int] = field(default_factory=set)

    def __init__(self) -> None:
        self.root = RadixTrie.Node()
        self.sequences: Dict[int, RadixTrie.Node] = {}
        self.sequence_lengths: Dict[int, int] = {}

    @staticmethod
    def _pick_sequence(candidates: set[int], preferred_sequences: Optional[Any]) -> int:
        if preferred_sequences is None:
            return next(iter(candidates))
        if isinstance(preferred_sequences, OrderedDict):
            for sequence_id in reversed(preferred_sequences):
                if sequence_id in candidates:
                    return sequence_id
            return next(iter(candidates))
        if isinstance(preferred_sequences, (list, tuple)):
            for sequence_id in reversed(preferred_sequences):
                if sequence_id in candidates:
                    return sequence_id
            return next(iter(candidates))
        preferred = candidates & preferred_sequences
        if preferred:
            return next(iter(preferred))
        return next(iter(candidates))

    @staticmethod
    def _common_prefix_len(
        label: Sequence[int],
        tokens: Sequence[int],
        offset: int,
    ) -> int:
        limit = min(len(label), len(tokens) - offset)
        match_len = 0
        while match_len < limit and label[match_len] == tokens[offset + match_len]:
            match_len += 1
        return match_len

    def _split_child(
        self,
        parent: "RadixTrie.Node",
        child: "RadixTrie.Node",
        prefix_len: int,
    ) -> "RadixTrie.Node":
        assert 0 < prefix_len < len(child.label)
        prefix = child.label[:prefix_len]
        suffix = child.label[prefix_len:]
        middle = RadixTrie.Node(
            label=prefix,
            parent=parent,
            sequences=set(child.sequences),
        )
        parent.children[prefix[0]] = middle
        child.label = suffix
        child.parent = middle
        middle.children[suffix[0]] = child
        return middle

    def _locate_prefix_node(
        self,
        sequence_id: int,
        keep_len: int,
    ) -> "RadixTrie.Node":
        total_len = self.sequence_lengths[sequence_id]
        assert 0 <= keep_len <= total_len
        if keep_len == 0:
            return self.root
        node = self.sequences[sequence_id]
        drop_len = total_len - keep_len
        while node is not self.root:
            label_len = len(node.label)
            if drop_len == 0:
                return node
            if drop_len < label_len:
                parent = node.parent
                assert parent is not None
                return self._split_child(parent, node, label_len - drop_len)
            drop_len -= label_len
            parent = node.parent
            assert parent is not None
            node = parent
        return self.root

    def extend(
        self,
        sequence_id: int,
        tokens: Sequence[int],
    ) -> None:
        assert sequence_id >= 0
        node = self.sequences.get(sequence_id, self.root)
        if tokens:
            node.tails.discard(sequence_id)
        length = self.sequence_lengths.get(sequence_id, 0)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            child = node.children.get(token)
            if child is None:
                child = RadixTrie.Node(
                    label=tuple(tokens[index:]),
                    parent=node,
                    sequences={sequence_id},
                )
                node.children[token] = child
                node = child
                length += len(tokens) - index
                index = len(tokens)
                break
            match_len = self._common_prefix_len(child.label, tokens, index)
            if match_len == len(child.label):
                child.sequences.add(sequence_id)
                node = child
                length += match_len
                index += match_len
                continue
            node = self._split_child(node, child, match_len)
            node.sequences.add(sequence_id)
            length += match_len
            index += match_len
            if index == len(tokens):
                break
            suffix = RadixTrie.Node(
                label=tuple(tokens[index:]),
                parent=node,
                sequences={sequence_id},
            )
            node.children[suffix.label[0]] = suffix
            node = suffix
            length += len(tokens) - index
            index = len(tokens)
            break
        if node is self.root:
            self.sequences.pop(sequence_id, None)
            self.sequence_lengths.pop(sequence_id, None)
            self.root.tails.discard(sequence_id)
        else:
            self.sequences[sequence_id] = node
            self.sequence_lengths[sequence_id] = length
            node.tails.add(sequence_id)

    def length(self, sequence_id: int) -> int:
        return self.sequence_lengths.get(sequence_id, 0)

    def truncate(self, sequence_id: int, keep_len: int) -> None:
        assert sequence_id >= 0
        assert sequence_id in self.sequence_lengths
        assert 0 <= keep_len <= self.sequence_lengths[sequence_id]
        current_len = self.sequence_lengths[sequence_id]
        if keep_len == current_len:
            return
        boundary = self._locate_prefix_node(sequence_id, keep_len)
        node = self.sequences.get(sequence_id, self.root)
        if node is not self.root:
            node.tails.discard(sequence_id)
        while node is not boundary:
            node.sequences.remove(sequence_id)
            parent = node.parent
            assert parent is not None
            if not node.sequences:
                del parent.children[node.label[0]]
            node = parent
        if boundary is self.root:
            self.sequences.pop(sequence_id, None)
            self.sequence_lengths.pop(sequence_id, None)
        else:
            self.sequences[sequence_id] = boundary
            self.sequence_lengths[sequence_id] = keep_len
            boundary.tails.add(sequence_id)

    def copy(self, source_sequence_id: int, dest_sequence_id: int, keep_len: int) -> None:
        assert source_sequence_id >= 0
        assert dest_sequence_id >= 0
        assert source_sequence_id in self.sequence_lengths
        assert dest_sequence_id not in self.sequence_lengths
        assert 0 <= keep_len <= self.sequence_lengths[source_sequence_id]
        if keep_len == 0:
            self.sequences[dest_sequence_id] = self.root
            self.sequence_lengths[dest_sequence_id] = 0
            self.root.tails.add(dest_sequence_id)
            return
        node = self._locate_prefix_node(source_sequence_id, keep_len)
        self.sequences[dest_sequence_id] = node
        self.sequence_lengths[dest_sequence_id] = keep_len
        node.tails.add(dest_sequence_id)
        while node is not self.root:
            node.sequences.add(dest_sequence_id)
            parent = node.parent
            assert parent is not None
            node = parent

    def tokens(self, sequence_id: int, keep_len: Optional[int] = None) -> List[int]:
        length = self.sequence_lengths[sequence_id]
        target_len = length if keep_len is None else keep_len
        assert 0 <= target_len <= length
        node = self.sequences[sequence_id]
        labels: List[Tuple[int, ...]] = []
        while node is not self.root:
            labels.append(node.label)
            parent = node.parent
            assert parent is not None
            node = parent
        values: List[int] = []
        for label in reversed(labels):
            values.extend(label)
        return values[:target_len]

    def longest_prefix(
        self,
        tokens: Sequence[int],
        preferred_sequences: Optional[Any] = None,
        *,
        exact_only: bool = False,
    ) -> Tuple[int, int]:
        node = self.root
        longest_sequence_id = -1
        longest_length = 0
        index = 0
        while index < len(tokens):
            child = node.children.get(tokens[index])
            if child is None:
                break
            match_len = self._common_prefix_len(child.label, tokens, index)
            if match_len < len(child.label):
                break
            node = child
            index += match_len
            candidates = node.tails if exact_only else node.sequences
            if candidates:
                longest_sequence_id = self._pick_sequence(candidates, preferred_sequences)
                longest_length = index
        return longest_sequence_id, longest_length


class SequenceHistory:
    __slots__ = ("_root", "_tails", "size")

    @dataclass
    class Node:
        token: Optional[int] = None
        parent: Optional["SequenceHistory.Node"] = None
        children: Dict[int, "SequenceHistory.Node"] = field(default_factory=dict)
        sequences: set[int] = field(default_factory=set)

    def __init__(self) -> None:
        self._root = SequenceHistory.Node()
        self._tails: Dict[int, SequenceHistory.Node] = {}
        self.size = 0

    def extend(self, sequence_id: int, tokens: Sequence[int]) -> None:
        assert sequence_id >= 0
        node = self._tails.get(sequence_id, self._root)
        for token in tokens:
            child = node.children.get(sequence_id)
            if child is None:
                child = SequenceHistory.Node(token=token, parent=node)
                node.children[sequence_id] = child
                self.size += 1
            else:
                assert child.parent is node
                assert child.token == token
            child.sequences.add(sequence_id)
            node = child
        if node is self._root:
            self._tails.pop(sequence_id, None)
        else:
            self._tails[sequence_id] = node

    def copy(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        source_length: int,
        keep_len: int,
    ) -> None:
        assert source_sequence_id >= 0
        assert dest_sequence_id >= 0
        assert source_sequence_id in self._tails
        assert dest_sequence_id not in self._tails
        assert 0 <= keep_len <= source_length
        node = self._tails[source_sequence_id]
        path: List[SequenceHistory.Node] = []
        for _ in range(source_length - keep_len):
            parent = node.parent
            assert parent is not None
            node = parent
        while node is not self._root:
            path.append(node)
            parent = node.parent
            assert parent is not None
            node = parent
        parent = self._root
        for child in reversed(path):
            parent.children[dest_sequence_id] = child
            child.sequences.add(dest_sequence_id)
            parent = child
        if keep_len == 0:
            self._tails.pop(dest_sequence_id, None)
        else:
            self._tails[dest_sequence_id] = path[0]

    def truncate(
        self,
        sequence_id: int,
        current_length: int,
        keep_len: int,
    ) -> None:
        assert sequence_id >= 0
        assert sequence_id in self._tails
        assert 0 <= keep_len <= current_length
        node = self._tails[sequence_id]
        drop = current_length - keep_len
        while node is not self._root and drop > 0:
            node.sequences.remove(sequence_id)
            parent = node.parent
            assert parent is not None
            child = parent.children.get(sequence_id)
            if child is node:
                del parent.children[sequence_id]
            if not node.sequences:
                self.size -= 1
            node = parent
            drop -= 1
        if node is self._root:
            self._tails.pop(sequence_id, None)
        else:
            self._tails[sequence_id] = node


class DraftProvider(abc.ABC):
    @abc.abstractmethod
    def draft(self, input_ids: np.ndarray, /) -> np.ndarray:
        raise NotImplementedError()


class PromptLookupDecoding(DraftProvider):
    def __init__(self, max_ngram_size: int = 2, num_pred_tokens: int = 10) -> None:
        self._max_ngram_size = max_ngram_size
        self._num_pred_tokens = num_pred_tokens

    def draft(self, input_ids: np.ndarray, /) -> np.ndarray:
        input_length = input_ids.shape[0]
        if input_length < 2:
            return np.array([], dtype=np.intc)
        max_ngram_size = min(self._max_ngram_size, input_length - 1)
        for ngram_size in range(max_ngram_size, 0, -1):
            windows = np.lib.stride_tricks.sliding_window_view(input_ids, (ngram_size,))
            ngram = input_ids[-ngram_size:]
            matches = np.all(windows == ngram, axis=1)
            match_indices = np.nonzero(matches)[0]
            for index in match_indices:
                start = index + ngram_size
                if start >= input_length:
                    continue
                end = min(start + self._num_pred_tokens, input_length)
                if start < end:
                    return input_ids[start:end].astype(np.intc, copy=False)
        return np.array([], dtype=np.intc)


class CompletionRequestCancelledError(RuntimeError):
    pass


class CompletionRequestValidationError(ValueError):
    pass


class CompletionResponseParsingError(RuntimeError):
    pass


class CompletionChunkLogprobs(TypedDict):
    tokens: List[str]
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChunkChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionChunkLogprobs]
    finish_reason: Optional[str]


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChunkChoice]


CompletionStream = Generator[CompletionChunk, None, OpenAICompletion]
CompletionPrompt = Union[str, List[int], List[str], List[List[int]]]


class CreateCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: CompletionPrompt = ""
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, ge=0)
    temperature: float = 0.8
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    logprobs: Optional[int] = Field(default=None, ge=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    n: int = Field(default=1, ge=1)
    best_of: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        result: Dict[str, float] = {}
        for key, bias in value.items():
            int(key)
            result[key] = float(bias)
        return result

    @model_validator(mode="after")
    def validate_after(self) -> "CreateCompletionRequest":
        if self.best_of is None:
            self.best_of = self.n
        if self.best_of < self.n:
            raise ValueError("best_of must be greater than or equal to n")
        if self.stream and self.best_of > 1:
            raise ValueError("best_of is not supported for streaming completions")
        if len(self.normalized_prompt()) > 1 and self.stream:
            raise ValueError("streaming does not support multiple prompts")
        return self

    def normalized_prompt(self) -> List[Union[str, List[int]]]:
        if isinstance(self.prompt, str):
            return [self.prompt]
        if all(isinstance(token, int) for token in self.prompt):
            return [cast(List[int], self.prompt)]
        if all(isinstance(prompt, str) for prompt in self.prompt):
            return cast(List[Union[str, List[int]]], list(cast(List[str], self.prompt)))
        if all(
            isinstance(prompt, list)
            and all(isinstance(token, int) for token in prompt)
            for prompt in self.prompt
        ):
            return cast(
                List[Union[str, List[int]]],
                list(cast(List[List[int]], self.prompt)),
            )
        raise ValueError("prompt must be a string, token ids, list of strings, or list of token-id lists")


class ChatCompletionRequestMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool", "function"] = Field(
        default="user"
    )
    content: Optional[str] = Field(default="")
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None)


class CreateChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: List[ChatCompletionRequestMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = Field(default=None, ge=0)
    temperature: float = 0.8
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    logprobs: Optional[bool] = Field(default=False)
    top_logprobs: Optional[int] = Field(default=None, ge=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    n: int = Field(default=1, ge=1)
    user: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        result: Dict[str, float] = {}
        for key, bias in value.items():
            int(key)
            result[key] = float(bias)
        return result

    @model_validator(mode="after")
    def validate_after(self) -> "CreateChatCompletionRequest":
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        return self


class CreateResponseRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: Any = ""
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0)
    model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    text: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Any] = None
    store: Optional[bool] = None
    truncation: Optional[Literal["auto", "disabled"]] = None


class ResponseCreateWebSocketRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["response.create"]
    model: Optional[str] = None
    input: Any = ""
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = True
    top_logprobs: Optional[int] = Field(default=None, ge=0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    text: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Any] = None
    store: Optional[bool] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    generate: Optional[bool] = None

    def to_create_response_request(self) -> CreateResponseRequest:
        return CreateResponseRequest.model_validate(
            self.model_dump(mode="python", exclude={"type"})
        )


@dataclass
class ResponsesWebSocketState:
    input_items: List[Any]
    output_items: List[Dict[str, Any]]


class ConfigFile(BaseModel):
    class ServerOptions(BaseModel):
        host: str = "127.0.0.1"
        port: int = 8000

    class FromPretrainedOptions(BaseModel):
        repo_id: str
        filename: str
        additional_files: Optional[List[str]] = None
        local_dir: Optional[str] = None
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto"
        cache_dir: Optional[str] = None

        @staticmethod
        def _pattern_has_glob(pattern: str) -> bool:
            return any(char in pattern for char in "*?[]")

        @staticmethod
        def _subfolder_for_repo_file(repo_file: str) -> Optional[str]:
            subfolder = str(Path(repo_file).parent)
            return None if subfolder == "." else subfolder

        def _cached_repo_file_list(self) -> List[str]:
            from huggingface_hub import scan_cache_dir

            try:
                cache_info = scan_cache_dir(self.cache_dir)
            except Exception:
                return []

            cached_files: set[str] = set()
            for repo in cache_info.repos:
                if repo.repo_type != "model" or repo.repo_id != self.repo_id:
                    continue
                for revision in repo.revisions:
                    for cached_file in revision.files:
                        cached_files.add(cached_file.file_name)
            return sorted(cached_files)

        def _download_repo_file(self, repo_file: str) -> str:
            from huggingface_hub import hf_hub_download

            filename = Path(repo_file).name
            subfolder = self._subfolder_for_repo_file(repo_file)

            download_kwargs = dict(
                repo_id=self.repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=self.local_dir,
                local_dir_use_symlinks=self.local_dir_use_symlinks,
                cache_dir=self.cache_dir,
            )

            try:
                resolved_path = hf_hub_download(**download_kwargs)
            except Exception as exc:
                try:
                    cached_path = cast(
                        str,
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=filename,
                            subfolder=subfolder,
                            cache_dir=self.cache_dir,
                            local_files_only=True,
                        ),
                    )
                except Exception:
                    raise exc

                if self.local_dir is None:
                    return cached_path

                resolved_path = os.path.join(self.local_dir, repo_file)
                resolved_parent = os.path.dirname(resolved_path)
                if resolved_parent:
                    os.makedirs(resolved_parent, exist_ok=True)
                shutil.copy2(cached_path, resolved_path)

            return cast(str, resolved_path)

        @staticmethod
        def _match_single_file(pattern: str, file_list: Sequence[str]) -> str:
            matching_files = [file for file in file_list if fnmatch.fnmatch(file, pattern)]
            if len(matching_files) == 0:
                raise ValueError(
                    f"No file found matching {pattern}\n\n"
                    f"Available Files:\n{json.dumps(list(file_list))}"
                )
            if len(matching_files) > 1:
                raise ValueError(
                    f"Multiple files found matching {pattern}\n\n"
                    f"Available Files:\n{json.dumps(list(file_list))}"
                )
            return matching_files[0]

        def resolve_model_path(self) -> str:
            try:
                from huggingface_hub import HfFileSystem
                from huggingface_hub.utils import validate_repo_id
            except ImportError as exc:
                raise ImportError(
                    "model.from_pretrained requires the huggingface-hub package. "
                    "You can install it with `pip install huggingface-hub`."
                ) from exc

            validate_repo_id(self.repo_id)
            requested_patterns = [self.filename, *(self.additional_files or [])]
            if not any(self._pattern_has_glob(pattern) for pattern in requested_patterns):
                model_path = self._download_repo_file(self.filename)
                if self.additional_files:
                    for additional_file_name in self.additional_files:
                        self._download_repo_file(additional_file_name)
                return model_path

            try:
                hffs = HfFileSystem()
                files = [
                    file["name"] if isinstance(file, dict) else file
                    for file in hffs.ls(self.repo_id, recursive=True)
                ]
                file_list = [str(Path(file).relative_to(self.repo_id)) for file in files]
            except Exception as exc:
                file_list = self._cached_repo_file_list()
                if not file_list:
                    raise exc

            matching_file = self._match_single_file(self.filename, file_list)
            model_path = self._download_repo_file(matching_file)

            if self.additional_files:
                for additional_file_name in self.additional_files:
                    matching_additional_file = self._match_single_file(
                        additional_file_name, file_list
                    )
                    self._download_repo_file(matching_additional_file)

            return model_path

    class ModelOptions(BaseModel):
        path: Optional[str] = None
        alias: Optional[str] = None
        chat_template: Optional[str] = None
        from_pretrained: Optional["ConfigFile.FromPretrainedOptions"] = None
        n_gpu_layers: Optional[int] = None
        split_mode: Optional[int] = None
        main_gpu: Optional[int] = None
        tensor_split: Optional[List[float]] = None
        vocab_only: Optional[bool] = None
        use_mmap: Optional[bool] = None
        use_mlock: Optional[bool] = None
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None
        n_ctx: Optional[int] = None
        n_batch: Optional[int] = None
        n_ubatch: Optional[int] = None
        n_seq_max: Optional[int] = None
        threads: Optional[int] = Field(
            default_factory=lambda: max(multiprocessing.cpu_count(), 1)
        )
        threads_batch: Optional[int] = Field(
            default_factory=lambda: max(multiprocessing.cpu_count(), 1)
        )
        rope_scaling_type: Optional[int] = None
        pooling_type: Optional[int] = None
        attention_type: Optional[int] = None
        rope_freq_base: Optional[float] = None
        rope_freq_scale: Optional[float] = None
        yarn_ext_factor: Optional[float] = None
        yarn_attn_factor: Optional[float] = None
        yarn_beta_fast: Optional[float] = None
        yarn_beta_slow: Optional[float] = None
        yarn_orig_ctx: Optional[int] = None
        offload_kqv: Optional[bool] = None
        flash_attn: Optional[bool] = None
        op_offload: Optional[bool] = None
        swa_full: Optional[bool] = None
        no_perf: Optional[bool] = None
        type_k: Optional[int] = None
        type_v: Optional[int] = None
        max_seq_len: Optional[int] = None
        max_output_tokens: Optional[int] = Field(default=None, ge=0)
        kv_unified: bool = True
        draft_model: Optional[Literal["prompt-lookup-decoding"]] = None
        draft_model_num_pred_tokens: int = 10
        draft_model_max_ngram_size: int = 2
        response_schema: Optional[Dict[str, Any]] = None
        store_logits: bool = True

        @model_validator(mode="after")
        def validate_source(self) -> "ConfigFile.ModelOptions":
            if (self.path is None) == (self.from_pretrained is None):
                raise ValueError("exactly one of model.path or model.from_pretrained is required")
            return self

        @field_validator("chat_template", mode="before")
        @classmethod
        def normalize_chat_template(cls, value: Any) -> Any:
            if isinstance(value, list):
                if not all(isinstance(item, str) for item in value):
                    raise TypeError("model.chat_template list entries must be strings")
                return "".join(value)
            return value

        def resolve_model_path(self) -> str:
            if self.from_pretrained is not None:
                return self.from_pretrained.resolve_model_path()
            assert self.path is not None
            return self.path

    server: "ConfigFile.ServerOptions" = Field(default_factory=lambda: ConfigFile.ServerOptions())
    model: "ConfigFile.ModelOptions"

    @classmethod
    def load(cls, path: str) -> "ConfigFile":
        return cls.model_validate_json(Path(path).read_text())


ConfigFile.model_rebuild()


class Jinja2ChatFormatter:
    def __init__(self, template: str, *, bos_token: str, eos_token: str) -> None:
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._template_text = template
        environment = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        environment.filters["from_json"] = self._from_json
        self._template = environment.from_string(template)

    @staticmethod
    def _from_json(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _render(
        self,
        *,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool,
        strftime_now: Callable[[str], str],
    ) -> str:
        def raise_exception(message: str) -> None:
            raise ValueError(message)

        return cast(
            str,
            self._template.render(
                messages=[message.model_dump(exclude_none=True) for message in messages],
                eos_token=self._eos_token,
                bos_token=self._bos_token,
                raise_exception=raise_exception,
                add_generation_prompt=add_generation_prompt,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                strftime_now=strftime_now,
            ),
        )

    def format(
        self,
        *,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[str, str, List[str]]:
        render_time = datetime.now()

        def strftime_now(format_string: str) -> str:
            return render_time.strftime(format_string)

        chat_template = self._render(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=False,
            strftime_now=strftime_now,
        )
        prompt = self._render(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=True,
            strftime_now=strftime_now,
        )
        generation_prompt = prompt[len(chat_template) :] if prompt.startswith(chat_template) else ""
        stop = [self._eos_token] if self._eos_token else []
        return prompt, generation_prompt, stop


@dataclass
class Token:
    token: int
    text_bytes: bytes
    token_logprob: Optional[float]
    top_logprobs: Optional[Dict[str, float]]

    @classmethod
    def from_token(
        cls,
        *,
        model: Model,
        prev_tokens: Sequence[int],
        prev_text_bytes: Optional[Union[bytes, bytearray]] = None,
        token: int,
    ) -> "Token":
        return cls(
            token=token,
            text_bytes=(
                model.token_bytes_with_prev_bytes(prev_tokens, prev_text_bytes, token)
                if prev_text_bytes is not None
                else model.token_bytes_with_prev(prev_tokens, token)
            ),
            token_logprob=None,
            top_logprobs=None,
        )

    @classmethod
    def from_logits(
        cls,
        *,
        model: Model,
        formatter: OpenAIFormatter,
        prev_tokens: Sequence[int],
        prev_text_bytes: Optional[Union[bytes, bytearray]] = None,
        token: int,
        logits: np.ndarray,
        logprobs_count: Optional[int],
        need_token_logprob: bool = False,
    ) -> "Token":
        text_bytes = (
            model.token_bytes_with_prev_bytes(prev_tokens, prev_text_bytes, token)
            if prev_text_bytes is not None
            else model.token_bytes_with_prev(prev_tokens, token)
        )
        if not model.store_logits:
            return cls(
                token=token,
                text_bytes=text_bytes,
                token_logprob=None,
                top_logprobs=None,
            )
        if logprobs_count is None and not need_token_logprob:
            return cls(
                token=token,
                text_bytes=text_bytes,
                token_logprob=None,
                top_logprobs=None,
            )
        logprobs = CompletionScheduler.logits_to_logprobs(logits)
        token_logprob = float(logprobs[token])
        top_logprobs: Optional[Dict[str, float]] = None
        if logprobs_count is not None:
            top_count = min(max(logprobs_count, 0), model.n_vocab)
            if top_count > 0:
                top_indices = np.argpartition(logprobs, -top_count)[-top_count:]
                top_indices = top_indices[np.argsort(logprobs[top_indices])[::-1]]
                top_logprobs = {
                    formatter.decode_text(model.token_bytes(int(index))): float(
                        logprobs[int(index)]
                    )
                    for index in top_indices
                }
            else:
                top_logprobs = {}
            top_logprobs[formatter.decode_text(text_bytes)] = token_logprob
        return cls(
            token=token,
            text_bytes=text_bytes,
            token_logprob=token_logprob,
            top_logprobs=top_logprobs,
        )


@dataclass
class Completion:
    request_id: str
    index: int
    seq_id: int
    sampler: "Sampler"
    prompt_tokens: List[int]
    prompt_text: str
    max_total_tokens: int
    stop_sequences: List[bytes]
    logprobs: Optional[int]
    completion_tokens: List[int] = field(default_factory=list)
    token_records: List[Token] = field(default_factory=list)
    rendered_bytes: bytearray = field(default_factory=bytearray)
    detokenized_prefix_bytes: bytearray = field(default_factory=bytearray)
    pending_input_tokens: List[int] = field(default_factory=list)
    draft_tokens: List[int] = field(default_factory=list)
    pending_finish_reason: Optional[str] = None
    returned_token_count: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None
    score_sum: float = 0.0
    rank_by_score: bool = False

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def completion_token_count(self) -> int:
        return len(self.completion_tokens)

    @property
    def needs_token_logprob(self) -> bool:
        return self.logprobs is not None or self.rank_by_score

    @property
    def max_stop_sequence_length(self) -> int:
        return max((len(stop) for stop in self.stop_sequences), default=0)


@dataclass
class CompletionRequest:
    payload: CreateCompletionRequest
    prompt_text: str
    prompt_tokens: List[int]
    effective_max_len: int
    internal_completion_count: int
    prompt_visible_start: int
    id: str = field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    created: int = field(default_factory=lambda: int(time.time()))
    prompt_cursor: int = 0
    match_sequence_id: int = -1
    match_length: int = 0
    prompt_logits: Optional[np.ndarray] = None
    base_seq_id: Optional[int] = None
    sibling_seq_ids: List[int] = field(default_factory=list)
    completion_seq_ids: List[int] = field(default_factory=list)
    completions: List[Completion] = field(default_factory=list)
    admitted: bool = False
    prompt_done: bool = False
    prompt_checkpoint_saved: bool = False
    cancelled: bool = False
    prompt_records: List[Token] = field(default_factory=list)
    grammar_text: Optional[str] = None
    grammar_root: str = "root"
    chat_tool_name: Optional[str] = None
    on_stream_chunk: Optional[Callable[[CompletionChunk], None]] = None
    on_done: Optional[Callable[[OpenAICompletion], None]] = None
    on_error: Optional[Callable[[BaseException], None]] = None

    @classmethod
    def from_payload(
        cls,
        *,
        model: Model,
        payload: CreateCompletionRequest,
        on_stream_chunk: Optional[Callable[[CompletionChunk], None]] = None,
        on_done: Optional[Callable[[OpenAICompletion], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> "CompletionRequest":
        prompts = payload.normalized_prompt()
        if len(prompts) != 1:
            raise CompletionRequestValidationError("multiple prompts are not supported")
        prompt_item = prompts[0]
        if isinstance(prompt_item, str):
            prompt_text = prompt_item
            try:
                prompt_tokens = model.build_prompt_tokens(prompt_text, payload.suffix)
            except ValueError as exc:
                raise CompletionRequestValidationError(str(exc)) from exc
        else:
            if payload.suffix is not None:
                raise CompletionRequestValidationError(
                    "suffix is not supported with token id prompts"
                )
            prompt_tokens = list(prompt_item)
            prompt_text = model.detokenize(prompt_tokens).decode("utf-8", errors="ignore")
        return cls.from_prepared(
            model=model,
            payload=payload,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            on_stream_chunk=on_stream_chunk,
            on_done=on_done,
            on_error=on_error,
        )

    @classmethod
    def from_prepared(
        cls,
        *,
        model: Model,
        payload: CreateCompletionRequest,
        prompt_text: str,
        prompt_tokens: List[int],
        grammar_text: Optional[str] = None,
        chat_tool_name: Optional[str] = None,
        request_id: Optional[str] = None,
        on_stream_chunk: Optional[Callable[[CompletionChunk], None]] = None,
        on_done: Optional[Callable[[OpenAICompletion], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> "CompletionRequest":
        ctx_limit = model.max_seq_len
        if model.max_output_tokens is not None:
            ctx_limit = min(ctx_limit, len(prompt_tokens) + model.max_output_tokens)
        if payload.max_tokens is None:
            effective_max_len = ctx_limit
        else:
            effective_max_len = min(ctx_limit, len(prompt_tokens) + payload.max_tokens)
        if effective_max_len < len(prompt_tokens):
            raise CompletionRequestValidationError("prompt exceeds context window")
        internal_completion_count = payload.best_of if payload.best_of is not None else payload.n
        request = cls(
            payload=payload,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            effective_max_len=effective_max_len,
            internal_completion_count=internal_completion_count,
            id=request_id or f"cmpl-{uuid.uuid4().hex}",
            prompt_visible_start=1
            if prompt_tokens and prompt_tokens[0] == model.bos_token
            else 0,
            grammar_text=grammar_text,
            chat_tool_name=chat_tool_name,
            on_stream_chunk=on_stream_chunk,
            on_done=on_done,
            on_error=on_error,
        )
        if payload.echo and payload.logprobs is not None:
            request.seed_prompt_records(model)
        return request

    def seed_prompt_records(self, model: Model) -> None:
        if self.prompt_visible_start >= len(self.prompt_tokens):
            return
        first_pos = self.prompt_visible_start
        first_token = self.prompt_tokens[first_pos]
        self.prompt_records.append(
            Token(
                token=first_token,
                text_bytes=model.token_bytes_with_prev(
                    self.prompt_tokens[:first_pos],
                    first_token,
                ),
                token_logprob=None,
                top_logprobs=None,
            )
        )

    def selected_completions(self) -> List[Completion]:
        completions = list(self.completions)
        best_of = self.payload.best_of if self.payload.best_of is not None else self.payload.n
        if best_of > self.payload.n:
            completions.sort(
                key=lambda completion: (
                    completion.score_sum / max(1, completion.completion_token_count),
                    completion.score_sum,
                ),
                reverse=True,
            )
        return sorted(completions[: self.payload.n], key=lambda completion: completion.index)

    def capture_prompt_logprobs(
        self,
        *,
        model: Model,
        formatter: OpenAIFormatter,
        start_pos: int,
        output_indices: Sequence[Optional[int]],
        output_count: int,
        output_arg: Callable[[Optional[int], int], Optional[int]],
    ) -> None:
        if self.payload.logprobs is None or not self.payload.echo:
            return
        for token_offset, output_index in enumerate(output_indices):
            if output_index is None:
                continue
            next_pos = start_pos + token_offset + 1
            if next_pos < self.prompt_visible_start or next_pos >= len(self.prompt_tokens):
                continue
            row_index = output_arg(output_index, output_count)
            assert row_index is not None
            next_token = self.prompt_tokens[next_pos]
            record = Token.from_logits(
                model=model,
                formatter=formatter,
                prev_tokens=self.prompt_tokens[:next_pos],
                token=next_token,
                logits=model.logits(row_index),
                logprobs_count=self.payload.logprobs,
            )
            expected_index = next_pos - self.prompt_visible_start
            if expected_index == len(self.prompt_records):
                self.prompt_records.append(record)
            elif expected_index < len(self.prompt_records):
                self.prompt_records[expected_index] = record
            else:
                raise RuntimeError("prompt logprob order mismatch")


class ResponseParser:
    @dataclass
    class PartialJsonValue:
        text: str
        schema_type: Optional[str] = None
        complete: bool = False

    @dataclass
    class PartialJsonObject:
        value: "OrderedDict[str, Any]"
        complete: bool = False

    @dataclass
    class DirectState:
        deltas: bool
        pending: str
        mode: int
        done: bool
        saw_tool_calls: bool
        tool_call_count: int
        assistant_prefix: str
        leading_capture_field: Optional[str]
        leading_capture_start: str
        leading_capture_end: str
        leading_capture_strip_after: bool
        leading_capture_implicit: bool
        content_end_markers: Tuple[str, ...]
        trim_before_iterator: bool
        iterator_start: str
        iterator_end: str
        stop_markers: Tuple[str, ...]
        function_start: str
        function_name_end: str
        function_end: str
        parameter_start: str
        parameter_name_end: str
        parameter_end: str

    @dataclass
    class ItemState:
        pending: str
        mode: int
        tool_call_index: int
        tool_name: str
        parameter_count: int
        visible_tool_call: Optional[Dict[str, Any]]
        visible_function: Optional[Dict[str, Any]]
        current_parameter: Optional[str]
        current_parameter_value: str
        current_schema_type: Optional[str]
        current_parameter_schema: Dict[str, Any]

    @dataclass
    class StreamState:
        kind: str
        pending: str
        mode: str
        current_item: Optional[Dict[str, Any]]
        current_segment: Optional[Dict[str, Any]]
        done: bool
        saw_tool_calls: bool
        parsed: Optional[Dict[str, Any]] = None
        buffer: str = ""
        tool_call_count: int = 0

    _STREAM_PLAN_CACHE: Dict[int, Tuple[Any, Optional[Dict[str, Any]]]] = {}
    _TOOL_SCHEMA_CACHE: Dict[int, Tuple[Any, Dict[str, Dict[str, Any]]]] = {}
    __slots__ = (
        "_schema",
        "_tools",
        "_completion_id",
        "_choice_index",
        "_generation_prompt",
        "_tool_schemas",
        "_started",
        "_text_parts",
        "_message",
        "_stream_plan",
        "_direct",
        "_item",
        "_stream_state",
        "_stream_failed",
    )
    DIRECT_MODE_ASSISTANT_PREFIX = 0
    DIRECT_MODE_PRELUDE = 1
    DIRECT_MODE_LEADING_CAPTURE = 2
    DIRECT_MODE_CONTENT = 3
    DIRECT_MODE_TOOL_ITEM = 4
    DIRECT_MODE_AFTER_TOOL_ITEM = 5
    ITEM_MODE_FUNCTION_HEADER = 0
    ITEM_MODE_AFTER_FUNCTION_HEADER = 1
    ITEM_MODE_PARAMETER_NAME = 2
    ITEM_MODE_PARAMETER_VALUE = 3
    ITEM_MODE_AFTER_PARAMETER = 4
    ITEM_MODE_DONE = 5

    def __init__(
        self,
        schema: Dict[str, Any],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        completion_id: str = "",
        choice_index: int = 0,
        generation_prompt: str = "",
    ) -> None:
        self._schema = schema
        self._tools = tools
        self._completion_id = completion_id
        self._choice_index = choice_index
        self._generation_prompt = generation_prompt
        self._tool_schemas = self._cached_tool_schema_map(tools)
        self._started = False
        self._text_parts: List[str] = []
        self._message: Dict[str, Any] = {}
        self._stream_plan = self._cached_stream_plan(schema)
        self._direct = ResponseParser.DirectState(
            deltas=bool(self._stream_plan is not None and self._stream_plan.get("direct_deltas")),
            pending="",
            mode=self.DIRECT_MODE_PRELUDE,
            done=False,
            saw_tool_calls=False,
            tool_call_count=0,
            assistant_prefix="",
            leading_capture_field=None,
            leading_capture_start="",
            leading_capture_end="",
            leading_capture_strip_after=False,
            leading_capture_implicit=False,
            content_end_markers=(),
            trim_before_iterator=False,
            iterator_start="",
            iterator_end="",
            stop_markers=(),
            function_start="",
            function_name_end="",
            function_end="",
            parameter_start="",
            parameter_name_end="",
            parameter_end="",
        )
        self._item = ResponseParser.ItemState(
            pending="",
            mode=self.ITEM_MODE_FUNCTION_HEADER,
            tool_call_index=0,
            tool_name="",
            parameter_count=0,
            visible_tool_call=None,
            visible_function=None,
            current_parameter=None,
            current_parameter_value="",
            current_schema_type=None,
            current_parameter_schema={},
        )
        if self._direct.deltas and self._stream_plan is not None:
            direct_init = self._stream_plan["direct_init"]
            (
                self._direct.assistant_prefix,
                self._direct.leading_capture_field,
                self._direct.leading_capture_start,
                self._direct.leading_capture_end,
                self._direct.leading_capture_strip_after,
                self._direct.leading_capture_implicit,
                self._direct.trim_before_iterator,
                self._direct.content_end_markers,
                self._direct.stop_markers,
                self._direct.iterator_start,
                self._direct.iterator_end,
                self._direct.function_start,
                self._direct.function_name_end,
                self._direct.function_end,
                self._direct.parameter_start,
                self._direct.parameter_name_end,
                self._direct.parameter_end,
            ) = direct_init
            self._direct.mode = (
                self.DIRECT_MODE_ASSISTANT_PREFIX
                if self._direct.assistant_prefix
                else self.DIRECT_MODE_PRELUDE
            )
            self._stream_state = None
        else:
            self._stream_state = (
                self._new_stream_state(self._stream_plan) if self._stream_plan is not None else None
            )
        self._stream_failed = False
        if self._generation_prompt and self._stream_plan is not None:
            success, _ignored = self._advance_stream_state(self._generation_prompt)
            if not success:
                self._stream_failed = True

    @property
    def started(self) -> bool:
        return self._started

    @staticmethod
    def _regex_capture(text: str, pattern: str) -> Optional[Any]:
        match = re.search(pattern, text, re.S)
        if match is None:
            return None
        group_dict = match.groupdict()
        if group_dict:
            return {key: value for key, value in group_dict.items() if value is not None}
        return match.group(1) if match.lastindex else match.group(0)

    @staticmethod
    def _gemma4_tool_call_to_json(text: str) -> str:
        strings: List[str] = []

        def capture(match: re.Match[str]) -> str:
            strings.append(match.group(1))
            return f"\x00{len(strings) - 1}\x00"

        text = re.sub(r'<\|"\|>(.*?)<\|"\|>', capture, text, flags=re.S)
        text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
        for index, value in enumerate(strings):
            text = text.replace(f"\x00{index}\x00", json.dumps(value))
        return text

    @staticmethod
    def _regex_literal_prefix(pattern: str) -> str:
        literal: List[str] = []
        index = 0
        while index < len(pattern):
            char = pattern[index]
            if char == "\\":
                index += 1
                if index >= len(pattern):
                    break
                escaped = pattern[index]
                if escaped == "n":
                    literal.append("\n")
                elif escaped == "t":
                    literal.append("\t")
                else:
                    literal.append(escaped)
                index += 1
                continue
            if char in "[](){}.*+?|^$":
                break
            literal.append(char)
            index += 1
        return "".join(literal)

    @staticmethod
    def _literal_suffix_prefix_length(text: str, literal: str) -> int:
        max_length = min(len(text), len(literal) - 1)
        if max_length <= 0:
            return 0
        tail = text[-max_length:]
        if literal[0] not in tail:
            return 0
        for prefix_length in range(max_length, 0, -1):
            if text.endswith(literal[:prefix_length]):
                return prefix_length
        return 0

    @classmethod
    def _consume_until_literal(
        cls,
        text: str,
        literal: str,
    ) -> Tuple[str, bool, str, str]:
        if not literal:
            return text, True, "", ""
        literal_length = len(literal)
        search_from = 0
        first_char = literal[0]
        while True:
            marker_index = text.find(first_char, search_from)
            if marker_index < 0:
                return text, False, "", ""
            if text.startswith(literal, marker_index):
                return text[:marker_index], True, text[marker_index + literal_length :], ""
            suffix = text[marker_index:]
            if literal.startswith(suffix):
                return text[:marker_index], False, "", suffix
            search_from = marker_index + 1

    @staticmethod
    def _compile_iterator_pattern(pattern: str) -> Optional[Tuple[str, str]]:
        if "(.*?)" not in pattern:
            return None
        prefix_pattern, suffix_pattern = pattern.split("(.*?)", 1)
        prefix = ResponseParser._regex_literal_prefix(prefix_pattern.lstrip("^"))
        suffix = ResponseParser._regex_literal_prefix(suffix_pattern)
        if not prefix:
            return None
        return prefix, suffix

    @classmethod
    def _compile_iterator_block_pattern(
        cls,
        pattern: str,
    ) -> Optional[Dict[str, Any]]:
        first_group = pattern.find("(")
        if first_group < 0:
            return None
        depth = 0
        last_group = -1
        escaped = False
        for index in range(first_group, len(pattern)):
            char = pattern[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "(":
                depth += 1
                continue
            if char == ")":
                depth -= 1
                if depth == 0:
                    last_group = index
                    break
        if last_group < first_group:
            return None
        start = cls._regex_literal_prefix(pattern[:first_group].lstrip("^"))
        if not start:
            return None
        suffix_pattern = pattern[last_group + 1 :]
        if suffix_pattern == "":
            return {"start": start, "end": "", "allow_eof": True}
        eof_variant = suffix_pattern.endswith("|$)") and suffix_pattern.startswith("(?:")
        if eof_variant:
            suffix_pattern = suffix_pattern[3:-3]
        end = cls._regex_literal_prefix(suffix_pattern)
        if not end:
            return None
        return {"start": start, "end": end, "allow_eof": eof_variant}

    @classmethod
    def _compile_capture_pattern(
        cls,
        pattern: str,
    ) -> Optional[Dict[str, Any]]:
        normalized = pattern.lstrip("^")
        if "(.*?)" in normalized:
            prefix_pattern, suffix_pattern = normalized.split("(.*?)", 1)
            start = cls._regex_literal_prefix(prefix_pattern)
            if not start:
                return None
            if suffix_pattern == "":
                return {"start": start, "end": "", "allow_eof": True}
            eof_variant = suffix_pattern.endswith("|$)") and suffix_pattern.startswith("(?:")
            if eof_variant:
                suffix_pattern = suffix_pattern[3:-3]
            end = cls._regex_literal_prefix(suffix_pattern)
            if not end:
                return None
            return {"start": start, "end": end, "allow_eof": eof_variant}
        if "(.*)" in normalized:
            prefix_pattern, suffix_pattern = normalized.split("(.*)", 1)
            start = cls._regex_literal_prefix(prefix_pattern)
            if not start:
                return None
            if suffix_pattern == "":
                return {"start": start, "end": "", "allow_eof": True}
            eof_variant = suffix_pattern.endswith("|$)") and suffix_pattern.startswith("(?:")
            if eof_variant:
                suffix_pattern = suffix_pattern[3:-3]
            end = cls._regex_literal_prefix(suffix_pattern)
            if not end:
                return None
            return {"start": start, "end": end, "allow_eof": eof_variant}
        return None

    @classmethod
    def _compile_word_capture_pattern(
        cls,
        pattern: str,
    ) -> Optional[Dict[str, Any]]:
        normalized = pattern.lstrip("^")
        if r"(\w+)" not in normalized:
            return None
        prefix_pattern, suffix_pattern = normalized.split(r"(\w+)", 1)
        start = cls._regex_literal_prefix(prefix_pattern)
        if not start:
            return None
        end = cls._regex_literal_prefix(suffix_pattern) if suffix_pattern else ""
        return {
            "kind": "word",
            "start": start,
            "end": end,
        }

    @classmethod
    def _consume_until_any_literal(
        cls,
        text: str,
        literals: Sequence[str],
    ) -> Tuple[str, Optional[str], str, str]:
        for literal in literals:
            if literal and len(text) < len(literal) and literal.startswith(text):
                return "", None, "", text
        literal_first_chars = {literal[0] for literal in literals if literal}
        if literal_first_chars and not any(char in text for char in literal_first_chars):
            return text, None, "", ""
        earliest_index: Optional[int] = None
        earliest_literal: Optional[str] = None
        for literal in literals:
            marker_index = text.find(literal)
            if marker_index < 0:
                continue
            if earliest_index is None or marker_index < earliest_index:
                earliest_index = marker_index
                earliest_literal = literal
        if earliest_index is not None and earliest_literal is not None:
            return (
                text[:earliest_index],
                earliest_literal,
                text[earliest_index + len(earliest_literal) :],
                "",
            )
        overlap = 0
        for literal in literals:
            overlap = max(overlap, cls._literal_suffix_prefix_length(text, literal))
        if overlap:
            return text[:-overlap], None, "", text[-overlap:]
        return text, None, "", ""

    @classmethod
    def _compile_tool_call_item_plan(
        cls,
        item_schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        item_properties = item_schema.get("properties")
        if not isinstance(item_properties, dict):
            return None
        function_schema = item_properties.get("function")
        if not isinstance(function_schema, dict):
            return None
        function_properties = function_schema.get("properties")
        if not isinstance(function_properties, dict):
            return None
        name_schema = function_properties.get("name")
        arguments_schema = function_properties.get("arguments")
        if not isinstance(name_schema, dict) or not isinstance(arguments_schema, dict):
            return None
        name_regex = name_schema.get("x-regex")
        argument_regex = arguments_schema.get("x-regex")
        argument_key_value = arguments_schema.get("x-regex-key-value")
        if (
            name_regex == r"^<function=([^>\n]+)>\n"
            and argument_regex == r"^<function=[^>\n]+>\n(.*?)\n</function>$"
            and argument_key_value
            == r"<parameter=(?P<key>[^>\n]+)>\n(?P<value>.*?)\n</parameter>"
        ):
            return {
                "kind": "tagged-parameters",
                "schema": item_schema,
                "function_start": "<function=",
                "function_name_end": ">\n",
                "function_end": "</function>",
                "parameter_start": "<parameter=",
                "parameter_name_end": ">\n",
                "parameter_end": "\n</parameter>",
            }
        name_capture = (
            cls._compile_word_capture_pattern(name_regex)
            if isinstance(name_regex, str)
            else None
        )
        arguments_capture = (
            cls._compile_capture_pattern(argument_regex)
            if isinstance(argument_regex, str)
            else None
        )
        if (
            isinstance(name_capture, dict)
            and isinstance(arguments_capture, dict)
            and arguments_schema.get("x-parser") == "json"
        ):
            arguments_value_schema = {
                key: value
                for key, value in arguments_schema.items()
                if key != "x-regex"
            }
            return {
                "kind": "json-message",
                "schema": item_schema,
                "name_capture": name_capture,
                "arguments_capture": arguments_capture,
                "arguments_schema": arguments_value_schema,
            }
        return {
            "kind": "buffered",
            "schema": item_schema,
        }

    @classmethod
    def _compile_segment_message_plan(
        cls,
        schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if schema.get("type") != "object":
            return None
        if isinstance(schema.get("x-regex"), str):
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None
        segments: List[Dict[str, Any]] = []
        field_segment_count = 0
        for field_name, value_schema in properties.items():
            if not isinstance(value_schema, dict):
                continue
            if field_name == "tool_calls":
                iterator_pattern = value_schema.get("x-regex-iterator")
                if not isinstance(iterator_pattern, str):
                    continue
                iterator_capture = cls._compile_iterator_block_pattern(iterator_pattern)
                if not isinstance(iterator_capture, dict) or not iterator_capture["start"]:
                    return None
                items_schema = value_schema.get("items")
                if not isinstance(items_schema, dict):
                    return None
                item_plan = cls._compile_tool_call_item_plan(items_schema)
                if item_plan is None:
                    return None
                segments.append(
                    {
                        "kind": "iterator",
                        "field": field_name,
                        "start": iterator_capture["start"],
                        "end": iterator_capture["end"],
                        "allow_eof": iterator_capture["allow_eof"],
                        "item": item_plan,
                    }
                )
                continue
            field_regex = value_schema.get("x-regex")
            if not isinstance(field_regex, str):
                continue
            capture = cls._compile_capture_pattern(field_regex)
            if not isinstance(capture, dict) or not capture["start"]:
                return None
            segments.append(
                {
                    "kind": "field",
                    "field": field_name,
                    "start": capture["start"],
                    "end": capture["end"],
                    "allow_eof": capture["allow_eof"],
                }
            )
            field_segment_count += 1
        if not segments or field_segment_count == 0:
            return None
        start_literals = tuple(segment["start"] for segment in segments)
        if len(start_literals) != len(set(start_literals)):
            return None
        return {
            "kind": "segment-message",
            "segments": segments,
            "segment_starts": start_literals,
            "segments_by_start": {segment["start"]: segment for segment in segments},
        }

    @classmethod
    def _compile_tagged_message_plan(
        cls,
        schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if schema.get("type") != "object":
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None
        tool_calls_schema = properties.get("tool_calls")
        if not isinstance(tool_calls_schema, dict):
            return None
        iterator_pattern = tool_calls_schema.get("x-regex-iterator")
        if not isinstance(iterator_pattern, str):
            return None
        iterator = cls._compile_iterator_pattern(iterator_pattern)
        if iterator is None:
            return None
        items_schema = tool_calls_schema.get("items")
        if not isinstance(items_schema, dict):
            return None
        item_plan = cls._compile_tool_call_item_plan(items_schema)
        if item_plan is None:
            return None
        content_schema = properties.get("content")
        content_regex = (
            content_schema.get("x-regex")
            if isinstance(content_schema, dict)
            else None
        )
        object_regex = schema.get("x-regex") if isinstance(schema.get("x-regex"), str) else None
        assistant_prefix: Optional[str] = None
        if isinstance(content_regex, str) and r"<\|im_start\|>assistant\n" in content_regex:
            assistant_prefix = "<|im_start|>assistant\n"
        leading_capture: Optional[Dict[str, Any]] = None
        for field_name, value_schema in properties.items():
            if not isinstance(value_schema, dict):
                continue
            field_regex = value_schema.get("x-regex")
            if not isinstance(field_regex, str):
                continue
            if "<think>\\n" in field_regex and "</think>" in field_regex:
                leading_capture = {
                    "field": field_name,
                    "start": "<think>\n",
                    "end": "</think>",
                    "strip_after": True,
                    "implicit_at_start": "(?:<think>\\n)?" in field_regex,
                }
                break
        if leading_capture is None and isinstance(object_regex, str):
            if (
                "(?P<thinking>" in object_regex
                and r"<\|channel\>thought\n" in object_regex
                and r"\<channel\|\>" in object_regex
            ):
                leading_capture = {
                    "field": "thinking",
                    "start": "<|channel>thought\n",
                    "end": "<channel|>",
                    "strip_after": False,
                    "implicit_at_start": False,
                }
        end_markers: List[str] = []
        iterator_start, iterator_end = iterator
        if "content" in properties:
            end_markers.append(iterator_start)
        if isinstance(content_regex, str) and r"<\|im_end\|>" in content_regex:
            end_markers.append("<|im_end|>")
        if isinstance(object_regex, str) and r"<turn\|>" in object_regex:
            end_markers.append("<turn|>")
        if not end_markers and iterator_start:
            end_markers.append(iterator_start)
        trim_before_iterator = (
            isinstance(content_regex, str)
            and r"\s*<tool_call>\n" in content_regex
        )
        end_marker_tuple = tuple(end_markers)
        direct_deltas = item_plan["kind"] == "tagged-parameters"
        direct_init = (
            (
                assistant_prefix or "",
                leading_capture["field"] if leading_capture is not None else None,
                leading_capture["start"] if leading_capture is not None else "",
                leading_capture["end"] if leading_capture is not None else "",
                bool(leading_capture.get("strip_after")) if leading_capture is not None else False,
                bool(leading_capture.get("implicit_at_start")) if leading_capture is not None else False,
                trim_before_iterator,
                end_marker_tuple,
                tuple(marker for marker in end_marker_tuple if marker != iterator_start),
                iterator_start,
                iterator_end,
                item_plan["function_start"],
                item_plan["function_name_end"],
                item_plan["function_end"],
                item_plan["parameter_start"],
                item_plan["parameter_name_end"],
                item_plan["parameter_end"],
            )
            if direct_deltas
            else None
        )
        return {
            "kind": "tagged-message",
            "direct_deltas": direct_deltas,
            "assistant_prefix": assistant_prefix,
            "leading_capture": leading_capture,
            "content_field": "content" if "content" in properties else None,
            "content_end_markers": end_marker_tuple,
            "trim_before_iterator": trim_before_iterator,
            "stop_markers": tuple(marker for marker in end_marker_tuple if marker != iterator_start),
            "direct_init": direct_init,
            "iterator": {
                "start": iterator_start,
                "end": iterator_end,
                "item": item_plan,
            },
        }

    @classmethod
    def _compile_stream_plan(
        cls,
        response_schema: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(response_schema, dict):
            return None
        if response_schema.get("x-parser") == "json":
            return {"kind": "json-root"}
        segment_plan = cls._compile_segment_message_plan(response_schema)
        if segment_plan is not None:
            return segment_plan
        return cls._compile_tagged_message_plan(response_schema)

    @classmethod
    def _cached_stream_plan(
        cls,
        response_schema: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(response_schema, dict):
            return None
        cache_key = id(response_schema)
        cached = cls._STREAM_PLAN_CACHE.get(cache_key)
        if cached is not None and cached[0] is response_schema:
            return cached[1]
        plan = cls._compile_stream_plan(response_schema)
        cls._STREAM_PLAN_CACHE[cache_key] = (response_schema, plan)
        return plan

    @staticmethod
    def _tool_schema_map(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if tools is None:
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            name = function.get("name")
            parameters = function.get("parameters")
            if isinstance(name, str) and isinstance(parameters, dict):
                mapping[name] = {
                    "parameters": parameters,
                    "content_type": function.get("content_type"),
                }
        return mapping

    @classmethod
    def _cached_tool_schema_map(
        cls,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if tools is None:
            return {}
        cache_key = id(tools)
        cached = cls._TOOL_SCHEMA_CACHE.get(cache_key)
        if cached is not None and cached[0] is tools:
            return cached[1]
        mapping = cls._tool_schema_map(tools)
        cls._TOOL_SCHEMA_CACHE[cache_key] = (tools, mapping)
        return mapping

    def _parameter_schema_for_tool(
        self, tool_name: str, parameter_name: str
    ) -> Dict[str, Any]:
        tool_schema = self._tool_schemas.get(tool_name)
        if not isinstance(tool_schema, dict):
            return {}
        parameters = tool_schema.get("parameters")
        if not isinstance(parameters, dict):
            return {}
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return {}
        parameter_schema = properties.get(parameter_name)
        if not isinstance(parameter_schema, dict):
            return {}
        return parameter_schema

    def _tool_content_type(self, tool_name: str) -> Optional[str]:
        tool_schema = self._tool_schemas.get(tool_name)
        if not isinstance(tool_schema, dict):
            return None
        content_type = tool_schema.get("content_type")
        if isinstance(content_type, str):
            return content_type
        return None

    def _has_text_tools(self) -> bool:
        return any(
            isinstance(tool_schema, dict) and tool_schema.get("content_type") == "text"
            for tool_schema in self._tool_schemas.values()
        )

    @staticmethod
    def _append_parsed_text(parsed: Dict[str, Any], key: str, text: str) -> None:
        if not text:
            return
        existing = parsed.get(key)
        if isinstance(existing, str):
            parsed[key] = existing + text
        else:
            parsed[key] = text

    def _append_visible_text(self, key: str, text: str) -> None:
        if not text:
            return
        existing = self._message.get(key)
        if isinstance(existing, str):
            self._message[key] = existing + text
        else:
            self._message[key] = text

    @staticmethod
    def _advance_json_scanner(
        item_state: Dict[str, Any],
        text: str,
        *,
        schema_type: Optional[str],
    ) -> bool:
        started = cast(bool, item_state["json_started"])
        complete = cast(bool, item_state["json_complete"])
        depth = cast(int, item_state["json_depth"])
        in_string = cast(bool, item_state["json_in_string"])
        escaped = cast(bool, item_state["json_escaped"])
        for char in text:
            if complete:
                if not char.isspace():
                    return False
                continue
            if not started:
                if char.isspace():
                    continue
                if schema_type == "object" and char != "{":
                    return False
                if schema_type == "array" and char != "[":
                    return False
                if char == "{":
                    started = True
                    depth = 1
                    continue
                if char == "[":
                    started = True
                    depth = 1
                    continue
                return False
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                depth += 1
                continue
            if char in "}]":
                depth -= 1
                if depth < 0:
                    return False
                if depth == 0:
                    complete = True
                continue
        item_state["json_started"] = started
        item_state["json_complete"] = complete
        item_state["json_depth"] = depth
        item_state["json_in_string"] = in_string
        item_state["json_escaped"] = escaped
        return True

    def _new_tool_call_state(self, item_plan: Dict[str, Any]) -> Dict[str, Any]:
        if item_plan["kind"] == "buffered":
            return {"kind": "buffered", "buffer": ""}
        if item_plan["kind"] == "json-message":
            return {
                "kind": "json-message",
                "pending": "",
                "mode": "function-name",
                "tool_call": {
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": "",
                    },
                },
                "arguments_text": "",
                "json_started": False,
                "json_complete": False,
                "json_depth": 0,
                "json_in_string": False,
                "json_escaped": False,
            }
        return {
            "kind": "tagged-parameters",
            "pending": "",
            "mode": "function-header",
            "tool_call": {
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": ResponseParser.PartialJsonObject(OrderedDict(), complete=False),
                },
            },
            "current_parameter": None,
            "current_schema_type": None,
        }

    def _new_stream_state(self, plan: Dict[str, Any]) -> ResponseParser.StreamState:
        if plan["kind"] == "json-root":
            return ResponseParser.StreamState(
                kind="json-root",
                pending="",
                mode="prelude",
                current_item=None,
                current_segment=None,
                done=False,
                saw_tool_calls=False,
                parsed={"role": "assistant"},
                buffer="",
            )
        if plan.get("direct_deltas"):
            return ResponseParser.StreamState(
                kind="tagged-message",
                pending="",
                mode="assistant-prefix" if plan.get("assistant_prefix") else "prelude",
                current_item=None,
                current_segment=None,
                done=False,
                saw_tool_calls=False,
                tool_call_count=0,
            )
        if plan["kind"] == "segment-message":
            return ResponseParser.StreamState(
                kind="segment-message",
                pending="",
                mode="segment-start",
                current_item=None,
                current_segment=None,
                done=False,
                saw_tool_calls=False,
                parsed={"role": "assistant"},
            )
        return ResponseParser.StreamState(
            kind="tagged-message",
            pending="",
            mode="assistant-prefix" if plan.get("assistant_prefix") else "prelude",
            current_item=None,
            current_segment=None,
            done=False,
            saw_tool_calls=False,
            parsed={"role": "assistant"},
        )

    def _start_direct_tool_call(self, tool_call_index: int) -> None:
        self._item.pending = ""
        self._item.mode = self.ITEM_MODE_FUNCTION_HEADER
        self._item.tool_call_index = tool_call_index
        self._item.tool_name = ""
        self._item.parameter_count = 0
        tool_calls = self._message.setdefault("tool_calls", [])
        assert isinstance(tool_calls, list)
        while len(tool_calls) <= tool_call_index:
            tool_calls.append({"function": {"name": "", "arguments": ""}})
        self._item.visible_tool_call = tool_calls[tool_call_index]
        function = self._item.visible_tool_call.setdefault("function", {})
        assert isinstance(function, dict)
        self._item.visible_function = function
        if tool_call_index == 0:
            self._message["function_call"] = self._item.visible_function
        self._item.current_parameter = None
        self._item.current_parameter_value = ""
        self._item.current_schema_type = None
        self._item.current_parameter_schema = {}

    def _advance_direct_tool_call_state(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        deltas: List[Dict[str, Any]] = []
        mode = self._item.mode
        tool_call_index = self._item.tool_call_index
        tool_name = self._item.tool_name
        parameter_count = self._item.parameter_count
        visible_tool_call = self._item.visible_tool_call
        visible_function = self._item.visible_function
        current_parameter = self._item.current_parameter
        current_parameter_value = self._item.current_parameter_value
        current_schema_type = self._item.current_schema_type
        current_parameter_schema = self._item.current_parameter_schema
        buffer = self._item.pending + text
        function_start = self._direct.function_start
        function_name_end = self._direct.function_name_end
        function_end = self._direct.function_end
        parameter_start = self._direct.parameter_start
        parameter_name_end = self._direct.parameter_name_end
        parameter_end = self._direct.parameter_end

        while True:
            if mode == self.ITEM_MODE_FUNCTION_HEADER:
                if buffer.startswith(function_start):
                    buffer = buffer[len(function_start) :]
                elif function_start.startswith(buffer):
                    self._item.pending = buffer
                    self._item.mode = mode
                    return True, deltas
                else:
                    self._item.pending = ""
                    return False, deltas
                delimiter_index = buffer.find(function_name_end)
                if delimiter_index < 0:
                    self._item.pending = function_start + buffer
                    self._item.mode = mode
                    return True, deltas
                function_name = buffer[:delimiter_index]
                if not function_name:
                    self._item.pending = ""
                    return False, deltas
                tool_name = function_name
                assert visible_tool_call is not None
                assert visible_function is not None
                visible_function["name"] = function_name
                visible_function["arguments"] = "{"
                visible_tool_call["id"] = (
                    f"call_{self._choice_index}_{function_name}_{self._completion_id}_{tool_call_index}"
                )
                visible_tool_call["type"] = "function"
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": tool_call_index,
                                "id": (
                                    f"call_{self._choice_index}_{function_name}_"
                                    f"{self._completion_id}_{tool_call_index}"
                                ),
                                "type": "function",
                                "function": {"name": function_name, "arguments": "{"},
                            }
                        ]
                    }
                )
                buffer = buffer[delimiter_index + len(function_name_end) :]
                mode = self.ITEM_MODE_AFTER_FUNCTION_HEADER
                continue
            if mode == self.ITEM_MODE_AFTER_FUNCTION_HEADER:
                if buffer.startswith("\n"):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith(parameter_start):
                    mode = self.ITEM_MODE_PARAMETER_NAME
                    continue
                if buffer.startswith(function_end):
                    buffer = buffer[len(function_end) :]
                    assert visible_function is not None
                    visible_function["arguments"] += "}"
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "function": {"arguments": "}"},
                                }
                            ]
                        }
                    )
                    mode = self.ITEM_MODE_DONE
                    continue
                if parameter_start.startswith(buffer) or function_end.startswith(buffer):
                    self._item.pending = buffer
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    return True, deltas
                if not buffer:
                    self._item.pending = ""
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    return True, deltas
                self._item.pending = ""
                return False, deltas
            if mode == self.ITEM_MODE_PARAMETER_NAME:
                if buffer.startswith(parameter_start):
                    buffer = buffer[len(parameter_start) :]
                elif parameter_start.startswith(buffer):
                    self._item.pending = buffer
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    self._item.parameter_count = parameter_count
                    return True, deltas
                else:
                    self._item.pending = ""
                    return False, deltas
                delimiter_index = buffer.find(parameter_name_end)
                if delimiter_index < 0:
                    self._item.pending = parameter_start + buffer
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    self._item.parameter_count = parameter_count
                    return True, deltas
                parameter_name = buffer[:delimiter_index]
                if not parameter_name:
                    self._item.pending = ""
                    return False, deltas
                parameter_schema = self._parameter_schema_for_tool(tool_name, parameter_name)
                schema_type = parameter_schema.get("type") if isinstance(parameter_schema, dict) else None
                current_parameter = parameter_name
                current_parameter_value = ""
                current_schema_type = schema_type if isinstance(schema_type, str) else None
                current_parameter_schema = parameter_schema
                key_prefix = json.dumps(parameter_name, ensure_ascii=False, separators=(",", ":")) + ":"
                if parameter_count > 0:
                    key_prefix = "," + key_prefix
                if schema_type in {None, "string"}:
                    key_prefix += '"'
                parameter_count += 1
                assert visible_function is not None
                visible_function["arguments"] += key_prefix
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": tool_call_index,
                                "function": {"arguments": key_prefix},
                            }
                        ]
                    }
                )
                buffer = buffer[delimiter_index + len(parameter_name_end) :]
                mode = self.ITEM_MODE_PARAMETER_VALUE
                continue
            if mode == self.ITEM_MODE_PARAMETER_VALUE:
                value_delta, matched, remainder, pending = self._consume_until_literal(buffer, parameter_end)
                current_parameter_value += value_delta
                if value_delta:
                    assert visible_function is not None
                    visible_function["arguments"] += value_delta
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "function": {"arguments": value_delta},
                                }
                            ]
                        }
                    )
                if not matched:
                    self._item.pending = pending
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    self._item.parameter_count = parameter_count
                    self._item.current_parameter = current_parameter
                    self._item.current_parameter_value = current_parameter_value
                    self._item.current_schema_type = current_schema_type
                    self._item.current_parameter_schema = current_parameter_schema
                    return True, deltas
                self._coerce_tool_argument(
                    cast(str, current_parameter_value),
                    cast(Dict[str, Any], current_parameter_schema),
                    tool_name=tool_name,
                    argument_name=cast(str, current_parameter),
                )
                if current_schema_type in {None, "string"}:
                    assert visible_function is not None
                    visible_function["arguments"] += '"'
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "function": {"arguments": '"'},
                                }
                            ]
                        }
                    )
                current_parameter = None
                current_parameter_value = ""
                current_schema_type = None
                current_parameter_schema = {}
                buffer = remainder
                mode = self.ITEM_MODE_AFTER_PARAMETER
                continue
            if mode == self.ITEM_MODE_AFTER_PARAMETER:
                if buffer.startswith("\n"):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith(parameter_start):
                    mode = self.ITEM_MODE_PARAMETER_NAME
                    continue
                if buffer.startswith(function_end):
                    buffer = buffer[len(function_end) :]
                    assert visible_function is not None
                    visible_function["arguments"] += "}"
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "function": {"arguments": "}"},
                                }
                            ]
                        }
                    )
                    mode = self.ITEM_MODE_DONE
                    continue
                if parameter_start.startswith(buffer) or function_end.startswith(buffer):
                    self._item.pending = buffer
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    self._item.parameter_count = parameter_count
                    return True, deltas
                if not buffer:
                    self._item.pending = ""
                    self._item.mode = mode
                    self._item.tool_name = tool_name
                    self._item.parameter_count = parameter_count
                    return True, deltas
                self._item.pending = ""
                return False, deltas
            if mode == self.ITEM_MODE_DONE:
                if buffer.strip():
                    self._item.pending = ""
                    return False, deltas
                self._item.pending = buffer
                self._item.mode = mode
                self._item.tool_name = tool_name
                self._item.parameter_count = parameter_count
                return True, deltas

    def _advance_direct_stream_state(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        deltas: List[Dict[str, Any]] = []
        mode = self._direct.mode
        tool_call_count = self._direct.tool_call_count
        saw_tool_calls = self._direct.saw_tool_calls
        done = self._direct.done
        buffer = self._direct.pending + text
        assistant_prefix = self._direct.assistant_prefix
        leading_capture_field = self._direct.leading_capture_field
        leading_capture_start = self._direct.leading_capture_start
        leading_capture_end = self._direct.leading_capture_end
        leading_capture_strip_after = self._direct.leading_capture_strip_after
        leading_capture_implicit = self._direct.leading_capture_implicit
        iterator_start = self._direct.iterator_start
        iterator_end = self._direct.iterator_end
        content_end_markers = self._direct.content_end_markers
        stop_markers = self._direct.stop_markers

        while True:
            if mode == self.DIRECT_MODE_ASSISTANT_PREFIX:
                if buffer.startswith(assistant_prefix):
                    buffer = buffer[len(assistant_prefix) :]
                    mode = self.DIRECT_MODE_PRELUDE
                    continue
                if assistant_prefix.startswith(buffer):
                    self._direct.pending = buffer
                    self._direct.mode = mode
                    return True, deltas
                mode = self.DIRECT_MODE_PRELUDE
                continue
            if mode == self.DIRECT_MODE_PRELUDE:
                if leading_capture_field is not None:
                    if buffer.startswith(leading_capture_start):
                        buffer = buffer[len(leading_capture_start) :]
                        mode = self.DIRECT_MODE_LEADING_CAPTURE
                        continue
                    if leading_capture_start.startswith(buffer):
                        self._direct.pending = buffer
                        self._direct.mode = mode
                        return True, deltas
                    if leading_capture_implicit:
                        tool_index = buffer.find(iterator_start)
                        end_index = buffer.find(leading_capture_end)
                        overlap = self._literal_suffix_prefix_length(buffer, leading_capture_end)
                        if ((end_index >= 0 and (tool_index < 0 or end_index < tool_index)) or overlap):
                            mode = self.DIRECT_MODE_LEADING_CAPTURE
                            continue
                mode = self.DIRECT_MODE_CONTENT
                continue
            if mode == self.DIRECT_MODE_LEADING_CAPTURE:
                segment, matched, remainder, pending = self._consume_until_literal(buffer, leading_capture_end)
                if segment:
                    assert leading_capture_field is not None
                    self._append_visible_text(leading_capture_field, segment)
                    deltas.append({leading_capture_field: segment})
                if not matched:
                    self._direct.pending = pending
                    self._direct.mode = mode
                    self._direct.tool_call_count = tool_call_count
                    self._direct.saw_tool_calls = saw_tool_calls
                    self._direct.done = done
                    return True, deltas
                buffer = remainder.lstrip() if leading_capture_strip_after else remainder
                mode = self.DIRECT_MODE_CONTENT
                continue
            if mode == self.DIRECT_MODE_CONTENT:
                segment, matched_marker, remainder, pending = self._consume_until_any_literal(
                    buffer,
                    content_end_markers,
                )
                content_segment = segment.rstrip() if matched_marker == iterator_start and self._direct.trim_before_iterator else segment
                if content_segment:
                    self._append_visible_text("content", content_segment)
                    deltas.append({"content": content_segment})
                if matched_marker is None:
                    self._direct.pending = pending
                    self._direct.mode = mode
                    self._direct.tool_call_count = tool_call_count
                    self._direct.saw_tool_calls = saw_tool_calls
                    self._direct.done = done
                    return True, deltas
                if matched_marker == iterator_start:
                    saw_tool_calls = True
                    self._start_direct_tool_call(tool_call_count)
                    tool_call_count += 1
                    mode = self.DIRECT_MODE_TOOL_ITEM
                    buffer = remainder
                    continue
                done = True
                self._direct.pending = ""
                self._direct.mode = mode
                self._direct.tool_call_count = tool_call_count
                self._direct.saw_tool_calls = saw_tool_calls
                self._direct.done = done
                return True, deltas
            if mode == self.DIRECT_MODE_TOOL_ITEM:
                item_text, matched, remainder, pending = self._consume_until_literal(buffer, iterator_end)
                if item_text:
                    success, item_deltas = self._advance_direct_tool_call_state(item_text)
                    if not success:
                        self._direct.pending = ""
                        self._direct.mode = mode
                        self._direct.tool_call_count = tool_call_count
                        self._direct.saw_tool_calls = saw_tool_calls
                        self._direct.done = done
                        return False, deltas
                    deltas.extend(item_deltas)
                if not matched:
                    self._direct.pending = pending
                    self._direct.mode = mode
                    self._direct.tool_call_count = tool_call_count
                    self._direct.saw_tool_calls = saw_tool_calls
                    self._direct.done = done
                    return True, deltas
                buffer = remainder.lstrip()
                mode = self.DIRECT_MODE_AFTER_TOOL_ITEM
                continue
            if mode == self.DIRECT_MODE_AFTER_TOOL_ITEM:
                if not buffer:
                    self._direct.pending = ""
                    self._direct.mode = mode
                    self._direct.tool_call_count = tool_call_count
                    self._direct.saw_tool_calls = saw_tool_calls
                    self._direct.done = done
                    return True, deltas
                if buffer.startswith(iterator_start):
                    saw_tool_calls = True
                    self._start_direct_tool_call(tool_call_count)
                    tool_call_count += 1
                    mode = self.DIRECT_MODE_TOOL_ITEM
                    buffer = buffer[len(iterator_start) :]
                    continue
                for marker in stop_markers:
                    if buffer.startswith(marker):
                        done = True
                        self._direct.pending = ""
                        self._direct.mode = mode
                        self._direct.tool_call_count = tool_call_count
                        self._direct.saw_tool_calls = saw_tool_calls
                        self._direct.done = done
                        return True, deltas
                if any(marker.startswith(buffer) for marker in content_end_markers) or buffer.isspace():
                    self._direct.pending = buffer
                    self._direct.mode = mode
                    self._direct.tool_call_count = tool_call_count
                    self._direct.saw_tool_calls = saw_tool_calls
                    self._direct.done = done
                    return True, deltas
                self._direct.pending = ""
                self._direct.mode = mode
                self._direct.tool_call_count = tool_call_count
                self._direct.saw_tool_calls = saw_tool_calls
                self._direct.done = done
                return False, deltas

    def _advance_tool_call_state(
        self,
        item_state: Dict[str, Any],
        text: str,
        item_plan: Dict[str, Any],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        deltas: List[Dict[str, Any]] = []
        if item_plan["kind"] == "buffered":
            item_state["buffer"] = item_state["buffer"] + text
            return True, deltas
        if item_plan["kind"] == "json-message":
            buffer = item_state["pending"] + text
            item_state["pending"] = ""
            while True:
                mode = item_state["mode"]
                if mode == "function-name":
                    name_capture = item_plan["name_capture"]
                    name_prefix = name_capture["start"]
                    if buffer.startswith(name_prefix):
                        remainder = buffer[len(name_prefix) :]
                    elif name_prefix.startswith(buffer):
                        item_state["pending"] = buffer
                        return True, deltas
                    else:
                        return False, deltas
                    name_end = 0
                    while name_end < len(remainder) and (
                        remainder[name_end].isalnum() or remainder[name_end] == "_"
                    ):
                        name_end += 1
                    if name_end == 0:
                        if not remainder:
                            item_state["pending"] = buffer
                            return True, deltas
                        return False, deltas
                    if name_end == len(remainder):
                        item_state["pending"] = buffer
                        return True, deltas
                    function_name = remainder[:name_end]
                    item_state["tool_call"]["function"]["name"] = function_name
                    tool_call_index = cast(int, item_state["tool_call_index"])
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": (
                                        f"call_{self._choice_index}_{function_name}_"
                                        f"{self._completion_id}_{tool_call_index}"
                                    ),
                                    "type": "function",
                                    "function": {"name": function_name},
                                }
                            ]
                        }
                    )
                    buffer = remainder[name_end:]
                    item_state["mode"] = "seek-arguments"
                    continue
                if mode == "seek-arguments":
                    arguments_capture = item_plan["arguments_capture"]
                    _ignored, matched, remainder, pending = self._consume_until_literal(
                        buffer,
                        arguments_capture["start"],
                    )
                    if not matched:
                        item_state["pending"] = pending
                        return True, deltas
                    buffer = remainder
                    item_state["mode"] = "arguments"
                    continue
                if mode == "arguments":
                    if buffer:
                        item_state["arguments_text"] = item_state["arguments_text"] + buffer
                        deltas.append(
                            {
                                "tool_calls": [
                                    {
                                        "index": cast(int, item_state["tool_call_index"]),
                                        "function": {"arguments": buffer},
                                    }
                                ]
                            }
                        )
                    item_state["pending"] = ""
                    arguments_schema = cast(Dict[str, Any], item_plan["arguments_schema"])
                    schema_type = arguments_schema.get("type")
                    if not self._advance_json_scanner(
                        item_state,
                        buffer,
                        schema_type=schema_type if isinstance(schema_type, str) else None,
                    ):
                        return False, deltas
                    return True, deltas

        buffer = item_state["pending"] + text
        item_state["pending"] = ""
        while True:
            mode = item_state["mode"]
            if mode == "function-header":
                function_start = item_plan["function_start"]
                if buffer.startswith(function_start):
                    buffer = buffer[len(function_start) :]
                elif function_start.startswith(buffer):
                    item_state["pending"] = buffer
                    return True, deltas
                else:
                    return False, deltas
                function_name_end = item_plan["function_name_end"]
                delimiter_index = buffer.find(function_name_end)
                if delimiter_index < 0:
                    item_state["pending"] = function_start + buffer
                    return True, deltas
                function_name = buffer[:delimiter_index]
                if not function_name:
                    return False, deltas
                item_state["tool_call"]["function"]["name"] = function_name
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": item_state["tool_call_index"],
                                "id": (
                                    f"call_{self._choice_index}_{function_name}_"
                                    f"{self._completion_id}_{item_state['tool_call_index']}"
                                ),
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": "{",
                                },
                            }
                        ]
                    }
                )
                buffer = buffer[delimiter_index + len(function_name_end) :]
                item_state["mode"] = "after-function-header"
                continue
            if mode == "after-function-header":
                if buffer.startswith("\n"):
                    buffer = buffer[1:]
                    continue
                function_end = item_plan["function_end"]
                parameter_start = item_plan["parameter_start"]
                if buffer.startswith(parameter_start):
                    item_state["mode"] = "parameter-name"
                    continue
                if buffer.startswith(function_end):
                    buffer = buffer[len(function_end) :]
                    item_state["mode"] = "done"
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": item_state["tool_call_index"],
                                    "function": {"arguments": "}"},
                                }
                            ]
                        }
                    )
                    continue
                if parameter_start.startswith(buffer) or function_end.startswith(buffer):
                    item_state["pending"] = buffer
                    return True, deltas
                if not buffer:
                    return True, deltas
                return False, deltas
            if mode == "parameter-name":
                parameter_start = item_plan["parameter_start"]
                if buffer.startswith(parameter_start):
                    buffer = buffer[len(parameter_start) :]
                elif parameter_start.startswith(buffer):
                    item_state["pending"] = buffer
                    return True, deltas
                else:
                    return False, deltas
                parameter_name_end = item_plan["parameter_name_end"]
                delimiter_index = buffer.find(parameter_name_end)
                if delimiter_index < 0:
                    item_state["pending"] = parameter_start + buffer
                    return True, deltas
                parameter_name = buffer[:delimiter_index]
                if not parameter_name:
                    return False, deltas
                function = item_state["tool_call"]["function"]
                arguments = cast(ResponseParser.PartialJsonObject, function["arguments"])
                tool_name = function["name"]
                parameter_schema = self._parameter_schema_for_tool(tool_name, parameter_name)
                schema_type = (
                    parameter_schema.get("type")
                    if isinstance(parameter_schema, dict)
                    else None
                )
                arguments.value[parameter_name] = ResponseParser.PartialJsonValue(
                    text="",
                    schema_type=schema_type if isinstance(schema_type, str) else None,
                    complete=False,
                )
                key_prefix = json.dumps(
                    parameter_name,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + ":"
                if len(arguments.value) > 1:
                    key_prefix = "," + key_prefix
                if schema_type in {None, "string"}:
                    key_prefix += '"'
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": item_state["tool_call_index"],
                                "function": {"arguments": key_prefix},
                            }
                        ]
                    }
                )
                item_state["current_parameter"] = parameter_name
                item_state["current_schema_type"] = (
                    schema_type if isinstance(schema_type, str) else None
                )
                buffer = buffer[delimiter_index + len(parameter_name_end) :]
                item_state["mode"] = "parameter-value"
                continue
            if mode == "parameter-value":
                parameter_end = item_plan["parameter_end"]
                value_delta, matched, remainder, pending = self._consume_until_literal(
                    buffer,
                    parameter_end,
                )
                function = item_state["tool_call"]["function"]
                arguments = cast(ResponseParser.PartialJsonObject, function["arguments"])
                parameter_name = cast(str, item_state["current_parameter"])
                current_value = arguments.value[parameter_name]
                assert isinstance(current_value, ResponseParser.PartialJsonValue)
                current_value.text = current_value.text + value_delta
                if value_delta:
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": item_state["tool_call_index"],
                                    "function": {"arguments": value_delta},
                                }
                            ]
                        }
                    )
                if not matched:
                    item_state["pending"] = pending
                    return True, deltas
                tool_name = function["name"]
                parameter_schema = self._parameter_schema_for_tool(tool_name, parameter_name)
                self._coerce_tool_argument(
                    current_value.text,
                    parameter_schema,
                    tool_name=tool_name,
                    argument_name=parameter_name,
                )
                current_value.complete = True
                if current_value.schema_type in {None, "string"}:
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": item_state["tool_call_index"],
                                    "function": {"arguments": '"'},
                                }
                            ]
                        }
                    )
                item_state["current_parameter"] = None
                item_state["current_schema_type"] = None
                buffer = remainder
                item_state["mode"] = "after-parameter"
                continue
            if mode == "after-parameter":
                if buffer.startswith("\n"):
                    buffer = buffer[1:]
                    continue
                function_end = item_plan["function_end"]
                parameter_start = item_plan["parameter_start"]
                if buffer.startswith(parameter_start):
                    item_state["mode"] = "parameter-name"
                    continue
                if buffer.startswith(function_end):
                    buffer = buffer[len(function_end) :]
                    cast(
                        ResponseParser.PartialJsonObject,
                        item_state["tool_call"]["function"]["arguments"],
                    ).complete = True
                    item_state["mode"] = "done"
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": item_state["tool_call_index"],
                                    "function": {"arguments": "}"},
                                }
                            ]
                        }
                    )
                    continue
                if parameter_start.startswith(buffer) or function_end.startswith(buffer):
                    item_state["pending"] = buffer
                    return True, deltas
                if not buffer:
                    return True, deltas
                return False, deltas
            if mode == "done":
                if buffer.strip():
                    return False, deltas
                item_state["pending"] = buffer
                return True, deltas

    def _finish_tool_call_state(
        self,
        item_state: Dict[str, Any],
        item_plan: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if item_plan["kind"] == "buffered":
            parsed_item = self._parse_response_value(
                item_state["buffer"],
                item_plan["schema"],
                partial=False,
            )
            return self._normalize_tool_call_item(parsed_item, partial=False)
        if item_plan["kind"] == "json-message":
            if item_state["mode"] != "arguments":
                return None
            if item_state["pending"] or not item_state["json_started"] or not item_state["json_complete"]:
                return None
            try:
                arguments = self._parse_response_value(
                    item_state["arguments_text"],
                    item_plan["arguments_schema"],
                    partial=False,
                )
            except CompletionResponseParsingError:
                return None
            if arguments is None:
                return None
            item_state["tool_call"]["function"]["arguments"] = arguments
            return cast(Dict[str, Any], item_state["tool_call"])
        if item_state["pending"] or item_state["current_parameter"] is not None:
            return None
        if item_state["mode"] not in {"done", "after-function-header"}:
            return None
        return cast(Dict[str, Any], item_state["tool_call"])

    def _tool_call_delta(
        self,
        *,
        tool_call: Dict[str, Any],
        tool_call_index: int,
        partial: bool,
    ) -> Dict[str, Any]:
        function = cast(Dict[str, Any], tool_call["function"])
        return {
            "tool_calls": [
                {
                    "index": tool_call_index,
                    "id": (
                        f"call_{self._choice_index}_{function['name']}_"
                        f"{self._completion_id}_{tool_call_index}"
                    ),
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": function["name"],
                        "arguments": self._serialize_tool_arguments(
                            function["arguments"],
                            partial=partial,
                        ),
                    },
                }
            ]
        }

    def _advance_stream_state(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        deltas: List[Dict[str, Any]] = []
        if self._stream_plan is None:
            return False, deltas
        if self._direct.deltas:
            return self._advance_direct_stream_state(text)
        if self._stream_state is None:
            return False, deltas
        if self._stream_plan["kind"] == "segment-message":
            state = self._stream_state
            parsed = cast(Dict[str, Any], state.parsed)
            buffer = state.pending + text
            state.pending = ""
            while True:
                if state.mode == "segment-start":
                    ignored, matched_start, remainder, pending = self._consume_until_any_literal(
                        buffer,
                        cast(Tuple[str, ...], self._stream_plan["segment_starts"]),
                    )
                    if matched_start is None:
                        state.pending = buffer if not pending else pending
                        return True, deltas
                    state.current_segment = cast(
                        Dict[str, Any],
                        self._stream_plan["segments_by_start"][matched_start],
                    )
                    buffer = remainder
                    state.mode = (
                        "segment-tool-item"
                        if state.current_segment["kind"] == "iterator"
                        else "segment-field"
                    )
                    if state.current_segment["kind"] == "iterator":
                        item_state = self._new_tool_call_state(state.current_segment["item"])
                        if item_state["kind"] in {"tagged-parameters", "json-message"}:
                            tool_calls = cast(
                                List[Dict[str, Any]],
                                parsed.setdefault("tool_calls", []),
                            )
                            tool_calls.append(item_state["tool_call"])
                            item_state["tool_call_index"] = len(tool_calls) - 1
                        state.current_item = item_state
                        state.saw_tool_calls = True
                    continue
                if state.mode == "segment-field":
                    current_segment = state.current_segment
                    if not isinstance(current_segment, dict):
                        return False, deltas
                    segment_text, matched, remainder, pending = self._consume_until_literal(
                        buffer,
                        cast(str, current_segment["end"]),
                    )
                    field_name = cast(str, current_segment["field"])
                    self._append_parsed_text(parsed, field_name, segment_text)
                    if segment_text:
                        deltas.append({field_name: segment_text})
                    if not matched:
                        state.pending = pending
                        return True, deltas
                    buffer = remainder
                    state.current_segment = None
                    state.mode = "segment-start"
                    continue
                if state.mode == "segment-tool-item":
                    current_segment = state.current_segment
                    if not isinstance(current_segment, dict):
                        return False, deltas
                    active_item_state = state.current_item
                    if not isinstance(active_item_state, dict):
                        return False, deltas
                    item_text, matched, remainder, pending = self._consume_until_literal(
                        buffer,
                        cast(str, current_segment["end"]),
                    )
                    if item_text:
                        success, item_deltas = self._advance_tool_call_state(
                            active_item_state,
                            item_text,
                            cast(Dict[str, Any], current_segment["item"]),
                        )
                        if not success:
                            return False, deltas
                        deltas.extend(item_deltas)
                    if not matched:
                        state.pending = pending
                        return True, deltas
                    tool_call = self._finish_tool_call_state(
                        active_item_state,
                        cast(Dict[str, Any], current_segment["item"]),
                    )
                    if tool_call is None:
                        return False, deltas
                    if active_item_state["kind"] == "buffered":
                        tool_calls = cast(List[Dict[str, Any]], parsed.setdefault("tool_calls", []))
                        tool_call_index = len(tool_calls)
                        tool_calls.append(tool_call)
                        deltas.append(
                            self._tool_call_delta(
                                tool_call=tool_call,
                                tool_call_index=tool_call_index,
                                partial=False,
                            )
                        )
                    state.current_item = None
                    state.current_segment = None
                    state.mode = "segment-start"
                    buffer = remainder
                    continue
                return False, deltas
        if self._stream_plan["kind"] == "json-root":
            buffer = self._stream_state.buffer + text
            try:
                parsed = self.parse_chat_response(buffer, partial=True)
            except CompletionResponseParsingError:
                return False, deltas
            self._stream_state.buffer = buffer
            self._stream_state.parsed = parsed
            return True, deltas

        state = self._stream_state
        plan = self._stream_plan
        buffer = state.pending + text
        state.pending = ""
        parsed = cast(Dict[str, Any], state.parsed)
        iterator_start = plan["iterator"]["start"]
        iterator_end = plan["iterator"]["end"]
        while True:
            mode = state.mode
            if mode == "assistant-prefix":
                assistant_prefix = plan.get("assistant_prefix")
                if not assistant_prefix:
                    state.mode = "prelude"
                    continue
                if buffer.startswith(assistant_prefix):
                    buffer = buffer[len(assistant_prefix) :]
                    state.mode = "prelude"
                    continue
                if assistant_prefix.startswith(buffer):
                    state.pending = buffer
                    return True, deltas
                state.mode = "prelude"
                continue
            if mode == "prelude":
                leading_capture = plan.get("leading_capture")
                if leading_capture is not None:
                    capture_start = leading_capture["start"]
                    if buffer.startswith(capture_start):
                        buffer = buffer[len(capture_start) :]
                        state.mode = "leading-capture"
                        continue
                    if capture_start.startswith(buffer):
                        state.pending = buffer
                        return True, deltas
                    if leading_capture.get("implicit_at_start"):
                        capture_end = leading_capture["end"]
                        tool_index = buffer.find(iterator_start)
                        end_index = buffer.find(capture_end)
                        overlap = self._literal_suffix_prefix_length(buffer, capture_end)
                        if (
                            end_index >= 0
                            and (tool_index < 0 or end_index < tool_index)
                        ) or overlap:
                            state.mode = "leading-capture"
                            continue
                state.mode = "content"
                continue
            if mode == "leading-capture":
                leading_capture = cast(Dict[str, Any], plan["leading_capture"])
                segment, matched, remainder, pending = self._consume_until_literal(
                    buffer,
                    leading_capture["end"],
                )
                self._append_parsed_text(parsed, leading_capture["field"], segment)
                if segment:
                    deltas.append({leading_capture["field"]: segment})
                if not matched:
                    state.pending = pending
                    return True, deltas
                buffer = remainder.lstrip() if leading_capture.get("strip_after") else remainder
                state.mode = "content"
                continue
            if mode == "content":
                content_field = plan.get("content_field")
                end_markers = cast(List[str], plan["content_end_markers"])
                segment, matched_marker, remainder, pending = self._consume_until_any_literal(
                    buffer,
                    end_markers,
                )
                if content_field is not None:
                    content_segment = (
                        segment.rstrip()
                        if matched_marker == iterator_start and plan.get("trim_before_iterator")
                        else segment
                    )
                    self._append_parsed_text(parsed, content_field, content_segment)
                    if content_segment:
                        deltas.append({content_field: content_segment})
                if matched_marker is None:
                    state.pending = pending
                    return True, deltas
                if matched_marker == iterator_start:
                    item_state = self._new_tool_call_state(plan["iterator"]["item"])
                    state.saw_tool_calls = True
                    if item_state["kind"] == "tagged-parameters":
                        tool_calls = cast(List[Dict[str, Any]], parsed.setdefault("tool_calls", []))
                        tool_calls.append(item_state["tool_call"])
                        item_state["tool_call_index"] = len(tool_calls) - 1
                    state.current_item = item_state
                    state.mode = "tool-item"
                    buffer = remainder
                    continue
                state.done = True
                state.pending = ""
                return True, deltas
            if mode == "tool-item":
                current_item_state: Optional[Dict[str, Any]] = state.current_item
                if not isinstance(current_item_state, dict):
                    return False, deltas
                item_text, matched, remainder, pending = self._consume_until_literal(
                    buffer,
                    iterator_end,
                )
                if item_text:
                    success, item_deltas = self._advance_tool_call_state(
                        current_item_state,
                        item_text,
                        plan["iterator"]["item"],
                    )
                    if not success:
                        return False, deltas
                    deltas.extend(item_deltas)
                if not matched:
                    state.pending = pending
                    return True, deltas
                tool_call = self._finish_tool_call_state(
                    current_item_state,
                    plan["iterator"]["item"],
                )
                if tool_call is None:
                    return False, deltas
                if current_item_state["kind"] == "buffered":
                    parsed.setdefault("tool_calls", []).append(tool_call)
                state.current_item = None
                buffer = remainder.lstrip()
                state.mode = "after-tool-item"
                continue
            if mode == "after-tool-item":
                end_markers = [
                    iterator_start,
                    *[
                        marker
                        for marker in cast(List[str], plan["content_end_markers"])
                        if marker != iterator_start
                    ],
                ]
                if not buffer:
                    state.pending = ""
                    return True, deltas
                if buffer.startswith(iterator_start):
                    item_state = self._new_tool_call_state(plan["iterator"]["item"])
                    state.saw_tool_calls = True
                    if item_state["kind"] == "tagged-parameters":
                        tool_calls = cast(List[Dict[str, Any]], parsed.setdefault("tool_calls", []))
                        tool_calls.append(item_state["tool_call"])
                        item_state["tool_call_index"] = len(tool_calls) - 1
                    state.current_item = item_state
                    state.mode = "tool-item"
                    buffer = buffer[len(iterator_start) :]
                    continue
                for marker in end_markers[1:]:
                    if buffer.startswith(marker):
                        state.done = True
                        state.pending = ""
                        return True, deltas
                if any(marker.startswith(buffer) for marker in end_markers):
                    state.pending = buffer
                    return True, deltas
                if buffer.isspace():
                    state.pending = buffer
                    return True, deltas
                return False, deltas

    @staticmethod
    def _partial_regex_key_value_item(
        pattern: str,
        text: str,
        *,
        min_start: int,
    ) -> Optional[Tuple[str, str]]:
        value_group = "(?P<value>"
        value_group_start = pattern.find(value_group)
        if value_group_start < 0:
            return None
        partial_pattern = pattern[:value_group_start] + r"(?P<value>.*)\Z"
        partial_match: Optional[re.Match[str]] = None
        for match in re.finditer(partial_pattern, text, re.S):
            if match.start() < min_start:
                continue
            partial_match = match
        if partial_match is None:
            return None
        group_dict = partial_match.groupdict()
        key = group_dict.get("key")
        value = group_dict.get("value")
        if key is None or value is None:
            return None
        value_pattern = "(?P<value>.*?)"
        if value_pattern in pattern:
            suffix_literal = ResponseParser._regex_literal_prefix(
                pattern.split(value_pattern, 1)[1]
            )
            for suffix_length in range(len(suffix_literal), 0, -1):
                suffix_prefix = suffix_literal[:suffix_length]
                if value.endswith(suffix_prefix):
                    value = value[:-suffix_length]
                    break
        return key, value

    def _trim_partial_tool_call_prefix(
        self,
        *,
        response_text: str,
        parsed: Dict[str, Any],
    ) -> None:
        if not isinstance(parsed.get("content"), str):
            return
        tool_calls_schema = self._schema.get("properties", {}).get("tool_calls")
        if not isinstance(tool_calls_schema, dict):
            return
        iterator_pattern = tool_calls_schema.get("x-regex-iterator")
        if not isinstance(iterator_pattern, str) or "(.*?)" not in iterator_pattern:
            return
        prefix_pattern = iterator_pattern.split("(.*?)", 1)[0]
        literal_prefix = self._regex_literal_prefix(prefix_pattern)
        if not literal_prefix:
            return
        content = cast(str, parsed["content"])
        for prefix_length in range(len(literal_prefix) - 1, 0, -1):
            prefix = literal_prefix[:prefix_length]
            if content.endswith(prefix) and response_text.endswith(prefix):
                trimmed = content[:-prefix_length]
                parsed["content"] = trimmed if trimmed else None
                break

    def _parse_response_value(
        self,
        text: Any,
        schema: Dict[str, Any],
        *,
        partial: bool,
    ) -> Any:
        if "const" in schema:
            return schema["const"]
        if text is None:
            return None
        node_content: Any = text
        node_regex = schema.get("x-regex")
        if node_regex is not None:
            if not isinstance(node_content, str):
                raise CompletionResponseParsingError(
                    "response_schema x-regex requires string input"
                )
            captured_content = self._regex_capture(node_content, node_regex)
            if captured_content is None:
                if (
                    partial
                    and schema.get("type") == "object"
                    and "x-regex-key-value" in schema
                ):
                    captured_content = node_content
                else:
                    return None
            node_content = captured_content
        node_regex_iterator = schema.get("x-regex-iterator")
        if node_regex_iterator is not None:
            if schema.get("type") != "array":
                raise CompletionResponseParsingError(
                    "response_schema x-regex-iterator requires array type"
                )
            if not isinstance(node_content, str):
                raise CompletionResponseParsingError(
                    "response_schema x-regex-iterator requires string input"
                )
            array_values = []
            matches = list(re.finditer(node_regex_iterator, node_content, re.S))
            for match in matches:
                item_text = self._regex_capture(match.group(0), node_regex_iterator)
                if item_text is not None:
                    array_values.append(item_text)
            if partial and "(.*?)" in node_regex_iterator:
                prefix_pattern, suffix_pattern = node_regex_iterator.split("(.*?)", 1)
                prefix_matches = list(re.finditer(prefix_pattern, node_content, re.S))
                if prefix_matches:
                    last_prefix_match = prefix_matches[-1]
                    if not matches or matches[-1].start() != last_prefix_match.start():
                        tail = node_content[last_prefix_match.end() :]
                        if re.search(suffix_pattern, tail, re.S) is None:
                            array_values.append(tail)
            if not array_values:
                return None
            node_content = array_values
        node_regex_key_value = schema.get("x-regex-key-value")
        if node_regex_key_value is not None:
            if schema.get("type") != "object":
                raise CompletionResponseParsingError(
                    "response_schema x-regex-key-value requires object type"
                )
            if not isinstance(node_content, str):
                raise CompletionResponseParsingError(
                    "response_schema x-regex-key-value requires string input"
                )
            key_values: Dict[str, str] = {}
            matches = list(re.finditer(node_regex_key_value, node_content, re.S))
            for match in matches:
                group_dict = match.groupdict()
                if "key" not in group_dict or "value" not in group_dict:
                    raise CompletionResponseParsingError(
                        "response_schema x-regex-key-value must define key and value groups"
                    )
                key = group_dict["key"]
                value = group_dict["value"]
                if key is None or value is None:
                    raise CompletionResponseParsingError(
                        "response_schema x-regex-key-value matched null key or value"
                    )
                key_values[key] = value
            if partial:
                min_start = matches[-1].end() if matches else 0
                partial_item = self._partial_regex_key_value_item(
                    node_regex_key_value,
                    node_content,
                    min_start=min_start,
                )
                if partial_item is not None:
                    key_values[partial_item[0]] = partial_item[1]
            if not key_values:
                return None
            node_content = key_values
        parser_name = schema.get("x-parser")
        if parser_name is not None:
            if parser_name != "json":
                if parser_name != "gemma4-tool-call":
                    raise CompletionResponseParsingError(
                        f"unsupported response_schema x-parser: {parser_name}"
                    )
                if not isinstance(node_content, str):
                    raise CompletionResponseParsingError(
                        "response_schema x-parser='gemma4-tool-call' requires string input"
                    )
                node_content = self._gemma4_tool_call_to_json(node_content)
                parser_name = "json"
            if parser_name == "json":
                if not isinstance(node_content, str):
                    raise CompletionResponseParsingError(
                        "response_schema x-parser='json' requires string input"
                    )
                try:
                    parsed = from_json(node_content, allow_partial=partial)
                except ValueError as exc:
                    if (
                        self._has_text_tools()
                        and schema.get("type") == "object"
                        and schema.get("additionalProperties") is True
                        and not schema.get("properties")
                    ):
                        return node_content
                    if partial:
                        return None
                    raise CompletionResponseParsingError(
                        "response did not match response_schema JSON parser"
                    ) from exc
                stripped_schema = {
                    key: value
                    for key, value in schema.items()
                    if key
                    not in {
                        "x-parser",
                        "x-parser-args",
                        "x-regex",
                        "x-regex-iterator",
                        "x-regex-key-value",
                    }
                }
                return self._parse_response_value(
                    parsed, stripped_schema, partial=partial
                )
        schema_type = schema.get("type")
        if schema_type == "string":
            return node_content
        if schema_type == "array":
            if isinstance(node_content, list):
                array_values = []
                item_schema = schema.get("items", {})
                for item in node_content:
                    parsed_item = self._parse_response_value(
                        item,
                        item_schema,
                        partial=partial,
                    )
                    if parsed_item is not None:
                        array_values.append(parsed_item)
                return array_values
            return []
        if schema_type == "object":
            properties = schema.get("properties", {})
            if isinstance(node_content, dict):
                parsed_object: Dict[str, Any] = {}
                for key, value_schema in properties.items():
                    value = self._parse_response_value(
                        node_content.get(key),
                        value_schema,
                        partial=partial,
                    )
                    if value is None:
                        continue
                    if isinstance(value, list) and not value:
                        continue
                    parsed_object[key] = value
                additional_properties = schema.get("additionalProperties")
                if additional_properties is True:
                    for key, value in node_content.items():
                        if key not in parsed_object and key not in properties:
                            parsed_object[key] = value
                elif isinstance(additional_properties, dict):
                    for key, value in node_content.items():
                        if key in parsed_object or key in properties:
                            continue
                        parsed_value = self._parse_response_value(
                            value,
                            additional_properties,
                            partial=partial,
                        )
                        if parsed_value is not None:
                            parsed_object[key] = parsed_value
                if not partial:
                    missing = [
                        key
                        for key in schema.get("required", [])
                        if key not in parsed_object
                    ]
                    if missing:
                        raise CompletionResponseParsingError(
                            f"response did not match response_schema: missing {', '.join(missing)}"
                        )
                return parsed_object
            parsed_object_from_text: Dict[str, Any] = {}
            for key, value_schema in properties.items():
                value = self._parse_response_value(
                    node_content, value_schema, partial=partial
                )
                if value is None:
                    continue
                if isinstance(value, list) and not value:
                    continue
                parsed_object_from_text[key] = value
            if not partial:
                missing = [
                    key
                    for key in schema.get("required", [])
                    if key not in parsed_object_from_text
                ]
                if missing:
                    raise CompletionResponseParsingError(
                        f"response did not match response_schema: missing {', '.join(missing)}"
                    )
            return parsed_object_from_text
        if schema_type == "integer":
            if isinstance(node_content, int) and not isinstance(node_content, bool):
                return node_content
            if partial and isinstance(node_content, str) and not node_content:
                return None
            try:
                return int(node_content)
            except (TypeError, ValueError):
                return None
        if schema_type == "number":
            if isinstance(node_content, (int, float)) and not isinstance(
                node_content, bool
            ):
                return node_content
            if partial and isinstance(node_content, str) and not node_content:
                return None
            try:
                return float(node_content)
            except (TypeError, ValueError):
                return None
        if schema_type == "boolean":
            if isinstance(node_content, bool):
                return node_content
            if node_content in {"true", "True", 1, "1"}:
                return True
            if node_content in {"false", "False", 0, "0"}:
                return False
            return None
        one_of = schema.get("oneOf")
        if isinstance(one_of, list):
            for option in one_of:
                value = self._parse_response_value(
                    node_content, option, partial=partial
                )
                if value is not None:
                    return value
            return None
        if schema_type is None or schema_type == "any":
            return node_content
        return None

    def _coerce_tool_argument(
        self,
        raw_value: str,
        schema: Dict[str, Any],
        *,
        tool_name: str,
        argument_name: str,
    ) -> Any:
        if "oneOf" in schema:
            last_error: Optional[BaseException] = None
            for variant in schema["oneOf"]:
                try:
                    return self._coerce_tool_argument(
                        raw_value,
                        variant,
                        tool_name=tool_name,
                        argument_name=argument_name,
                    )
                except BaseException as exc:
                    last_error = exc
            raise CompletionResponseParsingError(
                f"tool '{tool_name}' argument '{argument_name}' did not match any allowed schema"
            ) from last_error
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            last_type_error: Optional[BaseException] = None
            for variant_type in schema_type:
                try:
                    return self._coerce_tool_argument(
                        raw_value,
                        {**schema, "type": variant_type},
                        tool_name=tool_name,
                        argument_name=argument_name,
                    )
                except BaseException as exc:
                    last_type_error = exc
            raise CompletionResponseParsingError(
                f"tool '{tool_name}' argument '{argument_name}' did not match any allowed type"
            ) from last_type_error
        if schema_type in {None, "string"}:
            return raw_value
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise CompletionResponseParsingError(
                f"tool '{tool_name}' argument '{argument_name}' is not valid JSON for type '{schema_type}'"
            ) from exc
        if schema_type == "integer":
            if isinstance(decoded, bool) or not isinstance(decoded, int):
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' argument '{argument_name}' must be an integer"
                )
            return decoded
        if schema_type == "number":
            if isinstance(decoded, bool) or not isinstance(decoded, (int, float)):
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' argument '{argument_name}' must be a number"
                )
            return decoded
        if schema_type == "boolean":
            if not isinstance(decoded, bool):
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' argument '{argument_name}' must be a boolean"
                )
            return decoded
        if schema_type == "object":
            if not isinstance(decoded, dict):
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' argument '{argument_name}' must be an object"
                )
            return decoded
        if schema_type == "array":
            if not isinstance(decoded, list):
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' argument '{argument_name}' must be an array"
                )
            return decoded
        return decoded

    def _coerce_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, str],
        *,
        partial: bool,
    ) -> Dict[str, Any]:
        if self._tool_content_type(tool_name) == "text":
            raw_input = arguments.get("input", "")
            if not isinstance(raw_input, str):
                raw_input = str(raw_input)
            return {"input": raw_input}
        if self._tools is None:
            raise CompletionResponseParsingError(
                f"response included tool call '{tool_name}' but the request declared no tools"
            )
        tool = next(
            (
                candidate
                for candidate in self._tools
                if candidate.get("type") == "function"
                and candidate.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        if tool is None:
            raise CompletionResponseParsingError(
                f"response included undeclared tool call '{tool_name}'"
            )
        parameters = tool.get("function", {}).get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        properties = parameters.get("properties", {})
        coerced: Dict[str, Any] = {}
        for argument_name, raw_value in arguments.items():
            if argument_name not in properties:
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' returned unexpected argument '{argument_name}'"
                )
            coerced[argument_name] = self._coerce_tool_argument(
                raw_value,
                properties[argument_name],
                tool_name=tool_name,
                argument_name=argument_name,
            )
        if not partial:
            missing = [
                argument_name
                for argument_name in parameters.get("required", [])
                if argument_name not in coerced
            ]
            if missing:
                raise CompletionResponseParsingError(
                    f"tool '{tool_name}' response is missing required arguments: {', '.join(missing)}"
                )
        return coerced

    def parse_chat_response(
        self,
        response_text: str,
        *,
        partial: bool,
    ) -> Dict[str, Any]:
        full_response_text = self._generation_prompt + response_text
        parsed = self._parse_response_value(
            full_response_text,
            self._schema,
            partial=partial,
        )
        if not isinstance(parsed, dict):
            raise CompletionResponseParsingError("response_schema must produce an object")
        if partial:
            self._trim_partial_tool_call_prefix(
                response_text=full_response_text,
                parsed=parsed,
            )
        tool_calls = parsed.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_tool_calls: List[Dict[str, Any]] = []
            for tool_call in tool_calls:
                normalized = self._normalize_tool_call_item(tool_call, partial=partial)
                if normalized is None:
                    continue
                normalized_tool_calls.append(normalized)
            if normalized_tool_calls:
                parsed["tool_calls"] = normalized_tool_calls
            else:
                parsed.pop("tool_calls", None)
        for field in ("reasoning_content", "thinking"):
            value = parsed.get(field)
            if isinstance(value, str) and not value.strip():
                parsed.pop(field, None)
        return parsed

    def _normalize_tool_call_item(
        self,
        tool_call: Any,
        *,
        partial: bool,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_call, dict):
            if partial:
                return None
            raise CompletionResponseParsingError("tool_calls items must be objects")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            if partial:
                return None
            raise CompletionResponseParsingError(
                "tool_calls items must define a function object"
            )
        tool_name = function.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            if partial:
                return None
            raise CompletionResponseParsingError(
                "tool_calls function name must be a non-empty string"
            )
        if self._tool_content_type(tool_name) == "text":
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                if partial:
                    return None
                raise CompletionResponseParsingError(
                    "tool_calls function arguments must be a string for text tools"
                )
            return {
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        arguments = function.get("arguments", {})
        if not isinstance(arguments, (dict, ResponseParser.PartialJsonObject)):
            if partial:
                return None
            raise CompletionResponseParsingError(
                "tool_calls function arguments must parse to an object"
            )
        argument_values = (
            arguments.value
            if isinstance(arguments, ResponseParser.PartialJsonObject)
            else arguments
        )
        raw_arguments: Dict[str, str] = {}
        for argument_name, argument_value in argument_values.items():
            if isinstance(argument_value, ResponseParser.PartialJsonValue):
                raw_arguments[argument_name] = argument_value.text
            else:
                raw_arguments[argument_name] = str(argument_value)
        normalized_arguments = (
            arguments
            if isinstance(arguments, ResponseParser.PartialJsonObject)
            or any(
                isinstance(value, ResponseParser.PartialJsonValue)
                for value in argument_values.values()
            )
            else self._coerce_tool_arguments(
                tool_name,
                raw_arguments,
                partial=partial,
            )
        )
        return {
            "type": tool_call.get("type", "function"),
            "function": {
                "name": tool_name,
                "arguments": normalized_arguments,
            },
        }

    @classmethod
    def _serialize_partial_json_prefix(cls, value: Any) -> str:
        if isinstance(value, ResponseParser.PartialJsonValue):
            if value.complete and value.schema_type in {None, "string"}:
                return json.dumps(value.text, ensure_ascii=False, separators=(",", ":"))
            if value.schema_type in {None, "string"}:
                return json.dumps(value.text, ensure_ascii=False, separators=(",", ":"))[:-1]
            return value.text
        if isinstance(value, ResponseParser.PartialJsonObject):
            return cls._serialize_partial_json_prefix(value.value)
        if isinstance(value, dict):
            items = list(value.items())
            if not items:
                return "{"
            rendered = ["{"]
            last_index = len(items) - 1
            for index, (key, item_value) in enumerate(items):
                if index > 0:
                    rendered.append(",")
                rendered.append(json.dumps(key, ensure_ascii=False, separators=(",", ":")))
                rendered.append(":")
                if index == last_index:
                    rendered.append(cls._serialize_partial_json_prefix(item_value))
                else:
                    rendered.append(
                        json.dumps(item_value, ensure_ascii=False, separators=(",", ":"))
                    )
            return "".join(rendered)
        if isinstance(value, list):
            if not value:
                return "["
            rendered = ["["]
            last_index = len(value) - 1
            for index, item_value in enumerate(value):
                if index > 0:
                    rendered.append(",")
                if index == last_index:
                    rendered.append(cls._serialize_partial_json_prefix(item_value))
                else:
                    rendered.append(
                        json.dumps(item_value, ensure_ascii=False, separators=(",", ":"))
                    )
            return "".join(rendered)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:-1]
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _contains_partial_json_value(cls, value: Any) -> bool:
        if isinstance(value, ResponseParser.PartialJsonValue):
            return True
        if isinstance(value, ResponseParser.PartialJsonObject):
            return True
        if isinstance(value, dict):
            return any(cls._contains_partial_json_value(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_partial_json_value(item) for item in value)
        return False

    @classmethod
    def _serialize_partial_json_state(cls, value: Any) -> Tuple[str, bool]:
        if isinstance(value, ResponseParser.PartialJsonValue):
            if value.schema_type in {None, "string"}:
                rendered = json.dumps(value.text, ensure_ascii=False, separators=(",", ":"))
                if value.complete:
                    return rendered, True
                return rendered[:-1], False
            return value.text, value.complete
        if isinstance(value, ResponseParser.PartialJsonObject):
            rendered, children_complete = cls._serialize_partial_json_state(value.value)
            if value.complete and children_complete:
                return rendered, True
            if rendered.endswith("}"):
                return rendered[:-1], False
            return rendered, False
        if isinstance(value, dict):
            parts = ["{"]
            items = list(value.items())
            for index, (key, item_value) in enumerate(items):
                if index > 0:
                    parts.append(",")
                parts.append(json.dumps(key, ensure_ascii=False, separators=(",", ":")))
                parts.append(":")
                rendered_item, item_complete = cls._serialize_partial_json_state(item_value)
                parts.append(rendered_item)
                if index == len(items) - 1 and not item_complete:
                    return "".join(parts), False
            parts.append("}")
            return "".join(parts), True
        if isinstance(value, list):
            parts = ["["]
            for index, item_value in enumerate(value):
                if index > 0:
                    parts.append(",")
                rendered_item, item_complete = cls._serialize_partial_json_state(item_value)
                parts.append(rendered_item)
                if index == len(value) - 1 and not item_complete:
                    return "".join(parts), False
            parts.append("]")
            return "".join(parts), True
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")), True

    @classmethod
    def _serialize_tool_arguments(cls, arguments: Any, *, partial: bool = False) -> str:
        if partial:
            if cls._contains_partial_json_value(arguments):
                return cls._serialize_partial_json_state(arguments)[0]
            return cls._serialize_partial_json_prefix(arguments)
        if isinstance(arguments, ResponseParser.PartialJsonObject):
            arguments = arguments.value
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

    def _parsed_chat_message(
        self,
        *,
        parsed: Dict[str, Any],
        partial: bool = False,
    ) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": parsed.get("role", "assistant"),
        }
        for key, value in parsed.items():
            if key in {"role", "content", "tool_calls"}:
                continue
            if value is None or value == "":
                continue
            message[key] = value
        content = parsed.get("content")
        tool_calls = parsed.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls = []
            for tool_call_index, tool_call in enumerate(tool_calls):
                function = tool_call["function"]
                if self._tool_content_type(function["name"]) == "text":
                    raw_arguments = function["arguments"]
                    arguments = (
                        raw_arguments
                        if isinstance(raw_arguments, str)
                        else str(raw_arguments)
                    )
                else:
                    arguments = self._serialize_tool_arguments(
                        function["arguments"],
                        partial=partial,
                    )
                normalized_tool_calls.append(
                    {
                        "id": f"call_{self._choice_index}_{function['name']}_{self._completion_id}_{tool_call_index}",
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": function["name"],
                            "arguments": arguments,
                        },
                    }
                )
            message["content"] = content if content not in {None, ""} else None
            message["tool_calls"] = normalized_tool_calls
            if len(normalized_tool_calls) == 1:
                message["function_call"] = dict(normalized_tool_calls[0]["function"])
            return message
        message["content"] = content if content is not None else ""
        return message

    def _message_deltas(
        self,
        previous_message: Dict[str, Any],
        message: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        deltas: List[Dict[str, Any]] = []
        for key, value in message.items():
            if key in {"role", "content", "tool_calls", "function_call"}:
                continue
            old_value = previous_message.get(key, "")
            if not isinstance(value, str):
                if key not in previous_message and value is not None:
                    deltas.append({key: value})
                continue
            if not value:
                continue
            if isinstance(old_value, str) and value.startswith(old_value):
                delta_value = value[len(old_value) :]
            else:
                delta_value = value
            if delta_value:
                deltas.append({key: delta_value})

        new_content = message.get("content")
        old_content = previous_message.get("content", "")
        if isinstance(new_content, str) and new_content:
            if isinstance(old_content, str) and new_content.startswith(old_content):
                content_delta = new_content[len(old_content) :]
            else:
                content_delta = new_content
            if content_delta:
                deltas.append({"content": content_delta})

        new_tool_calls = cast(List[Dict[str, Any]], message.get("tool_calls", []))
        old_tool_calls = cast(List[Dict[str, Any]], previous_message.get("tool_calls", []))
        for tool_call_index, tool_call in enumerate(new_tool_calls):
            old_tool_call = old_tool_calls[tool_call_index] if tool_call_index < len(old_tool_calls) else None
            delta_tool_call: Dict[str, Any] = {"index": tool_call_index}
            if old_tool_call is None:
                delta_tool_call["id"] = tool_call["id"]
                delta_tool_call["type"] = tool_call["type"]
            function = cast(Dict[str, Any], tool_call["function"])
            old_function = (
                cast(Dict[str, Any], old_tool_call["function"])
                if old_tool_call is not None
                else {}
            )
            function_delta: Dict[str, Any] = {}
            if function.get("name") and function.get("name") != old_function.get("name"):
                function_delta["name"] = function["name"]
            arguments = cast(str, function.get("arguments", ""))
            old_arguments = cast(str, old_function.get("arguments", ""))
            if old_tool_call is None and arguments == "{}":
                argument_delta = ""
            elif arguments.startswith(old_arguments):
                argument_delta = arguments[len(old_arguments) :]
            else:
                argument_delta = arguments
            if argument_delta:
                function_delta["arguments"] = argument_delta
            if function_delta:
                delta_tool_call["function"] = function_delta
            if len(delta_tool_call) > 1:
                deltas.append({"tool_calls": [delta_tool_call]})
        return deltas

    @staticmethod
    def _apply_message_delta(message: Dict[str, Any], delta: Dict[str, Any]) -> None:
        if "role" in delta:
            message["role"] = delta["role"]
        for key, value in delta.items():
            if key in {"role", "tool_calls", "function_call"}:
                continue
            if isinstance(value, str):
                existing = message.get(key)
                if isinstance(existing, str):
                    message[key] = existing + value
                else:
                    message[key] = value
            else:
                message[key] = value
        tool_call_deltas = delta.get("tool_calls")
        if not isinstance(tool_call_deltas, list):
            return
        tool_calls = cast(List[Dict[str, Any]], message.setdefault("tool_calls", []))
        for tool_delta in tool_call_deltas:
            if not isinstance(tool_delta, dict):
                continue
            index = tool_delta.get("index")
            if not isinstance(index, int):
                continue
            while len(tool_calls) <= index:
                tool_calls.append({"function": {"name": "", "arguments": ""}})
            tool_call = tool_calls[index]
            if "id" in tool_delta:
                tool_call["id"] = tool_delta["id"]
            if "type" in tool_delta:
                tool_call["type"] = tool_delta["type"]
            function_delta = tool_delta.get("function")
            if not isinstance(function_delta, dict):
                continue
            function = cast(Dict[str, Any], tool_call.setdefault("function", {}))
            name_delta = function_delta.get("name")
            if isinstance(name_delta, str):
                function["name"] = cast(str, function.get("name", "")) + name_delta
            arguments_delta = function_delta.get("arguments")
            if isinstance(arguments_delta, str):
                function["arguments"] = cast(str, function.get("arguments", "")) + arguments_delta
        if tool_calls:
            message["function_call"] = dict(cast(Dict[str, Any], tool_calls[0]["function"]))

    def parse_completion_message(self, response_text: str) -> Dict[str, Any]:
        parsed = self.parse_chat_response(response_text, partial=False)
        return self._parsed_chat_message(parsed=parsed)

    def _stream_state_message(self, *, partial: bool) -> Dict[str, Any]:
        assert self._stream_state is not None
        if self._stream_plan is not None and self._stream_plan.get("direct_deltas"):
            copied = {
                key: (
                    [
                        {
                            child_key: (
                                dict(child_value)
                                if isinstance(child_value, dict)
                                else child_value
                            )
                            for child_key, child_value in tool_call.items()
                        }
                        for tool_call in cast(List[Dict[str, Any]], value)
                    ]
                    if key == "tool_calls" and isinstance(value, list)
                    else value
                )
                for key, value in self._message.items()
            }
            if copied.get("tool_calls"):
                copied["function_call"] = dict(
                    cast(List[Dict[str, Any]], copied["tool_calls"])[0]["function"]
                )
            return copied
        parsed = cast(Dict[str, Any], self._stream_state.parsed)
        return self._parsed_chat_message(parsed=parsed, partial=partial)

    def _stream_state_complete(self) -> bool:
        if self._stream_plan is None:
            return False
        if self._direct.deltas:
            return (
                not self._direct.pending
                and self._direct.mode in {self.DIRECT_MODE_CONTENT, self.DIRECT_MODE_AFTER_TOOL_ITEM}
            )
        if self._stream_state is None:
            return False
        if self._stream_plan["kind"] == "json-root":
            return True
        if self._stream_plan["kind"] == "segment-message":
            return (
                not self._stream_state.pending
                and self._stream_state.current_item is None
                and self._stream_state.current_segment is None
                and self._stream_state.mode == "segment-start"
            )
        return (
            not self._stream_state.pending
            and self._stream_state.current_item is None
            and self._stream_state.mode in {"content", "after-tool-item"}
        )

    @staticmethod
    def _chat_chunk_payload(
        *,
        chunk_id: str,
        created: int,
        model: str,
        index: int,
        delta: Dict[str, Any],
        finish_reason: Optional[str],
        logprobs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        choice: Dict[str, Any] = {
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {
            "id": "chat" + chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        }

    def _chunk_payloads(
        self,
        *,
        chunk_id: str,
        created: int,
        model: str,
        deltas: List[Dict[str, Any]],
        finish_reason: Optional[str],
        logprobs: Optional[Dict[str, Any]],
        leading_delta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        logprobs_sent = False
        chat_chunk_id = "chat" + chunk_id
        index = self._choice_index
        if leading_delta is not None:
            payloads.append(
                {
                    "id": chat_chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": leading_delta,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        for delta in deltas:
            payload_logprobs = None
            if (
                logprobs is not None
                and delta.get("role") != "assistant"
                and not logprobs_sent
            ):
                payload_logprobs = logprobs
                logprobs_sent = True
            choice: Dict[str, Any] = {
                "index": index,
                "delta": delta,
                "finish_reason": None,
            }
            if payload_logprobs is not None:
                choice["logprobs"] = payload_logprobs
            payloads.append(
                {
                    "id": chat_chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [choice],
                }
            )
        if finish_reason is not None:
            payloads.append(
                {
                    "id": chat_chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )
        return payloads

    def consume_completion_chunk(
        self,
        text: str,
        *,
        chunk_id: str,
        created: int,
        model: str,
        finish_reason: Optional[str],
        logprobs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        role_delta: Optional[Dict[str, Any]] = None
        if not self._started:
            self._started = True
            self._message["role"] = "assistant"
            role_delta = {"role": "assistant"}
        self._text_parts.append(text)

        if self._stream_plan is not None and not self._stream_failed:
            success, stream_deltas = self._advance_stream_state(text)
            if success:
                if self._direct.deltas:
                    if finish_reason is None:
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=stream_deltas,
                            finish_reason=None,
                            logprobs=logprobs,
                            leading_delta=role_delta,
                        )
                    if self._stream_state_complete():
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=stream_deltas,
                            finish_reason=(
                                "tool_calls" if self._direct.saw_tool_calls else finish_reason
                            ),
                            logprobs=logprobs,
                            leading_delta=role_delta,
                        )
                    self._stream_failed = True
                elif self._stream_plan["kind"] == "segment-message":
                    if role_delta is not None:
                        stream_deltas = [role_delta, *stream_deltas]
                    for delta in stream_deltas:
                        self._apply_message_delta(self._message, delta)
                    if finish_reason is None:
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=stream_deltas,
                            finish_reason=None,
                            logprobs=logprobs,
                        )
                    if self._stream_state_complete():
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=stream_deltas,
                            finish_reason=(
                                "tool_calls" if self._message.get("tool_calls") else finish_reason
                            ),
                            logprobs=logprobs,
                        )
                    self._stream_failed = True
                else:
                    previous_message = self._message
                    partial_deltas: List[Dict[str, Any]] = []
                    assert self._stream_state is not None
                    parsed = cast(Dict[str, Any], self._stream_state.parsed)
                    message = self._parsed_chat_message(
                        parsed=parsed,
                        partial=finish_reason is None,
                    )
                    if finish_reason is None:
                        if role_delta is not None:
                            partial_deltas.append(role_delta)
                        partial_deltas.extend(self._message_deltas(previous_message, message))
                        self._message = message
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=partial_deltas,
                            finish_reason=None,
                            logprobs=logprobs,
                        )
                    if self._stream_state_complete():
                        if role_delta is not None:
                            partial_deltas.append(role_delta)
                        partial_deltas.extend(self._message_deltas(previous_message, message))
                        self._message = message
                        return self._chunk_payloads(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            deltas=partial_deltas,
                            finish_reason=(
                                "tool_calls" if message.get("tool_calls") else finish_reason
                            ),
                            logprobs=logprobs,
                        )
                    self._stream_failed = True
            else:
                self._stream_failed = True

        response_text = "".join(self._text_parts)
        parsed = self.parse_chat_response(response_text, partial=finish_reason is None)
        message = self._parsed_chat_message(parsed=parsed, partial=finish_reason is None)
        previous_message = self._message
        deltas: List[Dict[str, Any]] = []
        if role_delta is not None:
            deltas.append(role_delta)
        deltas.extend(self._message_deltas(previous_message, message))
        self._message = message
        return self._chunk_payloads(
            chunk_id=chunk_id,
            created=created,
            model=model,
            deltas=deltas,
            finish_reason=(
                "tool_calls"
                if finish_reason is not None and message.get("tool_calls")
                else finish_reason
            ),
            logprobs=logprobs,
        )


class OpenAIFormatter:
    @dataclass
    class ReturnedToken:
        index: int
        text_bytes: bytes
        token: Token
        text_offset: int

    @dataclass
    class ResponsesOutputItem:
        output_index: int
        item: Dict[str, Any]
        content_index: Optional[int] = None

    @dataclass
    class ResponsesStream:
        body: CreateResponseRequest
        response_id: str
        created_at: float
        model: str
        output: List[Dict[str, Any]] = field(default_factory=list)
        sequence_number: int = 0
        started: bool = False
        assistant_meta: Dict[str, Any] = field(default_factory=dict)
        reasoning_item: Optional["OpenAIFormatter.ResponsesOutputItem"] = None
        message_item: Optional["OpenAIFormatter.ResponsesOutputItem"] = None
        tool_items: Dict[int, "OpenAIFormatter.ResponsesOutputItem"] = field(
            default_factory=dict
        )
        final_status: Optional[str] = None
        incomplete_details: Optional[Dict[str, Any]] = None

    def __init__(self, model: Model) -> None:
        self.model = model

    @staticmethod
    def decode_text(data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def encode_sse_payload(payload: BaseModel | Dict[str, Any]) -> bytes:
        data = (
            payload.model_dump(mode="json", exclude_none=True)
            if isinstance(payload, BaseModel)
            else payload
        )
        return (
            "data: "
            f"{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
        ).encode("utf-8")

    @staticmethod
    def next_stream_chunk(
        stream: Iterator[CompletionChunk],
    ) -> Tuple[bool, Optional[CompletionChunk]]:
        try:
            return False, next(stream)
        except StopIteration:
            return True, None

    @staticmethod
    def next_stream_output(
        stream: Iterator[CompletionChunk],
    ) -> Tuple[bool, Optional[CompletionChunk], Optional[OpenAICompletion]]:
        try:
            return False, next(stream), None
        except StopIteration as stop:
            return True, None, cast(Optional[OpenAICompletion], stop.value)

    @staticmethod
    def collect_completion(stream: Iterator[OpenAICompletion]) -> OpenAICompletion:
        iterator = iter(stream)
        while True:
            try:
                next(iterator)
            except StopIteration as stop:
                result = stop.value
                if result is None:
                    raise RuntimeError("missing completion result")
                return cast(OpenAICompletion, result)

    @staticmethod
    def _normalized_tools(
        *,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if functions is not None:
            return [{"type": "function", "function": function} for function in functions]
        return tools

    def _chat_template_text(self) -> str:
        if self.model.chat_formatter is None:
            return ""
        return cast(str, self.model.chat_formatter._template_text)

    def _uses_harmony_channels(self) -> bool:
        template = self._chat_template_text()
        return "<|channel|>" in template or "<|recipient|>" in template

    def _uses_reasoning_content(self) -> bool:
        template = self._chat_template_text()
        return "reasoning_content" in template or "<think>" in template

    @staticmethod
    def _chat_message(data: Dict[str, Any]) -> ChatCompletionRequestMessage:
        return ChatCompletionRequestMessage.model_validate(data)

    def _instructions_role(self) -> Literal["developer", "system"]:
        return "developer" if "developer" in self._chat_template_text() else "system"

    @staticmethod
    def _response_reasoning_effort(
        body: CreateResponseRequest,
    ) -> Optional[str]:
        reasoning = body.reasoning
        if not isinstance(reasoning, dict):
            return None
        effort = reasoning.get("effort")
        if effort in {"low", "medium", "high"}:
            return cast(str, effort)
        return None

    @staticmethod
    def _response_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise CompletionRequestValidationError("responses input content must be a string or list")
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise CompletionRequestValidationError("responses content parts must be objects")
            part_type = part.get("type")
            if part_type in {"input_text", "output_text", "reasoning_text", "summary_text", "text"}:
                text = part.get("text")
                if not isinstance(text, str):
                    raise CompletionRequestValidationError(
                        f"responses content part {part_type!r} requires string text"
                    )
                parts.append(text)
                continue
            raise CompletionRequestValidationError(
                f"unsupported responses content part type: {part_type!r}"
            )
        return "".join(parts)

    @staticmethod
    def _response_reasoning_text(item: Dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, list) and content:
            return OpenAIFormatter._response_text_from_content(content)
        summary = item.get("summary")
        if isinstance(summary, list) and summary:
            return OpenAIFormatter._response_text_from_content(summary)
        return ""

    def _response_reasoning_input_message(
        self,
        *,
        text: str,
    ) -> ChatCompletionRequestMessage:
        if self._uses_harmony_channels():
            return self._chat_message(
                {
                    "role": "assistant",
                    "content": text,
                    "channel": "analysis",
                }
            )
        if self._uses_reasoning_content():
            return self._chat_message(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": text,
                }
            )
        return self._chat_message(
            {
                "role": "assistant",
                "content": text,
            }
        )

    @staticmethod
    def _response_tool_call_input_message(
        *,
        name: str,
        arguments: str,
        call_id: str,
        content_type: Optional[str] = None,
    ) -> ChatCompletionRequestMessage:
        function: Dict[str, Any] = {
            "name": name,
            "arguments": arguments,
        }
        if isinstance(content_type, str) and content_type:
            function["content_type"] = content_type
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": function,
        }
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
            "function_call": dict(function),
        }
        return OpenAIFormatter._chat_message(message)

    def _response_function_call_input_message(
        self,
        item: Dict[str, Any],
    ) -> ChatCompletionRequestMessage:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise CompletionRequestValidationError("function_call input requires name")
        arguments = item.get("arguments", "")
        if not isinstance(arguments, str):
            raise CompletionRequestValidationError("function_call input requires string arguments")
        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
        return self._response_tool_call_input_message(
            name=name,
            arguments=arguments,
            call_id=call_id,
            content_type="json",
        )

    def _response_custom_tool_call_input_message(
        self,
        item: Dict[str, Any],
    ) -> ChatCompletionRequestMessage:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise CompletionRequestValidationError("custom_tool_call input requires name")
        input_text = item.get("input", "")
        if not isinstance(input_text, str):
            raise CompletionRequestValidationError("custom_tool_call input requires string input")
        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
        content_type = item.get("content_type")
        if not isinstance(content_type, str) or not content_type:
            content_type = "text"
        return self._response_tool_call_input_message(
            name=name,
            arguments=input_text,
            call_id=call_id,
            content_type=content_type,
        )

    def responses_input_to_chat_messages(
        self,
        body: CreateResponseRequest,
    ) -> List[ChatCompletionRequestMessage]:
        if body.conversation is not None:
            raise CompletionRequestValidationError(
                "conversation is not supported in stateless /v1/responses"
            )
        if body.store:
            raise CompletionRequestValidationError(
                "store=true is not supported in stateless /v1/responses"
            )
        if body.truncation not in {None, "disabled"}:
            raise CompletionRequestValidationError(
                "only truncation='disabled' is supported"
            )

        messages: List[ChatCompletionRequestMessage] = []
        if body.instructions is not None:
            messages.append(
                self._chat_message(
                    {
                        "role": self._instructions_role(),
                        "content": body.instructions,
                    }
                )
            )

        if isinstance(body.input, str):
            messages.append(
                self._chat_message(
                    {
                        "role": "user",
                        "content": body.input,
                    }
                )
            )
            return messages

        items = body.input
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise CompletionRequestValidationError(
                "responses input must be a string, object, or list"
            )

        leading_preamble: List[str] = []
        item_start_index = 0
        while item_start_index < len(items):
            item = items[item_start_index]
            if not isinstance(item, dict):
                break
            item_type = item.get("type")
            if item_type is None and "role" in item:
                item_type = "message"
            if item_type != "message":
                break
            role = item.get("role", "user")
            if role not in {"developer", "system"}:
                break
            leading_preamble.append(
                self._response_text_from_content(item.get("content", ""))
            )
            item_start_index += 1

        if leading_preamble:
            preamble_parts: List[str] = []
            if messages:
                first_content = messages[0].content
                if isinstance(first_content, str) and first_content:
                    preamble_parts.append(first_content)
                messages = []
            preamble_parts.extend(text for text in leading_preamble if text)
            if preamble_parts:
                messages.append(
                    self._chat_message(
                        {
                            "role": self._instructions_role(),
                            "content": "\n\n".join(preamble_parts),
                        }
                    )
                )

        function_names_by_call_id: Dict[str, str] = {}
        for item in items[item_start_index:]:
            if not isinstance(item, dict):
                raise CompletionRequestValidationError(
                    "responses input items must be objects"
                )
            item_type = item.get("type")
            if item_type is None and "role" in item:
                item_type = "message"
            if item_type == "message":
                role = item.get("role", "user")
                if role not in {
                    "user",
                    "assistant",
                    "system",
                    "developer",
                    "tool",
                    "function",
                }:
                    raise CompletionRequestValidationError(
                        f"unsupported responses message role: {role!r}"
                    )
                if role == "function":
                    role = "tool"
                data: Dict[str, Any] = {
                    "role": role,
                    "content": self._response_text_from_content(
                        item.get("content", "")
                    ),
                }
                phase = item.get("phase")
                if isinstance(phase, str):
                    data["phase"] = phase
                    if role == "assistant" and self._uses_harmony_channels():
                        if phase == "commentary":
                            data["channel"] = "commentary"
                        elif phase == "final_answer":
                            data["channel"] = "final"
                messages.append(self._chat_message(data))
                continue
            if item_type == "reasoning":
                text = self._response_reasoning_text(item)
                if text:
                    messages.append(self._response_reasoning_input_message(text=text))
                continue
            if item_type == "function_call":
                message = self._response_function_call_input_message(item)
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    function_names_by_call_id[call_id] = name
                messages.append(message)
                continue
            if item_type == "custom_tool_call":
                message = self._response_custom_tool_call_input_message(item)
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    function_names_by_call_id[call_id] = name
                messages.append(message)
                continue
            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise CompletionRequestValidationError(
                        "function_call_output input requires call_id"
                    )
                tool_output_data: Dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": self._response_text_from_content(item.get("output", "")),
                }
                name = function_names_by_call_id.get(call_id)
                if name is not None:
                    tool_output_data["name"] = name
                messages.append(self._chat_message(tool_output_data))
                continue
            if item_type == "custom_tool_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise CompletionRequestValidationError(
                        "custom_tool_call_output input requires call_id"
                    )
                tool_output_data = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": self._response_text_from_content(item.get("output", "")),
                }
                name = function_names_by_call_id.get(call_id)
                if name is not None:
                    tool_output_data["name"] = name
                messages.append(self._chat_message(tool_output_data))
                continue
            raise CompletionRequestValidationError(
                f"unsupported responses input item type: {item_type!r}"
            )
        return messages

    @staticmethod
    def _clone_response_input_items(input_items: Any) -> List[Any]:
        if isinstance(input_items, list):
            return copy.deepcopy(input_items)
        if isinstance(input_items, dict):
            return [copy.deepcopy(input_items)]
        return [copy.deepcopy(input_items)]

    @staticmethod
    def _normalize_response_output_item_for_input(
        item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role")
            if not isinstance(role, str):
                return None
            normalized: Dict[str, Any] = {
                "type": "message",
                "role": role,
                "content": copy.deepcopy(item.get("content", [])),
            }
            for key in ("id", "phase", "status"):
                value = item.get(key)
                if value is not None:
                    normalized[key] = copy.deepcopy(value)
            return normalized
        if item_type == "reasoning":
            normalized = {
                "type": "reasoning",
                "content": copy.deepcopy(item.get("content", [])),
            }
            for key in ("id", "summary", "status"):
                value = item.get(key)
                if value is not None:
                    normalized[key] = copy.deepcopy(value)
            return normalized
        if item_type == "function_call":
            normalized = {
                "type": "function_call",
                "call_id": copy.deepcopy(item.get("call_id")),
                "name": copy.deepcopy(item.get("name")),
                "arguments": copy.deepcopy(item.get("arguments", "")),
            }
            namespace = item.get("namespace")
            item_id = item.get("id")
            if item_id is not None:
                normalized["id"] = copy.deepcopy(item_id)
            if namespace is not None:
                normalized["namespace"] = copy.deepcopy(namespace)
            return normalized
        if item_type == "custom_tool_call":
            normalized = {
                "type": "custom_tool_call",
                "call_id": copy.deepcopy(item.get("call_id")),
                "name": copy.deepcopy(item.get("name")),
                "input": copy.deepcopy(item.get("input", "")),
            }
            namespace = item.get("namespace")
            content_type = item.get("content_type")
            item_id = item.get("id")
            if item_id is not None:
                normalized["id"] = copy.deepcopy(item_id)
            if namespace is not None:
                normalized["namespace"] = copy.deepcopy(namespace)
            if content_type is not None:
                normalized["content_type"] = copy.deepcopy(content_type)
            return normalized
        return None

    def _responses_tools_to_chat_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        if tools is None:
            return None
        normalized_tools: List[Dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "web_search":
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise CompletionRequestValidationError("responses tools require name")
            if tool_type == "function":
                normalized_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description"),
                            "parameters": tool.get("parameters"),
                            "strict": tool.get("strict"),
                        },
                    }
                )
                continue
            if tool_type == "custom":
                tool_format = tool.get("format")
                syntax = (
                    tool_format.get("syntax") if isinstance(tool_format, dict) else None
                )
                definition = (
                    tool_format.get("definition")
                    if isinstance(tool_format, dict)
                    else None
                )
                description = cast(Optional[str], tool.get("description")) or ""
                if isinstance(syntax, str) and isinstance(definition, str):
                    if description:
                        description = f"{description}\n\n{syntax}:\n{definition}"
                    else:
                        description = f"{syntax}:\n{definition}"
                normalized_tools.append(
                    {
                        "type": "function",
                        "original_type": "custom",
                        "function": {
                            "name": name,
                            "description": description or None,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True,
                            },
                            "strict": tool.get("strict"),
                            "content_type": "text",
                        },
                    }
                )
                continue
            raise CompletionRequestValidationError(
                f"unsupported responses tool type: {tool_type!r}"
            )
        return normalized_tools

    @staticmethod
    def _responses_tool_choice_to_chat_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if not isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice.get("type") in {"function", "custom"} and "name" in tool_choice:
            return {
                "type": "function",
                "function": {
                    "name": tool_choice["name"],
                },
            }
        return tool_choice

    @staticmethod
    def _response_format_type(response_format: Optional[Dict[str, Any]]) -> Optional[str]:
        if response_format is None:
            return None
        format_type = response_format.get("type")
        if isinstance(format_type, str):
            return format_type
        return None

    @staticmethod
    def _grammar_for_response_format(
        response_format: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        format_type = OpenAIFormatter._response_format_type(response_format)
        if format_type is None or format_type == "text":
            return None
        if format_type == "json_object":
            return JSON_GBNF
        if format_type == "json_schema":
            assert response_format is not None
            schema = response_format.get("schema")
            if schema is None and isinstance(response_format.get("json_schema"), dict):
                schema = cast(Dict[str, Any], response_format["json_schema"]).get("schema")
            if not isinstance(schema, dict):
                raise CompletionRequestValidationError(
                    "json_schema response format requires a schema object"
                )
            return JsonSchemaConverter.to_gbnf(
                json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        raise CompletionRequestValidationError(
            f"unsupported response format type: {format_type!r}"
        )

    def chat_request_from_responses_request(
        self,
        body: CreateResponseRequest,
    ) -> CreateChatCompletionRequest:
        chat_tools = self._responses_tools_to_chat_tools(body.tools)
        return CreateChatCompletionRequest(
            messages=self.responses_input_to_chat_messages(body),
            max_tokens=body.max_output_tokens,
            temperature=0.8 if body.temperature is None else body.temperature,
            top_p=0.95 if body.top_p is None else body.top_p,
            stream=body.stream,
            logprobs=body.top_logprobs is not None,
            top_logprobs=body.top_logprobs,
            model=body.model,
            user=body.user,
            tools=chat_tools,
            tool_choice=self._responses_tool_choice_to_chat_tool_choice(body.tool_choice),
            response_format=(
                None
                if body.text is None
                else body.text.get("format")
                if isinstance(body.text.get("format"), dict)
                else None
            ),
            reasoning_effort=self._response_reasoning_effort(body),
        )

    def _response_parser(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        completion_id: str = "",
        choice_index: int = 0,
        generation_prompt: str = "",
    ) -> ResponseParser:
        if self.model.response_schema is None:
            raise CompletionResponseParsingError("model does not define response_schema")
        return ResponseParser(
            self.model.response_schema,
            tools=tools,
            completion_id=completion_id,
            choice_index=choice_index,
            generation_prompt=generation_prompt,
        )

    def parse_chat_response(
        self,
        response_text: str,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        partial: bool,
        generation_prompt: str = "",
    ) -> Dict[str, Any]:
        return self._response_parser(
            tools=tools,
            generation_prompt=generation_prompt,
        ).parse_chat_response(
            response_text,
            partial=partial,
        )

    def _chat_tool_name_and_grammar(
        self,
        body: CreateChatCompletionRequest,
    ) -> Tuple[Optional[str], Optional[str]]:
        tools = self._normalized_tools(functions=body.functions, tools=body.tools)
        tool_choice: Optional[Union[str, Dict[str, Any]]] = body.tool_choice
        if body.function_call is not None:
            if isinstance(body.function_call, str) and body.function_call in {"none", "auto"}:
                tool_choice = body.function_call
            elif isinstance(body.function_call, dict) and "name" in body.function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": body.function_call["name"],
                    },
                }
        grammar_text = self._grammar_for_response_format(body.response_format)
        if not isinstance(tool_choice, dict):
            return None, grammar_text
        if tools is None:
            raise CompletionRequestValidationError("tool choice requires tools")
        tool_name = tool_choice["function"]["name"]
        tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
        if tool is None:
            raise CompletionRequestValidationError(
                f"Tool choice '{tool_name}' not found in tools."
            )
        if self.model.response_schema is None:
            return tool_name, JSON_GBNF
        return tool_name, grammar_text

    def completion_request_from_chat_request(
        self,
        body: CreateChatCompletionRequest,
    ) -> Tuple[
        CreateCompletionRequest,
        str,
        str,
        List[int],
        Optional[str],
        Optional[str],
    ]:
        try:
            prompt_text, generation_prompt, prompt_tokens, formatter_stop = self.model.build_chat_prompt(
                body.messages,
                functions=body.functions,
                function_call=body.function_call,
                tools=body.tools,
                tool_choice=body.tool_choice,
                reasoning_effort=body.reasoning_effort,
            )
            tool_name, grammar_text = self._chat_tool_name_and_grammar(body)
        except ValueError as exc:
            raise CompletionRequestValidationError(str(exc)) from exc
        request_stop: List[str] = []
        if body.stop is None:
            request_stop = []
        elif isinstance(body.stop, str):
            request_stop = [body.stop]
        else:
            request_stop = list(body.stop)
        stop_sequences = [stop for stop in [*request_stop, *formatter_stop] if stop]
        deduped_stop: List[str] = []
        seen_stop: set[str] = set()
        for stop in stop_sequences:
            if stop not in seen_stop:
                deduped_stop.append(stop)
                seen_stop.add(stop)
        payload = CreateCompletionRequest(
            prompt=prompt_text,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            echo=False,
            stop=deduped_stop or None,
            stream=body.stream,
            logprobs=(
                0 if body.logprobs and body.top_logprobs is None else body.top_logprobs
            ),
            presence_penalty=body.presence_penalty,
            frequency_penalty=body.frequency_penalty,
            logit_bias=body.logit_bias,
            seed=body.seed,
            model=body.model,
            n=body.n,
            user=body.user,
        )
        return payload, prompt_text, generation_prompt, prompt_tokens, grammar_text, tool_name

    @staticmethod
    def _response_phase_from_message(message: Dict[str, Any]) -> Optional[str]:
        phase = message.get("phase")
        if phase in {"commentary", "final_answer"}:
            return cast(str, phase)
        channel = message.get("channel")
        if channel == "commentary":
            return "commentary"
        if channel == "final":
            return "final_answer"
        return None

    @staticmethod
    def _response_reasoning_text_from_message(message: Dict[str, Any]) -> str:
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            return thinking
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            return reasoning_content
        if message.get("channel") == "analysis":
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _response_output_text_from_message(message: Dict[str, Any]) -> str:
        if message.get("channel") == "analysis":
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _response_logprobs_from_chat_logprobs(
        logprobs: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if logprobs is None:
            return []
        content = logprobs.get("content")
        if not isinstance(content, list):
            return []
        response_logprobs: List[Dict[str, Any]] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            token = entry.get("token")
            logprob = entry.get("logprob")
            if not isinstance(token, str) or not isinstance(logprob, (int, float)):
                continue
            top_logprobs = entry.get("top_logprobs")
            converted_top: List[Dict[str, Any]] = []
            if isinstance(top_logprobs, list):
                for top in top_logprobs:
                    if not isinstance(top, dict):
                        continue
                    converted_top.append(
                        {
                            "token": top.get("token"),
                            "logprob": top.get("logprob"),
                        }
                    )
            response_logprobs.append(
                {
                    "token": token,
                    "logprob": float(logprob),
                    "top_logprobs": converted_top or None,
                }
            )
        return response_logprobs

    def _response_logprobs_from_completion(
        self,
        logprobs: Optional[CompletionLogprobs],
    ) -> List[Dict[str, Any]]:
        if logprobs is None:
            return []
        response_logprobs: List[Dict[str, Any]] = []
        for token, token_logprob, top_logprobs in zip(
            logprobs.tokens,
            logprobs.token_logprobs,
            logprobs.top_logprobs,
        ):
            if token_logprob is None:
                continue
            converted_top: List[Dict[str, Any]] = []
            if top_logprobs is not None:
                for top_token, top_logprob in top_logprobs.items():
                    converted_top.append(
                        {
                            "token": top_token,
                            "logprob": float(top_logprob),
                        }
                    )
            response_logprobs.append(
                {
                    "token": token,
                    "logprob": float(token_logprob),
                    "top_logprobs": converted_top or None,
                }
            )
        return response_logprobs

    @staticmethod
    def _response_reasoning_item(
        *,
        item_id: str,
        text: str,
        status: str,
    ) -> Dict[str, Any]:
        return {
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": text,
                }
            ],
            "status": status,
        }

    @staticmethod
    def _response_message_item(
        *,
        item_id: str,
        text: str,
        status: str,
        phase: Optional[str],
        logprobs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        content_item: Dict[str, Any] = {
            "type": "output_text",
            "text": text,
            "annotations": [],
        }
        if logprobs:
            content_item["logprobs"] = logprobs
        item: Dict[str, Any] = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [content_item],
            "status": status,
        }
        if phase is not None:
            item["phase"] = phase
        return item

    @staticmethod
    def _response_function_call_item(
        *,
        item_id: str,
        call_id: str,
        name: str,
        arguments: str,
        status: str,
    ) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": status,
        }
        if "." in name:
            namespace, bare_name = name.split(".", 1)
            item["namespace"] = namespace
            item["name"] = bare_name
        return item

    @staticmethod
    def _response_custom_tool_call_item(
        *,
        item_id: str,
        call_id: str,
        name: str,
        input_text: str,
    ) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": item_id,
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name,
            "input": input_text,
        }
        if "." in name:
            namespace, bare_name = name.split(".", 1)
            item["namespace"] = namespace
            item["name"] = bare_name
        return item

    @staticmethod
    def _responses_tool_type_by_name(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        if tools is None:
            return {}
        tool_types: Dict[str, str] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("original_type") or tool.get("type")
            if not isinstance(tool_type, str):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            else:
                name = tool.get("name")
            if isinstance(name, str) and name:
                tool_types[name] = tool_type
        return tool_types

    def _response_output_items_from_message(
        self,
        *,
        response_id: str,
        message: Dict[str, Any],
        logprobs: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        tool_types_by_name = self._responses_tool_type_by_name(tools)
        reasoning_text = self._response_reasoning_text_from_message(message)
        if reasoning_text:
            items.append(
                self._response_reasoning_item(
                    item_id=f"rs_{response_id}_0",
                    text=reasoning_text,
                    status="completed",
                )
            )
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, str):
                    continue
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"call_{response_id}_{tool_call_index}"
                tool_type = tool_types_by_name.get(name, "function")
                if tool_type == "custom":
                    items.append(
                        self._response_custom_tool_call_item(
                            item_id=f"fc_{response_id}_{tool_call_index}",
                            call_id=call_id,
                            name=name,
                            input_text=arguments,
                        )
                    )
                    continue
                items.append(
                    self._response_function_call_item(
                        item_id=f"fc_{response_id}_{tool_call_index}",
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        status="completed",
                    )
                )
        output_text = self._response_output_text_from_message(message)
        if output_text:
            items.append(
                self._response_message_item(
                    item_id=f"msg_{response_id}_0",
                    text=output_text,
                    status="completed",
                    phase=self._response_phase_from_message(message),
                    logprobs=logprobs,
                )
            )
        return items

    def _responses_usage_from_completion(
        self,
        *,
        usage: Optional[CompletionUsage],
        output_items: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if usage is None:
            return None
        reasoning_tokens = 0
        for item in output_items:
            if item.get("type") != "reasoning":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "reasoning_text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        reasoning_tokens += len(
                            self.model.tokenize(text, add_bos=False, special=True)
                        )
        return {
            "input_tokens": usage.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": 0,
            },
            "output_tokens": usage.completion_tokens,
            "output_tokens_details": {
                "reasoning_tokens": min(reasoning_tokens, usage.completion_tokens),
            },
            "total_tokens": usage.total_tokens,
        }

    def _response_status_and_incomplete_details(
        self,
        *,
        finish_reason: Optional[str],
    ) -> Tuple[str, Optional[Dict[str, str]]]:
        if finish_reason == "length":
            return "incomplete", {"reason": "max_output_tokens"}
        return "completed", None

    def _response_object(
        self,
        *,
        body: CreateResponseRequest,
        response_id: str,
        created_at: float,
        model: str,
        output_items: Sequence[Dict[str, Any]],
        usage: Optional[Dict[str, Any]],
        status: str,
        incomplete_details: Optional[Dict[str, str]],
        completed_at: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "completed_at": completed_at,
            "error": None,
            "incomplete_details": incomplete_details,
            "instructions": body.instructions,
            "metadata": body.metadata,
            "model": model,
            "output": list(output_items),
            "parallel_tool_calls": body.parallel_tool_calls,
            "reasoning": {
                "effort": self._response_reasoning_effort(body),
                "summary": None,
            },
            "store": False,
            "temperature": body.temperature,
            "tool_choice": body.tool_choice or "auto",
            "tools": body.tools or [],
            "top_p": body.top_p,
            "max_output_tokens": body.max_output_tokens,
            "previous_response_id": None,
            "status": status,
            "text": body.text or {"format": {"type": "text"}},
            "top_logprobs": body.top_logprobs,
            "truncation": body.truncation,
            "usage": usage,
            "user": body.user,
        }

    def _response_from_chat_message(
        self,
        *,
        body: CreateResponseRequest,
        response_id: str,
        created_at: float,
        model: str,
        message: Dict[str, Any],
        finish_reason: Optional[str],
        usage: Optional[CompletionUsage],
        logprobs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        status, incomplete_details = self._response_status_and_incomplete_details(
            finish_reason=finish_reason,
        )
        output_items = self._response_output_items_from_message(
            response_id=response_id,
            message=message,
            logprobs=logprobs,
            tools=cast(Optional[List[Dict[str, Any]]], body.tools),
        )
        return self._response_object(
            body=body,
            response_id=response_id,
            created_at=created_at,
            model=model,
            output_items=output_items,
            usage=self._responses_usage_from_completion(
                usage=usage,
                output_items=output_items,
            ),
            status=status,
            incomplete_details=incomplete_details,
            completed_at=time.time() if status in {"completed", "incomplete"} else None,
        )

    def convert_completion_response_to_response(
        self,
        completion: OpenAICompletion,
        body: CreateResponseRequest,
        tool_name: Optional[str] = None,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_prompt: str = "",
    ) -> Dict[str, Any]:
        chat_response = self.convert_completion_response_to_chat(
            completion,
            tool_name,
            tools=tools,
            generation_prompt=generation_prompt,
        )
        if isinstance(chat_response, BaseModel):
            chat_payload = chat_response.model_dump(mode="json", exclude_none=True)
        else:
            chat_payload = chat_response
        choice = cast(Dict[str, Any], chat_payload["choices"][0])
        message = cast(Dict[str, Any], choice["message"])
        response_id = "resp_" + completion.id
        logprobs = self._response_logprobs_from_completion(completion.choices[0].logprobs)
        return self._response_from_chat_message(
            body=body,
            response_id=response_id,
            created_at=float(completion.created),
            model=completion.model,
            message=message,
            finish_reason=cast(Optional[str], choice.get("finish_reason")),
            usage=completion.usage,
            logprobs=logprobs or None,
        )

    @staticmethod
    def _response_event(
        state: "OpenAIFormatter.ResponsesStream",
        event_type: str,
        **payload: Any,
    ) -> Dict[str, Any]:
        state.sequence_number += 1
        event = {
            "type": event_type,
            "sequence_number": state.sequence_number,
        }
        event.update(payload)
        return event

    def _response_stream_response(
        self,
        state: "OpenAIFormatter.ResponsesStream",
        *,
        status: str,
        usage: Optional[Dict[str, Any]],
        incomplete_details: Optional[Dict[str, str]],
        completed_at: Optional[float],
    ) -> Dict[str, Any]:
        return self._response_object(
            body=state.body,
            response_id=state.response_id,
            created_at=state.created_at,
            model=state.model,
            output_items=state.output,
            usage=usage,
            status=status,
            incomplete_details=incomplete_details,
            completed_at=completed_at,
        )

    def start_response_stream(
        self,
        state: "OpenAIFormatter.ResponsesStream",
    ) -> List[Dict[str, Any]]:
        if state.started:
            return []
        state.started = True
        response = self._response_stream_response(
            state,
            status="in_progress",
            usage=None,
            incomplete_details=None,
            completed_at=None,
        )
        return [
            self._response_event(state, "response.created", response=response),
            self._response_event(state, "response.in_progress", response=response),
        ]

    @staticmethod
    def _response_item_status(
        finish_reason: Optional[str],
    ) -> str:
        if finish_reason == "length":
            return "incomplete"
        return "completed"

    def _add_response_stream_item(
        self,
        state: "OpenAIFormatter.ResponsesStream",
        item: Dict[str, Any],
        *,
        content_index: Optional[int] = None,
    ) -> "OpenAIFormatter.ResponsesOutputItem":
        item_state = OpenAIFormatter.ResponsesOutputItem(
            output_index=len(state.output),
            item=item,
            content_index=content_index,
        )
        state.output.append(item)
        return item_state

    def _ensure_reasoning_stream_item(
        self,
        state: "OpenAIFormatter.ResponsesStream",
    ) -> Tuple[List[Dict[str, Any]], "OpenAIFormatter.ResponsesOutputItem"]:
        if state.reasoning_item is not None:
            return [], state.reasoning_item
        item = self._response_reasoning_item(
            item_id=f"rs_{state.response_id}_{len(state.output)}",
            text="",
            status="in_progress",
        )
        item_state = self._add_response_stream_item(state, item, content_index=0)
        state.reasoning_item = item_state
        part = cast(List[Dict[str, Any]], item["content"])[0]
        return [
            self._response_event(
                state,
                "response.output_item.added",
                output_index=item_state.output_index,
                item=item,
            ),
            self._response_event(
                state,
                "response.content_part.added",
                item_id=cast(str, item["id"]),
                output_index=item_state.output_index,
                content_index=0,
                part=part,
            ),
        ], item_state

    def _ensure_message_stream_item(
        self,
        state: "OpenAIFormatter.ResponsesStream",
    ) -> Tuple[List[Dict[str, Any]], "OpenAIFormatter.ResponsesOutputItem"]:
        if state.message_item is not None:
            return [], state.message_item
        item = self._response_message_item(
            item_id=f"msg_{state.response_id}_{len(state.output)}",
            text="",
            status="in_progress",
            phase=cast(Optional[str], state.assistant_meta.get("phase")),
        )
        item_state = self._add_response_stream_item(state, item, content_index=0)
        state.message_item = item_state
        part = cast(List[Dict[str, Any]], item["content"])[0]
        return [
            self._response_event(
                state,
                "response.output_item.added",
                output_index=item_state.output_index,
                item=item,
            ),
            self._response_event(
                state,
                "response.content_part.added",
                item_id=cast(str, item["id"]),
                output_index=item_state.output_index,
                content_index=0,
                part=part,
            ),
        ], item_state

    def _ensure_tool_stream_item(
        self,
        state: "OpenAIFormatter.ResponsesStream",
        *,
        tool_call_index: int,
        call_id: Optional[str],
        name: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], "OpenAIFormatter.ResponsesOutputItem"]:
        existing = state.tool_items.get(tool_call_index)
        if existing is not None:
            return [], existing
        tool_types_by_name = self._responses_tool_type_by_name(state.body.tools)
        tool_type = tool_types_by_name.get(name or "", "function")
        item_id = f"fc_{state.response_id}_{tool_call_index}"
        resolved_call_id = call_id or f"call_{state.response_id}_{tool_call_index}"
        resolved_name = name or ""
        if tool_type == "custom":
            item = self._response_custom_tool_call_item(
                item_id=item_id,
                call_id=resolved_call_id,
                name=resolved_name,
                input_text="",
            )
        else:
            item = self._response_function_call_item(
                item_id=item_id,
                call_id=resolved_call_id,
                name=resolved_name,
                arguments="",
                status="in_progress",
            )
        item_state = self._add_response_stream_item(state, item)
        state.tool_items[tool_call_index] = item_state
        return [
            self._response_event(
                state,
                "response.output_item.added",
                output_index=item_state.output_index,
                item=item,
            )
        ], item_state

    @staticmethod
    def _update_tool_stream_item(
        item: Dict[str, Any],
        *,
        call_id: Optional[str],
        name_delta: Optional[str],
        arguments_delta: Optional[str],
    ) -> None:
        if isinstance(call_id, str) and call_id:
            item["call_id"] = call_id
        if isinstance(name_delta, str) and name_delta:
            current_name = cast(str, item.get("name", ""))
            if not current_name:
                item["name"] = name_delta
            elif name_delta == current_name or name_delta.startswith(current_name):
                item["name"] = name_delta
            elif not current_name.endswith(name_delta):
                item["name"] = current_name + name_delta
        if isinstance(arguments_delta, str) and arguments_delta:
            if item.get("type") == "custom_tool_call":
                item["input"] = cast(str, item.get("input", "")) + arguments_delta
            else:
                item["arguments"] = (
                    cast(str, item.get("arguments", "")) + arguments_delta
                )

    def _finalize_response_stream_items(
        self,
        state: "OpenAIFormatter.ResponsesStream",
        *,
        finish_reason: Optional[str],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        item_status = self._response_item_status(finish_reason)

        if (
            state.reasoning_item is not None
            and state.reasoning_item.item["status"] == "in_progress"
        ):
            item = state.reasoning_item.item
            item["status"] = item_status
            part = cast(List[Dict[str, Any]], item["content"])[0]
            events.append(
                self._response_event(
                    state,
                    "response.reasoning_text.done",
                    item_id=cast(str, item["id"]),
                    output_index=state.reasoning_item.output_index,
                    content_index=cast(int, state.reasoning_item.content_index),
                    text=part["text"],
                )
            )
            events.append(
                self._response_event(
                    state,
                    "response.content_part.done",
                    item_id=cast(str, item["id"]),
                    output_index=state.reasoning_item.output_index,
                    content_index=cast(int, state.reasoning_item.content_index),
                    part=part,
                )
            )
            events.append(
                self._response_event(
                    state,
                    "response.output_item.done",
                    output_index=state.reasoning_item.output_index,
                    item=item,
                )
            )

        if (
            state.message_item is not None
            and state.message_item.item["status"] == "in_progress"
        ):
            item = state.message_item.item
            item["status"] = item_status
            part = cast(List[Dict[str, Any]], item["content"])[0]
            events.append(
                self._response_event(
                    state,
                    "response.output_text.done",
                    item_id=cast(str, item["id"]),
                    output_index=state.message_item.output_index,
                    content_index=cast(int, state.message_item.content_index),
                    text=part["text"],
                    logprobs=part.get("logprobs", []),
                )
            )
            events.append(
                self._response_event(
                    state,
                    "response.content_part.done",
                    item_id=cast(str, item["id"]),
                    output_index=state.message_item.output_index,
                    content_index=cast(int, state.message_item.content_index),
                    part=part,
                )
            )
            events.append(
                self._response_event(
                    state,
                    "response.output_item.done",
                    output_index=state.message_item.output_index,
                    item=item,
                )
            )

        for tool_call_index in sorted(state.tool_items):
            item_state = state.tool_items[tool_call_index]
            item = item_state.item
            if (
                item.get("status") != "in_progress"
                and item.get("type") != "custom_tool_call"
            ):
                continue
            if item.get("type") == "custom_tool_call":
                events.append(
                    self._response_event(
                        state,
                        "response.custom_tool_call_input.done",
                        item_id=cast(str, item["id"]),
                        output_index=item_state.output_index,
                        input=cast(str, item.get("input", "")),
                    )
                )
            else:
                item["status"] = item_status
                events.append(
                    self._response_event(
                        state,
                        "response.function_call_arguments.done",
                        item_id=cast(str, item["id"]),
                        output_index=item_state.output_index,
                        name=cast(str, item["name"]),
                        arguments=cast(str, item["arguments"]),
                    )
                )
            events.append(
                self._response_event(
                    state,
                    "response.output_item.done",
                    output_index=item_state.output_index,
                    item=item,
                )
            )

        state.final_status, state.incomplete_details = (
            self._response_status_and_incomplete_details(
                finish_reason=finish_reason,
            )
        )
        return events

    def convert_chat_chunk_to_response_events(
        self,
        chunk: ChatCompletionChunk | Dict[str, Any],
        state: "OpenAIFormatter.ResponsesStream",
    ) -> List[Dict[str, Any]]:
        payload = (
            chunk.model_dump(mode="json", exclude_none=True)
            if isinstance(chunk, BaseModel)
            else chunk
        )
        events = self.start_response_stream(state)
        choice = cast(Dict[str, Any], payload["choices"][0])
        delta = cast(Dict[str, Any], choice.get("delta", {}))
        finish_reason = cast(Optional[str], choice.get("finish_reason"))
        logprobs = self._response_logprobs_from_chat_logprobs(
            cast(Optional[Dict[str, Any]], choice.get("logprobs"))
        )

        phase = delta.get("phase")
        if isinstance(phase, str):
            state.assistant_meta["phase"] = phase
            if state.message_item is not None:
                state.message_item.item["phase"] = phase

        reasoning_delta = delta.get("reasoning_content")
        if not isinstance(reasoning_delta, str) or not reasoning_delta:
            candidate = delta.get("thinking")
            reasoning_delta = candidate if isinstance(candidate, str) else None
        if isinstance(reasoning_delta, str) and reasoning_delta:
            added, item_state = self._ensure_reasoning_stream_item(state)
            events.extend(added)
            part = cast(List[Dict[str, Any]], item_state.item["content"])[0]
            part["text"] += reasoning_delta
            events.append(
                self._response_event(
                    state,
                    "response.reasoning_text.delta",
                    item_id=cast(str, item_state.item["id"]),
                    output_index=item_state.output_index,
                    content_index=cast(int, item_state.content_index),
                    delta=reasoning_delta,
                )
            )

        content_delta = delta.get("content")
        if isinstance(content_delta, str) and content_delta:
            added, item_state = self._ensure_message_stream_item(state)
            events.extend(added)
            part = cast(List[Dict[str, Any]], item_state.item["content"])[0]
            part["text"] += content_delta
            if logprobs:
                cast(List[Dict[str, Any]], part.setdefault("logprobs", [])).extend(
                    logprobs
                )
            events.append(
                self._response_event(
                    state,
                    "response.output_text.delta",
                    item_id=cast(str, item_state.item["id"]),
                    output_index=item_state.output_index,
                    content_index=cast(int, item_state.content_index),
                    delta=content_delta,
                    logprobs=logprobs,
                )
            )

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_index = tool_call.get("index", 0)
                if not isinstance(tool_call_index, int):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                added, item_state = self._ensure_tool_stream_item(
                    state,
                    tool_call_index=tool_call_index,
                    call_id=cast(Optional[str], tool_call.get("id")),
                    name=cast(Optional[str], function.get("name")),
                )
                events.extend(added)
                self._update_tool_stream_item(
                    item_state.item,
                    call_id=cast(Optional[str], tool_call.get("id")),
                    name_delta=cast(Optional[str], function.get("name")),
                    arguments_delta=cast(Optional[str], function.get("arguments")),
                )
                arguments_delta = function.get("arguments")
                if isinstance(arguments_delta, str) and arguments_delta:
                    if item_state.item.get("type") == "custom_tool_call":
                        events.append(
                            self._response_event(
                                state,
                                "response.custom_tool_call_input.delta",
                                item_id=cast(str, item_state.item["id"]),
                                output_index=item_state.output_index,
                                delta=arguments_delta,
                            )
                        )
                        continue
                    events.append(
                        self._response_event(
                            state,
                            "response.function_call_arguments.delta",
                            item_id=cast(str, item_state.item["id"]),
                            output_index=item_state.output_index,
                            delta=arguments_delta,
                        )
                    )

        if finish_reason is not None:
            events.extend(
                self._finalize_response_stream_items(
                    state,
                    finish_reason=finish_reason,
                )
            )
        return events

    def response_stream_terminal_events(
        self,
        state: "OpenAIFormatter.ResponsesStream",
        completion: Optional[OpenAICompletion],
    ) -> List[Dict[str, Any]]:
        if not state.started:
            state.started = True
        if completion is not None and state.final_status is None:
            finish_reason = None
            if completion.choices:
                finish_reason = completion.choices[0].finish_reason
            self._finalize_response_stream_items(state, finish_reason=finish_reason)
        status = state.final_status or "completed"
        response = self._response_stream_response(
            state,
            status=status,
            usage=(
                self._responses_usage_from_completion(
                    usage=completion.usage if completion is not None else None,
                    output_items=state.output,
                )
            ),
            incomplete_details=state.incomplete_details,
            completed_at=time.time() if status in {"completed", "incomplete"} else None,
        )
        event_type = "response.incomplete" if status == "incomplete" else "response.completed"
        return [self._response_event(state, event_type, response=response)]

    def aggregate_completion_results(
        self,
        results: Sequence[OpenAICompletion],
    ) -> OpenAICompletion:
        choices: List[CompletionChoice] = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for result in results:
            for choice in result.choices:
                choices.append(choice.model_copy(update={"index": len(choices)}))
            assert result.usage is not None
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens
            total_tokens += result.usage.total_tokens
        return OpenAICompletion(
            id=f"cmpl-{uuid.uuid4().hex}",
            object="text_completion",
            created=int(time.time()),
            model=self.model.model_path,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    @staticmethod
    def _convert_completion_logprobs_to_chat_choice(
        logprobs: Optional[CompletionLogprobs],
    ) -> Optional[ChatCompletionChoiceLogprobs]:
        if logprobs is None:
            return None
        return ChatCompletionChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=None,
                    logprob=token_logprob if token_logprob is not None else 0.0,
                    top_logprobs=(
                        [
                            TopLogprob(
                                token=top_token,
                                logprob=top_logprob,
                                bytes=None,
                            )
                            for top_token, top_logprob in top_logprobs.items()
                        ]
                        if top_logprobs is not None
                        else []
                    ),
                )
                for token, token_logprob, top_logprobs in zip(
                    logprobs.tokens,
                    logprobs.token_logprobs,
                    logprobs.top_logprobs,
                )
            ],
            refusal=None,
        )

    @staticmethod
    def _convert_completion_logprobs_to_chat_chunk(
        logprobs: Optional[CompletionLogprobs],
    ) -> Optional[ChatCompletionChunkChoiceLogprobs]:
        choice_logprobs = OpenAIFormatter._convert_completion_logprobs_to_chat_choice(logprobs)
        if choice_logprobs is None:
            return None
        return ChatCompletionChunkChoiceLogprobs.model_validate(
            choice_logprobs.model_dump(mode="python", exclude_none=True)
        )

    @staticmethod
    def _chat_message_from_completion_choice(
        completion_id: str,
        choice: CompletionChoice,
        tool_name: Optional[str],
    ) -> ChatCompletionMessage:
        if tool_name is None:
            return ChatCompletionMessage(
                role="assistant",
                content=choice.text,
            )
        tool_id = f"call_{choice.index}_{tool_name}_{completion_id}"
        arguments = choice.text
        return ChatCompletionMessage(
            role="assistant",
            content=None,
            function_call=ChatCompletionMessageFunctionCall(
                name=tool_name,
                arguments=arguments,
            ),
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id=tool_id,
                    type="function",
                    function=ChatCompletionMessageToolCallFunction(
                        name=tool_name,
                        arguments=arguments,
                    ),
                )
            ],
        )

    @staticmethod
    def _chat_chunk_payload(
        *,
        chunk_id: str,
        created: int,
        model: str,
        index: int,
        delta: Dict[str, Any],
        finish_reason: Optional[str],
        logprobs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        choice: Dict[str, Any] = {
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {
            "id": "chat" + chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        }

    def convert_completion_response_to_chat(
        self,
        completion: OpenAICompletion,
        tool_name: Optional[str] = None,
        *,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_prompt: str = "",
    ) -> ChatCompletion | Dict[str, Any]:
        normalized_tools = self._normalized_tools(functions=functions, tools=tools)
        if self.model.response_schema is not None:
            choices: List[Dict[str, Any]] = []
            for choice in completion.choices:
                parser = self._response_parser(
                    tools=normalized_tools,
                    completion_id=completion.id,
                    choice_index=choice.index,
                    generation_prompt=generation_prompt,
                )
                message = parser.parse_completion_message(choice.text)
                logprobs = self._convert_completion_logprobs_to_chat_choice(choice.logprobs)
                choices.append(
                    {
                        "index": choice.index,
                        "message": message,
                        "logprobs": (
                            logprobs.model_dump(mode="json", exclude_none=True)
                            if logprobs is not None
                            else None
                        ),
                        "finish_reason": (
                            "tool_calls"
                            if message.get("tool_calls")
                            else choice.finish_reason
                        ),
                    }
                )
            return {
                "id": "chat" + completion.id,
                "object": "chat.completion",
                "created": completion.created,
                "model": completion.model,
                "choices": choices,
                "usage": (
                    completion.usage.model_dump(mode="json", exclude_none=True)
                    if completion.usage is not None
                    else None
                ),
            }
        return ChatCompletion(
            id="chat" + completion.id,
            object="chat.completion",
            created=completion.created,
            model=completion.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=self._chat_message_from_completion_choice(
                        completion.id,
                        choice,
                        tool_name,
                    ),
                    logprobs=self._convert_completion_logprobs_to_chat_choice(choice.logprobs),
                    finish_reason=(
                        "tool_calls" if tool_name is not None else choice.finish_reason
                    ),
                )
                for choice in completion.choices
            ],
            usage=completion.usage,
        )

    def convert_completion_chunk_to_chat_chunks(
        self,
        chunk: CompletionChunk,
        started_indices: set[int],
        tool_name: Optional[str] = None,
        *,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        parsed_states: Optional[Dict[int, Any]] = None,
        generation_prompt: str = "",
    ) -> List[ChatCompletionChunk | Dict[str, Any]]:
        normalized_tools = self._normalized_tools(functions=functions, tools=tools)
        if self.model.response_schema is not None:
            parsed_chunks: List[Dict[str, Any]] = []
            if parsed_states is None:
                parsed_states = {}
            for choice in chunk["choices"]:
                index = choice["index"]
                parser = parsed_states.get(index)
                if not isinstance(parser, ResponseParser):
                    parser = self._response_parser(
                        tools=normalized_tools,
                        completion_id=chunk["id"],
                        choice_index=index,
                        generation_prompt=generation_prompt,
                    )
                    parsed_states[index] = parser
                logprobs = self._convert_completion_logprobs_to_chat_chunk(
                    CompletionLogprobs.model_validate(choice["logprobs"])
                    if choice["logprobs"] is not None
                    else None
                )
                parsed_chunks.extend(
                    parser.consume_completion_chunk(
                        choice["text"],
                        chunk_id=chunk["id"],
                        created=chunk["created"],
                        model=chunk["model"],
                        finish_reason=choice["finish_reason"],
                        logprobs=(
                            logprobs.model_dump(mode="json", exclude_none=True)
                            if logprobs is not None
                            else None
                        ),
                    )
                )
                if parser.started:
                    started_indices.add(index)
            return parsed_chunks
        chat_chunks: List[ChatCompletionChunk] = []
        for choice in chunk["choices"]:
            index = choice["index"]
            if index not in started_indices:
                started_indices.add(index)
                chat_chunks.append(
                    ChatCompletionChunk(
                        id="chat" + chunk["id"],
                        object="chat.completion.chunk",
                        created=chunk["created"],
                        model=chunk["model"],
                        choices=[
                            ChatCompletionChunkChoice(
                                index=index,
                                delta=(
                                    ChoiceDelta(
                                        role="assistant",
                                        content=None,
                                        function_call=None,
                                        tool_calls=None,
                                    )
                                    if tool_name is not None
                                    else ChoiceDelta(role="assistant")
                                ),
                                logprobs=None,
                                finish_reason=None,
                            )
                        ],
                    )
                )
            if tool_name is not None:
                delta: ChoiceDelta
                if choice["finish_reason"] is None:
                    tool_id = f"call_{index}_{tool_name}_{chunk['id']}"
                    delta = ChoiceDelta(
                        role=None,
                        content=None,
                        function_call=ChoiceDeltaFunctionCall(
                            name=tool_name,
                            arguments=choice["text"],
                        ),
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=tool_id,
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=tool_name,
                                    arguments=choice["text"],
                                ),
                            )
                        ],
                    )
                else:
                    delta = ChoiceDelta(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    )
                chat_chunks.append(
                    ChatCompletionChunk(
                        id="chat" + chunk["id"],
                        object="chat.completion.chunk",
                        created=chunk["created"],
                        model=chunk["model"],
                        choices=[
                            ChatCompletionChunkChoice(
                                index=index,
                                delta=delta,
                                logprobs=self._convert_completion_logprobs_to_chat_chunk(
                                    CompletionLogprobs.model_validate(choice["logprobs"])
                                    if choice["logprobs"] is not None
                                    else None
                                ),
                                finish_reason=(
                                    "tool_calls"
                                    if choice["finish_reason"] is not None
                                    else None
                                ),
                            )
                        ],
                    )
                )
                continue
            chat_chunks.append(
                ChatCompletionChunk(
                    id="chat" + chunk["id"],
                    object="chat.completion.chunk",
                    created=chunk["created"],
                    model=chunk["model"],
                    choices=[
                        ChatCompletionChunkChoice(
                            index=index,
                            delta=(
                                ChoiceDelta(content=choice["text"])
                                if choice["finish_reason"] is None
                                else ChoiceDelta()
                            ),
                            logprobs=self._convert_completion_logprobs_to_chat_chunk(
                                CompletionLogprobs.model_validate(choice["logprobs"])
                                if choice["logprobs"] is not None
                                else None
                            ),
                            finish_reason=choice["finish_reason"],
                        )
                    ],
                )
            )
        return chat_chunks

    def returned_output_end(
        self,
        completion: Completion,
        finish_reason: Optional[str],
    ) -> int:
        completion_bytes: bytes | bytearray
        if completion.rendered_bytes:
            completion_bytes = completion.rendered_bytes
        else:
            completion_bytes = b"".join(record.text_bytes for record in completion.token_records)
        returned_end = len(completion_bytes)
        if finish_reason == "stop":
            stops = [stop for stop in completion.stop_sequences if stop in completion_bytes]
            if stops:
                returned_end = min(completion_bytes.index(stop) for stop in stops)
        elif finish_reason is None:
            holdback = 0
            for stop in completion.stop_sequences:
                for size in range(min(len(stop), returned_end), 0, -1):
                    if completion_bytes.endswith(stop[:size]):
                        holdback = max(holdback, size)
                        break
            returned_end -= holdback
        return returned_end

    def returned_tokens(
        self,
        completion: Completion,
        finish_reason: Optional[str],
        *,
        start_index: int = 0,
    ) -> List["OpenAIFormatter.ReturnedToken"]:
        returned_end = self.returned_output_end(completion, finish_reason)
        returned_tokens: List[OpenAIFormatter.ReturnedToken] = []
        prefix_bytes = b""
        for index, record in enumerate(completion.token_records):
            token_start = len(prefix_bytes)
            if token_start >= returned_end:
                break
            token_end = token_start + len(record.text_bytes)
            text_bytes = record.text_bytes
            if token_end > returned_end:
                text_bytes = text_bytes[: returned_end - token_start]
            if index >= start_index:
                returned_tokens.append(
                    OpenAIFormatter.ReturnedToken(
                        index=index,
                        text_bytes=text_bytes,
                        token=record,
                        text_offset=len(self.decode_text(prefix_bytes)),
                    )
                )
            prefix_bytes += record.text_bytes
        return returned_tokens

    def stream_completion_chunks(
        self,
        request: CompletionRequest,
        completion: Completion,
        finish_reason: Optional[str],
    ) -> List[CompletionChunk]:
        returned_tokens = self.returned_tokens(
            completion,
            finish_reason,
            start_index=completion.returned_token_count,
        )
        chunks: List[CompletionChunk] = []
        if completion.logprobs is not None:
            for returned_token in returned_tokens:
                token = returned_token.token
                chunks.append(
                    {
                        "id": request.id,
                        "object": "text_completion",
                        "created": request.created,
                        "model": self.model.model_path,
                        "choices": [
                            {
                                "text": self.decode_text(returned_token.text_bytes),
                                "index": completion.index,
                                "logprobs": {
                                    "tokens": [self.decode_text(token.text_bytes)],
                                    "text_offset": [
                                        len(completion.prompt_text) + returned_token.text_offset
                                    ],
                                    "token_logprobs": [token.token_logprob],
                                    "top_logprobs": [token.top_logprobs],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                completion.returned_token_count = returned_token.index + 1
            return chunks

        chunk_tokens: List[OpenAIFormatter.ReturnedToken] = []
        for returned_token in returned_tokens:
            chunk_tokens.append(returned_token)
            chunk_bytes = b"".join(token.text_bytes for token in chunk_tokens)
            if returned_token.text_bytes != returned_token.token.text_bytes:
                chunks.append(
                    {
                        "id": request.id,
                        "object": "text_completion",
                        "created": request.created,
                        "model": self.model.model_path,
                        "choices": [
                            {
                                "text": self.decode_text(returned_token.text_bytes),
                                "index": completion.index,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                completion.returned_token_count = returned_token.index + 1
                chunk_tokens = []
                continue
            try:
                text = chunk_bytes.decode("utf-8")
            except UnicodeError:
                continue
            chunks.append(
                {
                    "id": request.id,
                    "object": "text_completion",
                    "created": request.created,
                    "model": self.model.model_path,
                    "choices": [
                        {
                            "text": text,
                            "index": completion.index,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
            completion.returned_token_count = returned_token.index + 1
            chunk_tokens = []
        return chunks

    def completion_finish_chunk(
        self,
        request: CompletionRequest,
        completion: Completion,
        finish_reason: str,
    ) -> CompletionChunk:
        return {
            "id": request.id,
            "object": "text_completion",
            "created": request.created,
            "model": self.model.model_path,
            "choices": [
                {
                    "text": "",
                    "index": completion.index,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _build_completion_choice(
        self,
        request: CompletionRequest,
        completion: Completion,
    ) -> CompletionChoice:
        returned_tokens = self.returned_tokens(completion, completion.finish_reason)
        text_bytes = b"".join(returned_token.text_bytes for returned_token in returned_tokens)
        text = self.decode_text(text_bytes)
        if request.payload.echo:
            text = completion.prompt_text + text
        logprobs: Optional[CompletionLogprobs] = None
        if completion.logprobs is not None:
            offsets: List[int] = []
            token_texts: List[str] = []
            token_logprobs: List[Optional[float]] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []
            text_cursor = request.prompt_text if not request.payload.echo else ""
            if request.payload.echo:
                for record in request.prompt_records:
                    offsets.append(len(text_cursor))
                    token_texts.append(self.decode_text(record.text_bytes))
                    token_logprobs.append(record.token_logprob)
                    top_logprobs.append(record.top_logprobs)
                    text_cursor += self.decode_text(record.text_bytes)
                text_cursor = request.prompt_text
            for returned_token in returned_tokens:
                token = returned_token.token
                offsets.append(len(completion.prompt_text) + returned_token.text_offset)
                if request.payload.echo:
                    offsets[-1] = len(text_cursor)
                token_texts.append(self.decode_text(token.text_bytes))
                token_logprobs.append(token.token_logprob)
                top_logprobs.append(token.top_logprobs)
                text_cursor += self.decode_text(token.text_bytes)
            logprobs = CompletionLogprobs(
                text_offset=offsets,
                token_logprobs=token_logprobs,
                tokens=token_texts,
                top_logprobs=top_logprobs,
            )
        return CompletionChoice(
            text=text,
            index=completion.index,
            logprobs=logprobs,
            finish_reason=completion.finish_reason,
        )

    def build_completion_response(
        self,
        request: CompletionRequest,
        completions: Sequence[Completion],
    ) -> OpenAICompletion:
        completion_tokens = sum(completion.completion_token_count for completion in completions)
        return OpenAICompletion(
            id=request.id,
            object="text_completion",
            created=request.created,
            model=self.model.model_path,
            choices=[
                self._build_completion_choice(request, completion)
                for completion in completions
            ],
            usage=CompletionUsage(
                prompt_tokens=len(request.prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=len(request.prompt_tokens) + completion_tokens,
            ),
        )


class Sampler:
    TOKEN_DATA_DTYPE = np.dtype(
        [("id", np.intc), ("logit", np.single), ("p", np.single)],
        align=True,
    )

    def __init__(
        self,
        *,
        seed: int,
        vocab: ctypes.c_void_p,
        n_vocab: int,
        top_p: float,
        temperature: float,
        frequency_penalty: float,
        presence_penalty: float,
        logit_bias: Optional[Dict[int, float]],
        grammar_text: Optional[str] = None,
        grammar_root: str = "root",
    ) -> None:
        params = llama_cpp.llama_sampler_chain_default_params()
        self._sampler = llama_cpp.llama_sampler_chain_init(params)
        self._closed = False
        self._sample_logits_size = 0
        self._sample_logits_token_data: Optional[Any] = None
        self._sample_logits_token_array: Optional[Any] = None
        self._sample_logits_recarray: Optional[np.recarray] = None
        if logit_bias:
            bias_array = (llama_cpp.llama_logit_bias * len(logit_bias))()
            for index, (token, bias) in enumerate(logit_bias.items()):
                bias_array[index].token = ctypes.c_int32(token)
                bias_array[index].bias = float(bias)
            llama_cpp.llama_sampler_chain_add(
                self._sampler,
                llama_cpp.llama_sampler_init_logit_bias(
                    n_vocab, len(logit_bias), bias_array
                ),
            )
            self.bias_array = bias_array
        llama_cpp.llama_sampler_chain_add(
            self._sampler,
            llama_cpp.llama_sampler_init_penalties(
                64,
                1.0,
                frequency_penalty,
                presence_penalty,
            ),
        )
        if grammar_text is not None:
            grammar_sampler = llama_cpp.llama_sampler_init_grammar(
                vocab,
                grammar_text.encode("utf-8"),
                grammar_root.encode("utf-8"),
            )
            if grammar_sampler is None:
                raise RuntimeError("failed to initialize grammar sampler")
            llama_cpp.llama_sampler_chain_add(self._sampler, grammar_sampler)
        if temperature < 0.0:
            llama_cpp.llama_sampler_chain_add(
                self._sampler, llama_cpp.llama_sampler_init_dist(seed)
            )
            return
        if temperature == 0.0:
            llama_cpp.llama_sampler_chain_add(
                self._sampler, llama_cpp.llama_sampler_init_greedy()
            )
            return
        min_keep = 1
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_top_p(top_p, min_keep)
        )
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_temp(temperature)
        )
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_dist(seed)
        )

    def sample(self, ctx: ctypes.c_void_p, row_index: int) -> int:
        return int(llama_cpp.llama_sampler_sample(self._sampler, ctx, row_index))

    def _ensure_sample_logits_buffer(self, size: int) -> None:
        if size == self._sample_logits_size and self._sample_logits_recarray is not None:
            return
        token_data = (llama_cpp.llama_token_data * size)()
        token_data_address = ctypes.addressof(token_data)
        recarray = np.recarray(
            shape=(size,),
            dtype=self.TOKEN_DATA_DTYPE,
            buf=(llama_cpp.llama_token_data * size).from_address(token_data_address),
        )
        recarray.id[:] = np.arange(size, dtype=np.intc)
        token_array = llama_cpp.llama_token_data_array(
            data=token_data,
            size=size,
            selected=-1,
            sorted=False,
        )
        self._sample_logits_size = size
        self._sample_logits_token_data = token_data
        self._sample_logits_token_array = token_array
        self._sample_logits_recarray = recarray

    def sample_logits(self, logits: np.ndarray) -> int:
        self._ensure_sample_logits_buffer(len(logits))
        assert self._sample_logits_recarray is not None
        assert self._sample_logits_token_array is not None
        self._sample_logits_recarray.logit[:] = logits
        self._sample_logits_recarray.p.fill(0.0)
        self._sample_logits_token_array.selected = -1
        self._sample_logits_token_array.sorted = False
        llama_cpp.llama_sampler_apply(self._sampler, ctypes.byref(self._sample_logits_token_array))
        token = int(self._sample_logits_recarray.id[self._sample_logits_token_array.selected])
        llama_cpp.llama_sampler_accept(self._sampler, token)
        return token

    def close(self) -> None:
        if not self._closed:
            llama_cpp.llama_sampler_free(self._sampler)
            self._closed = True


class Model:
    def __init__(
        self,
        *,
        model_path: str,
        model_alias: Optional[str] = None,
        chat_template: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        split_mode: Optional[int] = None,
        main_gpu: Optional[int] = None,
        tensor_split: Optional[List[float]] = None,
        vocab_only: Optional[bool] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,
        n_ctx: Optional[int],
        n_batch: Optional[int],
        n_ubatch: Optional[int] = None,
        n_seq_max: Optional[int],
        n_threads: Optional[int],
        n_threads_batch: Optional[int],
        rope_scaling_type: Optional[int] = None,
        pooling_type: Optional[int] = None,
        attention_type: Optional[int] = None,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        yarn_ext_factor: Optional[float] = None,
        yarn_attn_factor: Optional[float] = None,
        yarn_beta_fast: Optional[float] = None,
        yarn_beta_slow: Optional[float] = None,
        yarn_orig_ctx: Optional[int] = None,
        offload_kqv: Optional[bool] = None,
        flash_attn: Optional[bool] = None,
        op_offload: Optional[bool] = None,
        swa_full: Optional[bool] = None,
        no_perf: Optional[bool] = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        kv_unified: bool = True,
        max_seq_len: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        draft_model: Optional[str] = None,
        draft_model_num_pred_tokens: int = 10,
        draft_model_max_ngram_size: int = 2,
        response_schema: Optional[Dict[str, Any]] = None,
        store_logits: bool = True,
    ) -> None:
        llama_cpp.llama_backend_init()
        self.backend_initialized = True
        self.model_path = model_path
        self.model_alias = model_alias
        self.chat_template_override = chat_template
        self.response_schema = response_schema
        self.store_logits = store_logits
        self.max_output_tokens = max_output_tokens
        model_params, self._c_tensor_split, self._kv_overrides_array = (
            self.build_model_params(
                n_gpu_layers=n_gpu_layers,
                split_mode=split_mode,
                main_gpu=main_gpu,
                tensor_split=tensor_split,
                vocab_only=vocab_only,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                kv_overrides=kv_overrides,
            )
        )
        llama_model = llama_cpp.llama_model_load_from_file(
            model_path.encode("utf-8"),
            model_params,
        )
        if llama_model is None:
            raise RuntimeError(f"failed to load model: {model_path}")
        self.llama_model = llama_model
        self.vocab = llama_cpp.llama_model_get_vocab(llama_model)
        if self.vocab is None:
            raise RuntimeError("failed to access model vocabulary")
        if llama_cpp.llama_model_has_encoder(llama_model):
            raise RuntimeError("encoder models are not supported")
        if not llama_cpp.llama_model_has_decoder(llama_model):
            raise RuntimeError("decoder is required")
        if llama_cpp.llama_model_is_recurrent(llama_model):
            self.memory_model = "recurrent"
        elif llama_cpp.llama_model_is_hybrid(llama_model):
            self.memory_model = "hybrid"
        else:
            self.memory_model = (
                "attention-unified" if kv_unified else "attention-partitioned"
            )
        if draft_model is not None and not self.memory_model.startswith("attention"):
            raise RuntimeError(
                "speculative decoding is only supported for attention models"
            )
        n_ctx_train = int(llama_cpp.llama_model_n_ctx_train(llama_model))
        context_params = self.build_context_params(
            n_ctx=n_ctx if n_ctx is not None else n_ctx_train,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_seq_max=n_seq_max,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            rope_scaling_type=rope_scaling_type,
            pooling_type=pooling_type,
            attention_type=attention_type,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            yarn_ext_factor=yarn_ext_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_orig_ctx=yarn_orig_ctx,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn,
            op_offload=op_offload,
            swa_full=swa_full,
            no_perf=no_perf,
            type_k=type_k,
            type_v=type_v,
            kv_unified=kv_unified,
        )
        ctx = llama_cpp.llama_init_from_model(llama_model, context_params)
        if ctx is None:
            raise RuntimeError("failed to create context")
        self.ctx = ctx
        self.mem = llama_cpp.llama_get_memory(ctx)
        self.n_ctx = int(llama_cpp.llama_n_ctx(ctx))
        self.n_ctx_seq = int(llama_cpp.llama_n_ctx_seq(ctx))
        self.n_seq_max = int(llama_cpp.llama_n_seq_max(ctx))
        self.n_batch = int(llama_cpp.llama_n_batch(ctx))
        self.n_ctx_train = n_ctx_train
        self.n_vocab = int(llama_cpp.llama_vocab_n_tokens(self.vocab))
        self.kv_unified = kv_unified
        self.max_seq_len_limit = min(self.request_context_limit, self.n_ctx_train)
        if max_seq_len is None:
            self.max_seq_len = self.max_seq_len_limit
        else:
            if max_seq_len <= 0:
                llama_cpp.llama_free(self.ctx)
                llama_cpp.llama_model_free(self.llama_model)
                if self.backend_initialized:
                    llama_cpp.llama_backend_free()
                    self.backend_initialized = False
                raise RuntimeError("max_seq_len must be greater than 0")
            if max_seq_len > self.max_seq_len_limit:
                llama_cpp.llama_free(self.ctx)
                llama_cpp.llama_model_free(self.llama_model)
                if self.backend_initialized:
                    llama_cpp.llama_backend_free()
                    self.backend_initialized = False
                raise RuntimeError(
                    f"max_seq_len ({max_seq_len}) exceeds runtime limit ({self.max_seq_len_limit})"
                )
            self.max_seq_len = max_seq_len
        self.batch = llama_cpp.llama_batch_init(self.n_batch, 0, self.n_seq_max)
        self.bos_token = int(llama_cpp.llama_vocab_bos(self.vocab))
        self.eos_token = int(llama_cpp.llama_vocab_eos(self.vocab))
        self.cls_token = self.bos_token
        self.sep_token = int(llama_cpp.llama_vocab_sep(self.vocab))
        self.fim_pre_token = int(llama_cpp.llama_vocab_fim_pre(self.vocab))
        self.fim_mid_token = int(llama_cpp.llama_vocab_fim_mid(self.vocab))
        self.fim_suf_token = int(llama_cpp.llama_vocab_fim_suf(self.vocab))
        self.add_bos_token = bool(llama_cpp.llama_vocab_get_add_bos(self.vocab))
        self.add_eos_token = bool(llama_cpp.llama_vocab_get_add_eos(self.vocab))
        self.add_space_prefix = (
            self._meta_value("tokenizer.ggml.add_space_prefix") != "false"
        )
        if draft_model is None:
            self.draft_provider: Optional[DraftProvider] = None
        elif draft_model == "prompt-lookup-decoding":
            self.draft_provider = PromptLookupDecoding(
                max_ngram_size=draft_model_max_ngram_size,
                num_pred_tokens=draft_model_num_pred_tokens,
            )
        else:
            raise RuntimeError(f"unsupported draft model: {draft_model}")
        self.chat_formatter = self._build_chat_formatter()

    @staticmethod
    def build_model_params(
        *,
        n_gpu_layers: Optional[int],
        split_mode: Optional[int],
        main_gpu: Optional[int],
        tensor_split: Optional[List[float]],
        vocab_only: Optional[bool],
        use_mmap: Optional[bool],
        use_mlock: Optional[bool],
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]],
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        model_params = llama_cpp.llama_model_default_params()
        if n_gpu_layers is not None:
            model_params.n_gpu_layers = 0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers
        if split_mode is not None:
            model_params.split_mode = split_mode
        if main_gpu is not None:
            model_params.main_gpu = main_gpu
        tensor_split_ref = None
        if tensor_split is not None:
            if len(tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:
                raise ValueError(
                    "tensor_split exceeds maximum supported devices "
                    f"({llama_cpp.LLAMA_MAX_DEVICES})"
                )
            float_array = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES
            tensor_split_ref = float_array(*tensor_split)
            model_params.tensor_split = tensor_split_ref
        if vocab_only is not None:
            model_params.vocab_only = vocab_only
        if use_mmap is not None:
            model_params.use_mmap = use_mmap
        if use_mlock is not None:
            model_params.use_mlock = use_mlock

        kv_overrides_ref = None
        if kv_overrides is not None:
            kv_overrides_ref = (
                llama_cpp.llama_model_kv_override * (len(kv_overrides) + 1)
            )()
            for index, (key, value) in enumerate(kv_overrides.items()):
                kv_overrides_ref[index].key = key.encode("utf-8")
                if isinstance(value, bool):
                    kv_overrides_ref[index].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL
                    kv_overrides_ref[index].value.val_bool = value
                elif isinstance(value, int):
                    kv_overrides_ref[index].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT
                    kv_overrides_ref[index].value.val_i64 = value
                elif isinstance(value, float):
                    kv_overrides_ref[index].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT
                    kv_overrides_ref[index].value.val_f64 = value
                elif isinstance(value, str):
                    encoded = value.encode("utf-8")
                    if len(encoded) > 128:
                        raise ValueError(f"kv_overrides value for {key} is too long")
                    encoded = encoded.ljust(128, b"\0")
                    kv_overrides_ref[index].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR
                    address = cast(
                        int,
                        ctypes.addressof(kv_overrides_ref[index].value)
                        + llama_cpp.llama_model_kv_override_value.val_str.offset,
                    )
                    buffer_start = ctypes.cast(address, ctypes.POINTER(ctypes.c_char))
                    ctypes.memmove(buffer_start, encoded, 128)
                else:
                    raise ValueError(f"unsupported kv_override value for {key}: {value!r}")
            kv_overrides_ref[-1].key = b"\0"
            model_params.kv_overrides = kv_overrides_ref
        return model_params, tensor_split_ref, kv_overrides_ref

    @staticmethod
    def build_context_params(
        *,
        n_ctx: Optional[int],
        n_batch: Optional[int],
        n_ubatch: Optional[int],
        n_seq_max: Optional[int],
        n_threads: Optional[int],
        n_threads_batch: Optional[int],
        rope_scaling_type: Optional[int],
        pooling_type: Optional[int],
        attention_type: Optional[int],
        rope_freq_base: Optional[float],
        rope_freq_scale: Optional[float],
        yarn_ext_factor: Optional[float],
        yarn_attn_factor: Optional[float],
        yarn_beta_fast: Optional[float],
        yarn_beta_slow: Optional[float],
        yarn_orig_ctx: Optional[int],
        offload_kqv: Optional[bool],
        flash_attn: Optional[bool],
        op_offload: Optional[bool],
        swa_full: Optional[bool],
        no_perf: Optional[bool],
        type_k: Optional[int],
        type_v: Optional[int],
        kv_unified: bool,
    ) -> Any:
        context_params = llama_cpp.llama_context_default_params()
        if n_ctx is not None:
            context_params.n_ctx = n_ctx
        if n_batch is not None:
            context_params.n_batch = min(int(context_params.n_ctx), n_batch)
        if n_ubatch is not None:
            context_params.n_ubatch = min(int(context_params.n_batch), n_ubatch)
        if n_seq_max is not None:
            context_params.n_seq_max = n_seq_max
        if n_threads is not None:
            context_params.n_threads = n_threads
        if n_threads_batch is not None:
            context_params.n_threads_batch = n_threads_batch
        if rope_scaling_type is not None:
            context_params.rope_scaling_type = rope_scaling_type
        if pooling_type is not None:
            context_params.pooling_type = pooling_type
        if attention_type is not None:
            context_params.attention_type = attention_type
        if rope_freq_base is not None:
            context_params.rope_freq_base = rope_freq_base
        if rope_freq_scale is not None:
            context_params.rope_freq_scale = rope_freq_scale
        if yarn_ext_factor is not None:
            context_params.yarn_ext_factor = yarn_ext_factor
        if yarn_attn_factor is not None:
            context_params.yarn_attn_factor = yarn_attn_factor
        if yarn_beta_fast is not None:
            context_params.yarn_beta_fast = yarn_beta_fast
        if yarn_beta_slow is not None:
            context_params.yarn_beta_slow = yarn_beta_slow
        if yarn_orig_ctx is not None:
            context_params.yarn_orig_ctx = yarn_orig_ctx
        if offload_kqv is not None:
            context_params.offload_kqv = offload_kqv
        if flash_attn is not None:
            context_params.flash_attn_type = (
                llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
                if flash_attn
                else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
            )
        if op_offload is not None:
            context_params.op_offload = op_offload
        if swa_full is not None:
            context_params.swa_full = swa_full
        if no_perf is not None:
            context_params.no_perf = no_perf
        if type_k is not None:
            context_params.type_k = type_k
        if type_v is not None:
            context_params.type_v = type_v
        context_params.kv_unified = kv_unified
        return context_params

    @property
    def exact_checkpoints_only(self) -> bool:
        return self.memory_model in {"recurrent", "hybrid"}

    @property
    def has_attention_budget(self) -> bool:
        return self.memory_model != "recurrent"

    @property
    def attention_partitioned(self) -> bool:
        return self.memory_model == "attention-partitioned"

    @property
    def request_context_limit(self) -> int:
        if self.attention_partitioned:
            return self.n_ctx_seq
        return self.n_ctx

    def close(self) -> None:
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_free(self.ctx)
        llama_cpp.llama_model_free(self.llama_model)
        if self.backend_initialized:
            llama_cpp.llama_backend_free()
            self.backend_initialized = False

    def _meta_value(self, key: str) -> Optional[str]:
        encoded = key.encode("utf-8")
        capacity = 256
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            count = int(
                llama_cpp.llama_model_meta_val_str(
                    self.llama_model,
                    encoded,
                    buffer,
                    capacity,
                )
            )
            if count < 0:
                return None
            if count < capacity:
                return buffer.value.decode("utf-8", errors="ignore")
            capacity = count + 1

    def _build_chat_formatter(self) -> Optional[Jinja2ChatFormatter]:
        template_text = self.chat_template_override
        if template_text is None:
            template = llama_cpp.llama_model_chat_template(self.llama_model, None)
            if template:
                template_text = template.decode("utf-8", errors="ignore")
        if not template_text:
            return None
        bos_token = ""
        eos_token = ""
        if self.bos_token != -1:
            bos_text = llama_cpp.llama_vocab_get_text(self.vocab, self.bos_token)
            bos_token = bos_text.decode("utf-8", errors="ignore") if bos_text else ""
        if self.eos_token != -1:
            eos_text = llama_cpp.llama_vocab_get_text(self.vocab, self.eos_token)
            eos_token = eos_text.decode("utf-8", errors="ignore") if eos_text else ""
        return Jinja2ChatFormatter(
            template=template_text,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    def tokenize(
        self,
        text: str,
        *,
        add_bos: bool = True,
        special: bool = True,
    ) -> List[int]:
        encoded = text.encode("utf-8")
        capacity = max(32, len(encoded) + 32)
        while True:
            tokens = (llama_cpp.llama_token * capacity)()
            count = int(
                llama_cpp.llama_tokenize(
                    self.vocab,
                    encoded,
                    len(encoded),
                    tokens,
                    capacity,
                    add_bos,
                    special,
                )
            )
            if count >= 0:
                return [int(tokens[index]) for index in range(count)]
            capacity = max(capacity * 2, -count)

    def build_prompt_tokens(self, prompt: str, suffix: Optional[str]) -> List[int]:
        if suffix is None:
            return self.tokenize(prompt)
        if min(self.fim_pre_token, self.fim_mid_token, self.fim_suf_token) < 0:
            raise ValueError("suffix is not supported by this model")
        bos_tokens = [self.cls_token if self.cls_token != -1 else self.bos_token]
        eos_tokens = [self.sep_token if self.sep_token != -1 else self.eos_token]
        if not self.add_bos_token or bos_tokens[:1] == [-1]:
            bos_tokens = []
        if not self.add_eos_token and self.sep_token == -1:
            eos_tokens = []
        suffix_text = suffix
        suffix_space_prefix = 0
        if self.add_space_prefix and suffix_text:
            suffix_text = "☺" + suffix_text
            suffix_space_prefix = 2
        prefix_tokens = [self.fim_pre_token] + self.tokenize(
            prompt,
            add_bos=False,
            special=False,
        )
        suffix_tokens = [self.fim_suf_token]
        if suffix_text:
            suffix_tokens.extend(
                self.tokenize(
                    suffix_text,
                    add_bos=False,
                    special=False,
                )[suffix_space_prefix:]
            )
        return bos_tokens + prefix_tokens + suffix_tokens + [self.fim_mid_token] + eos_tokens

    def build_chat_prompt(
        self,
        messages: List[ChatCompletionRequestMessage],
        *,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[str, str, List[int], List[str]]:
        if self.chat_formatter is None:
            raise ValueError("model does not provide a GGUF chat template")
        prompt, generation_prompt, formatter_stop = self.chat_formatter.format(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        )
        prompt_tokens = self.tokenize(prompt, add_bos=False, special=True)
        return prompt, generation_prompt, prompt_tokens, formatter_stop

    def detokenize(self, tokens: Sequence[int]) -> bytes:
        if not tokens:
            return b""
        array = (llama_cpp.llama_token * len(tokens))(*tokens)
        capacity = max(64, len(tokens) * 16)
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            count = int(
                llama_cpp.llama_detokenize(
                    self.vocab,
                    array,
                    len(tokens),
                    buffer,
                    capacity,
                    True,
                    True,
                )
            )
            if count >= 0:
                return bytes(buffer.raw[:count])
            capacity = max(capacity * 2, -count)

    def token_bytes(self, token: int) -> bytes:
        return self.detokenize([token])

    def token_bytes_with_prev(self, prev_tokens: Sequence[int], token: int) -> bytes:
        current = self.detokenize([*prev_tokens, token])
        previous = self.detokenize(prev_tokens)
        return current[len(previous) :]

    def token_bytes_with_prev_bytes(
        self,
        prev_tokens: Sequence[int],
        prev_text_bytes: Union[bytes, bytearray],
        token: int,
    ) -> bytes:
        current = self.detokenize([*prev_tokens, token])
        return current[len(prev_text_bytes) :]

    def clear_batch(self) -> None:
        self.batch.n_tokens = 0

    def add_batch_tokens(
        self,
        *,
        seq_id: int,
        start_pos: int,
        tokens: Sequence[int],
        output_indices: Sequence[Optional[int]],
    ) -> None:
        if not tokens:
            return
        for index, token in enumerate(tokens):
            slot = self.batch.n_tokens
            self.batch.token[slot] = token
            self.batch.pos[slot] = start_pos + index
            self.batch.seq_id[slot][0] = seq_id
            self.batch.n_seq_id[slot] = 1
            self.batch.logits[slot] = int(output_indices[index] is not None)
            self.batch.n_tokens += 1

    def decode(self) -> None:
        result = int(llama_cpp.llama_decode(self.ctx, self.batch))
        if result != 0:
            raise RuntimeError(f"llama_decode failed with code {result}")

    def logits(self, row_index: int) -> np.ndarray:
        ptr = llama_cpp.llama_get_logits_ith(self.ctx, row_index)
        if not ptr:
            raise RuntimeError(f"missing logits row {row_index}")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()


class MemoryPolicy(abc.ABC):
    def __init__(self, scheduler: CompletionScheduler) -> None:
        self.scheduler = scheduler

    def reclaim_order(self, best_free: Optional[int]) -> List[int]:
        reclaim_order = [seq_id for seq_id in self.scheduler.free_sequences if seq_id != best_free]
        if best_free is not None:
            reclaim_order.append(best_free)
        return reclaim_order

    @abc.abstractmethod
    def can_admit(self, request: CompletionRequest) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def admit_request(self, request: CompletionRequest) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        raise NotImplementedError


class AttentionMemoryPolicy(MemoryPolicy):
    def match_prefix(self, tokens: Sequence[int]) -> Tuple[int, int]:
        return self.scheduler.radix_trie.longest_prefix(
            tokens,
            preferred_sequences=self.scheduler.free_sequences,
        )

    @staticmethod
    def reuse_len_for_request(request: CompletionRequest, match_length: int) -> int:
        needs_generation = (
            request.payload.max_tokens != 0
            and request.effective_max_len > len(request.prompt_tokens)
        )
        reuse_len = match_length
        if needs_generation and request.prompt_tokens:
            reuse_len = min(reuse_len, len(request.prompt_tokens) - 1)
        return reuse_len

    def admit_request(self, request: CompletionRequest) -> None:
        match_seq_id = request.match_sequence_id
        match_length = request.match_length
        reuse_len = self.reuse_len_for_request(request, match_length)
        claimable = match_seq_id in self.scheduler.free_sequences
        if claimable:
            base_seq_id = match_seq_id
            del self.scheduler.free_sequences[base_seq_id]
            self.scheduler.claimed_sequences.add(base_seq_id)
            if self.scheduler.radix_trie.length(base_seq_id) > reuse_len:
                self.scheduler.truncate_sequence(base_seq_id, reuse_len)
        else:
            base_seq_id = self.scheduler.unused_sequences.pop()
            self.scheduler.claimed_sequences.add(base_seq_id)
            if reuse_len > 0 and match_seq_id >= 0:
                self.copy_prompt_state(match_seq_id, base_seq_id, reuse_len)
        sibling_count = request.internal_completion_count - 1
        sibling_seq_ids: List[int] = []
        for _ in range(sibling_count):
            seq_id = self.scheduler.unused_sequences.pop()
            self.scheduler.claimed_sequences.add(seq_id)
            sibling_seq_ids.append(seq_id)
        request.base_seq_id = base_seq_id
        request.sibling_seq_ids = sibling_seq_ids
        request.completion_seq_ids = [base_seq_id, *sibling_seq_ids]
        request.prompt_cursor = reuse_len
        request.admitted = True
        self.scheduler.active_request_ids.add(request.id)
        if request.prompt_cursor == len(request.prompt_tokens):
            request.prompt_done = True
            self.scheduler.start_completions(request, prompt_row_index=None)


class UnifiedAttentionMemoryPolicy(AttentionMemoryPolicy):
    def can_admit(self, request: CompletionRequest) -> bool:
        match_seq_id, match_length = self.match_prefix(request.prompt_tokens)
        request.match_sequence_id = match_seq_id
        request.match_length = match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        required_sequence_ids = request.internal_completion_count - int(claimable)
        prompt_length = len(request.prompt_tokens)
        reuse_len = self.reuse_len_for_request(request, match_length)
        prefix_credit = match_length if claimable else reuse_len
        required_kv = (
            prompt_length
            - prefix_credit
            + request.internal_completion_count * (request.effective_max_len - prompt_length)
        )
        if (
            len(self.scheduler.unused_sequences) >= required_sequence_ids
            and self.scheduler.sequence_history.size + required_kv <= self.scheduler.model.n_ctx
        ):
            return True
        best_free = match_seq_id if claimable else None
        for seq_id in self.reclaim_order(best_free):
            if len(self.scheduler.unused_sequences) < required_sequence_ids:
                self.scheduler.delete_free_sequence(seq_id)
            elif self.scheduler.sequence_history.size + required_kv > self.scheduler.model.n_ctx:
                if seq_id == best_free and request.match_length > 0:
                    self.scheduler.truncate_free_sequence(seq_id, request.match_length)
                else:
                    self.scheduler.delete_free_sequence(seq_id)
            if (
                len(self.scheduler.unused_sequences) >= required_sequence_ids
                and self.scheduler.sequence_history.size + required_kv <= self.scheduler.model.n_ctx
            ):
                request.match_sequence_id, request.match_length = self.match_prefix(
                    request.prompt_tokens,
                )
                return True
        return False

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        if keep_len <= 0:
            return
        source_length = self.scheduler.radix_trie.length(source_sequence_id)
        copy_p0 = 0
        copy_p1 = keep_len
        if not self.scheduler.model.kv_unified:
            copy_p0 = -1
            copy_p1 = -1
        llama_cpp.llama_memory_seq_cp(
            self.scheduler.model.mem,
            source_sequence_id,
            dest_sequence_id,
            copy_p0,
            copy_p1,
        )
        self.scheduler.radix_trie.copy(source_sequence_id, dest_sequence_id, keep_len)
        self.scheduler.sequence_history.copy(
            source_sequence_id,
            dest_sequence_id,
            source_length,
            keep_len,
        )


class PartitionedAttentionMemoryPolicy(AttentionMemoryPolicy):
    def can_admit(self, request: CompletionRequest) -> bool:
        match_seq_id, match_length = self.match_prefix(request.prompt_tokens)
        request.match_sequence_id = match_seq_id
        request.match_length = match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        required_sequence_ids = request.internal_completion_count - int(claimable)
        if len(self.scheduler.unused_sequences) >= required_sequence_ids:
            return True
        best_free = match_seq_id if claimable else None
        for seq_id in self.reclaim_order(best_free):
            self.scheduler.delete_free_sequence(seq_id)
            if len(self.scheduler.unused_sequences) >= required_sequence_ids:
                request.match_sequence_id, request.match_length = self.match_prefix(
                    request.prompt_tokens,
                )
                return True
        return False

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        if keep_len <= 0:
            return
        source_length = self.scheduler.radix_trie.length(source_sequence_id)
        llama_cpp.llama_memory_seq_cp(
            self.scheduler.model.mem,
            source_sequence_id,
            dest_sequence_id,
            -1,
            -1,
        )
        if source_length > keep_len:
            llama_cpp.llama_memory_seq_rm(self.scheduler.model.mem, dest_sequence_id, keep_len, -1)
        prefix_tokens = self.scheduler.radix_trie.tokens(source_sequence_id, keep_len)
        self.scheduler.radix_trie.copy(source_sequence_id, dest_sequence_id, keep_len)
        self.scheduler.sequence_history.extend(dest_sequence_id, prefix_tokens)


class CheckpointMemoryPolicy(MemoryPolicy):
    def exact_checkpoint_match(self, tokens: Sequence[int]) -> Tuple[int, int]:
        match_seq_id, match_length = self.scheduler.radix_trie.longest_prefix(
            tokens,
            preferred_sequences=self.scheduler.free_sequences,
            exact_only=True,
        )
        if match_seq_id not in self.scheduler.free_sequences:
            return -1, 0
        return match_seq_id, match_length

    def can_admit(self, request: CompletionRequest) -> bool:
        match_seq_id, match_length = self.exact_checkpoint_match(request.prompt_tokens)
        request.match_sequence_id = match_seq_id
        request.match_length = match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        required_sequence_ids = request.internal_completion_count - int(claimable)
        required_attn_kv = (
            len(request.prompt_tokens) - match_length
            + request.internal_completion_count
            * (request.effective_max_len - len(request.prompt_tokens))
        )
        if len(self.scheduler.unused_sequences) >= required_sequence_ids and (
            not self.scheduler.model.has_attention_budget
            or self.scheduler.sequence_history.size + required_attn_kv <= self.scheduler.model.n_ctx
        ):
            return True
        best_free = match_seq_id if claimable else None
        for seq_id in self.reclaim_order(best_free):
            self.scheduler.delete_free_sequence(seq_id)
            if len(self.scheduler.unused_sequences) >= required_sequence_ids and (
                not self.scheduler.model.has_attention_budget
                or self.scheduler.sequence_history.size + required_attn_kv <= self.scheduler.model.n_ctx
            ):
                request.match_sequence_id, request.match_length = self.exact_checkpoint_match(
                    request.prompt_tokens,
                )
                return True
        return False

    def admit_request(self, request: CompletionRequest) -> None:
        match_seq_id = request.match_sequence_id
        match_length = request.match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        if claimable:
            base_seq_id = match_seq_id
            del self.scheduler.free_sequences[base_seq_id]
            self.scheduler.claimed_sequences.add(base_seq_id)
            request.prompt_logits = self.scheduler.checkpoint_logits.get(base_seq_id)
        else:
            base_seq_id = self.scheduler.unused_sequences.pop()
            self.scheduler.claimed_sequences.add(base_seq_id)
            request.prompt_logits = None
        sibling_count = request.internal_completion_count - 1
        sibling_seq_ids: List[int] = []
        for _ in range(sibling_count):
            seq_id = self.scheduler.unused_sequences.pop()
            self.scheduler.claimed_sequences.add(seq_id)
            sibling_seq_ids.append(seq_id)
        request.base_seq_id = base_seq_id
        request.sibling_seq_ids = sibling_seq_ids
        request.completion_seq_ids = [base_seq_id, *sibling_seq_ids]
        request.prompt_cursor = match_length
        request.admitted = True
        self.scheduler.active_request_ids.add(request.id)
        if request.prompt_cursor == len(request.prompt_tokens):
            request.prompt_done = True
            self.scheduler.maybe_save_prompt_checkpoint(request)
            self.scheduler.start_completions(
                request,
                prompt_row_index=None,
                prompt_logits=request.prompt_logits,
            )

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        if keep_len <= 0:
            return
        source_length = self.scheduler.radix_trie.length(source_sequence_id)
        copy_p0 = 0
        copy_p1 = keep_len
        if not self.scheduler.model.kv_unified:
            copy_p0 = -1
            copy_p1 = -1
        llama_cpp.llama_memory_seq_cp(
            self.scheduler.model.mem,
            source_sequence_id,
            dest_sequence_id,
            copy_p0,
            copy_p1,
        )
        self.scheduler.radix_trie.copy(source_sequence_id, dest_sequence_id, keep_len)
        self.scheduler.sequence_history.copy(
            source_sequence_id,
            dest_sequence_id,
            source_length,
            keep_len,
        )


class CompletionScheduler:
    @dataclass
    class BatchItem:
        kind: str
        request_id: str
        seq_id: int
        start_pos: int
        tokens: List[int]
        output_indices: List[Optional[int]]
        completion_index: Optional[int] = None
        pending_count: int = 0

    def __init__(self, model: Model) -> None:
        self.model = model
        self.formatter = OpenAIFormatter(model)
        self.radix_trie = RadixTrie()
        self.sequence_history = SequenceHistory()
        self.checkpoint_logits: Dict[int, np.ndarray] = {}
        self.claimed_sequences: set[int] = set()
        self.free_sequences: "OrderedDict[int, None]" = OrderedDict()
        self.unused_sequences: List[int] = list(range(self.model.n_seq_max - 1, -1, -1))
        self.requests: Dict[str, CompletionRequest] = {}
        self.pending_requests: Deque[CompletionRequest] = deque()
        self.active_request_ids: set[str] = set()
        self.closed = False
        self.sequence_round_robin = 0
        self.speculative_stats: Dict[str, int] = {
            "draft_proposals": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "draft_tokens_rejected": 0,
        }
        self.memory_policy = self.build_memory_policy()

    def build_memory_policy(self) -> MemoryPolicy:
        if self.model.exact_checkpoints_only:
            return CheckpointMemoryPolicy(self)
        if self.model.attention_partitioned:
            return PartitionedAttentionMemoryPolicy(self)
        return UnifiedAttentionMemoryPolicy(self)

    def close(self) -> None:
        self.closed = True
        self.model.close()

    def submit_request(self, request: CompletionRequest) -> str:
        if self.closed:
            raise RuntimeError("scheduler closed")
        self.requests[request.id] = request
        self.pending_requests.append(request)
        return request.id

    def cancel(self, request_id: str) -> None:
        request = self.requests.get(request_id)
        if request is not None:
            request.cancelled = True

    def is_idle(self) -> bool:
        return not self.pending_requests and not self.active_request_ids

    @staticmethod
    def logits_to_logprobs(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32, copy=False)
        max_logit = float(np.max(logits))
        shifted = logits - max_logit
        return shifted - math.log(float(np.sum(np.exp(shifted, dtype=np.float64))))

    def step(self) -> bool:
        if self.closed:
            return False
        self.admit_waiting()
        batch_items = self.build_batch()
        if not batch_items:
            return self.finalize_cancelled()
        self.model.clear_batch()
        for item in batch_items:
            self.model.add_batch_tokens(
                seq_id=item.seq_id,
                start_pos=item.start_pos,
                tokens=item.tokens,
                output_indices=item.output_indices,
            )
        try:
            self.model.decode()
        except BaseException as exc:  # noqa: BLE001
            for request_id in list(self.active_request_ids):
                self.fail_request(self.requests[request_id], exc)
            for request in list(self.pending_requests):
                self.pending_requests.remove(request)
                self.fail_request(request, exc)
            return True
        self.process_batch(batch_items)
        self.finalize_cancelled()
        return True

    def admit_waiting(self) -> None:
        while self.pending_requests:
            request = self.pending_requests[0]
            if request.cancelled:
                self.pending_requests.popleft()
                self.fail_request(request, CompletionRequestCancelledError("request cancelled"))
                continue
            if not self.can_admit(request):
                break
            self.pending_requests.popleft()
            self.admit_request(request)

    def can_admit(self, request: CompletionRequest) -> bool:
        return self.memory_policy.can_admit(request)

    def admit_request(self, request: CompletionRequest) -> None:
        self.memory_policy.admit_request(request)

    def build_batch(self) -> List[CompletionScheduler.BatchItem]:
        prompt_requests = [
            self.requests[request_id]
            for request_id in self.active_request_ids
            if self.requests[request_id].admitted
            and not self.requests[request_id].prompt_done
            and not self.requests[request_id].cancelled
        ]
        completions = [
            completion
            for request_id in self.active_request_ids
            for completion in self.requests[request_id].completions
            if (completion.pending_input_tokens or completion.draft_tokens)
            and not completion.finished
        ]
        if not prompt_requests and not completions:
            return []

        ordered_sequences = self._ordered_pending_sequences(prompt_requests, completions)
        allocations = self._allocate_pending_tokens(
            ordered_sequences,
            self.model.n_batch,
        )
        if ordered_sequences:
            self.sequence_round_robin += 1

        items: List[CompletionScheduler.BatchItem] = []
        output_index = 0
        for source in ordered_sequences:
            token_count = allocations.get(self._pending_sequence_id(source), 0)
            if token_count <= 0:
                continue
            item, output_index = self._build_pending_batch_item(
                source,
                token_count,
                output_index,
            )
            items.append(item)
        return items

    def _ordered_pending_sequences(
        self,
        prompt_requests: Sequence[CompletionRequest],
        completions: Sequence[Completion],
    ) -> List[Union[CompletionRequest, Completion]]:
        if not prompt_requests and not completions:
            return []
        prompt_by_request_id = {request.id: request for request in prompt_requests}
        completions_by_request_id: Dict[str, List[Completion]] = {}
        for completion in completions:
            completions_by_request_id.setdefault(completion.request_id, []).append(completion)
        sequence_list: List[Union[CompletionRequest, Completion]] = []
        for request_id, request in self.requests.items():
            if request_id not in self.active_request_ids or request.cancelled:
                continue
            prompt_request = prompt_by_request_id.get(request_id)
            if prompt_request is not None:
                sequence_list.append(prompt_request)
            sequence_list.extend(completions_by_request_id.get(request_id, []))
        if not sequence_list:
            return []
        start = self.sequence_round_robin % len(sequence_list)
        return sequence_list[start:] + sequence_list[:start]

    def _pending_sequence_id(
        self,
        source: Union[CompletionRequest, Completion],
    ) -> int:
        if isinstance(source, CompletionRequest):
            if source.base_seq_id is None:
                raise RuntimeError("prompt sequence is missing base seq id")
            return source.base_seq_id
        return source.seq_id

    def _pending_tokens_length(
        self,
        source: Union[CompletionRequest, Completion],
    ) -> int:
        if isinstance(source, CompletionRequest):
            return len(source.prompt_tokens) - source.prompt_cursor
        draft_count = len(source.draft_tokens) if source.pending_finish_reason is None else 0
        return len(source.pending_input_tokens) + draft_count

    def _allocate_pending_tokens(
        self,
        sources: Sequence[Union[CompletionRequest, Completion]],
        capacity: int,
    ) -> Dict[int, int]:
        allocations: Dict[int, int] = {}
        active = [source for source in sources if self._pending_tokens_length(source) > 0]
        remaining_capacity = capacity

        for source in active:
            if remaining_capacity <= 0:
                break
            seq_id = self._pending_sequence_id(source)
            allocations[seq_id] = 1
            remaining_capacity -= 1

        while remaining_capacity > 0 and active:
            share = max(1, remaining_capacity // len(active))
            progress = False
            next_active: List[Union[CompletionRequest, Completion]] = []
            for source in active:
                seq_id = self._pending_sequence_id(source)
                remaining_tokens = self._pending_tokens_length(source) - allocations.get(seq_id, 0)
                if remaining_tokens <= 0:
                    continue
                allocation = min(remaining_tokens, share, remaining_capacity)
                if allocation <= 0:
                    next_active.append(source)
                    continue
                allocations[seq_id] = allocations.get(seq_id, 0) + allocation
                remaining_capacity -= allocation
                progress = True
                if remaining_tokens > allocation and remaining_capacity > 0:
                    next_active.append(source)
            if not progress:
                break
            active = next_active
        return allocations

    def _build_pending_batch_item(
        self,
        source: Union[CompletionRequest, Completion],
        token_count: int,
        output_index: int,
    ) -> Tuple["CompletionScheduler.BatchItem", int]:
        if isinstance(source, CompletionRequest):
            if source.base_seq_id is None:
                raise RuntimeError("prompt sequence is missing base seq id")
            remaining_prompt = source.prompt_tokens[source.prompt_cursor :]
            chunk = list(remaining_prompt[:token_count])
            ends_prompt = source.prompt_cursor + len(chunk) == len(source.prompt_tokens)
            output_indices: List[Optional[int]] = [None] * len(chunk)
            if source.payload.echo and source.payload.logprobs is not None:
                for index in range(len(chunk)):
                    output_indices[index] = output_index
                    output_index += 1
            elif self.model.exact_checkpoints_only and chunk:
                output_indices[-1] = output_index
                output_index += 1
            elif ends_prompt and chunk:
                output_indices[-1] = output_index
                output_index += 1
            return (
                CompletionScheduler.BatchItem(
                    kind="prompt",
                    request_id=source.id,
                    seq_id=source.base_seq_id,
                    start_pos=source.prompt_cursor,
                    tokens=chunk,
                    output_indices=output_indices,
                ),
                output_index,
            )

        request = self.requests[source.request_id]
        pending_count = min(token_count, len(source.pending_input_tokens))
        draft_count = max(0, token_count - pending_count)
        scheduled_tokens = [
            *source.pending_input_tokens[:pending_count],
            *(
                source.draft_tokens[:draft_count]
                if source.pending_finish_reason is None
                else []
            ),
        ]
        output_indices = list(range(output_index, output_index + len(scheduled_tokens)))
        return (
            CompletionScheduler.BatchItem(
                kind="token",
                request_id=request.id,
                seq_id=source.seq_id,
                start_pos=self.radix_trie.length(source.seq_id),
                tokens=list(scheduled_tokens),
                output_indices=output_indices,
                completion_index=source.index,
                pending_count=pending_count,
            ),
            output_index + len(scheduled_tokens),
        )

    def process_batch(self, items: List[CompletionScheduler.BatchItem]) -> None:
        output_count = sum(
            output_index is not None
            for item in items
            for output_index in item.output_indices
        )
        for item in items:
            request = self.requests[item.request_id]
            if request.cancelled:
                continue
            self.radix_trie.extend(item.seq_id, item.tokens)
            self.sequence_history.extend(item.seq_id, item.tokens)
            if item.kind == "prompt":
                request.capture_prompt_logprobs(
                    model=self.model,
                    formatter=self.formatter,
                    start_pos=item.start_pos,
                    output_indices=item.output_indices,
                    output_count=output_count,
                    output_arg=self.output_arg,
                )
                prompt_row_index = self.output_arg(
                    self.last_output_index(item.output_indices),
                    output_count,
                )
                prompt_logits = (
                    self.model.logits(prompt_row_index)
                    if prompt_row_index is not None
                    else None
                )
                request.prompt_cursor += len(item.tokens)
                if request.prompt_cursor == len(request.prompt_tokens):
                    request.prompt_done = True
                    request.prompt_logits = prompt_logits
                    self.maybe_save_prompt_checkpoint(request)
                    self.start_completions(
                        request,
                        prompt_row_index=prompt_row_index,
                        prompt_logits=request.prompt_logits,
                    )
            else:
                assert item.completion_index is not None
                completion = request.completions[item.completion_index]
                self.process_generation_item(completion, item, output_count)
                self.finalize_request_if_ready(request)

    def maybe_fill_draft_tokens(self, completion: Completion) -> None:
        if (
            self.model.draft_provider is None
            or completion.finished
            or completion.pending_finish_reason is not None
            or completion.draft_tokens
        ):
            return
        remaining_tokens = completion.max_total_tokens - completion.total_tokens
        if remaining_tokens <= 0:
            return
        input_ids = np.array(
            [*completion.prompt_tokens, *completion.completion_tokens],
            dtype=np.intc,
        )
        proposed = self.model.draft_provider.draft(input_ids)
        if proposed.size == 0:
            return
        limited = [int(token) for token in proposed[:remaining_tokens]]
        if not limited:
            return
        completion.draft_tokens = limited
        self.speculative_stats["draft_proposals"] += 1
        self.speculative_stats["draft_tokens_proposed"] += len(limited)

    def process_generation_item(
        self,
        completion: Completion,
        item: CompletionScheduler.BatchItem,
        output_count: int,
    ) -> None:
        if completion.finished:
            return
        raw_row_indices = [
            self.output_arg(output_index, output_count)
            for output_index in item.output_indices
        ]
        if any(row_index is None for row_index in raw_row_indices):
            raise RuntimeError("generation rows are required")
        row_indices: List[int] = []
        for row_index in raw_row_indices:
            assert row_index is not None
            row_indices.append(int(row_index))
        if completion.pending_finish_reason is not None:
            self.checkpoint_logits[completion.seq_id] = self.model.logits(row_indices[-1])
            completion.pending_input_tokens = completion.pending_input_tokens[item.pending_count :]
            finish_reason: str = completion.pending_finish_reason
            completion.pending_finish_reason = None
            self.finish_completion(completion, finish_reason)
            return

        if item.pending_count:
            completion.pending_input_tokens = completion.pending_input_tokens[item.pending_count :]

        decoded_draft_tokens = item.tokens[item.pending_count :]
        accepted_draft_count = 0

        for draft_index, draft_token in enumerate(decoded_draft_tokens):
            row_index = row_indices[draft_index]
            logits = self.model.logits(row_index)
            self.checkpoint_logits[completion.seq_id] = logits
            sampled_token = completion.sampler.sample(self.model.ctx, row_index)
            if sampled_token != draft_token:
                rejected = len(completion.draft_tokens)
                if rejected > 0:
                    self.speculative_stats["draft_tokens_rejected"] += rejected
                keep_len = item.start_pos + item.pending_count + accepted_draft_count
                self.truncate_sequence(completion.seq_id, keep_len)
                completion.draft_tokens.clear()
                record = Token.from_logits(
                    model=self.model,
                    formatter=self.formatter,
                    prev_tokens=[*completion.prompt_tokens, *completion.completion_tokens],
                    prev_text_bytes=completion.detokenized_prefix_bytes,
                    token=sampled_token,
                    logits=logits,
                    logprobs_count=completion.logprobs,
                    need_token_logprob=completion.needs_token_logprob,
                )
                mismatch_finish_reason: Optional[str] = self.handle_completion_token(
                    completion,
                    sampled_token,
                    record,
                    decoded=False,
                )
                if mismatch_finish_reason is not None:
                    completion.pending_finish_reason = mismatch_finish_reason
                return
            record = Token.from_logits(
                model=self.model,
                formatter=self.formatter,
                prev_tokens=[*completion.prompt_tokens, *completion.completion_tokens],
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=draft_token,
                logits=logits,
                logprobs_count=completion.logprobs,
                need_token_logprob=completion.needs_token_logprob,
            )
            accepted_finish_reason: Optional[str] = self.handle_completion_token(
                completion,
                draft_token,
                record,
                decoded=True,
            )
            accepted_draft_count += 1
            self.speculative_stats["draft_tokens_accepted"] += 1
            if accepted_finish_reason is not None:
                rejected = max(0, len(completion.draft_tokens) - accepted_draft_count)
                if rejected > 0:
                    self.speculative_stats["draft_tokens_rejected"] += rejected
                keep_len = item.start_pos + item.pending_count + accepted_draft_count
                self.truncate_sequence(completion.seq_id, keep_len)
                completion.draft_tokens.clear()
                self.finish_completion(completion, accepted_finish_reason)
                return

        if accepted_draft_count:
            completion.draft_tokens = completion.draft_tokens[accepted_draft_count:]

        final_row_index = row_indices[-1]
        final_logits = self.model.logits(final_row_index)
        self.checkpoint_logits[completion.seq_id] = final_logits
        next_token = completion.sampler.sample(self.model.ctx, final_row_index)
        if completion.draft_tokens and next_token != completion.draft_tokens[0]:
            self.speculative_stats["draft_tokens_rejected"] += len(completion.draft_tokens)
            completion.draft_tokens.clear()
        elif completion.draft_tokens and next_token == completion.draft_tokens[0]:
            completion.draft_tokens = completion.draft_tokens[1:]
            self.speculative_stats["draft_tokens_accepted"] += 1

        record = Token.from_logits(
            model=self.model,
            formatter=self.formatter,
            prev_tokens=[*completion.prompt_tokens, *completion.completion_tokens],
            prev_text_bytes=completion.detokenized_prefix_bytes,
            token=next_token,
            logits=final_logits,
            logprobs_count=completion.logprobs,
            need_token_logprob=completion.needs_token_logprob,
        )
        final_finish_reason: Optional[str] = self.handle_completion_token(
            completion,
            next_token,
            record,
            decoded=False,
        )
        if final_finish_reason is not None:
            completion.pending_finish_reason = final_finish_reason

    def maybe_save_prompt_checkpoint(self, request: CompletionRequest) -> None:
        if (
            not self.model.exact_checkpoints_only
            or request.prompt_checkpoint_saved
            or request.base_seq_id is None
            or not request.prompt_tokens
            or request.prompt_logits is None
            or not self.unused_sequences
        ):
            return
        checkpoint_seq_id = self.unused_sequences.pop()
        copy_p0 = 0
        copy_p1 = len(request.prompt_tokens)
        if not self.model.kv_unified:
            copy_p0 = -1
            copy_p1 = -1
        llama_cpp.llama_memory_seq_cp(
            self.model.mem,
            request.base_seq_id,
            checkpoint_seq_id,
            copy_p0,
            copy_p1,
        )
        self.radix_trie.copy(request.base_seq_id, checkpoint_seq_id, len(request.prompt_tokens))
        self.sequence_history.copy(
            request.base_seq_id,
            checkpoint_seq_id,
            self.radix_trie.length(request.base_seq_id),
            len(request.prompt_tokens),
        )
        self.checkpoint_logits[checkpoint_seq_id] = request.prompt_logits
        self.free_sequences[checkpoint_seq_id] = None
        self.free_sequences.move_to_end(checkpoint_seq_id)
        request.prompt_checkpoint_saved = True

    @staticmethod
    def output_arg(output_index: Optional[int], output_count: int) -> Optional[int]:
        if output_index is None:
            return None
        return output_index - output_count

    @staticmethod
    def last_output_index(output_indices: Sequence[Optional[int]]) -> Optional[int]:
        for output_index in reversed(output_indices):
            if output_index is not None:
                return output_index
        return None

    def start_completions(
        self,
        request: CompletionRequest,
        prompt_row_index: Optional[int],
        prompt_logits: Optional[np.ndarray] = None,
    ) -> None:
        if request.completions:
            return
        assert request.base_seq_id is not None
        prompt_tokens = list(request.prompt_tokens)
        prompt_text = request.prompt_text
        if request.payload.stop is None:
            stop_sequences: List[bytes] = []
        elif isinstance(request.payload.stop, str):
            stop_sequences = [request.payload.stop.encode("utf-8")]
        else:
            stop_sequences = [item.encode("utf-8") for item in request.payload.stop]
        logit_bias = (
            {int(token): float(bias) for token, bias in request.payload.logit_bias.items()}
            if request.payload.logit_bias
            else None
        )
        prompt_text_bytes = self.model.detokenize(prompt_tokens) if prompt_tokens else b""
        for offset, seq_id in enumerate(request.completion_seq_ids):
            if offset > 0:
                if prompt_tokens:
                    self.memory_policy.copy_prompt_state(
                        request.base_seq_id,
                        seq_id,
                        len(prompt_tokens),
                    )
            sampler = Sampler(
                seed=(request.payload.seed or llama_cpp.LLAMA_DEFAULT_SEED) + offset,
                vocab=self.model.vocab,
                n_vocab=self.model.n_vocab,
                top_p=request.payload.top_p,
                temperature=request.payload.temperature,
                frequency_penalty=request.payload.frequency_penalty or 0.0,
                presence_penalty=request.payload.presence_penalty or 0.0,
                logit_bias=logit_bias,
                grammar_text=request.grammar_text,
                grammar_root=request.grammar_root,
            )
            request.completions.append(
                Completion(
                    request_id=request.id,
                    index=offset,
                    seq_id=seq_id,
                    sampler=sampler,
                    prompt_tokens=prompt_tokens,
                    prompt_text=prompt_text,
                    max_total_tokens=request.effective_max_len,
                    stop_sequences=stop_sequences,
                    logprobs=request.payload.logprobs,
                    detokenized_prefix_bytes=bytearray(prompt_text_bytes),
                    rank_by_score=(
                        request.payload.best_of is not None
                        and request.payload.best_of > request.payload.n
                    ),
                )
            )
        if request.payload.max_tokens == 0 or request.effective_max_len == len(prompt_tokens):
            for completion in request.completions:
                self.finish_completion(completion, "length")
            self.finalize_request_if_ready(request)
            return
        if prompt_row_index is None:
            if prompt_logits is not None:
                for completion in request.completions:
                    self.sample_completion_from_logits(completion, prompt_logits)
                return
            raise RuntimeError("prompt row is required to start generation")
        for completion in request.completions:
            self.sample_completion(completion, prompt_row_index)

    def sample_completion(self, completion: Completion, row_index: Optional[int]) -> None:
        if completion.finished:
            return
        if row_index is None:
            raise RuntimeError("missing logits row")
        token = completion.sampler.sample(self.model.ctx, row_index)
        prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
        if self.model.store_logits or completion.needs_token_logprob:
            logits = self.model.logits(row_index)
            if self.model.store_logits:
                self.checkpoint_logits[completion.seq_id] = logits
            record = Token.from_logits(
                model=self.model,
                formatter=self.formatter,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=token,
                logits=logits,
                logprobs_count=completion.logprobs,
                need_token_logprob=completion.needs_token_logprob,
            )
        else:
            record = Token.from_token(
                model=self.model,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=token,
            )
        finish_reason = self.handle_completion_token(
            completion,
            token,
            record,
            decoded=False,
        )
        if finish_reason is not None:
            completion.pending_finish_reason = finish_reason

    def sample_completion_from_logits(
        self,
        completion: Completion,
        logits: np.ndarray,
    ) -> None:
        if completion.finished:
            return
        token = completion.sampler.sample_logits(logits)
        prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
        if self.model.store_logits or completion.needs_token_logprob:
            if self.model.store_logits:
                self.checkpoint_logits[completion.seq_id] = logits.copy()
            record = Token.from_logits(
                model=self.model,
                formatter=self.formatter,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=token,
                logits=logits,
                logprobs_count=completion.logprobs,
                need_token_logprob=completion.needs_token_logprob,
            )
        else:
            record = Token.from_token(
                model=self.model,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=token,
            )
        finish_reason = self.handle_completion_token(
            completion,
            token,
            record,
            decoded=False,
        )
        if finish_reason is not None:
            completion.pending_finish_reason = finish_reason

    def handle_completion_token(
        self,
        completion: Completion,
        token: int,
        record: Token,
        *,
        decoded: bool,
    ) -> Optional[str]:
        if record.token_logprob is not None:
            completion.score_sum += record.token_logprob
        rendered_start = len(completion.rendered_bytes)
        completion.completion_tokens.append(token)
        completion.token_records.append(record)
        completion.rendered_bytes.extend(record.text_bytes)
        completion.detokenized_prefix_bytes.extend(record.text_bytes)
        finish_reason: Optional[str] = None
        if llama_cpp.llama_vocab_is_eog(self.model.vocab, token):
            finish_reason = "stop"
        elif completion.total_tokens >= completion.max_total_tokens:
            finish_reason = "length"
        else:
            max_stop_length = completion.max_stop_sequence_length
            search_start = max(0, rendered_start - max_stop_length + 1)
            if any(
                completion.rendered_bytes.find(stop, search_start) != -1
                for stop in completion.stop_sequences
            ):
                finish_reason = "stop"
        if not decoded:
            completion.pending_input_tokens.append(token)
        if (
            completion.request_id in self.requests
            and self.requests[completion.request_id].payload.stream
            and finish_reason is None
        ):
            self.flush_stream_updates(completion, finish_reason=None)
        if not decoded and finish_reason is None:
            self.maybe_fill_draft_tokens(completion)
        return finish_reason

    def flush_stream_updates(
        self,
        completion: Completion,
        finish_reason: Optional[str],
    ) -> None:
        request = self.requests[completion.request_id]
        for payload in self.formatter.stream_completion_chunks(
            request,
            completion,
            finish_reason,
        ):
            if request.on_stream_chunk is not None:
                request.on_stream_chunk(payload)

    def finish_completion(self, completion: Completion, finish_reason: str) -> None:
        if completion.finished:
            return
        completion.finished = True
        completion.finish_reason = finish_reason
        completion.pending_input_tokens.clear()
        completion.draft_tokens.clear()
        request = self.requests[completion.request_id]
        if request.payload.stream:
            self.flush_stream_updates(completion, finish_reason=finish_reason)
            if request.on_stream_chunk is not None:
                request.on_stream_chunk(
                    self.formatter.completion_finish_chunk(
                        request,
                        completion,
                        finish_reason,
                    )
                )

    def finalize_request_if_ready(self, request: CompletionRequest) -> None:
        if not request.completions or not all(completion.finished for completion in request.completions):
            return
        selected = request.selected_completions()
        result = self.formatter.build_completion_response(request, selected)
        self.release_request(request)
        if request.on_done is not None:
            request.on_done(result)

    def truncate_sequence(self, seq_id: int, keep_len: int) -> None:
        current_len = self.radix_trie.length(seq_id)
        if current_len <= keep_len:
            return
        llama_cpp.llama_memory_seq_rm(self.model.mem, seq_id, keep_len, -1)
        self.radix_trie.truncate(seq_id, keep_len)
        self.sequence_history.truncate(seq_id, current_len, keep_len)
        self.checkpoint_logits.pop(seq_id, None)

    def truncate_free_sequence(self, seq_id: int, keep_len: int) -> None:
        if seq_id not in self.free_sequences:
            return
        self.truncate_sequence(seq_id, keep_len)
        self.free_sequences.move_to_end(seq_id)

    def delete_free_sequence(self, seq_id: int) -> None:
        if seq_id not in self.free_sequences:
            return
        self.truncate_sequence(seq_id, 0)
        del self.free_sequences[seq_id]
        self.checkpoint_logits.pop(seq_id, None)
        self.unused_sequences.append(seq_id)

    def release_request(self, request: CompletionRequest) -> None:
        for completion in request.completions:
            completion.sampler.close()
        for seq_id in request.completion_seq_ids:
            if seq_id in self.claimed_sequences:
                self.claimed_sequences.remove(seq_id)
                self.free_sequences[seq_id] = None
                self.free_sequences.move_to_end(seq_id)
        self.active_request_ids.discard(request.id)
        self.requests.pop(request.id, None)

    def fail_request(self, request: CompletionRequest, exc: BaseException) -> None:
        if request.id in self.active_request_ids or request.admitted:
            self.release_request(request)
        else:
            self.requests.pop(request.id, None)
        if request.on_error is not None:
            request.on_error(exc)

    def finalize_cancelled(self) -> bool:
        finalized = False
        for request in list(self.pending_requests):
            if request.cancelled:
                self.pending_requests.remove(request)
                self.fail_request(request, CompletionRequestCancelledError("request cancelled"))
                finalized = True
        for request_id in list(self.active_request_ids):
            request = self.requests[request_id]
            if request.cancelled:
                self.fail_request(request, CompletionRequestCancelledError("request cancelled"))
                finalized = True
        return finalized


class CompletionService:
    def __init__(self, scheduler: CompletionScheduler) -> None:
        self.scheduler = scheduler
        self.formatter = scheduler.formatter
        self.condition = threading.Condition()
        self.commands: Deque[Callable[[], None]] = deque()
        self.closed = False
        self.worker = threading.Thread(
            target=self.run_loop,
            name="completion-service",
            daemon=True,
        )
        self.worker.start()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join()
        self.scheduler.close()

    def enqueue(self, command: Callable[[], None]) -> None:
        with self.condition:
            if self.closed:
                raise RuntimeError("completion service closed")
            self.commands.append(command)
            self.condition.notify_all()

    def run_loop(self) -> None:
        while True:
            with self.condition:
                while not self.closed and not self.commands and self.scheduler.is_idle():
                    self.condition.wait()
                if self.closed and not self.commands and self.scheduler.is_idle():
                    return
                commands = list(self.commands)
                self.commands.clear()
            for command in commands:
                command()
            progressed = self.scheduler.step()
            if progressed:
                continue
            with self.condition:
                if self.closed and not self.commands and self.scheduler.is_idle():
                    return
                if not self.commands:
                    self.condition.wait(timeout=0.01)

    def submit_request(
        self,
        request: CompletionRequest,
    ) -> Tuple[CompletionStream, Callable[[], None]]:
        mailbox: "queue.Queue[object]" = queue.Queue()
        done = threading.Event()
        cancelled = threading.Event()
        sentinel = object()
        result_box: Dict[str, OpenAICompletion] = {}
        error_box: Dict[str, BaseException] = {}

        def on_stream_chunk(chunk: CompletionChunk) -> None:
            mailbox.put(chunk)

        def on_done(result: OpenAICompletion) -> None:
            result_box["result"] = result
            done.set()
            mailbox.put(sentinel)

        def on_error(exc: BaseException) -> None:
            error_box["error"] = exc
            done.set()
            mailbox.put(sentinel)

        request.on_stream_chunk = on_stream_chunk
        request.on_done = on_done
        request.on_error = on_error

        def cancel() -> None:
            if cancelled.is_set():
                return
            cancelled.set()
            try:
                def cancel_request() -> None:
                    self.scheduler.cancel(request.id)

                self.enqueue(cancel_request)
            except RuntimeError:
                pass

        def stream() -> CompletionStream:
            try:
                while True:
                    item = mailbox.get()
                    if item is sentinel:
                        break
                    yield cast(CompletionChunk, item)
            finally:
                if not done.is_set():
                    cancel()
            if "error" in error_box:
                raise error_box["error"]
            result = result_box.get("result")
            if result is None:
                raise RuntimeError("missing completion result")
            return result

        def submit_request() -> None:
            self.scheduler.submit_request(request)

        self.enqueue(submit_request)
        return stream(), cancel

    def submit(
        self,
        payload: CreateCompletionRequest,
    ) -> Tuple[CompletionStream, Callable[[], None]]:
        request = CompletionRequest.from_payload(
            model=self.scheduler.model,
            payload=payload,
        )
        return self.submit_request(request)


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def watch_http_disconnect(
        http_request: Request,
        cancel: Callable[[], None],
    ) -> None:
        while True:
            message = await http_request.receive()
            if message["type"] == "http.disconnect":
                cancel()
                return

    async def disconnected_cancelled_response(
        http_request: Request,
        exc: BaseException,
    ) -> Optional[Response]:
        if not isinstance(exc, CompletionRequestCancelledError):
            return None
        if await http_request.is_disconnected():
            return Response(status_code=204)
        return None

    async def disconnected_cancelled_response_or_raise(
        http_request: Request,
        exc: BaseException,
    ) -> Response:
        response = await disconnected_cancelled_response(http_request, exc)
        if response is None:
            raise exc
        return response

    def bad_request(exc: CompletionRequestValidationError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    async def collect_completion_result(
        formatter: OpenAIFormatter,
        http_request: Request,
        stream: CompletionStream,
        cancel: Callable[[], None],
    ) -> OpenAICompletion | Response:
        disconnect_task = asyncio.create_task(
            watch_http_disconnect(http_request, cancel)
        )
        try:
            return await asyncio.to_thread(formatter.collect_completion, stream)
        except asyncio.CancelledError:
            cancel()
            raise
        except BaseException as exc:
            return await disconnected_cancelled_response_or_raise(http_request, exc)
        finally:
            disconnect_task.cancel()
            stream.close()

    async def collect_completion_results(
        formatter: OpenAIFormatter,
        http_request: Request,
        submissions: Sequence[
            Tuple[
                CompletionStream,
                Callable[[], None],
            ]
        ],
    ) -> List[OpenAICompletion] | Response:
        streams = [stream for stream, _ in submissions]
        cancel_all = [cancel for _, cancel in submissions]

        def cancel_all_requests() -> None:
            for cancel in cancel_all:
                cancel()

        disconnect_task = asyncio.create_task(
            watch_http_disconnect(
                http_request,
                cancel_all_requests,
            )
        )
        try:
            return await asyncio.gather(
                *(
                    asyncio.to_thread(formatter.collect_completion, stream)
                    for stream in streams
                )
            )
        except asyncio.CancelledError:
            cancel_all_requests()
            raise
        except BaseException as exc:
            return await disconnected_cancelled_response_or_raise(http_request, exc)
        finally:
            disconnect_task.cancel()
            for stream in streams:
                stream.close()

    async def stream_sse_chunks(
        formatter: OpenAIFormatter,
        http_request: Request,
        stream: CompletionStream,
        cancel: Callable[[], None],
        chunk_payloads: Callable[
            [CompletionChunk], Iterable[BaseModel | Dict[str, Any]]
        ],
    ) -> AsyncIterator[bytes]:
        disconnect_task = asyncio.create_task(
            watch_http_disconnect(http_request, cancel)
        )
        try:
            while True:
                done, chunk = await asyncio.to_thread(
                    formatter.next_stream_chunk,
                    stream,
                )
                if done:
                    break
                assert chunk is not None
                for payload in chunk_payloads(chunk):
                    yield formatter.encode_sse_payload(payload)
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            cancel()
            raise
        except BaseException as exc:
            cancel()
            if (
                isinstance(exc, CompletionRequestCancelledError)
                and await http_request.is_disconnected()
            ):
                return
            raise
        finally:
            disconnect_task.cancel()

    async def stream_sse_outputs(
        formatter: OpenAIFormatter,
        http_request: Request,
        stream: CompletionStream,
        cancel: Callable[[], None],
        *,
        initial_payloads: Iterable[BaseModel | Dict[str, Any]],
        chunk_payloads: Callable[
            [CompletionChunk], Iterable[BaseModel | Dict[str, Any]]
        ],
        done_payloads: Callable[
            [Optional[OpenAICompletion]], Iterable[BaseModel | Dict[str, Any]]
        ],
    ) -> AsyncIterator[bytes]:
        disconnect_task = asyncio.create_task(
            watch_http_disconnect(http_request, cancel)
        )
        try:
            for payload in initial_payloads:
                yield formatter.encode_sse_payload(payload)
            while True:
                done, chunk, result = await asyncio.to_thread(
                    formatter.next_stream_output,
                    stream,
                )
                if done:
                    for payload in done_payloads(result):
                        yield formatter.encode_sse_payload(payload)
                    break
                assert chunk is not None
                for payload in chunk_payloads(chunk):
                    yield formatter.encode_sse_payload(payload)
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            cancel()
            raise
        except BaseException as exc:
            cancel()
            if (
                isinstance(exc, CompletionRequestCancelledError)
                and await http_request.is_disconnected()
            ):
                return
            raise
        finally:
            disconnect_task.cancel()

    async def stream_websocket_responses(
        websocket: WebSocket,
        formatter: OpenAIFormatter,
        stream: CompletionStream,
        cancel: Callable[[], None],
        *,
        initial_payloads: Iterable[BaseModel | Dict[str, Any]],
        chunk_payloads: Callable[
            [CompletionChunk], Iterable[BaseModel | Dict[str, Any]]
        ],
        done_payloads: Callable[
            [Optional[OpenAICompletion]], Iterable[BaseModel | Dict[str, Any]]
        ],
    ) -> None:
        try:
            for payload in initial_payloads:
                await websocket.send_json(payload)
            while True:
                done, chunk, result = await asyncio.to_thread(
                    formatter.next_stream_output,
                    stream,
                )
                if done:
                    for payload in done_payloads(result):
                        await websocket.send_json(payload)
                    break
                assert chunk is not None
                for payload in chunk_payloads(chunk):
                    await websocket.send_json(payload)
        finally:
            stream.close()

    @app.post("/v1/completions")
    async def create_completion(http_request: Request, body: CreateCompletionRequest):
        service: CompletionService = app.state.service
        formatter = service.formatter
        prompts = body.normalized_prompt()
        if len(prompts) > 1:
            if body.stream:
                raise HTTPException(
                    status_code=400,
                    detail="streaming does not support multiple prompts",
                )
            try:
                submissions = [
                    service.submit(body.model_copy(update={"prompt": prompt}))
                    for prompt in prompts
                ]
            except CompletionRequestValidationError as exc:
                raise bad_request(exc) from exc
            results = await collect_completion_results(
                formatter, http_request, submissions
            )
            if isinstance(results, Response):
                return results
            return JSONResponse(
                formatter.aggregate_completion_results(results).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
        try:
            stream, cancel = service.submit(
                body.model_copy(update={"prompt": prompts[0]})
            )
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        if body.stream:
            return StreamingResponse(
                stream_sse_chunks(
                    formatter,
                    http_request,
                    stream,
                    cancel,
                    lambda chunk: [chunk],
                ),
                media_type="text/event-stream",
            )
        result = await collect_completion_result(
            formatter, http_request, stream, cancel
        )
        if isinstance(result, Response):
            return result
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        http_request: Request, body: CreateChatCompletionRequest
    ):
        service: CompletionService = app.state.service
        formatter = service.formatter
        try:
            payload, prompt_text, generation_prompt, prompt_tokens, grammar_text, tool_name = (
                formatter.completion_request_from_chat_request(
                    body,
                )
            )
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        try:
            request = CompletionRequest.from_prepared(
                model=service.scheduler.model,
                payload=payload,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                grammar_text=grammar_text,
                chat_tool_name=tool_name,
            )
            stream, cancel = service.submit_request(request)
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        if body.stream:
            started_indices: set[int] = set()

            def chat_chunk_payloads(
                completion_chunk: CompletionChunk,
            ) -> Iterable[BaseModel | Dict[str, Any]]:
                return formatter.convert_completion_chunk_to_chat_chunks(
                    completion_chunk,
                    started_indices,
                    tool_name,
                    functions=body.functions,
                    tools=body.tools,
                    parsed_states=parsed_states,
                    generation_prompt=generation_prompt,
                )

            parsed_states: Dict[int, Dict[str, Any]] = {}
            return StreamingResponse(
                stream_sse_chunks(
                    formatter,
                    http_request,
                    stream,
                    cancel,
                    chat_chunk_payloads,
                ),
                media_type="text/event-stream",
            )
        completion = await collect_completion_result(
            formatter, http_request, stream, cancel
        )
        if isinstance(completion, Response):
            return completion
        chat_response = formatter.convert_completion_response_to_chat(
            completion,
            tool_name,
            functions=body.functions,
            tools=body.tools,
            generation_prompt=generation_prompt,
        )
        if isinstance(chat_response, BaseModel):
            return JSONResponse(
                chat_response.model_dump(mode="json", exclude_none=True)
            )
        return JSONResponse(chat_response)

    @app.post("/v1/responses")
    async def create_response(http_request: Request, body: CreateResponseRequest):
        service: CompletionService = app.state.service
        formatter = service.formatter
        try:
            chat_body = formatter.chat_request_from_responses_request(body)
            payload, prompt_text, generation_prompt, prompt_tokens, grammar_text, tool_name = (
                formatter.completion_request_from_chat_request(
                    chat_body,
                )
            )
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        try:
            request = CompletionRequest.from_prepared(
                model=service.scheduler.model,
                payload=payload,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                grammar_text=grammar_text,
                chat_tool_name=tool_name,
            )
            stream, cancel = service.submit_request(request)
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        if body.stream:
            started_indices: set[int] = set()
            parsed_states: Dict[int, ResponseParser] = {}
            stream_state = OpenAIFormatter.ResponsesStream(
                body=body,
                response_id="resp_" + request.id,
                created_at=float(request.created),
                model=service.scheduler.model.model_path,
            )

            def response_chunk_payloads(
                completion_chunk: CompletionChunk,
            ) -> Iterable[BaseModel | Dict[str, Any]]:
                chat_chunks = formatter.convert_completion_chunk_to_chat_chunks(
                    completion_chunk,
                    started_indices,
                    tool_name,
                    tools=chat_body.tools,
                    parsed_states=parsed_states,
                    generation_prompt=generation_prompt,
                )
                payloads: List[BaseModel | Dict[str, Any]] = []
                for chat_chunk in chat_chunks:
                    payloads.extend(
                        formatter.convert_chat_chunk_to_response_events(
                            chat_chunk,
                            stream_state,
                        )
                    )
                return payloads

            return StreamingResponse(
                stream_sse_outputs(
                    formatter,
                    http_request,
                    stream,
                    cancel,
                    initial_payloads=formatter.start_response_stream(stream_state),
                    chunk_payloads=response_chunk_payloads,
                    done_payloads=lambda completion: (
                        formatter.response_stream_terminal_events(
                            stream_state,
                            completion,
                        )
                    ),
                ),
                media_type="text/event-stream",
            )
        completion = await collect_completion_result(
            formatter, http_request, stream, cancel
        )
        if isinstance(completion, Response):
            return completion
        return JSONResponse(
            formatter.convert_completion_response_to_response(
                completion,
                body,
                tool_name,
                tools=chat_body.tools,
                generation_prompt=generation_prompt,
            )
        )

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket):
        await websocket.accept()
        service: CompletionService = app.state.service
        formatter = service.formatter
        # HTTP /v1/responses remains stateless. The websocket transport keeps
        # per-connection response history so Codex can use previous_response_id
        # within a single live session.
        websocket_response_history: Dict[str, ResponsesWebSocketState] = {}

        def websocket_request_with_ephemeral_history(
            ws_body: ResponseCreateWebSocketRequest,
        ) -> CreateResponseRequest:
            body = ws_body.to_create_response_request()
            previous_response_id = body.previous_response_id
            if previous_response_id is None:
                return body
            prior_state = websocket_response_history.get(previous_response_id)
            if prior_state is None:
                raise CompletionRequestValidationError(
                    f"unknown previous_response_id: {previous_response_id}"
                )
            current_items = formatter._clone_response_input_items(body.input)
            replay_items = formatter._clone_response_input_items(
                prior_state.input_items
            )
            for item in prior_state.output_items:
                normalized = formatter._normalize_response_output_item_for_input(item)
                if normalized is not None:
                    replay_items.append(normalized)
            replay_items.extend(current_items)
            return body.model_copy(
                update={
                    "input": replay_items,
                    "previous_response_id": None,
                }
            )

        async def send_error(message: str) -> None:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": message,
                    },
                }
            )

        try:
            while True:
                try:
                    payload = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                except Exception:
                    await send_error("invalid websocket request payload")
                    continue

                try:
                    ws_body = ResponseCreateWebSocketRequest.model_validate(payload)
                    body = websocket_request_with_ephemeral_history(ws_body)
                    if ws_body.generate is False:
                        body = body.model_copy(update={"max_output_tokens": 0})
                    chat_body = formatter.chat_request_from_responses_request(body)
                    (
                        completion_payload,
                        prompt_text,
                        generation_prompt,
                        prompt_tokens,
                        grammar_text,
                        tool_name,
                    ) = formatter.completion_request_from_chat_request(chat_body)
                    request = CompletionRequest.from_prepared(
                        model=service.scheduler.model,
                        payload=completion_payload,
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                        grammar_text=grammar_text,
                        chat_tool_name=tool_name,
                    )
                    stream, cancel = service.submit_request(request)
                except CompletionRequestValidationError as exc:
                    await send_error(str(exc))
                    continue

                started_indices: set[int] = set()
                parsed_states: Dict[int, ResponseParser] = {}
                stream_state = OpenAIFormatter.ResponsesStream(
                    body=body,
                    response_id="resp_" + request.id,
                    created_at=float(request.created),
                    model=service.scheduler.model.model_path,
                )

                def response_chunk_payloads(
                    completion_chunk: CompletionChunk,
                ) -> Iterable[BaseModel | Dict[str, Any]]:
                    chat_chunks = formatter.convert_completion_chunk_to_chat_chunks(
                        completion_chunk,
                        started_indices,
                        tool_name,
                        tools=chat_body.tools,
                        parsed_states=parsed_states,
                        generation_prompt=generation_prompt,
                    )
                    payloads: List[BaseModel | Dict[str, Any]] = []
                    for chat_chunk in chat_chunks:
                        payloads.extend(
                            formatter.convert_chat_chunk_to_response_events(
                                chat_chunk,
                                stream_state,
                            )
                        )
                    return payloads

                try:
                    await stream_websocket_responses(
                        websocket,
                        formatter,
                        stream,
                        cancel,
                        initial_payloads=formatter.start_response_stream(stream_state),
                        chunk_payloads=response_chunk_payloads,
                        done_payloads=lambda completion: (
                            formatter.response_stream_terminal_events(
                                stream_state,
                                completion,
                            )
                        ),
                    )
                    websocket_response_history[stream_state.response_id] = (
                        ResponsesWebSocketState(
                            input_items=formatter._clone_response_input_items(
                                body.input
                            ),
                            output_items=copy.deepcopy(stream_state.output),
                        )
                    )
                except WebSocketDisconnect:
                    cancel()
                    break
                except BaseException as exc:
                    cancel()
                    await send_error(str(exc))
        except WebSocketDisconnect:
            pass

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        service = app.state.service
        model = service.scheduler.model
        model_id = getattr(model, "model_alias", None) or model.model_path
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "llama-cpp-python",
                }
            ],
        }

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    return app


APP = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config-file", required=True)
    args = parser.parse_args()
    config = ConfigFile.load(args.config_file)
    model_path = config.model.resolve_model_path()
    model = Model(
        model_path=model_path,
        model_alias=config.model.alias,
        chat_template=config.model.chat_template,
        n_gpu_layers=config.model.n_gpu_layers,
        split_mode=config.model.split_mode,
        main_gpu=config.model.main_gpu,
        tensor_split=config.model.tensor_split,
        vocab_only=config.model.vocab_only,
        use_mmap=config.model.use_mmap,
        use_mlock=config.model.use_mlock,
        kv_overrides=config.model.kv_overrides,
        n_ctx=config.model.n_ctx,
        n_batch=config.model.n_batch,
        n_ubatch=config.model.n_ubatch,
        n_seq_max=config.model.n_seq_max,
        n_threads=config.model.threads,
        n_threads_batch=config.model.threads_batch,
        rope_scaling_type=config.model.rope_scaling_type,
        pooling_type=config.model.pooling_type,
        attention_type=config.model.attention_type,
        rope_freq_base=config.model.rope_freq_base,
        rope_freq_scale=config.model.rope_freq_scale,
        yarn_ext_factor=config.model.yarn_ext_factor,
        yarn_attn_factor=config.model.yarn_attn_factor,
        yarn_beta_fast=config.model.yarn_beta_fast,
        yarn_beta_slow=config.model.yarn_beta_slow,
        yarn_orig_ctx=config.model.yarn_orig_ctx,
        offload_kqv=config.model.offload_kqv,
        flash_attn=config.model.flash_attn,
        op_offload=config.model.op_offload,
        swa_full=config.model.swa_full,
        no_perf=config.model.no_perf,
        type_k=config.model.type_k,
        type_v=config.model.type_v,
        kv_unified=config.model.kv_unified,
        max_seq_len=config.model.max_seq_len,
        max_output_tokens=config.model.max_output_tokens,
        draft_model=config.model.draft_model,
        draft_model_num_pred_tokens=config.model.draft_model_num_pred_tokens,
        draft_model_max_ngram_size=config.model.draft_model_max_ngram_size,
        response_schema=config.model.response_schema,
        store_logits=config.model.store_logits,
    )
    scheduler = CompletionScheduler(model)
    APP.state.service = CompletionService(scheduler)
    try:
        uvicorn.run(
            APP, host=config.server.host, port=config.server.port, log_level="info"
        )
    finally:
        APP.state.service.close()


if __name__ == "__main__":
    main()
