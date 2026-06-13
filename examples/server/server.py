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
#   "safetensors",
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
import base64
import hashlib
import binascii
import asyncio
import argparse
import threading
import multiprocessing
import copy
import shutil
import inspect
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

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
    Protocol,
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
from llama_cpp import llama_cpp_ext  # noqa: E402
from llama_cpp import mtmd_cpp  # noqa: E402


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
    @dataclass(frozen=True)
    class BuiltinRule:
        content: str
        deps: Sequence[str] = ()

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
    __slots__ = ("_position_lengths", "_root", "_tails", "size")

    @dataclass
    class Node:
        token: Optional[int] = None
        parent: Optional["SequenceHistory.Node"] = None
        children: Dict[int, "SequenceHistory.Node"] = field(default_factory=dict)
        sequences: set[int] = field(default_factory=set)
        position_increment: int = 1

    def __init__(self) -> None:
        self._root = SequenceHistory.Node()
        self._tails: Dict[int, SequenceHistory.Node] = {}
        self._position_lengths: Dict[int, int] = {}
        self.size = 0

    def extend(
        self,
        sequence_id: int,
        tokens: Sequence[int],
        position_increments: Optional[Sequence[int]] = None,
    ) -> None:
        assert sequence_id >= 0
        if position_increments is None:
            position_increments = [1] * len(tokens)
        assert len(position_increments) == len(tokens)
        node = self._tails.get(sequence_id, self._root)
        position_length = self._position_lengths.get(sequence_id, 0)
        for token, position_increment in zip(tokens, position_increments):
            position_increment = max(0, int(position_increment))
            child = node.children.get(sequence_id)
            if child is None:
                child = SequenceHistory.Node(
                    token=token,
                    parent=node,
                    position_increment=position_increment,
                )
                node.children[sequence_id] = child
                self.size += 1
            else:
                assert child.parent is node
                assert child.token == token
                assert child.position_increment == position_increment
            child.sequences.add(sequence_id)
            position_length += position_increment
            node = child
        if node is self._root:
            self._tails.pop(sequence_id, None)
            self._position_lengths.pop(sequence_id, None)
        else:
            self._tails[sequence_id] = node
            self._position_lengths[sequence_id] = position_length

    def position_length(self, sequence_id: int) -> int:
        return self._position_lengths.get(sequence_id, 0)

    def position_length_for_prefix(self, sequence_id: int, keep_len: int) -> int:
        if keep_len <= 0 or sequence_id not in self._tails:
            return 0
        node = self._tails[sequence_id]
        increments: List[int] = []
        while node is not self._root:
            increments.append(node.position_increment)
            parent = node.parent
            assert parent is not None
            node = parent
        increments.reverse()
        assert keep_len <= len(increments)
        return sum(increments[:keep_len])

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
        position_length = 0
        for child in reversed(path):
            parent.children[dest_sequence_id] = child
            child.sequences.add(dest_sequence_id)
            position_length += child.position_increment
            parent = child
        if keep_len == 0:
            self._tails.pop(dest_sequence_id, None)
            self._position_lengths.pop(dest_sequence_id, None)
        else:
            self._tails[dest_sequence_id] = path[0]
            self._position_lengths[dest_sequence_id] = position_length

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
        position_length = self._position_lengths.get(sequence_id, 0)
        while node is not self._root and drop > 0:
            node.sequences.remove(sequence_id)
            parent = node.parent
            assert parent is not None
            child = parent.children.get(sequence_id)
            if child is node:
                del parent.children[sequence_id]
            if not node.sequences:
                self.size -= 1
            position_length -= node.position_increment
            node = parent
            drop -= 1
        if node is self._root:
            self._tails.pop(sequence_id, None)
            self._position_lengths.pop(sequence_id, None)
        else:
            self._tails[sequence_id] = node
            self._position_lengths[sequence_id] = max(0, position_length)


class DraftProvider(abc.ABC):
    @abc.abstractmethod
    def draft(
        self,
        input_ids: np.ndarray,
        /,
        *,
        seq_id: int,
        max_tokens: Optional[int],
    ) -> np.ndarray:
        raise NotImplementedError()

    def can_draft(self, input_length: int, /, *, seq_id: int) -> bool:
        return True

    def process(self, batch: Any, /) -> None:
        pass

    def accept(self, seq_id: int, accepted_draft_tokens: int) -> None:
        pass

    def truncate(self, seq_id: int, keep_len: int) -> None:
        pass

    def copy_sequence(
        self,
        source_seq_id: int,
        dest_seq_id: int,
        p0: int,
        p1: int,
    ) -> None:
        pass

    def set_target_processing_enabled(self, enabled: bool) -> None:
        pass

    def close(self) -> None:
        pass


class PromptLookupDecoding(DraftProvider):
    def __init__(self, max_ngram_size: int = 2, num_pred_tokens: int = 10) -> None:
        self._max_ngram_size = max_ngram_size
        self._num_pred_tokens = num_pred_tokens

    def draft(
        self,
        input_ids: np.ndarray,
        /,
        *,
        seq_id: int,
        max_tokens: Optional[int],
    ) -> np.ndarray:
        input_length = input_ids.shape[0]
        if input_length < 2:
            return np.array([], dtype=np.intc)
        num_pred_tokens = self._num_pred_tokens
        if max_tokens is not None:
            num_pred_tokens = min(num_pred_tokens, max_tokens)
        if num_pred_tokens <= 0:
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
                end = min(start + num_pred_tokens, input_length)
                if start < end:
                    return input_ids[start:end].astype(np.intc, copy=False)
        return np.array([], dtype=np.intc)


class MTPDraftProvider(DraftProvider):
    batched_draft = True
    sampled_batch_draft = True

    @dataclass
    class DraftManyState:
        result_index: int
        seq_id: int
        first_pos: int
        keep_len: int
        n_predict: int
        token: int
        drafted: List[int]
        embedding: np.ndarray
        cache_key: Tuple[Tuple[int, ...], int]

    def __init__(
        self,
        *,
        model: "Model",
        draft_model: Any,
        context_params: Any,
        num_pred_tokens: int,
        top_k: int,
        p_min: float,
    ) -> None:
        self.target_ctx = model.ctx
        self.model = draft_model
        self.n_seq_max = model.n_seq_max
        self.n_vocab = model.n_vocab
        self.n_embd = int(llama_cpp.llama_model_n_embd_out(self.model))
        if self.n_embd <= 0:
            self.n_embd = int(llama_cpp.llama_model_n_embd(self.model))
        if self.n_embd != model.n_embd:
            raise RuntimeError(
                "MTP draft model output embedding size must match target model "
                f"embedding size ({self.n_embd} != {model.n_embd})"
            )
        self.num_pred_tokens = max(0, int(num_pred_tokens))
        self.top_k = max(1, int(top_k))
        self.p_min = max(0.0, min(1.0, float(p_min)))
        self.ctx = llama_cpp.llama_init_from_model(self.model, context_params)
        if self.ctx is None:
            raise RuntimeError("failed to create MTP draft context")
        ctx_other = llama_cpp_ext.llama_get_ctx_other(self.ctx)
        self.is_mem_shared = bool(ctx_other and ctx_other == self.target_ctx)
        self.sampled_batch_draft = not self.is_mem_shared
        self.n_batch = int(llama_cpp.llama_n_batch(self.ctx))
        mem = llama_cpp.llama_get_memory(self.ctx)
        if mem is None:
            llama_cpp.llama_free(self.ctx)
            raise RuntimeError("failed to access MTP draft memory")
        self.mem = mem
        self.batch = llama_cpp.llama_batch_init(self.n_batch, self.n_embd, 1)
        self._batch_tokens = (llama_cpp.llama_token * self.n_batch)()
        self.batch.token = self._batch_tokens
        self.batch_embeddings = np.ctypeslib.as_array(
            self.batch.embd,
            shape=(self.n_batch * self.n_embd,),
        )
        self._samplers: List[Any] = []
        self.pending_h = np.zeros((self.n_seq_max, self.n_embd), dtype=np.float32)
        self.verify_h: List[np.ndarray] = [
            np.empty((0, self.n_embd), dtype=np.float32)
            for _ in range(self.n_seq_max)
        ]
        self.verify_h_pos: List[List[int]] = [[] for _ in range(self.n_seq_max)]
        self.verify_h_rows = [0] * self.n_seq_max
        self.ready = [False] * self.n_seq_max
        self.ready_pos = [0] * self.n_seq_max
        self.context_pos = [0] * self.n_seq_max
        self.decode_seconds_total = 0.0
        self.decode_calls_total = 0
        self.decode_tokens_total = 0
        self.decode_failures_total = 0
        self.target_processing_enabled = False
        self.set_target_processing_enabled(True)
        llama_cpp_ext.llama_set_embeddings_nextn(
            self.ctx,
            True,
            True,
        )
        self._init_samplers()

    def _init_samplers(self) -> None:
        for seq_id in range(self.n_seq_max):
            params = llama_cpp.llama_sampler_chain_default_params()
            params.no_perf = True
            sampler = llama_cpp.llama_sampler_chain_init(params)
            if self.top_k > 1:
                llama_cpp.llama_sampler_chain_add(
                    sampler,
                    llama_cpp.llama_sampler_init_top_k(self.top_k),
                )
            llama_cpp.llama_sampler_chain_add(
                sampler,
                llama_cpp.llama_sampler_init_greedy(),
            )
            self._samplers.append(sampler)

    def _passes_p_min(self, output_index: int) -> bool:
        if self.p_min <= 0.0:
            return True
        logits_ptr = llama_cpp.llama_get_logits_ith(self.ctx, output_index)
        if not logits_ptr:
            return False
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.n_vocab,))
        n_values = min(self.top_k, self.n_vocab)
        if n_values <= 0:
            return False
        if n_values == self.n_vocab:
            values = logits
        else:
            top_indices = np.argpartition(logits, -n_values)[-n_values:]
            values = logits[top_indices]
        max_logit = float(np.max(values))
        weights = np.exp(values.astype(np.float64, copy=False) - max_logit)
        total = float(np.sum(weights))
        if total <= 0.0 or not math.isfinite(total):
            return False
        return 1.0 / total >= self.p_min

    def _sample_token(self, output_index: int = 0, *, seq_id: int = 0) -> Optional[int]:
        if seq_id < 0 or seq_id >= len(self._samplers):
            return None
        if not self._passes_p_min(output_index):
            return None
        sampler = self._samplers[seq_id]
        token = int(llama_cpp.llama_sampler_sample(sampler, self.ctx, output_index))
        if token == llama_cpp.LLAMA_TOKEN_NULL:
            return None
        return token

    def _reset_sampler(self, seq_id: int) -> None:
        if 0 <= seq_id < len(self._samplers):
            llama_cpp.llama_sampler_reset(self._samplers[seq_id])

    def close(self) -> None:
        self.set_target_processing_enabled(False)
        self.batch.token = ctypes.POINTER(llama_cpp.llama_token)()
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_free(self.ctx)
        for sampler in self._samplers:
            llama_cpp.llama_sampler_free(sampler)
        self._samplers.clear()

    def set_target_processing_enabled(self, enabled: bool) -> None:
        if self.target_processing_enabled == enabled:
            return
        llama_cpp_ext.llama_set_embeddings_nextn(
            self.target_ctx,
            enabled,
            False,
        )
        self.target_processing_enabled = enabled

    def _clear_batch(self) -> None:
        self.batch.n_tokens = 0

    def _batch_embeddings(self) -> np.ndarray:
        return self.batch_embeddings

    def _set_batch_embedding_row(
        self,
        row: int,
        embedding: Union[np.ndarray, ctypes.POINTER(ctypes.c_float)],
    ) -> None:
        row_start = row * self.n_embd
        row_end = row_start + self.n_embd
        if isinstance(embedding, np.ndarray):
            self._batch_embeddings()[row_start:row_end] = embedding
            return
        self._batch_embeddings()[row_start:row_end] = np.ctypeslib.as_array(
            embedding,
            shape=(self.n_embd,),
        )

    def _add_batch_token(
        self,
        *,
        token: int,
        pos: int,
        seq_id: int,
        logits: bool,
    ) -> None:
        slot = int(self.batch.n_tokens)
        if slot >= self.n_batch:
            raise RuntimeError("MTP draft batch capacity exceeded")
        self.batch.token[slot] = int(token)
        self.batch.pos[slot] = int(pos)
        self.batch.seq_id[slot][0] = int(seq_id)
        self.batch.n_seq_id[slot] = 1
        self.batch.logits[slot] = int(logits)
        self.batch.n_tokens += 1

    def _try_decode_batch(self) -> bool:
        n_tokens = int(self.batch.n_tokens)
        if n_tokens <= 0:
            return True
        started_at = time.perf_counter()
        result = int(llama_cpp.llama_decode(self.ctx, self.batch))
        self.decode_seconds_total += time.perf_counter() - started_at
        self.decode_calls_total += 1
        self.decode_tokens_total += n_tokens
        if result != 0:
            self.decode_failures_total += 1
            return False
        return True

    def _decode_batch(self) -> None:
        n_tokens = int(self.batch.n_tokens)
        if n_tokens <= 0:
            return
        started_at = time.perf_counter()
        result = int(llama_cpp.llama_decode(self.ctx, self.batch))
        self.decode_seconds_total += time.perf_counter() - started_at
        self.decode_calls_total += 1
        self.decode_tokens_total += n_tokens
        if result != 0:
            self.decode_failures_total += 1
            raise RuntimeError(f"MTP draft decode failed with code {result}")

    def metric_definitions(
        self,
    ) -> List[Tuple[str, str, str, Union[int, float]]]:
        decode_tokens_seconds = (
            self.decode_tokens_total / self.decode_seconds_total
            if self.decode_seconds_total > 0.0
            else 0.0
        )
        return [
            (
                "counter",
                "batch_server:mtp_decode_seconds_total",
                "Time spent inside llama_decode() for the MTP draft context.",
                self.decode_seconds_total,
            ),
            (
                "counter",
                "batch_server:mtp_decode_calls_total",
                "Number of llama_decode() calls made by the MTP draft context.",
                self.decode_calls_total,
            ),
            (
                "counter",
                "batch_server:mtp_decode_tokens_total",
                "Number of batch rows decoded by the MTP draft context.",
                self.decode_tokens_total,
            ),
            (
                "counter",
                "batch_server:mtp_decode_failures_total",
                "Number of failed MTP draft context decode calls.",
                self.decode_failures_total,
            ),
            (
                "gauge",
                "batch_server:mtp_decode_tokens_seconds",
                "Average MTP draft context decode throughput in batch rows/s.",
                decode_tokens_seconds,
            ),
        ]

    def can_draft(self, input_length: int, /, *, seq_id: int) -> bool:
        if (
            input_length <= 0
            or seq_id < 0
            or seq_id >= self.n_seq_max
            or not self.ready[seq_id]
        ):
            return False
        return self.ready_pos[seq_id] == input_length - 1

    def process(self, batch: Any, /) -> None:
        n_tokens = int(batch.n_tokens)
        if (
            n_tokens <= 0
            or not self.target_processing_enabled
            or not bool(batch.token)
            or bool(batch.embd)
        ):
            return

        h_tgt = llama_cpp_ext.llama_get_embeddings_nextn(self.target_ctx)
        if not h_tgt:
            raise RuntimeError("missing target nextn embeddings for MTP")
        h_tgt_rows = np.ctypeslib.as_array(
            h_tgt,
            shape=(n_tokens, self.n_embd),
        )

        previous_row_by_seq: Dict[int, int] = {}
        first_pos_by_seq: Dict[int, int] = {}
        target_rows_by_seq: Dict[int, List[int]] = {}
        aligned_by_seq: Dict[int, bool] = {}

        for start in range(0, n_tokens, self.n_batch):
            self._process_rows(
                batch,
                h_tgt_rows,
                start,
                min(start + self.n_batch, n_tokens),
                previous_row_by_seq,
                first_pos_by_seq,
                target_rows_by_seq,
                aligned_by_seq,
            )

        for seq_id, rows in target_rows_by_seq.items():
            if not aligned_by_seq.get(seq_id, False):
                self.ready[seq_id] = False
                continue
            if rows[-1] - rows[0] + 1 == len(rows):
                target_rows = h_tgt_rows[rows[0] : rows[-1] + 1]
            else:
                target_rows = h_tgt_rows[rows].copy()
            target_positions = [int(batch.pos[source_index]) for source_index in rows]
            self.verify_h[seq_id] = target_rows
            self.verify_h_pos[seq_id] = target_positions
            self.verify_h_rows[seq_id] = len(rows)
            self.pending_h[seq_id] = target_rows[-1]
            self.ready[seq_id] = True
            self.ready_pos[seq_id] = target_positions[-1] + 1

    def _process_rows(
        self,
        batch: Any,
        h_tgt_rows: np.ndarray,
        start: int,
        end: int,
        previous_row_by_seq: Dict[int, int],
        first_pos_by_seq: Dict[int, int],
        target_rows_by_seq: Dict[int, List[int]],
        aligned_by_seq: Dict[int, bool],
    ) -> None:
        added_pos_by_seq: Dict[int, int] = {}
        self._clear_batch()
        for index in range(start, end):
            if int(batch.n_seq_id[index]) != 1:
                raise RuntimeError("MTP requires one sequence id per batch token")
            seq_id = int(batch.seq_id[index][0])
            if seq_id < 0 or seq_id >= self.n_seq_max:
                raise RuntimeError(f"MTP sequence id out of range: {seq_id}")
            pos = int(batch.pos[index])
            first_pos = first_pos_by_seq.setdefault(seq_id, pos)
            aligned = first_pos <= 0 or (
                self.ready[seq_id] and self.ready_pos[seq_id] == first_pos
            )
            aligned_by_seq.setdefault(seq_id, aligned)
            previous_row = previous_row_by_seq.get(seq_id)
            if (
                aligned
                and not self.is_mem_shared
                and pos >= 0
                and pos >= self.context_pos[seq_id]
            ):
                slot = int(self.batch.n_tokens)
                self._add_batch_token(
                    token=int(batch.token[index]),
                    pos=pos,
                    seq_id=seq_id,
                    logits=False,
                )
                if previous_row is None:
                    self._set_batch_embedding_row(slot, self.pending_h[seq_id])
                else:
                    self._set_batch_embedding_row(slot, h_tgt_rows[previous_row])
                added_pos_by_seq[seq_id] = pos
            previous_row_by_seq[seq_id] = index
            target_rows_by_seq.setdefault(seq_id, []).append(index)

        if int(self.batch.n_tokens) > 0:
            self._decode_batch()
            for seq_id, pos in added_pos_by_seq.items():
                self.context_pos[seq_id] = max(self.context_pos[seq_id], pos + 1)

    def draft(
        self,
        input_ids: np.ndarray,
        /,
        *,
        seq_id: int,
        max_tokens: Optional[int],
    ) -> np.ndarray:
        if (
            self.num_pred_tokens <= 0
            or input_ids.size == 0
            or seq_id < 0
            or seq_id >= self.n_seq_max
            or not self.ready[seq_id]
        ):
            return np.array([], dtype=np.intc)
        n_predict = self.num_pred_tokens
        if max_tokens is not None:
            n_predict = min(n_predict, max_tokens)
        if n_predict <= 0:
            return np.array([], dtype=np.intc)

        n_past = int(input_ids.size) - 1
        if self.ready_pos[seq_id] != n_past:
            return np.array([], dtype=np.intc)
        first_pos = n_past
        if first_pos < 0:
            return np.array([], dtype=np.intc)

        token = int(input_ids[-1])
        drafted: List[int] = []
        if not self.is_mem_shared and self.context_pos[seq_id] > first_pos:
            self.truncate(seq_id, first_pos)
        if not self.is_mem_shared and self.context_pos[seq_id] < first_pos:
            self.ready[seq_id] = False
            return np.array([], dtype=np.intc)

        self._reset_sampler(seq_id)
        self._clear_batch()
        self._add_batch_token(
            token=token,
            pos=first_pos,
            seq_id=seq_id,
            logits=True,
        )
        self._set_batch_embedding_row(0, self.pending_h[seq_id])
        if not self._try_decode_batch():
            if not self.is_mem_shared:
                self.truncate(seq_id, first_pos)
            return np.array([], dtype=np.intc)
        if not self.is_mem_shared:
            self.context_pos[seq_id] = first_pos + 1

        while len(drafted) < n_predict:
            sampled_token = self._sample_token(seq_id=seq_id)
            if sampled_token is None:
                break
            token = sampled_token
            drafted.append(token)
            if len(drafted) >= n_predict:
                break
            h_row = llama_cpp_ext.llama_get_embeddings_nextn_ith(self.ctx, 0)
            if not h_row:
                break
            self._clear_batch()
            self._add_batch_token(
                token=token,
                pos=first_pos if self.is_mem_shared else first_pos + len(drafted),
                seq_id=seq_id,
                logits=True,
            )
            self._set_batch_embedding_row(0, h_row)
            if not self._try_decode_batch():
                break
            if not self.is_mem_shared:
                self.context_pos[seq_id] = first_pos + len(drafted) + 1

        if not drafted:
            if not self.is_mem_shared:
                self.truncate(seq_id, n_past)
            return np.array([], dtype=np.intc)
        if not self.is_mem_shared:
            self.truncate(seq_id, n_past)
        return np.asarray(drafted, dtype=np.intc)

    def draft_many(
        self,
        requests: Sequence[Tuple[np.ndarray, int, Optional[int]]],
        /,
    ) -> List[np.ndarray]:
        results = [np.array([], dtype=np.intc) for _ in requests]
        active: List["MTPDraftProvider.DraftManyState"] = []
        for result_index, (input_ids, seq_id, max_tokens) in enumerate(requests):
            if (
                self.num_pred_tokens <= 0
                or input_ids.size == 0
                or seq_id < 0
                or seq_id >= self.n_seq_max
                or not self.ready[seq_id]
            ):
                continue
            n_predict = self.num_pred_tokens
            if max_tokens is not None:
                n_predict = min(n_predict, max_tokens)
            if n_predict <= 0:
                continue
            if len(active) >= self.n_batch:
                break

            n_past = int(input_ids.size) - 1
            if self.ready_pos[seq_id] != n_past:
                continue
            first_pos = n_past
            if first_pos < 0:
                continue
            if not self.is_mem_shared and self.context_pos[seq_id] > first_pos:
                self.truncate(seq_id, first_pos)
            if not self.is_mem_shared and self.context_pos[seq_id] < first_pos:
                self.ready[seq_id] = False
                continue
            self._reset_sampler(seq_id)
            active.append(
                self.DraftManyState(
                    result_index=result_index,
                    seq_id=seq_id,
                    first_pos=first_pos,
                    keep_len=n_past,
                    n_predict=n_predict,
                    token=int(input_ids[-1]),
                    drafted=[],
                    embedding=self.pending_h[seq_id],
                    cache_key=(
                        tuple(int(token) for token in input_ids.tolist()),
                        n_predict,
                    ),
                )
            )

        if not active:
            return results

        touched = list(active)
        try:
            if all(state.n_predict == 1 for state in active):
                grouped: Dict[
                    Tuple[Tuple[int, ...], int],
                    List["MTPDraftProvider.DraftManyState"],
                ] = {}
                for state in active:
                    grouped.setdefault(state.cache_key, []).append(state)

                representatives = [states[0] for states in grouped.values()]
                self._clear_batch()
                for row, state in enumerate(representatives):
                    self._add_batch_token(
                        token=state.token,
                        pos=state.first_pos,
                        seq_id=state.seq_id,
                        logits=True,
                    )
                    self._set_batch_embedding_row(row, state.embedding)

                if self._try_decode_batch():
                    sampled_tokens = [
                        self._sample_token(row, seq_id=state.seq_id)
                        for row, state in enumerate(representatives)
                    ]
                    for representative, sampled_token in zip(
                        representatives,
                        sampled_tokens,
                    ):
                        if sampled_token is None:
                            continue
                        if not self.is_mem_shared:
                            self.context_pos[representative.seq_id] = max(
                                self.context_pos[representative.seq_id],
                                representative.first_pos + 1,
                            )
                        for state in grouped[representative.cache_key]:
                            state.drafted.append(sampled_token)
                    active = []

            while active:
                self._clear_batch()
                for row, state in enumerate(active):
                    self._add_batch_token(
                        token=state.token,
                        pos=(
                            state.first_pos
                            if self.is_mem_shared
                            else state.first_pos + len(state.drafted)
                        ),
                        seq_id=state.seq_id,
                        logits=True,
                    )
                    self._set_batch_embedding_row(row, state.embedding)

                if not self._try_decode_batch():
                    break

                next_active: List["MTPDraftProvider.DraftManyState"] = []
                sampled_tokens = [
                    self._sample_token(row, seq_id=state.seq_id)
                    for row, state in enumerate(active)
                ]
                for row, (state, sampled_token) in enumerate(zip(active, sampled_tokens)):
                    decoded_pos = (
                        state.first_pos
                        if self.is_mem_shared
                        else state.first_pos + len(state.drafted)
                    )
                    if not self.is_mem_shared:
                        self.context_pos[state.seq_id] = max(
                            self.context_pos[state.seq_id],
                            decoded_pos + 1,
                        )
                    if sampled_token is None:
                        continue
                    h_row_ptr = llama_cpp_ext.llama_get_embeddings_nextn_ith(
                        self.ctx, row
                    )
                    state.drafted.append(sampled_token)
                    if len(state.drafted) >= state.n_predict:
                        continue
                    if not h_row_ptr:
                        continue
                    state.token = sampled_token
                    state.embedding = np.ctypeslib.as_array(
                        h_row_ptr,
                        shape=(self.n_embd,),
                    ).copy()
                    next_active.append(state)
                active = next_active
        finally:
            if not self.is_mem_shared:
                for state in touched:
                    self.truncate(state.seq_id, state.keep_len)

        for state in touched:
            if state.drafted:
                results[state.result_index] = np.asarray(
                    state.drafted,
                    dtype=np.intc,
                )
        return results

    @dataclass(frozen=True)
    class SampledContextRow:
        seq_id: int
        draft_pos: int
        token: int
        source_row: Optional[int]

    @dataclass(frozen=True)
    class SampledPendingRow:
        update_index: int
        seq_id: int
        draft_pos: int
        token: int
        source_row: int

    @dataclass(frozen=True)
    class SampledOutput:
        update_index: int
        seq_id: int
        output_index: int
        keep_len: int
        ready_pos: int

    @dataclass(frozen=True)
    class SampledBatchUpdate:
        seq_id: int
        start_pos: int
        tokens: List[int]
        row_indices: List[int]
        target_count: int
        sample_index: int
        pending_token: Optional[int]
        max_tokens: Optional[int]

    @dataclass
    class SampledBatchPlan:
        context_rows: List["MTPDraftProvider.SampledContextRow"]
        pending_rows: List["MTPDraftProvider.SampledPendingRow"]
        sample_pending_index_by_update: Dict[int, int]

    @dataclass
    class SampledDraftState:
        update_index: int
        seq_id: int
        keep_len: int
        pos: int
        token: int
        drafted: List[int]
        n_predict: int
        embedding: np.ndarray

    def _build_sampled_batch_plan(
        self,
        updates: Sequence["MTPDraftProvider.SampledBatchUpdate"],
        /,
    ) -> "MTPDraftProvider.SampledBatchPlan":
        context_rows: List["MTPDraftProvider.SampledContextRow"] = []
        pending_rows: List["MTPDraftProvider.SampledPendingRow"] = []
        sample_pending_index_by_update: Dict[int, int] = {}

        for update_index, update in enumerate(updates):
            seq_id = update.seq_id
            if seq_id < 0 or seq_id >= self.n_seq_max:
                continue
            start_pos = update.start_pos
            tokens = update.tokens
            row_indices = update.row_indices
            target_count = update.target_count
            sample_index = update.sample_index
            pending_token = update.pending_token
            if (
                pending_token is None
                or start_pos < 0
                or target_count <= 0
                or target_count > len(tokens)
                or target_count > len(row_indices)
                or sample_index < 0
                or sample_index >= target_count
            ):
                continue

            for target_index in range(sample_index + 1):
                mtp_pos = start_pos + target_index
                source_row = (
                    None
                    if target_index == 0
                    else row_indices[target_index - 1]
                )
                context_rows.append(
                    self.SampledContextRow(
                        seq_id=seq_id,
                        draft_pos=mtp_pos,
                        token=tokens[target_index],
                        source_row=source_row,
                    )
                )

            actual_pos = start_pos + sample_index + 1
            pending_rows.append(
                self.SampledPendingRow(
                    update_index=update_index,
                    seq_id=seq_id,
                    draft_pos=actual_pos,
                    token=pending_token,
                    source_row=row_indices[sample_index],
                )
            )
            sample_pending_index_by_update[update_index] = len(pending_rows) - 1

        return self.SampledBatchPlan(
            context_rows=context_rows,
            pending_rows=pending_rows,
            sample_pending_index_by_update=sample_pending_index_by_update,
        )

    def _decode_sampled_context_rows(
        self,
        context_rows: Sequence["MTPDraftProvider.SampledContextRow"],
        h_tgt_rows: np.ndarray,
        /,
    ) -> None:
        self._clear_batch()
        decoded_context_rows: List[Tuple[int, int]] = []
        for row in context_rows:
            if self.is_mem_shared:
                continue
            if row.draft_pos < self.context_pos[row.seq_id]:
                continue
            if row.source_row is None:
                embedding = self.pending_h[row.seq_id]
            else:
                embedding = h_tgt_rows[row.source_row]
            self._add_batch_token(
                token=row.token,
                pos=row.draft_pos,
                seq_id=row.seq_id,
                logits=False,
            )
            self._set_batch_embedding_row(int(self.batch.n_tokens) - 1, embedding)
            decoded_context_rows.append((row.seq_id, row.draft_pos))

        if int(self.batch.n_tokens) > 0:
            self._decode_batch()
            for seq_id, draft_pos in decoded_context_rows:
                self.context_pos[seq_id] = max(
                    self.context_pos[seq_id],
                    draft_pos + 1,
                )

    def _decode_sampled_pending_rows(
        self,
        plan: "MTPDraftProvider.SampledBatchPlan",
        h_tgt_rows: np.ndarray,
        /,
    ) -> List["MTPDraftProvider.SampledOutput"]:
        sampled_outputs: List["MTPDraftProvider.SampledOutput"] = []
        pending_rows = plan.pending_rows

        self._clear_batch()
        for pending_index, row in enumerate(pending_rows):
            if (
                not self.is_mem_shared
                and row.draft_pos < self.context_pos[row.seq_id]
            ):
                continue
            is_sample_pending = (
                pending_index
                == plan.sample_pending_index_by_update.get(row.update_index)
            )
            slot = int(self.batch.n_tokens)
            self._add_batch_token(
                token=row.token,
                pos=row.draft_pos,
                seq_id=row.seq_id,
                logits=is_sample_pending,
            )
            self._set_batch_embedding_row(slot, h_tgt_rows[row.source_row])
            if is_sample_pending:
                sampled_outputs.append(
                    self.SampledOutput(
                        update_index=row.update_index,
                        seq_id=row.seq_id,
                        output_index=slot,
                        keep_len=row.draft_pos,
                        ready_pos=row.draft_pos,
                    )
                )

        self._decode_batch()
        if not self.is_mem_shared:
            for row in pending_rows:
                self.context_pos[row.seq_id] = max(
                    self.context_pos[row.seq_id],
                    row.draft_pos + 1,
                )

        return sampled_outputs

    def _start_sampled_draft_states(
        self,
        updates: Sequence["MTPDraftProvider.SampledBatchUpdate"],
        sampled_outputs: Sequence["MTPDraftProvider.SampledOutput"],
        results: List[np.ndarray],
        /,
    ) -> List["MTPDraftProvider.SampledDraftState"]:
        active: List["MTPDraftProvider.SampledDraftState"] = []
        for output in sampled_outputs:
            self._reset_sampler(output.seq_id)
            sampled_token = self._sample_token(
                output.output_index,
                seq_id=output.seq_id,
            )
            if sampled_token is None:
                continue
            update = updates[output.update_index]
            seq_id = update.seq_id
            self.ready[seq_id] = True
            n_predict = self.num_pred_tokens
            max_tokens = update.max_tokens
            if max_tokens is not None:
                n_predict = min(n_predict, max_tokens)
            if n_predict <= 0:
                continue
            if n_predict > 1:
                h_row = llama_cpp_ext.llama_get_embeddings_nextn_ith(
                    self.ctx, output.output_index
                )
                if h_row:
                    active.append(
                        self.SampledDraftState(
                            update_index=output.update_index,
                            seq_id=seq_id,
                            keep_len=output.keep_len,
                            pos=(
                                output.keep_len
                                if self.is_mem_shared
                                else output.keep_len + 1
                            ),
                            token=sampled_token,
                            drafted=[sampled_token],
                            n_predict=n_predict,
                            embedding=np.ctypeslib.as_array(
                                h_row,
                                shape=(self.n_embd,),
                            ).copy(),
                        )
                    )
            results[output.update_index] = np.asarray([sampled_token], dtype=np.intc)

        return active

    def _extend_sampled_draft_states(
        self,
        active: List["MTPDraftProvider.SampledDraftState"],
        results: List[np.ndarray],
        cleanup_keep_len_by_seq: Dict[int, int],
        /,
    ) -> None:
        touched = list(active)
        try:
            while active:
                self._clear_batch()
                for batch_row, state in enumerate(active):
                    self._add_batch_token(
                        token=state.token,
                        pos=state.pos,
                        seq_id=state.seq_id,
                        logits=True,
                    )
                    self._set_batch_embedding_row(batch_row, state.embedding)

                if not self._try_decode_batch():
                    break

                next_active: List["MTPDraftProvider.SampledDraftState"] = []
                sampled_tokens = [
                    self._sample_token(batch_row, seq_id=state.seq_id)
                    for batch_row, state in enumerate(active)
                ]
                for batch_row, (state, sampled_token) in enumerate(
                    zip(active, sampled_tokens)
                ):
                    if not self.is_mem_shared:
                        self.context_pos[state.seq_id] = max(
                            self.context_pos[state.seq_id],
                            state.pos + 1,
                        )
                    if sampled_token is None:
                        continue
                    state.drafted.append(sampled_token)
                    if len(state.drafted) >= state.n_predict:
                        continue
                    h_row = llama_cpp_ext.llama_get_embeddings_nextn_ith(
                        self.ctx, batch_row
                    )
                    if not h_row:
                        continue
                    state.token = sampled_token
                    state.embedding = np.ctypeslib.as_array(
                        h_row,
                        shape=(self.n_embd,),
                    ).copy()
                    if not self.is_mem_shared:
                        state.pos += 1
                    next_active.append(state)
                active = next_active
        finally:
            for state in touched:
                cleanup_keep_len_by_seq[state.seq_id] = state.keep_len
            if not self.is_mem_shared:
                for seq_id, keep_len in cleanup_keep_len_by_seq.items():
                    self._truncate_memory(seq_id, keep_len)

        for state in touched:
            if state.drafted:
                results[state.update_index] = np.asarray(
                    state.drafted,
                    dtype=np.intc,
                )

    def process_sampled_batch(
        self,
        updates: Sequence["MTPDraftProvider.SampledBatchUpdate"],
        /,
    ) -> List[np.ndarray]:
        results = [np.array([], dtype=np.intc) for _ in updates]
        if self.num_pred_tokens <= 0 or not updates:
            return results
        h_tgt = llama_cpp_ext.llama_get_embeddings_nextn(self.target_ctx)
        if not h_tgt:
            raise RuntimeError("missing target nextn embeddings for MTP")
        n_target_rows = max(
            (
                max(update.row_indices) + 1
                for update in updates
                if update.row_indices
            ),
            default=0,
        )
        if n_target_rows <= 0:
            return results
        h_tgt_rows = np.ctypeslib.as_array(h_tgt, shape=(n_target_rows, self.n_embd))

        plan = self._build_sampled_batch_plan(updates)
        if not plan.context_rows and not plan.pending_rows:
            return results
        if (
            len(plan.context_rows) > self.n_batch
            or len(plan.pending_rows) > self.n_batch
        ):
            raise RuntimeError("MTP draft batch capacity exceeded")

        self._decode_sampled_context_rows(plan.context_rows, h_tgt_rows)
        sampled_outputs = self._decode_sampled_pending_rows(plan, h_tgt_rows)
        cleanup_keep_len_by_seq: Dict[int, int] = {}
        for output in sampled_outputs:
            update = updates[output.update_index]
            sample_index = update.sample_index
            sample_source_row = update.row_indices[sample_index]
            self.pending_h[output.seq_id] = h_tgt_rows[sample_source_row]
            self.ready[output.seq_id] = True
            self.ready_pos[output.seq_id] = output.ready_pos
            cleanup_keep_len_by_seq[output.seq_id] = output.keep_len

        if not self.is_mem_shared:
            for seq_id, keep_len in cleanup_keep_len_by_seq.items():
                self._truncate_memory(seq_id, keep_len)

        if sampled_outputs:
            active = self._start_sampled_draft_states(
                updates,
                sampled_outputs,
                results,
            )
            self._extend_sampled_draft_states(
                active,
                results,
                cleanup_keep_len_by_seq,
            )

        return results

    def accept(self, seq_id: int, accepted_draft_tokens: int) -> None:
        if seq_id < 0 or seq_id >= self.n_seq_max:
            return
        n_rows = self.verify_h_rows[seq_id]
        if n_rows <= 0:
            return
        row = min(max(accepted_draft_tokens, 0), n_rows - 1)
        self.pending_h[seq_id] = self.verify_h[seq_id][row]
        self.ready[seq_id] = True
        self.ready_pos[seq_id] = self.verify_h_pos[seq_id][row] + 1

    def _truncate_memory(self, seq_id: int, keep_len: int) -> None:
        if seq_id < 0 or seq_id >= self.n_seq_max:
            return
        if self.is_mem_shared:
            self.context_pos[seq_id] = min(self.context_pos[seq_id], keep_len)
            return
        if not llama_cpp.llama_memory_seq_rm(
            self.mem,
            seq_id,
            keep_len,
            -1,
        ):
            raise RuntimeError(
                f"failed to truncate MTP draft sequence {seq_id} at position {keep_len}"
            )
        self.context_pos[seq_id] = min(self.context_pos[seq_id], keep_len)

    def truncate(self, seq_id: int, keep_len: int) -> None:
        if seq_id < 0 or seq_id >= self.n_seq_max:
            return
        self._truncate_memory(seq_id, keep_len)
        if keep_len <= 0:
            self.pending_h[seq_id].fill(0.0)
            self.verify_h[seq_id] = np.empty((0, self.n_embd), dtype=np.float32)
            self.verify_h_pos[seq_id] = []
            self.verify_h_rows[seq_id] = 0
            self.ready[seq_id] = False
            self.ready_pos[seq_id] = 0
            self.context_pos[seq_id] = 0
            return

        if self.ready_pos[seq_id] != keep_len:
            self.ready[seq_id] = False

    def copy_sequence(
        self,
        source_seq_id: int,
        dest_seq_id: int,
        p0: int,
        p1: int,
    ) -> None:
        if (
            source_seq_id < 0
            or source_seq_id >= self.n_seq_max
            or dest_seq_id < 0
            or dest_seq_id >= self.n_seq_max
        ):
            return
        if not self.is_mem_shared:
            llama_cpp.llama_memory_seq_cp(
                self.mem,
                source_seq_id,
                dest_seq_id,
                p0,
                p1,
            )
        source_ready_pos = self.ready_pos[source_seq_id]
        copied_full_ready_state = p1 < 0 or p1 == source_ready_pos
        if self.ready[source_seq_id] and copied_full_ready_state:
            self.pending_h[dest_seq_id] = self.pending_h[source_seq_id]
            self.verify_h[dest_seq_id] = self.verify_h[source_seq_id].copy()
            self.verify_h_pos[dest_seq_id] = list(self.verify_h_pos[source_seq_id])
            self.verify_h_rows[dest_seq_id] = self.verify_h_rows[source_seq_id]
            self.ready[dest_seq_id] = True
            self.ready_pos[dest_seq_id] = source_ready_pos
            self.context_pos[dest_seq_id] = min(
                self.context_pos[source_seq_id],
                source_ready_pos,
            )
            return

        self.pending_h[dest_seq_id].fill(0.0)
        self.verify_h[dest_seq_id] = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_h_pos[dest_seq_id] = []
        self.verify_h_rows[dest_seq_id] = 0
        self.ready[dest_seq_id] = False
        self.ready_pos[dest_seq_id] = 0
        self.context_pos[dest_seq_id] = 0


class CompletionRequestCancelledError(RuntimeError):
    pass


class CompletionRequestValidationError(ValueError):
    pass


class CompletionResponseParsingError(RuntimeError):
    pass


def omit_additional_properties(schema: Dict[str, Any]) -> None:
    schema.pop("additionalProperties", None)


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
EmbeddingInput = Union[str, List[str], List[int], List[List[int]]]


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


class CreateEmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: EmbeddingInput
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None

    @staticmethod
    def _validate_text_input(text: str) -> str:
        if text == "":
            raise ValueError("embedding input must not contain empty strings")
        return text

    @staticmethod
    def _validate_token_input(tokens: List[int]) -> List[int]:
        if not tokens:
            raise ValueError("embedding token input must not be empty")
        if len(tokens) > 2048:
            raise ValueError("embedding token input must not exceed 2048 tokens")
        return tokens

    @model_validator(mode="after")
    def validate_after(self) -> "CreateEmbeddingRequest":
        self.normalized_input()
        return self

    def normalized_input(self) -> List[Union[str, List[int]]]:
        if isinstance(self.input, str):
            return [self._validate_text_input(self.input)]
        if all(isinstance(token, int) for token in self.input):
            return [self._validate_token_input(cast(List[int], self.input))]
        if all(isinstance(item, str) for item in self.input):
            if len(self.input) > 2048:
                raise ValueError("embedding input array must not exceed 2048 items")
            return [
                self._validate_text_input(item)
                for item in cast(List[str], self.input)
            ]
        if all(
            isinstance(item, list)
            and all(isinstance(token, int) for token in item)
            for item in self.input
        ):
            if len(self.input) > 2048:
                raise ValueError("embedding input array must not exceed 2048 items")
            return [
                self._validate_token_input(item)
                for item in cast(List[List[int]], self.input)
            ]
        raise ValueError(
            "embedding input must be a string, list of strings, token ids, or list of token-id lists"
        )


class EmbeddingDataResponse(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], str]
    index: int


class EmbeddingUsageResponse(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingDataResponse]
    model: str
    usage: EmbeddingUsageResponse

    @staticmethod
    def encode_embedding(
        embedding: Sequence[float],
        encoding_format: Literal["float", "base64"],
        dimensions: Optional[int],
    ) -> Union[List[float], str]:
        if dimensions is not None:
            if dimensions > len(embedding):
                raise CompletionRequestValidationError(
                    f"dimensions ({dimensions}) exceeds embedding size ({len(embedding)})"
                )
            embedding = embedding[:dimensions]
        if encoding_format == "float":
            return [float(value) for value in embedding]
        array = np.asarray(embedding, dtype=np.float32)
        return base64.b64encode(array.tobytes()).decode("ascii")

    @classmethod
    def from_embeddings(
        cls,
        *,
        model: str,
        embeddings: Sequence[Sequence[float]],
        total_tokens: int,
        encoding_format: Literal["float", "base64"],
        dimensions: Optional[int],
    ) -> "CreateEmbeddingResponse":
        return cls(
            data=[
                EmbeddingDataResponse(
                    embedding=cls.encode_embedding(
                        embedding,
                        encoding_format,
                        dimensions,
                    ),
                    index=index,
                )
                for index, embedding in enumerate(embeddings)
            ],
            model=model,
            usage=EmbeddingUsageResponse(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )


class ChatCompletionFunctionCall(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra=omit_additional_properties,
    )

    name: str
    arguments: Optional[str] = None


class ChatCompletionToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: ChatCompletionFunctionCall


class ChatCompletionRequestMessage(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra=omit_additional_properties,
    )

    role: Literal["system", "developer", "user", "assistant", "tool", "function"] = Field(
        default="user"
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(default="")
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    function_call: Optional[ChatCompletionFunctionCall] = Field(default=None)
    tool_calls: Optional[List[ChatCompletionToolCall]] = Field(default=None)


class ChatTemplateFunctionDefinition(TypedDict, total=False):
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]
    strict: Optional[bool]
    content_type: Optional[str]


class ChatTemplateTool(TypedDict, total=False):
    type: Literal["function"]
    original_type: str
    function: ChatTemplateFunctionDefinition


class ChatTemplateFunctionCall(TypedDict, total=False):
    name: str


class ChatTemplateToolChoice(TypedDict, total=False):
    type: Literal["function"]
    function: ChatTemplateFunctionCall


class ChatTemplateResponseFormatJsonSchema(TypedDict, total=False):
    name: Optional[str]
    description: Optional[str]
    schema: Dict[str, Any]
    strict: Optional[bool]


class ChatTemplateResponseFormat(TypedDict, total=False):
    type: Literal["text", "json_object", "json_schema"]
    schema: Dict[str, Any]
    json_schema: ChatTemplateResponseFormatJsonSchema


class ChatCompletionFunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def to_template_function(self) -> ChatTemplateFunctionDefinition:
        function: ChatTemplateFunctionDefinition = {"name": self.name}
        if self.description is not None:
            function["description"] = self.description
        if self.parameters is not None:
            function["parameters"] = self.parameters
        return function


class ChatCompletionToolFunction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    def to_template_function(self) -> ChatTemplateFunctionDefinition:
        function: ChatTemplateFunctionDefinition = {"name": self.name}
        if self.description is not None:
            function["description"] = self.description
        if self.parameters is not None:
            function["parameters"] = self.parameters
        if self.strict is not None:
            function["strict"] = self.strict
        return function


class ChatCompletionTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    function: ChatCompletionToolFunction

    def to_template_tool(self) -> ChatTemplateTool:
        return {
            "type": self.type,
            "function": self.function.to_template_function(),
        }


class ChatCompletionFunctionCallOption(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str

    def to_template_function_call(self) -> ChatTemplateFunctionCall:
        return {"name": self.name}


class ChatCompletionToolChoiceObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    function: ChatCompletionFunctionCallOption

    def to_template_tool_choice(self) -> ChatTemplateToolChoice:
        return {
            "type": self.type,
            "function": self.function.to_template_function_call(),
        }


class ChatCompletionResponseFormatJsonSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: Optional[str] = None
    description: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None

    def to_template_json_schema(self) -> ChatTemplateResponseFormatJsonSchema:
        json_schema: ChatTemplateResponseFormatJsonSchema = {}
        if self.name is not None:
            json_schema["name"] = self.name
        if self.description is not None:
            json_schema["description"] = self.description
        if self.schema_ is not None:
            json_schema["schema"] = self.schema_
        if self.strict is not None:
            json_schema["strict"] = self.strict
        return json_schema


class ChatCompletionResponseFormat(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"]
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    json_schema: Optional[ChatCompletionResponseFormatJsonSchema] = None

    def to_template_response_format(self) -> ChatTemplateResponseFormat:
        response_format: ChatTemplateResponseFormat = {"type": self.type}
        if self.schema_ is not None:
            response_format["schema"] = self.schema_
        if self.json_schema is not None:
            response_format["json_schema"] = self.json_schema.to_template_json_schema()
        return response_format


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
    functions: Optional[List[ChatCompletionFunctionDefinition]] = None
    function_call: Optional[
        Union[Literal["none", "auto"], ChatCompletionFunctionCallOption]
    ] = None
    tools: Optional[List[ChatCompletionTool]] = None
    tool_choice: Optional[
        Union[Literal["none", "auto", "required"], ChatCompletionToolChoiceObject]
    ] = None
    response_format: Optional[ChatCompletionResponseFormat] = None
    reasoning_effort: Optional[
        Literal["low", "medium", "high", "minimal", "none"]
    ] = None

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


class ResponsesFunctionTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    def to_chat_template_tool(self) -> ChatTemplateTool:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }


class ResponsesCustomToolFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Optional[str] = None
    syntax: Optional[str] = None
    definition: Optional[str] = None


class ResponsesCustomTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["custom"]
    name: str
    description: Optional[str] = None
    format: Optional[ResponsesCustomToolFormat] = None
    strict: Optional[bool] = None

    def to_chat_template_tool(self) -> ChatTemplateTool:
        tool_format = self.format
        syntax = tool_format.syntax if tool_format is not None else None
        definition = tool_format.definition if tool_format is not None else None
        description = self.description or ""
        text_tool_guidance = (
            "This is a text tool. When calling it, the "
            "`function.arguments` field itself must be the raw input string. "
            "Do not wrap the input in JSON and do not use an object such as "
            "'{\"input\": \"...\"}' or '{\"patch\": \"...\"}'."
        )
        if isinstance(syntax, str) and isinstance(definition, str):
            if description:
                description = (
                    f"{description}\n\n{text_tool_guidance}\n\n"
                    f"{syntax}:\n{definition}"
                )
            else:
                description = f"{text_tool_guidance}\n\n{syntax}:\n{definition}"
        elif description:
            description = f"{description}\n\n{text_tool_guidance}"
        else:
            description = text_tool_guidance
        return {
            "type": "function",
            "original_type": "custom",
            "function": {
                "name": self.name,
                "description": description or None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": (
                                "Raw input text for this tool. "
                                "For apply_patch, put the full patch here."
                            ),
                        },
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
                "strict": self.strict,
                "content_type": "text",
            },
        }


class ResponsesWebSearchTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["web_search"]


class ResponsesNamespaceTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["namespace"]


class ResponsesImageGenerationTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["image_generation"]


ResponsesToolDefinition = Union[
    ResponsesFunctionTool,
    ResponsesCustomTool,
    ResponsesWebSearchTool,
    ResponsesNamespaceTool,
    ResponsesImageGenerationTool,
]


class ResponsesToolChoiceObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function", "custom"]
    name: str

    def to_chat_template_tool_choice(self) -> ChatTemplateToolChoice:
        return {
            "type": "function",
            "function": {
                "name": self.name,
            },
        }


ResponsesToolChoice = Union[
    Literal["auto", "none", "required"],
    ResponsesToolChoiceObject,
]


class ResponsesTextFormatJsonSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_: Dict[str, Any] = Field(alias="schema")

    def to_template_json_schema(self) -> ChatTemplateResponseFormatJsonSchema:
        return {"schema": self.schema_}


class ResponsesTextFormat(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"]
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    json_schema: Optional[ResponsesTextFormatJsonSchema] = None

    def to_chat_response_format(self) -> ChatTemplateResponseFormat:
        response_format: ChatTemplateResponseFormat = {"type": self.type}
        if self.schema_ is not None:
            response_format["schema"] = self.schema_
        if self.json_schema is not None:
            response_format["json_schema"] = self.json_schema.to_template_json_schema()
        return response_format


class ResponsesTextOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: Optional[ResponsesTextFormat] = None


class ResponsesReasoningOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    effort: Optional[Literal["low", "medium", "high"]] = None


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
    tools: Optional[List["ResponsesToolDefinition"]] = None
    tool_choice: Optional["ResponsesToolChoice"] = None
    parallel_tool_calls: bool = True
    text: Optional["ResponsesTextOptions"] = None
    reasoning: Optional["ResponsesReasoningOptions"] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[str] = None
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
    tools: Optional[List["ResponsesToolDefinition"]] = None
    tool_choice: Optional["ResponsesToolChoice"] = None
    parallel_tool_calls: bool = True
    text: Optional["ResponsesTextOptions"] = None
    reasoning: Optional["ResponsesReasoningOptions"] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[str] = None
    store: Optional[bool] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    generate: Optional[bool] = None

    def to_create_response_request(self) -> CreateResponseRequest:
        return CreateResponseRequest.model_validate(
            self.model_dump(mode="python", exclude={"type"})
        )


class ModelCardResponse(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCardResponse]


class HealthzResponse(BaseModel):
    status: Literal["ok"] = "ok"


@dataclass
class ResponsesWebSocketState:
    input_items: List[Any]
    output_items: List[Dict[str, Any]]


class ConfigFile(BaseModel):
    class ServerOptions(BaseModel):
        host: str = "127.0.0.1"
        port: int = 8000

    class DiskCacheOptions(BaseModel):
        path: str
        max_bytes: int = Field(ge=0)
        min_tokens: int = Field(default=128, ge=1)

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

            download_kwargs: Dict[str, Any] = dict(
                repo_id=self.repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=self.local_dir,
                cache_dir=self.cache_dir,
            )
            hf_hub_download_signature = inspect.signature(hf_hub_download)
            supports_local_dir_use_symlinks = (
                "local_dir_use_symlinks" in hf_hub_download_signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in hf_hub_download_signature.parameters.values()
                )
            )
            if supports_local_dir_use_symlinks:
                download_kwargs["local_dir_use_symlinks"] = self.local_dir_use_symlinks

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
                try:
                    from huggingface_hub import HfFileSystem
                    from huggingface_hub.utils._validators import validate_repo_id
                except ImportError:
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

    class LoraOptions(BaseModel):
        path: Optional[str] = None
        from_pretrained: Optional["ConfigFile.FromPretrainedOptions"] = None
        scale: float = 1.0

        @model_validator(mode="after")
        def validate_source(self) -> "ConfigFile.LoraOptions":
            if (self.path is None) == (self.from_pretrained is None):
                raise ValueError("exactly one of lora.path or lora.from_pretrained is required")
            return self

        def resolve_path(self) -> str:
            if self.from_pretrained is not None:
                return self.from_pretrained.resolve_model_path()
            assert self.path is not None
            return self.path

    class MTMDEmbeddingCacheOptions(BaseModel):
        path: str
        max_bytes: int = Field(ge=0)

    class MTMDOptions(BaseModel):
        mmproj_path: Optional[str] = None
        mmproj_from_pretrained: Optional["ConfigFile.FromPretrainedOptions"] = None
        embedding_cache: Optional["ConfigFile.MTMDEmbeddingCacheOptions"] = None
        allowed_media_domains: Optional[List[str]] = None
        allowed_local_media_path: Optional[str] = None
        batch_max_tokens: int = Field(default=1024, ge=1)
        image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
        audio_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
        video_max_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
        image_timeout_seconds: float = Field(default=10.0, gt=0.0)

        @model_validator(mode="after")
        def validate_source(self) -> "ConfigFile.MTMDOptions":
            if (self.mmproj_path is None) == (self.mmproj_from_pretrained is None):
                raise ValueError(
                    "exactly one of model.mtmd.mmproj_path or "
                    "model.mtmd.mmproj_from_pretrained is required"
                )
            return self

        def resolve_mmproj_path(self) -> str:
            if self.mmproj_from_pretrained is not None:
                return self.mmproj_from_pretrained.resolve_model_path()
            assert self.mmproj_path is not None
            return self.mmproj_path

    class ModelOptions(BaseModel):
        path: Optional[str] = None
        alias: Optional[str] = None
        chat_template: Optional[str] = None
        from_pretrained: Optional["ConfigFile.FromPretrainedOptions"] = None
        loras: List["ConfigFile.LoraOptions"] = Field(default_factory=list)
        mtmd: Optional["ConfigFile.MTMDOptions"] = None
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
        embedding: Optional[bool] = None
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
        draft_model: Optional[Literal["prompt-lookup-decoding", "draft-mtp"]] = None
        draft_model_path: Optional[str] = None
        draft_model_from_pretrained: Optional[
            "ConfigFile.FromPretrainedOptions"
        ] = None
        draft_model_num_pred_tokens: int = 16
        draft_model_max_ngram_size: int = 2
        draft_model_top_k: int = Field(default=1, ge=1)
        draft_model_p_min: float = Field(default=0.0, ge=0.0, le=1.0)
        draft_model_max_batch_size: Optional[int] = Field(default=None, ge=1)
        draft_model_threads: Optional[int] = Field(default=None, gt=0)
        draft_model_threads_batch: Optional[int] = Field(default=None, gt=0)
        response_schema: Optional[Dict[str, Any]] = None
        store_logits: bool = True

        @model_validator(mode="after")
        def validate_source(self) -> "ConfigFile.ModelOptions":
            if (self.path is None) == (self.from_pretrained is None):
                raise ValueError("exactly one of model.path or model.from_pretrained is required")
            if (
                self.draft_model_path is not None
                and self.draft_model_from_pretrained is not None
            ):
                raise ValueError(
                    "model.draft_model_path and model.draft_model_from_pretrained "
                    "are mutually exclusive"
                )
            return self

        def resolve_draft_model_path(self) -> Optional[str]:
            if self.draft_model_from_pretrained is not None:
                return self.draft_model_from_pretrained.resolve_model_path()
            return self.draft_model_path

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
    disk_cache: Optional["ConfigFile.DiskCacheOptions"] = None

    @classmethod
    def load(cls, path: str) -> "ConfigFile":
        return cls.model_validate_json(Path(path).read_text())


ConfigFile.model_rebuild()


@dataclass(frozen=True)
class MediaInput:
    kind: Literal["image", "audio", "video"]
    url: Optional[str] = None
    data: Optional[str] = None
    format: Optional[str] = None


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

    @staticmethod
    def media_inputs_from_messages(
        messages: Sequence[ChatCompletionRequestMessage],
    ) -> List[MediaInput]:
        media_inputs: List[MediaInput] = []
        for message in messages:
            content = message.content
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"image_url", "input_image", "image"}:
                    image_url = part.get("image_url") or part.get("url")
                    if isinstance(image_url, str):
                        media_inputs.append(MediaInput(kind="image", url=image_url))
                    elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                        media_inputs.append(MediaInput(kind="image", url=cast(str, image_url["url"])))
                    else:
                        raise ValueError("image_url content part requires a URL string")
                    continue
                if part_type == "audio_url":
                    audio_url = part.get("audio_url")
                    if isinstance(audio_url, str):
                        media_inputs.append(MediaInput(kind="audio", url=audio_url))
                    elif isinstance(audio_url, dict) and isinstance(audio_url.get("url"), str):
                        media_inputs.append(MediaInput(kind="audio", url=cast(str, audio_url["url"])))
                    else:
                        raise ValueError("audio_url content part requires a URL string")
                    continue
                if part_type == "input_audio":
                    input_audio = part.get("input_audio")
                    if isinstance(input_audio, dict):
                        data = input_audio.get("data")
                        audio_format = input_audio.get("format")
                    else:
                        data = part.get("data")
                        audio_format = part.get("format")
                    if not isinstance(data, str):
                        raise ValueError("input_audio content part requires base64 data")
                    if audio_format is not None and not isinstance(audio_format, str):
                        raise ValueError("input_audio format must be a string")
                    media_inputs.append(
                        MediaInput(
                            kind="audio",
                            data=data,
                            format=cast(Optional[str], audio_format),
                        )
                    )
                    continue
                if part_type in {"video_url", "input_video", "video"}:
                    input_video = part.get("input_video")
                    if isinstance(input_video, dict):
                        data = input_video.get("data")
                        video_url = input_video.get("video_url") or input_video.get("url")
                        video_format = input_video.get("format")
                    else:
                        data = part.get("data")
                        video_url = part.get("video_url") or part.get("url")
                        video_format = part.get("format")
                    if isinstance(video_url, dict):
                        video_url = video_url.get("url")
                    if isinstance(data, str):
                        if video_format is not None and not isinstance(video_format, str):
                            raise ValueError("input_video format must be a string")
                        media_inputs.append(
                            MediaInput(
                                kind="video",
                                data=data,
                                format=cast(Optional[str], video_format),
                            )
                        )
                    elif isinstance(video_url, str):
                        media_inputs.append(MediaInput(kind="video", url=video_url))
                    else:
                        raise ValueError("video content part requires base64 data or a URL string")
                    continue
        return media_inputs

    @staticmethod
    def _literal_content_for_template(
        text: str,
        media_marker: Optional[str],
    ) -> str:
        if media_marker is not None and media_marker in text:
            raise ValueError("message content contains reserved multimodal marker text")
        return text

    @staticmethod
    def _content_part_for_template(
        part: Any,
        media_marker: Optional[str],
    ) -> str:
        if isinstance(part, str):
            return Jinja2ChatFormatter._literal_content_for_template(part, media_marker)
        if not isinstance(part, dict):
            raise ValueError("content parts must be strings or objects")
        part_type = part.get("type")
        if part_type in {
            "image_url",
            "input_image",
            "image",
            "audio_url",
            "input_audio",
            "video_url",
            "input_video",
            "video",
        }:
            if media_marker is None:
                raise ValueError("multimodal content requires model.mtmd")
            return media_marker
        if part_type in {"text", "input_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"content part {part_type!r} requires string text")
            return Jinja2ChatFormatter._literal_content_for_template(text, media_marker)
        raise ValueError(f"unsupported content part type: {part_type!r}")

    @staticmethod
    def _messages_for_template(
        messages: Sequence[ChatCompletionRequestMessage],
        media_marker: Optional[str],
    ) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            data = message.model_dump(exclude_none=True)
            content = data.get("content")
            if isinstance(content, list):
                data["content"] = "".join(
                    Jinja2ChatFormatter._content_part_for_template(
                        part,
                        media_marker,
                    )
                    for part in content
                )
            elif isinstance(content, str):
                data["content"] = Jinja2ChatFormatter._literal_content_for_template(
                    content,
                    media_marker,
                )
            converted.append(data)
        return converted

    def _render(
        self,
        *,
        messages: List[ChatCompletionRequestMessage],
        media_marker: Optional[str] = None,
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        function_call: Optional[Union[str, ChatTemplateFunctionCall]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        tool_choice: Optional[Union[str, ChatTemplateToolChoice]] = None,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool,
        strftime_now: Callable[[str], str],
    ) -> str:
        def raise_exception(message: str) -> None:
            raise ValueError(message)

        return cast(
            str,
            self._template.render(
                messages=self._messages_for_template(messages, media_marker),
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
        media_marker: Optional[str] = None,
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        function_call: Optional[Union[str, ChatTemplateFunctionCall]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        tool_choice: Optional[Union[str, ChatTemplateToolChoice]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[str, str, List[str]]:
        render_time = datetime.now()

        def strftime_now(format_string: str) -> str:
            return render_time.strftime(format_string)

        chat_template = self._render(
            messages=messages,
            media_marker=media_marker,
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
            media_marker=media_marker,
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
    prompt_length: int
    prompt_text: str
    multimodal_prompt: bool
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
        return self.prompt_length + len(self.completion_tokens)

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
class PromptSegment:
    @dataclass
    class Media:
        embeddings: np.ndarray
        positions: np.ndarray
        non_causal: bool = False

    kind: Literal["text", "image", "audio", "video"]
    start_pos: int
    n_pos: int
    identity_tokens: List[int]
    decode_start_pos: int
    decode_n_pos: int
    text_tokens: List[int] = field(default_factory=list)
    media: Optional["PromptSegment.Media"] = None

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.n_pos

    @property
    def batch_rows(self) -> int:
        if self.kind != "text":
            if self.media is None:
                return 0
            return int(self.media.embeddings.shape[0])
        return len(self.text_tokens)

    @property
    def decoder_position_increments(self) -> List[int]:
        if not self.identity_tokens:
            return []
        if self.kind == "text":
            return [1] * len(self.identity_tokens)
        return [*([0] * (len(self.identity_tokens) - 1)), self.decode_n_pos]

    def rows_for_capacity(self, offset: int, capacity: int) -> int:
        if capacity <= 0:
            return 0
        return min(capacity, self.end_pos - self.start_pos - offset)

    def media_slice(
        self,
        row_offset: int,
        row_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        if self.media is None:
            raise RuntimeError("media segment is missing embeddings or positions")
        row_start = row_offset
        row_end = row_offset + row_count
        embeddings = self.media.embeddings[row_start:row_end]
        if len(self.media.positions) == self.batch_rows:
            positions = self.media.positions[row_start:row_end]
        else:
            positions = (
                self.media.positions.reshape(4, self.batch_rows)[:, row_start:row_end]
                .reshape(-1)
            )
        return (
            embeddings,
            positions,
            self.decoder_position_increments[row_start:row_end],
        )


@dataclass
class PromptPlan:
    text: str
    generation_prompt: str
    text_tokens: List[int]
    identity_tokens: List[int]
    segments: List[PromptSegment]
    text_token_index_by_pos: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_tokens(
        cls,
        text: str,
        tokens: List[int],
        *,
        generation_prompt: str = "",
    ) -> "PromptPlan":
        segment = PromptSegment(
            kind="text",
            start_pos=0,
            n_pos=len(tokens),
            identity_tokens=list(tokens),
            decode_start_pos=0,
            decode_n_pos=len(tokens),
            text_tokens=list(tokens),
        )
        return cls(
            text=text,
            generation_prompt=generation_prompt,
            text_tokens=list(tokens),
            identity_tokens=list(tokens),
            segments=[segment] if tokens else [],
            text_token_index_by_pos={pos: pos for pos in range(len(tokens))},
        )

    @property
    def length(self) -> int:
        return len(self.identity_tokens)

    @property
    def eval_token_count(self) -> int:
        return self.length

    def position_increments_up_to(self, pos: int) -> List[int]:
        increments: List[int] = []
        for segment in self.segments:
            if pos <= segment.start_pos:
                break
            take = min(pos, segment.end_pos) - segment.start_pos
            if take <= 0:
                continue
            increments.extend(segment.decoder_position_increments[:take])
            if pos <= segment.end_pos:
                break
        return increments

    def is_boundary(self, pos: int) -> bool:
        if pos <= 0 or pos >= self.length:
            return True
        return any(segment.start_pos == pos or segment.end_pos == pos for segment in self.segments)

    def is_reusable_boundary(self, pos: int) -> bool:
        if pos <= 0 or pos >= self.length:
            return True
        if self.is_boundary(pos):
            return True
        return self.segment_at(pos).kind == "text"

    def clamp_to_reusable_boundary(self, pos: int) -> int:
        if pos <= 0:
            return 0
        if pos >= self.length:
            return self.length
        if self.is_reusable_boundary(pos):
            return pos
        segment = self.segment_at(pos)
        return segment.start_pos

    def decoder_pos_up_to(self, pos: int) -> int:
        if not self.is_reusable_boundary(pos):
            raise RuntimeError("decoder position requested inside a non-reusable media segment")
        if pos <= 0:
            return 0
        if pos >= self.length:
            if not self.segments:
                return 0
            segment = self.segments[-1]
            return segment.decode_start_pos + segment.decode_n_pos
        segment = self.segment_at(pos)
        if segment.kind == "text":
            return segment.decode_start_pos + (pos - segment.start_pos)
        if pos == segment.start_pos:
            return segment.decode_start_pos
        if pos == segment.end_pos:
            return segment.decode_start_pos + segment.decode_n_pos
        raise RuntimeError("decoder position requested inside a media segment")

    def segment_at(self, pos: int) -> PromptSegment:
        for segment in self.segments:
            if segment.start_pos <= pos < segment.end_pos:
                return segment
        raise RuntimeError(f"missing prompt segment at position {pos}")

    def has_text_token_at(self, pos: int) -> bool:
        return pos in self.text_token_index_by_pos

    def text_token_at(self, pos: int) -> int:
        return self.text_tokens[self.text_token_index_by_pos[pos]]

    def prev_text_tokens_at(self, pos: int) -> List[int]:
        return self.text_tokens[: self.text_token_index_by_pos[pos]]


@dataclass(frozen=True)
class PreparedCompletionParts:
    payload: CreateCompletionRequest
    prompt_text: str
    generation_prompt: str
    prompt_plan: PromptPlan
    grammar_text: Optional[str]
    tool_name: Optional[str]


@dataclass(frozen=True)
class ResponsesChatRequestParts:
    messages: List[ChatCompletionRequestMessage]
    max_tokens: Optional[int]
    temperature: float
    top_p: float
    stream: bool
    logprobs: bool
    top_logprobs: Optional[int]
    model: Optional[str]
    user: Optional[str]
    tools: Optional[List[ChatTemplateTool]]
    tool_choice: Optional[Union[Literal["auto", "none", "required"], ChatTemplateToolChoice]]
    response_format: Optional[ChatTemplateResponseFormat]
    reasoning_effort: Optional[str]


@dataclass
class SchedulerMetrics:
    started_at: float = field(default_factory=time.time)
    requests_submitted_total: int = 0
    requests_admitted_total: int = 0
    requests_completed_total: int = 0
    requests_cancelled_total: int = 0
    requests_failed_total: int = 0
    prompt_tokens_total: int = 0
    prompt_seconds_total: float = 0.0
    tokens_predicted_total: int = 0
    tokens_predicted_seconds_total: float = 0.0
    scheduler_step_seconds_total: float = 0.0
    process_batch_seconds_total: float = 0.0
    sample_seconds_total: float = 0.0
    draft_seconds_total: float = 0.0
    draft_process_seconds_total: float = 0.0
    draft_generate_seconds_total: float = 0.0
    draft_sampled_batch_seconds_total: float = 0.0
    draft_process_calls_total: int = 0
    draft_generate_calls_total: int = 0
    draft_sampled_batch_calls_total: int = 0
    draft_batches_verified_total: int = 0
    draft_target_tokens_verified_total: int = 0
    draft_target_tokens_wasted_total: int = 0
    draft_tokens_reused_as_pending_total: int = 0
    n_decode_total: int = 0
    scheduler_step_calls_total: int = 0
    process_batch_calls_total: int = 0
    sample_calls_total: int = 0
    n_tokens_max: int = 0
    n_busy_slots_total: int = 0
    checkpoint_hits_total: int = 0
    checkpoint_saves_total: int = 0
    checkpoint_evictions_total: int = 0
    sequence_cache_hits_total: int = 0
    sequence_cache_save_requests_total: int = 0
    sequence_cache_lookup_failures_total: int = 0
    sequence_cache_load_failures_total: int = 0
    sequence_cache_save_failures_total: int = 0
    sequence_cache_tokens_loaded_total: int = 0

    def observe_decode(
        self,
        items: Sequence[Any],
        elapsed_seconds: float,
    ) -> None:
        if not items:
            return
        total_tokens = sum(
            item.batch_token_count
            for item in items
        )
        if total_tokens <= 0:
            return
        prompt_tokens = sum(
            item.batch_token_count
            for item in items
            if getattr(item, "kind", None) == "prefill"
        )
        generation_tokens = total_tokens - prompt_tokens
        self.n_decode_total += 1
        self.n_busy_slots_total += len(items)
        self.n_tokens_max = max(self.n_tokens_max, total_tokens)
        self.prompt_tokens_total += prompt_tokens
        if prompt_tokens > 0:
            self.prompt_seconds_total += elapsed_seconds * (prompt_tokens / total_tokens)
        if generation_tokens > 0:
            self.tokens_predicted_seconds_total += elapsed_seconds * (
                generation_tokens / total_tokens
            )

    def observe_predicted_token(self) -> None:
        self.tokens_predicted_total += 1


class SequenceCache(Protocol):
    """Optional external sequence-state cache; storage and eviction stay outside the scheduler."""

    @dataclass(frozen=True)
    class Match:
        tokens: Tuple[int, ...]
        has_prompt_logits: bool = False

        @property
        def length(self) -> int:
            return len(self.tokens)

    @dataclass
    class Load:
        tokens: List[int]
        state_bytes: np.ndarray
        prompt_logits: Optional[np.ndarray] = None

    def lookup(self, tokens: Sequence[int]) -> Optional["SequenceCache.Match"]:
        ...

    def load(
        self,
        match: "SequenceCache.Match",
    ) -> Optional["SequenceCache.Load"]:
        ...

    def save(
        self,
        tokens: Sequence[int],
        state_bytes: np.ndarray,
        prompt_logits: Optional[np.ndarray],
    ) -> None:
        ...


@dataclass
class CompletionRequest:
    payload: CreateCompletionRequest
    prompt_text: str
    prompt_tokens: List[int]
    prompt_plan: PromptPlan
    effective_max_len: int
    internal_completion_count: int
    prompt_visible_start: int
    id: str = field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    created: int = field(default_factory=lambda: int(time.time()))
    prompt_cursor: int = 0
    match_sequence_id: int = -1
    match_length: int = 0
    sequence_cache_match: Optional[SequenceCache.Match] = None
    sequence_cache_match_length: int = 0
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
    def from_prepared(
        cls,
        *,
        payload: CreateCompletionRequest,
        prompt_text: str,
        prompt_plan: PromptPlan,
        max_seq_len: int,
        max_output_tokens: Optional[int],
        prompt_visible_start: int,
        prompt_records: Optional[List[Token]] = None,
        grammar_text: Optional[str] = None,
        chat_tool_name: Optional[str] = None,
        on_stream_chunk: Optional[Callable[[CompletionChunk], None]] = None,
        on_done: Optional[Callable[[OpenAICompletion], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> "CompletionRequest":
        prompt_tokens = list(prompt_plan.identity_tokens)
        prompt_eval_tokens = prompt_plan.eval_token_count
        if prompt_eval_tokens > max_seq_len:
            raise CompletionRequestValidationError("prompt exceeds context window")
        ctx_limit = prompt_plan.length + (max_seq_len - prompt_eval_tokens)
        if max_output_tokens is not None:
            ctx_limit = min(ctx_limit, prompt_plan.length + max_output_tokens)
        if payload.max_tokens is None:
            effective_max_len = ctx_limit
        else:
            effective_max_len = min(ctx_limit, prompt_plan.length + payload.max_tokens)
        if effective_max_len < prompt_plan.length:
            raise CompletionRequestValidationError("prompt exceeds context window")
        internal_completion_count = payload.best_of if payload.best_of is not None else payload.n
        request = cls(
            payload=payload,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            prompt_plan=prompt_plan,
            effective_max_len=effective_max_len,
            internal_completion_count=internal_completion_count,
            prompt_visible_start=prompt_visible_start,
            prompt_records=list(prompt_records or []),
            grammar_text=grammar_text,
            chat_tool_name=chat_tool_name,
            on_stream_chunk=on_stream_chunk,
            on_done=on_done,
            on_error=on_error,
        )
        return request

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
        output_indices: Sequence[Optional[int]],
        output_positions: Sequence[int],
        output_count: int,
        output_index_to_logits_index: Callable[[Optional[int], int], Optional[int]],
    ) -> None:
        if self.payload.logprobs is None or not self.payload.echo:
            return
        for token_offset, output_index in enumerate(output_indices):
            if output_index is None:
                continue
            next_pos = output_positions[token_offset] + 1
            if not self.prompt_plan.has_text_token_at(next_pos):
                continue
            text_token_index = self.prompt_plan.text_token_index_by_pos[next_pos]
            if text_token_index < self.prompt_visible_start:
                continue
            logits_index = output_index_to_logits_index(output_index, output_count)
            assert logits_index is not None
            next_token = self.prompt_plan.text_token_at(next_pos)
            record = Token.from_logits(
                model=model,
                formatter=formatter,
                prev_tokens=self.prompt_plan.prev_text_tokens_at(next_pos),
                token=next_token,
                logits=model.logits(logits_index),
                logprobs_count=self.payload.logprobs,
            )
            expected_index = text_token_index - self.prompt_visible_start
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
        tools: Optional[List[ChatTemplateTool]] = None,
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
            success, _ = self._advance_stream_state(self._generation_prompt)
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

        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            text = "{" + stripped[1:-1] + "}"
        text = re.sub(r'<\|"\|>(.*?)<\|"\|>', capture, text, flags=re.S)
        text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
        for index, value in enumerate(strings):
            text = text.replace(f"\x00{index}\x00", json.dumps(value))
        return text

    @staticmethod
    def _regex_literal_prefix(pattern: str) -> str:
        literal, _ = ResponseParser._regex_literal_prefix_and_remainder(pattern)
        return literal

    @staticmethod
    def _regex_literal_prefix_and_remainder(pattern: str) -> Tuple[str, str]:
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
        return "".join(literal), pattern[index:]

    @staticmethod
    def _find_regex_group_end(pattern: str, start: int) -> int:
        depth = 0
        escaped = False
        in_character_class = False
        for index in range(start, len(pattern)):
            char = pattern[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "[":
                in_character_class = True
                continue
            if char == "]" and in_character_class:
                in_character_class = False
                continue
            if in_character_class:
                continue
            if char == "(":
                depth += 1
                continue
            if char == ")":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @classmethod
    def _consume_optional_literal_prefix(
        cls,
        pattern: str,
    ) -> Optional[Tuple[str, str]]:
        if not pattern.startswith("(?:"):
            return None
        group_end = cls._find_regex_group_end(pattern, 0)
        if group_end < 0 or group_end + 1 >= len(pattern) or pattern[group_end + 1] != "?":
            return None
        literal, remainder = cls._regex_literal_prefix_and_remainder(pattern[3:group_end])
        if not literal or remainder:
            return None
        return literal, pattern[group_end + 2 :]

    @staticmethod
    def _split_regex_alternatives(pattern: str) -> List[str]:
        alternatives: List[str] = []
        start = 0
        depth = 0
        escaped = False
        in_character_class = False
        for index, char in enumerate(pattern):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "[":
                in_character_class = True
                continue
            if char == "]" and in_character_class:
                in_character_class = False
                continue
            if in_character_class:
                continue
            if char == "(":
                depth += 1
                continue
            if char == ")":
                depth -= 1
                continue
            if char == "|" and depth == 0:
                alternatives.append(pattern[start:index])
                start = index + 1
        alternatives.append(pattern[start:])
        return alternatives

    @classmethod
    def _regex_lookahead_literal_specs(cls, pattern: str) -> List[Tuple[str, bool]]:
        if not pattern.startswith("(?="):
            return []
        group_end = cls._find_regex_group_end(pattern, 0)
        if group_end < 0:
            return []
        literals: List[Tuple[str, bool]] = []
        for alternative in cls._split_regex_alternatives(pattern[3:group_end]):
            strip_leading_whitespace = False
            while alternative.startswith(r"\s*"):
                strip_leading_whitespace = True
                alternative = alternative[3:]
            if alternative == "$":
                continue
            if alternative.endswith("$"):
                alternative = alternative[:-1]
            literal, _ = cls._regex_literal_prefix_and_remainder(alternative)
            if literal:
                literals.append((literal, strip_leading_whitespace))
        return literals

    @classmethod
    def _regex_capture_parts(
        cls,
        pattern: str,
    ) -> Optional[Tuple[str, str]]:
        normalized = pattern.lstrip("^")
        captures = [
            (index, token)
            for token in ("(.*?)", "(.*)")
            if (index := normalized.find(token)) >= 0
        ]
        if not captures:
            return None
        capture_index, capture_token = min(captures, key=lambda item: item[0])
        return normalized[:capture_index], normalized[capture_index + len(capture_token) :]

    @classmethod
    def _regex_capture_end_literal_specs(cls, pattern: str) -> List[Tuple[str, bool]]:
        capture_parts = cls._regex_capture_parts(pattern)
        if capture_parts is None:
            return []
        _, suffix_pattern = capture_parts
        literal_specs = cls._regex_lookahead_literal_specs(suffix_pattern)
        if literal_specs:
            return literal_specs
        literal, _ = cls._regex_literal_prefix_and_remainder(suffix_pattern)
        return [(literal, False)] if literal else []

    @classmethod
    def _regex_capture_end_literals(cls, pattern: str) -> List[str]:
        return [literal for literal, _ in cls._regex_capture_end_literal_specs(pattern)]

    @classmethod
    def _regex_leading_capture(
        cls,
        *,
        field_name: str,
        field_regex: str,
        content_regex: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        capture_parts = cls._regex_capture_parts(field_regex)
        if capture_parts is None:
            return None
        prefix_pattern, _ = capture_parts
        prefix_pattern = prefix_pattern.lstrip("^")
        optional_prefix = cls._consume_optional_literal_prefix(prefix_pattern)
        if optional_prefix is not None:
            prefix_pattern = optional_prefix[1]
        implicit_at_start = False
        optional_capture_start = cls._consume_optional_literal_prefix(prefix_pattern)
        if optional_capture_start is not None:
            capture_start, prefix_pattern = optional_capture_start
            implicit_at_start = True
        else:
            capture_start, prefix_pattern = cls._regex_literal_prefix_and_remainder(prefix_pattern)
        if not capture_start or prefix_pattern:
            return None
        end_literals = cls._regex_capture_end_literals(field_regex)
        if not end_literals:
            return None
        capture_end = end_literals[0]
        strip_after = False
        if isinstance(content_regex, str):
            escaped_end = re.escape(capture_end)
            strip_after = bool(re.search(escaped_end + r"\\s\*", content_regex))
        return {
            "field": field_name,
            "start": capture_start,
            "end": capture_end,
            "strip_after": strip_after,
            "implicit_at_start": implicit_at_start,
        }

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
            iterator_capture = cls._compile_iterator_block_pattern(iterator_pattern)
            if (
                not isinstance(iterator_capture, dict)
                or not iterator_capture["start"]
                or iterator_capture["allow_eof"]
            ):
                return None
            iterator = (iterator_capture["start"], iterator_capture["end"])
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
        assistant_prefix: Optional[str] = None
        if isinstance(content_regex, str):
            optional_prefix = cls._consume_optional_literal_prefix(content_regex.lstrip("^"))
            if optional_prefix is not None:
                assistant_prefix = optional_prefix[0]
        leading_capture: Optional[Dict[str, Any]] = None
        for field_name, value_schema in properties.items():
            if not isinstance(value_schema, dict):
                continue
            field_regex = value_schema.get("x-regex")
            if not isinstance(field_regex, str):
                continue
            if field_name == "content":
                continue
            capture = cls._regex_leading_capture(
                field_name=field_name,
                field_regex=field_regex,
                content_regex=content_regex,
            )
            if capture is not None:
                leading_capture = capture
                break
        end_markers: List[str] = []
        content_end_marker_specs: List[Tuple[str, bool]] = []
        iterator_start, iterator_end = iterator
        if "content" in properties:
            end_markers.append(iterator_start)
        if isinstance(content_regex, str):
            content_end_marker_specs = cls._regex_capture_end_literal_specs(content_regex)
            end_markers.extend(literal for literal, _ in content_end_marker_specs)
        if not end_markers and iterator_start:
            end_markers.append(iterator_start)
        deduped_end_markers = tuple(dict.fromkeys(end_markers))
        trim_before_iterator = any(
            literal == iterator_start and strip_leading_whitespace
            for literal, strip_leading_whitespace in content_end_marker_specs
        )
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
                deduped_end_markers,
                tuple(marker for marker in deduped_end_markers if marker != iterator_start),
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
            "content_end_markers": deduped_end_markers,
            "trim_before_iterator": trim_before_iterator,
            "stop_markers": tuple(marker for marker in deduped_end_markers if marker != iterator_start),
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
        tools: Optional[List[ChatTemplateTool]],
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
        tools: Optional[List[ChatTemplateTool]],
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

    def _raw_string_tool_arguments(self, tool_name: str, value: str) -> Optional[Dict[str, str]]:
        if self._tools is None:
            return None
        for tool in self._tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            if function.get("name") != tool_name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                return None
            required = parameters.get("required")
            if not isinstance(required, list) or len(required) != 1:
                return None
            argument_name = required[0]
            if not isinstance(argument_name, str):
                return None
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                return None
            argument_schema = properties.get(argument_name)
            if not isinstance(argument_schema, dict):
                return None
            argument_type = argument_schema.get("type")
            if argument_type == "string" or (
                isinstance(argument_type, list) and "string" in argument_type
            ):
                return {argument_name: value}
            return None
        return None

    def _single_string_tool_argument_name(self, tool_name: str) -> Optional[str]:
        if self._tools is None:
            return None
        for tool in self._tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            if function.get("name") != tool_name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                return None
            required = parameters.get("required")
            if not isinstance(required, list) or len(required) != 1:
                return None
            argument_name = required[0]
            if not isinstance(argument_name, str):
                return None
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                return None
            argument_schema = properties.get(argument_name)
            if not isinstance(argument_schema, dict):
                return None
            argument_type = argument_schema.get("type")
            if argument_type == "string" or (
                isinstance(argument_type, list) and "string" in argument_type
            ):
                return argument_name
            return None
        return None

    def _text_tool_argument_from_object(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[str]:
        input_value = arguments.get("input")
        if isinstance(input_value, str):
            return input_value
        argument_name = self._single_string_tool_argument_name(tool_name)
        if argument_name is not None:
            argument_value = arguments.get(argument_name)
            if isinstance(argument_value, str):
                return argument_value
        if len(arguments) == 1:
            argument_value = next(iter(arguments.values()))
            if isinstance(argument_value, str):
                return argument_value
        return None

    def _text_tool_arguments(
        self,
        tool_name: str,
        arguments: Any,
        *,
        partial: bool,
    ) -> Optional[str]:
        if isinstance(arguments, str):
            parsed_arguments = self._raw_object_tool_arguments(arguments)
            if parsed_arguments is not None:
                text = self._text_tool_argument_from_object(tool_name, parsed_arguments)
                if text is not None:
                    return text
                if partial:
                    return None
                return json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":"))
            return arguments
        if isinstance(arguments, ResponseParser.PartialJsonObject):
            arguments = arguments.value
        if isinstance(arguments, dict):
            text = self._text_tool_argument_from_object(tool_name, arguments)
            if text is not None:
                return text
            if partial:
                return None
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if partial:
            return None
        return str(arguments)

    @classmethod
    def _raw_object_tool_arguments(cls, value: str) -> Optional[Dict[str, Any]]:
        candidates = [value]
        stripped = value.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            candidates.append(stripped[1:-1])
        for candidate in candidates:
            normalized = cls._gemma4_tool_call_to_json(candidate)
            for allow_partial in (False, True):
                try:
                    parsed = from_json(normalized, allow_partial=allow_partial)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    return {
                        key: cls._trim_partial_gemma_quote_marker(value)
                        if isinstance(value, str)
                        else value
                        for key, value in parsed.items()
                    }
        return None

    @staticmethod
    def _trim_partial_gemma_quote_marker(value: str) -> str:
        quote_marker = '<|"|>'
        for prefix_length in range(len(quote_marker) - 1, 0, -1):
            if value.endswith(quote_marker[:prefix_length]):
                return value[:-prefix_length]
        return value

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
        visible_tool_call = tool_calls[tool_call_index]
        assert isinstance(visible_tool_call, dict)
        self._item.visible_tool_call = visible_tool_call
        function = visible_tool_call.setdefault("function", {})
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
                if leading_capture_field is not None:
                    if buffer.startswith(leading_capture_start):
                        buffer = buffer[len(leading_capture_start) :]
                        mode = self.DIRECT_MODE_LEADING_CAPTURE
                        continue
                    if leading_capture_start.startswith(buffer):
                        self._direct.pending = buffer
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
                    _, matched, remainder, pending = self._consume_until_literal(
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
                    _, matched_start, remainder, pending = self._consume_until_any_literal(
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
            arguments = self._text_tool_arguments(
                tool_name,
                function.get("arguments", ""),
                partial=partial,
            )
            if arguments is None:
                return None
            return {
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = self._raw_object_tool_arguments(arguments) or self._raw_string_tool_arguments(
                tool_name, arguments
            )
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
                    arguments = self._text_tool_arguments(
                        function["name"],
                        function["arguments"],
                        partial=partial,
                    )
                    if arguments is None:
                        continue
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
                else:
                    previous_message = self._message
                    partial_deltas: List[Dict[str, Any]] = []
                    assert self._stream_state is not None
                    parsed = cast(Dict[str, Any], self._stream_state.parsed)
                    message = self._parsed_chat_message(
                        parsed=parsed,
                        partial=finish_reason is None or not self._stream_state_complete(),
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
    def collect_completion(stream: Iterator[Any]) -> OpenAICompletion:
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
    def _tools_for_response_parser(
        *,
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
    ) -> Optional[List[ChatTemplateTool]]:
        if functions is not None:
            return [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]
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
        if reasoning is None:
            return None
        effort = reasoning.effort
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
    def _response_chat_content_from_content(
        content: Any,
    ) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise CompletionRequestValidationError("responses input content must be a string or list")
        parts: List[Dict[str, Any]] = []
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
                parts.append({"type": "text", "text": text})
                continue
            if part_type == "input_image":
                image_url = part.get("image_url")
                if not isinstance(image_url, str):
                    raise CompletionRequestValidationError(
                        "responses input_image content part requires image_url"
                    )
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
                continue
            if part_type == "input_audio":
                input_audio = part.get("input_audio")
                if isinstance(input_audio, dict):
                    data = input_audio.get("data")
                    audio_format = input_audio.get("format")
                else:
                    data = part.get("data")
                    audio_format = part.get("format")
                if not isinstance(data, str):
                    raise CompletionRequestValidationError(
                        "responses input_audio content part requires base64 data"
                    )
                if audio_format is not None and not isinstance(audio_format, str):
                    raise CompletionRequestValidationError(
                        "responses input_audio format must be a string"
                    )
                audio_part: Dict[str, Any] = {"data": data}
                if audio_format is not None:
                    audio_part["format"] = audio_format
                parts.append({"type": "input_audio", "input_audio": audio_part})
                continue
            if part_type == "audio_url":
                audio_url = part.get("audio_url")
                if not isinstance(audio_url, str):
                    raise CompletionRequestValidationError(
                        "responses audio_url content part requires audio_url"
                    )
                parts.append({"type": "audio_url", "audio_url": {"url": audio_url}})
                continue
            raise CompletionRequestValidationError(
                f"unsupported responses content part type: {part_type!r}"
            )
        if all(part.get("type") == "text" for part in parts):
            return "".join(cast(str, part.get("text", "")) for part in parts)
        return parts

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

        function_names_by_call_id: Dict[str, str] = {}
        for item in items:
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
                    "content": self._response_chat_content_from_content(
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
        tools: Optional[List[ResponsesToolDefinition]],
    ) -> Optional[List[ChatTemplateTool]]:
        if tools is None:
            return None
        chat_tools: List[ChatTemplateTool] = []
        for tool in tools:
            if isinstance(
                tool,
                (
                    ResponsesWebSearchTool,
                    ResponsesNamespaceTool,
                    ResponsesImageGenerationTool,
                ),
            ):
                continue
            if isinstance(tool, ResponsesFunctionTool):
                chat_tools.append(tool.to_chat_template_tool())
                continue
            if isinstance(tool, ResponsesCustomTool):
                chat_tools.append(tool.to_chat_template_tool())
                continue
            raise CompletionRequestValidationError(
                f"unsupported responses tool type: {tool.type!r}"
            )
        return chat_tools

    @staticmethod
    def _responses_tool_choice_to_chat_tool_choice(
        tool_choice: Optional[ResponsesToolChoice],
    ) -> Optional[Union[Literal["auto", "none", "required"], ChatTemplateToolChoice]]:
        if tool_choice is None or isinstance(tool_choice, str):
            return tool_choice
        if tool_choice.type in {"function", "custom"}:
            return tool_choice.to_chat_template_tool_choice()
        raise CompletionRequestValidationError(
            f"unsupported responses tool_choice type: {tool_choice.type!r}"
        )

    @staticmethod
    def _response_format_type(response_format: Optional[ChatTemplateResponseFormat]) -> Optional[str]:
        if response_format is None:
            return None
        format_type = response_format.get("type")
        if isinstance(format_type, str):
            return format_type
        return None

    @staticmethod
    def _grammar_for_response_format(
        response_format: Optional[ChatTemplateResponseFormat],
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

    def response_request_to_chat_parts(
        self,
        body: CreateResponseRequest,
    ) -> ResponsesChatRequestParts:
        chat_tools = self._responses_tools_to_chat_tools(body.tools)
        response_format = (
            None
            if body.text is None or body.text.format is None
            else body.text.format.to_chat_response_format()
        )
        return ResponsesChatRequestParts(
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
            response_format=response_format,
            reasoning_effort=self._response_reasoning_effort(body),
        )

    def _response_parser(
        self,
        *,
        tools: Optional[List[ChatTemplateTool]] = None,
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
        tools: Optional[List[ChatTemplateTool]] = None,
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
        *,
        tools: Optional[List[ChatTemplateTool]],
        function_call: Optional[Union[Literal["none", "auto"], ChatTemplateFunctionCall]],
        tool_choice: Optional[Union[Literal["none", "auto", "required"], ChatTemplateToolChoice]],
        response_format: Optional[ChatTemplateResponseFormat],
    ) -> Tuple[Optional[str], Optional[str]]:
        selected_tool_choice: Optional[Union[str, ChatTemplateToolChoice]] = tool_choice
        if function_call is not None:
            if isinstance(function_call, str):
                selected_tool_choice = function_call
            else:
                selected_tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }
        grammar_text = self._grammar_for_response_format(response_format)
        if not isinstance(selected_tool_choice, dict):
            return None, grammar_text
        if tools is None:
            raise CompletionRequestValidationError("tool choice requires tools")
        tool_name = selected_tool_choice["function"]["name"]
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
    ) -> PreparedCompletionParts:
        functions = (
            [function.to_template_function() for function in body.functions]
            if body.functions is not None
            else None
        )
        tools = (
            [tool.to_template_tool() for tool in body.tools]
            if body.tools is not None
            else None
        )
        function_call = (
            body.function_call
            if body.function_call is None or isinstance(body.function_call, str)
            else body.function_call.to_template_function_call()
        )
        tool_choice = (
            body.tool_choice
            if body.tool_choice is None or isinstance(body.tool_choice, str)
            else body.tool_choice.to_template_tool_choice()
        )
        response_format = (
            body.response_format.to_template_response_format()
            if body.response_format is not None
            else None
        )
        parser_tools = self._tools_for_response_parser(
            functions=functions,
            tools=tools,
        )
        try:
            prompt_text, generation_prompt, prompt_plan, formatter_stop = self.model.build_chat_prompt(
                body.messages,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=body.reasoning_effort,
            )
            tool_name, grammar_text = self._chat_tool_name_and_grammar(
                tools=parser_tools,
                function_call=function_call,
                tool_choice=tool_choice,
                response_format=response_format,
            )
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
        return PreparedCompletionParts(
            payload=payload,
            prompt_text=prompt_text,
            generation_prompt=generation_prompt,
            prompt_plan=prompt_plan,
            grammar_text=grammar_text,
            tool_name=tool_name,
        )

    def completion_request_from_response_chat_parts(
        self,
        body: ResponsesChatRequestParts,
    ) -> PreparedCompletionParts:
        try:
            prompt_text, generation_prompt, prompt_plan, formatter_stop = self.model.build_chat_prompt(
                body.messages,
                tools=body.tools,
                tool_choice=body.tool_choice,
                reasoning_effort=body.reasoning_effort,
            )
            tool_name, grammar_text = self._chat_tool_name_and_grammar(
                tools=body.tools,
                function_call=None,
                tool_choice=body.tool_choice,
                response_format=body.response_format,
            )
        except ValueError as exc:
            raise CompletionRequestValidationError(str(exc)) from exc
        deduped_stop: List[str] = []
        seen_stop: set[str] = set()
        for stop in formatter_stop:
            if stop and stop not in seen_stop:
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
            seed=None,
            model=body.model,
            n=1,
            user=body.user,
        )
        return PreparedCompletionParts(
            payload=payload,
            prompt_text=prompt_text,
            generation_prompt=generation_prompt,
            prompt_plan=prompt_plan,
            grammar_text=grammar_text,
            tool_name=tool_name,
        )

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
        tokens = logprobs.tokens or []
        token_logprobs_list = logprobs.token_logprobs or []
        top_logprobs_list = logprobs.top_logprobs or []
        for token, token_logprob, top_logprobs in zip(
            tokens,
            token_logprobs_list,
            top_logprobs_list,
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
        tools: Optional[List[Any]],
    ) -> Dict[str, str]:
        if tools is None:
            return {}
        tool_types: Dict[str, str] = {}
        for tool in tools:
            if isinstance(tool, BaseModel):
                tool = tool.model_dump(mode="python", exclude_none=True)
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
        tools: Optional[List[ChatTemplateTool]] = None,
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
            "tool_choice": (
                body.tool_choice.model_dump(mode="python", exclude_none=True)
                if isinstance(body.tool_choice, BaseModel)
                else body.tool_choice or "auto"
            ),
            "tools": (
                [
                    tool.model_dump(mode="python", exclude_none=True)
                    if isinstance(tool, BaseModel)
                    else tool
                    for tool in body.tools
                ]
                if body.tools
                else []
            ),
            "top_p": body.top_p,
            "max_output_tokens": body.max_output_tokens,
            "previous_response_id": None,
            "status": status,
            "text": (
                body.text.model_dump(mode="python", exclude_none=True, by_alias=True)
                if body.text is not None
                else {"format": {"type": "text"}}
            ),
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
            tools=cast(Optional[List[Any]], body.tools),
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
        tools: Optional[List[ChatTemplateTool]] = None,
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
                item=copy.deepcopy(item),
            ),
            self._response_event(
                state,
                "response.content_part.added",
                item_id=cast(str, item["id"]),
                output_index=item_state.output_index,
                content_index=0,
                part=copy.deepcopy(part),
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
                item=copy.deepcopy(item),
            ),
            self._response_event(
                state,
                "response.content_part.added",
                item_id=cast(str, item["id"]),
                output_index=item_state.output_index,
                content_index=0,
                part=copy.deepcopy(part),
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
                item=copy.deepcopy(item),
            )
        ], item_state

    def _update_tool_stream_item(
        self,
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
                raw_arguments = cast(str, item.get("_raw_arguments", "")) + arguments_delta
                item["_raw_arguments"] = raw_arguments
                normalized_input = self._normalize_text_tool_payload(raw_arguments)
                if normalized_input is not None:
                    item["input"] = normalized_input
            else:
                item["arguments"] = (
                    cast(str, item.get("arguments", "")) + arguments_delta
                )

    @staticmethod
    def _normalize_text_tool_payload(payload: str) -> Optional[str]:
        if payload == "":
            return ""
        stripped = payload.lstrip()
        if not stripped:
            return ""
        if stripped[0] not in '{["':
            return payload
        try:
            decoded = json.loads(payload)
        except Exception:
            return None
        if isinstance(decoded, str):
            return decoded
        if isinstance(decoded, dict):
            input_value = decoded.get("input")
            if isinstance(input_value, str):
                return input_value
            if len(decoded) == 1:
                sole_value = next(iter(decoded.values()))
                if isinstance(sole_value, str):
                    return sole_value
        return payload

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
            item.pop("_raw_arguments", None)
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
                previous_input = cast(str, item_state.item.get("input", ""))
                self._update_tool_stream_item(
                    item_state.item,
                    call_id=cast(Optional[str], tool_call.get("id")),
                    name_delta=cast(Optional[str], function.get("name")),
                    arguments_delta=cast(Optional[str], function.get("arguments")),
                )
                arguments_delta = function.get("arguments")
                if isinstance(arguments_delta, str) and arguments_delta:
                    if item_state.item.get("type") == "custom_tool_call":
                        current_input = cast(str, item_state.item.get("input", ""))
                        if not current_input or current_input == previous_input:
                            continue
                        delta_text = current_input
                        if current_input.startswith(previous_input):
                            delta_text = current_input[len(previous_input) :]
                        events.append(
                            self._response_event(
                                state,
                                "response.custom_tool_call_input.delta",
                                item_id=cast(str, item_state.item["id"]),
                                output_index=item_state.output_index,
                                delta=delta_text,
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
        tokens = logprobs.tokens or []
        token_logprobs_list = logprobs.token_logprobs or []
        top_logprobs_list = logprobs.top_logprobs or []
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
                    tokens,
                    token_logprobs_list,
                    top_logprobs_list,
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
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        generation_prompt: str = "",
    ) -> ChatCompletion | Dict[str, Any]:
        parser_tools = self._tools_for_response_parser(
            functions=functions,
            tools=tools,
        )
        if self.model.response_schema is not None:
            choices: List[Dict[str, Any]] = []
            for choice in completion.choices:
                parser = self._response_parser(
                    tools=parser_tools,
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
                    finish_reason=cast(Any, (
                        "tool_calls" if tool_name is not None else choice.finish_reason
                    )),
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
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        parsed_states: Optional[Dict[int, Any]] = None,
        generation_prompt: str = "",
    ) -> List[ChatCompletionChunk | Dict[str, Any]]:
        parser_tools = self._tools_for_response_parser(
            functions=functions,
            tools=tools,
        )
        if self.model.response_schema is not None:
            parsed_chunks: List[Dict[str, Any]] = []
            if parsed_states is None:
                parsed_states = {}
            for choice in chunk["choices"]:
                index = choice["index"]
                parser = parsed_states.get(index)
                if not isinstance(parser, ResponseParser):
                    parser = self._response_parser(
                        tools=parser_tools,
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
            return cast(List[ChatCompletionChunk | Dict[str, Any]], parsed_chunks)
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
                            finish_reason=cast(Any, choice["finish_reason"]),
                        )
                    ],
                )
            )
        return cast(List[ChatCompletionChunk | Dict[str, Any]], chat_chunks)

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
                if finish_reason is None:
                    break
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
            logprobs = CompletionLogprobs.model_construct(
                text_offset=offsets,
                token_logprobs=token_logprobs,
                tokens=token_texts,
                top_logprobs=top_logprobs,
            )
        return CompletionChoice(
            text=text,
            index=completion.index,
            logprobs=logprobs,
            finish_reason=cast(Any, completion.finish_reason),
        )

    def build_completion_response(
        self,
        request: CompletionRequest,
        completions: Sequence[Completion],
    ) -> OpenAICompletion:
        completion_tokens = sum(completion.completion_token_count for completion in completions)
        prompt_tokens = request.prompt_plan.eval_token_count
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
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
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
        vocab: llama_cpp.llama_vocab_p,
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
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
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

    def sample(self, ctx: llama_cpp.llama_context_p, output_index: int) -> int:
        return int(llama_cpp.llama_sampler_sample(self._sampler, ctx, output_index))

    def _ensure_sample_logits_buffer(self, size: int) -> None:
        if size == self._sample_logits_size and self._sample_logits_recarray is not None:
            return
        token_data = (llama_cpp.llama_token_data * size)()
        token_data_address = ctypes.addressof(token_data)
        recarray = np.recarray(
            shape=(size,),
            dtype=self.TOKEN_DATA_DTYPE,
            buf=cast(
                Any,
                (llama_cpp.llama_token_data * size).from_address(token_data_address),
            ),
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
        llama_cpp.llama_sampler_apply(
            self._sampler,
            cast(Any, ctypes.byref(self._sample_logits_token_array)),
        )
        token = int(self._sample_logits_recarray.id[self._sample_logits_token_array.selected])
        llama_cpp.llama_sampler_accept(self._sampler, token)
        return token

    def close(self) -> None:
        if not self._closed:
            llama_cpp.llama_sampler_free(self._sampler)
            self._closed = True


@dataclass
class MTMDEmbedding:
    key: str
    embeddings: np.ndarray


class MTMDEmbeddingCache:
    _metadata_version = "1"

    def __init__(
        self,
        *,
        path: str,
        max_bytes: int,
        model_fingerprint: str,
        mmproj_fingerprint: str,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.model_fingerprint = model_fingerprint
        self.mmproj_fingerprint = mmproj_fingerprint
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_open(path: Path) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "model.mtmd.embedding_cache requires safetensors. "
                "Install it with `pip install safetensors`."
            ) from exc
        return safe_open(str(path), framework="numpy")

    @staticmethod
    def _save_file(
        tensors: Dict[str, np.ndarray],
        path: Path,
        metadata: Dict[str, str],
    ) -> None:
        from safetensors.numpy import save_file

        save_file(tensors, str(path), metadata=metadata)

    @staticmethod
    def fingerprint_file(path: str) -> str:
        stat = os.stat(path)
        payload = f"{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_key(
        *,
        model_fingerprint: str,
        mmproj_fingerprint: str,
        kind: Literal["image", "audio", "video"],
        media_bytes: bytes,
    ) -> str:
        digest = hashlib.sha256(f"{kind}:".encode("utf-8") + media_bytes).hexdigest()
        payload = ":".join(
            [
                MTMDEmbeddingCache._metadata_version,
                model_fingerprint,
                mmproj_fingerprint,
                kind,
                digest,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def key_for_media(self, kind: Literal["image", "audio", "video"], media_bytes: bytes) -> str:
        return self._build_key(
            model_fingerprint=self.model_fingerprint,
            mmproj_fingerprint=self.mmproj_fingerprint,
            kind=kind,
            media_bytes=media_bytes,
        )

    def _path_for_key(self, key: str) -> Path:
        return self.path / f"{key}.safetensors"

    def load(self, key: str) -> Optional[MTMDEmbedding]:
        entry_path = self._path_for_key(key)
        if not entry_path.exists():
            return None
        with self._safe_open(entry_path) as tensors:
            metadata = tensors.metadata()
            if metadata.get("version") != self._metadata_version:
                return None
            if metadata.get("model") != self.model_fingerprint:
                return None
            if metadata.get("mmproj") != self.mmproj_fingerprint:
                return None
            embeddings = np.array(tensors.get_tensor("embeddings"), copy=True)
        return MTMDEmbedding(key=key, embeddings=embeddings.astype(np.float32, copy=False))

    def save(self, key: str, embeddings: np.ndarray) -> None:
        if self.max_bytes == 0:
            return
        tmp_path = self.path / f".{key}.{uuid.uuid4().hex}.tmp"
        final_path = self._path_for_key(key)
        metadata = {
            "version": self._metadata_version,
            "model": self.model_fingerprint,
            "mmproj": self.mmproj_fingerprint,
            "key": key,
        }
        self._save_file(
            {"embeddings": np.ascontiguousarray(embeddings, dtype=np.float32)},
            tmp_path,
            metadata,
        )
        os.replace(tmp_path, final_path)
        self.evict_if_needed()

    def evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        entries = [path for path in self.path.glob("*.safetensors") if path.is_file()]
        total = sum(path.stat().st_size for path in entries)
        if total <= self.max_bytes:
            return
        for path in sorted(entries, key=lambda item: item.stat().st_mtime_ns):
            if total <= self.max_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
            except OSError:
                continue


@dataclass
class MTMDLoadedMedia:
    media: MediaInput
    media_bytes: bytes
    key: str
    bitmap: Any
    video_ctx: Optional[Any] = None
    video_temp_path: Optional[Path] = None
    video_callback: Optional[Any] = None
    video_frame_count: int = 0
    video_frames_used: int = 0


class MTMDProcessor:
    @dataclass
    class MediaChunk:
        kind: Literal["image", "audio", "video"]
        key: str
        chunk: Any
        n_tokens: int
        decode_n_pos: int
        non_causal: bool
        embeddings: Optional[np.ndarray] = None

    @dataclass
    class ParsedChunk:
        text_tokens: Optional[List[int]] = None
        media: Optional["MTMDProcessor.MediaChunk"] = None

    def __init__(
        self,
        *,
        model_path: str,
        llama_model: Any,
        chat_formatter: Jinja2ChatFormatter,
        tokenize: Callable[..., List[int]],
        n_embd_inp: int,
        n_batch: int,
        n_ubatch: int,
        n_threads_batch: int,
        mmproj_path: str,
        batch_max_tokens: int,
        embedding_cache: Optional[MTMDEmbeddingCache],
        allowed_media_domains: Optional[List[str]],
        allowed_local_media_path: Optional[str],
        image_max_bytes: int,
        audio_max_bytes: int,
        video_max_bytes: int,
        image_timeout_seconds: float,
    ) -> None:
        self.chat_formatter = chat_formatter
        self.tokenize = tokenize
        self.n_embd_inp = n_embd_inp
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.mmproj_path = mmproj_path
        self.embedding_cache = embedding_cache
        self.batch_max_tokens = batch_max_tokens
        self.model_fingerprint = MTMDEmbeddingCache.fingerprint_file(model_path)
        self.mmproj_fingerprint = MTMDEmbeddingCache.fingerprint_file(mmproj_path)
        self.allowed_media_domains = (
            {domain.lower() for domain in allowed_media_domains}
            if allowed_media_domains is not None
            else set()
        )
        self.allowed_local_media_path = (
            Path(allowed_local_media_path).expanduser().resolve()
            if allowed_local_media_path is not None
            else None
        )
        self.image_max_bytes = image_max_bytes
        self.audio_max_bytes = audio_max_bytes
        self.video_max_bytes = video_max_bytes
        self.image_timeout_seconds = image_timeout_seconds
        self.lock = threading.Lock()
        params = mtmd_cpp.mtmd_context_params_default()
        params.n_threads = max(1, n_threads_batch)
        params.batch_max_tokens = batch_max_tokens
        self.ctx = mtmd_cpp.mtmd_init_from_file(
            mmproj_path.encode("utf-8"),
            llama_model,
            params,
        )
        if self.ctx is None:
            raise RuntimeError(f"failed to load MTMD context: {mmproj_path}")
        self.supports_vision = bool(mtmd_cpp.mtmd_support_vision(self.ctx))
        self.supports_audio = bool(mtmd_cpp.mtmd_support_audio(self.ctx))
        self.supports_video = self.supports_vision and bool(
            mtmd_cpp.mtmd_helper_support_video(self.ctx)
        )
        if not self.supports_vision and not self.supports_audio:
            mtmd_cpp.mtmd_free(self.ctx)
            self.ctx = None
            raise RuntimeError(f"MTMD projector does not support image or audio input: {mmproj_path}")
        media_marker = mtmd_cpp.mtmd_get_marker(self.ctx)
        if media_marker is None:
            mtmd_cpp.mtmd_free(self.ctx)
            self.ctx = None
            raise RuntimeError(f"MTMD projector does not expose a media marker: {mmproj_path}")
        self.media_marker = media_marker.decode("utf-8")

    def close(self) -> None:
        if self.ctx is not None:
            mtmd_cpp.mtmd_free(self.ctx)
            self.ctx = None

    def _max_bytes_for_media(self, kind: Literal["image", "audio", "video"]) -> int:
        if kind == "image":
            return self.image_max_bytes
        if kind == "audio":
            return self.audio_max_bytes
        return self.video_max_bytes

    def _load_media_file(self, kind: Literal["image", "audio", "video"], media_url: str) -> bytes:
        if self.allowed_local_media_path is None:
            raise CompletionRequestValidationError("local media path is not allowed")
        parsed = urllib.parse.urlsplit(media_url)
        if parsed.netloc not in {"", "localhost"}:
            raise CompletionRequestValidationError("local media path is not allowed")
        path = Path(urllib.parse.unquote(parsed.path)).expanduser().resolve()
        try:
            path.relative_to(self.allowed_local_media_path)
        except ValueError as exc:
            raise CompletionRequestValidationError("local media path is not allowed") from exc
        max_bytes = self._max_bytes_for_media(kind)
        try:
            if path.stat().st_size > max_bytes:
                raise CompletionRequestValidationError(f"{kind} exceeds model.mtmd.{kind}_max_bytes")
            data = path.read_bytes()
        except OSError as exc:
            raise CompletionRequestValidationError(f"failed to read local {kind}: {exc}") from exc
        if len(data) > max_bytes:
            raise CompletionRequestValidationError(f"{kind} exceeds model.mtmd.{kind}_max_bytes")
        return data

    def _validate_remote_media_url(self, media_url: str) -> str:
        parsed = urllib.parse.urlsplit(media_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise CompletionRequestValidationError(
                "only data:, file:, http:, and https: media URLs are supported"
            )
        if parsed.username is not None or parsed.password is not None:
            raise CompletionRequestValidationError("remote media domain is not allowed")
        hostname = parsed.hostname.lower()
        if self.allowed_media_domains and hostname not in self.allowed_media_domains:
            raise CompletionRequestValidationError("remote media domain is not allowed")
        return urllib.parse.urlunsplit(parsed)

    @staticmethod
    def _urlopen_without_redirects(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> Any:
        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
                return None

        opener = urllib.request.build_opener(NoRedirectHandler)
        return opener.open(request, timeout=timeout)

    def _load_media_url(self, kind: Literal["image", "audio", "video"], media_url: str) -> bytes:
        max_bytes = self._max_bytes_for_media(kind)
        if media_url.startswith("data:"):
            try:
                _, encoded = media_url.split(",", 1)
            except ValueError as exc:
                raise CompletionRequestValidationError(f"invalid data {kind} URL") from exc
            data = base64.b64decode(encoded, validate=False)
            if len(data) > max_bytes:
                raise CompletionRequestValidationError(f"{kind} exceeds model.mtmd.{kind}_max_bytes")
            return data
        if media_url.startswith("file:"):
            return self._load_media_file(kind, media_url)
        media_url = self._validate_remote_media_url(media_url)
        request = urllib.request.Request(
            media_url,
            headers={"User-Agent": "llama-cpp-python-batch-mtmd/0"},
        )
        try:
            with self._urlopen_without_redirects(request, timeout=self.image_timeout_seconds) as response:
                data = response.read(max_bytes + 1)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise CompletionRequestValidationError(
                f"failed to fetch {kind} URL: {exc}"
            ) from exc
        if len(data) > max_bytes:
            raise CompletionRequestValidationError(f"{kind} exceeds model.mtmd.{kind}_max_bytes")
        return data

    def load_media(self, media: MediaInput) -> bytes:
        if media.url is not None:
            return self._load_media_url(media.kind, media.url)
        if media.kind not in {"audio", "video"} or media.data is None:
            raise CompletionRequestValidationError(f"{media.kind} input requires a URL")
        try:
            data = base64.b64decode(media.data, validate=False)
        except (ValueError, binascii.Error) as exc:
            raise CompletionRequestValidationError(f"input_{media.kind} data must be valid base64") from exc
        max_bytes = self._max_bytes_for_media(media.kind)
        if len(data) > max_bytes:
            raise CompletionRequestValidationError(
                f"{media.kind} exceeds model.mtmd.{media.kind}_max_bytes"
            )
        return data

    def _create_loaded_media(
        self,
        media: MediaInput,
        media_bytes: bytes,
    ) -> MTMDLoadedMedia:
        key = (
            self.embedding_cache.key_for_media(media.kind, media_bytes)
            if self.embedding_cache is not None
            else MTMDEmbeddingCache._build_key(
                model_fingerprint=self.model_fingerprint,
                mmproj_fingerprint=self.mmproj_fingerprint,
                kind=media.kind,
                media_bytes=media_bytes,
            )
        )
        if media.kind == "video":
            return self._create_loaded_video_media(media, media_bytes, key)
        buffer = (ctypes.c_uint8 * len(media_bytes)).from_buffer_copy(media_bytes)
        wrapper = mtmd_cpp.mtmd_helper_bitmap_init_from_buf_wrapper(
            self.ctx,
            buffer,
            len(media_bytes),
            False,
        )
        bitmap = wrapper.bitmap
        if bitmap is None:
            raise CompletionRequestValidationError(f"failed to create MTMD {media.kind} bitmap")
        mtmd_cpp.mtmd_bitmap_set_id(bitmap, key.encode("utf-8"))
        video_frame_count = 0
        video_ctx = wrapper.video_ctx
        if video_ctx:
            video_info = mtmd_cpp.mtmd_helper_video_get_info(video_ctx)
            video_frame_count = max(0, int(video_info.n_frames))
        return MTMDLoadedMedia(
            media=media,
            media_bytes=media_bytes,
            key=key,
            bitmap=bitmap,
            video_ctx=video_ctx,
            video_frame_count=video_frame_count,
        )

    @staticmethod
    def _video_temp_suffix(media: MediaInput) -> str:
        extension = (media.format or "mp4").lstrip(".").lower()
        if not extension or any(not char.isalnum() for char in extension):
            extension = "video"
        return f".{extension}"

    def _create_loaded_video_media(
        self,
        media: MediaInput,
        media_bytes: bytes,
        key: str,
    ) -> MTMDLoadedMedia:
        temp_file = tempfile.NamedTemporaryFile(
            prefix="llama-cpp-python-mtmd-",
            suffix=self._video_temp_suffix(media),
            delete=False,
        )
        temp_path = Path(temp_file.name)
        try:
            with temp_file:
                temp_file.write(media_bytes)
            params = mtmd_cpp.mtmd_helper_video_init_params_default()
            video_ctx = mtmd_cpp.mtmd_helper_video_init(
                self.ctx,
                str(temp_path).encode("utf-8"),
                params,
            )
            if video_ctx is None:
                raise CompletionRequestValidationError("failed to create MTMD video context")

            def read_next(
                _chunk_index: int,
                _user_data: Any,
                out_bitmap: Any,
                out_text: Any,
            ) -> int:
                return int(mtmd_cpp.mtmd_helper_video_read_next(video_ctx, out_bitmap, out_text))

            callback = mtmd_cpp.mtmd_bitmap_lazy_callback(read_next)
            bitmap = mtmd_cpp.mtmd_bitmap_init_lazy(
                self.ctx,
                key.encode("utf-8"),
                ctypes.c_void_p(),
                callback,
            )
            if bitmap is None:
                mtmd_cpp.mtmd_helper_video_free(video_ctx)
                raise CompletionRequestValidationError("failed to create MTMD video bitmap")
            video_info = mtmd_cpp.mtmd_helper_video_get_info(video_ctx)
            return MTMDLoadedMedia(
                media=media,
                media_bytes=media_bytes,
                key=key,
                bitmap=bitmap,
                video_ctx=video_ctx,
                video_temp_path=temp_path,
                video_callback=callback,
                video_frame_count=max(0, int(video_info.n_frames)),
            )
        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _media_identity_tokens(
        self,
        kind: Literal["image", "audio", "video"],
        key: str,
        n_pos: int,
    ) -> List[int]:
        tokens: List[int] = []
        for index in range(n_pos):
            digest = hashlib.sha256(f"{kind}:{key}:{index}".encode("utf-8")).digest()
            tokens.append(-1 - (int.from_bytes(digest[:4], "little") & 0x3FFFFFFF))
        return tokens

    def _embeddings_from_pointer(self, output: Any, n_tokens: int) -> np.ndarray:
        flat = np.ctypeslib.as_array(output, shape=(n_tokens * self.n_embd_inp,))
        return np.array(flat, dtype=np.float32, copy=True).reshape(
            n_tokens,
            self.n_embd_inp,
        )

    def _load_cached_media_chunk(self, media_chunk: "MTMDProcessor.MediaChunk") -> bool:
        if self.embedding_cache is None:
            return False
        cached = self.embedding_cache.load(media_chunk.key)
        if cached is None or cached.embeddings.shape != (
            media_chunk.n_tokens,
            self.n_embd_inp,
        ):
            return False
        media_chunk.embeddings = cached.embeddings
        return True

    def _save_media_chunk(self, media_chunk: "MTMDProcessor.MediaChunk") -> None:
        if self.embedding_cache is None or media_chunk.embeddings is None:
            return
        self.embedding_cache.save(media_chunk.key, media_chunk.embeddings)

    def _encode_media_batch(
        self,
        media_chunks: Sequence["MTMDProcessor.MediaChunk"],
        start_index: int,
    ) -> int:
        batch = mtmd_cpp.mtmd_batch_init(self.ctx)
        if batch is None:
            raise CompletionRequestValidationError("failed to create MTMD media batch")
        try:
            first = media_chunks[start_index]
            result = int(mtmd_cpp.mtmd_batch_add_chunk(batch, first.chunk))
            if result != 0:
                raise CompletionRequestValidationError(
                    f"failed to add {first.kind} chunk to MTMD batch: error code {result}"
                )
            group = [first]
            next_index = start_index + 1
            while next_index < len(media_chunks):
                candidate = media_chunks[next_index]
                result = int(mtmd_cpp.mtmd_batch_add_chunk(batch, candidate.chunk))
                if result == 0:
                    group.append(candidate)
                    next_index += 1
                    continue
                if result in {2, 3}:
                    break
                raise CompletionRequestValidationError(
                    f"failed to add {candidate.kind} chunk to MTMD batch: error code {result}"
                )
            result = int(mtmd_cpp.mtmd_batch_encode(batch))
            if result != 0:
                raise CompletionRequestValidationError(
                    f"failed to encode MTMD media batch: error code {result}"
                )
            for media_chunk in group:
                output = mtmd_cpp.mtmd_batch_get_output_embd(batch, media_chunk.chunk)
                if output is None:
                    raise CompletionRequestValidationError(
                        f"MTMD {media_chunk.kind} encoder returned no embeddings"
                    )
                media_chunk.embeddings = self._embeddings_from_pointer(
                    output,
                    media_chunk.n_tokens,
                )
                self._save_media_chunk(media_chunk)
            return len(group)
        finally:
            mtmd_cpp.mtmd_batch_free(batch)

    def _encode_media_chunks(
        self,
        media_chunks: Sequence["MTMDProcessor.MediaChunk"],
    ) -> None:
        uncached = [
            media_chunk
            for media_chunk in media_chunks
            if not self._load_cached_media_chunk(media_chunk)
        ]
        index = 0
        while index < len(uncached):
            index += self._encode_media_batch(uncached, index)

    def _positions_for_chunk(self, chunk: Any, start_pos: int) -> np.ndarray:
        n_tokens = int(mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk))
        if not mtmd_cpp.mtmd_decode_use_mrope(self.ctx):
            return np.arange(start_pos, start_pos + n_tokens, dtype=np.int32)
        chunk_type = int(mtmd_cpp.mtmd_input_chunk_get_type(chunk))
        if chunk_type == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO:
            positions = np.empty((4, n_tokens), dtype=np.int32)
            positions[:] = np.arange(start_pos, start_pos + n_tokens, dtype=np.int32)
            return positions.reshape(-1)
        image_tokens = mtmd_cpp.mtmd_input_chunk_get_tokens_image(chunk)
        if image_tokens is None:
            raise CompletionRequestValidationError("MTMD image chunk has no image tokens")
        positions = np.empty((4, n_tokens), dtype=np.int32)
        for index in range(n_tokens):
            pos = mtmd_cpp.mtmd_image_tokens_get_decoder_pos(
                image_tokens,
                llama_cpp.llama_pos(start_pos),
                index,
            )
            positions[0, index] = int(pos.t)
            positions[1, index] = int(pos.y)
            positions[2, index] = int(pos.x)
            positions[3, index] = int(pos.z)
        return positions.reshape(-1)

    def build_prompt_plan(
        self,
        *,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        function_call: Optional[Union[str, ChatTemplateFunctionCall]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        tool_choice: Optional[Union[str, ChatTemplateToolChoice]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> PromptPlan:
        media_inputs = Jinja2ChatFormatter.media_inputs_from_messages(messages)
        if not media_inputs:
            prompt, generation_prompt, _ = self.chat_formatter.format(
                messages=messages,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
            )
            tokens = self.tokenize(prompt, add_bos=False, special=True)
            return PromptPlan.from_tokens(prompt, tokens, generation_prompt=generation_prompt)
        if any(media.kind == "image" for media in media_inputs) and not self.supports_vision:
            raise CompletionRequestValidationError("MTMD projector does not support images")
        if any(media.kind == "audio" for media in media_inputs) and not self.supports_audio:
            raise CompletionRequestValidationError("MTMD projector does not support audio")
        if any(media.kind == "video" for media in media_inputs) and not self.supports_video:
            raise CompletionRequestValidationError("MTMD projector does not support video")
        with self.lock:
            return self._build_prompt_plan_locked(
                messages=messages,
                media_inputs=media_inputs,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
            )

    def _build_prompt_plan_locked(
        self,
        *,
        messages: List[ChatCompletionRequestMessage],
        media_inputs: List[MediaInput],
        functions: Optional[List[ChatTemplateFunctionDefinition]],
        function_call: Optional[Union[str, ChatTemplateFunctionCall]],
        tools: Optional[List[ChatTemplateTool]],
        tool_choice: Optional[Union[str, ChatTemplateToolChoice]],
        reasoning_effort: Optional[str],
    ) -> PromptPlan:
        prompt, generation_prompt, _ = self.chat_formatter.format(
            messages=messages,
            media_marker=self.media_marker,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        )
        media_bytes_by_index = [self.load_media(media) for media in media_inputs]
        loaded_media: List[MTMDLoadedMedia] = []
        chunks: Optional[Any] = None
        try:
            loaded_media = [
                self._create_loaded_media(media, media_bytes)
                for media, media_bytes in zip(media_inputs, media_bytes_by_index)
            ]
            loaded_media_by_key = {media.key: media for media in loaded_media}
            video_media = [media for media in loaded_media if media.media.kind == "video"]
            if len(video_media) > 1 and any(media.video_frame_count <= 0 for media in video_media):
                raise CompletionRequestValidationError(
                    "multiple videos require MTMD to report frame counts"
                )
            input_text = mtmd_cpp.mtmd_input_text()
            input_text.text = prompt.encode("utf-8")
            input_text.add_special = False
            input_text.parse_special = True
            chunks = mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise CompletionRequestValidationError("failed to create MTMD input chunks")
            bitmap_array = (mtmd_cpp.mtmd_bitmap_p_ctypes * len(loaded_media))(
                *(media.bitmap for media in loaded_media)
            )
            result = int(
                mtmd_cpp.mtmd_tokenize(
                    self.ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(loaded_media),
                )
            )
            if result != 0:
                raise CompletionRequestValidationError(
                    f"failed to tokenize MTMD prompt: error code {result}"
                )
            parsed_chunks: List[MTMDProcessor.ParsedChunk] = []
            media_chunks: List[MTMDProcessor.MediaChunk] = []
            video_index = 0
            used_media_keys = set()
            n_chunks = int(mtmd_cpp.mtmd_input_chunks_size(chunks))
            for chunk_index in range(n_chunks):
                chunk = mtmd_cpp.mtmd_input_chunks_get(chunks, chunk_index)
                if chunk is None:
                    continue
                chunk_type = int(mtmd_cpp.mtmd_input_chunk_get_type(chunk))
                if chunk_type == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n_tokens_out = ctypes.c_size_t()
                    tokens_ptr = mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                        chunk,
                        ctypes.byref(n_tokens_out),
                    )
                    tokens = (
                        [int(tokens_ptr[index]) for index in range(int(n_tokens_out.value))]
                        if tokens_ptr
                        else []
                    )
                    if tokens:
                        parsed_chunks.append(
                            MTMDProcessor.ParsedChunk(text_tokens=tokens)
                        )
                    continue
                if chunk_type == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    chunk_kind: Literal["image", "audio"] = "image"
                    if not self.supports_vision:
                        raise CompletionRequestValidationError("MTMD projector does not support images")
                elif chunk_type == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO:
                    chunk_kind = "audio"
                    if not self.supports_audio:
                        raise CompletionRequestValidationError("MTMD projector does not support audio")
                else:
                    raise CompletionRequestValidationError("unsupported MTMD input chunk type")
                chunk_id_bytes = mtmd_cpp.mtmd_input_chunk_get_id(chunk)
                chunk_id = chunk_id_bytes.decode("utf-8") if chunk_id_bytes else ""
                media = loaded_media_by_key.get(chunk_id)
                video_frame_index: Optional[int] = None
                if media is None and chunk_kind == "image" and video_media:
                    while (
                        video_index < len(video_media)
                        and video_media[video_index].video_frame_count > 0
                        and video_media[video_index].video_frames_used
                        >= video_media[video_index].video_frame_count
                    ):
                        video_index += 1
                    if video_index >= len(video_media):
                        raise CompletionRequestValidationError("MTMD video frame count mismatch")
                    media = video_media[video_index]
                    video_frame_index = media.video_frames_used
                    media.video_frames_used += 1
                if media is None:
                    raise CompletionRequestValidationError("MTMD media chunk identity mismatch")
                if media.media.kind == "video":
                    if chunk_kind != "image":
                        raise CompletionRequestValidationError("MTMD video chunk modality mismatch")
                    kind: Literal["image", "audio", "video"] = "video"
                    if video_frame_index is None:
                        video_frame_index = media.video_frames_used
                        media.video_frames_used += 1
                    key = hashlib.sha256(
                        f"{media.key}:frame:{video_frame_index}".encode("utf-8")
                    ).hexdigest()
                else:
                    if media.media.kind != chunk_kind:
                        raise CompletionRequestValidationError("MTMD media chunk modality mismatch")
                    kind = media.media.kind
                    key = media.key
                used_media_keys.add(media.key)
                decode_n_pos = int(mtmd_cpp.mtmd_input_chunk_get_n_pos(chunk))
                if decode_n_pos <= 0:
                    raise CompletionRequestValidationError("MTMD media chunk has no decoder positions")
                n_tokens = int(mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk))
                if n_tokens <= 0:
                    raise CompletionRequestValidationError("MTMD media chunk has no embedding tokens")
                non_causal = bool(mtmd_cpp.mtmd_decode_use_non_causal(self.ctx, chunk))
                media_chunk = MTMDProcessor.MediaChunk(
                    kind=kind,
                    key=key,
                    chunk=chunk,
                    n_tokens=n_tokens,
                    decode_n_pos=decode_n_pos,
                    non_causal=non_causal,
                )
                parsed_chunks.append(MTMDProcessor.ParsedChunk(media=media_chunk))
                media_chunks.append(media_chunk)
            if used_media_keys != {media.key for media in loaded_media}:
                raise CompletionRequestValidationError("not all media inputs were consumed by MTMD")
            self._encode_media_chunks(media_chunks)
            segments: List[PromptSegment] = []
            identity_tokens: List[int] = []
            text_tokens: List[int] = []
            text_token_index_by_pos: Dict[int, int] = {}
            identity_pos = 0
            decode_pos = 0
            for parsed_chunk in parsed_chunks:
                if parsed_chunk.text_tokens is not None:
                    tokens = parsed_chunk.text_tokens
                    start_pos = identity_pos
                    segments.append(
                        PromptSegment(
                            kind="text",
                            start_pos=start_pos,
                            n_pos=len(tokens),
                            identity_tokens=list(tokens),
                            decode_start_pos=decode_pos,
                            decode_n_pos=len(tokens),
                            text_tokens=list(tokens),
                        )
                    )
                    for offset, token in enumerate(tokens):
                        text_token_index_by_pos[start_pos + offset] = len(text_tokens)
                        text_tokens.append(token)
                    identity_tokens.extend(tokens)
                    identity_pos += len(tokens)
                    decode_pos += len(tokens)
                    continue
                media_chunk = parsed_chunk.media
                if media_chunk is None or media_chunk.embeddings is None:
                    raise CompletionRequestValidationError("MTMD media chunk has no embeddings")
                embeddings = media_chunk.embeddings
                if media_chunk.non_causal and embeddings.shape[0] > min(self.n_batch, self.n_ubatch):
                    raise CompletionRequestValidationError(
                        f"non-causal {media_chunk.kind} embedding chunk exceeds model batch limits; "
                        "increase n_batch and n_ubatch"
                    )
                segment_identity = self._media_identity_tokens(
                    media_chunk.kind,
                    media_chunk.key,
                    media_chunk.n_tokens,
                )
                positions = self._positions_for_chunk(media_chunk.chunk, decode_pos)
                segments.append(
                    PromptSegment(
                        kind=media_chunk.kind,
                        start_pos=identity_pos,
                        n_pos=media_chunk.n_tokens,
                        identity_tokens=segment_identity,
                        decode_start_pos=decode_pos,
                        decode_n_pos=media_chunk.decode_n_pos,
                        media=PromptSegment.Media(
                            embeddings=embeddings,
                            positions=positions,
                            non_causal=media_chunk.non_causal,
                        ),
                    )
                )
                identity_tokens.extend(segment_identity)
                identity_pos += media_chunk.n_tokens
                decode_pos += media_chunk.decode_n_pos
            return PromptPlan(
                text=prompt,
                generation_prompt=generation_prompt,
                text_tokens=text_tokens,
                identity_tokens=identity_tokens,
                segments=segments,
                text_token_index_by_pos=text_token_index_by_pos,
            )
        finally:
            if chunks is not None:
                mtmd_cpp.mtmd_input_chunks_free(chunks)
            for media in loaded_media:
                mtmd_cpp.mtmd_bitmap_free(media.bitmap)
                if media.video_ctx:
                    mtmd_cpp.mtmd_helper_video_free(media.video_ctx)
                if media.video_temp_path is not None:
                    try:
                        media.video_temp_path.unlink()
                    except OSError:
                        pass


class Model:
    @dataclass(frozen=True)
    class LoraAdapter:
        path: str
        scale: float = 1.0

    def __init__(
        self,
        *,
        model_path: str,
        model_alias: Optional[str] = None,
        chat_template: Optional[str] = None,
        loras: Optional[List["Model.LoraAdapter"]] = None,
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
        embedding: Optional[bool] = None,
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
        draft_model_path: Optional[str] = None,
        draft_model_num_pred_tokens: int = 16,
        draft_model_max_ngram_size: int = 2,
        draft_model_top_k: int = 1,
        draft_model_p_min: float = 0.0,
        draft_model_max_batch_size: Optional[int] = None,
        draft_model_threads: Optional[int] = None,
        draft_model_threads_batch: Optional[int] = None,
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
        self.draft_model_max_batch_size = draft_model_max_batch_size
        self.draft_provider: Optional[DraftProvider] = None
        self.loras = list(loras or [])
        self._lora_adapters: List[Any] = []
        self._lora_adapter_array: Optional[Any] = None
        self._lora_scales_array: Optional[Any] = None
        self.draft_llama_model: Optional[Any] = None
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
        vocab = llama_cpp.llama_model_get_vocab(llama_model)
        if vocab is None:
            raise RuntimeError("failed to access model vocabulary")
        self.vocab = vocab
        embedding = self.resolve_embedding_mode(llama_model, embedding)
        self.embedding = embedding
        self.has_encoder = bool(llama_cpp.llama_model_has_encoder(llama_model))
        self.has_decoder = bool(llama_cpp.llama_model_has_decoder(llama_model))
        if self.has_encoder and not embedding:
            raise RuntimeError("encoder models are not supported")
        if not self.has_decoder and not (embedding and self.has_encoder):
            raise RuntimeError("decoder is required")
        if llama_cpp.llama_model_is_recurrent(llama_model):
            self.memory_model = "recurrent"
        elif llama_cpp.llama_model_is_hybrid(llama_model):
            self.memory_model = "hybrid"
        else:
            self.memory_model = (
                "attention-unified" if kv_unified else "attention-partitioned"
        )
        normalized_draft_model = draft_model
        required_mtp_batch = max(1, draft_model_num_pred_tokens + 1)
        if normalized_draft_model == "draft-mtp":
            if n_batch is not None and n_batch < required_mtp_batch:
                raise RuntimeError(
                    "MTP requires n_batch to fit the pending token plus draft tokens "
                    f"(required {required_mtp_batch}, got {n_batch})"
                )
        self.draft_target_batching = normalized_draft_model is not None
        if (
            normalized_draft_model is not None
            and normalized_draft_model != "draft-mtp"
            and not self.memory_model.startswith("attention")
        ):
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
            embedding=embedding,
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
            n_rs_seq=None,
            ctx_type=None,
        )
        ctx = llama_cpp.llama_init_from_model(llama_model, context_params)
        if ctx is None:
            raise RuntimeError("failed to create context")
        self.ctx = ctx
        mem = llama_cpp.llama_get_memory(ctx)
        if mem is None and not embedding:
            raise RuntimeError("failed to access model memory")
        self.mem = mem
        self.n_ctx = int(llama_cpp.llama_n_ctx(ctx))
        self.n_ctx_seq = int(llama_cpp.llama_n_ctx_seq(ctx))
        self.n_seq_max = int(llama_cpp.llama_n_seq_max(ctx))
        self.n_rs_seq = int(llama_cpp.llama_n_rs_seq(ctx))
        self.n_batch = int(llama_cpp.llama_n_batch(ctx))
        self.n_ubatch = int(llama_cpp.llama_n_ubatch(ctx))
        self.n_threads_batch = (
            n_threads_batch
            if n_threads_batch is not None
            else max(multiprocessing.cpu_count(), 1)
        )
        self.mtmd_processor: Optional[MTMDProcessor] = None
        self._embedding_batch: Optional[llama_cpp.llama_batch] = None
        self._embedding_batch_refs: List[Any] = []
        if normalized_draft_model == "draft-mtp" and self.n_batch < required_mtp_batch:
            raise RuntimeError(
                "MTP requires runtime n_batch to fit the pending token plus draft tokens "
                f"(required {required_mtp_batch}, got {self.n_batch})"
            )
        self.n_ctx_train = n_ctx_train
        self.n_vocab = int(llama_cpp.llama_vocab_n_tokens(self.vocab))
        self.n_embd = int(llama_cpp.llama_model_n_embd(self.llama_model))
        self.n_embd_inp = int(llama_cpp.llama_model_n_embd_inp(self.llama_model))
        self.n_embd_out = int(llama_cpp.llama_model_n_embd_out(self.llama_model))
        if self.n_embd_out <= 0:
            self.n_embd_out = self.n_embd
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
        if normalized_draft_model is None:
            self.draft_provider = None
        elif normalized_draft_model == "prompt-lookup-decoding":
            self.draft_provider = PromptLookupDecoding(
                max_ngram_size=draft_model_max_ngram_size,
                num_pred_tokens=draft_model_num_pred_tokens,
            )
        elif normalized_draft_model == "draft-mtp":
            draft_llama_model = self.llama_model
            if draft_model_path is not None:
                draft_llama_model = llama_cpp.llama_model_load_from_file(
                    draft_model_path.encode("utf-8"),
                    model_params,
                )
                if draft_llama_model is None:
                    llama_cpp.llama_batch_free(self.batch)
                    llama_cpp.llama_free(self.ctx)
                    self._free_lora_adapters()
                    llama_cpp.llama_model_free(self.llama_model)
                    if self.backend_initialized:
                        llama_cpp.llama_backend_free()
                        self.backend_initialized = False
                    raise RuntimeError(f"failed to load MTP draft model: {draft_model_path}")
                self.draft_llama_model = draft_llama_model
            if self.n_ubatch < self.n_seq_max:
                mtp_n_batch = self.n_batch
            else:
                mtp_n_batch = min(
                    self.n_batch,
                    max(self.n_ubatch, self.n_seq_max, required_mtp_batch),
                )
            mtp_context_params = self.build_context_params(
                n_ctx=self.n_ctx,
                n_batch=mtp_n_batch,
                n_ubatch=min(self.n_ubatch, mtp_n_batch),
                n_seq_max=self.n_seq_max,
                n_threads=(
                    draft_model_threads
                    if draft_model_threads is not None
                    else n_threads
                ),
                n_threads_batch=(
                    draft_model_threads_batch
                    if draft_model_threads_batch is not None
                    else (
                        draft_model_threads
                        if draft_model_threads is not None
                        else n_threads_batch
                    )
                ),
                rope_scaling_type=rope_scaling_type,
                pooling_type=pooling_type,
                attention_type=attention_type,
                embedding=embedding,
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
                n_rs_seq=0,
                ctx_type=llama_cpp.LLAMA_CONTEXT_TYPE_MTP,
                n_outputs_max=min(mtp_n_batch, self.n_seq_max),
                ctx_other=self.ctx,
            )
            try:
                self.draft_provider = MTPDraftProvider(
                    model=self,
                    draft_model=draft_llama_model,
                    context_params=mtp_context_params,
                    num_pred_tokens=draft_model_num_pred_tokens,
                    top_k=draft_model_top_k,
                    p_min=draft_model_p_min,
                )
            except BaseException:
                llama_cpp.llama_batch_free(self.batch)
                llama_cpp.llama_free(self.ctx)
                self._free_lora_adapters()
                if self.draft_llama_model is not None:
                    llama_cpp.llama_model_free(self.draft_llama_model)
                    self.draft_llama_model = None
                llama_cpp.llama_model_free(self.llama_model)
                if self.backend_initialized:
                    llama_cpp.llama_backend_free()
                    self.backend_initialized = False
                raise
        else:
            raise RuntimeError(f"unsupported draft model: {draft_model}")
        try:
            self._load_lora_adapters(self.loras)
            self._apply_lora_adapters(self.ctx, "target")
            if (
                isinstance(self.draft_provider, MTPDraftProvider)
                and self.draft_llama_model is None
            ):
                self._apply_lora_adapters(self.draft_provider.ctx, "MTP draft")
        except BaseException:
            if self.draft_provider is not None:
                self.draft_provider.close()
            llama_cpp.llama_batch_free(self.batch)
            llama_cpp.llama_free(self.ctx)
            self._free_lora_adapters()
            if self.draft_llama_model is not None:
                llama_cpp.llama_model_free(self.draft_llama_model)
                self.draft_llama_model = None
            llama_cpp.llama_model_free(self.llama_model)
            if self.backend_initialized:
                llama_cpp.llama_backend_free()
                self.backend_initialized = False
            raise
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
                        + cast(
                            Any,
                            llama_cpp.llama_model_kv_override_value.val_str,
                        ).offset,
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
        embedding: bool,
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
        n_rs_seq: Optional[int] = None,
        ctx_type: Optional[int] = None,
        n_outputs_max: Optional[int] = None,
        ctx_other: Optional[Any] = None,
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
        if n_rs_seq is not None:
            context_params.n_rs_seq = n_rs_seq
        if n_threads is not None:
            context_params.n_threads = n_threads
        if n_threads_batch is not None:
            context_params.n_threads_batch = n_threads_batch
        if ctx_type is not None:
            context_params.ctx_type = ctx_type
        if n_outputs_max is not None:
            context_params.n_outputs_max = n_outputs_max
        if ctx_other is not None:
            context_params.ctx_other = ctx_other
        if rope_scaling_type is not None:
            context_params.rope_scaling_type = rope_scaling_type
        if pooling_type is not None:
            context_params.pooling_type = pooling_type
        if attention_type is not None:
            context_params.attention_type = attention_type
        context_params.embeddings = embedding
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

    def _load_lora_adapters(self, loras: List["Model.LoraAdapter"]) -> None:
        for lora in loras:
            adapter = llama_cpp.llama_adapter_lora_init(
                self.llama_model,
                lora.path.encode("utf-8"),
            )
            if adapter is None:
                raise RuntimeError(f"failed to load LoRA adapter: {lora.path}")
            self._lora_adapters.append(adapter)

        if not self._lora_adapters:
            return

        adapter_array_type = llama_cpp.llama_adapter_lora_p_ctypes * len(
            self._lora_adapters
        )
        scale_array_type = ctypes.c_float * len(self._lora_adapters)
        self._lora_adapter_array = adapter_array_type(*self._lora_adapters)
        self._lora_scales_array = scale_array_type(
            *(float(lora.scale) for lora in loras)
        )

    def _apply_lora_adapters(self, ctx: Any, context_name: str) -> None:
        if not self._lora_adapters:
            return
        if self._lora_adapter_array is None or self._lora_scales_array is None:
            raise RuntimeError("LoRA adapter arrays are not initialized")
        result = llama_cpp.llama_set_adapters_lora(
            ctx,
            self._lora_adapter_array,
            len(self._lora_adapters),
            self._lora_scales_array,
        )
        if result:
            raise RuntimeError(f"failed to apply LoRA adapters to {context_name} context")

    def _free_lora_adapters(self) -> None:
        while self._lora_adapters:
            adapter = self._lora_adapters.pop()
            llama_cpp.llama_adapter_lora_free(adapter)
        self._lora_adapter_array = None
        self._lora_scales_array = None

    def close(self) -> None:
        if self.mtmd_processor is not None:
            self.mtmd_processor.close()
            self.mtmd_processor = None
        if self.draft_provider is not None:
            self.draft_provider.close()
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_free(self.ctx)
        self._free_lora_adapters()
        if self.draft_llama_model is not None:
            llama_cpp.llama_model_free(self.draft_llama_model)
            self.draft_llama_model = None
        llama_cpp.llama_model_free(self.llama_model)
        if self.backend_initialized:
            llama_cpp.llama_backend_free()
            self.backend_initialized = False

    @staticmethod
    def _model_meta_key_by_index(llama_model: Any, index: int) -> Optional[str]:
        capacity = 256
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            count = int(
                llama_cpp.llama_model_meta_key_by_index(
                    llama_model,
                    index,
                    cast(Any, buffer),
                    capacity,
                )
            )
            if count < 0:
                return None
            if count < capacity:
                return buffer.value.decode("utf-8", errors="ignore")
            capacity = count + 1

    @staticmethod
    def _model_meta_value(llama_model: Any, key: str) -> Optional[str]:
        encoded = key.encode("utf-8")
        capacity = 256
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            count = int(
                llama_cpp.llama_model_meta_val_str(
                    llama_model,
                    encoded,
                    cast(Any, buffer),
                    capacity,
                )
            )
            if count < 0:
                return None
            if count < capacity:
                return buffer.value.decode("utf-8", errors="ignore")
            capacity = count + 1

    @staticmethod
    def _parse_pooling_type(value: str) -> Optional[int]:
        normalized = value.strip().lower()
        try:
            return int(normalized)
        except ValueError:
            return {
                "none": llama_cpp.LLAMA_POOLING_TYPE_NONE,
                "mean": llama_cpp.LLAMA_POOLING_TYPE_MEAN,
                "cls": llama_cpp.LLAMA_POOLING_TYPE_CLS,
                "last": llama_cpp.LLAMA_POOLING_TYPE_LAST,
                "rank": llama_cpp.LLAMA_POOLING_TYPE_RANK,
            }.get(normalized)

    @classmethod
    def detect_embedding_model(cls, llama_model: Any) -> bool:
        for index in range(int(llama_cpp.llama_model_meta_count(llama_model))):
            key = cls._model_meta_key_by_index(llama_model, index)
            if key is None or not key.endswith(".pooling_type"):
                continue
            value = cls._model_meta_value(llama_model, key)
            if value is None:
                continue
            pooling_type = cls._parse_pooling_type(value)
            return pooling_type in {
                llama_cpp.LLAMA_POOLING_TYPE_MEAN,
                llama_cpp.LLAMA_POOLING_TYPE_CLS,
                llama_cpp.LLAMA_POOLING_TYPE_LAST,
            }
        return False

    @classmethod
    def resolve_embedding_mode(
        cls,
        llama_model: Any,
        embedding: Optional[bool],
    ) -> bool:
        if embedding is not None:
            return embedding
        return cls.detect_embedding_model(llama_model)

    def _meta_value(self, key: str) -> Optional[str]:
        return self._model_meta_value(self.llama_model, key)

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
        functions: Optional[List[ChatTemplateFunctionDefinition]] = None,
        function_call: Optional[Union[str, ChatTemplateFunctionCall]] = None,
        tools: Optional[List[ChatTemplateTool]] = None,
        tool_choice: Optional[Union[str, ChatTemplateToolChoice]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[str, str, PromptPlan, List[str]]:
        if self.chat_formatter is None:
            raise ValueError("model does not provide a GGUF chat template")
        if self.mtmd_processor is not None:
            prompt_plan = self.mtmd_processor.build_prompt_plan(
                messages=messages,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
            )
            formatter_stop = [self.chat_formatter._eos_token] if self.chat_formatter._eos_token else []
            return (
                prompt_plan.text,
                prompt_plan.generation_prompt,
                prompt_plan,
                formatter_stop,
            )
        prompt, generation_prompt, formatter_stop = self.chat_formatter.format(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        )
        prompt_tokens = self.tokenize(prompt, add_bos=False, special=True)
        return (
            prompt,
            generation_prompt,
            PromptPlan.from_tokens(
                prompt,
                prompt_tokens,
                generation_prompt=generation_prompt,
            ),
            formatter_stop,
        )

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
                    cast(Any, buffer),
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
        self._embedding_batch = None
        self._embedding_batch_refs = []

    def clear_memory(self) -> None:
        if self.mem is not None:
            llama_cpp.llama_memory_clear(self.mem, True)

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

    def add_batch_embeddings(
        self,
        *,
        seq_id: int,
        embeddings: np.ndarray,
        positions: np.ndarray,
        output_indices: Sequence[Optional[int]],
    ) -> None:
        if self.batch.n_tokens:
            raise RuntimeError("cannot mix token and embedding batches")
        if self._embedding_batch is not None:
            raise RuntimeError("only one embedding batch is supported per scheduler step")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        positions = np.ascontiguousarray(positions, dtype=np.int32).reshape(-1)
        n_tokens = int(embeddings.shape[0])
        if n_tokens == 0:
            return
        if embeddings.ndim != 2 or embeddings.shape[1] != self.n_embd_inp:
            raise RuntimeError("embedding batch shape does not match model input embedding size")
        if len(positions) not in {n_tokens, n_tokens * 4}:
            raise RuntimeError("embedding position length mismatch")
        if len(output_indices) != n_tokens:
            raise RuntimeError("embedding output index length mismatch")
        pos_array = (llama_cpp.llama_pos * len(positions))(
            *[int(pos) for pos in positions]
        )
        n_seq_id_array = (ctypes.c_int32 * n_tokens)(*[1] * n_tokens)
        seq_id_array = (llama_cpp.llama_seq_id * 1)(llama_cpp.llama_seq_id(seq_id))
        seq_ids_array = (ctypes.POINTER(llama_cpp.llama_seq_id) * (n_tokens + 1))()
        for index in range(n_tokens):
            seq_ids_array[index] = seq_id_array
        logits_array = (ctypes.c_int8 * n_tokens)(
            *[int(output_index is not None) for output_index in output_indices]
        )
        batch = llama_cpp.llama_batch(
            n_tokens=n_tokens,
            token=None,
            embd=embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos=pos_array,
            n_seq_id=n_seq_id_array,
            seq_id=seq_ids_array,
            logits=logits_array,
        )
        self._embedding_batch = batch
        self._embedding_batch_refs = [
            embeddings,
            positions,
            pos_array,
            n_seq_id_array,
            seq_id_array,
            seq_ids_array,
            logits_array,
        ]

    def decode(self) -> None:
        batch = self._embedding_batch if self._embedding_batch is not None else self.batch
        if self.embedding and self.has_encoder:
            operation = "llama_encode"
            result = int(llama_cpp.llama_encode(self.ctx, batch))
        else:
            operation = "llama_decode"
            result = int(llama_cpp.llama_decode(self.ctx, batch))
        if result != 0:
            raise RuntimeError(f"{operation} failed with code {result}")

    def process_draft_batch(self) -> None:
        if self.draft_provider is not None:
            self.draft_provider.process(self.batch)

    def set_draft_processing_enabled(self, enabled: bool) -> None:
        if self.draft_provider is not None:
            self.draft_provider.set_target_processing_enabled(enabled)

    def accept_draft_tokens(self, seq_id: int, accepted_draft_tokens: int) -> None:
        if self.draft_provider is not None:
            self.draft_provider.accept(seq_id, accepted_draft_tokens)

    def truncate_draft_sequence(self, seq_id: int, keep_len: int) -> None:
        if self.draft_provider is not None:
            self.draft_provider.truncate(seq_id, keep_len)

    def copy_draft_sequence(
        self,
        source_seq_id: int,
        dest_seq_id: int,
        p0: int,
        p1: int,
    ) -> None:
        if self.draft_provider is not None:
            self.draft_provider.copy_sequence(source_seq_id, dest_seq_id, p0, p1)

    def logits(self, output_index: int) -> np.ndarray:
        ptr = llama_cpp.llama_get_logits_ith(self.ctx, output_index)
        if not ptr:
            raise RuntimeError(f"missing logits output {output_index}")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def embed(
        self,
        inputs: Sequence[Union[str, List[int]]],
    ) -> Tuple[List[List[float]], int]:
        if not self.embedding:
            raise CompletionRequestValidationError(
                "model.embedding must be true to use /v1/embeddings"
            )
        pooling_type = int(llama_cpp.llama_pooling_type(self.ctx))
        if pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE:
            raise CompletionRequestValidationError(
                "/v1/embeddings requires a pooled embedding model; "
                "set model.pooling_type to MEAN, CLS, or LAST"
            )
        if pooling_type == llama_cpp.LLAMA_POOLING_TYPE_RANK:
            raise CompletionRequestValidationError(
                "/v1/embeddings does not support reranking pooling"
            )
        if len(inputs) > 2048:
            raise CompletionRequestValidationError(
                "embedding input batch size exceeds 2048"
            )

        embeddings: List[List[float]] = []
        total_tokens = 0
        batch_sizes: List[int] = []
        batch_token_count = 0

        def decode_embedding_batch() -> None:
            nonlocal batch_token_count
            if not batch_sizes:
                return
            self.clear_memory()
            self.decode()
            self.clear_batch()
            for seq_id in range(len(batch_sizes)):
                ptr = llama_cpp.llama_get_embeddings_seq(
                    self.ctx,
                    llama_cpp.llama_seq_id(seq_id),
                )
                if not ptr:
                    raise RuntimeError(f"missing embedding output for input {seq_id}")
                embeddings.append(
                    np.ctypeslib.as_array(ptr, shape=(self.n_embd_out,)).astype(
                        float
                    ).tolist()
                )
            batch_sizes.clear()
            batch_token_count = 0

        try:
            self.clear_batch()
            self.clear_memory()
            for input_item in inputs:
                tokens = (
                    self.tokenize(input_item)
                    if isinstance(input_item, str)
                    else list(input_item)
                )
                n_tokens = len(tokens)
                if n_tokens == 0:
                    raise CompletionRequestValidationError(
                        "embedding input must not be empty"
                    )
                if n_tokens > self.n_ctx_seq:
                    raise CompletionRequestValidationError(
                        f"embedding input has {n_tokens} tokens, exceeding n_ctx_seq ({self.n_ctx_seq})"
                    )
                if n_tokens > self.n_batch:
                    raise CompletionRequestValidationError(
                        f"embedding input has {n_tokens} tokens, exceeding n_batch ({self.n_batch})"
                    )
                if total_tokens + n_tokens > 300_000:
                    raise CompletionRequestValidationError(
                        "embedding request exceeds 300000 total tokens"
                    )
                if (
                    batch_token_count + n_tokens > self.n_batch
                    or len(batch_sizes) >= self.n_seq_max
                ):
                    decode_embedding_batch()
                seq_id = len(batch_sizes)
                self.add_batch_tokens(
                    seq_id=seq_id,
                    start_pos=0,
                    tokens=tokens,
                    output_indices=[0] * n_tokens,
                )
                batch_sizes.append(n_tokens)
                batch_token_count += n_tokens
                total_tokens += n_tokens
            decode_embedding_batch()
        finally:
            self.clear_batch()
            self.clear_memory()
        return embeddings, total_tokens


class SequenceDiskCache(SequenceCache):
    """Directory-backed cache for serialized llama.cpp sequence state."""

    @dataclass
    class Entry:
        entry_id: int
        path: Path
        tokens: Tuple[int, ...]
        size_bytes: int
        has_prompt_logits: bool
        last_accessed: float

    @dataclass(frozen=True)
    class Header:
        tokens: Tuple[int, ...]
        has_prompt_logits: bool

    @dataclass
    class Payload:
        tokens: List[int]
        state_bytes: np.ndarray
        prompt_logits: Optional[np.ndarray]

    FORMAT_VERSION = "1"

    TENSOR_TOKENS = "tokens"
    TENSOR_STATE = "state"
    TENSOR_PROMPT_LOGITS = "prompt_logits"

    METADATA_FORMAT = "sequence_cache_format"
    METADATA_COMPATIBILITY_KEY = "compatibility_key"

    def __init__(
        self,
        *,
        path: Union[str, Path],
        max_bytes: int,
        compatibility_key: str,
        min_tokens: int = 128,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(0, int(max_bytes))
        self.min_tokens = max(1, int(min_tokens))
        self.compatibility_key = compatibility_key
        self._metadata = {
            self.METADATA_FORMAT: self.FORMAT_VERSION,
            self.METADATA_COMPATIBILITY_KEY: compatibility_key,
        }
        self._trie = RadixTrie()
        self._entries_by_id: Dict[int, SequenceDiskCache.Entry] = {}
        self._entries_by_tokens: Dict[Tuple[int, ...], SequenceDiskCache.Entry] = {}
        self._next_entry_id = 0
        self._size_bytes = 0
        self._load_entries()
        self._evict_if_needed()

    @staticmethod
    def fingerprint_file(path: str) -> str:
        stat = os.stat(path)
        payload = f"{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def compatibility_key_for_model(cls, model: "Model") -> str:
        payload: Dict[str, Any] = {
            "model": cls.fingerprint_file(model.model_path),
            "memory_model": model.memory_model,
            "loras": [
                {
                    "adapter": cls.fingerprint_file(lora.path),
                    "scale": float(lora.scale),
                }
                for lora in model.loras
            ],
        }
        if model.memory_model == "attention-partitioned":
            payload["attention_streams"] = model.n_seq_max
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _safe_open(path: Path) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "disk_cache requires safetensors. Install it with `pip install safetensors`."
            ) from exc

        return safe_open(str(path), framework="numpy")

    @staticmethod
    def _save_file(
        tensors: Dict[str, np.ndarray],
        path: Path,
        metadata: Dict[str, str],
    ) -> None:
        from safetensors.numpy import save_file

        save_file(tensors, str(path), metadata=metadata)

    def _metadata_compatible(self, metadata: Optional[Dict[str, str]]) -> bool:
        if metadata is None:
            return False
        return all(metadata.get(key) == value for key, value in self._metadata.items())

    def _read_entry_header(self, path: Path) -> Optional["SequenceDiskCache.Header"]:
        with self._safe_open(path) as tensors:
            if not self._metadata_compatible(tensors.metadata()):
                return None
            tokens_array = tensors.get_tensor(self.TENSOR_TOKENS)
            has_prompt_logits = self.TENSOR_PROMPT_LOGITS in tensors.keys()
        tokens = tuple(int(token) for token in tokens_array.tolist())
        if len(tokens) < self.min_tokens:
            return None
        return SequenceDiskCache.Header(
            tokens=tokens,
            has_prompt_logits=has_prompt_logits,
        )

    def _read_entry_payload(
        self,
        entry: "SequenceDiskCache.Entry",
    ) -> "SequenceDiskCache.Payload":
        with self._safe_open(entry.path) as tensors:
            tokens = [
                int(token)
                for token in tensors.get_tensor(self.TENSOR_TOKENS).tolist()
            ]
            state_bytes = np.ascontiguousarray(
                tensors.get_tensor(self.TENSOR_STATE),
                dtype=np.uint8,
            ).copy()
            prompt_logits = (
                tensors.get_tensor(self.TENSOR_PROMPT_LOGITS)
                if entry.has_prompt_logits
                and self.TENSOR_PROMPT_LOGITS in tensors.keys()
                else None
            )
        return SequenceDiskCache.Payload(
            tokens=tokens,
            state_bytes=state_bytes,
            prompt_logits=prompt_logits,
        )

    def _write_entry(
        self,
        path: Path,
        tensors: Dict[str, np.ndarray],
    ) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            self._save_file(tensors, tmp_path, self._metadata)
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_entries(self) -> None:
        for path in sorted(self.path.glob("*.safetensors")):
            try:
                header = self._read_entry_header(path)
                if header is None:
                    continue
                self._add_entry(
                    path=path,
                    tokens=header.tokens,
                    has_prompt_logits=header.has_prompt_logits,
                    last_accessed=path.stat().st_mtime,
                )
            except Exception:  # noqa: BLE001
                continue

    def _add_entry(
        self,
        *,
        path: Path,
        tokens: Tuple[int, ...],
        has_prompt_logits: bool,
        last_accessed: Optional[float] = None,
    ) -> None:
        existing = self._entries_by_tokens.get(tokens)
        if existing is not None:
            self._remove_entry(existing)
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        size_bytes = path.stat().st_size
        entry = SequenceDiskCache.Entry(
            entry_id=entry_id,
            path=path,
            tokens=tokens,
            size_bytes=size_bytes,
            has_prompt_logits=has_prompt_logits,
            last_accessed=last_accessed if last_accessed is not None else time.time(),
        )
        self._entries_by_id[entry_id] = entry
        self._entries_by_tokens[tokens] = entry
        self._trie.extend(entry_id, tokens)
        self._size_bytes += size_bytes

    def _remove_entry(self, entry: "SequenceDiskCache.Entry") -> None:
        if entry.entry_id in self._trie.sequence_lengths:
            self._trie.truncate(entry.entry_id, 0)
        self._entries_by_id.pop(entry.entry_id, None)
        if self._entries_by_tokens.get(entry.tokens) is entry:
            self._entries_by_tokens.pop(entry.tokens, None)
        self._size_bytes = max(0, self._size_bytes - entry.size_bytes)
        try:
            entry.path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _touch(entry: "SequenceDiskCache.Entry") -> None:
        entry.last_accessed = time.time()
        try:
            os.utime(entry.path, None)
        except OSError:
            pass

    def _oldest_entry(self) -> Optional["SequenceDiskCache.Entry"]:
        if not self._entries_by_id:
            return None
        return min(
            self._entries_by_id.values(),
            key=lambda item: (item.last_accessed, item.size_bytes),
        )

    def _evict_until(self, target_bytes: int) -> None:
        while self._size_bytes > target_bytes:
            entry = self._oldest_entry()
            if entry is None:
                return
            self._remove_entry(entry)

    def _evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            for entry in list(self._entries_by_id.values()):
                self._remove_entry(entry)
            return
        if self._size_bytes > self.max_bytes:
            self._evict_until(self.max_bytes)
            self._evict_until(int(self.max_bytes * 0.9))

    def lookup(self, tokens: Sequence[int]) -> Optional[SequenceCache.Match]:
        if not tokens:
            return None
        entry_id, _ = self._trie.longest_prefix(tokens)
        entry = self._entries_by_id.get(entry_id)
        if entry is None:
            return None
        self._touch(entry)
        return SequenceCache.Match(
            tokens=entry.tokens,
            has_prompt_logits=entry.has_prompt_logits,
        )

    def load(
        self,
        match: SequenceCache.Match,
    ) -> Optional[SequenceCache.Load]:
        entry = self._entries_by_tokens.get(match.tokens)
        if entry is None:
            return None

        payload = self._read_entry_payload(entry)
        self._touch(entry)
        return SequenceCache.Load(
            tokens=payload.tokens,
            state_bytes=payload.state_bytes,
            prompt_logits=(
                np.asarray(payload.prompt_logits, dtype=np.float32)
                if payload.prompt_logits is not None
                else None
            ),
        )

    def save(
        self,
        tokens: Sequence[int],
        state_bytes: np.ndarray,
        prompt_logits: Optional[np.ndarray],
    ) -> None:
        if len(tokens) < self.min_tokens or self.max_bytes <= 0:
            return
        state = np.asarray(state_bytes, dtype=np.uint8)
        if state.size <= 0:
            return
        state = np.ascontiguousarray(state).copy()
        entry_tokens = tuple(int(token) for token in tokens)
        tensors: Dict[str, np.ndarray] = {
            self.TENSOR_TOKENS: np.asarray(entry_tokens, dtype=np.int32),
            self.TENSOR_STATE: state,
        }
        if prompt_logits is not None:
            tensors[self.TENSOR_PROMPT_LOGITS] = np.asarray(prompt_logits, dtype=np.float32)
        path = self.path / f"{uuid.uuid4().hex}.safetensors"
        self._write_entry(path, tensors)
        self._add_entry(
            path=path,
            tokens=entry_tokens,
            has_prompt_logits=prompt_logits is not None,
        )
        self._evict_if_needed()


class MemoryPolicy(abc.ABC):
    def __init__(self, scheduler: CompletionScheduler) -> None:
        self.scheduler = scheduler

    def reclaim_order(self, best_free: Optional[int]) -> List[int]:
        reclaim_order = [seq_id for seq_id in self.scheduler.free_sequences if seq_id != best_free]
        if best_free is not None:
            reclaim_order.append(best_free)
        return reclaim_order

    @staticmethod
    def generation_kv_for_request(
        request: CompletionRequest,
        prompt_length: int,
    ) -> int:
        return request.internal_completion_count * (
            request.effective_max_len - prompt_length
        )

    @staticmethod
    def attention_kv_required(
        prompt_kv: int,
        reused_kv: int,
        generation_kv: int,
    ) -> int:
        return prompt_kv - reused_kv + generation_kv

    def try_set_sequence_cache_match(
        self,
        request: CompletionRequest,
        resident_reuse_len: int,
        required_sequence_ids: int,
        required_attn_kv: Optional[int] = None,
        skip_attention_budget_when_unbounded: bool = False,
    ) -> bool:
        cache_match_length = self.scheduler.find_sequence_cache_match(
            request,
            resident_reuse_len,
        )
        if cache_match_length <= resident_reuse_len:
            return False
        has_sequence_budget = len(self.scheduler.unused_sequences) >= required_sequence_ids
        has_attention_budget = required_attn_kv is None or (
            skip_attention_budget_when_unbounded
            and not self.scheduler.model.has_attention_budget
        ) or self.scheduler.sequence_history.size + required_attn_kv <= self.scheduler.model.n_ctx
        if has_sequence_budget and has_attention_budget:
            return True
        self.scheduler.clear_sequence_cache_match(request)
        return False

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
        return request.prompt_plan.clamp_to_reusable_boundary(reuse_len)

    def admit_request(self, request: CompletionRequest) -> None:
        match_seq_id = request.match_sequence_id
        match_length = request.match_length
        reuse_len = self.reuse_len_for_request(request, match_length)
        claimable = match_seq_id in self.scheduler.free_sequences
        if request.sequence_cache_match is not None:
            base_seq_id = self.scheduler.claim_unused_sequence()
            reuse_len, request.prompt_logits = self.scheduler.hydrate_sequence_cache_match(
                request,
                base_seq_id,
            )
        elif claimable:
            base_seq_id = self.scheduler.claim_free_sequence(match_seq_id)
            if self.scheduler.radix_trie.length(base_seq_id) > reuse_len:
                self.scheduler.truncate_sequence(base_seq_id, reuse_len)
        else:
            base_seq_id = self.scheduler.claim_unused_sequence()
            if reuse_len > 0 and match_seq_id >= 0:
                self.copy_prompt_state(match_seq_id, base_seq_id, reuse_len)
        sibling_count = request.internal_completion_count - 1
        sibling_seq_ids: List[int] = []
        for _ in range(sibling_count):
            seq_id = self.scheduler.claim_unused_sequence()
            sibling_seq_ids.append(seq_id)
        self.scheduler.activate_request(
            request,
            base_seq_id=base_seq_id,
            sibling_seq_ids=sibling_seq_ids,
        )
        request.prompt_cursor = reuse_len
        if request.prompt_cursor == len(request.prompt_tokens):
            request.prompt_done = True
            self.scheduler.maybe_save_sequence_cache(request)
            self.scheduler.start_completions(
                request,
                prompt_output_index=None,
                prompt_logits=request.prompt_logits,
            )


class UnifiedAttentionMemoryPolicy(AttentionMemoryPolicy):
    def can_admit(self, request: CompletionRequest) -> bool:
        match_seq_id, match_length = self.match_prefix(request.prompt_tokens)
        match_length = request.prompt_plan.clamp_to_reusable_boundary(match_length)
        request.match_sequence_id = match_seq_id
        request.match_length = match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        required_sequence_ids = request.internal_completion_count - int(claimable)
        prompt_length = len(request.prompt_tokens)
        prompt_kv = request.prompt_plan.eval_token_count
        reuse_len = self.reuse_len_for_request(request, match_length)
        prefix_credit = match_length if claimable else reuse_len
        prefix_credit_kv = max(0, min(prefix_credit, request.prompt_plan.length))
        generation_kv = self.generation_kv_for_request(request, prompt_length)
        if self.try_set_sequence_cache_match(
            request,
            resident_reuse_len=reuse_len,
            required_sequence_ids=request.internal_completion_count,
            required_attn_kv=self.attention_kv_required(
                prompt_kv,
                reused_kv=0,
                generation_kv=generation_kv,
            ),
        ):
            return True
        required_kv = self.attention_kv_required(
            prompt_kv,
            reused_kv=prefix_credit_kv,
            generation_kv=generation_kv,
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
                request.match_length = request.prompt_plan.clamp_to_reusable_boundary(
                    request.match_length
                )
                return True
        return False

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        self.scheduler.copy_sequence_state(
            source_sequence_id,
            dest_sequence_id,
            keep_len,
        )


class PartitionedAttentionMemoryPolicy(AttentionMemoryPolicy):
    def can_admit(self, request: CompletionRequest) -> bool:
        match_seq_id, match_length = self.match_prefix(request.prompt_tokens)
        match_length = request.prompt_plan.clamp_to_reusable_boundary(match_length)
        request.match_sequence_id = match_seq_id
        request.match_length = match_length
        claimable = match_seq_id in self.scheduler.free_sequences
        required_sequence_ids = request.internal_completion_count - int(claimable)
        reuse_len = self.reuse_len_for_request(request, match_length)
        if self.try_set_sequence_cache_match(
            request,
            resident_reuse_len=reuse_len,
            required_sequence_ids=request.internal_completion_count,
        ):
            return True
        if len(self.scheduler.unused_sequences) >= required_sequence_ids:
            return True
        best_free = match_seq_id if claimable else None
        for seq_id in self.reclaim_order(best_free):
            self.scheduler.delete_free_sequence(seq_id)
            if len(self.scheduler.unused_sequences) >= required_sequence_ids:
                request.match_sequence_id, request.match_length = self.match_prefix(
                    request.prompt_tokens,
                )
                request.match_length = request.prompt_plan.clamp_to_reusable_boundary(
                    request.match_length
                )
                return True
        return False

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        self.scheduler.copy_sequence_state(
            source_sequence_id,
            dest_sequence_id,
            keep_len,
            copy_all_state=True,
        )


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
        prompt_length = len(request.prompt_tokens)
        prompt_kv = request.prompt_plan.eval_token_count
        generation_kv = self.generation_kv_for_request(request, prompt_length)
        if self.try_set_sequence_cache_match(
            request,
            resident_reuse_len=match_length,
            required_sequence_ids=request.internal_completion_count,
            required_attn_kv=self.attention_kv_required(
                prompt_kv,
                reused_kv=0,
                generation_kv=generation_kv,
            ),
            skip_attention_budget_when_unbounded=True,
        ):
            return True
        required_attn_kv = self.attention_kv_required(
            prompt_kv,
            reused_kv=max(0, min(match_length, request.prompt_plan.length)),
            generation_kv=generation_kv,
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
        if request.sequence_cache_match is not None:
            base_seq_id = self.scheduler.claim_unused_sequence()
            match_length, request.prompt_logits = self.scheduler.hydrate_sequence_cache_match(
                request,
                base_seq_id,
            )
        elif claimable:
            base_seq_id = self.scheduler.claim_free_sequence(match_seq_id)
            request.prompt_logits = self.scheduler.checkpoint_logits.get(base_seq_id)
            self.scheduler.metrics.checkpoint_hits_total += 1
        else:
            base_seq_id = self.scheduler.claim_unused_sequence()
            request.prompt_logits = None
        sibling_count = request.internal_completion_count - 1
        sibling_seq_ids: List[int] = []
        for _ in range(sibling_count):
            seq_id = self.scheduler.claim_unused_sequence()
            sibling_seq_ids.append(seq_id)
        self.scheduler.activate_request(
            request,
            base_seq_id=base_seq_id,
            sibling_seq_ids=sibling_seq_ids,
        )
        request.prompt_cursor = match_length
        if request.prompt_cursor == len(request.prompt_tokens):
            request.prompt_done = True
            self.scheduler.maybe_save_prompt_checkpoint(request)
            self.scheduler.maybe_save_sequence_cache(request)
            self.scheduler.start_completions(
                request,
                prompt_output_index=None,
                prompt_logits=request.prompt_logits,
            )

    def copy_prompt_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
    ) -> None:
        self.scheduler.copy_sequence_state(
            source_sequence_id,
            dest_sequence_id,
            keep_len,
        )


class CompletionScheduler:
    @dataclass
    class BatchItem:
        @dataclass
        class Prefill:
            embeddings: Optional[np.ndarray] = None
            positions: Optional[np.ndarray] = None
            non_causal: bool = False
            prompt_advance_to: Optional[int] = None

        @dataclass
        class Decode:
            completion_index: int
            pending_count: int
            accepted_draft_count: int = 0
            sampled_pending_token: Optional[int] = None
            rollback_keep_len: Optional[int] = None
            rollback_accepted_draft_count: int = 0
            rollback_draft_processed: bool = False
            deferred_accept_draft_count: Optional[int] = None
            deferred_truncate_draft_len: Optional[int] = None

        kind: Literal["prefill", "decode"]
        request_id: str
        seq_id: int
        start_pos: int
        llama_start_pos: int
        tokens: List[int]
        identity_tokens: List[int]
        output_indices: List[Optional[int]]
        output_positions: List[int]
        position_increments: List[int]
        prefill_state: Optional["CompletionScheduler.BatchItem.Prefill"] = None
        decode_state: Optional["CompletionScheduler.BatchItem.Decode"] = None

        def require_prefill(self) -> "CompletionScheduler.BatchItem.Prefill":
            if self.prefill_state is None:
                raise RuntimeError("batch item is not a prefill item")
            return self.prefill_state

        def require_decode(self) -> "CompletionScheduler.BatchItem.Decode":
            if self.decode_state is None:
                raise RuntimeError("batch item is not a decode item")
            return self.decode_state

        @property
        def batch_token_count(self) -> int:
            if (
                self.prefill_state is not None
                and self.prefill_state.embeddings is not None
            ):
                return int(self.prefill_state.embeddings.shape[0])
            return len(self.tokens)

        @classmethod
        def prefill(
            cls,
            *,
            request_id: str,
            seq_id: int,
            start_pos: int,
            llama_start_pos: int,
            tokens: List[int],
            identity_tokens: List[int],
            output_indices: List[Optional[int]],
            output_positions: List[int],
            position_increments: List[int],
            embeddings: Optional[np.ndarray] = None,
            positions: Optional[np.ndarray] = None,
            non_causal: bool = False,
            prompt_advance_to: Optional[int] = None,
        ) -> "CompletionScheduler.BatchItem":
            return cls(
                kind="prefill",
                request_id=request_id,
                seq_id=seq_id,
                start_pos=start_pos,
                llama_start_pos=llama_start_pos,
                tokens=tokens,
                identity_tokens=identity_tokens,
                output_indices=output_indices,
                output_positions=output_positions,
                position_increments=position_increments,
                prefill_state=cls.Prefill(
                    embeddings=embeddings,
                    positions=positions,
                    non_causal=non_causal,
                    prompt_advance_to=prompt_advance_to,
                ),
            )

        @classmethod
        def decode(
            cls,
            *,
            request_id: str,
            seq_id: int,
            start_pos: int,
            llama_start_pos: int,
            tokens: List[int],
            identity_tokens: List[int],
            output_indices: List[Optional[int]],
            output_positions: List[int],
            position_increments: List[int],
            completion_index: int,
            pending_count: int,
        ) -> "CompletionScheduler.BatchItem":
            return cls(
                kind="decode",
                request_id=request_id,
                seq_id=seq_id,
                start_pos=start_pos,
                llama_start_pos=llama_start_pos,
                tokens=tokens,
                identity_tokens=identity_tokens,
                output_indices=output_indices,
                output_positions=output_positions,
                position_increments=position_increments,
                decode_state=cls.Decode(
                    completion_index=completion_index,
                    pending_count=pending_count,
                ),
            )

    def __init__(
        self,
        model: Model,
        sequence_cache: Optional[SequenceCache] = None,
    ) -> None:
        self.model = model
        self.sequence_cache = sequence_cache
        self.formatter = OpenAIFormatter(model)
        self.radix_trie = RadixTrie()
        self.sequence_history = SequenceHistory()
        self.metrics = SchedulerMetrics()
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
        self.draft_acceptance_length_counts: Dict[int, int] = {}
        self.defer_sampled_draft_processing = False
        self.memory_policy = self.build_memory_policy()

    def build_memory_policy(self) -> MemoryPolicy:
        if self.model.exact_checkpoints_only:
            return CheckpointMemoryPolicy(self)
        if self.model.attention_partitioned:
            return PartitionedAttentionMemoryPolicy(self)
        return UnifiedAttentionMemoryPolicy(self)

    def clear_resident_state(self) -> None:
        self.model.clear_memory()
        self.model.clear_batch()
        self.radix_trie = RadixTrie()
        self.sequence_history = SequenceHistory()
        self.checkpoint_logits.clear()
        self.claimed_sequences.clear()
        self.free_sequences.clear()
        self.unused_sequences = list(range(self.model.n_seq_max - 1, -1, -1))
        for seq_id in range(self.model.n_seq_max):
            self.model.truncate_draft_sequence(seq_id, 0)

    def create_embedding(
        self,
        payload: CreateEmbeddingRequest,
    ) -> CreateEmbeddingResponse:
        if not self.is_idle():
            raise RuntimeError("embedding requests require an idle scheduler")
        self.clear_resident_state()
        try:
            embeddings, total_tokens = self.model.embed(payload.normalized_input())
            return CreateEmbeddingResponse.from_embeddings(
                model=payload.model,
                embeddings=embeddings,
                total_tokens=total_tokens,
                encoding_format=payload.encoding_format,
                dimensions=payload.dimensions,
            )
        finally:
            self.clear_resident_state()

    @staticmethod
    def request_needs_prompt_logits(request: CompletionRequest) -> bool:
        return request.payload.max_tokens != 0 and request.effective_max_len > len(
            request.prompt_tokens
        )

    @staticmethod
    def request_needs_uncached_prompt_logprobs(request: CompletionRequest) -> bool:
        return request.payload.echo and request.payload.logprobs is not None

    @staticmethod
    def clear_sequence_cache_match(request: CompletionRequest) -> None:
        request.sequence_cache_match = None
        request.sequence_cache_match_length = 0

    def can_lookup_sequence_cache(self, request: CompletionRequest) -> bool:
        return (
            self.sequence_cache is not None
            and bool(request.prompt_tokens)
            and not self.request_needs_uncached_prompt_logprobs(request)
        )

    def is_sequence_cache_match_usable(
        self,
        request: CompletionRequest,
        match: SequenceCache.Match,
        resident_reuse_len: int,
    ) -> bool:
        match_length = match.length
        if (
            match_length <= resident_reuse_len
            or match_length > len(request.prompt_tokens)
            or tuple(request.prompt_tokens[:match_length]) != match.tokens
            or not request.prompt_plan.is_reusable_boundary(match_length)
        ):
            return False
        return not (
            self.request_needs_prompt_logits(request)
            and match_length == len(request.prompt_tokens)
            and not match.has_prompt_logits
        )

    def find_sequence_cache_match(
        self,
        request: CompletionRequest,
        resident_reuse_len: int,
    ) -> int:
        self.clear_sequence_cache_match(request)
        if not self.can_lookup_sequence_cache(request):
            return 0
        assert self.sequence_cache is not None
        try:
            match = self.sequence_cache.lookup(request.prompt_tokens)
        except Exception:  # noqa: BLE001
            self.metrics.sequence_cache_lookup_failures_total += 1
            return 0
        if match is None:
            return 0
        if not self.is_sequence_cache_match_usable(
            request,
            match,
            resident_reuse_len,
        ):
            return 0
        request.sequence_cache_match = match
        request.sequence_cache_match_length = match.length
        return match.length

    def fail_sequence_cache_load(
        self,
        request: CompletionRequest,
        seq_id: int,
    ) -> Tuple[int, Optional[np.ndarray]]:
        llama_cpp.llama_memory_seq_rm(self.model.mem, seq_id, 0, -1)
        self.model.truncate_draft_sequence(seq_id, 0)
        self.clear_sequence_cache_match(request)
        self.metrics.sequence_cache_load_failures_total += 1
        return 0, None

    def load_sequence_state_bytes(
        self,
        seq_id: int,
        state_bytes: np.ndarray,
    ) -> bool:
        state = np.ascontiguousarray(state_bytes, dtype=np.uint8)
        state_size = int(state.size)
        if state_size <= 0:
            return False
        state_buffer = (ctypes.c_uint8 * state_size).from_buffer(cast(Any, state))
        loaded_bytes = int(
            llama_cpp.llama_state_seq_set_data(
                self.model.ctx,
                state_buffer,
                state_size,
                cast(llama_cpp.llama_seq_id, seq_id),
            )
        )
        return loaded_bytes == state_size

    def save_sequence_state_bytes(self, seq_id: int) -> Optional[np.ndarray]:
        c_seq_id = cast(llama_cpp.llama_seq_id, seq_id)
        state_size = int(llama_cpp.llama_state_seq_get_size(self.model.ctx, c_seq_id))
        if state_size <= 0:
            return None
        state_buffer = (ctypes.c_uint8 * state_size)()
        state_bytes = int(
            llama_cpp.llama_state_seq_get_data(
                self.model.ctx,
                state_buffer,
                state_size,
                c_seq_id,
            )
        )
        if state_bytes <= 0:
            return None
        return np.ctypeslib.as_array(state_buffer, shape=(state_bytes,)).copy()

    def hydrate_sequence_cache_match(
        self,
        request: CompletionRequest,
        seq_id: int,
    ) -> Tuple[int, Optional[np.ndarray]]:
        match = request.sequence_cache_match
        if self.sequence_cache is None or match is None:
            return 0, None
        try:
            loaded = self.sequence_cache.load(match)
        except Exception:  # noqa: BLE001
            return self.fail_sequence_cache_load(request, seq_id)
        if loaded is None:
            return self.fail_sequence_cache_load(request, seq_id)
        tokens = list(loaded.tokens)
        expected_length = request.sequence_cache_match_length
        if (
            len(tokens) != expected_length
            or tokens != request.prompt_tokens[:expected_length]
        ):
            return self.fail_sequence_cache_load(request, seq_id)
        if (
            len(tokens) == len(request.prompt_tokens)
            and self.request_needs_prompt_logits(request)
            and loaded.prompt_logits is None
        ):
            return self.fail_sequence_cache_load(request, seq_id)
        if not self.load_sequence_state_bytes(seq_id, loaded.state_bytes):
            return self.fail_sequence_cache_load(request, seq_id)
        self.radix_trie.extend(seq_id, tokens)
        self.sequence_history.extend(
            seq_id,
            tokens,
            request.prompt_plan.position_increments_up_to(len(tokens)),
        )
        self.model.truncate_draft_sequence(seq_id, 0)
        self.metrics.sequence_cache_hits_total += 1
        self.metrics.sequence_cache_tokens_loaded_total += len(tokens)
        if len(tokens) == len(request.prompt_tokens):
            return len(tokens), loaded.prompt_logits
        return len(tokens), None

    def maybe_save_sequence_cache(self, request: CompletionRequest) -> None:
        if (
            self.sequence_cache is None
            or request.base_seq_id is None
            or not request.prompt_tokens
            or request.sequence_cache_match_length == len(request.prompt_tokens)
        ):
            return
        state_bytes = self.save_sequence_state_bytes(request.base_seq_id)
        if state_bytes is None:
            return
        try:
            self.sequence_cache.save(
                request.prompt_tokens,
                state_bytes,
                request.prompt_logits,
            )
        except Exception:  # noqa: BLE001
            self.metrics.sequence_cache_save_failures_total += 1
            return
        self.metrics.sequence_cache_save_requests_total += 1

    def close(self) -> None:
        self.closed = True
        self.model.close()

    def submit_request(self, request: CompletionRequest) -> str:
        if self.closed:
            raise RuntimeError("scheduler closed")
        self.requests[request.id] = request
        self.pending_requests.append(request)
        self.metrics.requests_submitted_total += 1
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
        step_started_at = time.perf_counter()
        if self.closed:
            return False
        try:
            self.admit_waiting()
            batch_items = self.build_batch()
            if not batch_items:
                if self.maybe_fill_batched_draft_tokens():
                    self.finalize_cancelled()
                    return True
                return self.finalize_cancelled()
            draft_processing_enabled = self.batch_needs_draft_processing(batch_items)
            self.defer_sampled_draft_processing = (
                draft_processing_enabled
                and self.supports_sampled_draft_processing
                and all(item.kind == "decode" for item in batch_items)
            )
            self.model.set_draft_processing_enabled(draft_processing_enabled)
            self.model.clear_batch()
            for item in batch_items:
                prefill = item.prefill_state
                if prefill is not None and prefill.embeddings is not None:
                    if prefill.positions is None:
                        raise RuntimeError("embedding batch item is missing positions")
                    self.model.add_batch_embeddings(
                        seq_id=item.seq_id,
                        embeddings=prefill.embeddings,
                        positions=prefill.positions,
                        output_indices=item.output_indices,
                    )
                else:
                    self.model.add_batch_tokens(
                        seq_id=item.seq_id,
                        start_pos=item.llama_start_pos,
                        tokens=item.tokens,
                        output_indices=item.output_indices,
                    )
            decode_started_at = time.perf_counter()
            try:
                non_causal_decode = any(
                    item.prefill_state is not None and item.prefill_state.non_causal
                    for item in batch_items
                )
                try:
                    if non_causal_decode:
                        llama_cpp.llama_set_causal_attn(self.model.ctx, False)
                    self.model.decode()
                finally:
                    if non_causal_decode:
                        llama_cpp.llama_set_causal_attn(self.model.ctx, True)
                decode_elapsed = time.perf_counter() - decode_started_at
                if draft_processing_enabled and not self.defer_sampled_draft_processing:
                    draft_process_started_at = time.perf_counter()
                    self.model.process_draft_batch()
                    draft_process_elapsed = time.perf_counter() - draft_process_started_at
                    self.metrics.draft_process_seconds_total += draft_process_elapsed
                    self.metrics.draft_process_calls_total += 1
                    self.observe_draft_process(batch_items, draft_process_elapsed)
            except BaseException as exc:  # noqa: BLE001
                for request_id in list(self.active_request_ids):
                    self.fail_request(self.requests[request_id], exc)
                for request in list(self.pending_requests):
                    self.fail_request(request, exc)
                return True
            self.metrics.observe_decode(
                batch_items,
                decode_elapsed,
            )
            process_batch_started_at = time.perf_counter()
            self.process_batch(batch_items)
            self.metrics.process_batch_seconds_total += (
                time.perf_counter() - process_batch_started_at
            )
            self.metrics.process_batch_calls_total += 1
            if self.defer_sampled_draft_processing:
                try:
                    self.process_sampled_draft_batch(batch_items)
                finally:
                    self.apply_deferred_draft_rollbacks(batch_items)
                    self.defer_sampled_draft_processing = False
            self.finalize_cancelled()
            return True
        finally:
            self.metrics.scheduler_step_seconds_total += (
                time.perf_counter() - step_started_at
            )
            self.metrics.scheduler_step_calls_total += 1

    def observe_draft_process(
        self,
        items: Sequence["CompletionScheduler.BatchItem"],
        elapsed_seconds: float,
    ) -> None:
        if self.model.draft_provider is None or elapsed_seconds <= 0.0:
            return
        self.metrics.draft_seconds_total += elapsed_seconds

    def item_allows_mtp_processing(self, item: "CompletionScheduler.BatchItem") -> bool:
        request = self.requests.get(item.request_id)
        if request is None:
            return False
        if item.kind == "prefill":
            return not any(segment.kind != "text" for segment in request.prompt_plan.segments)
        decode = item.decode_state
        if decode is None:
            return False
        return not request.completions[decode.completion_index].multimodal_prompt

    def batch_needs_draft_processing(
        self,
        batch_items: Sequence["CompletionScheduler.BatchItem"],
    ) -> bool:
        if self.model.draft_provider is None:
            return False
        if not self.draft_batch_size_allowed(batch_items):
            return False
        if isinstance(self.model.draft_provider, MTPDraftProvider) and not all(
            self.item_allows_mtp_processing(item) for item in batch_items
        ):
            return False
        for item in batch_items:
            request = self.requests.get(item.request_id)
            if request is None:
                continue
            if item.kind == "prefill":
                if self.request_needs_prompt_logits(request):
                    return True
                continue
            decode = item.decode_state
            if decode is None:
                continue
            completion = request.completions[decode.completion_index]
            if not completion.finished:
                return True
        return False

    def draft_batch_size_allowed(
        self,
        batch_items: Sequence["CompletionScheduler.BatchItem"],
    ) -> bool:
        max_batch_size = getattr(self.model, "draft_model_max_batch_size", None)
        if max_batch_size is None:
            return True
        draft_batch_size = sum(
            1
            for item in batch_items
            if item.kind == "decode" and item.decode_state is not None
        )
        return draft_batch_size <= max_batch_size

    def active_draft_batch_size_allowed(self, draft_batch_size: int) -> bool:
        max_batch_size = getattr(self.model, "draft_model_max_batch_size", None)
        return max_batch_size is None or draft_batch_size <= max_batch_size

    def admit_waiting(self) -> None:
        while self.pending_requests:
            request = self.pending_requests[0]
            if request.cancelled:
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
        if request.admitted:
            self.metrics.requests_admitted_total += 1

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

        for request in prompt_requests:
            if request.prompt_cursor < len(request.prompt_tokens):
                segment = self._current_prompt_segment(request)
                if segment.kind != "text":
                    token_count = min(
                        segment.batch_rows
                        if segment.media is not None and segment.media.non_causal
                        else self._pending_tokens_length(request),
                        self.model.n_batch,
                    )
                    if segment.batch_rows > self.model.n_batch:
                        if segment.media is not None and segment.media.non_causal:
                            raise RuntimeError(
                                "non-causal media prompt segment exceeds model.n_batch; "
                                "increase n_batch"
                            )
                    item, _ = self._build_pending_batch_item(
                        request,
                        token_count,
                        0,
                    )
                    return [item]

        ordered_sequences = self._ordered_pending_sequences(prompt_requests, completions)
        allocations = self._allocate_pending_tokens_for_model(ordered_sequences)
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
        return self._split_mixed_mtp_batch(items)

    def _split_mixed_mtp_batch(
        self,
        items: List["CompletionScheduler.BatchItem"],
    ) -> List["CompletionScheduler.BatchItem"]:
        if not isinstance(self.model.draft_provider, MTPDraftProvider) or len(items) <= 1:
            return items
        eligibility = [self.item_allows_mtp_processing(item) for item in items]
        if all(eligibility) or not any(eligibility):
            return items
        keep_eligible = eligibility[0]
        split_items = [
            item
            for item, item_eligible in zip(items, eligibility)
            if item_eligible == keep_eligible
        ]
        self._renumber_output_indices(split_items)
        return split_items

    @staticmethod
    def _renumber_output_indices(
        items: Sequence["CompletionScheduler.BatchItem"],
    ) -> None:
        next_output_index = 0
        for item in items:
            for index, output_index in enumerate(item.output_indices):
                if output_index is None:
                    continue
                item.output_indices[index] = next_output_index
                next_output_index += 1

    def _allocate_pending_tokens_for_model(
        self,
        ordered_sequences: Sequence[Union[CompletionRequest, Completion]],
    ) -> Dict[int, int]:
        if not self._needs_homogeneous_recurrent_draft_batching():
            return self._allocate_pending_tokens(ordered_sequences, self.model.n_batch)

        active = [
            source
            for source in ordered_sequences
            if self._pending_tokens_length(source) > 0
        ]
        if not active:
            return {}

        first_draft_count = self._recurrent_draft_batch_token_count(active[0])
        if first_draft_count > 0:
            return self._allocate_homogeneous_recurrent_draft_tokens(
                active,
                first_draft_count,
            )

        non_draft_sources = [
            source
            for source in active
            if self._recurrent_draft_batch_token_count(source) == 0
        ]
        return self._allocate_pending_tokens(non_draft_sources, self.model.n_batch)

    def _needs_homogeneous_recurrent_draft_batching(self) -> bool:
        return bool(
            self.model.draft_provider is not None
            and getattr(self.model, "draft_target_batching", False)
            and self.model.exact_checkpoints_only
            and self.model.n_rs_seq > 0
        )

    def _recurrent_draft_batch_token_count(
        self,
        source: Union[CompletionRequest, Completion],
    ) -> int:
        if isinstance(source, CompletionRequest):
            return 0
        if (
            source.finished
            or source.pending_finish_reason is not None
            or not source.pending_input_tokens
            or not source.draft_tokens
        ):
            return 0
        token_count = min(
            len(source.pending_input_tokens) + len(source.draft_tokens),
            len(source.pending_input_tokens) + self.model.n_rs_seq,
        )
        return token_count if token_count > len(source.pending_input_tokens) else 0

    def _allocate_homogeneous_recurrent_draft_tokens(
        self,
        sources: Sequence[Union[CompletionRequest, Completion]],
        token_count: int,
    ) -> Dict[int, int]:
        # Recurrent rollback snapshots are only valid if the whole per-sequence
        # pending+draft group fits in a single llama.cpp ubatch.
        safe_capacity = min(self.model.n_batch, self.model.n_ubatch)
        if token_count > safe_capacity:
            return self._allocate_recurrent_pending_only_tokens(sources, safe_capacity)

        allocations: Dict[int, int] = {}
        remaining_capacity = safe_capacity
        for source in sources:
            if remaining_capacity < token_count:
                break
            if self._recurrent_draft_batch_token_count(source) != token_count:
                continue
            seq_id = self._pending_sequence_id(source)
            allocations[seq_id] = token_count
            remaining_capacity -= token_count
        return allocations

    def _allocate_recurrent_pending_only_tokens(
        self,
        sources: Sequence[Union[CompletionRequest, Completion]],
        capacity: int,
    ) -> Dict[int, int]:
        allocations: Dict[int, int] = {}
        remaining_capacity = capacity
        for source in sources:
            if remaining_capacity <= 0:
                break
            if self._recurrent_draft_batch_token_count(source) == 0:
                continue
            if isinstance(source, CompletionRequest):
                continue
            pending_count = min(len(source.pending_input_tokens), remaining_capacity)
            if pending_count <= 0:
                continue
            allocations[source.seq_id] = pending_count
            remaining_capacity -= pending_count
        return allocations

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

    def _current_prompt_segment(self, request: CompletionRequest) -> PromptSegment:
        return request.prompt_plan.segment_at(request.prompt_cursor)

    def _pending_tokens_length(
        self,
        source: Union[CompletionRequest, Completion],
    ) -> int:
        if isinstance(source, CompletionRequest):
            if source.prompt_cursor >= len(source.prompt_tokens):
                return 0
            segment = self._current_prompt_segment(source)
            if segment.kind != "text":
                segment_offset = source.prompt_cursor - segment.start_pos
                if segment.media is not None and segment.media.non_causal:
                    if segment_offset != 0:
                        raise RuntimeError("non-causal media segment was partially scheduled")
                    return segment.batch_rows
                return segment.end_pos - source.prompt_cursor
            return segment.end_pos - source.prompt_cursor
        draft_count = (
            len(source.draft_tokens)
            if source.pending_finish_reason is None
            and getattr(self.model, "draft_target_batching", True)
            else 0
        )
        return len(source.pending_input_tokens) + draft_count

    def _pending_allocation_for_capacity(
        self,
        source: Union[CompletionRequest, Completion],
        already_allocated: int,
        capacity: int,
    ) -> int:
        if capacity <= 0:
            return 0
        if not isinstance(source, CompletionRequest):
            remaining = self._pending_tokens_length(source) - already_allocated
            return min(remaining, capacity)
        segment = self._current_prompt_segment(source)
        if segment.kind == "text":
            remaining = self._pending_tokens_length(source) - already_allocated
            return min(remaining, capacity)
        if segment.media is not None and segment.media.non_causal:
            if already_allocated == 0 and segment.batch_rows <= capacity:
                return segment.batch_rows
            return 0
        segment_offset = source.prompt_cursor - segment.start_pos + already_allocated
        return segment.rows_for_capacity(segment_offset, capacity)

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
            allocation = self._pending_allocation_for_capacity(
                source,
                allocations.get(seq_id, 0),
                remaining_capacity,
            )
            if allocation <= 0:
                continue
            allocations[seq_id] = allocations.get(seq_id, 0) + allocation
            remaining_capacity -= allocation

        while remaining_capacity > 0 and active:
            share = max(1, remaining_capacity // len(active))
            progress = False
            next_active: List[Union[CompletionRequest, Completion]] = []
            for source in active:
                seq_id = self._pending_sequence_id(source)
                remaining_tokens = self._pending_tokens_length(source) - allocations.get(seq_id, 0)
                if remaining_tokens <= 0:
                    continue
                allocation = self._pending_allocation_for_capacity(
                    source,
                    allocations.get(seq_id, 0),
                    min(share, remaining_capacity),
                )
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
            segment = self._current_prompt_segment(source)
            if segment.kind != "text":
                segment_offset = source.prompt_cursor - segment.start_pos
                non_causal = segment.media is not None and segment.media.non_causal
                if non_causal:
                    if segment_offset != 0 or token_count < segment.batch_rows:
                        raise RuntimeError("non-causal media segment must be scheduled atomically")
                    row_count = segment.batch_rows
                    embeddings, positions, position_increments = segment.media_slice(
                        0,
                        row_count,
                    )
                else:
                    row_count = segment.rows_for_capacity(segment_offset, token_count)
                    if row_count <= 0:
                        raise RuntimeError("media prompt allocation is too small")
                    embeddings, positions, position_increments = segment.media_slice(
                        segment_offset,
                        row_count,
                    )
                logical_start = source.prompt_cursor
                logical_end = logical_start + row_count
                output_indices = [None] * row_count
                output_positions = [logical_start] * row_count
                if output_positions:
                    output_positions[-1] = logical_end - 1
                needs_last_output = (
                    (source.payload.echo and source.payload.logprobs is not None
                     and source.prompt_plan.has_text_token_at(logical_end))
                    or self.model.exact_checkpoints_only
                    or logical_end == len(source.prompt_tokens)
                )
                if needs_last_output and output_indices:
                    output_indices[-1] = output_index
                    output_index += 1
                return (
                    CompletionScheduler.BatchItem.prefill(
                        request_id=source.id,
                        seq_id=source.base_seq_id,
                        start_pos=logical_start,
                        llama_start_pos=segment.decode_start_pos + sum(
                            segment.decoder_position_increments[:segment_offset]
                        ),
                        tokens=[],
                        identity_tokens=list(
                            segment.identity_tokens[
                                segment_offset : segment_offset + row_count
                            ]
                        ),
                        output_indices=output_indices,
                        output_positions=output_positions,
                        position_increments=position_increments,
                        embeddings=embeddings,
                        positions=positions,
                        non_causal=non_causal,
                        prompt_advance_to=logical_end,
                    ),
                    output_index,
                )
            segment_offset = source.prompt_cursor - segment.start_pos
            max_count = min(token_count, segment.end_pos - source.prompt_cursor)
            chunk = list(segment.text_tokens[segment_offset : segment_offset + max_count])
            identity_chunk = list(
                segment.identity_tokens[segment_offset : segment_offset + max_count]
            )
            ends_prompt = source.prompt_cursor + len(identity_chunk) == len(source.prompt_tokens)
            output_indices: List[Optional[int]] = [None] * len(chunk)
            output_positions = [source.prompt_cursor + index for index in range(len(chunk))]
            for index, output_position in enumerate(output_positions):
                if (
                    source.payload.echo
                    and source.payload.logprobs is not None
                    and source.prompt_plan.has_text_token_at(output_position + 1)
                ):
                    output_indices[index] = output_index
                    output_index += 1
            if self.model.exact_checkpoints_only and chunk and output_indices[-1] is None:
                output_indices[-1] = output_index
                output_index += 1
            elif ends_prompt and chunk and output_indices[-1] is None:
                output_indices[-1] = output_index
                output_index += 1
            return (
                CompletionScheduler.BatchItem.prefill(
                    request_id=source.id,
                    seq_id=source.base_seq_id,
                    start_pos=source.prompt_cursor,
                    llama_start_pos=segment.decode_start_pos + segment_offset,
                    tokens=chunk,
                    identity_tokens=identity_chunk,
                    output_indices=output_indices,
                    output_positions=output_positions,
                    position_increments=segment.decoder_position_increments[
                        segment_offset : segment_offset + len(identity_chunk)
                    ],
                ),
                output_index,
            )

        request = self.requests[source.request_id]
        pending_count = min(token_count, len(source.pending_input_tokens))
        draft_count = (
            max(0, token_count - pending_count)
            if getattr(self.model, "draft_target_batching", True)
            else 0
        )
        scheduled_tokens = [
            *source.pending_input_tokens[:pending_count],
            *(
                source.draft_tokens[:draft_count]
                if source.pending_finish_reason is None
                else []
            ),
        ]
        output_indices = list(range(output_index, output_index + len(scheduled_tokens)))
        start_pos = self.radix_trie.length(source.seq_id)
        return (
            CompletionScheduler.BatchItem.decode(
                request_id=request.id,
                seq_id=source.seq_id,
                start_pos=start_pos,
                llama_start_pos=self.sequence_history.position_length(source.seq_id),
                tokens=list(scheduled_tokens),
                identity_tokens=list(scheduled_tokens),
                output_indices=output_indices,
                output_positions=[
                    start_pos + index
                    for index in range(len(scheduled_tokens))
                ],
                position_increments=[1] * len(scheduled_tokens),
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
            self.radix_trie.extend(item.seq_id, item.identity_tokens)
            self.sequence_history.extend(
                item.seq_id,
                item.identity_tokens,
                item.position_increments,
            )
            if item.kind == "prefill":
                request.capture_prompt_logprobs(
                    model=self.model,
                    formatter=self.formatter,
                    output_indices=item.output_indices,
                    output_positions=item.output_positions,
                    output_count=output_count,
                    output_index_to_logits_index=self.output_index_to_logits_index,
                )
                prompt_output_index = self.output_index_to_logits_index(
                    self.last_output_index(item.output_indices),
                    output_count,
                )
                prompt_logits = (
                    self.model.logits(prompt_output_index)
                    if prompt_output_index is not None
                    else None
                )
                prefill = item.require_prefill()
                if prefill.prompt_advance_to is not None:
                    request.prompt_cursor = prefill.prompt_advance_to
                else:
                    request.prompt_cursor += len(item.identity_tokens)
                if request.prompt_cursor == len(request.prompt_tokens):
                    request.prompt_done = True
                    request.prompt_logits = prompt_logits
                    self.maybe_save_prompt_checkpoint(request)
                    self.maybe_save_sequence_cache(request)
                    self.start_completions(
                        request,
                        prompt_output_index=prompt_output_index,
                        prompt_logits=request.prompt_logits,
                    )
            else:
                decode = item.require_decode()
                completion = request.completions[decode.completion_index]
                self.process_generation_item(
                    completion,
                    item,
                    output_count,
                )
                self.finalize_request_if_ready(request)
        if not self.defer_sampled_draft_processing:
            self.maybe_fill_batched_draft_tokens()

    @property
    def should_defer_draft_fill(self) -> bool:
        return bool(
            self.model.draft_provider is not None
            and getattr(self.model.draft_provider, "batched_draft", False)
        )

    @property
    def supports_sampled_draft_processing(self) -> bool:
        return bool(
            self.model.draft_provider is not None
            and getattr(self.model.draft_provider, "sampled_batch_draft", False)
        )

    def maybe_fill_draft_tokens(self, completion: Completion) -> None:
        if (
            self.model.draft_provider is None
            or completion.finished
            or completion.pending_finish_reason is not None
            or completion.draft_tokens
        ):
            return
        remaining_tokens = completion.max_total_tokens - completion.total_tokens
        if not getattr(self.model, "draft_target_batching", True):
            remaining_tokens = min(remaining_tokens, 1)
        if remaining_tokens <= 0:
            return
        input_ids = self.draft_input_ids(completion)
        if isinstance(self.model.draft_provider, MTPDraftProvider) and completion.multimodal_prompt:
            return
        can_draft = getattr(self.model.draft_provider, "can_draft", None)
        if can_draft is not None and not can_draft(
            int(input_ids.shape[0]),
            seq_id=completion.seq_id,
        ):
            return
        draft_started_at = time.perf_counter()
        try:
            proposed = self.model.draft_provider.draft(
                input_ids,
                seq_id=completion.seq_id,
                max_tokens=remaining_tokens,
            )
        finally:
            draft_elapsed = time.perf_counter() - draft_started_at
            self.metrics.draft_seconds_total += draft_elapsed
            self.metrics.draft_generate_seconds_total += draft_elapsed
            self.metrics.draft_generate_calls_total += 1
        if proposed.size == 0:
            return
        limited = [int(token) for token in proposed[:remaining_tokens]]
        if not limited:
            return
        completion.draft_tokens = limited
        self.speculative_stats["draft_proposals"] += 1
        self.speculative_stats["draft_tokens_proposed"] += len(limited)

    def maybe_fill_batched_draft_tokens(self) -> bool:
        if not self.should_defer_draft_fill or self.model.draft_provider is None:
            return False
        draft_many = getattr(self.model.draft_provider, "draft_many", None)
        if draft_many is None:
            return False

        completions: List[Completion] = []
        draft_requests: List[Tuple[np.ndarray, int, Optional[int]]] = []
        remaining_by_completion: List[int] = []
        for request_id in self.active_request_ids:
            request = self.requests.get(request_id)
            if request is None or not request.admitted:
                continue
            for completion in request.completions:
                if (
                    completion.finished
                    or completion.pending_finish_reason is not None
                    or completion.draft_tokens
                ):
                    continue
                remaining_tokens = completion.max_total_tokens - completion.total_tokens
                if not getattr(self.model, "draft_target_batching", True):
                    remaining_tokens = min(remaining_tokens, 1)
                if remaining_tokens <= 0:
                    continue
                input_ids = self.draft_input_ids(completion)
                if (
                    isinstance(self.model.draft_provider, MTPDraftProvider)
                    and completion.multimodal_prompt
                ):
                    continue
                can_draft = getattr(self.model.draft_provider, "can_draft", None)
                if can_draft is not None and not can_draft(
                    int(input_ids.shape[0]),
                    seq_id=completion.seq_id,
                ):
                    continue
                completions.append(completion)
                draft_requests.append((input_ids, completion.seq_id, remaining_tokens))
                remaining_by_completion.append(remaining_tokens)

        if not draft_requests:
            return False
        if not self.active_draft_batch_size_allowed(len(draft_requests)):
            return False

        draft_started_at = time.perf_counter()
        proposed_many = draft_many(draft_requests)
        draft_elapsed = time.perf_counter() - draft_started_at
        self.metrics.draft_seconds_total += draft_elapsed
        self.metrics.draft_generate_seconds_total += draft_elapsed
        self.metrics.draft_generate_calls_total += 1
        made_progress = False

        for completion, proposed, remaining_tokens in zip(
            completions,
            proposed_many,
            remaining_by_completion,
        ):
            if proposed.size == 0:
                continue
            limited = [int(token) for token in proposed[:remaining_tokens]]
            if not limited:
                continue
            self.speculative_stats["draft_proposals"] += 1
            self.speculative_stats["draft_tokens_proposed"] += len(limited)
            completion.draft_tokens = limited
            made_progress = True
        return made_progress

    def draft_input_ids(self, completion: Completion) -> np.ndarray:
        return np.array(
            [*completion.prompt_tokens, *completion.completion_tokens],
            dtype=np.intc,
        )

    def process_sampled_draft_batch(
        self,
        items: Sequence["CompletionScheduler.BatchItem"],
    ) -> None:
        if self.model.draft_provider is None:
            return
        process_sampled_batch = getattr(
            self.model.draft_provider,
            "process_sampled_batch",
            None,
        )
        if process_sampled_batch is None:
            return
        if not self.draft_batch_size_allowed(items):
            return

        output_count = sum(
            output_index is not None
            for item in items
            for output_index in item.output_indices
        )
        completions: List[Completion] = []
        update_items: List["CompletionScheduler.BatchItem"] = []
        updates: List["MTPDraftProvider.SampledBatchUpdate"] = []
        remaining_by_completion: List[int] = []
        batch_row_offset = 0
        for item in items:
            item_batch_rows = list(
                range(batch_row_offset, batch_row_offset + len(item.tokens))
            )
            batch_row_offset += len(item.tokens)
            decode = item.decode_state
            if item.kind != "decode" or decode is None:
                continue
            request = self.requests.get(item.request_id)
            if request is None:
                continue
            completion = request.completions[decode.completion_index]
            if completion.finished or completion.pending_finish_reason is not None:
                continue
            remaining_tokens = completion.max_total_tokens - completion.total_tokens
            if remaining_tokens <= 0:
                continue
            resolved_output_indices = [
                self.output_index_to_logits_index(output_index, output_count)
                for output_index in item.output_indices
            ]
            if any(output_index is None for output_index in resolved_output_indices):
                continue
            sample_index = decode.pending_count + decode.accepted_draft_count - 1
            target_count = len(item.tokens)
            if sample_index < 0 or sample_index >= target_count:
                continue
            if decode.sampled_pending_token is None:
                continue
            completions.append(completion)
            update_items.append(item)
            updates.append(
                MTPDraftProvider.SampledBatchUpdate(
                    seq_id=item.seq_id,
                    start_pos=item.llama_start_pos,
                    tokens=item.tokens,
                    row_indices=item_batch_rows,
                    target_count=target_count,
                    sample_index=sample_index,
                    pending_token=decode.sampled_pending_token,
                    max_tokens=remaining_tokens,
                )
            )
            remaining_by_completion.append(remaining_tokens)

        if not updates:
            return

        draft_started_at = time.perf_counter()
        proposed_many = process_sampled_batch(updates)
        for item in update_items:
            item.require_decode().rollback_draft_processed = True
        draft_elapsed = time.perf_counter() - draft_started_at
        self.metrics.draft_seconds_total += draft_elapsed
        self.metrics.draft_sampled_batch_seconds_total += draft_elapsed
        self.metrics.draft_sampled_batch_calls_total += 1
        for completion, proposed, remaining_tokens in zip(
            completions,
            proposed_many,
            remaining_by_completion,
        ):
            if proposed.size == 0:
                continue
            limited = [int(token) for token in proposed[:remaining_tokens]]
            if not limited:
                continue
            completion.draft_tokens = limited
            self.speculative_stats["draft_proposals"] += 1
            self.speculative_stats["draft_tokens_proposed"] += len(limited)

    def rollback_draft_verification(
        self,
        item: "CompletionScheduler.BatchItem",
        keep_len: int,
        accepted_draft_count: int,
        *,
        defer_draft_state: bool,
    ) -> None:
        decode = item.require_decode()
        if defer_draft_state:
            decode.rollback_keep_len = keep_len
            decode.rollback_accepted_draft_count = accepted_draft_count
            decode.rollback_draft_processed = False
            return
        self.model.accept_draft_tokens(item.seq_id, accepted_draft_count)
        self.truncate_sequence(item.seq_id, keep_len)

    def apply_deferred_draft_rollbacks(
        self,
        items: Sequence["CompletionScheduler.BatchItem"],
    ) -> None:
        for item in items:
            decode = item.require_decode()
            rollback_keep_len = decode.rollback_keep_len
            if rollback_keep_len is None:
                deferred_accept_draft_count = decode.deferred_accept_draft_count
                if (
                    not decode.rollback_draft_processed
                    and deferred_accept_draft_count is not None
                ):
                    self.model.accept_draft_tokens(
                        item.seq_id,
                        deferred_accept_draft_count,
                    )
                    deferred_truncate_draft_len = decode.deferred_truncate_draft_len
                    if deferred_truncate_draft_len is not None:
                        self.model.truncate_draft_sequence(
                            item.seq_id,
                            deferred_truncate_draft_len,
                        )
                decode.deferred_accept_draft_count = None
                decode.deferred_truncate_draft_len = None
                decode.rollback_draft_processed = False
                continue

            draft_state_needs_fallback = not decode.rollback_draft_processed
            if draft_state_needs_fallback:
                self.model.accept_draft_tokens(
                    item.seq_id,
                    decode.rollback_accepted_draft_count,
                )
            self.truncate_sequence(
                item.seq_id,
                rollback_keep_len,
                truncate_draft=draft_state_needs_fallback,
            )
            decode.rollback_keep_len = None
            decode.rollback_accepted_draft_count = 0
            decode.rollback_draft_processed = False
            decode.deferred_accept_draft_count = None
            decode.deferred_truncate_draft_len = None

    def sample_completion_token(self, completion: Completion, output_index: int) -> int:
        sample_started_at = time.perf_counter()
        try:
            return completion.sampler.sample(self.model.ctx, output_index)
        finally:
            self.metrics.sample_seconds_total += time.perf_counter() - sample_started_at
            self.metrics.sample_calls_total += 1

    def process_generation_item(
        self,
        completion: Completion,
        item: CompletionScheduler.BatchItem,
        output_count: int,
    ) -> None:
        if completion.finished:
            return
        decode = item.require_decode()
        pending_count = decode.pending_count
        resolved_output_indices = [
            self.output_index_to_logits_index(output_index, output_count)
            for output_index in item.output_indices
        ]
        if any(output_index is None for output_index in resolved_output_indices):
            raise RuntimeError("generation outputs are required")
        logits_indices: List[int] = []
        for output_index in resolved_output_indices:
            assert output_index is not None
            logits_indices.append(int(output_index))
        if completion.pending_finish_reason is not None:
            if self.model.store_logits:
                self.checkpoint_logits[completion.seq_id] = self.model.logits(
                    logits_indices[-1]
                )
            completion.pending_input_tokens = completion.pending_input_tokens[pending_count:]
            finish_reason: str = completion.pending_finish_reason
            completion.pending_finish_reason = None
            self.finish_completion(completion, finish_reason)
            return

        if pending_count:
            completion.pending_input_tokens = completion.pending_input_tokens[pending_count:]

        decoded_draft_tokens = item.tokens[pending_count:]
        accepted_draft_count = 0
        defer_draft_state = self.defer_sampled_draft_processing
        if decoded_draft_tokens and pending_count <= 0:
            raise RuntimeError("draft verification requires at least one pending token")
        if decoded_draft_tokens:
            self.metrics.draft_batches_verified_total += 1
            self.metrics.draft_target_tokens_verified_total += len(decoded_draft_tokens)

        for draft_index, draft_token in enumerate(decoded_draft_tokens):
            verify_output_index = pending_count + draft_index - 1
            if verify_output_index >= len(logits_indices):
                raise RuntimeError("missing target output for draft verification")
            logits_index = logits_indices[verify_output_index]
            sampled_token = self.sample_completion_token(completion, logits_index)
            need_logits = self.model.store_logits or completion.needs_token_logprob
            logits = self.model.logits(logits_index) if need_logits else None
            if self.model.store_logits and logits is not None:
                self.checkpoint_logits[completion.seq_id] = logits
            if sampled_token != draft_token:
                self.record_draft_acceptance_length(accepted_draft_count)
                rejected = max(0, len(completion.draft_tokens) - accepted_draft_count)
                if rejected > 0:
                    self.speculative_stats["draft_tokens_rejected"] += rejected
                self.metrics.draft_target_tokens_wasted_total += (
                    max(0, len(decoded_draft_tokens) - accepted_draft_count - 1)
                )
                keep_len = item.start_pos + pending_count + accepted_draft_count
                self.rollback_draft_verification(
                    item,
                    keep_len,
                    accepted_draft_count,
                    defer_draft_state=defer_draft_state,
                )
                completion.draft_tokens.clear()
                prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
                if logits is not None:
                    record = Token.from_logits(
                        model=self.model,
                        formatter=self.formatter,
                        prev_tokens=prev_tokens,
                        prev_text_bytes=completion.detokenized_prefix_bytes,
                        token=sampled_token,
                        logits=logits,
                        logprobs_count=completion.logprobs,
                        need_token_logprob=completion.needs_token_logprob,
                    )
                else:
                    record = Token.from_token(
                        model=self.model,
                        prev_tokens=prev_tokens,
                        prev_text_bytes=completion.detokenized_prefix_bytes,
                        token=sampled_token,
                    )
                mismatch_finish_reason: Optional[str] = self.handle_completion_token(
                    completion,
                    sampled_token,
                    record,
                    decoded=False,
                )
                if mismatch_finish_reason is not None:
                    completion.pending_finish_reason = mismatch_finish_reason
                else:
                    decode.sampled_pending_token = sampled_token
                decode.accepted_draft_count = accepted_draft_count
                return
            prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
            if logits is not None:
                record = Token.from_logits(
                    model=self.model,
                    formatter=self.formatter,
                    prev_tokens=prev_tokens,
                    prev_text_bytes=completion.detokenized_prefix_bytes,
                    token=draft_token,
                    logits=logits,
                    logprobs_count=completion.logprobs,
                    need_token_logprob=completion.needs_token_logprob,
                )
            else:
                record = Token.from_token(
                    model=self.model,
                    prev_tokens=prev_tokens,
                    prev_text_bytes=completion.detokenized_prefix_bytes,
                    token=draft_token,
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
                self.record_draft_acceptance_length(accepted_draft_count)
                rejected = max(0, len(completion.draft_tokens) - accepted_draft_count)
                if rejected > 0:
                    self.speculative_stats["draft_tokens_rejected"] += rejected
                self.metrics.draft_target_tokens_wasted_total += (
                    len(decoded_draft_tokens) - accepted_draft_count
                )
                keep_len = item.start_pos + pending_count + accepted_draft_count
                self.rollback_draft_verification(
                    item,
                    keep_len,
                    accepted_draft_count,
                    defer_draft_state=defer_draft_state,
                )
                completion.draft_tokens.clear()
                decode.accepted_draft_count = accepted_draft_count
                self.finish_completion(completion, accepted_finish_reason)
                return

        decode.accepted_draft_count = accepted_draft_count
        if decoded_draft_tokens:
            self.record_draft_acceptance_length(accepted_draft_count)
        if not defer_draft_state:
            self.model.accept_draft_tokens(completion.seq_id, accepted_draft_count)
        elif decoded_draft_tokens:
            decode.deferred_accept_draft_count = accepted_draft_count
        if accepted_draft_count:
            completion.draft_tokens = completion.draft_tokens[accepted_draft_count:]

        final_logits_index = logits_indices[-1]
        next_token = self.sample_completion_token(completion, final_logits_index)
        need_final_logits = self.model.store_logits or completion.needs_token_logprob
        final_logits = (
            self.model.logits(final_logits_index) if need_final_logits else None
        )
        if self.model.store_logits and final_logits is not None:
            self.checkpoint_logits[completion.seq_id] = final_logits
        if completion.draft_tokens and next_token != completion.draft_tokens[0]:
            self.speculative_stats["draft_tokens_rejected"] += len(completion.draft_tokens)
            if not defer_draft_state:
                self.model.truncate_draft_sequence(
                    completion.seq_id,
                    item.start_pos + len(item.tokens),
                )
            else:
                decode.deferred_truncate_draft_len = item.start_pos + len(item.tokens)
            completion.draft_tokens.clear()
        elif completion.draft_tokens and next_token == completion.draft_tokens[0]:
            if getattr(self.model, "draft_target_batching", True):
                completion.draft_tokens = completion.draft_tokens[1:]
            else:
                completion.draft_tokens.clear()
            self.speculative_stats["draft_tokens_accepted"] += 1
            self.metrics.draft_tokens_reused_as_pending_total += 1

        prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
        if final_logits is not None:
            record = Token.from_logits(
                model=self.model,
                formatter=self.formatter,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=next_token,
                logits=final_logits,
                logprobs_count=completion.logprobs,
                need_token_logprob=completion.needs_token_logprob,
            )
        else:
            record = Token.from_token(
                model=self.model,
                prev_tokens=prev_tokens,
                prev_text_bytes=completion.detokenized_prefix_bytes,
                token=next_token,
            )
        final_finish_reason: Optional[str] = self.handle_completion_token(
            completion,
            next_token,
            record,
            decoded=False,
        )
        if final_finish_reason is not None:
            completion.pending_finish_reason = final_finish_reason
        else:
            decode.sampled_pending_token = next_token

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
        self.copy_sequence_state(
            request.base_seq_id,
            checkpoint_seq_id,
            len(request.prompt_tokens),
        )
        self.checkpoint_logits[checkpoint_seq_id] = request.prompt_logits
        self.free_sequences[checkpoint_seq_id] = None
        self.free_sequences.move_to_end(checkpoint_seq_id)
        request.prompt_checkpoint_saved = True
        self.metrics.checkpoint_saves_total += 1

    @staticmethod
    def output_index_to_logits_index(
        output_index: Optional[int],
        output_count: int,
    ) -> Optional[int]:
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
        prompt_output_index: Optional[int],
        prompt_logits: Optional[np.ndarray] = None,
    ) -> None:
        if request.completions:
            return
        assert request.base_seq_id is not None
        prompt_tokens = list(request.prompt_plan.text_tokens)
        prompt_length = len(request.prompt_tokens)
        multimodal_prompt = any(
            segment.kind != "text" for segment in request.prompt_plan.segments
        )
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
                if prompt_length:
                    self.memory_policy.copy_prompt_state(
                        request.base_seq_id,
                        seq_id,
                        prompt_length,
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
                    prompt_length=prompt_length,
                    prompt_text=prompt_text,
                    multimodal_prompt=multimodal_prompt,
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
        if request.payload.max_tokens == 0 or request.effective_max_len == prompt_length:
            for completion in request.completions:
                self.finish_completion(completion, "length")
            self.finalize_request_if_ready(request)
            return
        if prompt_output_index is None:
            if prompt_logits is not None:
                for completion in request.completions:
                    self.sample_completion_from_logits(completion, prompt_logits)
                return
            raise RuntimeError("prompt output is required to start generation")
        for completion in request.completions:
            self.sample_completion(completion, prompt_output_index)

    def sample_completion(
        self,
        completion: Completion,
        output_index: Optional[int],
    ) -> None:
        if completion.finished:
            return
        if output_index is None:
            raise RuntimeError("missing logits output")
        token = self.sample_completion_token(completion, output_index)
        prev_tokens = [*completion.prompt_tokens, *completion.completion_tokens]
        if self.model.store_logits or completion.needs_token_logprob:
            logits = self.model.logits(output_index)
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
        self.metrics.observe_predicted_token()
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
        if not decoded and finish_reason is None and not self.should_defer_draft_fill:
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
        self.metrics.requests_completed_total += 1
        self.release_request(request)
        if request.on_done is not None:
            request.on_done(result)

    def truncate_sequence(
        self,
        seq_id: int,
        keep_len: int,
        *,
        truncate_draft: bool = True,
    ) -> None:
        current_len = self.radix_trie.length(seq_id)
        if current_len <= keep_len:
            return
        keep_pos = self.sequence_history.position_length_for_prefix(seq_id, keep_len)
        if not llama_cpp.llama_memory_seq_rm(self.model.mem, seq_id, keep_pos, -1):
            raise RuntimeError(
                f"failed to truncate model sequence {seq_id} at position {keep_pos}"
            )
        if truncate_draft:
            self.model.truncate_draft_sequence(seq_id, keep_pos)
        self.truncate_sequence_metadata(seq_id, current_len, keep_len)

    def copy_sequence_state(
        self,
        source_sequence_id: int,
        dest_sequence_id: int,
        keep_len: int,
        *,
        copy_all_state: bool = False,
    ) -> None:
        if keep_len <= 0:
            return
        source_length = self.radix_trie.length(source_sequence_id)
        keep_pos = self.sequence_history.position_length_for_prefix(
            source_sequence_id,
            keep_len,
        )
        copy_p0 = 0
        copy_p1 = keep_pos
        if copy_all_state or not self.model.kv_unified:
            copy_p0 = -1
            copy_p1 = -1
        llama_cpp.llama_memory_seq_cp(
            self.model.mem,
            source_sequence_id,
            dest_sequence_id,
            copy_p0,
            copy_p1,
        )
        self.model.copy_draft_sequence(
            source_sequence_id,
            dest_sequence_id,
            copy_p0,
            copy_p1,
        )
        if copy_all_state and source_length > keep_len:
            if not llama_cpp.llama_memory_seq_rm(
                self.model.mem,
                dest_sequence_id,
                keep_pos,
                -1,
            ):
                raise RuntimeError(
                    f"failed to truncate copied model sequence {dest_sequence_id} "
                    f"at position {keep_pos}"
                )
            self.model.truncate_draft_sequence(dest_sequence_id, keep_pos)
        self.radix_trie.copy(source_sequence_id, dest_sequence_id, keep_len)
        self.sequence_history.copy(
            source_sequence_id,
            dest_sequence_id,
            source_length,
            keep_len,
        )

    def truncate_sequence_metadata(
        self,
        seq_id: int,
        current_len: int,
        keep_len: int,
    ) -> None:
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
        self.metrics.checkpoint_evictions_total += 1

    def claim_unused_sequence(self) -> int:
        seq_id = self.unused_sequences.pop()
        self.claimed_sequences.add(seq_id)
        return seq_id

    def claim_free_sequence(self, seq_id: int) -> int:
        del self.free_sequences[seq_id]
        self.claimed_sequences.add(seq_id)
        return seq_id

    def activate_request(
        self,
        request: CompletionRequest,
        *,
        base_seq_id: int,
        sibling_seq_ids: List[int],
    ) -> None:
        request.base_seq_id = base_seq_id
        request.sibling_seq_ids = sibling_seq_ids
        request.completion_seq_ids = [base_seq_id, *sibling_seq_ids]
        request.admitted = True
        self.active_request_ids.add(request.id)

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

    def remove_pending_request(self, request: CompletionRequest) -> None:
        try:
            self.pending_requests.remove(request)
        except ValueError:
            pass

    def fail_request(self, request: CompletionRequest, exc: BaseException) -> None:
        self.remove_pending_request(request)
        if request.id in self.active_request_ids or request.admitted:
            self.release_request(request)
        else:
            self.requests.pop(request.id, None)
        if isinstance(exc, CompletionRequestCancelledError):
            self.metrics.requests_cancelled_total += 1
        else:
            self.metrics.requests_failed_total += 1
        if request.on_error is not None:
            request.on_error(exc)

    @staticmethod
    def _format_prometheus_value(value: Union[int, float]) -> str:
        if isinstance(value, int):
            return str(value)
        if not math.isfinite(value):
            raise ValueError(f"invalid Prometheus metric value: {value}")
        return format(value, ".16g")

    def sequence_cache_metric_definitions(
        self,
    ) -> List[Tuple[str, str, str, Union[int, float]]]:
        return [
            (
                "counter",
                "batch_server:sequence_cache_hits_total",
                "Number of external sequence cache entries hydrated.",
                self.metrics.sequence_cache_hits_total,
            ),
            (
                "counter",
                "batch_server:sequence_cache_save_requests_total",
                "Number of external sequence cache save requests.",
                self.metrics.sequence_cache_save_requests_total,
            ),
            (
                "counter",
                "batch_server:sequence_cache_load_failures_total",
                "Number of external sequence cache load failures.",
                self.metrics.sequence_cache_load_failures_total,
            ),
            (
                "counter",
                "batch_server:sequence_cache_lookup_failures_total",
                "Number of external sequence cache lookup failures.",
                self.metrics.sequence_cache_lookup_failures_total,
            ),
            (
                "counter",
                "batch_server:sequence_cache_save_failures_total",
                "Number of external sequence cache save failures.",
                self.metrics.sequence_cache_save_failures_total,
            ),
            (
                "counter",
                "batch_server:sequence_cache_tokens_loaded_total",
                "Number of prompt tokens hydrated from external sequence cache.",
                self.metrics.sequence_cache_tokens_loaded_total,
            ),
        ]

    def speculative_metric_definitions(
        self,
    ) -> List[Tuple[str, str, str, Union[int, float]]]:
        proposed = self.speculative_stats["draft_tokens_proposed"]
        accepted = self.speculative_stats["draft_tokens_accepted"]
        proposals = self.speculative_stats["draft_proposals"]
        verified_proposals = sum(self.draft_acceptance_length_counts.values())
        verified_accepted = sum(
            accepted_tokens * count
            for accepted_tokens, count in self.draft_acceptance_length_counts.items()
        )
        acceptance_rate = accepted / proposed if proposed > 0 else 0.0
        average_proposed_tokens = proposed / proposals if proposals > 0 else 0.0
        average_verified_acceptance_length = (
            verified_accepted / verified_proposals if verified_proposals > 0 else 0.0
        )
        return [
            (
                "counter",
                "batch_server:draft_proposals_total",
                "Number of speculative draft proposal batches generated.",
                proposals,
            ),
            (
                "counter",
                "batch_server:draft_tokens_proposed_total",
                "Number of speculative draft tokens proposed.",
                proposed,
            ),
            (
                "counter",
                "batch_server:draft_tokens_accepted_total",
                "Number of speculative draft tokens accepted.",
                accepted,
            ),
            (
                "counter",
                "batch_server:draft_tokens_rejected_total",
                "Number of speculative draft tokens rejected.",
                self.speculative_stats["draft_tokens_rejected"],
            ),
            (
                "counter",
                "batch_server:draft_batches_verified_total",
                "Number of completion items that target-verified draft tokens.",
                self.metrics.draft_batches_verified_total,
            ),
            (
                "counter",
                "batch_server:draft_target_tokens_verified_total",
                "Number of draft tokens decoded by the target for verification.",
                self.metrics.draft_target_tokens_verified_total,
            ),
            (
                "counter",
                "batch_server:draft_target_tokens_wasted_total",
                "Number of target-verified draft tokens rejected before emission.",
                self.metrics.draft_target_tokens_wasted_total,
            ),
            (
                "counter",
                "batch_server:draft_tokens_reused_as_pending_total",
                "Number of draft tokens accepted by the next pending-token sample.",
                self.metrics.draft_tokens_reused_as_pending_total,
            ),
            (
                "gauge",
                "batch_server:draft_acceptance_rate",
                "Fraction of proposed draft tokens accepted, including reused pending matches.",
                acceptance_rate,
            ),
            (
                "gauge",
                "batch_server:draft_average_proposed_tokens",
                "Average number of tokens in generated draft proposals.",
                average_proposed_tokens,
            ),
            (
                "gauge",
                "batch_server:draft_average_verified_acceptance_length",
                "Average accepted draft-token prefix length for target-verified proposals only.",
                average_verified_acceptance_length,
            ),
        ]

    def record_draft_acceptance_length(self, accepted_tokens: int) -> None:
        accepted_tokens = max(0, int(accepted_tokens))
        self.draft_acceptance_length_counts[accepted_tokens] = (
            self.draft_acceptance_length_counts.get(accepted_tokens, 0) + 1
        )

    def draft_acceptance_length_metric_lines(self) -> List[str]:
        lines = [
            "# HELP batch_server:draft_acceptance_length_total Number of verified draft proposals by accepted-token prefix length.",
            "# TYPE batch_server:draft_acceptance_length_total counter",
        ]
        for accepted_tokens, count in sorted(self.draft_acceptance_length_counts.items()):
            lines.append(
                f'batch_server:draft_acceptance_length_total{{accepted_tokens="{accepted_tokens}"}} {count}'
            )
        return lines

    def render_prometheus_metrics(self) -> str:
        active_completions = sum(
            1
            for request_id in self.active_request_ids
            for completion in self.requests[request_id].completions
            if not completion.finished
        )
        checkpoint_entries = sum(
            1 for seq_id in self.free_sequences if seq_id in self.checkpoint_logits
        )
        prompt_tokens_seconds = (
            self.metrics.prompt_tokens_total / self.metrics.prompt_seconds_total
            if self.metrics.prompt_seconds_total > 0
            else 0.0
        )
        predicted_tokens_seconds = (
            self.metrics.tokens_predicted_total
            / self.metrics.tokens_predicted_seconds_total
            if self.metrics.tokens_predicted_seconds_total > 0
            else 0.0
        )
        n_busy_slots_per_decode = (
            self.metrics.n_busy_slots_total / self.metrics.n_decode_total
            if self.metrics.n_decode_total > 0
            else 0.0
        )
        draft_provider_metrics: List[Tuple[str, str, str, Union[int, float]]] = []
        if self.model.draft_provider is not None:
            draft_metric_definitions = getattr(
                self.model.draft_provider,
                "metric_definitions",
                None,
            )
            if draft_metric_definitions is not None:
                draft_provider_metrics = cast(
                    List[Tuple[str, str, str, Union[int, float]]],
                    draft_metric_definitions(),
                )
        metrics_def: List[Tuple[str, str, str, Union[int, float]]] = [
            (
                "counter",
                "llamacpp:prompt_tokens_total",
                "Number of prompt tokens processed.",
                self.metrics.prompt_tokens_total,
            ),
            (
                "counter",
                "llamacpp:prompt_seconds_total",
                "Estimated prompt processing time in seconds.",
                self.metrics.prompt_seconds_total,
            ),
            (
                "counter",
                "llamacpp:tokens_predicted_total",
                "Number of generated tokens processed.",
                self.metrics.tokens_predicted_total,
            ),
            (
                "counter",
                "llamacpp:tokens_predicted_seconds_total",
                "Estimated generation processing time in seconds.",
                self.metrics.tokens_predicted_seconds_total,
            ),
            (
                "counter",
                "batch_server:scheduler_step_seconds_total",
                "Total scheduler step wall time.",
                self.metrics.scheduler_step_seconds_total,
            ),
            (
                "counter",
                "batch_server:process_batch_seconds_total",
                "Time spent processing decoded batches after llama_decode().",
                self.metrics.process_batch_seconds_total,
            ),
            (
                "counter",
                "batch_server:sample_seconds_total",
                "Time spent sampling target logits.",
                self.metrics.sample_seconds_total,
            ),
            (
                "counter",
                "batch_server:draft_seconds_total",
                "Speculative draft processing time in seconds.",
                self.metrics.draft_seconds_total,
            ),
            (
                "counter",
                "batch_server:draft_process_seconds_total",
                "Time spent feeding target batches into the draft context.",
                self.metrics.draft_process_seconds_total,
            ),
            (
                "counter",
                "batch_server:draft_generate_seconds_total",
                "Time spent generating speculative draft tokens.",
                self.metrics.draft_generate_seconds_total,
            ),
            (
                "counter",
                "batch_server:draft_sampled_batch_seconds_total",
                "Time spent generating speculative tokens from sampled target batches.",
                self.metrics.draft_sampled_batch_seconds_total,
            ),
            (
                "counter",
                "batch_server:draft_process_calls_total",
                "Number of target-to-draft context processing phases.",
                self.metrics.draft_process_calls_total,
            ),
            (
                "counter",
                "batch_server:draft_generate_calls_total",
                "Number of speculative draft generation phases.",
                self.metrics.draft_generate_calls_total,
            ),
            (
                "counter",
                "batch_server:draft_sampled_batch_calls_total",
                "Number of sampled target batch draft phases.",
                self.metrics.draft_sampled_batch_calls_total,
            ),
            (
                "counter",
                "llamacpp:n_decode_total",
                "Total number of llama_decode() calls.",
                self.metrics.n_decode_total,
            ),
            (
                "counter",
                "batch_server:scheduler_step_calls_total",
                "Number of scheduler steps.",
                self.metrics.scheduler_step_calls_total,
            ),
            (
                "counter",
                "batch_server:process_batch_calls_total",
                "Number of decoded batch processing phases.",
                self.metrics.process_batch_calls_total,
            ),
            (
                "counter",
                "batch_server:sample_calls_total",
                "Number of target logit sampling calls.",
                self.metrics.sample_calls_total,
            ),
            (
                "counter",
                "llamacpp:n_tokens_max",
                "Largest observed n_tokens.",
                self.metrics.n_tokens_max,
            ),
            (
                "gauge",
                "llamacpp:n_busy_slots_per_decode",
                "Average number of busy sequences per llama_decode() call.",
                n_busy_slots_per_decode,
            ),
            (
                "gauge",
                "llamacpp:prompt_tokens_seconds",
                "Average prompt throughput in tokens/s.",
                prompt_tokens_seconds,
            ),
            (
                "gauge",
                "llamacpp:predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                predicted_tokens_seconds,
            ),
            (
                "gauge",
                "llamacpp:requests_processing",
                "Number of requests processing.",
                len(self.active_request_ids),
            ),
            (
                "gauge",
                "llamacpp:requests_deferred",
                "Number of requests deferred.",
                len(self.pending_requests),
            ),
            (
                "counter",
                "batch_server:requests_submitted_total",
                "Number of requests submitted.",
                self.metrics.requests_submitted_total,
            ),
            (
                "counter",
                "batch_server:requests_admitted_total",
                "Number of requests admitted.",
                self.metrics.requests_admitted_total,
            ),
            (
                "counter",
                "batch_server:requests_completed_total",
                "Number of requests completed successfully.",
                self.metrics.requests_completed_total,
            ),
            (
                "counter",
                "batch_server:requests_cancelled_total",
                "Number of requests cancelled.",
                self.metrics.requests_cancelled_total,
            ),
            (
                "counter",
                "batch_server:requests_failed_total",
                "Number of requests failed.",
                self.metrics.requests_failed_total,
            ),
            (
                "gauge",
                "batch_server:active_completions",
                "Number of active unfinished completions.",
                active_completions,
            ),
            (
                "gauge",
                "batch_server:claimed_sequences",
                "Number of claimed sequence ids.",
                len(self.claimed_sequences),
            ),
            (
                "gauge",
                "batch_server:free_sequences",
                "Number of reusable free sequence ids.",
                len(self.free_sequences),
            ),
            (
                "gauge",
                "batch_server:unused_sequences",
                "Number of unused sequence ids.",
                len(self.unused_sequences),
            ),
            (
                "gauge",
                "batch_server:checkpoint_entries",
                "Number of free sequence checkpoints with stored logits.",
                checkpoint_entries,
            ),
            (
                "counter",
                "batch_server:checkpoint_hits_total",
                "Number of exact checkpoint hits.",
                self.metrics.checkpoint_hits_total,
            ),
            (
                "counter",
                "batch_server:checkpoint_saves_total",
                "Number of prompt checkpoints saved.",
                self.metrics.checkpoint_saves_total,
            ),
            (
                "counter",
                "batch_server:checkpoint_evictions_total",
                "Number of free checkpoints evicted.",
                self.metrics.checkpoint_evictions_total,
            ),
            *self.sequence_cache_metric_definitions(),
            *self.speculative_metric_definitions(),
            *draft_provider_metrics,
            (
                "gauge",
                "batch_server:radix_trie_sequences",
                "Number of sequences tracked in the radix trie.",
                len(self.radix_trie.sequence_lengths),
            ),
            (
                "gauge",
                "batch_server:radix_trie_tokens",
                "Total tokens tracked in the radix trie.",
                self.sequence_history.size,
            ),
        ]
        lines: List[str] = []
        for metric_type, name, help_text, value in metrics_def:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {self._format_prometheus_value(value)}")
        lines.extend(self.draft_acceptance_length_metric_lines())
        lines.append("")
        return "\n".join(lines)

    def finalize_cancelled(self) -> bool:
        finalized = False
        for request in list(self.pending_requests):
            if request.cancelled:
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

    def call_on_scheduler(self, callback: Callable[[], Any]) -> Any:
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}
        done = threading.Event()

        def run_callback() -> None:
            try:
                result_box["result"] = callback()
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc
            finally:
                done.set()

        self.enqueue(run_callback)
        done.wait()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result")

    def call_on_idle_scheduler(self, callback: Callable[[], Any]) -> Any:
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}
        done = threading.Event()

        def run_callback() -> None:
            if not self.scheduler.is_idle():
                with self.condition:
                    self.commands.appendleft(run_callback)
                    self.condition.notify_all()
                return
            try:
                result_box["result"] = callback()
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc
            finally:
                done.set()

        self.enqueue(run_callback)
        done.wait()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result")

    def create_embedding(
        self,
        payload: CreateEmbeddingRequest,
    ) -> CreateEmbeddingResponse:
        embedding = self.call_on_idle_scheduler(
            lambda: self.scheduler.create_embedding(payload)
        )
        return cast(CreateEmbeddingResponse, embedding)

    def render_prometheus_metrics(self) -> str:
        metrics = self.call_on_scheduler(self.scheduler.render_prometheus_metrics)
        return cast(str, metrics)

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
        request = self.request_from_payload(payload)
        return self.submit_request(request)

    def request_from_payload(
        self,
        payload: CreateCompletionRequest,
    ) -> CompletionRequest:
        prompt_text, prompt_plan = self.prepare_completion_prompt(payload)
        return self.request_from_prepared(
            payload=payload,
            prompt_text=prompt_text,
            prompt_plan=prompt_plan,
        )

    def request_from_prepared(
        self,
        *,
        payload: CreateCompletionRequest,
        prompt_text: str,
        prompt_plan: PromptPlan,
        grammar_text: Optional[str] = None,
        chat_tool_name: Optional[str] = None,
        on_stream_chunk: Optional[Callable[[CompletionChunk], None]] = None,
        on_done: Optional[Callable[[OpenAICompletion], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> CompletionRequest:
        model = self.scheduler.model
        prompt_visible_start = self.prompt_visible_start(prompt_plan)
        return CompletionRequest.from_prepared(
            payload=payload,
            prompt_text=prompt_text,
            prompt_plan=prompt_plan,
            max_seq_len=model.max_seq_len,
            max_output_tokens=model.max_output_tokens,
            prompt_visible_start=prompt_visible_start,
            prompt_records=self.initial_prompt_records(
                payload,
                prompt_plan,
                prompt_visible_start,
            ),
            grammar_text=grammar_text,
            chat_tool_name=chat_tool_name,
            on_stream_chunk=on_stream_chunk,
            on_done=on_done,
            on_error=on_error,
        )

    def prepare_completion_prompt(
        self,
        payload: CreateCompletionRequest,
    ) -> Tuple[str, PromptPlan]:
        model = self.scheduler.model
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
        return prompt_text, PromptPlan.from_tokens(prompt_text, prompt_tokens)

    def prompt_visible_start(self, prompt_plan: PromptPlan) -> int:
        model = self.scheduler.model
        if prompt_plan.text_tokens and prompt_plan.text_tokens[0] == model.bos_token:
            return 1
        return 0

    def initial_prompt_records(
        self,
        payload: CreateCompletionRequest,
        prompt_plan: PromptPlan,
        prompt_visible_start: int,
    ) -> List[Token]:
        if not (payload.echo and payload.logprobs is not None):
            return []
        if prompt_visible_start >= len(prompt_plan.text_tokens):
            return []
        model = self.scheduler.model
        first_pos = prompt_visible_start
        first_token = prompt_plan.text_tokens[first_pos]
        return [
            Token(
                token=first_token,
                text_bytes=model.token_bytes_with_prev(
                    prompt_plan.text_tokens[:first_pos],
                    first_token,
                ),
                token_logprob=None,
                top_logprobs=None,
            )
        ]


def create_app() -> FastAPI:
    app = FastAPI()
    app.state.service = None
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

    async def disconnected_cancelled_response_or_raise(
        http_request: Request,
        exc: BaseException,
    ) -> Response:
        if isinstance(exc, CompletionRequestCancelledError):
            if await http_request.is_disconnected():
                return Response(status_code=204)
        raise exc

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
        chunk_payloads: Callable[[CompletionChunk], Iterable[Any]],
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
        chunk_payloads: Callable[[CompletionChunk], Iterable[Any]],
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
        _: Callable[[], None],
        *,
        initial_payloads: Iterable[BaseModel | Dict[str, Any]],
        chunk_payloads: Callable[[CompletionChunk], Iterable[Any]],
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
    async def create_completion(  # pyright: ignore[reportUnusedFunction]
        http_request: Request,
        body: CreateCompletionRequest,
    ):
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

    @app.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
    async def create_embedding(  # pyright: ignore[reportUnusedFunction]
        body: CreateEmbeddingRequest,
    ) -> JSONResponse:
        service: CompletionService = app.state.service
        try:
            embedding = await asyncio.to_thread(service.create_embedding, body)
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        return JSONResponse(embedding.model_dump(mode="json", exclude_none=True))

    @app.post("/v1/chat/completions")
    async def create_chat_completion(  # pyright: ignore[reportUnusedFunction]
        http_request: Request, body: CreateChatCompletionRequest
    ):
        service: CompletionService = app.state.service
        formatter = service.formatter
        try:
            parts = formatter.completion_request_from_chat_request(
                body,
            )
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        template_functions = (
            [function.to_template_function() for function in body.functions]
            if body.functions is not None
            else None
        )
        template_tools = (
            [tool.to_template_tool() for tool in body.tools]
            if body.tools is not None
            else None
        )
        try:
            request = service.request_from_prepared(
                payload=parts.payload,
                prompt_text=parts.prompt_text,
                prompt_plan=parts.prompt_plan,
                grammar_text=parts.grammar_text,
                chat_tool_name=parts.tool_name,
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
                    parts.tool_name,
                    functions=template_functions,
                    tools=template_tools,
                    parsed_states=parsed_states,
                    generation_prompt=parts.generation_prompt,
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
            parts.tool_name,
            functions=template_functions,
            tools=template_tools,
            generation_prompt=parts.generation_prompt,
        )
        if isinstance(chat_response, BaseModel):
            return JSONResponse(
                chat_response.model_dump(mode="json", exclude_none=True)
            )
        return JSONResponse(chat_response)

    @app.post("/v1/responses")
    async def create_response(  # pyright: ignore[reportUnusedFunction]
        http_request: Request,
        body: CreateResponseRequest,
    ):
        service: CompletionService = app.state.service
        formatter = service.formatter
        try:
            chat_parts = formatter.response_request_to_chat_parts(body)
            parts = formatter.completion_request_from_response_chat_parts(chat_parts)
        except CompletionRequestValidationError as exc:
            raise bad_request(exc) from exc
        response_tools = chat_parts.tools
        try:
            request = service.request_from_prepared(
                payload=parts.payload,
                prompt_text=parts.prompt_text,
                prompt_plan=parts.prompt_plan,
                grammar_text=parts.grammar_text,
                chat_tool_name=parts.tool_name,
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
                    parts.tool_name,
                    tools=response_tools,
                    parsed_states=parsed_states,
                    generation_prompt=parts.generation_prompt,
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
                parts.tool_name,
                tools=response_tools,
                generation_prompt=parts.generation_prompt,
            )
        )

    @app.websocket("/v1/responses")
    async def responses_websocket(  # pyright: ignore[reportUnusedFunction]
        websocket: WebSocket,
    ):
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
                    chat_parts = formatter.response_request_to_chat_parts(body)
                    parts = formatter.completion_request_from_response_chat_parts(chat_parts)
                    response_tools = chat_parts.tools
                    request = service.request_from_prepared(
                        payload=parts.payload,
                        prompt_text=parts.prompt_text,
                        prompt_plan=parts.prompt_plan,
                        grammar_text=parts.grammar_text,
                        chat_tool_name=parts.tool_name,
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
                        parts.tool_name,
                        tools=response_tools,
                        parsed_states=parsed_states,
                        generation_prompt=parts.generation_prompt,
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

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:  # pyright: ignore[reportUnusedFunction]
        service = app.state.service
        model = service.scheduler.model
        model_id = getattr(model, "model_alias", None) or model.model_path
        created = int(time.time())
        return ModelListResponse(
            data=[
                ModelCardResponse(
                    id=model_id,
                    created=created,
                    owned_by="llama-cpp-python",
                )
            ]
        )

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz() -> HealthzResponse:  # pyright: ignore[reportUnusedFunction]
        return HealthzResponse()

    @app.get("/metrics")
    async def metrics() -> Response:  # pyright: ignore[reportUnusedFunction]
        service = cast(Optional[CompletionService], getattr(app.state, "service", None))
        if service is None:
            raise HTTPException(status_code=503, detail="completion service unavailable")
        payload = await asyncio.to_thread(service.render_prometheus_metrics)
        return Response(
            payload,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


APP = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config-file", required=True)
    args = parser.parse_args()
    config = ConfigFile.load(args.config_file)
    model_path = config.model.resolve_model_path()
    loras = [
        Model.LoraAdapter(path=lora.resolve_path(), scale=lora.scale)
        for lora in config.model.loras
    ]
    model = Model(
        model_path=model_path,
        model_alias=config.model.alias,
        chat_template=config.model.chat_template,
        loras=loras,
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
        embedding=config.model.embedding,
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
        draft_model_path=config.model.resolve_draft_model_path(),
        draft_model_num_pred_tokens=config.model.draft_model_num_pred_tokens,
        draft_model_max_ngram_size=config.model.draft_model_max_ngram_size,
        draft_model_top_k=config.model.draft_model_top_k,
        draft_model_p_min=config.model.draft_model_p_min,
        draft_model_max_batch_size=config.model.draft_model_max_batch_size,
        draft_model_threads=config.model.draft_model_threads,
        draft_model_threads_batch=config.model.draft_model_threads_batch,
        response_schema=config.model.response_schema,
        store_logits=config.model.store_logits,
    )
    if config.model.mtmd is not None:
        if model.chat_formatter is None:
            raise RuntimeError("MTMD requires a GGUF chat template")
        mmproj_path = config.model.mtmd.resolve_mmproj_path()
        embedding_cache: Optional[MTMDEmbeddingCache] = None
        if config.model.mtmd.embedding_cache is not None:
            embedding_cache = MTMDEmbeddingCache(
                path=config.model.mtmd.embedding_cache.path,
                max_bytes=config.model.mtmd.embedding_cache.max_bytes,
                model_fingerprint=MTMDEmbeddingCache.fingerprint_file(model_path),
                mmproj_fingerprint=MTMDEmbeddingCache.fingerprint_file(mmproj_path),
            )
        model.mtmd_processor = MTMDProcessor(
            model_path=model.model_path,
            llama_model=model.llama_model,
            chat_formatter=model.chat_formatter,
            tokenize=model.tokenize,
            n_embd_inp=model.n_embd_inp,
            n_batch=model.n_batch,
            n_ubatch=model.n_ubatch,
            n_threads_batch=model.n_threads_batch,
            mmproj_path=mmproj_path,
            batch_max_tokens=config.model.mtmd.batch_max_tokens,
            embedding_cache=embedding_cache,
            allowed_media_domains=config.model.mtmd.allowed_media_domains,
            allowed_local_media_path=config.model.mtmd.allowed_local_media_path,
            image_max_bytes=config.model.mtmd.image_max_bytes,
            audio_max_bytes=config.model.mtmd.audio_max_bytes,
            video_max_bytes=config.model.mtmd.video_max_bytes,
            image_timeout_seconds=config.model.mtmd.image_timeout_seconds,
        )
    sequence_cache: Optional[SequenceCache] = None
    if config.disk_cache is not None:
        sequence_cache = SequenceDiskCache(
            path=config.disk_cache.path,
            max_bytes=config.disk_cache.max_bytes,
            min_tokens=config.disk_cache.min_tokens,
            compatibility_key=SequenceDiskCache.compatibility_key_for_model(model),
        )
    scheduler = CompletionScheduler(model, sequence_cache=sequence_cache)
    APP.state.service = CompletionService(scheduler)
    try:
        uvicorn.run(
            APP, host=config.server.host, port=config.server.port, log_level="info"
        )
    finally:
        APP.state.service.close()


if __name__ == "__main__":
    main()
