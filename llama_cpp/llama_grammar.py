"""Python implementation of llama grammar parser directly translated from C++ source file in vendor/llama.cpp/common/grammar-parser.cpp."""

# flake8: noqa
from itertools import groupby
from typing import (
    Any,
    Set,
    List,
    Optional,
    Tuple,
    Union,
)

import enum
import typing

import llama_cpp.llama_cpp as llama_cpp

class GrammarElementType(enum.IntEnum):
    END = llama_cpp.LLAMA_GRETYPE_END
    ALT = llama_cpp.LLAMA_GRETYPE_ALT
    RULE_REF = llama_cpp.LLAMA_GRETYPE_RULE_REF
    CHAR = llama_cpp.LLAMA_GRETYPE_CHAR
    CHAR_NOT = llama_cpp.LLAMA_GRETYPE_CHAR_NOT
    CHAR_RNG_UPPER = llama_cpp.LLAMA_GRETYPE_CHAR_RNG_UPPER
    CHAR_ALT = llama_cpp.LLAMA_GRETYPE_CHAR_ALT
    CHAR_ANY = llama_cpp.LLAMA_GRETYPE_CHAR_ANY

import dataclasses

@dataclasses.dataclass
class GrammarElement:
    type: GrammarElementType
    value: int

@dataclasses.dataclass
class ParseState:
    symbol_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
    rules: typing.List[typing.List[GrammarElement]] = dataclasses.field(default_factory=list)


def decode_utf8(src: str) -> typing.Tuple[int, str]:
    lookup: list[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4]
    first_byte: int = ord(src[0])
    highbits: int = first_byte >> 4
    length: int = lookup[highbits]
    mask: int = (1 << (8 - length)) - 1
    value: int = first_byte & mask
    end: int = min(len(src), length)  # Prevent overrun

    pos: int = 1
    for pos in range(1, end):
        if not src[pos]:
            break
        value = (value << 6) + (ord(src[pos]) & 0x3F)

    return value, src[pos:] if pos < len(src) else ""


def get_symbol_id(state: ParseState, name: str) -> int:
    next_id = len(state.symbol_ids)
    return state.symbol_ids.setdefault(name, next_id)


def generate_symbol_id(state: ParseState, base_name: str) -> int:
    next_id = len(state.symbol_ids)
    state.symbol_ids[f"{base_name}_{next_id}"] = next_id
    return next_id


def add_rule(state: ParseState, rule_id: int, rule: typing.List[GrammarElement]) -> None:
    if len(state.rules) <= rule_id:
        state.rules.extend([[]] * (rule_id + 1 - len(state.rules)))
    state.rules[rule_id] = rule


def is_digit_char(c: str) -> bool:
    return "0" <= c <= "9"


def is_word_char(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "-" or is_digit_char(c)


def parse_hex(src: str, size: int) -> typing.Tuple[int, str]:
    pos = 0
    value = 0
    for _ in range(size):
        value <<= 4
        c = src[pos]
        if "a" <= c <= "f":
            value += ord(c) - ord("a") + 10
        elif "A" <= c <= "F":
            value += ord(c) - ord("A") + 10
        elif "0" <= c <= "9":
            value += ord(c) - ord("0")
        else:
            break
        pos += 1
    if pos != size:
        raise ValueError(f"expecting {size} hex chars at {src}")
    return value, src[pos:]


def parse_space(src: str, newline_ok: bool) -> str:
    pos = 0
    while pos < len(src) and (src[pos] in (" ", "\t", "#") or (newline_ok and src[pos] in ("\r", "\n"))):
        if src[pos] == "#":
            while pos < len(src) and src[pos] not in ("\r", "\n"):
                pos += 1
        pos += 1
    return src[pos:]


def parse_name(src: str) -> typing.Tuple[str, str]:
    pos = 0
    try:
        while is_word_char(src[pos]):
            pos += 1
    except IndexError:
        return src, ""
    if pos == 0:
        raise ValueError(f"expecting name at {src}")
    return src[:pos], src[pos:]


def parse_int(src: str) -> typing.Tuple[int, str]:
    pos = 0
    while is_digit_char(src[pos]):
        pos += 1
    if pos == 0:
        raise ValueError(f"expecting integer at {src}")
    return int(src[:pos]), src[pos:]


def parse_char(src: str) -> typing.Tuple[int, str]:
    if src[0] == "\\":
        if src[1] == "x":
            return parse_hex(src[2:], 2)
        elif src[1] == "u":
            return parse_hex(src[2:], 4)
        elif src[1] == "U":
            return parse_hex(src[2:], 8)
        elif src[1] == "t":
            return ord("\t"), src[2:]
        elif src[1] == "r":
            return ord("\r"), src[2:]
        elif src[1] == "n":
            return ord("\n"), src[2:]
        elif src[1] in ('\\', '"', '[', ']'):
            return ord(src[1]), src[2:]
        else:
            raise ValueError(f"unknown escape at {src}")
    elif src:
        return decode_utf8(src)
    raise ValueError("unexpected end of input")


def parse_sequence(state: ParseState, src: str, rule_name: str, out_elements: typing.List[GrammarElement], is_nested: bool) -> str:
    last_sym_start = len(out_elements)
    pos = src

    def handle_repetitions(min_times: int, max_times: int) -> None:
        nonlocal last_sym_start
        nonlocal pos
        nonlocal out_elements

        if last_sym_start == len(out_elements):
            raise ValueError(f"expecting preceding item to */+/?/{{ at {pos}")

        previous_elements = out_elements[last_sym_start:]
        if min_times == 0:
            out_elements = out_elements[:last_sym_start]
        else:
            for i in range(1, min_times):
                out_elements.extend(previous_elements)
        last_rec_rule_id = 0
        n_opt = 1 if max_times < 0 else max_times - min_times

        rec_rule = list(previous_elements)
        for i in range(n_opt):
            rec_rule = rec_rule[:len(previous_elements)]
            rec_rule_id = generate_symbol_id(state, rule_name)
            if i > 0 or max_times < 0:
                rec_rule.append(GrammarElement(GrammarElementType.RULE_REF, rec_rule_id if max_times < 0 else last_rec_rule_id))
            rec_rule.append(GrammarElement(GrammarElementType.ALT, 0))
            rec_rule.append(GrammarElement(GrammarElementType.END, 0))
            add_rule(state, rec_rule_id, rec_rule)
            last_rec_rule_id = rec_rule_id
        if n_opt > 0:
            out_elements.append(GrammarElement(GrammarElementType.RULE_REF, last_rec_rule_id))

    while pos:
        if pos.startswith('"'):
            pos = pos[1:]
            last_sym_start = len(out_elements)
            while pos[0] != '"':
                if not pos:
                    raise ValueError("unexpected end of input")
                char, pos = parse_char(pos)
                out_elements.append(GrammarElement(GrammarElementType.CHAR, char))
            pos = parse_space(pos[1:], is_nested)
        elif pos.startswith("["):
            pos = pos[1:]
            start_type = GrammarElementType.CHAR
            if pos[0] == "^":
                start_type = GrammarElementType.CHAR_NOT
                pos = pos[1:]
            last_sym_start = len(out_elements)
            while pos[0] != "]":
                if not pos:
                    raise ValueError("unexpected end of input")
                char, pos = parse_char(pos)
                type = GrammarElementType.CHAR_ALT if last_sym_start < len(out_elements) else start_type
                out_elements.append(GrammarElement(type, char))
                if pos[0] == "-" and pos[1] != "]":
                    if not pos[1]:
                        raise ValueError("unexpected end of input")
                    endchar, pos = parse_char(pos[1:])
                    out_elements.append(GrammarElement(GrammarElementType.CHAR_RNG_UPPER, endchar))
            pos = parse_space(pos[1:], is_nested)
        elif is_word_char(pos[0]):
            name, rest = parse_name(pos)
            ref_rule_id = get_symbol_id(state, name)
            pos = parse_space(rest, is_nested)
            last_sym_start = len(out_elements)
            out_elements.append(GrammarElement(GrammarElementType.RULE_REF, ref_rule_id))
        elif pos.startswith("("):
            pos = parse_space(pos[1:], newline_ok=True)
            sub_rule_id = generate_symbol_id(state, rule_name)
            pos = parse_alternates(state, pos, rule_name, sub_rule_id, is_nested=True)
            last_sym_start = len(out_elements)
            out_elements.append(GrammarElement(GrammarElementType.RULE_REF, sub_rule_id))
            if pos[0] != ")":
                raise ValueError(f"expecting ')' at {pos}")
            pos = parse_space(pos[1:], is_nested)
        elif pos.startswith("."):
            last_sym_start = len(out_elements)
            out_elements.append(GrammarElement(GrammarElementType.CHAR_ANY, 0))
            pos = parse_space(pos[1:], is_nested)
        elif pos.startswith("*"):
            pos = parse_space(pos[1:], is_nested)
            handle_repetitions(0, -1)
        elif pos.startswith("+"):
            pos = parse_space(pos[1:], is_nested)
            handle_repetitions(1, -1)
        elif pos.startswith("?"):
            pos = parse_space(pos[1:], is_nested)
            handle_repetitions(0, 1)
        elif pos.startswith("{"):
            pos = parse_space(pos[1:], is_nested)
            if not is_digit_char(pos[0]):
                raise ValueError(f"expecting an int at {pos}")
            min_times, pos = parse_int(pos)
            pos = parse_space(pos, is_nested)

            max_times = -1

            if pos[0] == "}":
                max_times = min_times
                pos = parse_space(pos[1:], is_nested)
            elif pos[0] == ",":
                pos = parse_space(pos[1:], is_nested)
                if is_digit_char(pos[0]):
                    max_times, pos = parse_int(pos)
                    pos = parse_space(pos, is_nested)
                if pos[0] != "}":
                    raise ValueError("expecting '}' at {}".format(pos))
                pos = parse_space(pos[1:], is_nested)
            else:
                raise ValueError(f"expecting ',' at {pos}")
            handle_repetitions(min_times, max_times)
        else:
            break
    return pos


def parse_alternates(state: ParseState, src: str, rule_name: str, rule_id: int, is_nested: bool) -> str:
    rule = []
    pos = parse_sequence(state, src, rule_name, rule, is_nested)
    while pos.startswith("|"):
        rule.append(GrammarElement(GrammarElementType.ALT, 0))
        pos = parse_space(pos[1:], newline_ok=True)
        pos = parse_sequence(state, pos, rule_name, rule, is_nested)
    rule.append(GrammarElement(GrammarElementType.END, 0))
    add_rule(state, rule_id, rule)
    return pos


def parse_rule(state: ParseState, src: str) -> str:
    name, s = parse_name(src)
    s = parse_space(s, newline_ok=False)
    rule_id = get_symbol_id(state, name)

    if not s.startswith("::="):
        raise ValueError(f"expecting ::= at {s}")

    s = s[3:]

    s = parse_space(s, newline_ok=True)

    s = parse_alternates(state, s, name, rule_id, is_nested=False)

    if s.startswith("\r"):
        s = s[2:] if s[1] == "\n" else s[1:]
    elif s.startswith("\n"):
        s = s[1:]
    elif s:
        raise ValueError(f"expecting newline or end at {s}")
    return parse_space(s, newline_ok=True)


def parse(gbnf: str) -> ParseState:
    state = ParseState()
    s = parse_space(gbnf, newline_ok=True)
    while s:
        s = parse_rule(state, s)
    # validate
    for rule in state.rules:
        for elem in rule:
            if elem.type == GrammarElementType.RULE_REF:
                if elem.value >= len(state.rules) or not state.rules[elem.value]:
                    for k, v in state.symbol_ids.items():
                        if v == elem.value:
                            raise ValueError(f"Undefined rule identifier '{k}'")
    return state


def is_char_element(elem: GrammarElement) -> bool:
    return elem.type in (
        GrammarElementType.CHAR, 
        GrammarElementType.CHAR_NOT,
        GrammarElementType.CHAR_ALT,
        GrammarElementType.CHAR_RNG_UPPER,
        GrammarElementType.CHAR_ANY
    )


def print_grammar_char(file: typing.TextIO, c: int) -> None:
    if 0x20 <= c <= 0x7f:
        print(chr(c), end="", file=file)
    else:
        print(f"<U+{c:04X}>", end="", file=file)


def print_rule(
    file: typing.TextIO,
    rule_id: int,
    rule: typing.List[GrammarElement],
    symbol_id_names: typing.Dict[int, str],
) -> None:
    if not rule or rule[-1].type != GrammarElementType.END:
        raise ValueError(f"malformed rule, does not end with LLAMA_GRETYPE_END: {rule_id}")

    print(f"{symbol_id_names[rule_id]} ::=", end=" ", file=file)

    for i, elem in enumerate(rule[:-1]):
        if elem.type == GrammarElementType.END:
            raise ValueError(f"unexpected end of rule: {rule_id}, {i}")
        if elem.type == GrammarElementType.ALT:
            print("| ", end="", file=file)
        elif elem.type == GrammarElementType.RULE_REF:
            print(f"{symbol_id_names[elem.value]} ", end="", file=file)
        elif elem.type == GrammarElementType.CHAR:
            print("[", end="", file=file)
            print_grammar_char(file, elem.value)
        elif elem.type == GrammarElementType.CHAR_NOT:
            print("[^", end="", file=file)
            print_grammar_char(file, elem.value)
        elif elem.type == GrammarElementType.CHAR_RNG_UPPER:
            if i == 0 or not is_char_element(rule[i - 1]):
                raise ValueError(f"LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: {rule_id}, {i}")
            print(f"-", end="", file=file)
            print_grammar_char(file, elem.value)
        elif elem.type == GrammarElementType.CHAR_ALT:
            if i == 0 or not is_char_element(rule[i - 1]):
                raise ValueError(f"LLAMA_GRETYPE_CHAR_ALT without preceding char: {rule_id}, {i}")
            print_grammar_char(file, elem.value)
        elif elem.type == GrammarElementType.CHAR_ANY:
            print(".", end="", file=file)
        if is_char_element(elem):
            if rule[i + 1].type in (GrammarElementType.CHAR_ALT, GrammarElementType.CHAR_RNG_UPPER, GrammarElementType.CHAR_ANY):
                continue
            print("] ", end="", file=file)
    print(file=file)


def print_grammar(file: typing.TextIO, state: ParseState) -> None:
    try:
        symbol_id_names = {v: k for k, v in state.symbol_ids.items()}
        for i, rule in enumerate(state.rules):
            print_rule(file, i, rule, symbol_id_names)
    except Exception as err:
        print(f"\nerror printing grammar: {err}", file=file)
        raise err

import ctypes

class LlamaGrammar:
    def __init__(self, parse_state: ParseState):
        self.parse_state = parse_state

        self._grammar_rules = parse_state.rules
        self._n_rules = len(self._grammar_rules)
        self._start_rule_index = parse_state.symbol_ids["root"]

        self._element_lists = [
            [
                llama_cpp.llama_grammar_element(ctypes.c_int(elem.type.value), ctypes.c_uint32(elem.value))
                for elem in subvector
            ]
            for subvector in self._grammar_rules
        ]

        # Step 2: Convert each list to llama_grammar_element array and get pointer
        self._element_arrays = [
            (llama_cpp.llama_grammar_element * len(sublist))(*sublist)
            for sublist in self._element_lists
        ]

        # Step 3: Get pointer of each array
        self._element_array_pointers = [
            ctypes.cast(subarray, llama_cpp.llama_grammar_element_p) for subarray in self._element_arrays
        ]

        # Step 4: Make array of these pointers and get its pointer
        self._rules = (llama_cpp.llama_grammar_element_p * len(self._element_array_pointers))(
            *self._element_array_pointers
        )

        self.grammar = None
        self._init_grammar()


    def _init_grammar(self):
        grammar = llama_cpp.llama_grammar_init(
            self._rules, ctypes.c_size_t(self._n_rules), ctypes.c_size_t(self._start_rule_index)
        )

        if grammar is None:
            raise ValueError("Failed to create grammar")

        self.grammar = grammar

    def __del__(self):
        if self.grammar is not None:
            llama_cpp.llama_grammar_free(self.grammar)
            self.grammar = None

    def reset(self):
        if self.grammar is not None:
            llama_cpp.llama_grammar_free(self.grammar)
        self._init_grammar()

    @classmethod
    def from_string(cls, grammar: str, verbose: bool = True) -> "LlamaGrammar":
        parsed_grammar = parse(grammar)
        return cls(parsed_grammar)

    @classmethod
    def from_json_schema(cls, json_schema: str, verbose: bool = True) -> "LlamaGrammar":
        return cls.from_string(json_schema_to_gbnf(json_schema), verbose=verbose)


"""llama.cpp gbnf rules from vendor/llama.cpp/grammars"""

ARITHMETIC_GBNF = r"""
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"""

C_GBNF = r"""
root ::= (declaration)*

declaration ::= dataType identifier "(" parameter? ")" "{" statement* "}"

dataType  ::= "int" ws | "float" ws | "char" ws
identifier ::= [a-zA-Z_] [a-zA-Z_0-9]*

parameter ::= dataType identifier

statement ::=
    ( dataType identifier ws "=" ws expression ";" ) |
    ( identifier ws "=" ws expression ";" ) |
    ( identifier ws "(" argList? ")" ";" ) |
    ( "return" ws expression ";" ) |
    ( "while" "(" condition ")" "{" statement* "}" ) |
    ( "for" "(" forInit ";" ws condition ";" ws forUpdate ")" "{" statement* "}" ) |
    ( "if" "(" condition ")" "{" statement* "}" ("else" "{" statement* "}")? ) |
    ( singleLineComment ) |
    ( multiLineComment )

forInit ::= dataType identifier ws "=" ws expression | identifier ws "=" ws expression
forUpdate ::= identifier ws "=" ws expression

condition ::= expression relationOperator expression
relationOperator ::= ("<=" | "<" | "==" | "!=" | ">=" | ">")

expression ::= term (("+" | "-") term)*
term ::= factor(("*" | "/") factor)*

factor ::= identifier | number | unaryTerm | funcCall | parenExpression
unaryTerm ::= "-" factor
funcCall ::= identifier "(" argList? ")"
parenExpression ::= "(" ws expression ws ")"

argList ::= expression ("," ws expression)*

number ::= [0-9]+

singleLineComment ::= "//" [^\n]* "\n"
multiLineComment ::= "/*" ( [^*] | ("*" [^/]) )* "*/"

ws ::= ([ \t\n]+)
"""

CHESS_GBNF = r"""
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
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""

JAPANESE_GBNF = r"""
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
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""

JSON_ARR_GBNF = r"""
# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays

root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"

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
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""


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
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"""

LIST_GBNF = r"""
root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"
"""

"""llama.cpp json-schema to grammar converter from vendor/llama.cpp/examples/json-schema-to-grammar.py"""
import json
import re
from typing import List, Optional

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'


INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'


def _build_repetition(
    item_rule, min_items, max_items, separator_rule=None, item_rule_is_literal=False
):
    if not separator_rule:
        if min_items == 0 and max_items == 1:
            return f"{item_rule}?"
        elif min_items == 1 and max_items is None:
            return f"{item_rule}+"

    result = ""

    if min_items > 0:
        if item_rule_is_literal and separator_rule is None:
            result = '"' + (item_rule[1:-1] * min_items) + '"'
        else:
            result = (f" {separator_rule} " if separator_rule else " ").join(
                [item_rule] * min_items
            )

    def opt_repetitions(up_to_n, prefix_with_sep=False):
        """
        - n=4, no sep:             '(a (a (a (a)?)?)?)?'
        - n=4, sep=',', prefix:    '("," a ("," a ("," a ("," a)?)?)?)?'
        - n=4, sep=',', no prefix: '(a ("," a ("," a ("," a)?)?)?)?'
        """

        content = (
            f"{separator_rule} {item_rule}"
            if prefix_with_sep and separator_rule
            else item_rule
        )
        if up_to_n == 0:
            return ""
        elif up_to_n == 1:
            return f"({content})?"
        elif separator_rule and not prefix_with_sep:
            return f"({content} {opt_repetitions(up_to_n - 1, prefix_with_sep=True)})?"
        else:
            return (f"({content} " * up_to_n).rstrip() + (")?" * up_to_n)

    if min_items > 0 and max_items != min_items:
        result += " "

    if max_items is not None:
        result += opt_repetitions(max_items - min_items, prefix_with_sep=min_items > 0)
    else:
        item_operator = f'({separator_rule + " " if separator_rule else ""}{item_rule})'

        if min_items == 0 and separator_rule:
            result = f"({item_rule} {item_operator}*)?"
        else:
            result += f"{item_operator}*"

    return result


class BuiltinRule:
    def __init__(self, content: str, deps: list = None):
        self.content = content
        self.deps = deps or []


_up_to_15_digits = _build_repetition("[0-9]", 0, 15)

PRIMITIVE_RULES = {
    "boolean": BuiltinRule('("true" | "false") space', []),
    "decimal-part": BuiltinRule("[0-9] " + _up_to_15_digits, []),
    "integral-part": BuiltinRule("[0-9] | [1-9] " + _up_to_15_digits, []),
    "number": BuiltinRule(
        '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
        ["integral-part", "decimal-part"],
    ),
    "integer": BuiltinRule('("-"? integral-part) space', ["integral-part"]),
    "value": BuiltinRule(
        "object | array | string | number | boolean | null",
        ["object", "array", "string", "number", "boolean", "null"],
    ),
    "object": BuiltinRule(
        '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
        ["string", "value"],
    ),
    "array": BuiltinRule(
        '"[" space ( value ("," space value)* )? "]" space', ["value"]
    ),
    "uuid": BuiltinRule(
        r'"\"" '
        + ' "-" '.join("[0-9a-fA-F]" * n for n in [8, 4, 4, 4, 12])
        + r' "\"" space',
        [],
    ),
    "char": BuiltinRule(
        r'[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
        [],
    ),
    "string": BuiltinRule(r'"\"" char* "\"" space', ["char"]),
    "null": BuiltinRule('"null" space', []),
}

# TODO: support "uri", "email" string formats
STRING_FORMAT_RULES = {
    "date": BuiltinRule(
        '[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
        [],
    ),
    "time": BuiltinRule(
        '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
        [],
    ),
    "date-time": BuiltinRule('date "T" time', ["date", "time"]),
    "date-string": BuiltinRule('"\\"" date "\\"" space', ["date"]),
    "time-string": BuiltinRule('"\\"" time "\\"" space', ["time"]),
    "date-time-string": BuiltinRule('"\\"" date-time "\\"" space', ["date-time"]),
}

DOTALL = "[\\U00000000-\\U0010FFFF]"
DOT = "[^\\x0A\\x0D]"

RESERVED_NAMES = set(
    ["root", "dot", *PRIMITIVE_RULES.keys(), *STRING_FORMAT_RULES.keys()]
)


NON_LITERAL_SET = set("|.()[]{}*+?")
ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set("[]()|{}*+?")


class SchemaConverter:
    def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {
            "space": SPACE_RULE,
        }
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), literal
        )
        return f'"{escaped}"'

    def not_literal(
        self, literal: str, dotall: bool = True, maybe_escaped_underscores=False
    ) -> str:
        """
        not_literal('a') -> '[^a]'
        not_literal('abc') -> '([^a] | "a" ([^b] | "b" ([^c])?)?)?'
        """
        assert len(literal) > 0, "Empty literal not supported"

        def recurse(i: int):
            c = literal[i]
            if maybe_escaped_underscores and c == "_":
                yield f"[^{c}\\\\]"
                yield " | "
                yield f'"\\\\"? "{c}"'
            else:
                yield f"[^{c}]"
            if i < len(literal) - 1:
                yield " | "
                yield self._format_literal(c)
                yield " ("
                yield from recurse(i + 1)
                yield ")?"

        return "".join(("(", *recurse(0), ")"))

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while (
                f"{esc_name}{i}" in self._rules
                and self._rules[f"{esc_name}{i}"] != rule
            ):
                i += 1
            key = f"{esc_name}{i}"
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        """
        Resolves all $ref fields in the given schema, fetching any remote schemas,
        replacing $ref with absolute reference URL and populating self._refs with the
        respective referenced (sub)schema dictionaries.
        """

        def visit(n: dict):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get("$ref")
                if ref is not None and ref not in self._refs:
                    if ref.startswith("https://"):
                        assert (
                            self._allow_fetch
                        ), "Fetching remote schemas is not allowed (use --allow-fetch for force)"
                        import requests

                        frag_split = ref.split("#")
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(
                                requests.get(ref).json(), base_url
                            )
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == "":
                            return target
                    elif ref.startswith("#/"):
                        target = schema
                        ref = f"{url}{ref}"
                        n["$ref"] = ref
                    else:
                        raise ValueError(f"Unsupported ref {ref}")

                    for sel in ref.split("#")[-1].split("/")[1:]:
                        assert (
                            target is not None and sel in target
                        ), f"Error resolving ref {ref}: {sel} not in {target}"
                        target = target[sel]

                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)

            return n

        return visit(schema)

    def _generate_union_rule(self, name, alt_schemas):
        return " | ".join(
            (
                self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
                for i, alt_schema in enumerate(alt_schemas)
            )
        )

    def _visit_pattern(self, pattern, name):
        """
        Transforms a regular expression pattern into a GBNF rule.

        Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
        Output: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

        Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

        Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
        we define sub-rules to keep the output lean.
        """

        assert pattern.startswith("^") and pattern.endswith(
            "$"
        ), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        sub_rule_ids = {}

        i = 0
        length = len(pattern)

        def to_rule(s: Tuple[str, bool]) -> str:
            (txt, is_literal) = s
            return '"' + txt + '"' if is_literal else txt

        def transform() -> Tuple[str, bool]:
            """
            Parse a unit at index i (advancing it), and return its string representation + whether it's a literal.
            """
            nonlocal i
            nonlocal pattern
            nonlocal sub_rule_ids

            start = i
            # For each component of this sequence, store its string representation and whether it's a literal.
            # We only need a flat structure here to apply repetition operators to the last item, and
            # to merge literals at the and (we're parsing grouped ( sequences ) recursively and don't treat '|' specially
            # (GBNF's syntax is luckily very close to regular expressions!)
            seq: list[Tuple[str, bool]] = []

            def get_dot():
                if self._dotall:
                    rule = DOTALL
                else:
                    # Accept any character... except \n and \r line break chars (\x0A and \xOD)
                    rule = DOT
                return self._add_rule(f"dot", rule)

            def join_seq():
                nonlocal seq
                ret = []
                for is_literal, g in groupby(seq, lambda x: x[1]):
                    if is_literal:
                        ret.append(("".join(x[0] for x in g), True))
                    else:
                        ret.extend(g)
                if len(ret) == 1:
                    return ret[0]
                return (" ".join(to_rule(x) for x in seq), False)

            while i < length:
                c = pattern[i]
                if c == ".":
                    seq.append((get_dot(), False))
                    i += 1
                elif c == "(":
                    i += 1
                    if i < length:
                        assert (
                            pattern[i] != "?"
                        ), f'Unsupported pattern syntax "{pattern[i]}" at index {i} of /{pattern}/'
                    seq.append((f"({to_rule(transform())})", False))
                elif c == ")":
                    i += 1
                    assert (
                        start > 0 and pattern[start - 1] == "("
                    ), f"Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}"
                    return join_seq()
                elif c == "[":
                    square_brackets = c
                    i += 1
                    while i < length and pattern[i] != "]":
                        if pattern[i] == "\\":
                            square_brackets += pattern[i : i + 2]
                            i += 2
                        else:
                            square_brackets += pattern[i]
                            i += 1
                    assert (
                        i < length
                    ), f"Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}"
                    square_brackets += "]"
                    i += 1
                    seq.append((square_brackets, False))
                elif c == "|":
                    seq.append(("|", False))
                    i += 1
                elif c in ("*", "+", "?"):
                    seq[-1] = (to_rule(seq[-1]) + c, False)
                    i += 1
                elif c == "{":
                    curly_brackets = c
                    i += 1
                    while i < length and pattern[i] != "}":
                        curly_brackets += pattern[i]
                        i += 1
                    assert (
                        i < length
                    ), f"Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}"
                    curly_brackets += "}"
                    i += 1
                    nums = [s.strip() for s in curly_brackets[1:-1].split(",")]
                    min_times = 0
                    max_times = None
                    try:
                        if len(nums) == 1:
                            min_times = int(nums[0])
                            max_times = min_times
                        else:
                            assert len(nums) == 2
                            min_times = int(nums[0]) if nums[0] else 0
                            max_times = int(nums[1]) if nums[1] else None
                    except ValueError:
                        raise ValueError(
                            f"Invalid quantifier {curly_brackets} in /{pattern}/"
                        )

                    (sub, sub_is_literal) = seq[-1]

                    if not sub_is_literal:
                        id = sub_rule_ids.get(sub)
                        if id is None:
                            id = self._add_rule(f"{name}-{len(sub_rule_ids) + 1}", sub)
                            sub_rule_ids[sub] = id
                        sub = id

                    seq[-1] = (
                        _build_repetition(
                            f'"{sub}"' if sub_is_literal else sub,
                            min_times,
                            max_times,
                            item_rule_is_literal=sub_is_literal,
                        ),
                        False,
                    )
                else:
                    literal = ""
                    while i < length:
                        if pattern[i] == "\\" and i < length - 1:
                            next = pattern[i + 1]
                            if next in ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                                i += 1
                                literal += pattern[i]
                                i += 1
                            else:
                                literal += pattern[i : i + 2]
                                i += 2
                        elif pattern[i] == '"' and not self._raw_pattern:
                            literal += '\\"'
                            i += 1
                        elif pattern[i] not in NON_LITERAL_SET and (
                            i == length - 1
                            or literal == ""
                            or pattern[i + 1] == "."
                            or pattern[i + 1] not in NON_LITERAL_SET
                        ):
                            literal += pattern[i]
                            i += 1
                        else:
                            break
                    if literal:
                        seq.append((literal, True))

            return join_seq()

        return self._add_rule(
            name,
            (
                to_rule(transform())
                if self._raw_pattern
                else '"\\"" ' + to_rule(transform()) + ' "\\"" space'
            ),
        )

    def _resolve_ref(self, ref):
        ref_name = ref.split("/")[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name

    def _generate_constant_rule(self, value):
        return self._format_literal(json.dumps(value))

    def visit(self, schema, name):
        schema_type = schema.get("type")
        schema_format = schema.get("format")
        rule_name = name + "-" if name in RESERVED_NAMES else name or "root"

        if (ref := schema.get("$ref")) is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        elif "oneOf" in schema or "anyOf" in schema:
            return self._add_rule(
                rule_name,
                self._generate_union_rule(name, schema.get("oneOf") or schema["anyOf"]),
            )

        elif isinstance(schema_type, list):
            return self._add_rule(
                rule_name,
                self._generate_union_rule(name, [{"type": t} for t in schema_type]),
            )

        elif "const" in schema:
            return self._add_rule(
                rule_name, self._generate_constant_rule(schema["const"])
            )

        elif "enum" in schema:
            rule = " | ".join((self._generate_constant_rule(v) for v in schema["enum"]))
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, "object") and (
            "properties" in schema
            or (
                "additionalProperties" in schema
                and schema["additionalProperties"] is not True
            )
        ):
            required = set(schema.get("required", []))
            properties = list(schema.get("properties", {}).items())
            return self._add_rule(
                rule_name,
                self._build_object_rule(
                    properties, required, name, schema.get("additionalProperties")
                ),
            )

        elif schema_type in (None, "object") and "allOf" in schema:
            required = set()
            properties = []
            hybrid_name = name

            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get("$ref")) is not None:
                    comp_schema = self._refs[ref]

                if "properties" in comp_schema:
                    for prop_name, prop_schema in comp_schema["properties"].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema["allOf"]:
                if "anyOf" in t:
                    for tt in t["anyOf"]:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._add_rule(
                rule_name,
                self._build_object_rule(
                    properties, required, hybrid_name, additional_properties=[]
                ),
            )

        elif schema_type in (None, "array") and (
            "items" in schema or "prefixItems" in schema
        ):
            items = schema.get("items") or schema["prefixItems"]
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space '
                    + ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
                        for i, item in enumerate(items)
                    )
                    + ' "]" space',
                )
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                return self._add_rule(
                    rule_name,
                    '"[" space '
                    + _build_repetition(
                        item_rule_name, min_items, max_items, separator_rule='"," space'
                    )
                    + ' "]" space',
                )

        elif schema_type in (None, "string") and "pattern" in schema:
            return self._visit_pattern(schema["pattern"], rule_name)

        elif schema_type in (None, "string") and re.match(
            r"^uuid[1-5]?$", schema_format or ""
        ):
            return self._add_primitive(
                "root" if rule_name == "root" else schema_format,
                PRIMITIVE_RULES["uuid"],
            )

        elif (
            schema_type in (None, "string")
            and f"{schema_format}-string" in STRING_FORMAT_RULES
        ):
            prim_name = f"{schema_format}-string"
            return self._add_rule(
                rule_name,
                self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]),
            )

        elif schema_type == "string" and (
            "minLength" in schema or "maxLength" in schema
        ):
            char_rule = self._add_primitive("char", PRIMITIVE_RULES["char"])
            min_len = schema.get("minLength", 0)
            max_len = schema.get("maxLength")

            return self._add_rule(
                rule_name,
                r'"\"" '
                + _build_repetition(char_rule, min_len, max_len)
                + r' "\"" space',
            )

        elif (schema_type == "object") or (len(schema) == 0):
            return self._add_rule(
                rule_name, self._add_primitive("object", PRIMITIVE_RULES["object"])
            )

        else:
            assert schema_type in PRIMITIVE_RULES, f"Unrecognized schema: {schema}"
            # TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_primitive(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

    def _add_primitive(self, name: str, rule: BuiltinRule):
        n = self._add_rule(name, rule.content)

        for dep in rule.deps:
            dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
            assert dep_rule, f"Rule {dep} not known"
            if dep not in self._rules:
                self._add_primitive(dep, dep_rule)
        return n

    def _build_object_rule(
        self,
        properties: List[Tuple[str, Any]],
        required: Set[str],
        name: str,
        additional_properties: Union[bool, Any],
    ):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [
            kv[0]
            for _, kv in sorted(
                enumerate(properties),
                key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]),
            )
        ]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(
                prop_schema, f'{name}{"-" if name else ""}{prop_name}'
            )
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                rf'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}',
            )
        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]

        if additional_properties == True or isinstance(additional_properties, dict):
            sub_name = f'{name}{"-" if name else ""}additional'
            value_rule = self.visit(
                {} if additional_properties == True else additional_properties,
                f"{sub_name}-value",
            )
            prop_kv_rule_names["*"] = self._add_rule(
                f"{sub_name}-kv",
                self._add_primitive("string", PRIMITIVE_RULES["string"])
                + f' ":" space {value_rule}',
            )
            optional_props.append("*")

        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

        if optional_props:
            rule += " ("
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                if k == "*":
                    res = self._add_rule(
                        f'{name}{"-" if name else ""}additional-kvs',
                        f'{kv_rule_name} ( "," space ' + kv_rule_name + " )*",
                    )
                elif first_is_optional:
                    res = f'( "," space {kv_rule_name} )?'
                else:
                    res = kv_rule_name
                if len(rest) > 0:
                    res += " " + self._add_rule(
                        f'{name}{"-" if name else ""}{k}-rest',
                        get_recursive_refs(rest, first_is_optional=True),
                    )
                return res

            rule += " | ".join(
                get_recursive_refs(optional_props[i:], first_is_optional=False)
                for i in range(len(optional_props))
            )
            if required_props:
                rule += " )"
            rule += " )?"

        rule += ' "}" space'

        return rule

    def format_grammar(self):
        return "\n".join(
            f"{name} ::= {rule}"
            for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
        )


def json_schema_to_gbnf(schema: str, prop_order: Optional[List[str]] = None):
    prop_order = prop_order or []
    schema = json.loads(schema)
    prop_order = {name: idx for idx, name in enumerate(prop_order)}
    converter = SchemaConverter(
        prop_order=prop_order, allow_fetch=False, dotall=False, raw_pattern=False
    )
    schema = converter.resolve_refs(schema, "stdin")
    converter.visit(schema, "")
    return converter.format_grammar()
