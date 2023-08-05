"""C++ implementation of the llama grammar parser."""
# flake8: noqa
import argparse
from pathlib import Path
import sys
from ctypes import Array, c_int, c_size_t, c_uint32, cast
from enum import Enum
from itertools import islice
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    OrderedDict,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)

import llama_cpp

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
size_t = uint8_t = uint32_t = int
static_cast_uint8_t = ord


class Sentinel:
    pass


class const_char_p:
    """C++ implementation of const char*."""

    def __init__(self, value: Union[str, "const_char_p"]):
        if isinstance(value, const_char_p):
            # We're copying an existing const_char_p
            self.value = value.value
            self.pos = value.pos
            return

        # We're creating a new const_char_p
        self.value = value
        self.pos = 0

    def __str__(self) -> str:
        return self.value[self.pos :]

    def __add__(self, increment: int) -> "const_char_p":
        # To avoid side effects, we create a new const_char_p object
        new = self.__class__(self.value)
        new.pos = self.pos + increment
        return new

    def __sub__(self, decrement: int) -> "const_char_p":
        # To avoid side effects, we create a new const_char_p object
        new = self.__class__(self.value)
        new.pos = self.pos - decrement
        return new

    def __lt__(self, other: "const_char_p") -> bool:
        return self.pos < other.pos and self.value == other.value

    def __gt__(self, other: "const_char_p") -> bool:
        return self.pos > other.pos and self.value == other.value

    def __eq__(self, other: "const_char_p") -> bool:
        return self.pos == other.pos and self.value == other.value

    def add(self, other: "const_char_p") -> int:
        if self.value != other.value:
            raise ValueError("Can't add pointers to different strings")
        return self.pos + other.pos

    def sub(self, other: "const_char_p") -> int:
        if self.value != other.value:
            raise ValueError("Can't subtract pointers to different strings")
        return self.pos - other.pos

    def plus_plus(self) -> None:
        self.pos += 1

    def minus_minus(self) -> None:
        self.pos -= 1

    @property
    def derefer(self) -> Optional[str]:
        if self.pos >= len(self.value):
            # We've reached the end of the string
            return None

        return self.value[self.pos]


class std__vector(Generic[T], List[T]):
    """C++ implementation of std::vector."""

    class iterator:
        def __init__(self, vector: "std__vector[T]", index: int):
            self._vector = vector
            self._index = index
            self._version = vector._version

        def _check_version(self):
            if self._version != self._vector._version:
                raise RuntimeError("Iterator used after vector was modified.")

        def __iter__(self):
            return self

        def __next__(self) -> T:
            self._check_version()
            if self._index >= self._vector.size():
                raise StopIteration
            value = self._vector[self._index]
            self._index += 1
            return value

        def __add__(self, value: int) -> "std__vector[T].iterator":
            return self.__class__(self._vector, self._index + value)

        def __sub__(self, value: int) -> "std__vector[T].iterator":
            return self.__class__(self._vector, self._index - value)

    def __init__(self):
        self._version = 0

    def modify(self):
        # This is a bit of a hack to make sure iterators are invalidated
        self._version += 1

    def push_back(self, value: T) -> None:
        self.modify()
        self.append(value)

    def pop_back(self) -> None:
        self.modify()
        if not self.empty():
            self.pop()

    def back(self) -> T:
        return self[-1]

    def size(self) -> int:
        return len(self)

    # def clear(self) -> None:
    #     super().clear()

    def empty(self) -> bool:
        return self.size() == 0

    def data(self) -> "std__vector[T]":
        return self

    def resize(
        self,
        new_size: int,
        fill_value_factory: Optional[Callable[[], T]] = None,
    ) -> None:
        if new_size > self.size():
            if fill_value_factory is None:
                raise ValueError(
                    "A fill value factory function must be provided."
                )
            self.reserve(new_size, fill_value_factory)
        elif new_size < self.size():
            self[:] = self[:new_size]

    def reserve(
        self, capacity: int, fill_value_factory: Callable[[], T]
    ) -> None:
        if capacity > self.size():
            fill_value = fill_value_factory()
            self.extend([fill_value] * (capacity - self.size()))

    def front(self) -> T:
        if not self.empty():
            return self[0]
        else:
            raise IndexError("Vector is empty.")

    def assign(self, count: int, value: T) -> None:
        self.clear()
        self.extend([value] * count)

    def insert(
        self,
        pos: "std__vector[T].iterator",
        first: "std__vector[T].iterator",
        last: "std__vector[T].iterator",
    ) -> None:
        self[pos._index : pos._index] = list(
            islice(first._vector, first._index, last._index)
        )

    def begin(self) -> "std__vector[T].iterator":
        return self.iterator(self, 0)

    def end(self) -> "std__vector[T].iterator":
        return self.iterator(self, self.size())


class std__map(Generic[T, U], OrderedDict[T, U]):
    """C++ implementation of std::map."""

    class iterator(Generic[V, W]):
        def __init__(self, _map: "std__map[T, U]", key: Union[T, Sentinel]):
            self._map = _map
            self.iter = iter(_map)
            self.key = key
            self._advance()

        def _sanitize_key(self) -> T:
            if isinstance(self.key, Sentinel):
                raise StopIteration
            return self.key

        def _advance(self) -> None:
            try:
                while next(self.iter) != self.key:
                    pass
            except StopIteration:
                self.key = Sentinel()

        def __next__(self) -> Tuple[T, U]:
            key = self._sanitize_key()
            if key in self._map:
                value = self._map[key]
                self._advance()
                return key, value
            else:
                raise StopIteration

        def get(self) -> Tuple[T, U]:
            key = self._sanitize_key()
            return key, self._map[key]

        @property
        def first(self) -> T:
            return self._sanitize_key()

        @property
        def second(self) -> U:
            return self._map[self._sanitize_key()]

    def insert(
        self, key: T, value: U
    ) -> Tuple["std__map[T, U].iterator[T, U]", bool]:
        if key in self:
            return self.iterator(self, key), False
        else:
            self[key] = value
            return self.iterator(self, key), True

    def find(self, key: T) -> "std__map[T, U].iterator[T, U]":
        if key in self:
            return self.iterator(self, key)
        else:
            return self.end()

    def at(self, key: T) -> U:
        if key in self:
            return self[key]
        else:
            raise KeyError("The provided key is not found in the map.")

    def erase(self, iterator: "std__map[T, U].iterator[T, U]") -> None:
        key = iterator.first
        if key in self:
            del self[key]

    def size(self) -> int:
        return len(self)

    def empty(self) -> bool:
        return self.size() == 0

    def lower_bound(self, key: T) -> "std__map[T, U].iterator[T, U]":
        try:
            keys = sorted(list(self.keys()))  # type: ignore
            for k in keys:
                if k >= key:
                    return self.iterator(self, k)
            raise ValueError(
                "No key found that is not less than the input key"
            )
        except TypeError:
            raise TypeError("Keys of type T cannot be sorted.")

    def begin(self) -> "std__map[T, U].iterator[T, U]":
        return self.iterator(self, next(iter(self)))

    def end(self) -> "std__map[T, U].iterator[T, U]":
        return self.iterator(self, Sentinel())


class std__string(str):
    def __new__(cls, ptr: const_char_p, length: Optional[int] = None):
        if length is not None:
            return super().__new__(cls, str(ptr)[:length])
        return super().__new__(cls, str(ptr))


# // grammar element type
# enum llama_gretype {
#     // end of rule definition
#     LLAMA_GRETYPE_END            = 0,

#     // start of alternate definition for rule
#     LLAMA_GRETYPE_ALT            = 1,

#     // non-terminal element: reference to rule
#     LLAMA_GRETYPE_RULE_REF       = 2,

#     // terminal element: character (code point)
#     LLAMA_GRETYPE_CHAR           = 3,

#     // inverse char(s) ([^a], [^a-b] [^abc])
#     LLAMA_GRETYPE_CHAR_NOT       = 4,

#     // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
#     // be an inclusive range ([a-z])
#     LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,


#     // modifies a preceding LLAMA_GRETYPE_CHAR or
#     // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
#     LLAMA_GRETYPE_CHAR_ALT       = 6,
# };
class llama_gretype(Enum):
    """grammar element type"""

    LLAMA_GRETYPE_END = 0  # end of rule definition
    LLAMA_GRETYPE_ALT = 1  # start of alternate definition for rule
    LLAMA_GRETYPE_RULE_REF = 2  # non-terminal element: reference to rule
    LLAMA_GRETYPE_CHAR = 3  # terminal element: character (code point)
    LLAMA_GRETYPE_CHAR_NOT = 4  # inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5  # modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_ALT = 6  # modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])


# typedef struct llama_grammar_element {
#     enum llama_gretype type;
#     uint32_t           value; // Unicode code point or rule ID
# } llama_grammar_element;


# class llama_grammar_element(Structure):
#     _fields_ = [
#         ("type", c_int),
#         ("value", c_uint32),
#     ]


class llama_grammar_element:
    def __init__(self, type: llama_gretype, value: uint32_t):
        self.type = type
        self.value = value  # Unicode code point or rule ID

    def __repr__(self):  # debug
        return f"llama_grammar_element({self.type}, {self.value})"


# struct parse_state {
#     std::map<std::string, uint32_t>                 symbol_ids;
#     std::vector<std::vector<llama_grammar_element>> rules;
#     std::vector<const llama_grammar_element *> c_rules();
# };
class parse_state:
    def __init__(self):
        self.symbol_ids: std__map[str, uint32_t] = std__map()
        self.rules: std__vector[
            std__vector[llama_grammar_element]
        ] = std__vector()

    # std::vector<const llama_grammar_element *> parse_state::c_rules() {
    #     std::vector<const llama_grammar_element *> ret;
    #     for (const auto & rule : rules) {
    #         ret.push_back(rule.data());
    #     }
    #     return ret;
    # }
    def c_rules(self) -> std__vector[std__vector[llama_grammar_element]]:
        ret = (
            std__vector()
        )  # type: std__vector[std__vector[llama_grammar_element]]
        for rule in self.rules:
            ret.push_back(rule.data())
        return ret


# struct llama_grammar {
#     const std::vector<std::vector<llama_grammar_element>>   rules;
#     std::vector<std::vector<const llama_grammar_element *>> stacks;
# };
class llama_grammar:
    def __init__(
        self,
        rules: std__vector[std__vector[llama_grammar_element]],
        stacks: std__vector[std__vector[llama_grammar_element]],
    ):
        self.rules = rules
        self.stacks = stacks


# uint32_t get_symbol_id(parse_state & state, const char * src, size_t len) {
#     uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
#     auto result = state.symbol_ids.insert(std::make_pair(std::string(src, len), next_id));
#     return result.first->second;
# }
def get_symbol_id(state: parse_state, src: const_char_p, len: size_t) -> int:
    next_id = uint32_t(state.symbol_ids.size())  # type: uint32_t
    result = state.symbol_ids.insert(str(std__string(src, len)), next_id)
    return result[0].second  # type: ignore


# uint32_t generate_symbol_id(parse_state & state, const std::string & base_name) {
#     uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
#     state.symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
#     return next_id;
# }
def generate_symbol_id(state: parse_state, base_name: str) -> uint32_t:
    next_id = state.symbol_ids.size()  # type: uint32_t
    state.symbol_ids[base_name + "_" + str(next_id)] = next_id
    return next_id


# void add_rule(
#         parse_state & state,
#         uint32_t      rule_id,
#         const std::vector<llama_grammar_element> & rule) {
#     if (state.rules.size() <= rule_id) {
#         state.rules.resize(rule_id + 1);
#     }
#     state.rules[rule_id] = rule;
# }
def add_rule(
    state: parse_state,
    rule_id: uint32_t,
    rule: std__vector[llama_grammar_element],
) -> None:
    if state.rules.size() <= rule_id:
        state.rules.resize(
            rule_id + 1, fill_value_factory=std__vector[llama_grammar_element]
        )
    state.rules[rule_id] = rule


# std::pair<uint32_t, const char *> decode_utf8(const char * src) {
#     static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
#     uint8_t  first_byte = static_cast<uint8_t>(*src);
#     uint8_t  highbits   = first_byte >> 4;
#     int      len        = lookup[highbits];
#     uint8_t  mask       = (1 << (8 - len)) - 1;
#     uint32_t value      = first_byte & mask;
#     const char * end    = src + len; // may overrun!
#     const char * pos    = src + 1;
#     for ( ; pos < end && *pos; pos++) {
#         value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
#     }
#     return std::make_pair(value, pos);
# }
def decode_utf8(src: const_char_p) -> Tuple[uint32_t, const_char_p]:
    """Decodes a UTF-8 character from the source string."""
    lookup = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)
    first_byte = static_cast_uint8_t(src.derefer or "")  # type: uint8_t
    highbits = first_byte >> 4  # type: uint8_t
    len = lookup[highbits]  # type: int
    mask = (1 << (8 - len)) - 1  # type: uint8_t
    value = first_byte & mask  # type: uint32_t
    end = src + len  # type: const_char_p # may overrun!
    pos = src + 1  # type: const_char_p
    while pos < end and pos.derefer:
        value = (value << 6) + (static_cast_uint8_t(src.derefer or "") & 0x3F)
        pos.plus_plus()
    return value, pos


# bool is_word_char(char c) {
#     return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || ('0' <= c && c <= '9');
# }
def is_word_char(c: str) -> bool:
    return (
        ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "-" or ("0" <= c <= "9")
    )


# std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
#     const char * pos   = src;
#     const char * end   = src + size;
#     uint32_t     value = 0;
#     for ( ; pos < end && *pos; pos++) {
#         value <<= 4;
#         char c = *pos;
#         if ('a' <= c && c <= 'f') {
#             value += c - 'a' + 10;
#         } else if ('A' <= c && c <= 'F') {
#             value += c - 'A' + 10;
#         } else if ('0' <= c && c <= '9') {
#             value += c - '0';
#         } else {
#             break;
#         }
#     }
#     if (pos != end) {
#         throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
#     }
#     return std::make_pair(value, pos);
# }
def parse_hex(src: const_char_p, size: int) -> Tuple[uint32_t, const_char_p]:
    pos = const_char_p(src)  # type: const_char_p
    end = src + size  # type: const_char_p
    value = 0  # type: uint32_t
    while pos < end and pos.derefer:
        value <<= 4
        c = pos.derefer  # type: str
        if "a" <= c <= "f":
            value += static_cast_uint8_t(c) - static_cast_uint8_t("a") + 10
        elif "A" <= c <= "F":
            value += static_cast_uint8_t(c) - static_cast_uint8_t("A") + 10
        elif "0" <= c <= "9":
            value += static_cast_uint8_t(c) - static_cast_uint8_t("0")
        else:
            break
        pos.plus_plus()
    if pos != end:
        raise RuntimeError(
            "expecting " + str(size) + " hex chars at " + str(src)
        )
    return (value, pos)


# std::pair<uint32_t, const char *> parse_char(const char * src) {
#     if (*src == '\\') {
#         switch (src[1]) {
#             case 'x': return parse_hex(src + 2, 2);
#             case 'u': return parse_hex(src + 2, 4);
#             case 'U': return parse_hex(src + 2, 8);
#             case 't': return std::make_pair('\t', src + 2);
#             case 'r': return std::make_pair('\r', src + 2);
#             case 'n': return std::make_pair('\n', src + 2);
#             case '\\':
#             case '"':
#             case '[':
#             case ']':
#                 return std::make_pair(src[1], src + 2);
#             default:
#                 throw std::runtime_error(std::string("unknown escape at ") + src);
#         }
#     } else if (*src) {
#         return decode_utf8(src);
#     }
#     throw std::runtime_error("unexpected end of input");
# }
def parse_char(src: const_char_p) -> Tuple[uint32_t, const_char_p]:
    if src.derefer == "\\":
        switch = (src + 1).derefer  # type: Optional[str]
        if switch == "x":
            return parse_hex(src + 2, 2)
        elif switch == "u":
            return parse_hex(src + 2, 4)
        elif switch == "U":
            return parse_hex(src + 2, 8)
        elif switch == "t":
            return (static_cast_uint8_t("\t"), src + 2)  # implicit cast
        elif switch == "r":
            return (static_cast_uint8_t("\r"), src + 2)  # implicit cast
        elif switch == "n":
            return (static_cast_uint8_t("\n"), src + 2)  # implicit cast
        elif switch in ("\\", '"', "[", "]"):
            return (static_cast_uint8_t(switch), src + 2)  # implicit cast
        else:
            raise RuntimeError("unknown escape at " + str(src))
    elif src.derefer:
        return decode_utf8(src)
    else:
        raise RuntimeError("unexpected end of input")


# const char * parse_name(const char * src) {
#     const char * pos = src;
#     while (is_word_char(*pos)) {
#         pos++;
#     }
#     if (pos == src) {
#         throw std::runtime_error(std::string("expecting name at ") + src);
#     }
#     return pos;
# }
def parse_name(src: const_char_p) -> const_char_p:
    pos = const_char_p(src)  # type: const_char_p
    while is_word_char(pos.derefer or ""):
        pos.plus_plus()
    if pos == src:
        raise RuntimeError("expecting name at " + str(src))
    return pos


# const char * parse_space(const char * src, bool newline_ok) {
#     const char * pos = src;
#     while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
#             (newline_ok && (*pos == '\r' || *pos == '\n'))) {
#         if (*pos == '#') {
#             while (*pos && *pos != '\r' && *pos != '\n') {
#                 pos++;
#             }
#         } else {
#             pos++;
#         }
#     }
#     return pos;
# }
def parse_space(src: const_char_p, newline_ok: bool) -> const_char_p:
    # Using a copy of `src` to avoid side effects
    pos = const_char_p(src)

    while pos.derefer in (" ", "\t", "#") or (
        newline_ok and pos.derefer in ("\r", "\n")
    ):
        if pos.derefer == "#":
            while pos.derefer is not None and pos.derefer not in ("\r", "\n"):
                pos.plus_plus()
        else:
            pos.plus_plus()

    return pos


# const char * parse_sequence(
#         parse_state                        & state,
#         const char                         * src,
#         const std::string                  & rule_name,
#         std::vector<llama_grammar_element> & out_elements,
#         bool                                 is_nested) {
def parse_sequence(
    state: parse_state,
    src: const_char_p,
    rule_name: str,
    out_elements: std__vector[llama_grammar_element],
    is_nested: bool,
) -> const_char_p:
    # size_t last_sym_start = out_elements.size();
    # const char * pos = src;
    last_sym_start = out_elements.size()  # type: size_t
    pos = const_char_p(src)  # type: const_char_p
    # while (*pos) {
    while pos.derefer:
        # if (*pos == '"') { // literal string
        #     pos++;
        #     last_sym_start = out_elements.size();
        #     while (*pos != '"') {
        #         auto char_pair = parse_char(pos);
        #                 pos       = char_pair.second;
        #         out_elements.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
        #     }
        #     pos = parse_space(pos + 1, is_nested);
        if pos.derefer == '"':  # literal string
            pos.plus_plus()
            last_sym_start = out_elements.size()
            while pos.derefer != '"':
                char_pair = parse_char(
                    pos
                )  # type: Tuple[uint32_t, const_char_p]
                pos = char_pair[1]
                out_elements.push_back(
                    llama_grammar_element(
                        llama_gretype.LLAMA_GRETYPE_CHAR, char_pair[0]
                    )
                )
            pos = parse_space(pos + 1, is_nested)
        # } else if (*pos == '[') { // char range(s)
        #     pos++;
        #     enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
        elif pos.derefer == "[":  # char range(s)
            pos.plus_plus()
            start_type = (
                llama_gretype.LLAMA_GRETYPE_CHAR
            )  # type: llama_gretype
            # if (*pos == '^') {
            #     pos++;
            #     start_type = LLAMA_GRETYPE_CHAR_NOT;
            # }
            # last_sym_start = out_elements.size();
            if pos.derefer == "^":
                pos.plus_plus()
                start_type = llama_gretype.LLAMA_GRETYPE_CHAR_NOT
            last_sym_start = out_elements.size()
            # while (*pos != ']') {
            #     auto char_pair = parse_char(pos);
            #             pos       = char_pair.second;
            #     enum llama_gretype type = last_sym_start < out_elements.size()
            #         ? LLAMA_GRETYPE_CHAR_ALT
            #         : start_type;
            #     out_elements.push_back({type, char_pair.first});
            while pos.derefer != "]":
                char_pair = parse_char(
                    pos
                )  # type: Tuple[uint32_t, const_char_p]
                pos = char_pair[1]
                type = (
                    llama_gretype.LLAMA_GRETYPE_CHAR_ALT
                    if last_sym_start < out_elements.size()
                    else start_type
                )  # type: llama_gretype
                out_elements.push_back(
                    llama_grammar_element(type, char_pair[0])
                )
                #     if (pos[0] == '-' && pos[1] != ']') {
                #         auto endchar_pair = parse_char(pos + 1);
                #                 pos          = endchar_pair.second;
                #         out_elements.push_back({LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                #     }
                # }
                if pos.derefer == "-" and (pos + 1).derefer != "]":
                    endchar_pair = parse_char(
                        pos + 1
                    )  # type: Tuple[uint32_t, const_char_p]
                    pos = endchar_pair[1]
                    out_elements.push_back(
                        llama_grammar_element(
                            llama_gretype.LLAMA_GRETYPE_CHAR_RNG_UPPER,
                            endchar_pair[0],
                        )
                    )
            # pos = parse_space(pos + 1, is_nested);
            pos = parse_space(pos + 1, is_nested)
        # } else if (is_word_char(*pos)) { // rule reference
        #     const char * name_end    = parse_name(pos);
        #     uint32_t     ref_rule_id = get_symbol_id(state, pos, name_end - pos);
        #     pos = parse_space(name_end, is_nested);
        #     last_sym_start = out_elements.size();
        #     out_elements.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
        elif is_word_char(pos.derefer):  # rule reference
            name_end = parse_name(pos)  # type: const_char_p
            ref_rule_id = get_symbol_id(
                state, pos, name_end.sub(pos)
            )  # type: uint32_t
            pos = parse_space(name_end, is_nested)
            last_sym_start = out_elements.size()
            out_elements.push_back(
                llama_grammar_element(
                    llama_gretype.LLAMA_GRETYPE_RULE_REF, ref_rule_id
                )
            )
        # } else if (*pos == '(') { // grouping
        #     // parse nested alternates into synthesized rule
        #     pos = parse_space(pos + 1, true);
        #     uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
        #     pos = parse_alternates(state, pos, rule_name, sub_rule_id, true);
        #     last_sym_start = out_elements.size();
        #     // output reference to synthesized rule
        #     out_elements.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
        #     if (*pos != ')') {
        #         throw std::runtime_error(std::string("expecting ')' at ") + pos);
        #     }
        #     pos = parse_space(pos + 1, is_nested);
        elif pos.derefer == "(":  # grouping
            pos = parse_space(pos + 1, True)
            sub_rule_id = generate_symbol_id(
                state, rule_name
            )  # type: uint32_t
            pos = parse_alternates(state, pos, rule_name, sub_rule_id, True)
            last_sym_start = out_elements.size()
            out_elements.push_back(
                llama_grammar_element(
                    llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id
                )
            )
            if pos.derefer != ")":
                raise RuntimeError("expecting ')' at " + str(pos))
            pos = parse_space(pos + 1, is_nested)
        # } else if (*pos == '*' || *pos == '+' || *pos == '?') { // repetition operator
        #     if (last_sym_start == out_elements.size()) {
        #         throw std::runtime_error(std::string("expecting preceeding item to */+/? at ") + pos);
        #     }
        elif pos.derefer in ("*", "+", "?"):  # repetition operator
            if last_sym_start == out_elements.size():
                raise RuntimeError(
                    "expecting preceding item to */+/? at " + str(pos)
                )
            # // apply transformation to previous symbol (last_sym_start to end) according to
            # // rewrite rules:
            # // S* --> S' ::= S S' |
            # // S+ --> S' ::= S S' | S
            # // S? --> S' ::= S |
            # uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
            # std::vector<llama_grammar_element> sub_rule;
            # // add preceding symbol to generated rule
            # sub_rule.insert(
            #     sub_rule.end(), out_elements.begin() + last_sym_start, out_elements.end());
            sub_rule_id = generate_symbol_id(
                state, rule_name
            )  # type: uint32_t
            sub_rule = std__vector[llama_grammar_element]()
            sub_rule.insert(
                sub_rule.end(),
                out_elements.begin() + last_sym_start,
                out_elements.end(),
            )
            # if (*pos == '*' || *pos == '+') {
            #     // cause generated rule to recurse
            #     sub_rule.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
            # }
            # // mark start of alternate def
            # sub_rule.push_back({LLAMA_GRETYPE_ALT, 0});
            if pos.derefer in ("*", "+"):
                sub_rule.push_back(
                    llama_grammar_element(
                        llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id
                    )
                )
            sub_rule.push_back(
                llama_grammar_element(llama_gretype.LLAMA_GRETYPE_ALT, 0)
            )
            # if (*pos == '+') {
            #     // add preceding symbol as alternate only for '+' (otherwise empty)
            #     sub_rule.insert(
            #         sub_rule.end(), out_elements.begin() + last_sym_start, out_elements.end());
            # }
            # sub_rule.push_back({LLAMA_GRETYPE_END, 0});
            # add_rule(state, sub_rule_id, sub_rule);
            # // in original rule, replace previous symbol with reference to generated rule
            # out_elements.resize(last_sym_start);
            # out_elements.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
            # pos = parse_space(pos + 1, is_nested);
            if pos.derefer == "+":
                sub_rule.insert(
                    sub_rule.end(),
                    out_elements.begin() + last_sym_start,
                    out_elements.end(),
                )
            sub_rule.push_back(
                llama_grammar_element(llama_gretype.LLAMA_GRETYPE_END, 0)
            )
            add_rule(state, sub_rule_id, sub_rule)
            out_elements.resize(last_sym_start)
            out_elements.push_back(
                llama_grammar_element(
                    llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id
                )
            )
            pos = parse_space(pos + 1, is_nested)
        # } else {
        #     break;
        # }
        else:
            break
    #     }
    #     return pos;
    # }
    return pos


# const char * parse_alternates(
#         parse_state       & state,
#         const char        * src,
#         const std::string & rule_name,
#         uint32_t            rule_id,
#         bool                is_nested) {
#     std::vector<llama_grammar_element> rule;
#     const char * pos = parse_sequence(state, src, rule_name, rule, is_nested);
#     while (*pos == '|') {
#         rule.push_back({LLAMA_GRETYPE_ALT, 0});
#         pos = parse_space(pos + 1, true);
#         pos = parse_sequence(state, pos, rule_name, rule, is_nested);
#     }
#     rule.push_back({LLAMA_GRETYPE_END, 0});
#     add_rule(state, rule_id, rule);
#     return pos;
# }
def parse_alternates(
    state: parse_state,
    src: const_char_p,
    rule_name: str,
    rule_id: uint32_t,
    is_nested: bool,
) -> const_char_p:
    rule = std__vector()  # type: std__vector[llama_grammar_element]
    pos = parse_sequence(
        state, src, rule_name, rule, is_nested
    )  # type: const_char_p
    while pos.derefer == "|":
        rule.push_back(
            llama_grammar_element(llama_gretype.LLAMA_GRETYPE_ALT, 0)
        )
        pos = parse_space(pos + 1, True)
        pos = parse_sequence(state, pos, rule_name, rule, is_nested)
    rule.push_back(llama_grammar_element(llama_gretype.LLAMA_GRETYPE_END, 0))
    add_rule(state, rule_id, rule)
    return pos


# const char * parse_rule(parse_state & state, const char * src) {
#     const char * name_end = parse_name(src);
#     const char * pos      = parse_space(name_end, false);
#     size_t       name_len = name_end - src;
#     uint32_t     rule_id  = get_symbol_id(state, src, name_len);
#     const std::string name(src, name_len);

#     if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
#         throw std::runtime_error(std::string("expecting ::= at ") + pos);
#     }
#     pos = parse_space(pos + 3, true);

#     pos = parse_alternates(state, pos, name, rule_id, false);


#     if (*pos == '\r') {
#         pos += pos[1] == '\n' ? 2 : 1;
#     } else if (*pos == '\n') {
#         pos++;
#     } else if (*pos) {
#         throw std::runtime_error(std::string("expecting newline or end at ") + pos);
#     }
#     return parse_space(pos, true);
# }
def parse_rule(state: parse_state, src: const_char_p) -> const_char_p:
    name_end = parse_name(src)  # type: const_char_p
    pos = parse_space(name_end, False)  # type: const_char_p
    name_len = name_end.sub(src)  # type: size_t
    rule_id = get_symbol_id(state, src, name_len)  # type: uint32_t
    name = std__string(src, name_len)  # type: std__string

    if not (
        pos.derefer == ":"
        and (pos + 1).derefer == ":"
        and (pos + 2).derefer == "="
    ):
        raise RuntimeError("expecting ::= at " + str(pos))

    pos = parse_space(pos + 3, True)  # type: const_char_p
    pos = parse_alternates(
        state, pos, name, rule_id, False
    )  # type: const_char_p

    if pos.derefer == "\r":
        pos += 2 if (pos + 1).derefer == "\n" else 1
    elif pos.derefer == "\n":
        pos.plus_plus()
    elif pos.derefer:
        raise RuntimeError("expecting newline or end at " + str(pos))
    return parse_space(pos, True)


# parse_state parse(const char * src) {
#     try {
#         parse_state state;
#         const char * pos = parse_space(src, true);
#         while (*pos) {
#             pos = parse_rule(state, pos);
#         }
#         return state;
#     } catch (const std::exception & err) {
#         fprintf(stderr, "%s: error parsing grammar: %s\n", __func__, err.what());
#         return parse_state();
#     }
# }
def parse(src: const_char_p) -> parse_state:
    try:
        state = parse_state()  # type: parse_state
        pos = parse_space(src, True)  # type: const_char_p
        while pos.derefer:
            pos = parse_rule(state, pos)
        return state
    except Exception as err:
        print(f"{parse.__name__}: error parsing grammar: {err}")
        return parse_state()


# void print_grammar_char(FILE * file, uint32_t c) {
#     if (0x20 <= c && c <= 0x7f) {
#         fprintf(file, "%c", static_cast<char>(c));
#     } else {
#         // cop out of encoding UTF-8
#         fprintf(file, "<U+%04X>", c);
#     }
# }
def print_grammar_char(file: TextIO, c: uint32_t) -> None:
    if 0x20 <= c and c <= 0x7F:
        file.write(chr(c))
    else:
        # cop out of encoding UTF-8
        file.write(f"<U+{c:04X}>")


# bool is_char_element(llama_grammar_element elem) {
#     switch (elem.type) {
#         case LLAMA_GRETYPE_CHAR:           return true;
#         case LLAMA_GRETYPE_CHAR_NOT:       return true;
#         case LLAMA_GRETYPE_CHAR_ALT:       return true;
#         case LLAMA_GRETYPE_CHAR_RNG_UPPER: return true;
#         default:                           return false;
#     }
# }
def is_char_element(elem: llama_grammar_element) -> bool:
    return elem.type in (
        llama_gretype.LLAMA_GRETYPE_CHAR,
        llama_gretype.LLAMA_GRETYPE_CHAR_NOT,
        llama_gretype.LLAMA_GRETYPE_CHAR_ALT,
        llama_gretype.LLAMA_GRETYPE_CHAR_RNG_UPPER,
    )


# void print_rule(
#         FILE     * file,
#         uint32_t   rule_id,
#         const std::vector<llama_grammar_element> & rule,
#         const std::map<uint32_t, std::string>    & symbol_id_names) {
def print_rule(
    file: TextIO,
    rule_id: uint32_t,
    rule: std__vector[llama_grammar_element],
    symbol_id_names: std__map[uint32_t, str],
) -> None:
    #     if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
    #         throw std::runtime_error(
    #             "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    #     }
    #     fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    if rule.empty() or rule.back().type != llama_gretype.LLAMA_GRETYPE_END:
        raise RuntimeError(
            "malformed rule, does not end with LLAMA_GRETYPE_END: "
            + str(rule_id)
        )
    print(f"{symbol_id_names.at(rule_id)} ::=", file=file, end=" ")
    #     for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
    #         llama_grammar_element elem = rule[i];
    #         switch (elem.type) {
    #             case LLAMA_GRETYPE_END:
    #                 throw std::runtime_error(
    #                     "unexpected end of rule: " + std::to_string(rule_id) + "," +
    #                     std::to_string(i));
    #             case LLAMA_GRETYPE_ALT:
    #                 fprintf(file, "| ");
    #                 break;
    #             case LLAMA_GRETYPE_RULE_REF:
    #                 fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
    #                 break;
    #             case LLAMA_GRETYPE_CHAR:
    #                 fprintf(file, "[");
    #                 print_grammar_char(file, elem.value);
    #                 break;
    #             case LLAMA_GRETYPE_CHAR_NOT:
    #                 fprintf(file, "[^");
    #                 print_grammar_char(file, elem.value);
    #                 break;
    #             case LLAMA_GRETYPE_CHAR_RNG_UPPER:
    #                 if (i == 0 || !is_char_element(rule[i - 1])) {
    #                     throw std::runtime_error(
    #                         "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
    #                         std::to_string(rule_id) + "," + std::to_string(i));
    #                 }
    #                 fprintf(file, "-");
    #                 print_grammar_char(file, elem.value);
    #                 break;
    #             case LLAMA_GRETYPE_CHAR_ALT:
    #                 if (i == 0 || !is_char_element(rule[i - 1])) {
    #                     throw std::runtime_error(
    #                         "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
    #                         std::to_string(rule_id) + "," + std::to_string(i));
    #                 }
    #                 print_grammar_char(file, elem.value);
    #                 break;
    #         }
    for i, elem in enumerate(rule[:-1]):
        switch = elem.type  # type: llama_gretype
        if switch == llama_gretype.LLAMA_GRETYPE_END:
            raise RuntimeError(
                "unexpected end of rule: " + str(rule_id) + "," + str(i)
            )
        elif switch == llama_gretype.LLAMA_GRETYPE_ALT:
            print("| ", file=file, end="")
        elif switch == llama_gretype.LLAMA_GRETYPE_RULE_REF:
            print(f"{symbol_id_names.at(elem.value)} ", file=file, end="")
        elif switch == llama_gretype.LLAMA_GRETYPE_CHAR:
            print("[", file=file, end="")
            print_grammar_char(file, elem.value)
        elif switch == llama_gretype.LLAMA_GRETYPE_CHAR_NOT:
            print("[^", file=file, end="")
            print_grammar_char(file, elem.value)
        elif switch == llama_gretype.LLAMA_GRETYPE_CHAR_RNG_UPPER:
            if i == 0 or not is_char_element(rule[i - 1]):
                raise RuntimeError(
                    "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: "
                    + str(rule_id)
                    + ","
                    + str(i)
                )
            print("-", file=file, end="")
            print_grammar_char(file, elem.value)
        elif switch == llama_gretype.LLAMA_GRETYPE_CHAR_ALT:
            if i == 0 or not is_char_element(rule[i - 1]):
                raise RuntimeError(
                    "LLAMA_GRETYPE_CHAR_ALT without preceding char: "
                    + str(rule_id)
                    + ","
                    + str(i)
                )
            print_grammar_char(file, elem.value)
        # if (is_char_element(elem)) {
        #     switch (rule[i + 1].type) {
        #         case LLAMA_GRETYPE_CHAR_ALT:
        #         case LLAMA_GRETYPE_CHAR_RNG_UPPER:
        #             break;
        #         default:
        #             fprintf(file, "] ");
        if is_char_element(elem):
            if rule[i + 1].type in (
                llama_gretype.LLAMA_GRETYPE_CHAR_ALT,
                llama_gretype.LLAMA_GRETYPE_CHAR_RNG_UPPER,
            ):
                pass
            else:
                print("] ", file=file, end="")
    #             }
    #         }
    #     }
    #     fprintf(file, "\n");
    # }
    print(file=file)


# void print_grammar(FILE * file, const parse_state & state) {
#     try {
#         std::map<uint32_t, std::string> symbol_id_names;
#         for (auto kv : state.symbol_ids) {
#             symbol_id_names[kv.second] = kv.first;
#         }
#         for (size_t i = 0, end = state.rules.size(); i < end; i++) {
#             // fprintf(file, "%zu: ", i);
#             // print_rule_binary(file, state.rules[i]);
#             print_rule(file, i, state.rules[i], symbol_id_names);
#             // fprintf(file, "\n");
#         }
#     } catch (const std::exception & err) {
#         fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, err.what());
#     }
# }
def print_grammar(file: TextIO, state: parse_state) -> None:
    try:
        symbol_id_names = std__map()  # type: std__map[uint32_t, str]
        for kv in state.symbol_ids.items():
            symbol_id_names[kv[1]] = kv[0]

        for i, rule in enumerate(state.rules):
            print_rule(file, i, rule, symbol_id_names)
    except Exception as err:
        print(
            f"{print_grammar.__name__}: error printing grammar: {err}",
            file=sys.stderr,
        )


def convert_to_rules(
    llama_grammar_elements: std__vector[std__vector[llama_grammar_element]],
) -> Array[llama_cpp.llama_grammar_element_p]:
    """Make an Array object that is used for `llama_grammer_init`"""

    # Step 1: Convert to c_llama_grammar_element
    llama_grammar_element_p_p = (
        []
    )  # type: List[List[llama_cpp.llama_grammar_element]]
    for subvector in llama_grammar_elements:
        llama_grammar_element_p_p.append([])
        for elem in subvector:
            c_llama_grammar_element = llama_cpp.llama_grammar_element()
            c_llama_grammar_element.type = c_int(elem.type.value)
            c_llama_grammar_element.value = c_uint32(elem.value)
            llama_grammar_element_p_p[-1].append(c_llama_grammar_element)

    # Step 2: Convert each list to llama_grammar_element array and get pointer
    element_arrays = [
        (llama_cpp.llama_grammar_element * len(sublist))(*sublist)
        for sublist in llama_grammar_element_p_p
    ]  # type: List[Array[llama_cpp.llama_grammar_element]]

    # Step 3: Get pointer of each array
    element_array_pointers = [
        cast(sublist, llama_cpp.llama_grammar_element_p)
        for sublist in element_arrays
    ]  # type: List[llama_cpp.llama_grammar_element_p]

    # Step 4: Make array of these pointers and get its pointer
    return (llama_cpp.llama_grammar_element_p * len(element_array_pointers))(
        *element_array_pointers
    )


def parse_grammar_init_args(
    bnf: str,
) -> Tuple[Array[llama_cpp.llama_grammar_element_p], c_size_t, c_size_t]:
    """Parse a GBNF string and return tuple of `grammar rules` and `root symbol id`"""
    parsed_grammar = parse(const_char_p(bnf))  # type: parse_state
    if parsed_grammar.rules.empty():
        raise Exception(
            f"{parse_grammar_init_args.__name__}: error parsing grammar file: parsed_grammar.rules is empty"
        )
    print(f"{parse_grammar_init_args.__name__} grammar:", file=sys.stderr)
    print_grammar(sys.stdout, parsed_grammar)
    print(file=sys.stderr)
    grammar_rules = (
        parsed_grammar.c_rules()
    )  # type: std__vector[std__vector[llama_grammar_element]]
    return (
        convert_to_rules(grammar_rules),
        c_size_t(grammar_rules.size()),
        c_size_t(parsed_grammar.symbol_ids.at("root")),
    )


def parse_grammar_init_args_from_file(
    bnf_path: Union[str, Path]
) -> Tuple[Array[llama_cpp.llama_grammar_element_p], c_size_t, c_size_t]:
    """Parse a GBNF file and return tuple of `grammar rules` and `root symbol id`"""
    try:
        with open(bnf_path) as f:
            params_grammer = f.read()
    except Exception as err:
        raise Exception(
            f"{parse_grammar_init_args_from_file.__name__}: error reading grammar file: {err}"
        )

    if params_grammer:
        return parse_grammar_init_args(params_grammer)

    raise Exception(
        f"{parse_grammar_init_args_from_file.__name__}: error parsing grammar file: params_grammer is empty"
    )


# def get_grammar_p(bnf: str) -> llama_cpp.llama_grammar_p:
#     """Parse a GBNF string and return pointer to `llama_grammar`"""

#     grammar_rules, root_symbol_id = parse_rules(bnf)

#     grammar_element_p_p = convert_to_double_ptr(
#         grammar_rules
#     )  # type: llama_cpp.llama_grammar_element_p_p

#     c_llama_grammar_p = llama_cpp.llama_grammar_init(
#         grammar_element_p_p,
#         c_size_t(grammar_rules.size()),
#         c_size_t(root_symbol_id),
#     )  # type: llama_cpp.llama_grammar_p
#     return c_llama_grammar_p


# def get_grammar_p_from_file(
#     bnf_path: Union[str, Path]
# ) -> llama_cpp.llama_grammar_p:
#     """Parse a GBNF file and return pointer to `llama_grammar`"""
#     try:
#         with open(bnf_path) as f:
#             params_grammer = f.read()
#     except Exception as err:
#         raise Exception(
#             f"{get_grammar_p_from_file.__name__}: error reading grammar file: {err}"
#         )

#     if params_grammer:
#         return get_grammar_p(params_grammer)

#     raise Exception(
#         f"{get_grammar_p_from_file.__name__}: error parsing grammar file: params_grammer is empty"
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate C++ parser from GBNF grammar"
    )
    parser.add_argument(
        "-g",
        "--grammar",
        type=str,
        default="./vendor/llama.cpp/grammars/json.gbnf",
        help="path to GBNF grammar file",
    )

    args = parser.parse_args()
    rules, n_rules, start_rule_index = parse_grammar_init_args_from_file(
        args.grammar
    )
    llama_grammar_p = llama_cpp.llama_grammar_init(
        rules,
        n_rules,
        start_rule_index,
    )  # type: llama_cpp.llama_grammar_p

    # ----- USAGE:
    # llama_cpp.llama_sample_grammar(ctx=..., candidates=..., grammar=llama_grammar_p)
    # llama_cpp.llama_grammar_accept_token(ctx=..., grammar=llama_grammar_p, token=...)

    # ----- SAMPLE OUTPUT:
    # main grammar:
    # root ::= object
    # object ::= [{] ws object_11 [}] ws
    # value ::= object | array | string | number | value_6 ws
    # array ::= [[] ws array_15 []] ws
    # string ::= ["] string_18 ["] ws
    # number ::= number_19 number_25 number_29 ws
    # value_6 ::= [t] [r] [u] [e] | [f] [a] [l] [s] [e] | [n] [u] [l] [l]
    # ws ::= ws_31
    # object_8 ::= string [:] ws value object_10
    # object_9 ::= [,] ws string [:] ws value
    # object_10 ::= object_9 object_10 |
    # object_11 ::= object_8 |
    # array_12 ::= value array_14
    # array_13 ::= [,] ws value
    # array_14 ::= array_13 array_14 |
    # array_15 ::= array_12 |
    # string_16 ::= [^"\] | [\] string_17
    # string_17 ::= ["\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
    # string_18 ::= string_16 string_18 |
    # number_19 ::= number_20 number_21
    # number_20 ::= [-] |
    # number_21 ::= [0-9] | [1-9] number_22
    # number_22 ::= [0-9] number_22 |
    # number_23 ::= [.] number_24
    # number_24 ::= [0-9] number_24 | [0-9]
    # number_25 ::= number_23 |
    # number_26 ::= [eE] number_27 number_28
    # number_27 ::= [-+] |
    # number_28 ::= [0-9] number_28 | [0-9]
    # number_29 ::= number_26 |
    # ws_30 ::= [ <U+0009><U+000A>] ws
    # ws_31 ::= ws_30 |
