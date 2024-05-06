"""Python implementation of llama grammar parser directly translated from C++ source file in vendor/llama.cpp/common/grammar-parser.cpp."""

# flake8: noqa
from pathlib import Path
import sys
from ctypes import *  # type: ignore
from enum import Enum
from itertools import islice, groupby
from typing import (
    Any,
    Callable,
    Dict,
    Set,
    Generic,
    List,
    Optional,
    OrderedDict,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import llama_cpp.llama_cpp as llama_cpp

# Type aliases
llama_grammar_element = llama_cpp.llama_grammar_element
llama_grammar_element_p = llama_cpp.llama_grammar_element_p
llama_grammar_p = llama_cpp.llama_grammar_p

# Type variables
Ptr = TypeVar("Ptr", bound="const_char_p")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class Sentinel:
    """Used to mark the end of a iterator of std::vector & std::map."""


class LlamaGrammar:
    """Keeps reference counts of all the arguments, so that they are not
    garbage collected by Python."""

    def __del__(self) -> None:
        """Free the grammar pointer when the object is deleted."""
        if self.grammar is not None:
            llama_cpp.llama_grammar_free(self.grammar)
            self.grammar = None

    def __init__(
        self,
        parsed_grammar: "parse_state",
    ) -> None:
        """Initialize the grammar pointer from the parsed state."""
        self._grammar_rules = (
            parsed_grammar.c_rules()
        )  # type: std.vector[std.vector[LlamaGrammarElement]]
        self._n_rules = self._grammar_rules.size()  # type: int
        self._start_rule_index = parsed_grammar.symbol_ids.at("root")  # type: int
        self.init()

    @classmethod
    def from_string(cls, grammar: str, verbose: bool = True) -> "LlamaGrammar":
        """Convert a GBNF grammar to a Llama grammar."""
        parsed_grammar = parse(const_char_p(grammar))  # type: parse_state
        if parsed_grammar.rules.empty():
            raise ValueError(
                f"{cls.from_string.__name__}: error parsing grammar file: parsed_grammar.rules is empty"
            )
        if verbose:
            print(f"{cls.from_string.__name__} grammar:", file=sys.stderr)
            print_grammar(sys.stderr, parsed_grammar)
            print(file=sys.stderr)
        return cls(parsed_grammar)

    @classmethod
    def from_json_schema(
        cls,
        json_schema: str,
        verbose: bool = True,
    ) -> "LlamaGrammar":
        """Convert a JSON schema to a Llama grammar."""
        return cls.from_string(json_schema_to_gbnf(json_schema), verbose=verbose)

    @classmethod
    def from_file(cls, file: Union[str, Path], verbose: bool = True) -> "LlamaGrammar":
        try:
            with open(file) as f:
                grammar = f.read()
        except Exception as err:
            raise Exception(
                f"{cls.from_file.__name__}: error reading grammar file: {err}"
            )

        if grammar:
            return cls.from_string(grammar, verbose=verbose)

        raise ValueError(
            f"{cls.from_file.__name__}: error parsing grammar file: params_grammer is empty"
        )

    def init(self) -> None:
        # Step 1: Convert LlamaGrammarElement to llama_grammar_element
        self._element_lists = [
            [
                llama_grammar_element(c_int(elem.type.value), c_uint32(elem.value))
                for elem in subvector
            ]
            for subvector in self._grammar_rules
        ]  # type: List[List[llama_grammar_element]]

        # Step 2: Convert each list to llama_grammar_element array and get pointer
        self._element_arrays = [
            (llama_grammar_element * len(sublist))(*sublist)
            for sublist in self._element_lists
        ]  # type: List[Array[llama_grammar_element]]

        # Step 3: Get pointer of each array
        self._element_array_pointers = [
            cast(subarray, llama_grammar_element_p) for subarray in self._element_arrays
        ]  # type: List[llama_grammar_element_p]

        # Step 4: Make array of these pointers and get its pointer
        self._rules = (llama_grammar_element_p * len(self._element_array_pointers))(
            *self._element_array_pointers
        )
        self.grammar = llama_cpp.llama_grammar_init(
            self._rules, c_size_t(self._n_rules), c_size_t(self._start_rule_index)
        )

    def reset(self) -> None:
        if self.grammar is not None:
            llama_cpp.llama_grammar_free(self.grammar)
        self.init()


class LlamaGrammarElement:
    def __init__(self, type: "llama_gretype", value: int):
        self.type = type
        self.value = value  # Unicode code point or rule ID


class const_char_p:
    """C++ implementation of const char *."""

    def __init__(self, value: Union[str, Ptr], move: Optional[int] = None):
        if isinstance(value, const_char_p):
            # We're copying an existing const_char_p
            self.value = value.value
            self.pos = value.pos + (move or 0)
            return

        # We're creating a new const_char_p
        self.value = value
        self.pos = move or 0

    def __str__(self) -> str:
        assert self.value is not None, "null pointer"
        return self.value[self.pos :]

    def __getitem__(self, index: int) -> str:
        value = str(self)
        return value[index] if index < len(value) else ""

    @overload
    def __add__(self: Ptr, other: int) -> Ptr:
        ...

    @overload
    def __add__(self: Ptr, other: Ptr) -> int:
        ...

    def __add__(self: Ptr, other: Union[int, Ptr]) -> Union[int, Ptr]:
        return (
            self.__class__(self.value, self.pos + other)
            if isinstance(other, int)
            else self.pos + other.pos
        )

    @overload
    def __sub__(self: Ptr, other: int) -> Ptr:
        ...

    @overload
    def __sub__(self: Ptr, other: Ptr) -> int:
        ...

    def __sub__(self: Ptr, other: Union[int, Ptr]) -> Union[int, Ptr]:
        return (
            self.__class__(self.value, self.pos - other)
            if isinstance(other, int)
            else self.pos - other.pos
        )

    def __eq__(self: Ptr, other: Ptr) -> bool:
        assert self.value == other.value, "comparing pointers from different strings"
        return self.pos == other.pos

    def __lt__(self: Ptr, other: Ptr) -> bool:
        assert self.value == other.value, "comparing pointers from different strings"
        return self.pos < other.pos

    def __gt__(self: Ptr, other: Ptr) -> bool:
        assert self.value == other.value, "comparing pointers from different strings"
        return self.pos > other.pos


class std:
    @staticmethod
    def string(ptr: const_char_p, length: Optional[int] = None) -> str:
        """C++ implementation of std::string constructor."""
        value = str(ptr)
        if length is not None:
            value = value[:length]
        return value

    class vector(Generic[T], List[T]):
        """C++ implementation of std::vector."""

        class iterator:
            def __init__(self, vector: "std.vector[T]", index: int):
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

            def __add__(self, value: int) -> "std.vector[T].iterator":
                return self.__class__(self._vector, self._index + value)

            def __sub__(self, value: int) -> "std.vector[T].iterator":
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

        def clear(self) -> None:
            self.modify()
            super().clear()

        def empty(self) -> bool:
            return self.size() == 0

        def data(self) -> "std.vector[T]":
            return self

        def resize(
            self,
            new_size: int,
            fill_value_factory: Optional[Callable[[], T]] = None,
        ) -> None:
            if new_size > self.size():
                if fill_value_factory is None:
                    raise ValueError("A fill value factory function must be provided.")
                self.reserve(new_size, fill_value_factory)
            elif new_size < self.size():
                self[:] = self[:new_size]

        def reserve(self, capacity: int, fill_value_factory: Callable[[], T]) -> None:
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
            pos: "std.vector[T].iterator",
            first: "std.vector[T].iterator",
            last: "std.vector[T].iterator",
        ) -> None:
            self[pos._index : pos._index] = list(
                islice(first._vector, first._index, last._index)
            )

        def begin(self) -> "std.vector[T].iterator":
            return self.iterator(self, 0)

        def end(self) -> "std.vector[T].iterator":
            return self.iterator(self, self.size())

    class map(Generic[T, U], OrderedDict[T, U]):
        """C++ implementation of std::map."""

        class iterator(Generic[V, W]):
            def __init__(self, _map: "std.map[T, U]", key: Union[T, Sentinel]):
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
        ) -> Tuple["std.map[T, U].iterator[T, U]", bool]:
            if key in self:
                return self.iterator(self, key), False
            else:
                self[key] = value
                return self.iterator(self, key), True

        def find(self, key: T) -> "std.map[T, U].iterator[T, U]":
            if key in self:
                return self.iterator(self, key)
            else:
                return self.end()

        def at(self, key: T) -> U:
            if key in self:
                return self[key]
            else:
                raise KeyError("The provided key is not found in the map.")

        def erase(self, iterator: "std.map[T, U].iterator[T, U]") -> None:
            key = iterator.first
            if key in self:
                del self[key]

        def size(self) -> int:
            return len(self)

        def empty(self) -> bool:
            return self.size() == 0

        def lower_bound(self, key: T) -> "std.map[T, U].iterator[T, U]":
            try:
                keys = sorted(list(self.keys()))  # type: ignore
                for k in keys:
                    if k >= key:
                        return self.iterator(self, k)
                raise ValueError("No key found that is not less than the input key")
            except TypeError:
                raise TypeError("Keys of type T cannot be sorted.")

        def begin(self) -> "std.map[T, U].iterator[T, U]":
            return self.iterator(self, next(iter(self)))

        def end(self) -> "std.map[T, U].iterator[T, U]":
            return self.iterator(self, Sentinel())


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


# struct parse_state {
#     std::map<std::string, uint32_t>                 symbol_ids;
#     std::vector<std::vector<llama_grammar_element>> rules;
#     std::vector<const llama_grammar_element *> c_rules();
# };
class parse_state:
    def __init__(self):
        self.symbol_ids: std.map[str, int] = std.map()
        self.rules: std.vector[std.vector[LlamaGrammarElement]] = std.vector()

    # std::vector<const llama_grammar_element *> parse_state::c_rules() {
    #     std::vector<const llama_grammar_element *> ret;
    #     for (const auto & rule : rules) {
    #         ret.push_back(rule.data());
    #     }
    #     return ret;
    # }
    def c_rules(self) -> std.vector[std.vector[LlamaGrammarElement]]:
        ret = std.vector()  # type: std.vector[std.vector[LlamaGrammarElement]]
        for rule in self.rules:
            ret.push_back(rule.data())
        return ret

    def __repr__(self) -> str:
        return (
            f"parse_state(symbol_ids={len(self.symbol_ids)}, rules={len(self.rules)})"
        )


# struct llama_grammar {
#     const std::vector<std::vector<llama_grammar_element>>   rules;
#     std::vector<std::vector<const llama_grammar_element *>> stacks;
# };
# class llama_grammar:
#     def __init__(
#         self,
#         rules: std.vector[std.vector[llama_grammar_element]],
#         stacks: std.vector[std.vector[llama_grammar_element]],
#     ):
#         self.rules = rules
#         self.stacks = stacks


# uint32_t get_symbol_id(parse_state & state, const char * src, size_t len) {
#     uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
#     auto result = state.symbol_ids.insert(std::make_pair(std::string(src, len), next_id));
#     return result.first->second;
# }
def get_symbol_id(state: parse_state, src: const_char_p, len: int) -> int:
    next_id = state.symbol_ids.size()  # type: int
    result = state.symbol_ids.insert(std.string(src, len), next_id)
    return result[0].second  # type: ignore


# uint32_t generate_symbol_id(parse_state & state, const std::string & base_name) {
#     uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
#     state.symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
#     return next_id;
# }
def generate_symbol_id(state: parse_state, base_name: str) -> int:
    next_id = state.symbol_ids.size()  # type: int
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
    rule_id: int,
    rule: std.vector[LlamaGrammarElement],
) -> None:
    if state.rules.size() <= rule_id:
        state.rules.resize(
            rule_id + 1,
            fill_value_factory=std.vector[LlamaGrammarElement],
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
def decode_utf8(src: const_char_p) -> Tuple[int, const_char_p]:
    """Decodes a UTF-8 character from the source string."""
    # Get the codepoint of the first character
    value = ord(src[0])
    # Move the pointer ahead one character
    pos = src + 1

    return value, pos


# bool is_word_char(char c) {
#     return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || ('0' <= c && c <= '9');
# }
def is_word_char(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "-" or ("0" <= c <= "9")


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
def parse_hex(src: const_char_p, size: int) -> Tuple[int, const_char_p]:
    pos = const_char_p(src)  # type: const_char_p
    end = src + size  # type: const_char_p
    value = 0  # type: int
    while pos < end and pos[0]:
        value <<= 4
        c = pos[0]  # type: str
        if "a" <= c <= "f":
            value += ord(c) - ord("a") + 10
        elif "A" <= c <= "F":
            value += ord(c) - ord("A") + 10
        elif "0" <= c <= "9":
            value += ord(c) - ord("0")
        else:
            break
        pos += 1
    if pos != end:
        raise RuntimeError("expecting " + str(size) + " hex chars at " + str(src))
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
def parse_char(src: const_char_p) -> Tuple[int, const_char_p]:
    if src[0] == "\\":
        case = src[1]  # type: str
        if case == "x":
            return parse_hex(src + 2, 2)
        elif case == "u":
            return parse_hex(src + 2, 4)
        elif case == "U":
            return parse_hex(src + 2, 8)
        elif case == "t":
            return (ord("\t"), src + 2)  # implicit cast
        elif case == "r":
            return (ord("\r"), src + 2)  # implicit cast
        elif case == "n":
            return (ord("\n"), src + 2)  # implicit cast
        elif case in ("\\", '"', "[", "]"):
            return (ord(case), src + 2)  # implicit cast
        else:
            raise RuntimeError("unknown escape at " + str(src))
    elif src[0]:
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
    while is_word_char(pos[0]):
        pos += 1
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
    pos = const_char_p(src)  # type: const_char_p
    while pos[0] in (" ", "\t", "#") or (newline_ok and pos[0] in ("\r", "\n")):
        if pos[0] == "#":
            while pos[0] is not None and pos[0] not in ("\r", "\n"):
                pos += 1
        else:
            pos += 1
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
    out_elements: std.vector[LlamaGrammarElement],
    is_nested: bool,
) -> const_char_p:
    # size_t last_sym_start = out_elements.size();
    # const char * pos = src;
    last_sym_start = out_elements.size()  # type: int
    pos = const_char_p(src)  # type: const_char_p
    # while (*pos) {
    while pos[0]:
        # if (*pos == '"') { // literal string
        #     pos++;
        #     last_sym_start = out_elements.size();
        #     while (*pos != '"') {
        #         auto char_pair = parse_char(pos);
        #                 pos       = char_pair.second;
        #         out_elements.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
        #     }
        #     pos = parse_space(pos + 1, is_nested);
        if pos[0] == '"':  # literal string
            pos += 1
            last_sym_start = out_elements.size()
            while pos[0] != '"':
                char_pair = parse_char(pos)  # type: Tuple[int, const_char_p]
                pos = char_pair[1]
                out_elements.push_back(
                    LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_CHAR, char_pair[0])
                )
            pos = parse_space(pos + 1, is_nested)
        # } else if (*pos == '[') { // char range(s)
        #     pos++;
        #     enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
        elif pos[0] == "[":  # char range(s)
            pos += 1
            start_type = llama_gretype.LLAMA_GRETYPE_CHAR  # type: llama_gretype
            # if (*pos == '^') {
            #     pos++;
            #     start_type = LLAMA_GRETYPE_CHAR_NOT;
            # }
            # last_sym_start = out_elements.size();
            if pos[0] == "^":
                pos += 1
                start_type = llama_gretype.LLAMA_GRETYPE_CHAR_NOT
            last_sym_start = out_elements.size()
            # while (*pos != ']') {
            #     auto char_pair = parse_char(pos);
            #             pos       = char_pair.second;
            #     enum llama_gretype type = last_sym_start < out_elements.size()
            #         ? LLAMA_GRETYPE_CHAR_ALT
            #         : start_type;
            #     out_elements.push_back({type, char_pair.first});
            while pos[0] != "]":
                char_pair = parse_char(pos)  # type: Tuple[int, const_char_p]
                pos = char_pair[1]
                type = (
                    llama_gretype.LLAMA_GRETYPE_CHAR_ALT
                    if last_sym_start < out_elements.size()
                    else start_type
                )  # type: llama_gretype
                out_elements.push_back(LlamaGrammarElement(type, char_pair[0]))
                #     if (pos[0] == '-' && pos[1] != ']') {
                #         auto endchar_pair = parse_char(pos + 1);
                #                 pos          = endchar_pair.second;
                #         out_elements.push_back({LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                #     }
                # }
                if pos[0] == "-" and pos[1] != "]":
                    endchar_pair = parse_char(pos + 1)  # type: Tuple[int, const_char_p]
                    pos = endchar_pair[1]
                    out_elements.push_back(
                        LlamaGrammarElement(
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
        elif is_word_char(pos[0]):  # rule reference
            name_end = parse_name(pos)  # type: const_char_p
            ref_rule_id = get_symbol_id(state, pos, name_end - pos)  # type: int
            pos = parse_space(name_end, is_nested)
            last_sym_start = out_elements.size()
            out_elements.push_back(
                LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_RULE_REF, ref_rule_id)
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
        elif pos[0] == "(":  # grouping
            # parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, True)
            sub_rule_id = generate_symbol_id(state, rule_name)  # type: int
            pos = parse_alternates(state, pos, rule_name, sub_rule_id, True)
            last_sym_start = out_elements.size()
            # output reference to synthesized rule
            out_elements.push_back(
                LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id)
            )
            if pos[0] != ")":
                raise RuntimeError("expecting ')' at " + str(pos))
            pos = parse_space(pos + 1, is_nested)
        # } else if (*pos == '*' || *pos == '+' || *pos == '?') { // repetition operator
        #     if (last_sym_start == out_elements.size()) {
        #         throw std::runtime_error(std::string("expecting preceeding item to */+/? at ") + pos);
        #     }
        elif pos[0] in ("*", "+", "?"):  # repetition operator
            if last_sym_start == out_elements.size():
                raise RuntimeError("expecting preceding item to */+/? at " + str(pos))
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
            sub_rule_id = generate_symbol_id(state, rule_name)  # type: int
            sub_rule = std.vector[
                LlamaGrammarElement
            ]()  # type: std.vector[LlamaGrammarElement]
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
            if pos[0] in ("*", "+"):
                sub_rule.push_back(
                    LlamaGrammarElement(
                        llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id
                    )
                )
            sub_rule.push_back(LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_ALT, 0))
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
            if pos[0] == "+":
                # add preceding symbol as alternate only for '+' (otherwise empty)
                sub_rule.insert(
                    sub_rule.end(),
                    out_elements.begin() + last_sym_start,
                    out_elements.end(),
                )
            sub_rule.push_back(LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_END, 0))
            add_rule(state, sub_rule_id, sub_rule)
            # in original rule, replace previous symbol with reference to generated rule
            out_elements.resize(last_sym_start)
            out_elements.push_back(
                LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_RULE_REF, sub_rule_id)
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
    rule_id: int,
    is_nested: bool,
) -> const_char_p:
    rule = std.vector()  # type: std.vector[LlamaGrammarElement]
    pos = parse_sequence(state, src, rule_name, rule, is_nested)  # type: const_char_p
    while pos[0] == "|":
        rule.push_back(LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_ALT, 0))
        pos = parse_space(pos + 1, True)
        pos = parse_sequence(state, pos, rule_name, rule, is_nested)
    rule.push_back(LlamaGrammarElement(llama_gretype.LLAMA_GRETYPE_END, 0))
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
    name_len = name_end - src  # type: int
    rule_id = get_symbol_id(state, src, name_len)  # type: int
    name = std.string(src, name_len)  # type: str

    if not (pos[0] == ":" and pos[1] == ":" and pos[2] == "="):
        raise RuntimeError("expecting ::= at " + str(pos))

    pos = parse_space(pos + 3, True)  # type: const_char_p
    pos = parse_alternates(state, pos, name, rule_id, False)  # type: const_char_p

    if pos[0] == "\r":
        pos += 2 if pos[1] == "\n" else 1
    elif pos[0] == "\n":
        pos += 1
    elif pos[0]:
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
        while pos[0]:
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
def print_grammar_char(file: TextIO, c: int) -> None:
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
def is_char_element(elem: LlamaGrammarElement) -> bool:
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
    rule_id: int,
    rule: std.vector[LlamaGrammarElement],
    symbol_id_names: std.map[int, str],
) -> None:
    #     if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
    #         throw std::runtime_error(
    #             "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    #     }
    #     fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    if rule.empty() or rule.back().type != llama_gretype.LLAMA_GRETYPE_END:
        raise RuntimeError(
            "malformed rule, does not end with LLAMA_GRETYPE_END: " + str(rule_id)
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
        case = elem.type  # type: llama_gretype
        if case is llama_gretype.LLAMA_GRETYPE_END:
            raise RuntimeError("unexpected end of rule: " + str(rule_id) + "," + str(i))
        elif case is llama_gretype.LLAMA_GRETYPE_ALT:
            print("| ", file=file, end="")
        elif case is llama_gretype.LLAMA_GRETYPE_RULE_REF:
            print(f"{symbol_id_names.at(elem.value)} ", file=file, end="")
        elif case is llama_gretype.LLAMA_GRETYPE_CHAR:
            print("[", file=file, end="")
            print_grammar_char(file, elem.value)
        elif case is llama_gretype.LLAMA_GRETYPE_CHAR_NOT:
            print("[^", file=file, end="")
            print_grammar_char(file, elem.value)
        elif case is llama_gretype.LLAMA_GRETYPE_CHAR_RNG_UPPER:
            if i == 0 or not is_char_element(rule[i - 1]):
                raise RuntimeError(
                    "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: "
                    + str(rule_id)
                    + ","
                    + str(i)
                )
            print("-", file=file, end="")
            print_grammar_char(file, elem.value)
        elif case is llama_gretype.LLAMA_GRETYPE_CHAR_ALT:
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
        symbol_id_names = std.map()  # type: std.map[int, str]
        for kv in state.symbol_ids.items():
            symbol_id_names[kv[1]] = kv[0]

        for i, rule in enumerate(state.rules):
            print_rule(file, i, rule, symbol_id_names)
    except Exception as err:
        print(
            f"{print_grammar.__name__}: error printing grammar: {err}",
            file=sys.stderr,
        )


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
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
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


def _build_repetition(item_rule, min_items, max_items, separator_rule=None, item_rule_is_literal=False):
    if not separator_rule:
        if min_items == 0 and max_items == 1:
            return f'{item_rule}?'
        elif min_items == 1 and max_items is None:
            return f'{item_rule}+'

    result = ''

    if min_items > 0:
        if item_rule_is_literal and separator_rule is None:
            result = '"' + (item_rule[1:-1] * min_items) + '"'
        else:
            result = (f' {separator_rule} ' if separator_rule else ' ').join([item_rule] * min_items)

    def opt_repetitions(up_to_n, prefix_with_sep=False):
        '''
            - n=4, no sep:             '(a (a (a (a)?)?)?)?'
            - n=4, sep=',', prefix:    '("," a ("," a ("," a ("," a)?)?)?)?'
            - n=4, sep=',', no prefix: '(a ("," a ("," a ("," a)?)?)?)?'
        '''

        content = f'{separator_rule} {item_rule}' if prefix_with_sep and separator_rule else item_rule
        if up_to_n == 0:
            return ''
        elif up_to_n == 1:
            return f'({content})?'
        elif separator_rule and not prefix_with_sep:
            return f'({content} {opt_repetitions(up_to_n - 1, prefix_with_sep=True)})?'
        else:
            return (f'({content} ' * up_to_n).rstrip() + (')?' * up_to_n)

    if min_items > 0 and max_items != min_items:
        result += ' '

    if max_items is not None:
        result += opt_repetitions(max_items - min_items, prefix_with_sep=min_items > 0)
    else:
        item_operator = f'({separator_rule + " " if separator_rule else ""}{item_rule})'

        if min_items == 0 and separator_rule:
            result = f'({item_rule} {item_operator}*)?'
        else:
            result += f'{item_operator}*'

    return result



class BuiltinRule:
    def __init__(self, content: str, deps: list = None):
        self.content = content
        self.deps = deps or []

_up_to_15_digits = _build_repetition('[0-9]', 0, 15)

PRIMITIVE_RULES = {
    'boolean'      : BuiltinRule('("true" | "false") space', []),
    'decimal-part' : BuiltinRule('[0-9] ' + _up_to_15_digits, []),
    'integral-part': BuiltinRule('[0-9] | [1-9] ' + _up_to_15_digits, []),
    'number'       : BuiltinRule('("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space', ['integral-part', 'decimal-part']),
    'integer'      : BuiltinRule('("-"? integral-part) space', ['integral-part']),
    'value'        : BuiltinRule('object | array | string | number | boolean | null', ['object', 'array', 'string', 'number', 'boolean', 'null']),
    'object'       : BuiltinRule('"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space', ['string', 'value']),
    'array'        : BuiltinRule('"[" space ( value ("," space value)* )? "]" space', ['value']),
    'uuid'         : BuiltinRule(r'"\"" ' + ' "-" '.join('[0-9a-fA-F]' * n for n in [8, 4, 4, 4, 12]) + r' "\"" space', []),
    'char'         : BuiltinRule(r'[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])', []),
    'string'       : BuiltinRule(r'"\"" char* "\"" space', ['char']),
    'null'         : BuiltinRule('"null" space', []),
}

# TODO: support "uri", "email" string formats
STRING_FORMAT_RULES = {
    'date'            : BuiltinRule('[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )', []),
    'time'            : BuiltinRule('([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )', []),
    'date-time'       : BuiltinRule('date "T" time', ['date', 'time']),
    'date-string'     : BuiltinRule('"\\"" date "\\"" space', ['date']),
    'time-string'     : BuiltinRule('"\\"" time "\\"" space', ['time']),
    'date-time-string': BuiltinRule('"\\"" date-time "\\"" space', ['date-time']),
}

DOTALL = '[\\U00000000-\\U0010FFFF]'
DOT = '[^\\x0A\\x0D]'

RESERVED_NAMES = set(["root", "dot", *PRIMITIVE_RULES.keys(), *STRING_FORMAT_RULES.keys()])


NON_LITERAL_SET = set('|.()[]{}*+?')
ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set('[]()|{}*+?')




class SchemaConverter:
    def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {
            'space': SPACE_RULE,
        }
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), literal
        )
        return f'"{escaped}"'

    def not_literal(self, literal: str, dotall: bool = True, maybe_escaped_underscores = False) -> str:
        '''
            not_literal('a') -> '[^a]'
            not_literal('abc') -> '([^a] | "a" ([^b] | "b" ([^c])?)?)?'
        '''
        assert len(literal) > 0, 'Empty literal not supported'
        def recurse(i: int):
            c = literal[i]
            if maybe_escaped_underscores and c == '_':
                yield f'[^{c}\\\\]'
                yield ' | '
                yield f'"\\\\"? "{c}"'
            else:
                yield f'[^{c}]'
            if i < len(literal) - 1:
                yield ' | '
                yield self._format_literal(c)
                yield ' ('
                yield from recurse(i + 1)
                yield ')?'

        return ''.join(('(', *recurse(0), ')'))

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f'{esc_name}{i}' in self._rules and self._rules[f'{esc_name}{i}'] != rule:
                i += 1
            key = f'{esc_name}{i}'
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        '''
            Resolves all $ref fields in the given schema, fetching any remote schemas,
            replacing $ref with absolute reference URL and populating self._refs with the
            respective referenced (sub)schema dictionaries.
        '''
        def visit(n: dict):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get('$ref')
                if ref is not None and ref not in self._refs:
                    if ref.startswith('https://'):
                        assert self._allow_fetch, 'Fetching remote schemas is not allowed (use --allow-fetch for force)'
                        import requests

                        frag_split = ref.split('#')
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(requests.get(ref).json(), base_url)
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == '':
                            return target
                    elif ref.startswith('#/'):
                        target = schema
                        ref = f'{url}{ref}'
                        n['$ref'] = ref
                    else:
                        raise ValueError(f'Unsupported ref {ref}')

                    for sel in ref.split('#')[-1].split('/')[1:]:
                        assert target is not None and sel in target, f'Error resolving ref {ref}: {sel} not in {target}'
                        target = target[sel]

                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)

            return n
        return visit(schema)

    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _visit_pattern(self, pattern, name):
        '''
            Transforms a regular expression pattern into a GBNF rule.

            Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
            Output: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

            Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

            Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
            we define sub-rules to keep the output lean.
        '''

        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        sub_rule_ids = {}

        i = 0
        length = len(pattern)

        def to_rule(s: Tuple[str, bool]) -> str:
            (txt, is_literal) = s
            return "\"" + txt + "\"" if is_literal else txt

        def transform() -> Tuple[str, bool]:
            '''
                Parse a unit at index i (advancing it), and return its string representation + whether it's a literal.
            '''
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
                return self._add_rule(f'dot', rule)

            def join_seq():
                nonlocal seq
                ret = []
                for is_literal, g in groupby(seq, lambda x: x[1]):
                    if is_literal:
                        ret.append((''.join(x[0] for x in g), True))
                    else:
                        ret.extend(g)
                if len(ret) == 1:
                    return ret[0]
                return (' '.join(to_rule(x) for x in seq), False)

            while i < length:
                c = pattern[i]
                if c == '.':
                    seq.append((get_dot(), False))
                    i += 1
                elif c == '(':
                    i += 1
                    if i < length:
                        assert pattern[i] != '?', f'Unsupported pattern syntax "{pattern[i]}" at index {i} of /{pattern}/'
                    seq.append((f'({to_rule(transform())})', False))
                elif c == ')':
                    i += 1
                    assert start > 0 and pattern[start-1] == '(', f'Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}'
                    return join_seq()
                elif c == '[':
                    square_brackets = c
                    i += 1
                    while i < length and pattern[i] != ']':
                        if pattern[i] == '\\':
                            square_brackets += pattern[i:i+2]
                            i += 2
                        else:
                            square_brackets += pattern[i]
                            i += 1
                    assert i < length, f'Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}'
                    square_brackets += ']'
                    i += 1
                    seq.append((square_brackets, False))
                elif c == '|':
                    seq.append(('|', False))
                    i += 1
                elif c in ('*', '+', '?'):
                    seq[-1] = (to_rule(seq[-1]) + c, False)
                    i += 1
                elif c == '{':
                    curly_brackets = c
                    i += 1
                    while i < length and pattern[i] != '}':
                        curly_brackets += pattern[i]
                        i += 1
                    assert i < length, f'Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}'
                    curly_brackets += '}'
                    i += 1
                    nums = [s.strip() for s in curly_brackets[1:-1].split(',')]
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
                        raise ValueError(f'Invalid quantifier {curly_brackets} in /{pattern}/')

                    (sub, sub_is_literal) = seq[-1]

                    if not sub_is_literal:
                        id = sub_rule_ids.get(sub)
                        if id is None:
                            id = self._add_rule(f'{name}-{len(sub_rule_ids) + 1}', sub)
                            sub_rule_ids[sub] = id
                        sub = id

                    seq[-1] = (_build_repetition(f'"{sub}"' if sub_is_literal else sub, min_times, max_times, item_rule_is_literal=sub_is_literal), False)
                else:
                    literal = ''
                    while i < length:
                        if pattern[i] == '\\' and i < length - 1:
                            next = pattern[i + 1]
                            if next in ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                                i += 1
                                literal += pattern[i]
                                i += 1
                            else:
                                literal += pattern[i:i+2]
                                i += 2
                        elif pattern[i] == '"' and not self._raw_pattern:
                            literal += '\\"'
                            i += 1
                        elif pattern[i] not in NON_LITERAL_SET and \
                                (i == length - 1 or literal == '' or pattern[i+1] == '.' or pattern[i+1] not in NON_LITERAL_SET):
                            literal += pattern[i]
                            i += 1
                        else:
                            break
                    if literal:
                        seq.append((literal, True))

            return join_seq()

        return self._add_rule(
            name,
            to_rule(transform()) if self._raw_pattern \
                else "\"\\\"\" " + to_rule(transform()) + " \"\\\"\" space")


    def _resolve_ref(self, ref):
        ref_name = ref.split('/')[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name

    def _generate_constant_rule(self, value):
        return self._format_literal(json.dumps(value))

    def visit(self, schema, name):
        schema_type = schema.get('type')
        schema_format = schema.get('format')
        rule_name = name + '-' if name in RESERVED_NAMES else name or 'root'

        if (ref := schema.get('$ref')) is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        elif 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))

        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._generate_constant_rule(schema['const']))

        elif 'enum' in schema:
            rule = ' | '.join((self._generate_constant_rule(v) for v in schema['enum']))
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, 'object') and \
             ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._add_rule(rule_name, self._build_object_rule(properties, required, name, schema.get('additionalProperties')))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
            hybrid_name = name
            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get('$ref')) is not None:
                    comp_schema = self._refs[ref]

                if 'properties' in comp_schema:
                    for prop_name, prop_schema in comp_schema['properties'].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema['allOf']:
                if 'anyOf' in t:
                    for tt in t['anyOf']:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._add_rule(rule_name, self._build_object_rule(properties, required, hybrid_name, additional_properties=[]))

        elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
            items = schema.get('items') or schema['prefixItems']
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
                        for i, item in enumerate(items)) +
                    ' "]" space')
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                return self._add_rule(rule_name, '"[" space ' + _build_repetition(item_rule_name, min_items, max_items, separator_rule='"," space') + ' "]" space')

        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._visit_pattern(schema['pattern'], rule_name)

        elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema_format or ''):
            return self._add_primitive(
                'root' if rule_name == 'root' else schema_format,
                PRIMITIVE_RULES['uuid']
            )

        elif schema_type in (None, 'string') and f'{schema_format}-string' in STRING_FORMAT_RULES:
            prim_name = f'{schema_format}-string'
            return self._add_rule(rule_name, self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]))

        elif schema_type == 'string' and ('minLength' in schema or 'maxLength' in schema):
            char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
            min_len = schema.get('minLength', 0)
            max_len = schema.get('maxLength')

            return self._add_rule(rule_name, r'"\"" ' + _build_repetition(char_rule, min_len, max_len) + r' "\"" space')

        elif (schema_type == 'object') or (len(schema) == 0):
            return self._add_rule(rule_name, self._add_primitive('object', PRIMITIVE_RULES['object']))

        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            # TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_primitive('root' if rule_name == 'root' else schema_type, PRIMITIVE_RULES[schema_type])

    def _add_primitive(self, name: str, rule: BuiltinRule):
        n = self._add_rule(name, rule.content)

        for dep in rule.deps:
            dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
            assert dep_rule, f'Rule {dep} not known'
            if dep not in self._rules:
                self._add_primitive(dep, dep_rule)
        return n

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str, additional_properties: Union[bool, Any]):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}'
            )
        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]

        if additional_properties == True or isinstance(additional_properties, dict):
            sub_name = f'{name}{"-" if name else ""}additional'
            value_rule = self.visit({} if additional_properties == True else additional_properties, f'{sub_name}-value')
            prop_kv_rule_names["*"] = self._add_rule(
                f'{sub_name}-kv',
                self._add_primitive('string', PRIMITIVE_RULES['string']) + f' ":" space {value_rule}'
            )
            optional_props.append("*")

        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                if k == '*':
                    res = self._add_rule(
                        f'{name}{"-" if name else ""}additional-kvs',
                        f'{kv_rule_name} ( "," space ' + kv_rule_name + ' )*'
                    )
                elif first_is_optional:
                    res = f'( "," space {kv_rule_name} )?'
                else:
                    res = kv_rule_name
                if len(rest) > 0:
                    res += ' ' + self._add_rule(
                        f'{name}{"-" if name else ""}{k}-rest',
                        get_recursive_refs(rest, first_is_optional=True)
                    )
                return res

            rule += ' | '.join(
                get_recursive_refs(optional_props[i:], first_is_optional=False)
                for i in range(len(optional_props))
            )
            if required_props:
                rule += ' )'
            rule += ' )?'

        rule += ' "}" space'

        return rule

    def format_grammar(self):
        return '\n'.join(
            f'{name} ::= {rule}'
            for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
        )
def json_schema_to_gbnf(schema: str, prop_order: Optional[List[str]] = None):
    prop_order = prop_order or []
    schema = json.loads(schema)
    prop_order = {name: idx for idx, name in enumerate(prop_order)}
    converter = SchemaConverter(prop_order=prop_order, allow_fetch=False, dotall=False, raw_pattern=False)
    schema = converter.resolve_refs(schema, "stdin")
    converter.visit(schema, "")
    return converter.format_grammar()
