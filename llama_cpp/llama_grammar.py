"""Python implementation of llama grammar parser directly translated from C++ source file in vendor/llama.cpp/common/grammar-parser.cpp."""

# flake8: noqa
from pathlib import Path
import sys
from ctypes import *  # type: ignore
from enum import Enum
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
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
    lookup = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)
    first_byte = ord(src[0])  # type: int
    highbits = first_byte >> 4  # type: int
    len = lookup[highbits]  # type: int
    mask = (1 << (8 - len)) - 1  # type: int
    value = first_byte & mask  # type: int
    end = src + len  # type: const_char_p # may overrun!
    pos = src + 1  # type: const_char_p
    while pos < end and pos[0]:
        value = (value << 6) + (ord(pos[0]) & 0x3F)
        pos += 1
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
    [^"\\] |
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
    [^"\\] |
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

PRIMITIVE_RULES = {
    "boolean": '("true" | "false") space',
    "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
    "string": r""" "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space """,
    "null": '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}


class SchemaConverter:
    def __init__(self, prop_order):
        self._prop_order = prop_order
        self._rules = {"space": SPACE_RULE}
        self._defs: Dict[str, Any] = {}

    def _format_literal(self, literal: str):
        escaped: str = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal)
        )
        return f'"{escaped}"'

    def _add_rule(self, name: str, rule: str):
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f"{esc_name}{i}" in self._rules:
                i += 1
            key = f"{esc_name}{i}"
        self._rules[key] = rule
        return key

    def visit(self, schema: Dict[str, Any], name: str) -> str:
        rule_name = name or "root"

        if "$defs" in schema:
            # add defs to self._defs for later inlining
            for def_name, def_schema in schema["$defs"].items():
                self._defs[def_name] = def_schema

        if "oneOf" in schema or "anyOf" in schema:
            rule = " | ".join(
                (
                    self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
                    for i, alt_schema in enumerate(
                        schema.get("oneOf") or schema["anyOf"]
                    )
                )
            )
            return self._add_rule(rule_name, rule)

        elif "const" in schema:
            return self._add_rule(rule_name, self._format_literal(schema["const"]))

        elif "enum" in schema:
            rule = " | ".join((self._format_literal(v) for v in schema["enum"]))
            return self._add_rule(rule_name, rule)

        elif "$ref" in schema:
            ref = schema["$ref"]
            assert ref.startswith("#/$defs/"), f"Unrecognized schema: {schema}"
            # inline $defs
            def_name = ref[len("#/$defs/") :]
            def_schema = self._defs[def_name]
            return self.visit(def_schema, f'{name}{"-" if name else ""}{def_name}')


        schema_type: Optional[str] = schema.get("type") # type: ignore
        assert isinstance(schema_type, str), f"Unrecognized schema: {schema}"

        if schema_type == "object" and "properties" in schema:
            # TODO: `required` keyword
            if self._prop_order:
                prop_order = self._prop_order
                prop_pairs = sorted(
                    schema["properties"].items(),
                    # sort by position in prop_order (if specified) then by key
                    key=lambda kv: (prop_order.get(kv[0], len(prop_order)), kv[0]),
                )
            else:
                prop_pairs = schema["properties"].items()

            rule = '"{" space'
            for i, (prop_name, prop_schema) in enumerate(prop_pairs):
                prop_rule_name = self.visit(
                    prop_schema, f'{name}{"-" if name else ""}{prop_name}'
                )
                if i > 0:
                    rule += ' "," space'
                rule += rf' {self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            rule += ' "}" space'

            return self._add_rule(rule_name, rule)

        elif schema_type == "array" and "items" in schema:
            # TODO `prefixItems` keyword
            item_rule_name = self.visit(
                schema["items"], f'{name}{"-" if name else ""}item'
            )
            list_item_operator = f'("," space {item_rule_name})'
            successive_items = ""
            min_items = schema.get("minItems", 0)
            if min_items > 0:
               first_item = f"({item_rule_name})"
               successive_items = list_item_operator * (min_items - 1)
               min_items -= 1
            else:
               first_item = f"({item_rule_name})?"
            max_items = schema.get("maxItems")
            if max_items is not None and max_items > min_items:
                successive_items += (list_item_operator + "?") * (max_items - min_items - 1)
            else:
                successive_items += list_item_operator + "*"
            rule = f'"[" space {first_item} {successive_items} "]" space'
            return self._add_rule(rule_name, rule)

        else:
            assert schema_type in PRIMITIVE_RULES, f"Unrecognized schema: {schema}"
            return self._add_rule(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

    def format_grammar(self):
        return "\n".join((f"{name} ::= {rule}" for name, rule in self._rules.items()))


def json_schema_to_gbnf(schema: str, prop_order: Optional[List[str]] = None):
    prop_order = prop_order or []
    schema = json.loads(schema)
    prop_order = {name: idx for idx, name in enumerate(prop_order)}
    converter = SchemaConverter(prop_order)
    converter.visit(schema, "")
    return converter.format_grammar()
