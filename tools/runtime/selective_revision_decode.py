"""Runtime JSON decoding shim for the model worker environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, NoReturn

type JsonScalar = str | int | float | bool | None
type DecodedJson = JsonScalar | list["DecodedJson"] | dict[str, "DecodedJson"]

_ESCAPES: Final[dict[str, str]] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS: Final = frozenset("0123456789abcdefABCDEF")
_HEX_QUAD_LENGTH: Final = 4
_CONTROL_CODEPOINT_LIMIT: Final = 0x20
_HIGH_SURROGATE_START: Final = 0xD800
_HIGH_SURROGATE_END: Final = 0xDBFF
_LOW_SURROGATE_START: Final = 0xDC00
_LOW_SURROGATE_END: Final = 0xDFFF
_SUPPLEMENTARY_PLANE_START: Final = 0x10000
_SURROGATE_SHIFT: Final = 10


def decode_json_value(line: str) -> DecodedJson:
    """Decode JSON with a typed stdlib-free fallback for the model venv."""
    parser = _JsonParser(line)
    value = parser.parse_value()
    parser.require_end()
    return value


def coerce_json_value(value: object) -> DecodedJson:
    """Coerce trusted Python containers produced by dataclass serialization."""
    try:
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError as error:
        message = f"unsupported JSON value: {type(value).__name__}"
        raise ValueError(message) from error
    return decode_json_value(encoded)


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(slots=True)
class _JsonParser:
    """Small recursive JSON parser used only when Pydantic is unavailable."""

    source: str
    index: int = 0

    def parse_value(self) -> DecodedJson:
        self._skip_whitespace()
        match self._peek():
            case '"':
                return self._string()
            case "{":
                return self._object()
            case "[":
                return self._array()
            case "t" | "f" | "n":
                return self._literal_value()
            case "-" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                return self._number()
            case "":
                _fail("unexpected end of JSON")
            case character:
                message = f"unexpected JSON character: {character}"
                _fail(message)

    def require_end(self) -> None:
        self._skip_whitespace()
        if self.index != len(self.source):
            message = f"trailing JSON data at byte {self.index}"
            _fail(message)

    def _object(self) -> dict[str, DecodedJson]:
        self._consume("{")
        parsed: dict[str, DecodedJson] = {}
        self._skip_whitespace()
        if self._peek() == "}":
            self._consume("}")
            return parsed
        while True:
            self._skip_whitespace()
            if self._peek() != '"':
                _fail("JSON object key must be a string")
            key = self._string()
            self._skip_whitespace()
            self._consume(":")
            parsed[key] = self.parse_value()
            self._skip_whitespace()
            match self._peek():
                case ",":
                    self._consume(",")
                case "}":
                    self._consume("}")
                    return parsed
                case character:
                    message = f"unexpected JSON object character: {character}"
                    _fail(message)

    def _array(self) -> list[DecodedJson]:
        self._consume("[")
        parsed: list[DecodedJson] = []
        self._skip_whitespace()
        if self._peek() == "]":
            self._consume("]")
            return parsed
        while True:
            parsed.append(self.parse_value())
            self._skip_whitespace()
            match self._peek():
                case ",":
                    self._consume(",")
                case "]":
                    self._consume("]")
                    return parsed
                case character:
                    message = f"unexpected JSON array character: {character}"
                    _fail(message)

    def _string(self) -> str:
        self._consume('"')
        pieces: list[str] = []
        while True:
            character = self._next()
            match character:
                case "":
                    _fail("unterminated JSON string")
                case '"':
                    return "".join(pieces)
                case "\\":
                    pieces.append(self._escape())
                case _:
                    if ord(character) < _CONTROL_CODEPOINT_LIMIT:
                        _fail("unescaped control character in JSON string")
                    pieces.append(character)

    def _escape(self) -> str:
        escaped = self._next()
        match escaped:
            case "u":
                return self._unicode_escape()
            case key if key in _ESCAPES:
                return _ESCAPES[key]
            case "":
                _fail("unterminated JSON escape")
            case _:
                message = f"invalid JSON escape: {escaped}"
                _fail(message)

    def _unicode_escape(self) -> str:
        codepoint = self._hex_quad()
        if (
            _HIGH_SURROGATE_START <= codepoint <= _HIGH_SURROGATE_END
            and self.source[self.index : self.index + 2] == "\\u"
        ):
            self.index += 2
            low = self._hex_quad()
            if not _LOW_SURROGATE_START <= low <= _LOW_SURROGATE_END:
                _fail("invalid low surrogate in JSON string")
            combined = (
                _SUPPLEMENTARY_PLANE_START
                + ((codepoint - _HIGH_SURROGATE_START) << _SURROGATE_SHIFT)
                + (low - _LOW_SURROGATE_START)
            )
            return chr(combined)
        if _LOW_SURROGATE_START <= codepoint <= _LOW_SURROGATE_END:
            _fail("unpaired low surrogate in JSON string")
        return chr(codepoint)

    def _hex_quad(self) -> int:
        raw = self.source[self.index : self.index + _HEX_QUAD_LENGTH]
        if len(raw) != _HEX_QUAD_LENGTH or any(character not in _HEX_DIGITS for character in raw):
            _fail("invalid JSON unicode escape")
        self.index += _HEX_QUAD_LENGTH
        return int(raw, 16)

    def _number(self) -> int | float:
        start = self.index
        if self._peek() == "-":
            self.index += 1
        self._integer_digits()
        if self._peek() == ".":
            self.index += 1
            self._digits("fraction")
        if self._peek() in {"e", "E"}:
            self.index += 1
            if self._peek() in {"+", "-"}:
                self.index += 1
            self._digits("exponent")
        raw = self.source[start : self.index]
        return float(raw) if any(character in raw for character in ".eE") else int(raw)

    def _integer_digits(self) -> None:
        match self._peek():
            case "0":
                self.index += 1
                if self._peek().isdigit():
                    _fail("leading zero in JSON number")
            case digit if digit.isdigit():
                self._digits("integer")
            case _:
                _fail("missing integer digits in JSON number")

    def _digits(self, label: str) -> None:
        start = self.index
        while self._peek().isdigit():
            self.index += 1
        if self.index == start:
            message = f"missing {label} digits in JSON number"
            _fail(message)

    def _consume_literal(self, literal: str) -> None:
        if not self.source.startswith(literal, self.index):
            message = f"invalid JSON literal at byte {self.index}"
            _fail(message)
        self.index += len(literal)

    def _consume(self, expected: str) -> None:
        actual = self._next()
        if actual != expected:
            message = f"expected {expected}, got {actual}"
            _fail(message)

    def _skip_whitespace(self) -> None:
        while self._peek() in {" ", "\n", "\r", "\t"}:
            self.index += 1

    def _peek(self) -> str:
        return "" if self.index >= len(self.source) else self.source[self.index]

    def _next(self) -> str:
        character = self._peek()
        if character != "":
            self.index += 1
        return character

    def _literal_value(self) -> bool | None:
        match self._peek():
            case "t":
                self._consume_literal("true")
                return True
            case "f":
                self._consume_literal("false")
                return False
            case "n":
                self._consume_literal("null")
                return None
            case character:
                message = f"invalid JSON literal starter: {character}"
                _fail(message)
