# ruff: noqa: D100,D101,D103

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import NoReturn

type JsonArray = list[JsonValue]
type JsonObject = dict[str, JsonValue]
type JsonValue = bool | int | float | str | JsonArray | JsonObject | None


class JsonParseError(Exception):
    pass


def parse_json_object(raw: bytes) -> JsonObject:
    value = _Parser(raw.decode("utf-8")).parse()
    if isinstance(value, dict):
        return value
    return _fail("expected JSON object")


def require_object(payload: JsonObject, key: str) -> JsonObject:
    value = _required(payload, key)
    if isinstance(value, dict):
        return value
    return _fail(f"{key}: expected object")


def require_array(payload: JsonObject, key: str) -> JsonArray:
    value = _required(payload, key)
    if isinstance(value, list):
        return value
    return _fail(f"{key}: expected array")


def require_str(payload: JsonObject, key: str) -> str:
    value = _required(payload, key)
    if isinstance(value, str):
        return value
    return _fail(f"{key}: expected string")


def require_int(payload: JsonObject, key: str) -> int:
    value = _required(payload, key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _fail(f"{key}: expected integer")


def require_bool(payload: JsonObject, key: str) -> bool:
    value = _required(payload, key)
    if isinstance(value, bool):
        return value
    return _fail(f"{key}: expected boolean")


def require_number(payload: JsonObject, key: str) -> float:
    value = _required(payload, key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return _fail(f"{key}: expected number")


def require_number_or_none(payload: JsonObject, key: str) -> float | None:
    value = _required(payload, key)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return _fail(f"{key}: expected number or null")


def stable_json(payload: JsonObject) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def strings(values: tuple[str, ...]) -> JsonArray:
    result: JsonArray = []
    result.extend(values)
    return result


def ints(values: tuple[int, ...]) -> JsonArray:
    result: JsonArray = []
    result.extend(values)
    return result


def arrays_equal(left: JsonValue, right: JsonValue) -> bool:
    return left == right


def _required(payload: JsonObject, key: str) -> JsonValue:
    try:
        return payload[key]
    except KeyError:
        _fail(f"{key}: missing")


def _fail(message: str) -> NoReturn:
    raise JsonParseError(message)


@dataclass(slots=True)
class _Parser:
    text: str
    index: int = 0

    def parse(self) -> JsonValue:
        value = self._value()
        self._space()
        if self.index != len(self.text):
            self._die("trailing data")
        return value

    def _value(self) -> JsonValue:  # noqa: PLR0911 - JSON token dispatch is explicit.
        self._space()
        char = self._peek()
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        if char == '"':
            return self._string()
        if char == "t":
            self._literal("true")
            return True
        if char == "f":
            self._literal("false")
            return False
        if char == "n":
            self._literal("null")
            return None
        return self._number()

    def _object(self) -> JsonObject:
        self._take("{")
        result: JsonObject = {}
        self._space()
        if self._peek() == "}":
            self._take("}")
            return result
        while True:
            key = self._string()
            self._space()
            self._take(":")
            result[key] = self._value()
            self._space()
            if self._peek() == "}":
                self._take("}")
                return result
            self._take(",")
            self._space()

    def _array(self) -> JsonArray:
        self._take("[")
        result: JsonArray = []
        self._space()
        if self._peek() == "]":
            self._take("]")
            return result
        while True:
            result.append(self._value())
            self._space()
            if self._peek() == "]":
                self._take("]")
                return result
            self._take(",")
            self._space()

    def _string(self) -> str:
        self._take('"')
        chars: list[str] = []
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == '"':
                return "".join(chars)
            if char == "\\":
                chars.append(self._escape())
            else:
                chars.append(char)
        return self._die("unterminated string")

    def _escape(self) -> str:
        char = self._next()
        escapes = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if char in escapes:
            return escapes[char]
        return self._die("unsupported string escape")

    def _number(self) -> int | float:
        start = self.index
        if self._peek() == "-":
            self.index += 1
        while self.index < len(self.text) and self.text[self.index].isdigit():
            self.index += 1
        if self.index < len(self.text) and self.text[self.index] == ".":
            self.index += 1
            while self.index < len(self.text) and self.text[self.index].isdigit():
                self.index += 1
            return float(self.text[start : self.index])
        return int(self.text[start : self.index])

    def _literal(self, value: str) -> None:
        if not self.text.startswith(value, self.index):
            self._die(f"expected {value}")
        self.index += len(value)

    def _space(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \n\r\t":
            self.index += 1

    def _take(self, expected: str) -> None:
        if self._peek() != expected:
            self._die(f"expected {expected}")
        self.index += 1

    def _peek(self) -> str:
        if self.index >= len(self.text):
            self._die("unexpected end")
        return self.text[self.index]

    def _next(self) -> str:
        char = self._peek()
        self.index += 1
        return char

    def _die(self, detail: str) -> NoReturn:
        raise JsonParseError(detail)
