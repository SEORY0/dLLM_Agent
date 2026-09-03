"""Runtime JSON fallback decoder tests."""

from __future__ import annotations

import pytest
from tools.runtime.selective_revision_decode import coerce_json_value, decode_json_value


def test_decode_json_value_parses_nested_json_without_pydantic() -> None:
    # Given
    line = '{"outer":[1,true,null,{"escaped":"A"}]}'

    # When
    decoded = decode_json_value(line)

    # Then
    assert decoded == {"outer": [1, True, None, {"escaped": "A"}]}


def test_decode_json_value_rejects_nested_invalid_escape() -> None:
    # Given
    line = '{"outer":["\\q"]}'

    # When / Then
    with pytest.raises(ValueError, match="invalid JSON escape"):
        _ = decode_json_value(line)


def test_coerce_json_value_rejects_unsupported_nested_value() -> None:
    # Given
    unsupported: object = {"outer": {1, 2}}

    # When / Then
    with pytest.raises(ValueError, match="unsupported JSON value"):
        _ = coerce_json_value(unsupported)
