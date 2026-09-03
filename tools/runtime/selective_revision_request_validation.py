"""Raw request validation helpers for selective revision."""

from __future__ import annotations

from typing import Final

from dllm_agent.targets.openexr.particle import FIELD_ORDER, MATERIALIZED_MUTABLE_FIELDS
from tools.runtime.selective_revision_json import JsonValue, WireProtocolError, integer, string

_HEX_DIGITS: Final = frozenset("0123456789abcdef")
_SHA256_LENGTH: Final = 64
_ALL_SLOT_BITS: Final = (1 << 10) - 1
_MUTABLE_INDEXES: Final = frozenset(
    FIELD_ORDER.index(field) for field in MATERIALIZED_MUTABLE_FIELDS
)


def require_positive(payload: dict[str, JsonValue], key: str) -> None:
    """Require a positive integer field."""
    if integer(payload, key) <= 0:
        raise WireProtocolError(reason=f"{key} must be positive")


def require_nonnegative(payload: dict[str, JsonValue], key: str) -> None:
    """Require a nonnegative integer field."""
    if integer(payload, key) < 0:
        raise WireProtocolError(reason=f"{key} must be nonnegative")


def require_sha256(payload: dict[str, JsonValue], key: str) -> None:
    """Require a lowercase SHA-256 field."""
    value = string(payload, key)
    if len(value) != _SHA256_LENGTH or any(character not in _HEX_DIGITS for character in value):
        raise WireProtocolError(reason=f"{key} must be lowercase sha256")


def require_bitmaps(payload: dict[str, JsonValue]) -> None:
    """Require valid complementary ten-slot remask/frozen bitmaps."""
    remask = integer(payload, "remask_bitmap")
    frozen = integer(payload, "frozen_bitmap")
    if remask <= 0 or remask > _ALL_SLOT_BITS:
        raise WireProtocolError(reason="remask_bitmap out of range")
    if frozen < 0 or frozen > _ALL_SLOT_BITS:
        raise WireProtocolError(reason="frozen_bitmap out of range")
    if remask & frozen:
        raise WireProtocolError(reason="remask and frozen bitmaps overlap")
    if (remask | frozen) != _ALL_SLOT_BITS:
        raise WireProtocolError(reason="remask and frozen bitmaps must cover every slot")
    if any(
        remask & (1 << index) for index in range(len(FIELD_ORDER)) if index not in _MUTABLE_INDEXES
    ):
        raise WireProtocolError(reason="remask selects non-materialized field")
