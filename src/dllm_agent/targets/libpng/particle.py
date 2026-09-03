"""Deterministic valid PNG structured particles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from struct import pack
from types import MappingProxyType
from typing import Final, Literal, override
from zlib import compress, crc32

type PngField = Literal[
    "width",
    "height",
    "bit_depth",
    "color_type",
    "compression_method",
    "filter_method",
    "interlace_method",
    "text_key",
    "text_value",
    "image_payload",
]

FIELD_ORDER: Final[tuple[PngField, ...]] = (
    "width",
    "height",
    "bit_depth",
    "color_type",
    "compression_method",
    "filter_method",
    "interlace_method",
    "text_key",
    "text_value",
    "image_payload",
)
_WIDTHS: Final = (1, 2, 3, 4)
_HEIGHTS: Final = (1, 2, 4, 8)
_BIT_DEPTHS: Final = (8,)
_COLOR_TYPES: Final = (2,)
_PNG_METHOD: Final = (0,)
_INTERLACE: Final = (0,)
_TEXT_KEYS: Final = (b"k", b"note", b"seed", b"case")
_TEXT_VALUES: Final = (b"a", b"b", b"c", b"d")
_PAYLOAD_BYTES: Final = (0, 85, 170, 255)
_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_SELECTOR_COUNT: Final = 10
_CARDINALITIES: Final[dict[str, int]] = dict.fromkeys(FIELD_ORDER, 1)
_CARDINALITIES.update(
    {
        "width": len(_WIDTHS),
        "height": len(_HEIGHTS),
        "text_key": len(_TEXT_KEYS),
        "text_value": len(_TEXT_VALUES),
        "image_payload": len(_PAYLOAD_BYTES),
    },
)


@dataclass(frozen=True, slots=True)
class StructuredTargetError(Exception):
    """Raised when selectors cannot form a typed PNG particle."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the error reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Selector metadata for one structured particle field."""

    name: str
    cardinality: int
    mutable: bool


@dataclass(frozen=True, slots=True)
class FieldSpan:
    """Byte range controlled by one structured particle field."""

    field: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    """Immutable materialized input bytes with hash and span evidence."""

    content: bytes
    sha256: str
    anchor_sha256: str
    field_spans: tuple[FieldSpan, ...]


@dataclass(frozen=True, slots=True)
class _PngValues:
    width: int
    height: int
    text_key: bytes
    text_value: bytes
    payload_byte: int


@dataclass(frozen=True, slots=True)
class LibpngStructuredTarget:
    """Structured libpng adapter producing minimal valid PNG images."""

    target_id: Literal["libpng"] = "libpng"
    field_order: tuple[str, ...] = FIELD_ORDER
    fields: tuple[FieldMetadata, ...] = tuple(
        FieldMetadata(
            name=field,
            cardinality=_CARDINALITIES[field],
            mutable=_CARDINALITIES[field] > 1,
        )
        for field in FIELD_ORDER
    )
    mutable_fields: tuple[str, ...] = tuple(field.name for field in fields if field.mutable)
    cardinalities: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType(_CARDINALITIES),
    )

    def materialize(self, selectors: tuple[int, ...]) -> MaterializedSnapshot:
        """Materialize selectors into a syntactically valid PNG."""
        values = _values(require_ten_selectors(selectors))
        content = bytearray(_SIGNATURE)
        ihdr_start = _append_chunk(
            content,
            b"IHDR",
            pack(">IIBBBBB", values.width, values.height, 8, 2, 0, 0, 0),
        )
        text_payload = values.text_key + b"\x00" + values.text_value
        text_start = _append_chunk(content, b"tEXt", text_payload)
        raw = b"".join(
            bytes((0,))
            + bytes((values.payload_byte, values.payload_byte, values.payload_byte)) * values.width
            for _ in range(values.height)
        )
        compressed = compress(raw)
        idat_start = _append_chunk(content, b"IDAT", compressed)
        _ = _append_chunk(content, b"IEND", b"")
        spans = (
            FieldSpan("width", ihdr_start, ihdr_start + 4),
            FieldSpan("height", ihdr_start + 4, ihdr_start + 8),
            FieldSpan("bit_depth", ihdr_start + 8, ihdr_start + 9),
            FieldSpan("color_type", ihdr_start + 9, ihdr_start + 10),
            FieldSpan("compression_method", ihdr_start + 10, ihdr_start + 11),
            FieldSpan("filter_method", ihdr_start + 11, ihdr_start + 12),
            FieldSpan("interlace_method", ihdr_start + 12, ihdr_start + 13),
            FieldSpan("text_key", text_start, text_start + len(values.text_key)),
            FieldSpan(
                "text_value",
                text_start + len(values.text_key) + 1,
                text_start + len(text_payload),
            ),
            FieldSpan("image_payload", idat_start, idat_start + len(compressed)),
        )
        return snapshot_for(bytes(content), bytes(content[: ihdr_start + 13]), spans)


def _values(selectors: tuple[int, ...]) -> _PngValues:
    return _PngValues(
        width=selector_value(_WIDTHS, selectors[0]),
        height=selector_value(_HEIGHTS, selectors[1]),
        text_key=selector_value(_TEXT_KEYS, selectors[7]),
        text_value=selector_value(_TEXT_VALUES, selectors[8]),
        payload_byte=selector_value(_PAYLOAD_BYTES, selectors[9]),
    )


def require_ten_selectors(selectors: tuple[int, ...]) -> tuple[int, ...]:
    """Parse the fixed ten-slot structured particle boundary."""
    if len(selectors) != _SELECTOR_COUNT:
        raise StructuredTargetError(reason=f"expected 10 selectors, got {len(selectors)}")
    return selectors


def selector_value[T](catalog: tuple[T, ...], selector: int) -> T:
    """Select a catalog value using modulo selector routing."""
    return catalog[selector % len(catalog)]


def snapshot_for(
    content: bytes,
    anchor: bytes,
    spans: tuple[FieldSpan, ...],
) -> MaterializedSnapshot:
    """Build an immutable snapshot with deterministic hash evidence."""
    return MaterializedSnapshot(
        content=content,
        sha256=sha256(content).hexdigest(),
        anchor_sha256=sha256(anchor).hexdigest(),
        field_spans=spans,
    )


def _append_chunk(content: bytearray, kind: bytes, payload: bytes) -> int:
    start = len(content) + 8
    content.extend(pack(">I", len(payload)))
    content.extend(kind)
    content.extend(payload)
    content.extend(pack(">I", crc32(kind + payload) & 0xFFFFFFFF))
    return start
