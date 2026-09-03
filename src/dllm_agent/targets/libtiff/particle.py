"""Deterministic little-endian TIFF structured particles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from struct import pack
from types import MappingProxyType
from typing import Final, Literal, override

type TiffField = Literal[
    "image_width",
    "image_length",
    "bits_per_sample",
    "compression",
    "photometric",
    "strip_offsets",
    "samples_per_pixel",
    "rows_per_strip",
    "strip_byte_counts",
    "image_payload",
]

FIELD_ORDER: Final[tuple[TiffField, ...]] = (
    "image_width",
    "image_length",
    "bits_per_sample",
    "compression",
    "photometric",
    "strip_offsets",
    "samples_per_pixel",
    "rows_per_strip",
    "strip_byte_counts",
    "image_payload",
)
_WIDTHS: Final = (1, 2, 3, 4)
_HEIGHTS: Final = (1, 2, 4, 8)
_BITS: Final = (8, 16)
_PHOTOMETRIC: Final = (0, 1)
_PAYLOAD_BYTES: Final = (0, 85, 170, 255)
_CARDINALITIES: Final[dict[str, int]] = dict.fromkeys(FIELD_ORDER, 1)
_CARDINALITIES.update(
    {
        "image_width": len(_WIDTHS),
        "image_length": len(_HEIGHTS),
        "bits_per_sample": len(_BITS),
        "photometric": len(_PHOTOMETRIC),
        "image_payload": len(_PAYLOAD_BYTES),
    },
)
_IFD_OFFSET: Final = 8
_ENTRY_COUNT: Final = 10
_SELECTOR_COUNT: Final = 10
_PIXEL_OFFSET: Final = _IFD_OFFSET + 2 + (_ENTRY_COUNT * 12) + 4


@dataclass(frozen=True, slots=True)
class StructuredTargetError(Exception):
    """Raised when selectors cannot form a typed TIFF particle."""

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
class _TiffValues:
    width: int
    height: int
    bits_per_sample: int
    photometric: int


@dataclass(frozen=True, slots=True)
class LibTiffEntry:
    """Packed TIFF IFD entry with source field name."""

    field: str | None
    tag: int
    value_type: int
    count: int
    value: int


@dataclass(frozen=True, slots=True)
class LibtiffStructuredTarget:
    """Structured libtiff adapter producing minimal uncompressed TIFF images."""

    target_id: Literal["libtiff"] = "libtiff"
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
        """Materialize selectors into a little-endian uncompressed TIFF."""
        parsed_selectors = require_ten_selectors(selectors)
        values = _values(parsed_selectors)
        bytes_per_sample = values.bits_per_sample // 8
        byte_count = values.width * values.height * bytes_per_sample
        entries = _entries(values, byte_count)
        content = bytearray(b"II*\x00" + pack("<I", _IFD_OFFSET) + pack("<H", len(entries)))
        spans: list[FieldSpan] = []
        for entry in entries:
            entry_start = len(content)
            content.extend(pack("<HHII", entry.tag, entry.value_type, entry.count, entry.value))
            if entry.field is not None:
                spans.append(FieldSpan(entry.field, entry_start + 8, entry_start + 12))
        content.extend(pack("<I", 0))
        payload_start = len(content)
        payload_byte = selector_value(_PAYLOAD_BYTES, parsed_selectors[9])
        content.extend(bytes((payload_byte,)) * byte_count)
        spans.append(FieldSpan("image_payload", payload_start, len(content)))
        return snapshot_for(bytes(content), bytes(content[:_PIXEL_OFFSET]), tuple(spans))


def _values(selectors: tuple[int, ...]) -> _TiffValues:
    return _TiffValues(
        width=selector_value(_WIDTHS, selectors[0]),
        height=selector_value(_HEIGHTS, selectors[1]),
        bits_per_sample=selector_value(_BITS, selectors[2]),
        photometric=selector_value(_PHOTOMETRIC, selectors[4]),
    )


def _entries(values: _TiffValues, byte_count: int) -> tuple[LibTiffEntry, ...]:
    return (
        LibTiffEntry("image_width", 256, 4, 1, values.width),
        LibTiffEntry("image_length", 257, 4, 1, values.height),
        LibTiffEntry("bits_per_sample", 258, 3, 1, values.bits_per_sample),
        LibTiffEntry("compression", 259, 3, 1, 1),
        LibTiffEntry("photometric", 262, 3, 1, values.photometric),
        LibTiffEntry("strip_offsets", 273, 4, 1, _PIXEL_OFFSET),
        LibTiffEntry("samples_per_pixel", 277, 3, 1, 1),
        LibTiffEntry("rows_per_strip", 278, 4, 1, values.height),
        LibTiffEntry("strip_byte_counts", 279, 4, 1, byte_count),
        LibTiffEntry(None, 284, 3, 1, 1),
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
