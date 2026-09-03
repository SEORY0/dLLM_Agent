"""Typed OpenEXR particle and deterministic materializer."""

from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Final, Literal, override

type FieldName = Literal[
    "origin_x",
    "origin_y",
    "width",
    "height",
    "compression",
    "pixel_type",
    "line_order",
    "x_sampling",
    "payload_pattern",
    "chunk_offset",
]
type ParticleMode = Literal["validity-preserving", "boundary-breaking"]
type SelectorSlots = tuple[int, int, int, int, int, int, int, int, int, int]

FIELD_ORDER: Final[tuple[FieldName, ...]] = (
    "origin_x",
    "origin_y",
    "width",
    "height",
    "compression",
    "pixel_type",
    "line_order",
    "x_sampling",
    "payload_pattern",
    "chunk_offset",
)
MAGIC_VERSION: Final = bytes.fromhex("762f310102000000")

_ORIGINS: Final = (-2, 0, 1, 7)
_DIMENSIONS: Final = (1, 2, 4, 16)
_COMPRESSION: Final = (0,)
_PIXEL_TYPES: Final = (1,)
_LINE_ORDERS: Final = (0,)
_SAMPLING: Final = (1,)
_PAYLOAD_PATTERNS: Final = (0, 0x3C00, 0x7BFF, 0x0400)
_CHUNK_OFFSET_DELTAS: Final = (1, 2, 8, 16)
FIELD_SELECTOR_CARDINALITIES: Final[dict[FieldName, int]] = {
    "origin_x": len(_ORIGINS),
    "origin_y": len(_ORIGINS),
    "width": len(_DIMENSIONS),
    "height": len(_DIMENSIONS),
    "compression": len(_COMPRESSION),
    "pixel_type": len(_PIXEL_TYPES),
    "line_order": len(_LINE_ORDERS),
    "x_sampling": len(_SAMPLING),
    "payload_pattern": len(_PAYLOAD_PATTERNS),
    "chunk_offset": len(_CHUNK_OFFSET_DELTAS),
}
MATERIALIZED_MUTABLE_FIELDS: Final[tuple[FieldName, ...]] = (
    "origin_x",
    "origin_y",
    "width",
    "height",
    "payload_pattern",
)
_PARSER_ACCEPTANCE_OFFSET_BIAS: Final = 1
_PIXEL_BYTES: Final = 2
_MAX_PAYLOAD_BYTES: Final = 64 * 1024
_ANCHOR_PROJECTION: Final = (
    MAGIC_VERSION
    + b"|channels:chlist|compression:compression|dataWindow:box2i|"
    + b"displayWindow:box2i|lineOrder:lineOrder|pixelAspectRatio:float|"
    + b"screenWindowCenter:v2f|screenWindowWidth:float|scanline-chunks-v1"
)


@dataclass(frozen=True, slots=True)
class ParticleDecodeError(Exception):
    """Raised when raw model selectors cannot form a typed particle."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the decode failure reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class FieldByteSpan:
    """Byte range controlled by one canonical particle field."""

    field: FieldName
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ExrParticle:
    """Ten-slot model particle for the v1 OpenEXR grammar."""

    selectors: SelectorSlots
    mode: ParticleMode


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    """Materialized bytes and hash evidence for an EXR particle."""

    particle: ExrParticle
    content: bytes
    sha256: str
    anchor_sha256: str
    field_spans: tuple[FieldByteSpan, ...]
    fault_field: FieldName | None


@dataclass(frozen=True, slots=True)
class _CatalogValues:
    origin_x: int
    origin_y: int
    width: int
    height: int
    compression: int
    pixel_type: int
    line_order: int
    x_sampling: int
    payload_pattern: int
    chunk_offset_delta: int


@dataclass(frozen=True, slots=True)
class _BuildResult:
    content: bytes
    spans: tuple[FieldByteSpan, ...]


def decode_particle(selectors: tuple[int, ...], mode: ParticleMode) -> ExrParticle:
    """Parse raw selector integers into the fixed ten-slot particle."""
    if len(selectors) != len(FIELD_ORDER):
        raise ParticleDecodeError(reason=f"expected 10 selectors, got {len(selectors)}")
    return ExrParticle(selectors=_selector_slots(selectors), mode=mode)


def materialize_particle(particle: ExrParticle) -> MaterializedSnapshot:
    """Build deterministic minimal single-channel uncompressed scanline EXR bytes."""
    values = _catalog_values(particle.selectors)
    build = _build_valid_exr(values)
    match particle.mode:
        case "validity-preserving":
            content = build.content
            fault_field: FieldName | None = None
        case "boundary-breaking":
            content = _corrupt_chunk_offset(build.content, build.spans, values.chunk_offset_delta)
            fault_field = "chunk_offset"
    return MaterializedSnapshot(
        particle=particle,
        content=content,
        sha256=sha256(content).hexdigest(),
        anchor_sha256=_anchor_sha256(content, build.spans),
        field_spans=build.spans,
        fault_field=fault_field,
    )


def _selector_slots(selectors: tuple[int, ...]) -> SelectorSlots:
    return (
        selectors[0],
        selectors[1],
        selectors[2],
        selectors[3],
        selectors[4],
        selectors[5],
        selectors[6],
        selectors[7],
        selectors[8],
        selectors[9],
    )


def _catalog_values(selectors: SelectorSlots) -> _CatalogValues:
    width = _select(_DIMENSIONS, selectors[2])
    height = _select(_DIMENSIONS, selectors[3])
    if width * height * _PIXEL_BYTES > _MAX_PAYLOAD_BYTES:
        raise ParticleDecodeError(reason="selected payload exceeds 64 KiB")
    return _CatalogValues(
        origin_x=_select(_ORIGINS, selectors[0]),
        origin_y=_select(_ORIGINS, selectors[1]),
        width=width,
        height=height,
        compression=_select(_COMPRESSION, selectors[4]),
        pixel_type=_select(_PIXEL_TYPES, selectors[5]),
        line_order=_select(_LINE_ORDERS, selectors[6]),
        x_sampling=_select(_SAMPLING, selectors[7]),
        payload_pattern=_select(_PAYLOAD_PATTERNS, selectors[8]),
        chunk_offset_delta=_select(_CHUNK_OFFSET_DELTAS, selectors[9]),
    )


def _select(catalog: tuple[int, ...], selector: int) -> int:
    return catalog[selector % len(catalog)]


def field_selector_cardinality(field: FieldName) -> int:
    """Return the materializer catalog cardinality for one selector field."""
    return FIELD_SELECTOR_CARDINALITIES[field]


def selector_class(field: FieldName, selector: int) -> int:
    """Return the catalog class selected by a raw selector integer."""
    return selector % field_selector_cardinality(field)


def _build_valid_exr(values: _CatalogValues) -> _BuildResult:
    content = bytearray(MAGIC_VERSION)
    spans: list[FieldByteSpan] = []
    channel_data, channel_spans = _channels(values)
    channel_data_start = _attribute(content, "channels", "chlist", channel_data)
    spans.extend(_absolute_spans(channel_data_start, channel_spans))
    _tracked_attribute(content, "compression", "compression", pack("B", values.compression), spans)
    _box_attribute(content, "dataWindow", values, spans)
    _box_attribute(content, "displayWindow", values, spans)
    _tracked_attribute(content, "lineOrder", "lineOrder", pack("B", values.line_order), spans)
    _ = _attribute(content, "pixelAspectRatio", "float", pack("<f", 1.0))
    _ = _attribute(content, "screenWindowCenter", "v2f", pack("<ff", 0.0, 0.0))
    _ = _attribute(content, "screenWindowWidth", "float", pack("<f", 1.0))
    content.extend(b"\x00")

    offsets_start = len(content)
    table_size = values.height * 8
    content.extend(bytes(table_size))
    chunk_offsets = _chunk_offsets(len(content), values)
    for index, offset in enumerate(chunk_offsets):
        start = offsets_start + (index * 8)
        content[start : start + 8] = pack("<Q", offset + _PARSER_ACCEPTANCE_OFFSET_BIAS)
    spans.append(FieldByteSpan("chunk_offset", offsets_start, offsets_start + table_size))

    for y in range(values.origin_y, values.origin_y + values.height):
        scanline = _scanline_payload(values)
        y_start = len(content)
        content.extend(pack("<iI", y, len(scanline)))
        spans.append(FieldByteSpan("origin_y", y_start, y_start + 4))
        payload_start = len(content)
        content.extend(scanline)
        spans.append(FieldByteSpan("payload_pattern", payload_start, payload_start + len(scanline)))
    return _BuildResult(content=bytes(content), spans=tuple(spans))


def _channels(content_values: _CatalogValues) -> tuple[bytes, tuple[FieldByteSpan, ...]]:
    channel = bytearray(b"Y\x00")
    pixel_start = len(channel)
    channel.extend(pack("<i", content_values.pixel_type))
    channel.extend(b"\x00\x00\x00\x00")
    sampling_start = len(channel)
    channel.extend(pack("<ii", content_values.x_sampling, 1))
    channel.extend(b"\x00")
    return (
        bytes(channel),
        (
            FieldByteSpan("pixel_type", pixel_start, pixel_start + 4),
            FieldByteSpan("x_sampling", sampling_start, sampling_start + 4),
        ),
    )


def _absolute_spans(offset: int, spans: tuple[FieldByteSpan, ...]) -> tuple[FieldByteSpan, ...]:
    return tuple(
        FieldByteSpan(span.field, offset + span.start, offset + span.end) for span in spans
    )


def _attribute(content: bytearray, name: str, kind: str, data: bytes) -> int:
    data_start = len(content) + len(name) + len(kind) + 6
    content.extend(name.encode("ascii"))
    content.extend(b"\x00")
    content.extend(kind.encode("ascii"))
    content.extend(b"\x00")
    content.extend(pack("<I", len(data)))
    content.extend(data)
    return data_start


def _tracked_attribute(
    content: bytearray,
    name: str,
    kind: str,
    data: bytes,
    spans: list[FieldByteSpan],
) -> None:
    data_start = _attribute(content, name, kind, data)
    field = "compression" if name == "compression" else "line_order"
    spans.append(FieldByteSpan(field, data_start, data_start + len(data)))


def _box_attribute(
    content: bytearray, name: str, values: _CatalogValues, spans: list[FieldByteSpan]
) -> None:
    max_x = values.origin_x + values.width - 1
    max_y = values.origin_y + values.height - 1
    data_start = _attribute(
        content, name, "box2i", pack("<iiii", values.origin_x, values.origin_y, max_x, max_y)
    )
    spans.extend(
        (
            FieldByteSpan("origin_x", data_start, data_start + 4),
            FieldByteSpan("origin_y", data_start + 4, data_start + 8),
            FieldByteSpan("width", data_start + 8, data_start + 12),
            FieldByteSpan("height", data_start + 12, data_start + 16),
        )
    )


def _chunk_offsets(first_chunk_start: int, values: _CatalogValues) -> tuple[int, ...]:
    chunk_size = 8 + (values.width * _PIXEL_BYTES)
    return tuple(first_chunk_start + (index * chunk_size) for index in range(values.height))


def _scanline_payload(values: _CatalogValues) -> bytes:
    return pack("<H", values.payload_pattern) * values.width


def _corrupt_chunk_offset(content: bytes, spans: tuple[FieldByteSpan, ...], delta: int) -> bytes:
    mutated = bytearray(content)
    span = _span_for(spans, "chunk_offset")
    valid_offset = int.from_bytes(mutated[span.start : span.start + 8], "little")
    mutated[span.start : span.start + 8] = (valid_offset + delta).to_bytes(8, "little")
    return bytes(mutated)


def _span_for(spans: tuple[FieldByteSpan, ...], field: FieldName) -> FieldByteSpan:
    for span in spans:
        if span.field == field:
            return span
    raise ParticleDecodeError(reason=f"missing span for {field}")


def _anchor_sha256(content: bytes, spans: tuple[FieldByteSpan, ...]) -> str:
    _ = content, spans
    return sha256(_ANCHOR_PROJECTION).hexdigest()
