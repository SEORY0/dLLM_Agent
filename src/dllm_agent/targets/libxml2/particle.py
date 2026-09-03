"""Deterministic well-formed XML structured particles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Literal, override

type XmlField = Literal[
    "xml_version",
    "encoding",
    "namespace_uri",
    "root_name",
    "item_name",
    "attr_name",
    "attr_value",
    "text_value",
    "child_count",
    "whitespace",
]

FIELD_ORDER: Final[tuple[XmlField, ...]] = (
    "xml_version",
    "encoding",
    "namespace_uri",
    "root_name",
    "item_name",
    "attr_name",
    "attr_value",
    "text_value",
    "child_count",
    "whitespace",
)
_VERSIONS: Final = ('version="1.0"', "version='1.0'", 'version = "1.0"', "version = '1.0'")
_ENCODINGS: Final = ("UTF-8", "US-ASCII", "ISO-8859-1", "UTF-8")
_NAMESPACES: Final = ("urn:dllm:a", "urn:dllm:b", "urn:dllm:c", "urn:dllm:d")
_ROOTS: Final = ("root",)
_ITEMS: Final = ("item", "node", "case", "leaf")
_ATTR_NAMES: Final = ("id", "kind", "slot", "flag")
_ATTR_VALUES: Final = ("a", "b", "c", "d")
_TEXT_VALUES: Final = ("alpha", "beta", "gamma", "delta")
_CHILD_COUNTS: Final = (1, 2, 3, 4)
_WHITESPACE: Final = ("\n", "\n  ", " ", "\n\t")
_CARDINALITIES: Final[dict[str, int]] = {
    "xml_version": len(_VERSIONS),
    "encoding": len(_ENCODINGS),
    "namespace_uri": len(_NAMESPACES),
    "root_name": len(_ROOTS),
    "item_name": len(_ITEMS),
    "attr_name": len(_ATTR_NAMES),
    "attr_value": len(_ATTR_VALUES),
    "text_value": len(_TEXT_VALUES),
    "child_count": len(_CHILD_COUNTS),
    "whitespace": len(_WHITESPACE),
}
_SELECTOR_COUNT: Final = 10


@dataclass(frozen=True, slots=True)
class StructuredTargetError(Exception):
    """Raised when selectors cannot form a typed XML particle."""

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
class _XmlValues:
    version_token: str
    encoding: str
    item_name: str
    namespace_uri: str
    attr_name: str
    attr_value: str
    text_value: str
    child_count: int
    whitespace: str


@dataclass(frozen=True, slots=True)
class Libxml2StructuredTarget:
    """Structured libxml2 adapter producing well-formed XML documents."""

    target_id: Literal["libxml2"] = "libxml2"
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
        """Materialize selectors into a well-formed XML document."""
        values = _values(require_ten_selectors(selectors))
        children = values.whitespace.join(
            _child_xml(values, index) for index in range(values.child_count)
        )
        document = (
            f'<?xml {values.version_token} encoding="{values.encoding}"?>'
            f'{values.whitespace}<root xmlns="{values.namespace_uri}">'
            f"{values.whitespace}{children}{values.whitespace}</root>"
        ).encode()
        spans = (
            _span(document, values.version_token.encode(), "xml_version"),
            _span(document, values.encoding.encode(), "encoding"),
            _span(document, values.namespace_uri.encode(), "namespace_uri"),
            _span(document, b"root", "root_name"),
            _span(document, values.item_name.encode(), "item_name"),
            _span(document, values.attr_name.encode(), "attr_name"),
            _span(document, values.attr_value.encode(), "attr_value"),
            _span(document, values.text_value.encode(), "text_value"),
            _last_span(document, str(values.child_count - 1).encode(), "child_count"),
            _span(document, values.whitespace.encode(), "whitespace"),
        )
        return snapshot_for(document, document.split(b">", maxsplit=1)[0] + b">", spans)


def _values(selectors: tuple[int, ...]) -> _XmlValues:
    return _XmlValues(
        version_token=selector_value(_VERSIONS, selectors[0]),
        encoding=selector_value(_ENCODINGS, selectors[1]),
        item_name=selector_value(_ITEMS, selectors[4]),
        namespace_uri=selector_value(_NAMESPACES, selectors[2]),
        attr_name=selector_value(_ATTR_NAMES, selectors[5]),
        attr_value=selector_value(_ATTR_VALUES, selectors[6]),
        text_value=selector_value(_TEXT_VALUES, selectors[7]),
        child_count=selector_value(_CHILD_COUNTS, selectors[8]),
        whitespace=selector_value(_WHITESPACE, selectors[9]),
    )


def _child_xml(values: _XmlValues, index: int) -> str:
    return (
        f'<{values.item_name} {values.attr_name}="{values.attr_value}{index}">'
        f"{values.text_value}</{values.item_name}>"
    )


def _span(document: bytes, marker: bytes, field: str) -> FieldSpan:
    start = document.index(marker)
    return FieldSpan(field, start, start + len(marker))


def _last_span(document: bytes, marker: bytes, field: str) -> FieldSpan:
    start = document.rindex(marker)
    return FieldSpan(field, start, start + len(marker))


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
