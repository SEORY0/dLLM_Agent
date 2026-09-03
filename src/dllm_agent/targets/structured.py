"""Structured Magma target particle contracts and loader."""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, Protocol, override

from dllm_agent.targets.libpng.particle import LibpngStructuredTarget
from dllm_agent.targets.libtiff.particle import LibtiffStructuredTarget
from dllm_agent.targets.libxml2.particle import Libxml2StructuredTarget

type TargetId = Literal["libpng", "libtiff", "libxml2"]

TARGET_IDS: Final[tuple[TargetId, ...]] = ("libpng", "libtiff", "libxml2")
SELECTOR_COUNT: Final = 10


@dataclass(frozen=True, slots=True)
class StructuredTargetError(Exception):
    """Raised when a structured target cannot be loaded or materialized."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the error reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class FieldSpan:
    """Byte range controlled by one structured particle field."""

    field: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Selector metadata for one structured particle field."""

    name: str
    cardinality: int
    mutable: bool


class FieldMetadataLike(Protocol):
    """Read-only structural metadata for one selector field."""

    @property
    def name(self) -> str:
        """Return the selector field name."""
        ...

    @property
    def cardinality(self) -> int:
        """Return the selector catalog cardinality."""
        ...

    @property
    def mutable(self) -> bool:
        """Return whether this field changes materialized bytes."""
        ...


class FieldSpanLike(Protocol):
    """Read-only structural byte span for one selector field."""

    @property
    def field(self) -> str:
        """Return the selector field name."""
        ...

    @property
    def start(self) -> int:
        """Return the inclusive span start offset."""
        ...

    @property
    def end(self) -> int:
        """Return the exclusive span end offset."""
        ...


class SnapshotLike(Protocol):
    """Read-only structural materialized snapshot contract."""

    @property
    def content(self) -> bytes:
        """Return materialized target bytes."""
        ...

    @property
    def sha256(self) -> str:
        """Return SHA-256 of the materialized target bytes."""
        ...

    @property
    def anchor_sha256(self) -> str:
        """Return SHA-256 of the target anchor projection."""
        ...

    @property
    def field_spans(self) -> tuple[FieldSpanLike, ...]:
        """Return byte spans for every structured selector field."""
        ...


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    """Immutable materialized input bytes with hash and span evidence."""

    content: bytes
    sha256: str
    anchor_sha256: str
    field_spans: tuple[FieldSpanLike, ...]


class StructuredTarget(Protocol):
    """Materializer contract shared by deterministic structured targets."""

    @property
    def target_id(self) -> TargetId:
        """Return the Magma target identifier."""
        ...

    @property
    def field_order(self) -> tuple[str, ...]:
        """Return the canonical ten-field selector order."""
        ...

    @property
    def fields(self) -> tuple[FieldMetadataLike, ...]:
        """Return field cardinality and mutability metadata."""
        ...

    @property
    def mutable_fields(self) -> tuple[str, ...]:
        """Return fields whose selector changes materialized bytes."""
        ...

    @property
    def cardinalities(self) -> Mapping[str, int]:
        """Return selector cardinalities by field name."""
        ...

    def materialize(self, selectors: tuple[int, ...]) -> SnapshotLike:
        """Materialize selectors into immutable target input bytes."""
        ...


def load_structured_target(target_id: str) -> StructuredTarget:
    """Load one deterministic structured Magma target adapter."""
    match target_id:
        case "libpng":
            return LibpngStructuredTarget()
        case "libtiff":
            return LibtiffStructuredTarget()
        case "libxml2":
            return Libxml2StructuredTarget()
        case _:
            raise StructuredTargetError(reason=f"unknown structured target: {target_id}")


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


def selector_value[T](catalog: tuple[T, ...], selector: int) -> T:
    """Select a catalog value using modulo selector routing."""
    return catalog[selector % len(catalog)]


def require_ten_selectors(selectors: tuple[int, ...]) -> tuple[int, ...]:
    """Parse the fixed ten-slot structured particle boundary."""
    if len(selectors) != SELECTOR_COUNT:
        raise StructuredTargetError(reason=f"expected 10 selectors, got {len(selectors)}")
    return selectors
