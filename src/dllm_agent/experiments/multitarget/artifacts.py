"""Canonical JSON artifact persistence for multitarget experiments."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeGuard, override

from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type CanonicalValue = object

_TEMP_PREFIX = ".artifact-"
_TEMP_SUFFIX = ".tmp"


class DataclassInstance(Protocol):
    """Structural view shared by frozen dataclass artifact values."""

    __dataclass_fields__: Mapping[str, object]

    @override
    def __getattribute__(self, name: str) -> object:
        """Return one dataclass field value."""
        ...


@dataclass(slots=True)
class ArtifactPersistenceError(Exception):
    """Raised when canonical artifact persistence would be unsafe or ambiguous."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def canonical_json_bytes(value: CanonicalValue) -> bytes:
    """Serialize a supported artifact value into deterministic UTF-8 JSON bytes."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def write_json_artifact(
    *,
    experiment_root: Path,
    relative_path: Path,
    value: CanonicalValue,
) -> ArtifactEvidence:
    """Write one canonical JSON artifact below the experiment root exactly once."""
    payload = canonical_json_bytes(value)
    destination = _resolve_destination(experiment_root, relative_path)
    _write_atomic_if_absent(destination, payload)
    return ArtifactEvidence(
        path=destination.relative_to(experiment_root.resolve()).as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _canonicalize(value: CanonicalValue) -> JsonValue:
    if _is_dataclass_instance(value):
        result: JsonValue = _canonicalize_dataclass(value)
    elif isinstance(value, str | bool | int | float) or value is None:
        result = value
    elif isinstance(value, bytes):
        result = _canonicalize_bytes(value)
    elif isinstance(value, Path):
        result = value.as_posix()
    elif isinstance(value, Enum):
        result = _canonicalize_enum(value)
    elif _is_mapping(value):
        result = _canonicalize_mapping(value)
    elif isinstance(value, Sequence):
        result = [_canonicalize(item) for item in value]
    else:
        raise ArtifactPersistenceError(reason=f"unsupported artifact value: {type(value).__name__}")
    return result


def _is_dataclass_instance(value: CanonicalValue) -> TypeGuard[DataclassInstance]:
    return is_dataclass(value) and not isinstance(value, type)


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _canonicalize_dataclass(value: DataclassInstance) -> dict[str, JsonValue]:
    return {
        name: _canonicalize(value.__getattribute__(name))
        for name in sorted(value.__dataclass_fields__)
    }


def _canonicalize_bytes(value: bytes) -> dict[str, JsonValue]:
    return {
        "data_base64": base64.b64encode(value).decode("ascii"),
        "sha256": hashlib.sha256(value).hexdigest(),
        "size_bytes": len(value),
    }


def _canonicalize_enum(value: Enum) -> str:
    return value.name


def _canonicalize_mapping(value: Mapping[object, object]) -> dict[str, JsonValue]:
    items: list[tuple[str, object]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise ArtifactPersistenceError(reason="mapping keys must be strings")
        items.append((key, item))
    return {key: _canonicalize(item) for key, item in sorted(items, key=lambda pair: pair[0])}


def _resolve_destination(experiment_root: Path, relative_path: Path) -> Path:
    root = experiment_root.resolve()
    destination = (root / relative_path).resolve()
    try:
        _ = destination.relative_to(root)
    except ValueError as error:
        raise ArtifactPersistenceError(reason="artifact path is outside experiment root") from error
    if destination == root:
        raise ArtifactPersistenceError(reason="artifact path must name a file")
    return destination


def _write_atomic_if_absent(destination: Path, payload: bytes) -> None:
    if destination.exists():
        if destination.read_bytes() == payload:
            return
        raise ArtifactPersistenceError(reason=f"artifact already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=_TEMP_PREFIX,
            suffix=_TEMP_SUFFIX,
        )
        with os.fdopen(descriptor, "wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _link_or_accept_identical(Path(temp_name), destination, payload)
    except OSError as error:
        detail = error.strerror or "cannot write artifact"
        raise ArtifactPersistenceError(reason=f"{destination}: {detail}") from error
    finally:
        if temp_name:
            with suppress(OSError):
                Path(temp_name).unlink(missing_ok=True)


def _link_or_accept_identical(source: Path, destination: Path, payload: bytes) -> None:
    try:
        destination.hardlink_to(source)
    except FileExistsError:
        if destination.read_bytes() == payload:
            return
        raise ArtifactPersistenceError(reason=f"artifact already exists: {destination}") from None
