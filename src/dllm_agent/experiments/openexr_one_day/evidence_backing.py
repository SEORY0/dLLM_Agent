"""Dereferenceable byte evidence for OpenEXR one-day outputs."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Self, override
from uuid import uuid4

from pydantic import Field, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256


class EvidenceBackingError(Exception):
    """Raised when backing evidence is missing, unsafe, or content-mismatched."""

    reason: str

    def __init__(self, reason: str) -> None:  # noqa: D107
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class ArtifactBackingEvidence(FrozenModel):
    """Canonical relative path, exact byte size, and SHA-256 for a stored blob."""

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def _canonical_relative_path(self) -> Self:
        _ = _canonical_relative(self.relative_path)
        return self


def persist_backing_bytes(
    root: Path, relative_path: str, content: bytes
) -> ArtifactBackingEvidence:
    """Atomically persist bytes below root and return their dereferenceable evidence."""
    path = _backing_path(root, relative_path)
    if path.exists() or path.is_symlink():
        evidence = ArtifactBackingEvidence(
            relative_path=relative_path,
            size_bytes=len(content),
            sha256=sha256_bytes(content),
        )
        _ = validate_backing_file(root, evidence)
        return evidence
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    _require_below(root.resolve(), path.parent.resolve())
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            _ = handle.write(content)
        _ = temporary.replace(path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            _ = temporary.unlink()
    return ArtifactBackingEvidence(
        relative_path=relative_path,
        size_bytes=len(content),
        sha256=sha256_bytes(content),
    )


def validate_backing_file(root: Path, evidence: ArtifactBackingEvidence) -> bytes:
    """Reopen backing evidence, reject unsafe files, and verify size plus SHA-256."""
    path = _backing_path(root, evidence.relative_path)
    if path.is_symlink():
        raise EvidenceBackingError(reason="backing evidence path is a symlink")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvidenceBackingError(reason="backing evidence file is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise EvidenceBackingError(reason="backing evidence is not a regular file")
    content = path.read_bytes()
    if len(content) != evidence.size_bytes or sha256_bytes(content) != evidence.sha256:
        raise EvidenceBackingError(reason="backing evidence content mismatch")
    return content


def _backing_path(root: Path, relative_path: str) -> Path:
    relative = _canonical_relative(relative_path)
    resolved_root = root.resolve()
    path = resolved_root / relative
    _reject_symlink_parents(resolved_root, relative)
    _require_below(resolved_root, path.resolve(strict=False))
    return path


def _reject_symlink_parents(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise EvidenceBackingError(reason="backing evidence path contains a symlink")


def _canonical_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise EvidenceBackingError(reason="backing evidence path must be canonical relative")
    if path == Path() or not path.parts:
        raise EvidenceBackingError(reason="backing evidence path must be canonical relative")
    return path


def _require_below(root: Path, path: Path) -> None:
    try:
        _ = path.relative_to(root)
    except ValueError as error:
        raise EvidenceBackingError(reason="backing evidence path escapes root") from error
