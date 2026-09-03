"""Exact C0 corpus source loading for the OpenEXR one-day campaign."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256


class C0SourceError(ValueError):
    """Raised when exact C0 source evidence is unsafe or mismatched."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the C0 evidence failure reason."""
        self.reason = reason
        super().__init__(reason)


class ExactC0Source(FrozenModel):
    """Hash-bound local hex fixture for the baseline C0 corpus seed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    schema_version: Literal["dllm-openexr-c0-source/v1"] = "dllm-openexr-c0-source/v1"
    path: Path
    sha256: Sha256

    @model_validator(mode="after")
    def _safe_path(self) -> Self:
        absolute = self.path if self.path.is_absolute() else Path.cwd() / self.path
        for candidate in (*tuple(reversed(absolute.parents)), absolute):
            if candidate.is_symlink():
                raise C0SourceError(reason="C0 source path contains symlink")
        if self.path.is_symlink():
            raise C0SourceError(reason="C0 source may not be a symlink")
        if self.path.is_absolute():
            return self
        root = Path.cwd().resolve()
        resolved = (root / self.path).resolve()
        try:
            _ = resolved.relative_to(root)
        except ValueError as error:
            raise C0SourceError(reason="C0 source path escapes repository") from error
        object.__setattr__(self, "path", resolved)
        return self


class ExactC0Bytes(FrozenModel):
    """Verified exact C0 bytes and digest."""

    content: bytes = Field(min_length=1)
    sha256: Sha256


def load_exact_c0_source(source: ExactC0Source) -> ExactC0Bytes:
    """Decode exact C0 bytes from a hex fixture and verify the configured hash."""
    try:
        raw = source.path.read_text(encoding="ascii").strip()
        content = bytes.fromhex(raw)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise C0SourceError(reason="invalid C0 source fixture") from error
    digest = sha256_bytes(content)
    if digest != source.sha256:
        raise C0SourceError(reason="C0 sha256 mismatch")
    return ExactC0Bytes(content=content, sha256=digest)
