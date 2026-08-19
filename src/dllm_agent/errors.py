"""Typed application errors."""

from dataclasses import dataclass
from pathlib import Path
from typing import override


@dataclass(frozen=True, slots=True)
class BoundaryError(Exception):
    """An untrusted input failed parsing."""

    boundary: str
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        """Render a concise CLI-safe error."""
        return f"{self.boundary} boundary error: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ValidationSetupError(Exception):
    """The validator could not prepare the target."""

    detail: str

    @override
    def __str__(self) -> str:
        """Render a concise validator error."""
        return f"validator setup error: {self.detail}"
