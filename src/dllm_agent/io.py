"""Typed JSON and file boundaries."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ValidationError

from dllm_agent.errors import BoundaryError


def sha256_bytes(data: bytes) -> str:
    """Return lowercase SHA-256 for bytes."""
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path, boundary: str) -> bytes:
    """Read a required regular file."""
    try:
        if not path.is_file():
            raise BoundaryError(boundary, path, "file does not exist")
        return path.read_bytes()
    except OSError as error:
        raise BoundaryError(boundary, path, error.strerror or "cannot read file") from error


def parse_json[SchemaT: BaseModel](path: Path, model: type[SchemaT], boundary: str) -> SchemaT:
    """Parse JSON bytes once into a frozen schema."""
    raw = read_bytes(path, boundary)
    try:
        return model.model_validate_json(raw)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first["loc"])
        raise BoundaryError(boundary, path, f"{location}: {first['msg']}") from error


def write_model(path: Path, model: BaseModel) -> None:
    """Atomically create a JSON model output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _ = temporary.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = temporary.replace(path)
