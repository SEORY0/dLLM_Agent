"""Append-only JSONL provenance ledger for multi-target experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from pathlib import Path

EMPTY_STREAM_SHA256: Final = "0" * 64

type JsonValue = (
    str | int | float | bool | tuple[JsonValue, ...] | list[JsonValue] | dict[str, JsonValue] | None
)
type JsonObject = Mapping[str, JsonValue]

_RECORD_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(slots=True)
class LedgerError(Exception):
    """Raised when ledger evidence cannot be appended or validated."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the fail-closed ledger reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    """Hash-only evidence for process argv and environment."""

    argv_sha256: str
    env_sha256: str
    env_keys: tuple[str, ...]

    def to_json(self) -> dict[str, JsonValue]:
        """Return a JSON-safe representation without environment values."""
        return {
            "argv_sha256": self.argv_sha256,
            "env_sha256": self.env_sha256,
            "env_keys": list(self.env_keys),
        }


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Content hash and byte size for one persisted artifact."""

    path: str
    sha256: str
    size_bytes: int

    def to_json(self) -> dict[str, JsonValue]:
        """Return a JSON-safe artifact evidence object."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class LedgerAppendResult:
    """Append result with the new sequence and stream hash."""

    sequence: int
    record_sha256: str
    stream_sha256: str


@dataclass(frozen=True, slots=True)
class LedgerValidation:
    """Validation summary for an existing ledger stream."""

    record_count: int
    final_sequence: int
    final_stream_sha256: str


def command_evidence(argv: tuple[str, ...], env: Mapping[str, str]) -> CommandEvidence:
    """Hash command inputs while exposing only environment key names."""
    env_items = tuple(sorted(env.items()))
    return CommandEvidence(
        argv_sha256=hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest(),
        env_sha256=_sha256_json(env_items),
        env_keys=tuple(key for key, _value in env_items),
    )


def artifact_evidence(path: Path) -> ArtifactEvidence:
    """Hash one artifact file and record its byte length."""
    try:
        content = path.read_bytes()
    except OSError as error:
        detail = error.strerror or "cannot read artifact"
        raise LedgerError(reason=f"{path}: {detail}") from error
    return ArtifactEvidence(
        path=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def append_event(
    path: Path,
    *,
    timestamp_utc: str,
    event_type: str,
    payload: JsonObject,
) -> LedgerAppendResult:
    """Append one canonical JSONL event after validating any existing stream."""
    prior = validate_ledger(path) if path.exists() else _empty_validation()
    record_without_hash: dict[str, JsonValue] = {
        "sequence": prior.final_sequence + 1,
        "timestamp_utc": timestamp_utc,
        "event_type": event_type,
        "payload": dict(payload),
        "previous_stream_sha256": prior.final_stream_sha256,
    }
    record_sha256 = _sha256_json(record_without_hash)
    stream_sha256 = _stream_sha256(prior.final_stream_sha256, record_sha256)
    record = {
        **record_without_hash,
        "record_sha256": record_sha256,
        "stream_sha256": stream_sha256,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            _ = handle.write(_canonical_json(record).decode() + "\n")
    except OSError as error:
        detail = error.strerror or "cannot append ledger"
        raise LedgerError(reason=f"{path}: {detail}") from error
    return LedgerAppendResult(
        sequence=prior.final_sequence + 1,
        record_sha256=record_sha256,
        stream_sha256=stream_sha256,
    )


def validate_ledger(path: Path) -> LedgerValidation:
    """Validate sequence, per-record hashes, and rolling stream hash."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        detail = error.strerror or "cannot read ledger"
        raise LedgerError(reason=f"{path}: {detail}") from error
    if raw == b"":
        return _empty_validation()
    if not raw.endswith(b"\n"):
        raise LedgerError(reason="truncated ledger: missing terminal newline")

    previous_stream = EMPTY_STREAM_SHA256
    final_sequence = -1
    for expected_sequence, line in enumerate(raw.splitlines()):
        record = _decode_record(line, expected_sequence)
        sequence = _required_int(record, "sequence", expected_sequence)
        if sequence != expected_sequence:
            raise LedgerError(reason=f"sequence gap at line {expected_sequence + 1}")

        expected_record_hash = _sha256_json(_record_without_hashes(record))
        if _required_text(record, "record_sha256", expected_sequence) != expected_record_hash:
            raise LedgerError(reason=f"record hash mismatch at line {expected_sequence + 1}")

        if _required_text(record, "previous_stream_sha256", expected_sequence) != previous_stream:
            raise LedgerError(reason=f"stream predecessor mismatch at line {expected_sequence + 1}")
        expected_stream = _stream_sha256(previous_stream, expected_record_hash)
        if _required_text(record, "stream_sha256", expected_sequence) != expected_stream:
            raise LedgerError(reason=f"stream hash mismatch at line {expected_sequence + 1}")

        previous_stream = expected_stream
        final_sequence = sequence

    return LedgerValidation(
        record_count=final_sequence + 1,
        final_sequence=final_sequence,
        final_stream_sha256=previous_stream,
    )


def _empty_validation() -> LedgerValidation:
    return LedgerValidation(
        record_count=0,
        final_sequence=-1,
        final_stream_sha256=EMPTY_STREAM_SHA256,
    )


def _decode_record(line: bytes, expected_sequence: int) -> dict[str, JsonValue]:
    try:
        return _RECORD_ADAPTER.validate_json(line)
    except ValidationError as error:
        raise LedgerError(reason=f"invalid JSON at line {expected_sequence + 1}") from error


def _record_without_hashes(record: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: value for key, value in record.items() if key not in {"record_sha256", "stream_sha256"}
    }


def _required_int(record: Mapping[str, JsonValue], key: str, line_index: int) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise LedgerError(reason=f"{key} must be int at line {line_index + 1}")
    return value


def _required_text(record: Mapping[str, JsonValue], key: str, line_index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise LedgerError(reason=f"{key} must be string at line {line_index + 1}")
    return value


def _stream_sha256(previous_stream_sha256: str, record_sha256: str) -> str:
    return hashlib.sha256(f"{previous_stream_sha256}\n{record_sha256}".encode()).hexdigest()


def _sha256_json(value: JsonValue | JsonObject) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: JsonValue | JsonObject) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
