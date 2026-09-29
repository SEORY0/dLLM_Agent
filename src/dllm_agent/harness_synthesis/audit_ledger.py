"""Append-only per-iteration harness identity ledger.

Resolves spec §10's audit-chain risk: the existing pipeline pins harness identity to
one Docker image digest (``tools/openexr_runtime/Dockerfile``'s ``COPY``), which works
because that harness never changes within a campaign. Harness synthesis changes the
harness every iteration, so identity must instead be recorded per iteration — matching
``docs/paper-abstract-and-evaluation-claims.md``'s append-only JSONL audit ledger
principle. Design and field rationale live in
``docs/harness-synthesis-audit-ledger.md``; this module is that design's schema and
storage, structured exactly like ``revision/failure_registry.py`` (JSONL, exclusive-file
locking, fail-closed parse) rather than a new pattern.
"""

import contextlib
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from pydantic import Field, ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256

_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.01


class HarnessSynthesisLedgerRecord(FrozenModel):
    """One iteration's complete harness identity: source, knowledge base, and outcome."""

    schema_version: Literal["dllm-harness-synthesis-ledger/v1"] = (
        "dllm-harness-synthesis-ledger/v1"
    )
    run_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    arm: str = Field(min_length=1)
    seed: int
    harness_source_sha256: Sha256
    knowledge_base_sha256: Sha256
    usage_example_names: tuple[str, ...]
    compiled: bool
    compile_error: str | None = None
    coverage_edges: int | None = Field(default=None, ge=0)
    recorded_at_unix_seconds: float = Field(ge=0)


@dataclass(frozen=True, slots=True)
class AuditLedgerError(ValueError):
    """Raised when a ledger file cannot be parsed."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Return a fail-closed ledger parse error."""
        return f"{self.path}: {self.reason}"


def load(path: Path) -> tuple[HarnessSynthesisLedgerRecord, ...]:
    """Load persisted ledger records, or return empty if the file is absent."""
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[HarnessSynthesisLedgerRecord] = []
    for line_index, line in enumerate(lines):
        if line == "":
            continue
        try:
            records.append(HarnessSynthesisLedgerRecord.model_validate_json(line))
        except ValidationError as error:
            reason = f"line {line_index + 1}: {error}"
            raise AuditLedgerError(path=path, reason=reason) from error
    return tuple(records)


def append(path: Path, record: HarnessSynthesisLedgerRecord) -> None:
    """Atomically append one record, serialized against other concurrent appenders."""
    append_many(path, (record,))


def append_many(path: Path, records: tuple[HarnessSynthesisLedgerRecord, ...]) -> None:
    """Atomically append many records under a single lock/read/write round trip.

    Mirrors ``failure_registry.append_many``: ``_write_all`` replaces the whole file, so
    two overlapping read-modify-write calls would otherwise let the second replace
    silently discard the first's records — ``_locked`` serializes them.
    """
    if len(records) == 0:
        return
    with _locked(path):
        _write_all(path, (*load(path), *records))


def records_sha256(records: tuple[HarnessSynthesisLedgerRecord, ...]) -> str | None:
    """Hash the exact serialized content of a set of loaded records, or None if empty."""
    if len(records) == 0:
        return None
    return sha256_bytes(_jsonl(records).encode())


def _write_all(path: Path, records: tuple[HarnessSynthesisLedgerRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _ = temporary.write_text(_jsonl(records), encoding="utf-8")
    _ = temporary.replace(path)


def _jsonl(records: tuple[HarnessSynthesisLedgerRecord, ...]) -> str:
    return "".join(record.model_dump_json() + "\n" for record in records)


@contextlib.contextmanager
def _locked(path: Path) -> Generator[None]:
    """Hold an exclusive cross-process lock across one read-modify-write update.

    Uses exclusive file creation (``"x"`` mode), atomic on both POSIX (O_EXCL) and
    Windows (CREATE_NEW). A lock left behind by a crashed writer fails closed after the
    timeout rather than blocking every future append forever.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            with lock_path.open("x"):
                break
        except FileExistsError:
            if time.monotonic() > deadline:
                reason = f"lock held longer than {_LOCK_TIMEOUT_SECONDS}s"
                raise AuditLedgerError(path=path, reason=reason) from None
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)
