"""Byte-stable JSONL artifacts for PoC revision evidence."""

from dataclasses import dataclass
from typing import override

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.models import RevisionEvent


@dataclass(frozen=True, slots=True)
class RevisionLedgerSequenceError(ValueError):
    """Raised when revision ledger events are not strictly monotonic."""

    previous: int
    current: int

    @override
    def __str__(self) -> str:
        """Return a concise sequence failure."""
        return f"revision event sequence must be monotonic: {self.previous} then {self.current}"


def revision_events_bytes(events: tuple[RevisionEvent, ...]) -> bytes:
    """Serialize events as byte-stable timestamp-free JSONL."""
    _require_monotonic(events)
    return b"".join(_event_line(event) for event in events)


def revision_events_sha256(payload: bytes) -> str:
    """Hash the exact raw revision JSONL bytes."""
    return sha256_bytes(payload)


def _require_monotonic(events: tuple[RevisionEvent, ...]) -> None:
    previous = 0
    for event in events:
        if event.sequence <= previous:
            raise RevisionLedgerSequenceError(previous=previous, current=event.sequence)
        previous = event.sequence


def _event_line(event: RevisionEvent) -> bytes:
    return (
        b'{"kind":"'
        + event.kind.encode()
        + b'","event":'
        + event.model_dump_json().encode()
        + b"}\n"
    )
