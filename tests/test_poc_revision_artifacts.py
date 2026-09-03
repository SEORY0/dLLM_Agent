"""Deterministic JSONL ledger tests for PoC revision artifacts."""

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.artifacts import (
    revision_events_bytes,
    revision_events_sha256,
)
from dllm_agent.revision.models import RevisionEvent, RevisionEventKind

_ZERO_SHA = "0" * 64


def test_revision_events_bytes_are_stable_and_timestamp_free() -> None:
    """Given identical events, when serialized twice, then bytes are identical."""
    events = (_event(1, "particle_started"), _event(2, "remask_decided"))

    first = revision_events_bytes(events)
    second = revision_events_bytes(events)

    assert first == second
    assert b"timestamp" not in first
    assert b'"sequence":1' in first
    assert b'"sequence":2' in first
    assert revision_events_sha256(first) == sha256_bytes(first)


def test_revision_events_reject_non_monotonic_sequence() -> None:
    """Given a repeated sequence, when serializing, then ledger creation fails."""
    with pytest.raises(ValueError, match="monotonic"):
        _ = revision_events_bytes((_event(1, "particle_started"), _event(1, "probe_completed")))


def _event(sequence: int, kind: RevisionEventKind) -> RevisionEvent:
    return RevisionEvent(
        sequence=sequence,
        kind=kind,
        particle_id="p0",
        snapshot_sha256=_ZERO_SHA,
        parent_sha256=None,
        feedback_sha256=None,
        remask_sha256=None,
        fields=(),
        nfe=0,
        forward_calls=0,
        gpu_ms=0,
        validator_ms=0,
        fuzz_seconds=0,
        executions_per_second=None,
        cache_hit=False,
        dedup_hit=False,
        artifact_sha256=(),
    )
