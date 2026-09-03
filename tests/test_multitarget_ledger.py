from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.ledger import (
    LedgerError,
    append_event,
    artifact_evidence,
    command_evidence,
    validate_ledger,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_append_event_records_canonical_jsonl_without_leaking_environment(
    tmp_path: Path,
) -> None:
    """Given command evidence, when appended, then argv/env hashes bind private inputs."""
    # Given
    ledger_path = tmp_path / "campaign.jsonl"
    command = command_evidence(
        argv=("/usr/bin/docker", "run", "--rm", "image"),
        env={"ASAN_OPTIONS": "abort_on_error=1", "SECRET_TOKEN": "do-not-write"},
    )

    # When
    first = append_event(
        ledger_path,
        timestamp_utc="2026-08-21T02:00:00Z",
        event_type="cell_started",
        payload={"target": "libpng", "command": command.to_json()},
    )
    second = append_event(
        ledger_path,
        timestamp_utc="2026-08-21T02:00:01Z",
        event_type="cell_finished",
        payload={"target": "libpng", "status": "completed"},
    )

    # Then
    raw_text = ledger_path.read_text(encoding="utf-8")
    assert "do-not-write" not in raw_text
    assert "SECRET_TOKEN" in raw_text
    assert first.sequence == 0
    assert second.sequence == 1
    assert validate_ledger(ledger_path).final_stream_sha256 == second.stream_sha256

    expected_argv_sha256 = hashlib.sha256(b"/usr/bin/docker\0run\0--rm\0image").hexdigest()
    assert f'"argv_sha256":"{expected_argv_sha256}"' in raw_text
    assert '"env_keys":["ASAN_OPTIONS","SECRET_TOKEN"]' in raw_text
    assert f'"previous_stream_sha256":"{first.stream_sha256}"' in raw_text


def test_artifact_evidence_records_size_and_content_hash(tmp_path: Path) -> None:
    """Given an artifact file, when evidenced, then size and sha256 bind the bytes."""
    # Given
    artifact_path = tmp_path / "asan.log"
    _ = artifact_path.write_bytes(b"AddressSanitizer: heap-buffer-overflow\n")

    # When
    evidence = artifact_evidence(artifact_path)

    # Then
    assert evidence.path == str(artifact_path)
    assert evidence.size_bytes == 39
    assert (
        evidence.sha256 == hashlib.sha256(b"AddressSanitizer: heap-buffer-overflow\n").hexdigest()
    )


def test_validate_ledger_fails_closed_on_tamper_gap_and_truncated_line(
    tmp_path: Path,
) -> None:
    """Given a valid ledger, when bytes are corrupted, then validation rejects it."""
    # Given
    ledger_path = tmp_path / "campaign.jsonl"
    _ = append_event(
        ledger_path,
        timestamp_utc="2026-08-21T02:00:00Z",
        event_type="build_started",
        payload={"target": "libxml2"},
    )
    _ = append_event(
        ledger_path,
        timestamp_utc="2026-08-21T02:00:01Z",
        event_type="build_finished",
        payload={"target": "libxml2"},
    )

    # When / Then
    tampered = tmp_path / "tampered.jsonl"
    _ = tampered.write_text(
        ledger_path.read_text(encoding="utf-8").replace("libxml2", "libtiff", 1),
        encoding="utf-8",
    )
    with pytest.raises(LedgerError, match="record hash mismatch"):
        _ = validate_ledger(tampered)

    gap = tmp_path / "gap.jsonl"
    _ = gap.write_text(
        ledger_path.read_text(encoding="utf-8").replace('"sequence":1', '"sequence":3'),
        encoding="utf-8",
    )
    with pytest.raises(LedgerError, match="sequence gap"):
        _ = validate_ledger(gap)

    truncated = tmp_path / "truncated.jsonl"
    _ = truncated.write_text(ledger_path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    with pytest.raises(LedgerError, match="truncated"):
        _ = validate_ledger(truncated)
