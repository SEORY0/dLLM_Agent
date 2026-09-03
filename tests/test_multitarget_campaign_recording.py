"""Hash-chained experiment recording facade contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dllm_agent.experiments.multitarget.campaign_recording import ExperimentRecorder
from dllm_agent.experiments.multitarget.ledger import validate_ledger


def test_recorder_hashes_json_and_existing_artifacts_into_one_ledger(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    recorder = ExperimentRecorder(
        experiment_root=root,
        now=lambda: datetime(2026, 8, 21, 3, 0, tzinfo=UTC),
    )

    first = recorder.record_json(
        event_type="schedule-frozen",
        relative_path=Path("manifests/schedule.json"),
        value={"cells": 15},
        payload={"manifest_sha256": "a" * 64},
    )
    raw = root / "logs" / "worker.log"
    raw.parent.mkdir()
    _ = raw.write_text("exact output\n", encoding="utf-8")
    second = recorder.record_existing(
        event_type="worker-finished",
        artifact_path=raw,
        payload={"exit_code": 0},
    )

    assert first.artifact.path == "manifests/schedule.json"
    assert second.artifact.path == "logs/worker.log"
    validation = validate_ledger(root / "events.jsonl")
    assert validation.record_count == 2
    assert validation.final_sequence == 1
    assert len(validation.final_stream_sha256) == 64


def test_recorder_appends_phase_events_without_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    recorder = ExperimentRecorder(
        experiment_root=root,
        now=lambda: datetime(2026, 8, 21, 3, 1, tzinfo=UTC),
    )

    result = recorder.event("phase-started", {"phase": "initial-bank"})

    assert result.sequence == 0
    assert validate_ledger(root / "events.jsonl").record_count == 1
