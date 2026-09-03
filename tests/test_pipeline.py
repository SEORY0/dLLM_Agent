"""Pipeline integration tests."""

from pathlib import Path

import pytest

from dllm_agent.core.errors import BoundaryError
from dllm_agent.core.models import ReplayFixture
from dllm_agent.replay.pipeline import scan, verify_replay_fixture

ROOT = Path(__file__).parents[1]


def test_replay_fixture_rejects_mismatched_proof_hash() -> None:
    # Given
    source = ROOT / "fixtures/replay/stable-diffcoder-cwe121.json"
    fixture = ReplayFixture.model_validate_json(source.read_bytes()).model_copy(
        update={"proof_input_sha256": "0" * 64}
    )

    # When / Then
    with pytest.raises(BoundaryError, match="proof input SHA-256"):
        _ = verify_replay_fixture(fixture, source, fixture.target_sha256)


def test_scan_emits_auditable_proved_report() -> None:
    # Given
    target = ROOT / "fixtures/cwe121/vuln.c"
    fixture = ROOT / "fixtures/replay/stable-diffcoder-cwe121.json"

    # When
    report = scan(target, fixture)

    # Then
    assert (
        report.schema_version,
        report.finding.cwe,
        report.validator.status,
        report.accounting.validated_candidates,
    ) == ("dllm-scan-report/v1", "CWE-121", "proved", 1)


def test_scan_report_names_replay_trace_rotation_when_serialized() -> None:
    # Given
    target = ROOT / "fixtures/cwe121/vuln.c"
    fixture = ROOT / "fixtures/replay/stable-diffcoder-cwe121.json"

    # When
    report_json = scan(target, fixture).model_dump(mode="json")

    # Then
    assert "synthetic_trace_rotation" in report_json
    assert "permutation_control" not in report_json
