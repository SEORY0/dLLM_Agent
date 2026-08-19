"""Pipeline integration tests."""

from pathlib import Path

import pytest

from dllm_agent.errors import BoundaryError
from dllm_agent.pipeline import scan

ROOT = Path(__file__).parents[1]


def test_scan_rejects_mismatched_proof_hash(tmp_path: Path) -> None:
    # Given
    source = ROOT / "fixtures/replay/stable-diffcoder-cwe121.json"
    fixture = tmp_path / "fixture.json"
    raw = source.read_text(encoding="utf-8").replace(
        "2db6f282624b12b61f6d147229e59635e929a05fdb88daa2ce0da7f85e6dac88",
        "0" * 64,
        1,
    )
    _ = fixture.write_text(raw, encoding="utf-8")

    # When / Then
    with pytest.raises(BoundaryError, match="proof input SHA-256"):
        _ = scan(ROOT / "fixtures/cwe121/vuln.c", fixture)


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
