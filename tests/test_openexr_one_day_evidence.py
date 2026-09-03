"""End-to-end Task 6 evidence audit and atomic-output contracts. # noqa: SIZE_OK"""

from __future__ import annotations

import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.openexr_one_day.analysis import (
    AnalysisError,
    EvidenceTreeContent,
    analyze_campaign,
    audit_campaign,
)
from tests.openexr_one_day_task6_fixtures import build_task6_campaign

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def campaign_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    case = tmp_path_factory.mktemp("task6-campaign")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    root = Path("/data/dllm-agent") / f"task6-{suffix}"
    if root.exists():
        raise AssertionError(root)
    _ = build_task6_campaign(root)
    yield root
    shutil.rmtree(root)


def test_missing_coverage_backing_fails_before_results(campaign_root: Path) -> None:
    # Given
    audit = audit_campaign(campaign_root)
    first = audit.arms[0]
    evidence = first.coverage.coverage_series.points[0].profdata_evidence
    assert evidence is not None
    backing = first.row.artifact_root / evidence.relative_path
    content = backing.read_bytes()

    # When / Then
    try:
        backing.unlink()
        with pytest.raises(
            AnalysisError,
            match=r"backing evidence file is missing|completed post-search evidence invalid",
        ):
            _ = audit_campaign(campaign_root)
        assert not (campaign_root / "results").exists()
    finally:
        _ = backing.write_bytes(content)


def test_audit_and_analysis_write_only_complete_canonical_results(campaign_root: Path) -> None:
    # Given
    source_raw = (campaign_root / "raw-arm-reports.jsonl").read_bytes()

    # When
    result = analyze_campaign(campaign_root)

    # Then
    output_names = {path.name for path in (campaign_root / "results").iterdir()}
    assert output_names == {
        "raw-arm-reports.jsonl",
        "paired-summary.json",
        "plots-data.json",
        "report.md",
        "evidence-tree.sha256",
    }
    assert (campaign_root / "results" / "raw-arm-reports.jsonl").read_bytes() == source_raw
    assert len(result.audit.rows) == 50
    assert result.summary.complete_pair_count == 10
    assert result.summary.wins >= 9
    assert result.summary.paired_median_uplift >= 0.10
    assert result.summary.final_coverage_retention >= 0.95
    assert result.summary.deep_parse_drop_percentage_points <= 5.0
    assert result.summary.resource_usage.resident_model_load_duration_seconds is None
    assert result.summary.resource_usage.resident_model_load_rss_bytes is None
    assert result.summary.resource_usage.resident_model_load_vram_bytes is None
    assert result.summary.resource_usage.resource_integrity is False
    assert result.summary.resource_usage.resource_integrity_reasons == (
        "missing-upstream-value:CampaignAccounting.resident_model_load",
        "missing-upstream-value:CampaignAccounting.resident_model_load_evidence",
    )
    assert result.summary.decision == "INCONCLUSIVE"
    assert len(result.summary.event_endpoints) == 50
    assert all(endpoint.first_proved_cause.censored for endpoint in result.summary.event_endpoints)
    assert (
        b'"first_proved_cause_event"'
        in (campaign_root / "results" / "plots-data.json").read_bytes()
    )
    assert "first proved-cause time" in (campaign_root / "results" / "report.md").read_text(
        encoding="utf-8"
    )
    assert "missing-upstream-value:CampaignAccounting.resident_model_load" in (
        campaign_root / "results" / "report.md"
    ).read_text(encoding="utf-8")


def test_rerun_is_byte_identical_including_evidence_tree_root(campaign_root: Path) -> None:
    # Given
    first = analyze_campaign(campaign_root)
    first_bytes = {
        path.name: path.read_bytes() for path in sorted((campaign_root / "results").iterdir())
    }

    # When
    second = analyze_campaign(campaign_root)
    second_bytes = {
        path.name: path.read_bytes() for path in sorted((campaign_root / "results").iterdir())
    }

    # Then
    assert second_bytes == first_bytes
    assert second.evidence_tree_sha256 == first.evidence_tree_sha256
    tree_path = campaign_root / "results" / "evidence-tree.sha256"
    preimage = EvidenceTreeContent.model_validate_json(tree_path.read_bytes())
    assert set(preimage.model_dump()) == {"schema_version", "entries"}
    assert preimage.schema_version == "dllm-openexr-evidence-tree-content/v1"
    assert [entry.path for entry in preimage.entries] == sorted(
        entry.path for entry in preimage.entries
    )
    assert sha256(tree_path.read_bytes()).hexdigest() == first.evidence_tree_sha256


def test_one_byte_raw_mutation_fails_closed_without_changing_existing_outputs(
    campaign_root: Path,
) -> None:
    # Given
    _ = analyze_campaign(campaign_root)
    results = campaign_root / "results"
    before = {path.name: path.read_bytes() for path in sorted(results.iterdir())}
    raw_path = campaign_root / "raw-arm-reports.jsonl"
    original = raw_path.read_bytes()
    mutated = bytes([original[0] ^ 1]) + original[1:]

    # When / Then
    try:
        _ = raw_path.write_bytes(mutated)
        with pytest.raises(AnalysisError, match=r"campaign outcome|raw report"):
            _ = analyze_campaign(campaign_root)
        assert {path.name: path.read_bytes() for path in sorted(results.iterdir())} == before
    finally:
        _ = raw_path.write_bytes(original)


def test_unknown_file_and_symlink_are_rejected_before_analysis(campaign_root: Path) -> None:
    # Given
    unknown = campaign_root / "rogue.bin"
    symlink = campaign_root / "rogue-link"

    # When / Then
    try:
        _ = unknown.write_bytes(b"rogue")
        with pytest.raises(AnalysisError, match="unknown campaign artifact"):
            _ = audit_campaign(campaign_root)
        unknown.unlink()
        symlink.symlink_to(campaign_root / "raw-arm-reports.jsonl")
        with pytest.raises(AnalysisError, match="symlink"):
            _ = audit_campaign(campaign_root)
    finally:
        unknown.unlink(missing_ok=True)
        symlink.unlink(missing_ok=True)


def test_partial_or_unknown_results_are_rejected(campaign_root: Path) -> None:
    # Given
    _ = analyze_campaign(campaign_root)
    unknown = campaign_root / "results" / "partial.tmp"

    # When / Then
    try:
        _ = unknown.write_bytes(b"partial")
        with pytest.raises(AnalysisError, match="results artifact set"):
            _ = audit_campaign(campaign_root)
    finally:
        unknown.unlink(missing_ok=True)


def test_cli_exits_two_without_partial_outputs_for_invalid_campaign(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "invalid-campaign"
    root.mkdir()
    script = Path(__file__).parents[1] / "tools" / "analyze_openexr_one_day.py"

    # When
    completed = subprocess.run(  # noqa: S603
        [sys.executable, script.as_posix(), root.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 2
    assert "analysis failed:" in completed.stderr
    assert not (root / "results").exists()
