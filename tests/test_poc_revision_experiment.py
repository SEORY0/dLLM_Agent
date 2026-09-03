"""Paired PoC revision experiment runner and analyzer tests."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tools import analyze_poc_revision_experiment as analyzer
from tools import run_poc_revision_experiment as runner
from tools.poc_revision_experiment_json import (
    JsonValue,
    parse_json_object,
    require_array,
    require_number,
    require_object,
    stable_json,
)

ROOT = Path(__file__).parents[1]
ANALYZER_PATH = ROOT / "tools/analyze_poc_revision_experiment.py"


def _number_value(value: JsonValue) -> float:
    assert isinstance(value, int | float)
    assert not isinstance(value, bool)
    return float(value)


def test_seed_schedule_is_paired_and_pseudo_feedback_has_no_fixed_points() -> None:
    """Given pilot defaults, when schedule is built, then exact seeds are paired."""
    # Given
    # When
    schedule = runner.build_schedule(repetitions=5, particles=4)

    # Then
    assert schedule[0].particle_seeds == (202608200000, 202608200001, 202608200002, 202608200003)
    assert schedule[0].fuzzer_seeds == (202608210000, 202608210001)
    assert schedule[0].retrieval_seed == 202608220000
    assert schedule[0].permutation_seed == 202608230000
    assert schedule[0].proof_seed == 202608240000
    assert schedule[4].particle_seeds == (202608204000, 202608204001, 202608204002, 202608204003)
    assert schedule[4].fuzzer_seeds == (202608214000, 202608214001)
    assert schedule[4].retrieval_seed == 202608220004
    assert schedule[4].permutation_seed == 202608230004
    assert schedule[4].proof_seed == 202608240004
    assert runner.BOOTSTRAP_SEED == 202608209999
    assert set(schedule[0].pseudo_permutation) == {0, 1, 2, 3}
    assert all(source != target for source, target in enumerate(schedule[0].pseudo_permutation))


def test_default_budget_is_equal_for_model_arms_and_records_reference_separately(
    tmp_path: Path,
) -> None:
    """Given default pilot budget, when manifests are built, then arms request equal work."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=1, particles=4)

    # Then
    model_arms = [arm for arm in runner.ARMS if arm != "libfuzzer-only"]
    manifests = {
        arm: parse_json_object((tmp_path / f"rep-000/{arm}/run-manifest.json").read_bytes())
        for arm in runner.ARMS
    }
    requested = {arm: require_object(manifests[arm], "budget") for arm in model_arms}
    assert {_number_value(budget["max_model_nfe"]) for budget in requested.values()} == {704.0}
    assert {_number_value(budget["requested_fuzz_seconds"]) for budget in requested.values()} == {
        126.0
    }
    assert {_number_value(budget["proof_seconds"]) for budget in requested.values()} == {30.0}
    assert {
        tuple(require_array(require_object(manifests[arm], "schedule"), "particle_seeds"))
        for arm in model_arms
    } == {(202608200000, 202608200001, 202608200002, 202608200003)}
    reference = runner.reference_manifest(runner.build_schedule(repetitions=1, particles=4)[0])
    assert reference["reference_kind"] == "non-isomorphic"


def test_dry_run_fixture_writes_deterministic_raw_json_bytes(tmp_path: Path) -> None:
    """Given dry-run fixture mode, when run twice, then report bytes and hashes match."""
    # Given
    first = tmp_path / "first"
    second = tmp_path / "second"

    # When
    first_summary = runner.write_fixture_experiment(first, repetitions=3, particles=4)
    second_summary = runner.write_fixture_experiment(second, repetitions=3, particles=4)

    # Then
    first_bytes = (first / "raw-arm-reports.jsonl").read_bytes()
    second_bytes = (second / "raw-arm-reports.jsonl").read_bytes()
    assert first_bytes == second_bytes
    assert first_summary["raw_reports_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert second_summary["raw_reports_sha256"] == first_summary["raw_reports_sha256"]
    assert first_summary["mode"] == "fixture-validation-not-model-efficacy"


def test_analysis_reports_paired_effect_and_censored_proof_time(tmp_path: Path) -> None:
    """Given synthetic reports, when analyzed, then paired effects and censoring are reported."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=5, particles=4)

    # When
    summary = analyzer.analyze_experiment(tmp_path)

    # Then
    contrast = require_object(
        require_object(summary, "contrasts"), "selective-revision_vs_pseudo-feedback"
    )
    assert contrast["paired_n"] == 5
    assert require_number(require_object(contrast, "mean_diff"), "estimate") > 0
    assert require_number(require_object(contrast, "median_diff"), "estimate") >= 0
    assert _number_value(contrast["wins"]) > _number_value(contrast["losses"])
    assert _number_value(contrast["rank_biserial"]) > 0
    assert require_number(require_object(summary, "proof_time"), "right_censored") >= 1
    assert summary["go_no_go"] in {"go", "no-go", "inconclusive"}


def test_analyzer_fails_closed_for_malformed_or_mismatched_arm_reports(tmp_path: Path) -> None:
    """Given corrupt report evidence, when analyzed, then validation rejects it."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=1, particles=4)
    revision_path = tmp_path / "rep-000/selective-revision/revision-report.json"
    _ = revision_path.write_text(
        '{"schema_version":"dllm-poc-revision-report/v1"}\n', encoding="utf-8"
    )

    # When / Then
    with pytest.raises(analyzer.ExperimentAnalysisError, match="sha256"):
        _ = analyzer.analyze_experiment(tmp_path)


def test_analyzer_rejects_tampered_manifest_even_when_reference_hash_matches(
    tmp_path: Path,
) -> None:
    """Given recomputed manifest hash, when manifest disagrees, then analysis fails."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=1, particles=4)
    arm_dir = tmp_path / "rep-000/selective-revision"
    manifest_path = arm_dir / "run-manifest.json"
    manifest = parse_json_object(manifest_path.read_bytes())
    manifest["arm"] = "pseudo-feedback"
    _ = manifest_path.write_bytes(stable_json(manifest) + b"\n")
    arm_report_path = arm_dir / "arm-report.json"
    arm_report = parse_json_object(arm_report_path.read_bytes())
    require_object(arm_report, "run_manifest")["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _ = arm_report_path.write_bytes(stable_json(arm_report) + b"\n")

    # When / Then
    with pytest.raises(analyzer.ExperimentAnalysisError, match="manifest arm"):
        _ = analyzer.analyze_experiment(tmp_path)


def test_analyzer_rejects_missing_arm_in_paired_repetition_set(tmp_path: Path) -> None:
    """Given a missing five-arm cell, when analyzed, then analysis fails closed."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=2, particles=4)
    (tmp_path / "rep-001/pseudo-feedback/arm-report.json").unlink()

    # When / Then
    with pytest.raises(analyzer.ExperimentAnalysisError, match="repetition set"):
        _ = analyzer.analyze_experiment(tmp_path)


def test_analyzer_rejects_duplicate_arm_repetition_cell(tmp_path: Path) -> None:
    """Given duplicated arm/repetition evidence, when analyzed, then analysis fails closed."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=1, particles=4)
    source = tmp_path / "rep-000/selective-revision"
    duplicate = tmp_path / "rep-000/selective-revision-copy"
    _ = shutil.copytree(source, duplicate)
    duplicate_report = duplicate / "arm-report.json"
    payload = parse_json_object(duplicate_report.read_bytes())
    require_object(payload, "metrics")["target_branches"] = 999
    _ = duplicate_report.write_bytes(stable_json(payload) + b"\n")

    # When / Then
    with pytest.raises(analyzer.ExperimentAnalysisError, match="duplicate arm/repetition cell"):
        _ = analyzer.analyze_experiment(tmp_path)


def test_analyzer_rejects_fixture_metric_tamper_not_present_in_raw_rows(
    tmp_path: Path,
) -> None:
    """Given fixture arm-report metric tamper, when raw rows differ, then analysis fails."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=1, particles=4)
    report_path = tmp_path / "rep-000/selective-revision/arm-report.json"
    report = parse_json_object(report_path.read_bytes())
    require_object(report, "metrics")["validated_success"] = False
    _ = report_path.write_bytes(stable_json(report) + b"\n")

    # When / Then
    with pytest.raises(analyzer.ExperimentAnalysisError, match="fixture raw row mismatch"):
        _ = analyzer.analyze_experiment(tmp_path)


def test_thirty_repetition_fixture_recovers_known_treatment_effect(tmp_path: Path) -> None:
    """Given fixture validation, when 30 reps are analyzed, then known effect is recovered."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=30, particles=4)

    # When
    summary = analyzer.analyze_experiment(tmp_path)

    # Then
    primary = require_object(
        require_object(summary, "contrasts"), "selective-revision_vs_pseudo-feedback"
    )
    assert summary["mode"] == "fixture-validation-not-model-efficacy"
    assert summary["go_no_go"] == "go"
    assert primary["paired_n"] == 30
    assert primary["wins"] == 30
    assert primary["losses"] == 0
    mean_diff = require_object(primary, "mean_diff")
    assert require_number(mean_diff, "estimate") >= 0.05
    assert _number_value(require_array(mean_diff, "ci95")[0]) > 0


def test_analyzer_cli_writes_paired_summary_json(tmp_path: Path) -> None:
    """Given fixture reports, when analyzer CLI runs, then summary JSON is byte-stable."""
    # Given
    _ = runner.write_fixture_experiment(tmp_path, repetitions=2, particles=4)

    # When
    result = subprocess.run(  # noqa: S603 - checked-in analyzer path with fixture tmpdir.
        (sys.executable, str(ANALYZER_PATH), str(tmp_path)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 0
    summary_path = tmp_path / "paired-summary.json"
    first = summary_path.read_bytes()
    assert b'"bootstrap_seed":202608209999' in first
    result_again = subprocess.run(  # noqa: S603 - checked-in analyzer path with fixture tmpdir.
        (sys.executable, str(ANALYZER_PATH), str(tmp_path)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result_again.returncode == 0
    assert summary_path.read_bytes() == first
