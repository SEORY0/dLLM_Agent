"""Installed CLI end-to-end tests."""

import subprocess
import sys
from pathlib import Path

from dllm_agent.models import ExpandedPlan, ScanReport

ROOT = Path(__file__).parents[1]
CLI = str(Path(sys.executable).parent / "dllm-agent")


def test_installed_scan_writes_versioned_report(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "report.json"
    command = (
        CLI,
        "scan",
        "fixtures/cwe121/vuln.c",
        "--fixture",
        "fixtures/replay/stable-diffcoder-cwe121.json",
        "--output",
        str(output),
    )

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )

    # Then
    assert result.returncode == 0
    report = ScanReport.model_validate_json(output.read_bytes())
    assert report.validator.status == "proved"


def test_installed_plan_writes_expanded_plan(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "plan.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "plan",
            "configs/owned-fixture-mvp.json",
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 0
    plan = ExpandedPlan.model_validate_json(output.read_bytes())
    assert plan.schema_version == "dllm-plan/v1"


def test_installed_cli_returns_two_without_output_for_malformed_config(tmp_path: Path) -> None:
    # Given
    config = tmp_path / "bad.json"
    _ = config.write_text("{", encoding="utf-8")
    output = tmp_path / "plan.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (CLI, "plan", str(config), "--output", str(output)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert not output.exists()


def test_installed_cli_returns_two_without_output_for_missing_target(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "scan",
            "missing.c",
            "--fixture",
            "fixtures/replay/stable-diffcoder-cwe121.json",
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert not output.exists()


def test_installed_cli_returns_two_without_output_for_malformed_fixture(tmp_path: Path) -> None:
    # Given
    fixture = tmp_path / "bad-fixture.json"
    _ = fixture.write_text("{", encoding="utf-8")
    output = tmp_path / "report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "scan",
            "fixtures/cwe121/vuln.c",
            "--fixture",
            str(fixture),
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert not output.exists()
