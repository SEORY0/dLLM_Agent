"""Versioned schema boundary tests."""

from pathlib import Path

import pytest

from dllm_agent.core.errors import BoundaryError
from dllm_agent.core.io import parse_json, read_bytes
from dllm_agent.core.models import ExperimentConfig, ReplayFixture


def test_fixture_rejects_unknown_version_at_boundary(tmp_path: Path) -> None:
    # Given
    fixture = tmp_path / "fixture.json"
    _ = fixture.write_text('{"schema_version":"future"}', encoding="utf-8")

    # When / Then
    with pytest.raises(BoundaryError, match="schema_version"):
        _ = parse_json(fixture, ReplayFixture, "fixture")


def test_fixture_rejects_negative_nfe_at_boundary(tmp_path: Path) -> None:
    # Given
    source = Path(__file__).parents[1] / "fixtures/replay/stable-diffcoder-cwe121.json"
    fixture = tmp_path / "fixture.json"
    raw = source.read_text(encoding="utf-8").replace('"nfe": 12', '"nfe": -1', 1)
    _ = fixture.write_text(raw, encoding="utf-8")

    # When / Then
    with pytest.raises(BoundaryError, match="nfe"):
        _ = parse_json(fixture, ReplayFixture, "fixture")


def test_config_rejects_malformed_json_at_boundary(tmp_path: Path) -> None:
    # Given
    config = tmp_path / "config.json"
    _ = config.write_text("{", encoding="utf-8")

    # When / Then
    with pytest.raises(BoundaryError, match="config boundary error"):
        _ = parse_json(config, ExperimentConfig, "config")


def test_target_rejects_missing_path_at_boundary(tmp_path: Path) -> None:
    # Given
    missing = tmp_path / "missing.c"

    # When / Then
    with pytest.raises(BoundaryError, match="file does not exist"):
        _ = read_bytes(missing, "target")
