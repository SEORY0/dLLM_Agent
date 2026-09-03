"""Web catalog boundary tests."""

import hashlib
from pathlib import Path

import pytest

from dllm_agent.web.catalog import CatalogBoundaryError, WebCatalog
from dllm_agent.web.contracts import ResourceKind


def test_catalog_lists_only_allowed_regular_files_and_rejects_escapes(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "artifacts/live").mkdir(parents=True)
    (root / "artifacts/web/jobs/job-1").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "prompts").mkdir()
    _ = (root / "configs/hunt.json").write_text("{}", encoding="utf-8")
    _ = (root / "docs/live-hunt.md").write_text("notes\n", encoding="utf-8")
    _ = (root / "artifacts/live/report.json").write_text("{}", encoding="utf-8")
    _ = (root / "artifacts/web/jobs/job-1/candidate-bank.json").write_text("{}", encoding="utf-8")
    _ = (root / "artifacts/web/jobs/job-1/status.json").write_text("{}", encoding="utf-8")
    (root / "artifacts/live/run/build.log").parent.mkdir()
    _ = (root / "artifacts/live/run/build.log").write_text("private log\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    _ = outside.write_text("{}", encoding="utf-8")
    (root / "configs/escape.json").symlink_to(outside)

    # When
    catalog = WebCatalog(root)
    listing = catalog.list_resources()

    # Then
    paths = tuple(item.relative_path for item in listing.resources)
    assert paths == (
        "artifacts/live/report.json",
        "artifacts/web/jobs/job-1/candidate-bank.json",
        "configs/hunt.json",
        "docs/live-hunt.md",
    )
    assert tuple(item.display_name for item in listing.resources) == (
        "report.json",
        "candidate-bank.json",
        "hunt.json",
        "live-hunt.md",
    )
    assert all(item.resource_id != item.relative_path for item in listing.resources)
    with pytest.raises(CatalogBoundaryError, match="unknown resource"):
        _ = catalog.resolve_resource("configs/escape.json")


def test_catalog_uses_stable_opaque_ids_and_loads_known_schemas() -> None:
    # Given
    root = Path.cwd()
    bank_path = Path("artifacts/live/jsoncons-ubjson-candidate-bank.json")
    report_path = Path("artifacts/live/jsoncons-ubjson-report.json")
    bank_id = _resource_id(bank_path)
    report_id = _resource_id(report_path)

    # When
    catalog = WebCatalog(root)
    bank = catalog.load_candidate_bank(bank_id)
    report = catalog.load_hunt_report(report_id)
    view = catalog.load_trajectory_view(bank_id)

    # Then
    assert bank.schema_version == "dllm-candidate-bank/v1"
    assert report.schema_version == "dllm-hunt-report/v1"
    assert view.resource_id == bank_id
    assert view.candidates == bank.candidates
    assert catalog.resolve_resource(bank_id).relative_path == str(bank_path)
    assert catalog.resolve_resource(report_id).kind is ResourceKind.HUNT_REPORT


def _resource_id(path: Path) -> str:
    digest = hashlib.sha256(path.as_posix().encode()).hexdigest()[:24]
    return f"res_{digest}"
