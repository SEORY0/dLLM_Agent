"""HTTP surface tests for the loopback dLLM Hunt Console."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from dllm_agent.web.app import WebSettings, create_app
from dllm_agent.web.contracts import CatalogListing, ResourceKind, TrajectoryView

ROOT = Path(__file__).parents[1]


def _client(tmp_path: Path) -> TestClient:
    settings = WebSettings(
        workspace=ROOT,
        runtime_python=Path(sys.executable),
        cache_dir=tmp_path / "cache",
        source_root=tmp_path / "target",
        jobs_root=tmp_path / "jobs",
    )
    return TestClient(create_app(settings))


def test_web_app_serves_console_showcase_and_security_headers(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        console = client.get("/")
        showcase = client.get("/showcase")

    assert console.status_code == 200
    assert "dLLM Hunt Dashboard" in console.text
    assert 'data-role="runs-table-body"' in console.text
    assert "dLLM Instances" in console.text
    assert "Code Coverage" in console.text
    assert 'src="/assets/app.js"' in console.text
    assert showcase.status_code == 200
    assert "Primitive Showcase" in showcase.text
    assert console.headers["content-security-policy"].startswith("default-src 'self'")
    assert console.headers["x-content-type-options"] == "nosniff"
    assert console.headers["cache-control"] == "no-store"


def test_web_app_serves_report_detail_route(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        catalog_response = client.get("/api/v1/catalog")
        catalog = CatalogListing.model_validate_json(catalog_response.content)
        report = next(item for item in catalog.resources if item.kind is ResourceKind.HUNT_REPORT)
        response = client.get(f"/run/{report.resource_id}")

    assert response.status_code == 200
    assert "dLLM Hunt Dashboard" in response.text
    assert 'data-view="detail"' in response.text
    assert 'data-role="instance-grid"' in response.text
    assert 'data-role="coverage-grid"' in response.text


def test_web_app_catalog_and_trajectory_are_repo_owned(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        catalog_response = client.get("/api/v1/catalog")
        assert catalog_response.status_code == 200
        catalog = CatalogListing.model_validate_json(catalog_response.content)
        bank = next(item for item in catalog.resources if item.kind is ResourceKind.CANDIDATE_BANK)

        trajectory_response = client.get(f"/api/v1/trajectories/{bank.resource_id}")
        missing_response = client.get("/api/v1/trajectories/not-a-resource")

    assert trajectory_response.status_code == 200
    trajectory = TrajectoryView.model_validate_json(trajectory_response.content)
    assert trajectory.resource_id == bank.resource_id
    assert len(trajectory.candidates) >= 2
    assert trajectory.candidates[0].steps
    assert missing_response.status_code == 404


def test_web_app_health_exposes_no_host_paths(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"service": "dllm-agent-web", "status": "ok"}
    assert str(ROOT) not in response.text
