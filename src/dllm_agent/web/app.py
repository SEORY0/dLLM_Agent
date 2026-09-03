"""Loopback WebUI application for the dLLM Hunt Console."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dllm_agent.core.errors import BoundaryError
from dllm_agent.hunt.models import HuntReport
from dllm_agent.web.catalog import (
    CatalogBoundaryError,
    UnknownResourceError,
    WebCatalog,
    WrongResourceKindError,
)
from dllm_agent.web.contracts import (
    CatalogListing,
    HuntJobSpec,
    JobEvent,
    JobRuntimeConfig,
    JobStatus,
    ResourceKind,
    SampleJobSpec,
    TrajectoryView,
)
from dllm_agent.web.jobs import JobBusyError, JobManager, UnknownJobError

SecurityHandler = Callable[[Request], Awaitable[Response]]


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Trusted server-side paths for the local console."""

    workspace: Path
    runtime_python: Path
    cache_dir: Path
    source_root: Path
    jobs_root: Path


def create_app(settings: WebSettings) -> FastAPI:
    """Create a repository-bound FastAPI control plane."""
    workspace = settings.workspace.resolve()
    static_root = Path(__file__).with_name("static")
    catalog = WebCatalog(workspace)
    manager = JobManager(
        JobRuntimeConfig(
            repo_root=workspace,
            jobs_root=settings.jobs_root,
            runtime_python=settings.runtime_python,
            cache_dir=settings.cache_dir,
            source_root=settings.source_root,
        ),
        resolver=catalog.resource_path,
    )
    app = FastAPI(
        title="dLLM Hunt Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")
    _install_security_headers(app)
    _install_document_routes(app, static_root)
    _install_catalog_routes(app, catalog)
    _install_job_routes(app, catalog, manager)
    return app


def _install_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: SecurityHandler) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; "
            "img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    _ = add_security_headers


def _install_document_routes(app: FastAPI, static_root: Path) -> None:
    @app.get("/", include_in_schema=False)
    def console() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/run/{resource_id}", include_in_schema=False)
    def run_detail(resource_id: str) -> FileResponse:
        _ = resource_id
        return FileResponse(static_root / "index.html")

    @app.get("/showcase", include_in_schema=False)
    def showcase() -> FileResponse:
        return FileResponse(static_root / "showcase.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"service": "dllm-agent-web", "status": "ok"}

    _ = (console, run_detail, showcase, health)


def _install_catalog_routes(app: FastAPI, catalog: WebCatalog) -> None:
    @app.get("/api/v1/catalog", response_model=CatalogListing)
    def list_catalog() -> CatalogListing:
        return catalog.list_resources()

    @app.get("/api/v1/trajectories/{resource_id}", response_model=TrajectoryView)
    def get_trajectory(resource_id: str) -> TrajectoryView:
        try:
            return catalog.load_trajectory_view(resource_id)
        except UnknownResourceError as error:
            raise _not_found(error) from error
        except (WrongResourceKindError, BoundaryError) as error:
            raise _unprocessable(error) from error

    @app.get("/api/v1/reports/{resource_id}", response_model=HuntReport)
    def get_report(resource_id: str) -> HuntReport:
        try:
            return catalog.load_hunt_report(resource_id)
        except UnknownResourceError as error:
            raise _not_found(error) from error
        except (WrongResourceKindError, BoundaryError) as error:
            raise _unprocessable(error) from error

    _ = (list_catalog, get_trajectory, get_report)


def _install_job_routes(app: FastAPI, catalog: WebCatalog, manager: JobManager) -> None:
    @app.get("/api/v1/jobs")
    def list_jobs() -> dict[str, tuple[JobStatus, ...]]:
        _ = manager.refresh()
        return {"jobs": manager.list_statuses()}

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str) -> JobStatus:
        _ = manager.refresh()
        try:
            return manager.get_status(job_id)
        except UnknownJobError as error:
            raise _not_found(error) from error

    @app.get("/api/v1/jobs/{job_id}/events")
    def get_job_events(job_id: str) -> dict[str, tuple[JobEvent, ...]]:
        _ = manager.refresh()
        try:
            return {"events": manager.events(job_id)}
        except UnknownJobError as error:
            raise _not_found(error) from error

    @app.post("/api/v1/jobs/sample", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
    def start_sample(spec: SampleJobSpec) -> JobStatus:
        _require_kind(catalog, spec.manifest_id, ResourceKind.CONFIG)
        return _start_job(manager, spec)

    @app.post("/api/v1/jobs/hunt", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
    def start_hunt(spec: HuntJobSpec) -> JobStatus:
        _require_kind(catalog, spec.manifest_id, ResourceKind.CONFIG)
        _require_kind(catalog, spec.candidate_bank_id, ResourceKind.CANDIDATE_BANK)
        return _start_job(manager, spec)

    _ = (list_jobs, get_job, get_job_events, start_sample, start_hunt)


def _require_kind(catalog: WebCatalog, resource_id: str, expected: ResourceKind) -> None:
    try:
        resource = catalog.resolve_resource(resource_id)
    except CatalogBoundaryError as error:
        raise _not_found(error) from error
    if resource.kind is not expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"resource must be a {expected.value}",
        )


def _start_job(manager: JobManager, spec: SampleJobSpec | HuntJobSpec) -> JobStatus:
    try:
        return manager.start(spec)
    except JobBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


def _not_found(error: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


def _unprocessable(error: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error))
