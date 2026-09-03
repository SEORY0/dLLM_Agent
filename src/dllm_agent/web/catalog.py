"""Repository-local resource catalog for the WebUI."""

import hashlib
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from dllm_agent.core.errors import BoundaryError
from dllm_agent.core.io import parse_json
from dllm_agent.hunt.models import CandidateBank, HuntReport
from dllm_agent.web.contracts import (
    CatalogListing,
    ResourceDescriptor,
    ResourceKind,
    TrajectoryView,
)


class CatalogBoundaryError(Exception):
    """Raised when a browser resource id cannot resolve to an approved file."""


class UnknownResourceError(CatalogBoundaryError):
    """Raised when a browser resource id is not in the current catalog."""

    def __init__(self) -> None:
        """Create an unknown-resource error."""
        super().__init__("unknown resource")


class WrongResourceKindError(CatalogBoundaryError):
    """Raised when a resource id points at the wrong schema kind."""

    def __init__(self, expected: ResourceKind) -> None:
        """Create a wrong-kind error."""
        super().__init__(f"resource is not a {expected.value.replace('_', ' ')}")


class UnhandledCatalogRootError(CatalogBoundaryError):
    """Raised when catalog enumeration reaches an unsupported root."""

    def __init__(self, root: str) -> None:
        """Create an unsupported-root error."""
        super().__init__(f"unhandled catalog root {root}")


class WebCatalog:
    """Catalog approved non-symlink files under server-owned repository roots."""

    def __init__(self, repo_root: Path) -> None:
        """Create a catalog rooted at the trusted repository path."""
        self._repo_root: Path = repo_root.resolve()

    def list_resources(self) -> CatalogListing:
        """Return a stable snapshot of browser-visible files."""
        resources = tuple(self._describe(path) for path in self._candidate_paths())
        return CatalogListing(resources=resources)

    def resolve_resource(self, resource_id: str) -> ResourceDescriptor:
        """Resolve an opaque resource id without accepting a browser path."""
        for resource in self.list_resources().resources:
            if resource.resource_id == resource_id:
                return resource
        raise UnknownResourceError

    def resource_path(self, resource_id: str) -> Path:
        """Return the approved local path for an opaque resource id."""
        resource = self.resolve_resource(resource_id)
        return self._repo_root / resource.relative_path

    def load_candidate_bank(self, resource_id: str) -> CandidateBank:
        """Parse a catalog resource through the strict candidate-bank schema."""
        resource = self.resolve_resource(resource_id)
        if resource.kind is not ResourceKind.CANDIDATE_BANK:
            raise WrongResourceKindError(ResourceKind.CANDIDATE_BANK)
        return parse_json(self._repo_root / resource.relative_path, CandidateBank, "candidate bank")

    def load_hunt_report(self, resource_id: str) -> HuntReport:
        """Parse a catalog resource through the strict hunt-report schema."""
        resource = self.resolve_resource(resource_id)
        if resource.kind is not ResourceKind.HUNT_REPORT:
            raise WrongResourceKindError(ResourceKind.HUNT_REPORT)
        return parse_json(self._repo_root / resource.relative_path, HuntReport, "hunt report")

    def load_trajectory_view(self, resource_id: str) -> TrajectoryView:
        """Load browser-safe candidate trajectories from a candidate bank."""
        bank = self.load_candidate_bank(resource_id)
        return TrajectoryView(resource_id=resource_id, candidates=bank.candidates)

    def _candidate_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for directory in ("configs", "prompts", "docs"):
            paths.extend(self._regular_files_under(self._repo_root / directory))
        live = self._repo_root / "artifacts/live"
        if live.is_dir():
            paths.extend(path for path in live.iterdir() if self._is_top_level_live_artifact(path))
        jobs = self._repo_root / "artifacts/web/jobs"
        if jobs.is_dir():
            paths.extend(path for path in jobs.glob("*/*.json") if self._is_job_artifact(path))
        return tuple(sorted(paths, key=lambda path: path.relative_to(self._repo_root).as_posix()))

    def _regular_files_under(self, directory: Path) -> tuple[Path, ...]:
        if not directory.is_dir():
            return ()
        return tuple(path for path in directory.rglob("*") if self._is_owned_regular_file(path))

    def _is_top_level_live_artifact(self, path: Path) -> bool:
        if not self._is_owned_regular_file(path):
            return False
        name = path.name
        return path.suffix == ".json" and ("report" in name or "bank" in name)

    def _is_job_artifact(self, path: Path) -> bool:
        return self._is_owned_regular_file(path) and path.name in {
            "candidate-bank.json",
            "hunt-report.json",
        }

    def _is_owned_regular_file(self, path: Path) -> bool:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(self._repo_root)
        )

    def _describe(self, path: Path) -> ResourceDescriptor:
        relative = path.relative_to(self._repo_root).as_posix()
        data = path.read_bytes()
        return ResourceDescriptor(
            resource_id=_resource_id(relative),
            kind=self._kind_for(path),
            relative_path=relative,
            display_name=path.name,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )

    def _kind_for(self, path: Path) -> ResourceKind:
        relative = path.relative_to(self._repo_root)
        first = relative.parts[0]
        match first:
            case "configs":
                return ResourceKind.CONFIG
            case "prompts":
                return ResourceKind.PROMPT
            case "docs":
                return ResourceKind.DOC
            case "artifacts":
                return self._artifact_kind(path)
            case unreachable:
                raise UnhandledCatalogRootError(unreachable)

    def _artifact_kind(self, path: Path) -> ResourceKind:
        try:
            _ = parse_json(path, CandidateBank, "candidate bank")
        except (BoundaryError, ValidationError):
            pass
        else:
            return ResourceKind.CANDIDATE_BANK
        try:
            _ = parse_json(path, HuntReport, "hunt report")
        except (BoundaryError, ValidationError):
            return ResourceKind.ARTIFACT
        else:
            return ResourceKind.HUNT_REPORT


_RESOURCE_ID_BYTES: Final = 24


def _resource_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode()).hexdigest()[:_RESOURCE_ID_BYTES]
    return f"res_{digest}"
