"""Hash-bound resident model-load evidence for the OpenEXR campaign."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Self, override

from pydantic import Field, ValidationError, model_validator

from dllm_agent.core.models import FrozenModel
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    persist_backing_bytes,
    validate_backing_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.experiments.openexr_one_day.resident_pool import (
        ResourceSnapshot,
        WorkerLoadTelemetry,
    )

_RELATIVE_PATH = "provenance/resident-model-load.json"
_WORKER_COUNT = 8


class ResidentEvidenceError(Exception):
    """Raised when resident-load evidence is incomplete or byte-inconsistent."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the validation failure reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class ResourceSnapshotEvidence(FrozenModel):
    """Serializable resource values captured at one load instant."""

    host_available_bytes: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)
    gpu_free_bytes: int = Field(ge=0)
    process_rss_bytes: int = Field(ge=0)


class ResidentWorkerLoadEvidence(FrozenModel):
    """Exact load interval and extrema for one physical GPU worker."""

    gpu_index: int = Field(ge=0, le=7)
    requested_device: str = Field(pattern=r"^cuda:[0-7]$")
    ready_device: str = Field(pattern=r"^cuda:[0-7]$")
    load_seconds: float = Field(ge=0.0)
    process_id: int = Field(gt=0)
    before: ResourceSnapshotEvidence
    after: ResourceSnapshotEvidence
    samples: tuple[ResourceSnapshotEvidence, ...] = Field(min_length=2)
    minimum_host_available_bytes: int = Field(ge=0)
    maximum_swap_used_bytes: int = Field(ge=0)
    minimum_gpu_free_bytes: int = Field(ge=0)
    peak_process_rss_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_measurements(self) -> Self:
        expected_device = f"cuda:{self.gpu_index}"
        if self.requested_device != expected_device or self.ready_device != expected_device:
            message = "resident worker device does not match GPU index"
            raise ValueError(message)
        if self.samples[0] != self.before or self.samples[-1] != self.after:
            message = "resident worker boundary samples mismatch"
            raise ValueError(message)
        if (
            self.minimum_host_available_bytes
            != min(sample.host_available_bytes for sample in self.samples)
            or self.maximum_swap_used_bytes
            != max(sample.swap_used_bytes for sample in self.samples)
            or self.minimum_gpu_free_bytes != min(sample.gpu_free_bytes for sample in self.samples)
            or self.peak_process_rss_bytes
            != max(sample.process_rss_bytes for sample in self.samples)
        ):
            message = "resident worker extrema mismatch"
            raise ValueError(message)
        return self

    @property
    def peak_vram_bytes(self) -> int:
        """Return peak observed GPU-memory consumption relative to preload free memory."""
        return max(0, self.before.gpu_free_bytes - self.minimum_gpu_free_bytes)


class ResidentModelLoadEvidence(FrozenModel):
    """Campaign-wide setup cost, persisted once before the main clock."""

    schema_version: Literal["dllm-openexr-resident-model-load/v1"] = (
        "dllm-openexr-resident-model-load/v1"
    )
    workers: tuple[ResidentWorkerLoadEvidence, ...] = Field(min_length=8, max_length=8)
    total_load_seconds: float = Field(ge=0.0)
    peak_process_rss_bytes: int = Field(ge=0)
    peak_vram_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_aggregate(self) -> Self:
        if tuple(worker.gpu_index for worker in self.workers) != tuple(range(_WORKER_COUNT)):
            message = "resident load evidence requires ordered GPUs 0 through 7"
            raise ValueError(message)
        if len({worker.process_id for worker in self.workers}) != _WORKER_COUNT:
            message = "resident load evidence requires unique worker PIDs"
            raise ValueError(message)
        if self.total_load_seconds != math.fsum(worker.load_seconds for worker in self.workers):
            message = "resident model-load duration mismatch"
            raise ValueError(message)
        if self.peak_process_rss_bytes != max(
            worker.peak_process_rss_bytes for worker in self.workers
        ):
            message = "resident model-load RSS mismatch"
            raise ValueError(message)
        if self.peak_vram_bytes != max(worker.peak_vram_bytes for worker in self.workers):
            message = "resident model-load VRAM mismatch"
            raise ValueError(message)
        return self


def build_resident_model_load_evidence(
    telemetry: tuple[WorkerLoadTelemetry, ...],
) -> ResidentModelLoadEvidence:
    """Convert the complete resident-pool telemetry into a validated artifact model."""
    if len(telemetry) != _WORKER_COUNT:
        raise ResidentEvidenceError(reason="resident load evidence requires exactly eight workers")
    workers = tuple(_worker_evidence(item) for item in telemetry)
    try:
        return ResidentModelLoadEvidence(
            workers=workers,
            total_load_seconds=math.fsum(worker.load_seconds for worker in workers),
            peak_process_rss_bytes=max(worker.peak_process_rss_bytes for worker in workers),
            peak_vram_bytes=max(worker.peak_vram_bytes for worker in workers),
        )
    except ValidationError as error:
        raise ResidentEvidenceError(reason="resident load evidence is invalid") from error


def persist_resident_model_load_evidence(
    root: Path,
    evidence: ResidentModelLoadEvidence,
) -> ArtifactBackingEvidence:
    """Persist canonical resident-load evidence below a campaign root."""
    try:
        return persist_backing_bytes(
            root,
            _RELATIVE_PATH,
            canonical_json_bytes(evidence.model_dump(mode="json")),
        )
    except EvidenceBackingError as error:
        raise ResidentEvidenceError(reason=str(error)) from error


def load_resident_model_load_evidence(
    root: Path,
    backing: ArtifactBackingEvidence,
    *,
    expected_worker_pids: tuple[int, ...] | None = None,
) -> ResidentModelLoadEvidence:
    """Reopen exact bytes and optionally bind them to the campaign worker PID set."""
    if backing.relative_path != _RELATIVE_PATH:
        raise ResidentEvidenceError(reason="resident load backing evidence path mismatch")
    try:
        raw = validate_backing_file(root, backing)
        evidence = ResidentModelLoadEvidence.model_validate_json(raw)
    except (EvidenceBackingError, ValidationError) as error:
        message = f"resident load backing evidence invalid: {error}"
        raise ResidentEvidenceError(reason=message) from error
    if raw != canonical_json_bytes(evidence.model_dump(mode="json")):
        raise ResidentEvidenceError(reason="resident load backing evidence is noncanonical")
    observed_pids = tuple(worker.process_id for worker in evidence.workers)
    if expected_worker_pids is not None and observed_pids != expected_worker_pids:
        raise ResidentEvidenceError(reason="resident load worker PID mismatch")
    return evidence


def _worker_evidence(telemetry: WorkerLoadTelemetry) -> ResidentWorkerLoadEvidence:
    if telemetry.ready_device is None or telemetry.process_id is None:
        raise ResidentEvidenceError(reason="resident load evidence requires a ready worker PID")
    try:
        return ResidentWorkerLoadEvidence(
            gpu_index=int(telemetry.gpu_index),
            requested_device=telemetry.requested_device,
            ready_device=telemetry.ready_device,
            load_seconds=telemetry.load_seconds,
            process_id=telemetry.process_id,
            before=_snapshot(telemetry.before),
            after=_snapshot(telemetry.after),
            samples=tuple(_snapshot(sample) for sample in telemetry.samples),
            minimum_host_available_bytes=telemetry.minimum_host_available_bytes,
            maximum_swap_used_bytes=telemetry.maximum_swap_used_bytes,
            minimum_gpu_free_bytes=telemetry.minimum_gpu_free_bytes,
            peak_process_rss_bytes=telemetry.peak_process_rss_bytes,
        )
    except ValidationError as error:
        raise ResidentEvidenceError(reason="resident worker load evidence is invalid") from error


def _snapshot(snapshot: ResourceSnapshot) -> ResourceSnapshotEvidence:
    return ResourceSnapshotEvidence(
        host_available_bytes=snapshot.host_available_bytes,
        swap_used_bytes=snapshot.swap_used_bytes,
        gpu_free_bytes=snapshot.gpu_free_bytes,
        process_rss_bytes=snapshot.process_rss_bytes,
    )
