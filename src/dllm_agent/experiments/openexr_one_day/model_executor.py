"""Model-arm corpus preparation before OpenEXR long search."""

# ruff: noqa: PLR0913

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, override

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CoordinatorError,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
)
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS, ModelArm
from dllm_agent.revision.controller_models import (
    ProbeRunner,
    ProofRequest,
    ProofRunnerResult,
    RevisionRunRequest,
    RevisionSampler,
)
from dllm_agent.revision.models import PocRevisionReport, RevisionConfig
from dllm_agent.revision.pipeline import run_poc_revision

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult


class PocRevisionRunner(Protocol):
    """Narrow Task2 runner seam used by one-day model arms."""

    def __call__(
        self,
        *,
        config: RevisionConfig,
        arm: ModelArm,
        sampler: RevisionSampler,
        run_request: RevisionRunRequest,
        probe: ProbeRunner,
        proof_runner: DeferredProofAdapter,
        artifact_dir: Path,
    ) -> PocRevisionReport:
        """Run one model arm and return the Task2 revision report."""
        ...


class LegacyPocRevisionRunner(Protocol):
    """Compatibility seam for earlier Task5 tests that returned raw snapshots."""

    def __call__(
        self,
        *,
        arm: ModelArm,
        sampler: RevisionSampler,
        proof_runner: DeferredProofAdapter,
        artifact_dir: Path,
    ) -> tuple[bytes, ...]:
        """Run one model arm and return materialized snapshots."""
        ...


@dataclass(frozen=True, slots=True)
class DeferredProofAdapter:
    """Proof adapter proving model preparation never performs exact replay."""

    exact_replay_count: int = 0

    def run(self) -> None:
        """Reject exact proof during model-arm corpus preparation."""
        raise CoordinatorError(reason="exact proof is deferred until post-search")


@dataclass(frozen=True, slots=True)
class ModelPreparationError(CoordinatorError):
    """Typed model-preparation failure for one campaign cell."""

    cell_id: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.cell_id}: {self.reason}"


@dataclass(frozen=True, slots=True)
class _SkippedProbeRunner:
    def run(self, request: ProbeRequest) -> ProbeResult:
        del request
        raise CoordinatorError(reason="production OpenEXR probe is not configured")


@dataclass(frozen=True, slots=True)
class ProductionModelArmExecutor:
    """Prepare model-arm corpora through Task2 runner and frozen bank adapters."""

    run_revision: PocRevisionRunner = field(default_factory=lambda: _run_task2_revision)
    probe: ProbeRunner = field(default_factory=_SkippedProbeRunner)
    legacy_run_revision: LegacyPocRevisionRunner | None = None

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        """Run the requested model arm and persist verified snapshot bytes."""
        if request.arm not in MODEL_ARMS or request.cell.arm != request.arm:
            raise CoordinatorError(reason="model arm execution request mismatch")
        sampler = request.frozen_adapter or request.revision_delegate
        if sampler is None:
            raise CoordinatorError(reason="model arm requires a frozen bank adapter")
        staging_root = request.staging_root
        if staging_root is None:
            staging_root = request.cell.artifact_root / "corpus"
        if self.legacy_run_revision is not None:
            snapshots = self.legacy_run_revision(
                arm=request.arm,
                sampler=sampler,
                proof_runner=DeferredProofAdapter(),
                artifact_dir=request.cell.artifact_root / "model-prep",
            )
            result = _write_snapshots(staging_root, snapshots)
            return ModelArmExecutionResult(
                written_snapshot_count=result.written_snapshot_count,
                written_paths=result.written_paths,
                nfe=result.nfe,
                forward_calls=result.forward_calls,
                gpu_ms=result.gpu_ms,
                bank_ready_elapsed_seconds=request.bank_ready_elapsed_seconds,
            )
        revision_config = _revision_config(request)
        run_request = RevisionRunRequest(
            schema_version="dllm-poc-revision-run/v1",
            arm=request.arm,
            pseudo_feedback_rotation_seed=request.cell.fuzzer_seed,
            particle_seed_base=request.cell.fuzzer_seed,
            proof_seed=0,
        )
        report = self.run_revision(
            config=revision_config,
            arm=request.arm,
            sampler=sampler,
            run_request=run_request,
            probe=self.probe,
            proof_runner=DeferredProofAdapter(),
            artifact_dir=request.cell.artifact_root / "model-prep",
        )
        snapshots = _snapshot_bytes(report)
        result = _write_snapshots(staging_root, snapshots)
        accounting = report.runtime_accounting
        return ModelArmExecutionResult(
            written_snapshot_count=result.written_snapshot_count,
            written_paths=result.written_paths,
            nfe=accounting.recorded_nfe,
            forward_calls=accounting.recorded_forward_calls,
            gpu_ms=accounting.recorded_gpu_ms,
            bank_ready_elapsed_seconds=request.bank_ready_elapsed_seconds,
        )


def _write_snapshots(corpus_root: Path, snapshots: tuple[bytes, ...]) -> ModelArmExecutionResult:
    corpus_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for snapshot in snapshots:
        if not snapshot:
            raise CoordinatorError(reason="model arm produced empty snapshot")
        digest = sha256_bytes(snapshot)
        path = corpus_root / f"model-{digest}.exr"
        if path.exists():
            existing = path.read_bytes()
            if existing != snapshot or len(existing) != len(snapshot):
                raise CoordinatorError(reason="model snapshot artifact mismatch")
        else:
            _ = path.write_bytes(snapshot)
        paths.append(path)
    return ModelArmExecutionResult(
        written_snapshot_count=len(paths),
        written_paths=tuple(paths),
        nfe=0,
        forward_calls=0,
        gpu_ms=0,
        bank_ready_elapsed_seconds=0.0,
    )


def _snapshot_bytes(report: PocRevisionReport) -> tuple[bytes, ...]:
    artifacts = {artifact.snapshot_sha256: artifact for artifact in report.snapshot_artifacts}
    if set(artifacts) != {snapshot.snapshot_sha256 for snapshot in report.snapshots}:
        cell_id = "unknown"
        reason = "Task2 snapshot artifact index mismatch"
        raise ModelPreparationError(cell_id, reason)
    payloads: list[bytes] = []
    for snapshot in report.snapshots:
        artifact = artifacts[snapshot.snapshot_sha256]
        payload = artifact.path.read_bytes()
        if len(payload) != snapshot.byte_length or len(payload) != artifact.size_bytes:
            cell_id = "unknown"
            reason = "Task2 snapshot artifact length mismatch"
            raise ModelPreparationError(cell_id, reason)
        if sha256_bytes(payload) != snapshot.snapshot_sha256:
            cell_id = "unknown"
            reason = "Task2 snapshot artifact hash mismatch"
            raise ModelPreparationError(cell_id, reason)
        payloads.append(payload)
    return tuple(payloads)


def _revision_config(request: ModelArmExecutionRequest) -> RevisionConfig:
    seed = request.cell.fuzzer_seed
    return RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=16,
        epochs=1,
        initial_steps=request.config.candidate_steps,
        revision_steps=request.config.candidate_steps,
        probe_seconds=1,
        fuzzer_seeds=(seed, seed + 1),
        proof_seconds=1,
        controller_mode="online-portfolio",
    )


def _run_task2_revision(
    *,
    config: RevisionConfig,
    arm: ModelArm,
    sampler: RevisionSampler,
    run_request: RevisionRunRequest,
    probe: ProbeRunner,
    proof_runner: DeferredProofAdapter,
    artifact_dir: Path,
) -> PocRevisionReport:
    del arm
    return run_poc_revision(
        config,
        run_request,
        sampler,
        probe,
        _DeferredRevisionProofRunner(proof_runner),
        artifact_dir,
    )


@dataclass(frozen=True, slots=True)
class _SkippedProofResult:
    """Protocol-correct skipped proof result for model preparation."""

    oracle_status: Literal["skipped"] = "skipped"
    input_sha256: None = None
    proof_log_sha256: None = None


@dataclass(frozen=True, slots=True)
class _DeferredRevisionProofRunner:
    """Adapt deferred proof policy to the Task2 proof-runner protocol."""

    adapter: DeferredProofAdapter

    def run(self, request: ProofRequest) -> ProofRunnerResult:
        """Return skipped proof evidence without exact replay."""
        del request
        return _SkippedProofResult()
