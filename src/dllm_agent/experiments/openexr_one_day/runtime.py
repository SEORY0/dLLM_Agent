"""Production dependency composition for the OpenEXR one-day coordinator.

# noqa: SIZE_OK
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final, Protocol, Self, cast

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.openexr_one_day.candidate_bank import (
    ArmBankBinding,
    CandidateBank,
    build_candidate_bank,
    build_frozen_bank_adapters,
)
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellResult,
    CellRunRequest,
    DockerSearchLauncher,
    SearchBuildProvenance,
    run_openexr_cell,
    validate_cell_evidence,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CellOutcome,
    CoordinatorError,
    CorpusReleaseEvidence,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    ResidentPoolSurface,
    _candidate_bank_elapsed_seconds,
    _load_model_prep_result,
    _model_prep_evidence_path,
    _persist_candidate_bank,
    _persist_corpus_release,
    _persist_model_prep_result,
    _promote_treatment_corpus,
    build_campaign_manifest,
    coordinate_campaign,
    load_persisted_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.corpus_seed import seed_cell_corpus
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRequest,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.model_executor import ProductionModelArmExecutor
from dllm_agent.experiments.openexr_one_day.models import (
    ARMS,
    MODEL_ARMS,
    TARGET_COMMIT,
    CellSchedule,
    GpuIndex,
    ModelArm,
    MonotonicSeconds,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    PostSearchOutcome,
    ProductionPostSearch,
    load_post_search_outcome,
)
from dllm_agent.experiments.openexr_one_day.qualification import ProductionGateRunner
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeBinding,
    QualificationSmokeCell,
    SmokeModelPreparation,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_production import (
    ProductionSmokeCellBinding,
)
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    ResidentWorkerBinding,
    ResidentWorkerPool,
    ResourceLimits,
    ResourceSnapshot,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from dllm_agent.revision.controller_models import RevisionSampler
from dllm_agent.runtime.revision_client import RevisionWorkerClient
from dllm_agent.runtime.revision_process import WorkerCpuAffinity, worker_cpu_affinity_for_core
from dllm_agent.targets.openexr.deep_parse import DockerDeepParseReplayer
from dllm_agent.targets.openexr.probe import ProbeConfig, prepare_openexr_probe

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from subprocess import CompletedProcess
    from types import TracebackType

    from tools.runtime.selective_revision_json import ReadyRecord

    from dllm_agent.targets.openexr.probe import PreparedProbe

_MODEL_TIMEOUT_SECONDS: Final = 600.0
_MIN_MEMINFO_PARTS: Final = 2
_MODEL_ARMS: Final[tuple[ModelArm, ...]] = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
)


class RevisionWorkerFactory(Protocol):
    """Factory for production resident revision workers."""

    def create(self, *, device: str, cpu_affinity: WorkerCpuAffinity) -> ResidentRevisionWorker:
        """Create a worker client for one CUDA device."""
        ...


class ResidentRevisionWorker(RevisionSampler, Protocol):
    """Worker surface that is both resident and usable for revision sampling."""

    device: str
    cache_dir: Path
    read_only_cache: bool
    ready: ReadyRecord | None

    @property
    def process_id_if_started(self) -> int | None:
        """Return the started process ID, if any."""
        ...

    @property
    def last_process_id(self) -> int | None:
        """Return the most recent process ID, if any."""
        ...

    @property
    def is_alive(self) -> bool:
        """Return whether the worker is alive."""
        ...

    def __enter__(self) -> Self:
        """Enter the worker process."""
        ...

    def close(self) -> None:
        """Close the worker process."""
        ...


@dataclass(frozen=True, slots=True)
class DefaultRevisionWorkerFactory:
    """Build production `RevisionWorkerClient` instances."""

    runtime_python: Path
    cache_dir: Path
    prompt_path: Path
    prompt_sha256: str

    def create(self, *, device: str, cpu_affinity: WorkerCpuAffinity) -> RevisionWorkerClient:
        """Create one read-only-cache revision worker client."""
        return RevisionWorkerClient(
            runtime_python=self.runtime_python,
            cache_dir=self.cache_dir,
            prompt_path=self.prompt_path,
            prompt_sha256=self.prompt_sha256,
            device=device,
            timeout_seconds=_MODEL_TIMEOUT_SECONDS,
            read_only_cache=True,
            cpu_affinity=cpu_affinity,
        )


class OneDayRuntimeFactory(Protocol):
    """Runtime factory surface shared by CLI production and tests."""

    def build(self, config: CampaignConfig) -> RuntimeDependencyGraph:
        """Construct runtime dependencies without starting side effects."""
        ...

    def run(self, config: CampaignConfig) -> CampaignOutcome:
        """Run the qualified one-day campaign."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeDependencyGraph:
    """Concrete production objects reachable by a qualified one-day run."""

    resident_pool: ResidentWorkerPool
    cell_runner: ProductionCellRunner
    model_executor: ProductionModelArmExecutor
    post_search: ProductionPostSearch
    gate_runner: ProductionGateRunner
    probe_context: AbstractContextManager[PreparedProbe] | None = None
    probe_context_entered: bool = False

    def __enter__(self) -> Self:
        """Enter graph-owned resources and inject the prepared probe."""
        if self.probe_context is not None and not self.probe_context_entered:
            probe = self.probe_context.__enter__()
            object.__setattr__(
                self,
                "model_executor",
                ProductionModelArmExecutor(probe=probe),
            )
            object.__setattr__(self, "probe_context_entered", True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release graph-owned probe resources."""
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Close graph-owned resources."""
        if self.probe_context is not None and self.probe_context_entered:
            _ = self.probe_context.__exit__(None, None, None)
            object.__setattr__(self, "probe_context_entered", False)

    @property
    def concrete_dependency_names(self) -> tuple[str, ...]:
        """Expose dependency class names for composition smoke tests."""
        return (
            "RevisionWorkerClient",
            self.resident_pool.__class__.__name__,
            "DockerSearchLauncher",
            "OpenExrProofRunner",
        )

    @property
    def bank_sampler(self) -> RevisionSampler:
        """Compatibility view over the first entered-worker bank sampler."""
        return cast("RevisionSampler", cast("object", self.resident_pool.bindings[0].worker))

    @property
    def revision_delegates(self) -> dict[ModelArm, RevisionSampler]:
        """Compatibility view over the four production arm workers."""
        return {
            arm: cast("RevisionSampler", cast("object", self.resident_pool.bindings[index].worker))
            for index, arm in enumerate(_MODEL_ARMS)
        }


@dataclass(frozen=True, slots=True)
class ProductionQualificationSmokeAdapter:
    """Bind the real one-day runtime graph to the canonical qualification smoke."""

    smoke_root: Path
    resident_pool: ResidentWorkerPool
    cell_runner: ProductionCellRunner
    model_executor: ProductionModelArmExecutor
    post_search: ProductionPostSearch

    def build_candidate_bank(self, config: CampaignConfig, repetition: int) -> CandidateBank:
        """Build or reload one real smoke candidate bank under the smoke root."""
        path = _smoke_bank_path(self.smoke_root, repetition)
        if path.exists():
            return load_persisted_candidate_bank(path, repetition)
        schedule = build_campaign_schedule(
            main_start_monotonic_seconds=config.main_start_monotonic_seconds
        )
        sampler = _smoke_bank_sampler(schedule.cells, self.resident_pool, repetition)
        producer = _smoke_producer_gpu(schedule.cells, repetition)
        started = monotonic()
        bank = build_candidate_bank(
            sampler,
            schedule.seed_namespaces[repetition],
            steps=config.candidate_steps,
        )
        elapsed = max(0.0, monotonic() - started)
        _persist_candidate_bank(
            path,
            repetition,
            bank,
            elapsed,
            bank_ready_elapsed_seconds=elapsed,
            producer_gpu_index=producer,
        )
        return bank

    def bind_cell_request(
        self,
        config: CampaignConfig,
        plan_cell: QualificationSmokeCell,
        bank: CandidateBank,
    ) -> ProductionSmokeCellBinding:
        """Build the concrete smoke cell request and seed its private C0 corpus."""
        request = _smoke_cell_request(config, plan_cell, bank.selected_bank_sha256)
        _ = seed_cell_corpus(config, _smoke_pending_cell(config, plan_cell, request), None)
        return ProductionSmokeCellBinding(
            config=config, plan_cell=plan_cell, request=request, bank=bank
        )

    def bind_persisted_cell_request(
        self,
        config: CampaignConfig,
        plan_cell: QualificationSmokeCell,
        binding: QualificationSmokeBinding,
    ) -> ProductionSmokeCellBinding:
        """Rebind a persisted smoke cell without regenerating banks or corpora."""
        bank = load_persisted_candidate_bank(
            _smoke_bank_path(self.smoke_root, plan_cell.repetition),
            plan_cell.repetition,
        )
        request = _smoke_cell_request(config, plan_cell, binding.bank_sha256)
        return ProductionSmokeCellBinding(
            config=config, plan_cell=plan_cell, request=request, bank=bank
        )

    def prepare_model_corpus(self, binding: ProductionSmokeCellBinding) -> SmokeModelPreparation:
        """Prepare and release the real model corpus for one smoke model arm."""
        prep_path = _model_prep_evidence_path(binding.request)
        if prep_path.exists():
            result = _load_model_prep_result(prep_path, binding.request.cell_id)
        else:
            result = self.model_executor.prepare_corpus(
                _smoke_model_request(
                    binding=binding,
                    resident_pool=self.resident_pool,
                    smoke_root=self.smoke_root,
                    bank_ready_elapsed_seconds=_candidate_bank_elapsed_seconds(
                        _smoke_bank_path(self.smoke_root, binding.plan_cell.repetition)
                    ),
                )
            )
            result = _persist_smoke_model_prep(binding, result)
        if not (binding.request.artifact_root / "corpus-release.json").exists():
            _persist_smoke_corpus_release(binding, result)
        return SmokeModelPreparation(
            cell_id=binding.plan_cell.cell_id,
            preparation_sha256=sha256_bytes(prep_path.read_bytes()),
        )

    def run_cell(self, binding: ProductionSmokeCellBinding) -> CellResult:
        """Run the real Docker-backed smoke cell."""
        return self.cell_runner.run(binding.request)

    def validate_cell(self, binding: ProductionSmokeCellBinding) -> CellResult:
        """Reload and validate persisted smoke cell evidence."""
        return validate_cell_evidence(binding.request)

    def run_post_search(
        self, binding: ProductionSmokeCellBinding, result: CellResult
    ) -> PostSearchOutcome:
        """Build real coverage, proof, and deep-parse envelopes for one smoke cell."""
        return self.post_search.coverage_and_proof(binding.request, result)

    def load_post_search(
        self, binding: ProductionSmokeCellBinding, result: CellResult
    ) -> PostSearchOutcome:
        """Reload persisted smoke post-search envelopes without rebuilding them."""
        return load_post_search_outcome(binding.request, result)


def _smoke_bank_path(smoke_root: Path, repetition: int) -> Path:
    return smoke_root / "banks" / f"rep-{repetition:02d}" / "candidate-bank.json"


def _smoke_cell_request(
    config: CampaignConfig,
    plan_cell: QualificationSmokeCell,
    bank_sha256: str,
) -> CellRunRequest:
    return CellRunRequest(
        execution_profile="qualification-smoke",
        schedule_seconds=(0, 300),
        cell_id=plan_cell.cell_id,
        artifact_root=plan_cell.artifact_root,
        corpus_root=plan_cell.artifact_root / "corpus",
        crash_root=plan_cell.artifact_root / "crashes",
        log_root=plan_cell.artifact_root / "logs",
        target_commit=TARGET_COMMIT,
        harness_sha256=config.target_harness_sha256,
        config_sha256=config.config_sha256,
        bank_sha256=bank_sha256,
        start_monotonic_seconds=plan_cell.start_monotonic_seconds,
        deadline_monotonic_seconds=plan_cell.deadline_monotonic_seconds,
        fuzzer_seed=plan_cell.fuzzer_seed,
        max_input_length=config.max_input_length,
        target_timeout_seconds=config.target_timeout_seconds,
        rss_limit_mb=config.rss_limit_mb,
        cpu_index=config.resource_map.search_cores[_smoke_search_core_index(plan_cell)].core_id,
        dictionary_path=config.dictionary_path,
        dictionary_sha256=config.dictionary_sha256,
        build=SearchBuildProvenance(
            image_digest=config.search_image_digest,
            binary_sha256=config.search_binary_sha256,
            sanitizers=("address", "undefined"),
        ),
    )


def _smoke_model_request(
    *,
    binding: ProductionSmokeCellBinding,
    resident_pool: ResidentWorkerPool,
    smoke_root: Path,
    bank_ready_elapsed_seconds: float,
) -> ModelArmExecutionRequest:
    plan_cell = binding.plan_cell
    schedule_cells = _smoke_schedule_cells(plan_cell.start_monotonic_seconds)
    adapters = _smoke_adapters(
        schedule_cells, resident_pool, binding.bank, smoke_root, plan_cell.repetition
    )
    schedule_cell = _matching_smoke_schedule_cell(schedule_cells, plan_cell)
    bank_sampler = _smoke_bank_sampler(schedule_cells, resident_pool, plan_cell.repetition)
    if schedule_cell.gpu_index is None:
        raise CoordinatorError(reason="smoke model cell has no producer GPU")
    staging_root = binding.request.artifact_root / "model-prep-staging"
    return ModelArmExecutionRequest(
        config=binding.config,
        cell=_smoke_pending_cell_from_schedule(plan_cell, binding.request, schedule_cell),
        bank_sampler=bank_sampler,
        frozen_adapter=adapters[cast("ModelArm", plan_cell.arm)],
        arm=cast("ModelArm", plan_cell.arm),
        bank_sampler_id=id(bank_sampler),
        revision_delegate_id=id(
            getattr(
                adapters[cast("ModelArm", plan_cell.arm)],
                "revise_delegate",
                adapters[cast("ModelArm", plan_cell.arm)],
            )
        ),
        bank_ready_elapsed_seconds=bank_ready_elapsed_seconds,
        bank_physical_duration_seconds=bank_ready_elapsed_seconds,
        staging_root=staging_root / "corpus",
        live_corpus_root=binding.request.corpus_root,
        producer_gpu_index=int(schedule_cell.gpu_index),
    )


def _persist_smoke_model_prep(
    binding: ProductionSmokeCellBinding,
    result: ModelArmExecutionResult,
) -> ModelArmExecutionResult:
    index_request = CorpusIndexRequest(
        execution_profile="qualification-smoke",
        schedule_seconds=(0, 300),
        artifact_root=binding.request.artifact_root,
        corpus_root=binding.request.corpus_root,
        store_root=binding.request.artifact_root / "corpus-inputs",
        index_path=binding.request.artifact_root / "corpus-index.jsonl",
    )
    rows = load_corpus_index(index_request)
    if not rows:
        rows = (snapshot_corpus(index_request, 0, 0.0),)
    enriched = replace(
        result,
        pre_release_corpus_set_sha256=rows[-1].cumulative_set_sha256,
        producer_gpu_index=_smoke_model_gpu_index(binding.plan_cell),
    )
    _persist_model_prep_result(
        _model_prep_evidence_path(binding.request),
        binding.request.cell_id,
        enriched,
        bank_physical_duration_seconds=enriched.bank_ready_elapsed_seconds,
    )
    return _load_model_prep_result(
        _model_prep_evidence_path(binding.request),
        binding.request.cell_id,
    )


def _persist_smoke_corpus_release(
    binding: ProductionSmokeCellBinding,
    result: ModelArmExecutionResult,
) -> None:
    if result.pre_release_corpus_set_sha256 is None:
        raise CoordinatorError(reason="smoke model prep missing pre-release corpus hash")
    promoted = _promote_treatment_corpus(binding.request, binding.bank, result)
    _ = seed_cell_corpus(
        binding.config,
        _smoke_pending_cell(binding.config, binding.plan_cell, binding.request),
        binding.bank,
    )
    release = CorpusReleaseEvidence(
        cell_id=binding.request.cell_id,
        bank_sha256=binding.bank.selected_bank_sha256,
        bank_physical_duration_seconds=result.bank_ready_elapsed_seconds,
        bank_ready_elapsed_seconds=result.bank_ready_elapsed_seconds,
        model_ready_elapsed_seconds=result.model_ready_elapsed_seconds,
        corpus_release_elapsed_seconds=0.0,
        pre_release_corpus_set_sha256=result.pre_release_corpus_set_sha256,
        producer_gpu_index=_smoke_model_gpu_index(binding.plan_cell),
        promoted_sha256=promoted,
    )
    _persist_corpus_release(binding.request.artifact_root / "corpus-release.json", release)


def _smoke_pending_cell(
    config: CampaignConfig,
    plan_cell: QualificationSmokeCell,
    request: CellRunRequest,
) -> CellOutcome:
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=plan_cell.repetition,
        wave_index=0,
        arm=plan_cell.arm,
        fuzzer_seed=plan_cell.fuzzer_seed,
        deadline_monotonic_seconds=plan_cell.deadline_monotonic_seconds,
        artifact_root=request.artifact_root,
        mutable_root=config.artifact_root
        / "smoke-mutable"
        / f"rep-{plan_cell.repetition:02d}"
        / plan_cell.arm,
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=sha256_bytes(
            f"{config.config_sha256}:{plan_cell.repetition}:{request.bank_sha256}".encode()
        )
        if plan_cell.arm in MODEL_ARMS
        else "0" * 64,
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason="completed",
        coverage_frozen_through_seconds=None,
        search_cpu_seconds=0.0,
    )


def _smoke_pending_cell_from_schedule(
    plan_cell: QualificationSmokeCell,
    request: CellRunRequest,
    schedule_cell: CellSchedule,
) -> CellOutcome:
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=plan_cell.repetition,
        wave_index=0,
        arm=plan_cell.arm,
        fuzzer_seed=plan_cell.fuzzer_seed,
        deadline_monotonic_seconds=plan_cell.deadline_monotonic_seconds,
        artifact_root=request.artifact_root / "model-prep-staging",
        mutable_root=request.artifact_root / "mutable",
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=sha256_bytes(
            f"{request.config_sha256}:{int(schedule_cell.repetition)}:{request.bank_sha256}".encode()
        ),
        config_sha256=request.config_sha256,
        resource_map_sha256="0" * 64,
        manifest_sha256="0" * 64,
        terminal_reason="completed",
        coverage_frozen_through_seconds=None,
        search_cpu_seconds=0.0,
    )


def _smoke_schedule_cells(start_monotonic_seconds: float) -> tuple[CellSchedule, ...]:
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=MonotonicSeconds(start_monotonic_seconds)
    )
    return schedule.waves[0].cells


def _matching_smoke_schedule_cell(
    cells: tuple[CellSchedule, ...], plan_cell: QualificationSmokeCell
) -> CellSchedule:
    for cell in cells:
        if int(cell.repetition) == plan_cell.repetition and cell.arm == plan_cell.arm:
            return cell
    raise CoordinatorError(reason="smoke schedule cell missing")


def _smoke_adapters(
    cells: tuple[CellSchedule, ...],
    resident_pool: ResidentWorkerPool,
    bank: CandidateBank,
    smoke_root: Path,
    repetition: int,
) -> dict[ModelArm, RevisionSampler]:
    bindings = tuple(
        ArmBankBinding(
            cell.arm,
            cast("RevisionSampler", cast("object", resident_pool.worker_for(cell))),
            smoke_root / "mutable" / f"rep-{int(cell.repetition):02d}" / cell.arm,
        )
        for cell in cells
        if cell.arm in MODEL_ARMS and int(cell.repetition) == repetition
    )
    return {adapter.arm: adapter for adapter in build_frozen_bank_adapters(bank, bindings)}


def _smoke_bank_sampler(
    cells: tuple[CellSchedule, ...],
    resident_pool: ResidentWorkerPool,
    repetition: int,
) -> RevisionSampler:
    for cell in cells:
        if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS:
            return cast("RevisionSampler", cast("object", resident_pool.worker_for(cell)))
    raise CoordinatorError(reason="smoke bank sampler missing")


def _smoke_producer_gpu(cells: tuple[CellSchedule, ...], repetition: int) -> int:
    for cell in cells:
        if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS:
            if cell.gpu_index is None:
                raise CoordinatorError(reason="smoke bank producer missing")
            return int(cell.gpu_index)
    raise CoordinatorError(reason="smoke bank producer missing")


def _smoke_model_gpu_index(plan_cell: QualificationSmokeCell) -> int:
    cell = _matching_smoke_schedule_cell(
        _smoke_schedule_cells(plan_cell.start_monotonic_seconds),
        plan_cell,
    )
    if cell.gpu_index is None:
        raise CoordinatorError(reason="smoke model cell has no producer GPU")
    return int(cell.gpu_index)


def _smoke_search_core_index(plan_cell: QualificationSmokeCell) -> int:
    return (plan_cell.repetition * len(ARMS)) + ARMS.index(plan_cell.arm)


@dataclass(frozen=True, slots=True)
class ProductionRuntimeFactory:
    """Build the real production dependency graph without starting side effects."""

    runtime_python: Path
    source_root: Path
    cache_dir: Path
    prompt_path: Path
    qualification_evidence_path: Path = Path("qualification.json")
    worker_factory: RevisionWorkerFactory | None = None

    def build(self, config: CampaignConfig) -> RuntimeDependencyGraph:
        """Construct production classes; workers and Docker start only during run."""
        factory = self.worker_factory or DefaultRevisionWorkerFactory(
            runtime_python=self.runtime_python,
            cache_dir=self.cache_dir,
            prompt_path=self.prompt_path,
            prompt_sha256=config.prompt_sha256,
        )
        affinities = tuple(
            worker_cpu_affinity_for_core(assignment.core_id, assignment.sibling_core_id)
            for assignment in config.resource_map.model_cores
        )
        workers: tuple[ResidentRevisionWorker, ...] = tuple(
            factory.create(device=f"cuda:{index}", cpu_affinity=affinities[index])
            for index in range(8)
        )
        pool = ResidentWorkerPool(
            bindings=tuple(
                ResidentWorkerBinding(GpuIndex(index), worker)
                for index, worker in enumerate(workers)
            ),
            telemetry_probe=ProductionTelemetryProbe(),
            limits=ResourceLimits(max_process_rss_bytes=64 * 1024**3),
        )
        probe_context = prepare_openexr_probe(
            ProbeConfig(
                image_id=config.search_image_digest,
                harness_sha256=config.target_harness_sha256,
                rss_limit_mb=config.rss_limit_mb,
                timeout_seconds=config.target_timeout_seconds,
            )
        )
        return RuntimeDependencyGraph(
            resident_pool=pool,
            cell_runner=ProductionCellRunner(),
            model_executor=ProductionModelArmExecutor(),
            post_search=ProductionPostSearch(
                config=config,
                deep_parse_replayer=DockerDeepParseReplayer(config.deep_parse_checker_sha256),
            ),
            gate_runner=ProductionGateRunner(self.qualification_evidence_path),
            probe_context=probe_context,
        )

    def run(self, config: CampaignConfig) -> CampaignOutcome:
        """Run a qualified campaign with the production dependency graph."""
        with self.build(config) as graph:
            return coordinate_campaign(
                config,
                resident_pool=cast("ResidentPoolSurface", cast("object", graph.resident_pool)),
                cell_runner=graph.cell_runner,
                post_search=graph.post_search,
                gate_runner=graph.gate_runner,
                model_executor=graph.model_executor,
            )


@dataclass(frozen=True, slots=True)
class ProductionCellRunner:
    """Adapter from coordinator cell requests to the Task4 long-search runner."""

    launcher: DockerSearchLauncher = field(default_factory=DockerSearchLauncher)

    def wave_monotonic(self, scheduled_start: float) -> float:
        """Return the real clock used for wave-start and deadline decisions."""
        del scheduled_start
        return monotonic()

    def run(self, request: CellRunRequest) -> CellResult:
        """Run one long-search cell with the production Docker launcher."""
        index_request = CorpusIndexRequest(
            artifact_root=request.artifact_root,
            corpus_root=request.corpus_root,
            store_root=request.artifact_root / "corpus-inputs",
            index_path=request.artifact_root / "corpus-index.jsonl",
        )
        prefix = load_corpus_index(index_request) if index_request.index_path.exists() else ()
        return run_openexr_cell(
            request,
            self.launcher,
            _ProductionCellClock(),
            preinitialized_corpus_index=prefix,
        )

    def cancel_active(self) -> None:
        """Cancel and reap every Docker search owned by this runner."""
        self.launcher.cancel_active()


class ProcReader(Protocol):
    """Shell-free `/proc` reader."""

    def meminfo(self) -> str:
        """Read `/proc/meminfo`."""
        ...

    def status(self, pid: int) -> str:
        """Read `/proc/<pid>/status`."""
        ...


class CommandRunner(Protocol):
    """Shell-free command runner."""

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Run one argv vector without a shell."""
        ...


@dataclass(frozen=True, slots=True)
class FileProcReader:
    """Read Linux procfs telemetry files."""

    root: Path = Path("/proc")

    def meminfo(self) -> str:
        """Read host memory info."""
        return (self.root / "meminfo").read_text(encoding="utf-8")

    def status(self, pid: int) -> str:
        """Read one process status file."""
        return (self.root / str(pid) / "status").read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class SubprocessCommandRunner:
    """Run telemetry commands without invoking a shell."""

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Run one command with captured byte output."""
        return subprocess.run(argv, check=False, capture_output=True)  # noqa: S603


@dataclass(frozen=True, slots=True)
class ProductionTelemetryProbe:
    """Conservative telemetry adapter used by the resident pool."""

    proc_reader: ProcReader = field(default_factory=FileProcReader)
    command_runner: CommandRunner = field(default_factory=SubprocessCommandRunner)

    def capture(self, gpu_index: GpuIndex, process_id: int | None) -> ResourceSnapshot:
        """Capture host/GPU resource evidence for a resident-worker load gate."""
        meminfo = _parse_meminfo(self.proc_reader.meminfo())
        rss = 0 if process_id is None else _parse_status_rss(self.proc_reader.status(process_id))
        gpu = _gpu_free_bytes(self.command_runner, int(gpu_index))
        return ResourceSnapshot(
            host_available_bytes=meminfo.mem_available_kib * 1024,
            swap_used_bytes=(meminfo.swap_total_kib - meminfo.swap_free_kib) * 1024,
            gpu_free_bytes=gpu,
            process_rss_bytes=rss,
        )


@dataclass(frozen=True, slots=True)
class _MemInfo:
    mem_available_kib: int
    swap_free_kib: int
    swap_total_kib: int


def _parse_meminfo(text: str) -> _MemInfo:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= _MIN_MEMINFO_PARTS:
            key = parts[0].removesuffix(":")
            values[key] = int(parts[1])
    return _MemInfo(
        mem_available_kib=values["MemAvailable"],
        swap_free_kib=values["SwapFree"],
        swap_total_kib=values["SwapTotal"],
    )


def _parse_status_rss(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _gpu_free_bytes(runner: CommandRunner, gpu_index: int) -> int:
    process = runner.run(
        (
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        )
    )
    if process.returncode == 0:
        return int(process.stdout.decode("ascii").strip().split()[0]) * 1024 * 1024
    raise CoordinatorError(reason="nvidia-smi telemetry failed")


class _ProductionCellClock:
    """Production monotonic clock for Task4 search cells."""

    def monotonic(self) -> float:
        """Return current monotonic time."""
        return monotonic()

    def process_time(self) -> float:
        """Return current process CPU time."""
        return monotonic()
