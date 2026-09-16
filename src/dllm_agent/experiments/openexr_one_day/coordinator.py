"""Task 5 coordinator for the OpenEXR one-day campaign.

# noqa: SIZE_OK
"""

# pyright: reportUnnecessaryComparison=false
# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false
# ruff: noqa: C901, EM101, PLR0913, TRY003

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic, sleep
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    assert_never,
    cast,
    override,
)

from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from tools.selective_revision_payloads import sha256_token_ids
from tools.stable_diffcoder_generation import (
    DuplicateParent,
    FrozenParent,
    revision_trajectory_sha256,
)

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import (
    JsonValue,
    LedgerError,
    append_event,
    validate_ledger,
)
from dllm_agent.experiments.openexr_one_day.c0_source import C0SourceError
from dllm_agent.experiments.openexr_one_day.candidate_bank import (
    ArmBankBinding,
    CandidateBank,
    build_candidate_bank,
    build_frozen_bank_adapters,
)
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellEvidenceError,
    CellResult,
    CellRunRequest,
    SearchBuildProvenance,
    validate_cell_evidence,
)
from dllm_agent.experiments.openexr_one_day.corpus_seed import (
    CorpusSeedSummary,
    seed_cell_corpus,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRecord,
    CorpusIndexRequest,
    CoverageSeriesError,
    freeze_corpus_index,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,  # noqa: TC001  # Pydantic resolves this field at runtime.
)
from dllm_agent.experiments.openexr_one_day.failure_taxonomy import ExogenousCampaignError
from dllm_agent.experiments.openexr_one_day.models import (
    MODEL_ARMS,
    TARGET_COMMIT,
    Arm,
    CampaignSchedule,
    CellSchedule,
    ModelArm,
    MonotonicSeconds,
    SeedNamespace,
    WaveSchedule,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    CoverageSeriesEnvelope,
    DeepParseEnvelope,
    PostSearchOutcome,
    ProofBatchEnvelope,
    load_post_search_outcome,
)
from dllm_agent.experiments.openexr_one_day.resident_evidence import (
    ResidentEvidenceError,
    ResidentModelLoadEvidence,
    build_resident_model_load_evidence,
    load_resident_model_load_evidence,
    persist_resident_model_load_evidence,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    RevisionResult,
    RevisionSampler,
)
from dllm_agent.runtime.revision_process import BankGenerationCost
from dllm_agent.targets.openexr.particle import decode_particle, materialize_particle

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from dllm_agent.experiments.openexr_one_day.resident_pool import WorkerLoadTelemetry

_EVENTS_NAME: Final = "campaign-events.jsonl"
_MANIFEST_NAME: Final = "campaign-manifest.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COMPLETE_CELL_COUNT: Final = 50
_WAVE_DURATION_SECONDS: Final = 10_800.0

type CoreInput = CoreAssignment | dict[str, JsonValue]
type ScheduleJson = dict[str, JsonValue]
type RawArtifactStatus = Literal["available", "unavailable"]


class CampaignValidationError(ValueError):
    """Raised by Pydantic validators for invalid campaign config data."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the validation failure reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CoordinatorError(Exception):
    """Raised when the campaign must fail closed."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the coordination failure reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CoreAssignment(FrozenModel):
    """Physical core plus explicit SMT sibling evidence."""

    core_id: int = Field(ge=0)
    sibling_core_id: int = Field(ge=0)

    @field_validator("core_id", "sibling_core_id", mode="before")
    @classmethod
    def _parse_cpu_id(cls, value: JsonValue) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise CampaignValidationError("resource-map CPU identifiers must be integers")
        return value


class GpuIdentityRecord(FrozenModel):
    """Explicit GPU identity and NUMA evidence."""

    index: int = Field(ge=0)
    uuid: str = Field(pattern=r"^GPU-[0-9A-Za-z-]+$")
    numa_node: int = Field(ge=0)

    @field_validator("index", "numa_node", mode="before")
    @classmethod
    def _parse_gpu_int(cls, value: JsonValue) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise CampaignValidationError("resource-map GPU identifiers must be integers")
        return value


class ResourceMap(FrozenModel):
    """Immutable campaign CPU/GPU/NUMA resource map."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    model_cores: tuple[CoreAssignment, ...] = Field(min_length=1, max_length=8)
    search_cores: tuple[CoreAssignment, ...] = Field(min_length=10, max_length=10)
    proof_cores: tuple[CoreAssignment, ...] = Field(min_length=1)
    monitor_cores: tuple[CoreAssignment, ...] = Field(min_length=1)
    gpu_numa_nodes: tuple[int, ...] = Field(min_length=1, max_length=8)
    gpu_records: tuple[GpuIdentityRecord, ...] = ()
    core_numa_nodes: dict[int, int] = Field(min_length=20)

    @field_validator("model_cores", "search_cores", "proof_cores", "monitor_cores", mode="before")
    @classmethod
    def _parse_cores(
        cls, value: tuple[CoreInput, ...] | list[CoreInput]
    ) -> tuple[CoreAssignment, ...]:
        if not isinstance(value, list | tuple):
            raise CampaignValidationError("resource-map cores must be a sequence")
        items = tuple(value)
        parsed = [
            item if isinstance(item, CoreAssignment) else CoreAssignment.model_validate(item)
            for item in items
        ]
        return tuple(parsed)

    @field_validator("gpu_numa_nodes", mode="before")
    @classmethod
    def _parse_gpu_numa(cls, value: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        if not isinstance(value, list | tuple):
            raise CampaignValidationError("resource-map GPU NUMA evidence must be a sequence")
        parsed: list[int] = []
        for item in value:
            if not isinstance(item, int) or isinstance(item, bool):
                raise CampaignValidationError("resource-map GPU NUMA evidence must be integer")
            parsed.append(item)
        return tuple(parsed)

    @field_validator("core_numa_nodes", mode="before")
    @classmethod
    def _parse_core_numa(cls, value: dict[int | str, int]) -> dict[int, int]:
        if not isinstance(value, dict):
            raise CampaignValidationError("resource-map core NUMA evidence must be an object")
        parsed: dict[int, int] = {}
        for key, item in value.items():
            if (
                not isinstance(key, int | str)
                or isinstance(key, bool)
                or not isinstance(item, int)
                or isinstance(item, bool)
            ):
                raise CampaignValidationError(
                    "resource-map core NUMA evidence must map integer cores"
                )
            parsed[int(key)] = item
        return parsed

    @model_validator(mode="after")
    def _valid_resource_map(self) -> Self:  # noqa: PLR0912
        assignments = self.model_cores + self.search_cores + self.proof_cores + self.monitor_cores
        cores = tuple(item.core_id for item in assignments)
        siblings = tuple(item.sibling_core_id for item in assignments)
        if len(set(cores)) != len(cores):
            raise CampaignValidationError("resource-map duplicate CPU core")
        if set(cores) & set(siblings):
            raise CampaignValidationError("resource-map uses SMT siblings")
        if any(core not in self.core_numa_nodes for core in cores + siblings):
            raise CampaignValidationError("resource-map missing CPU topology evidence")
        expected_topology = set(range(max(self.core_numa_nodes) + 1))
        if set(self.core_numa_nodes) != expected_topology:
            raise CampaignValidationError("resource-map CPU topology evidence is incomplete")
        if len(set(siblings)) != len(siblings):
            raise CampaignValidationError("resource-map duplicate SMT sibling")
        for assignment in assignments:
            if (
                self.core_numa_nodes[assignment.core_id]
                != self.core_numa_nodes[assignment.sibling_core_id]
            ):
                raise CampaignValidationError("resource-map CPU sibling NUMA mismatch")
        if len(self.model_cores) != len(self.gpu_numa_nodes):
            raise CampaignValidationError("resource-map model core count must match GPU count")
        if any(node < 0 for node in self.gpu_numa_nodes):
            raise CampaignValidationError("resource-map GPU NUMA evidence is incomplete")
        if self.gpu_records:
            expected_count = len(self.gpu_numa_nodes)
            # record.index is the real physical CUDA/nvidia-smi device for
            # that pool slot -- not required to be 0-based/contiguous, since
            # a config may pin a smaller GPU count to specific physical
            # devices on a larger box (e.g. slot 0 -> physical GPU 1).
            if len(self.gpu_records) != expected_count:
                raise CampaignValidationError("resource-map GPU identity evidence is incomplete")
            if len({record.index for record in self.gpu_records}) != expected_count:
                raise CampaignValidationError("resource-map duplicate physical GPU index")
            if tuple(record.numa_node for record in self.gpu_records) != self.gpu_numa_nodes:
                raise CampaignValidationError("resource-map GPU identity NUMA mismatch")
            if len({record.uuid for record in self.gpu_records}) != len(self.gpu_numa_nodes):
                raise CampaignValidationError("resource-map duplicate GPU identity")
        for gpu_index, assignment in enumerate(self.model_cores):
            if self.core_numa_nodes.get(assignment.core_id) != self.gpu_numa_nodes[gpu_index]:
                raise CampaignValidationError("resource-map NUMA/GPU mismatch")
        return self


class CampaignConfig(FrozenModel):
    """Boundary-parsed one-day campaign configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    schema_version: Literal["dllm-openexr-one-day-config/v1"]
    config_id: str = Field(min_length=1)
    qualification_state: Literal["draft", "qualified"]
    artifact_root: Path
    main_start_monotonic_seconds: MonotonicSeconds
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    target_harness_sha256: Sha256
    prompt_sha256: Sha256
    model_revision: str = Field(min_length=1)
    remote_code_sha256: Sha256
    config_sha256: Sha256
    search_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    search_binary_sha256: Sha256
    sanitizers: tuple[Literal["address", "undefined"], ...]
    coverage_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    coverage_binary_sha256: Sha256
    coverage_build_manifest_sha256: Sha256
    deep_parse_checker_sha256: Sha256 = (
        "621ca050c5b65d5537885288d9f42bb7bcee8d9d038d5ee4782e75b6e01372ba"
    )
    c0_source_path: Path = Path("fixtures/openexr/c0-validity-preserving-zero.hex")
    c0_sha256: Sha256
    dictionary_path: Path = Path("configs/openexr-v3.4.14.dict")
    dictionary_sha256: Sha256
    max_input_length: int = Field(gt=0)
    target_timeout_seconds: int = Field(gt=0)
    rss_limit_mb: int = Field(gt=0)
    candidate_steps: int = Field(gt=0)
    proof_duration_seconds: int = Field(gt=0)
    proof_seed_base: int = Field(default=9_000_000, ge=0)
    runtime_python: Path = Path("/usr/bin/python3")
    model_cache_dir: Path = Path("/data/dllm-agent/cache")
    prompt_path: Path = Path("prompts/openexr-system.md")
    qualification_evidence_path: Path = Path("qualification.json")
    resource_map: ResourceMap

    @field_validator(
        "artifact_root",
        "runtime_python",
        "model_cache_dir",
        "prompt_path",
        "qualification_evidence_path",
        "dictionary_path",
        mode="before",
    )
    @classmethod
    def _parse_path(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise CampaignValidationError("artifact root must be a path")

    @field_validator("sanitizers", mode="before")
    @classmethod
    def _parse_sanitizers(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise CampaignValidationError("sanitizers must be a sequence")
        parsed: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise CampaignValidationError("sanitizers must be strings")
            parsed.append(item)
        return tuple(parsed)

    @model_validator(mode="after")
    def _valid_config(self) -> Self:
        _require_private_data_root(self.artifact_root)
        _validate_dictionary_file(self.dictionary_path, self.dictionary_sha256)
        if self.qualification_state == "qualified" and not self.resource_map.gpu_records:
            raise CampaignValidationError("qualified config requires GPU identity evidence")
        return self


class StartGateEvidence(FrozenModel):
    """Explicit evidence for every main NO-START gate."""

    hashes_match: bool
    banks_complete: bool
    read_only_cache: bool
    external_coverage_check: bool
    control_arm_equivalence: bool
    resource_map_valid: bool
    disk_space: bool
    vram: bool
    ram: bool
    swap_stable: bool
    reserved_core_load: bool
    smoke_passed: bool

    def failing_gate(self) -> str | None:
        """Return the first missing gate in preregistered order."""
        for name in (
            "hashes_match",
            "banks_complete",
            "read_only_cache",
            "external_coverage_check",
            "control_arm_equivalence",
            "resource_map_valid",
            "disk_space",
            "vram",
            "ram",
            "swap_stable",
            "reserved_core_load",
            "smoke_passed",
        ):
            if not getattr(self, name):
                return name.replace("_", "-")
        return None


class CampaignManifest(FrozenModel):
    """Immutable campaign manifest persisted before launch."""

    schema_version: Literal["dllm-openexr-campaign-manifest/v1"] = (
        "dllm-openexr-campaign-manifest/v1"
    )
    config: CampaignConfig
    schedule_hash: Sha256
    resource_map_sha256: Sha256
    manifest_sha256: Sha256 | None = None


class CellOutcome(FrozenModel):
    """Campaign-level outcome for one arm/repetition cell."""

    cell_id: str
    repetition: int
    wave_index: int
    arm: Arm
    fuzzer_seed: int
    deadline_monotonic_seconds: float
    artifact_root: Path
    mutable_root: Path
    bank_sha256: Sha256
    trajectory_set_sha256: Sha256
    config_sha256: Sha256
    resource_map_sha256: Sha256
    manifest_sha256: Sha256
    terminal_reason: Literal[
        "completed",
        "deadline",
        "treatment-failed",
        "exogenous-invalidated",
        "expired-frozen",
    ]
    coverage_frozen_through_seconds: int | None
    search_cpu_seconds: float
    treatment_failure_path: Path | None = None
    treatment_failure_sha256: Sha256 | None = None
    exogenous_evidence_path: Path | None = None
    exogenous_evidence_sha256: Sha256 | None = None


class PairOutcome(FrozenModel):
    """Status for the two paired repetitions in one wave."""

    wave_index: int
    repetitions: tuple[int, int]
    status: Literal["complete", "exogenous-invalidated"]
    exogenous_evidence_path: Path | None = None
    exogenous_evidence_sha256: Sha256 | None = None


class CampaignAccounting(FrozenModel):
    """Separate bank, search, coverage, and proof accounting."""

    bank_physical_initial_requests: int
    bank_logical_initial_requests_by_arm: dict[ModelArm, int]
    search_cpu_seconds: float
    coverage_replay_count: int
    proof_replay_count: int
    resident_model_load: ResidentModelLoadEvidence | None = None
    resident_model_load_evidence: ArtifactBackingEvidence | None = None


class CampaignOutcome(FrozenModel):
    """Complete campaign-level result."""

    manifest_sha256: Sha256
    resident_worker_pids: tuple[int, ...]
    launched_cell_count: int
    cells: tuple[CellOutcome, ...]
    pairs: tuple[PairOutcome, ...]
    accounting: CampaignAccounting


class CampaignStatus(FrozenModel):
    """Read-only status summary usable without model or Docker calls."""

    state: Literal["missing", "resumable", "complete", "corrupt", "qualified-draft-only"]
    event_count: int
    model_or_docker_calls: Literal[0] = 0


class ResidentWorkerSurface(RevisionSampler, Protocol):
    """Worker fields the coordinator observes from the resident pool."""

    @property
    def process_id_if_started(self) -> int | None:
        """Return the already-started resident worker process ID."""
        ...


class ResidentPoolSurface(Protocol):
    """Resident worker pool surface consumed by the campaign coordinator."""

    def __enter__(self) -> Self:
        """Enter the resident pool exactly once."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the resident pool."""
        ...

    def worker_for(self, cell: CellSchedule) -> ResidentWorkerSurface:
        """Return the already-resident worker for a model cell."""
        ...


class CellRunner(Protocol):
    """Task 4 search-cell runner boundary."""

    def run(self, request: CellRunRequest) -> CellResult:
        """Run one OpenEXR search cell."""
        ...


class PostSearchRunner(Protocol):
    """Post-search coverage and proof boundary."""

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        """Run coverage replay and deferred proof for one completed cell."""
        ...

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        """Run frozen coverage replay for one treatment-failed cell."""
        ...


class GateRunner(Protocol):
    """Main NO-START gate evaluator."""

    def evaluate(self, manifest: CampaignManifest) -> StartGateEvidence:
        """Return explicit gate evidence for the immutable campaign manifest."""
        ...


@dataclass(frozen=True, slots=True)
class ModelArmExecutionRequest:
    """Entered-worker inputs required to prepare one model-arm corpus."""

    config: CampaignConfig
    cell: CellOutcome
    arm: ModelArm
    bank_sampler: RevisionSampler | None = None
    frozen_adapter: RevisionSampler | None = None
    revision_delegate: RevisionSampler | None = None
    bank_sampler_id: int = 0
    revision_delegate_id: int = 0
    bank_ready_elapsed_seconds: float = 0.0
    bank_physical_duration_seconds: float = 0.0
    staging_root: Path | None = None
    live_corpus_root: Path | None = None
    producer_gpu_index: int | None = None


@dataclass(frozen=True, slots=True)
class ModelArmExecutionResult:
    """Materialized snapshots written before long search."""

    written_snapshot_count: int
    written_paths: tuple[Path, ...]
    nfe: int
    forward_calls: int
    gpu_ms: int
    bank_ready_elapsed_seconds: float
    model_ready_elapsed_seconds: float = 0.0
    corpus_release_elapsed_seconds: float | None = None
    pre_release_corpus_set_sha256: Sha256 | None = None
    producer_gpu_index: int | None = None


class ModelExecutor(Protocol):
    """Task2 model-preparation boundary."""

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        """Prepare model-arm corpus from the prebuilt shared bank and frozen adapter."""
        ...


@dataclass(frozen=True, slots=True)
class CellExecutionRecord:
    """Actual Task4 cell result plus model-preparation evidence."""

    outcome: CellOutcome
    result: CellResult | None
    model_prep: ModelArmExecutionResult | None


@dataclass(frozen=True, slots=True)
class _BankReady:
    repetition: int
    bank: CandidateBank
    sampler: RevisionSampler
    physical_duration_seconds: float
    bank_ready_elapsed_seconds: float
    producer_gpu_index: int


@dataclass(frozen=True, slots=True)
class _PrepReady:
    request: CellRunRequest
    cell: CellSchedule
    result: ModelArmExecutionResult
    bank: _BankReady


@dataclass(frozen=True, slots=True)
class _PrepFailed:
    request: CellRunRequest
    cell: CellSchedule
    error: CoordinatorError | OSError | ValueError


@dataclass(frozen=True, slots=True)
class _PrepExogenous:
    request: CellRunRequest
    cell: CellSchedule
    error: ExogenousCampaignError


@dataclass(frozen=True, slots=True)
class _SearchReady:
    request: CellRunRequest
    cell: CellSchedule
    result: CellResult


@dataclass(frozen=True, slots=True)
class _SearchFailed:
    request: CellRunRequest
    cell: CellSchedule
    error: OSError


@dataclass(frozen=True, slots=True)
class _SearchExogenous:
    request: CellRunRequest
    cell: CellSchedule
    error: ExogenousCampaignError


@dataclass(frozen=True, slots=True)
class _SearchInterrupted:
    request: CellRunRequest
    cell: CellSchedule
    error: KeyboardInterrupt


type _WaveFutureResult = (
    _BankReady
    | _PrepReady
    | _PrepFailed
    | _PrepExogenous
    | _SearchReady
    | _SearchFailed
    | _SearchExogenous
    | _SearchInterrupted
)


class BankGenerationCostEvidence(FrozenModel):
    """JSON-stable view of candidate-bank generation cost."""

    initial_requests: int = Field(ge=0)
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    gpu_ms: int = Field(ge=0)


class DuplicateParentEvidence(FrozenModel):
    """JSON-stable duplicate-parent evidence."""

    materialized_sha256: Sha256
    duplicate_raw_parent_sha256: Sha256
    kept_raw_parent_sha256: Sha256


class BankParentEvidence(FrozenModel):
    """Persisted raw parent with materialized bytes and trajectory hashes."""

    request: InitialParticleRequest
    result: RevisionResult
    content_hex: str = Field(pattern=r"^[0-9a-f]*$")
    materialized_sha256: Sha256
    raw_parent_sha256: Sha256


class CandidateBankEvidence(FrozenModel):
    """Canonical write-once candidate-bank artifact for one repetition."""

    schema_version: Literal["dllm-openexr-candidate-bank/v1", "dllm-openexr-candidate-bank/v2"] = (
        "dllm-openexr-candidate-bank/v2"
    )
    repetition: int = Field(ge=0, le=9)
    requested_seed_base: int = Field(ge=0)
    raw_parents: tuple[BankParentEvidence, ...] = Field(min_length=32, max_length=48)
    selected_raw_parent_sha256: tuple[Sha256, ...] = Field(min_length=16, max_length=16)
    duplicates: tuple[DuplicateParentEvidence, ...]
    primary_raw_count: int = Field(ge=32, le=32)
    supplement_raw_count: int = Field(ge=0, le=16)
    unique_parent_count: int = Field(ge=16)
    raw_token_states_sha256: Sha256
    raw_trajectories_sha256: Sha256
    raw_materialized_sha256: Sha256
    duplicate_evidence_sha256: Sha256
    selected_bank_sha256: Sha256
    physical_cost: BankGenerationCostEvidence
    measured_elapsed_seconds: float = Field(ge=0.0)
    physical_duration_seconds: float = Field(default=0.0, ge=0.0)
    bank_ready_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    producer_gpu_index: int = Field(default=0, ge=0, le=7)
    evidence_sha256: Sha256 | None = None


class SnapshotBackingEvidence(FrozenModel):
    """Model-prep snapshot path and content binding."""

    path: Path
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class ModelPrepEvidence(FrozenModel):
    """Canonical write-once model-preparation result artifact."""

    schema_version: Literal[
        "dllm-openexr-model-prep-result/v1", "dllm-openexr-model-prep-result/v2"
    ] = "dllm-openexr-model-prep-result/v2"
    cell_id: str
    snapshot_count: int = Field(ge=0)
    snapshots: tuple[SnapshotBackingEvidence, ...]
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    gpu_ms: int = Field(ge=0)
    bank_ready_elapsed_seconds: float = Field(ge=0.0)
    bank_physical_duration_seconds: float = Field(default=0.0, ge=0.0)
    model_ready_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    pre_release_corpus_set_sha256: Sha256 | None = None
    producer_gpu_index: int = Field(default=0, ge=0, le=7)
    evidence_sha256: Sha256 | None = None


class CorpusReleaseEvidence(FrozenModel):
    """Canonical binding for one atomic C0-to-treatment corpus release."""

    schema_version: Literal["dllm-openexr-corpus-release/v1"] = "dllm-openexr-corpus-release/v1"
    cell_id: str
    bank_sha256: Sha256
    bank_physical_duration_seconds: float = Field(ge=0.0)
    bank_ready_elapsed_seconds: float = Field(ge=0.0)
    model_ready_elapsed_seconds: float = Field(ge=0.0)
    corpus_release_elapsed_seconds: float = Field(ge=0.0)
    pre_release_corpus_set_sha256: Sha256
    producer_gpu_index: int = Field(ge=0, le=7)
    promoted_sha256: tuple[Sha256, ...]
    evidence_sha256: Sha256 | None = None


class LaunchRequestEvidence(FrozenModel):
    """Full request evidence bound into each launch checkpoint."""

    schema_version: Literal["dllm-openexr-launch-request/v1"] = "dllm-openexr-launch-request/v1"
    request: CellRunRequest
    request_sha256: Sha256
    mutable_root: Path
    bank_sha256: Sha256
    build_sha256: Sha256
    corpus_seed_manifest_sha256: Sha256


class CampaignOutcomeEnvelope(FrozenModel):
    """Completion envelope binding outcome, manifest, event stream, and raw report."""

    schema_version: Literal["dllm-openexr-campaign-outcome/v1"] = "dllm-openexr-campaign-outcome/v1"
    outcome: CampaignOutcome
    manifest_artifact_sha256: Sha256
    event_stream_sha256: Sha256
    raw_report_sha256: Sha256
    raw_report_size_bytes: int = Field(ge=0)
    evidence_tree_sha256: Sha256


class TreatmentFailureEvidence(FrozenModel):
    """Canonical evidence for one treatment-local model/search failure."""

    schema_version: Literal["dllm-openexr-treatment-failure/v1"] = (
        "dllm-openexr-treatment-failure/v1"
    )
    cell_id: str
    failure_phase: Literal["model-prep", "search"]
    exception_class: str = Field(min_length=1)
    message_sha256: Sha256
    request_sha256: Sha256
    bank_sha256: Sha256
    config_sha256: Sha256
    manifest_sha256: Sha256


class ExogenousFailureEvidence(FrozenModel):
    """Canonical evidence for one preregistered exogenous wave invalidation."""

    schema_version: Literal["dllm-openexr-exogenous-failure/v1"] = (
        "dllm-openexr-exogenous-failure/v1"
    )
    cause: str = Field(min_length=1)
    source_cell_id: str
    raw_evidence: str = Field(min_length=1)
    raw_evidence_sha256: Sha256
    affected_wave_index: int
    affected_repetitions: tuple[int, int]
    affected_cell_ids: tuple[str, ...]
    prior_event_stream_sha256: Sha256
    manifest_sha256: Sha256


class RawArmReportRow(FrozenModel):
    """Strict raw row binding one cell to request, result, bank, and post-search evidence."""

    schema_version: Literal["dllm-openexr-raw-arm-report/v1", "dllm-openexr-raw-arm-report/v2"] = (
        "dllm-openexr-raw-arm-report/v2"
    )
    campaign_root: Path
    repetition: int
    wave_index: int
    arm: Arm
    cell_id: str
    cell_request_sha256: Sha256
    cell_result_sha256: Sha256 | None
    artifact_root: Path
    coverage_status: RawArtifactStatus
    coverage_series_path: Path | None
    coverage_series_sha256: Sha256 | None
    coverage_envelope_sha256: Sha256 | None
    proof_status: RawArtifactStatus
    proof_result_path: Path | None
    proof_result_sha256: Sha256 | None
    proof_envelope_sha256: Sha256 | None
    deep_parse_status: RawArtifactStatus = "unavailable"
    deep_parse_envelope_path: Path | None = None
    deep_parse_result_sha256: Sha256 | None = None
    deep_parse_envelope_sha256: Sha256 | None = None
    deep_parse_replay_count: int | None = Field(default=None, ge=0)
    deep_parse_success_count: int | None = Field(default=None, ge=0)
    deep_parse_total_count: int | None = Field(default=None, ge=0)
    coverage_replay_count: int | None = Field(default=None, ge=0)
    c0_baseline_sha256: Sha256
    bank_evidence_path: Path | None
    bank_evidence_sha256: Sha256 | None
    bank_measured_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    bank_physical_duration_seconds: float | None = Field(default=None, ge=0.0)
    bank_ready_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    treatment_failure_path: Path | None = None
    treatment_failure_sha256: Sha256 | None = None
    exogenous_evidence_path: Path | None = None
    exogenous_evidence_sha256: Sha256 | None = None
    model_prep_result_path: Path | None = None
    model_prep_result_sha256: Sha256 | None = None
    model_snapshot_count: int | None = Field(default=None, ge=0)
    model_snapshot_paths: tuple[Path, ...]
    model_snapshot_sha256: tuple[Sha256, ...]
    model_snapshot_size_bytes: tuple[int, ...]
    model_nfe: int | None = Field(default=None, ge=0)
    model_forward_calls: int | None = Field(default=None, ge=0)
    model_gpu_ms: int | None = Field(default=None, ge=0)
    model_ready_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    corpus_release_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    pre_release_corpus_set_sha256: Sha256 | None = None
    producer_gpu_index: int | None = Field(default=None, ge=0, le=7)
    corpus_release_path: Path | None = None
    corpus_release_sha256: Sha256 | None = None
    proof_replay_count: int | None = Field(default=None, ge=0)
    proof_wall_seconds: float | None = Field(default=None, ge=0.0)
    proof_cpu_seconds: float | None = Field(default=None, ge=0.0)
    terminal_reason: Literal[
        "completed",
        "deadline",
        "treatment-failed",
        "exogenous-invalidated",
        "expired-frozen",
    ]
    pair_status: Literal["complete", "exogenous-invalidated"]
    search_cpu_seconds: float = Field(ge=0.0)
    fuzzer_seed: int
    manifest_sha256: Sha256
    resource_map_sha256: Sha256
    previous_row_sha256: Sha256
    row_sha256: Sha256


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load the strict one-day JSON config."""
    try:
        return CampaignConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise CoordinatorError(reason=f"invalid one-day config: {error}") from error


def build_campaign_manifest(config: CampaignConfig, schedule: CampaignSchedule) -> CampaignManifest:
    """Build and hash immutable campaign provenance before launches."""
    try:
        resource_hash = _hash_json(config.resource_map.model_dump(mode="json"))
    except ValueError as error:
        raise CoordinatorError(reason=str(error)) from error
    if config.target_commit != schedule.preregistration.target_commit:
        raise CoordinatorError(reason="target commit mismatch")
    unsigned = CampaignManifest(
        config=config,
        schedule_hash=_hash_json(_schedule_json(schedule)),
        resource_map_sha256=resource_hash,
    )
    return unsigned.model_copy(
        update={"manifest_sha256": _hash_json(unsigned.model_dump(mode="json"))}
    )


def build_cell_requests(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    schedule: CampaignSchedule,
    bank_by_repetition: dict[int, str],
    resource_map: ResourceMap,
) -> tuple[CellRunRequest, ...]:
    """Build real Task 4 cell requests with private per-cell roots."""
    return tuple(
        _cell_request(config, manifest, cell, bank_by_repetition, resource_map)
        for cell in schedule.cells
    )


def coordinate_campaign(
    config: CampaignConfig,
    *,
    resident_pool: ResidentPoolSurface,
    cell_runner: CellRunner,
    post_search: PostSearchRunner,
    gate_runner: GateRunner,
    model_executor: ModelExecutor | None = None,
    bank_sampler: RevisionSampler | None = None,
    revision_delegates: Mapping[ModelArm, RevisionSampler] | None = None,
) -> CampaignOutcome:
    """Run or continue the deterministic fake/production-composable campaign."""
    del bank_sampler, revision_delegates
    if config.qualification_state != "qualified":
        raise CoordinatorError(reason="draft config cannot start the main campaign")
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds,
        gpu_count=len(config.resource_map.model_cores),
    )
    manifest = build_campaign_manifest(config, schedule)
    _assert_new_or_resumable_root(config.artifact_root, manifest)
    if (config.artifact_root / "campaign-outcome.json").exists():
        return load_complete_campaign_outcome(config.artifact_root, manifest)
    gates = gate_runner.evaluate(manifest)
    if failed := gates.failing_gate():
        raise CoordinatorError(reason=f"NO-START gate failed: {failed}")
    _persist_manifest(config.artifact_root, manifest)
    existing_events = _existing_events(config.artifact_root)
    completed = {event.cell_id: event for event in existing_events if event.phase == "terminal"}
    if len(completed) == _COMPLETE_CELL_COUNT:
        raise CoordinatorError(reason="complete campaign outcome envelope missing")
    if not existing_events:
        _append_campaign_event(config.artifact_root, "manifest", _manifest_payload(manifest))
    records_by_id: dict[str, CellExecutionRecord] = {}
    requests_by_id: dict[str, CellRunRequest] = {}
    banks: dict[int, CandidateBank] = {}
    worker_pids: set[int] = set()
    resident_model_load: ResidentModelLoadEvidence | None = None
    resident_model_load_evidence: ArtifactBackingEvidence | None = None
    resident_worker_pids: tuple[int, ...] = ()
    with resident_pool as pool:
        (
            resident_model_load,
            resident_model_load_evidence,
            resident_worker_pids,
        ) = _resident_model_load_accounting(config, pool)
        worker_pids.update(resident_worker_pids)
        for wave in schedule.waves:
            _wait_for_wave_start(wave, cell_runner)
            wave_records, wave_requests, wave_banks = _run_runtime_wave(
                config=config,
                manifest=manifest,
                schedule=schedule,
                wave=wave,
                pool=pool,
                completed=completed,
                cell_runner=cell_runner,
                model_executor=model_executor,
                worker_pids=worker_pids,
            )
            records_by_id.update({record.outcome.cell_id: record for record in wave_records})
            requests_by_id.update({request.cell_id: request for request in wave_requests})
            banks.update(wave_banks)
            _apply_wave_exogenous_overlay(records_by_id, wave)
    ordered_records = tuple(records_by_id[_cell_id(cell)] for cell in schedule.cells)
    request_tuple = tuple(requests_by_id[_cell_id(cell)] for cell in schedule.cells)
    cells = [record.outcome for record in ordered_records]
    post_search_by_cell = _run_deferred_post_search(
        list(ordered_records), request_tuple, post_search
    )
    outcome = _campaign_outcome(
        manifest,
        resident_worker_pids or tuple(sorted(worker_pids)),
        tuple(cells),
        schedule,
        banks,
        post_search_by_cell,
        resident_model_load=resident_model_load,
        resident_model_load_evidence=resident_model_load_evidence,
    )
    _persist_raw_arm_reports(
        config.artifact_root,
        outcome,
        post_search_by_cell,
        config.c0_sha256,
        ordered_records,
        request_tuple,
    )
    _persist_campaign_outcome(config.artifact_root, outcome)
    return outcome


def _run_runtime_wave(  # noqa: PLR0912, PLR0915
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    schedule: CampaignSchedule,
    wave: WaveSchedule,
    pool: ResidentPoolSurface,
    completed: dict[str, _ExistingEvent],
    cell_runner: CellRunner,
    model_executor: ModelExecutor | None,
    worker_pids: set[int],
) -> tuple[tuple[CellExecutionRecord, ...], tuple[CellRunRequest, ...], dict[int, CandidateBank]]:
    """Run one wave with dynamic bank, preparation, release, and search readiness."""
    existing = _existing_events(config.artifact_root)
    launches = {event.cell_id: event for event in existing if event.phase == "launch"}
    launched_ids = set(launches)
    terminal_ids = set(completed)
    provisional = _initialize_wave_corpora(config, manifest, wave, launched_ids)
    records: dict[str, CellExecutionRecord] = {}
    requests: dict[str, CellRunRequest] = {}
    banks: dict[int, CandidateBank] = {}
    bank_ready: dict[int, _BankReady] = {}
    pending: set[Future[_WaveFutureResult]] = set()
    exogenous: _PrepExogenous | _SearchExogenous | None = None
    interrupted: KeyboardInterrupt | None = None
    executor = ThreadPoolExecutor(max_workers=12)

    def wave_now() -> float:
        return _runtime_monotonic(cell_runner, wave)

    def append_terminal(record: CellExecutionRecord) -> None:
        records[record.outcome.cell_id] = record
        if record.outcome.cell_id in terminal_ids:
            return
        _append_campaign_event(
            config.artifact_root,
            "terminal",
            _outcome_payload(record.outcome),
        )
        terminal_ids.add(record.outcome.cell_id)

    def append_launch(cell: CellSchedule, request: CellRunRequest) -> CellRunRequest:
        prior = launches.get(request.cell_id)
        if prior is not None:
            return _request_from_launch(prior, request)
        _append_campaign_event(
            config.artifact_root,
            "launch",
            _cell_payload(config, manifest, cell, request, banks),
        )
        launched_ids.add(request.cell_id)
        return request

    def submit_search(
        cell: CellSchedule,
        request: CellRunRequest,
        model_prep: ModelArmExecutionResult | None,
    ) -> None:
        prior = records.get(request.cell_id)
        records[request.cell_id] = CellExecutionRecord(
            outcome=_pending_cell(config, manifest, cell, request),
            result=None,
            model_prep=model_prep if prior is None else prior.model_prep,
        )
        pending.add(executor.submit(_search_job, cell_runner, request, cell))

    def prepared(ready: _PrepReady) -> None:
        request = ready.request
        cell = ready.cell
        elapsed = _wave_elapsed_seconds(wave, wave_now)
        rows = _append_due_pre_release_checkpoints(request, elapsed)
        pre_release_hash = rows[-1].cumulative_set_sha256
        prep_path = _model_prep_evidence_path(request)
        prep_persisted = prep_path.exists()
        model_ready = ready.result.model_ready_elapsed_seconds if prep_persisted else elapsed
        enriched = replace(
            ready.result,
            bank_ready_elapsed_seconds=ready.bank.bank_ready_elapsed_seconds,
            model_ready_elapsed_seconds=model_ready,
            pre_release_corpus_set_sha256=pre_release_hash,
            producer_gpu_index=ready.bank.producer_gpu_index
            if cell.gpu_index is None
            else int(cell.gpu_index),
        )
        if prep_persisted:
            enriched = _load_model_prep_result(prep_path, request.cell_id)
        else:
            _persist_model_prep_result(
                prep_path,
                request.cell_id,
                enriched,
                bank_physical_duration_seconds=ready.bank.physical_duration_seconds,
            )
        if elapsed >= _WAVE_DURATION_SECONDS:
            expired = CellExecutionRecord(
                outcome=_expired_cell(config, manifest, cell, request),
                result=None,
                model_prep=enriched,
            )
            append_terminal(expired)
            return
        release_path = _corpus_release_path(request)
        if release_path.exists():
            release = _load_corpus_release(release_path, request.cell_id)
        else:
            promoted = _promote_treatment_corpus(request, ready.bank.bank, enriched)
            _ = seed_cell_corpus(
                config,
                _pending_cell(config, manifest, cell, request),
                ready.bank.bank,
            )
            release = CorpusReleaseEvidence(
                cell_id=request.cell_id,
                bank_sha256=ready.bank.bank.selected_bank_sha256,
                bank_physical_duration_seconds=ready.bank.physical_duration_seconds,
                bank_ready_elapsed_seconds=ready.bank.bank_ready_elapsed_seconds,
                model_ready_elapsed_seconds=enriched.model_ready_elapsed_seconds,
                corpus_release_elapsed_seconds=elapsed,
                pre_release_corpus_set_sha256=pre_release_hash,
                producer_gpu_index=int(cell.gpu_index or 0),
                promoted_sha256=promoted,
            )
            _persist_corpus_release(release_path, release)
        enriched = replace(
            enriched,
            corpus_release_elapsed_seconds=release.corpus_release_elapsed_seconds,
        )
        request = append_launch(cell, request)
        submit_search(cell, request, enriched)

    def schedule_models(ready: _BankReady) -> None:
        banks[ready.repetition] = ready.bank
        bank_ready[ready.repetition] = ready
        if exogenous is not None:
            return
        adapters = _adapters_by_repetition(
            config,
            schedule,
            pool,
            {ready.repetition: ready.bank},
            worker_pids,
            repetitions=(ready.repetition,),
        )[ready.repetition]
        cells = tuple(
            cell
            for cell in wave.cells
            if int(cell.repetition) == ready.repetition and cell.arm in MODEL_ARMS
        )
        for cell in cells:
            request = _cell_request(
                config,
                manifest,
                cell,
                {ready.repetition: ready.bank.selected_bank_sha256},
                config.resource_map,
            )
            launch = launches.get(request.cell_id)
            if launch is not None:
                request = _request_from_launch(launch, request)
            requests[request.cell_id] = request
            terminal = completed.get(request.cell_id)
            if terminal is not None:
                request = _request_for_terminal_event(
                    config=config,
                    manifest=manifest,
                    cell=cell,
                    generated=request,
                    launch=launch,
                    terminal=terminal,
                )
                requests[request.cell_id] = request
                records[request.cell_id] = _terminal_record_from_evidence(terminal, request)
                continue
            prep_path = _model_prep_evidence_path(request)
            if prep_path.exists():
                prepared(
                    _PrepReady(
                        request=request,
                        cell=cell,
                        result=_load_model_prep_result(prep_path, request.cell_id),
                        bank=ready,
                    )
                )
                continue
            if model_executor is None:
                prepared(
                    _PrepReady(
                        request=request,
                        cell=cell,
                        result=ModelArmExecutionResult(
                            written_snapshot_count=0,
                            written_paths=(),
                            nfe=0,
                            forward_calls=0,
                            gpu_ms=0,
                            bank_ready_elapsed_seconds=ready.bank_ready_elapsed_seconds,
                        ),
                        bank=ready,
                    )
                )
                continue
            if cell.gpu_index is None:
                raise CoordinatorError(reason="model cell has no producer GPU")
            staging_artifact_root = request.artifact_root / "model-prep-staging"
            prep_cell = _pending_cell(config, manifest, cell, request).model_copy(
                update={"artifact_root": staging_artifact_root}
            )
            execution_request = ModelArmExecutionRequest(
                config=config,
                cell=prep_cell,
                bank_sampler=ready.sampler,
                frozen_adapter=adapters[cast("ModelArm", cell.arm)],
                arm=cast("ModelArm", cell.arm),
                bank_sampler_id=id(ready.sampler),
                revision_delegate_id=id(
                    getattr(
                        adapters[cast("ModelArm", cell.arm)],
                        "revise_delegate",
                        adapters[cast("ModelArm", cell.arm)],
                    )
                ),
                bank_ready_elapsed_seconds=ready.bank_ready_elapsed_seconds,
                bank_physical_duration_seconds=ready.physical_duration_seconds,
                staging_root=staging_artifact_root / "corpus",
                live_corpus_root=request.corpus_root,
                producer_gpu_index=int(cell.gpu_index),
            )
            pending.add(
                executor.submit(
                    _prep_job,
                    model_executor,
                    execution_request,
                    request,
                    cell,
                    ready,
                )
            )

    try:
        for cell in wave.cells:
            if cell.arm != "libfuzzer-only":
                continue
            request = provisional[_cell_id(cell)]
            terminal = completed.get(request.cell_id)
            if terminal is not None:
                request = _request_for_terminal_event(
                    config=config,
                    manifest=manifest,
                    cell=cell,
                    generated=request,
                    launch=launches.get(request.cell_id),
                    terminal=terminal,
                )
                requests[request.cell_id] = request
                records[request.cell_id] = _terminal_record_from_evidence(terminal, request)
                continue
            if _wave_elapsed_seconds(wave, wave_now) >= _WAVE_DURATION_SECONDS:
                try:
                    _ = freeze_corpus_index(
                        _corpus_index_request(request), freeze_elapsed_seconds=0.0
                    )
                except CoverageSeriesError as error:
                    raise CoordinatorError(
                        reason="expired baseline corpus index invalid"
                    ) from error
                requests[request.cell_id] = request
                append_terminal(
                    CellExecutionRecord(
                        outcome=_expired_cell(config, manifest, cell, request),
                        result=None,
                        model_prep=None,
                    )
                )
                continue
            request = append_launch(cell, request)
            requests[request.cell_id] = request
            submit_search(cell, request, None)

        for repetition in (int(wave.repetitions[0]), int(wave.repetitions[1])):
            sampler = _bank_sampler_for_repetition(schedule, pool, repetition)
            producer_gpu = _bank_producer_gpu(wave, repetition)
            path = _bank_evidence_path(config, repetition)
            if path.exists():
                evidence = _load_candidate_bank_evidence(path, repetition)
                schedule_models(
                    _BankReady(
                        repetition=repetition,
                        bank=_load_candidate_bank(path, repetition),
                        sampler=sampler,
                        physical_duration_seconds=evidence.physical_duration_seconds,
                        bank_ready_elapsed_seconds=evidence.bank_ready_elapsed_seconds,
                        producer_gpu_index=evidence.producer_gpu_index,
                    )
                )
            else:
                namespace = schedule.seed_namespaces[repetition]
                pending.add(
                    executor.submit(
                        _bank_job,
                        config,
                        wave,
                        repetition,
                        producer_gpu,
                        sampler,
                        namespace,
                        wave_now,
                    )
                )

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                result = future.result()
                match result:
                    case _BankReady():
                        path = _bank_evidence_path(config, result.repetition)
                        if not path.exists():
                            _persist_candidate_bank(
                                path,
                                result.repetition,
                                result.bank,
                                result.physical_duration_seconds,
                                bank_ready_elapsed_seconds=result.bank_ready_elapsed_seconds,
                                producer_gpu_index=result.producer_gpu_index,
                            )
                        schedule_models(result)
                    case _PrepReady():
                        if exogenous is None:
                            prepared(result)
                    case _PrepFailed():
                        if exogenous is None:
                            request = append_launch(result.cell, result.request)
                            requests[request.cell_id] = request
                            outcome = _model_prep_failed_cell(
                                config,
                                manifest,
                                result.cell,
                                request,
                                result.error,
                            )
                            append_terminal(CellExecutionRecord(outcome, None, None))
                    case _PrepExogenous():
                        if exogenous is None:
                            exogenous = result
                            _cancel_active_searches(cell_runner)
                    case _SearchReady():
                        if exogenous is None:
                            prior = records.get(result.request.cell_id)
                            outcome = _cell_outcome(
                                config,
                                manifest,
                                result.cell,
                                result.request,
                                result.result,
                            )
                            append_terminal(
                                CellExecutionRecord(
                                    outcome,
                                    result.result,
                                    None if prior is None else prior.model_prep,
                                )
                            )
                    case _SearchFailed():
                        if exogenous is None:
                            prior = records.get(result.request.cell_id)
                            outcome = _search_failed_cell(
                                config,
                                manifest,
                                result.cell,
                                result.request,
                                result.error,
                            )
                            append_terminal(
                                CellExecutionRecord(
                                    outcome,
                                    None,
                                    None if prior is None else prior.model_prep,
                                )
                            )
                    case _SearchExogenous():
                        if exogenous is None:
                            exogenous = result
                            _cancel_active_searches(cell_runner)
                    case _SearchInterrupted():
                        interrupted = result.error
                        _cancel_active_searches(cell_runner)
                    case unreachable:
                        assert_never(unreachable)
            if interrupted is not None:
                break
        if interrupted is not None:
            for future in pending:
                _ = future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise interrupted
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        _cancel_active_searches(cell_runner)
        for future in pending:
            _ = future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise

    if exogenous is not None:
        records, requests = _invalidate_runtime_wave(
            config=config,
            manifest=manifest,
            wave=wave,
            banks=banks,
            records=records,
            requests=requests,
            provisional=provisional,
            launched_ids=launched_ids,
            terminal_ids=terminal_ids,
            source_request=exogenous.request,
            source_cell=exogenous.cell,
            error=exogenous.error,
        )
    ordered_records = tuple(records[_cell_id(cell)] for cell in wave.cells)
    ordered_requests = tuple(requests[_cell_id(cell)] for cell in wave.cells)
    return ordered_records, ordered_requests, banks


def _wait_for_wave_start(wave: WaveSchedule, cell_runner: CellRunner) -> None:
    """Wait for the fixed wave anchor without changing its absolute deadline."""
    remaining = float(wave.start_monotonic_seconds) - _runtime_monotonic(cell_runner, wave)
    if remaining > 0.0:
        sleep(remaining)


def _runtime_monotonic(cell_runner: CellRunner, wave: WaveSchedule) -> float:
    provider = getattr(cell_runner, "wave_monotonic", None)
    if callable(provider):
        clock = cast("Callable[[float], float]", provider)
        return clock(float(wave.start_monotonic_seconds))
    delegate = getattr(cell_runner, "delegate", None)
    provider = getattr(delegate, "wave_monotonic", None)
    if callable(provider):
        clock = cast("Callable[[float], float]", provider)
        return clock(float(wave.start_monotonic_seconds))
    return monotonic()


def _initialize_wave_corpora(
    config: CampaignConfig,
    manifest: CampaignManifest,
    wave: WaveSchedule,
    launched_ids: set[str],
) -> dict[str, CellRunRequest]:
    bank_hashes = {int(repetition): config.c0_sha256 for repetition in wave.repetitions}
    requests: dict[str, CellRunRequest] = {}
    for cell in wave.cells:
        request = _cell_request(config, manifest, cell, bank_hashes, config.resource_map)
        requests[request.cell_id] = request
        if request.cell_id in launched_ids or _corpus_release_path(request).exists():
            continue
        pending = _pending_cell(config, manifest, cell, request)
        try:
            summary = seed_cell_corpus(config, pending, None)
        except C0SourceError as error:
            raise CoordinatorError(reason=error.reason) from error
        if summary.bank_parent_count != 0:
            raise CoordinatorError(reason="pre-release corpus seed is not C0-only")
        index_request = _corpus_index_request(request)
        rows = load_corpus_index(index_request)
        if not rows:
            rows = (snapshot_corpus(index_request, 0, 0.0),)
        if (
            len(rows) != 1
            or rows[0].scheduled_elapsed_seconds != 0
            or rows[0].cumulative_hashes != (config.c0_sha256,)
        ):
            raise CoordinatorError(reason="pre-release corpus index is not exact C0")
        live_hashes = tuple(
            sorted(sha256_bytes(path.read_bytes()) for path in request.corpus_root.iterdir())
        )
        if live_hashes != (config.c0_sha256,):
            raise CoordinatorError(reason="pre-release live corpus is not exact C0")
    return requests


def _bank_job(  # noqa: PLR0917
    config: CampaignConfig,
    wave: WaveSchedule,
    repetition: int,
    producer_gpu_index: int,
    sampler: RevisionSampler,
    namespace: SeedNamespace,
    wave_now: Callable[[], float],
) -> _BankReady:
    started = monotonic()
    bank = build_candidate_bank(sampler, namespace, steps=config.candidate_steps)
    finished = monotonic()
    return _BankReady(
        repetition=repetition,
        bank=bank,
        sampler=sampler,
        physical_duration_seconds=max(0.0, finished - started),
        bank_ready_elapsed_seconds=max(0.0, wave_now() - float(wave.start_monotonic_seconds)),
        producer_gpu_index=producer_gpu_index,
    )


def _prep_job(
    model_executor: ModelExecutor,
    execution_request: ModelArmExecutionRequest,
    request: CellRunRequest,
    cell: CellSchedule,
    bank: _BankReady,
) -> _PrepReady | _PrepFailed | _PrepExogenous:
    try:
        result = model_executor.prepare_corpus(execution_request)
    except ExogenousCampaignError as error:
        return _PrepExogenous(request, cell, error)
    except (CoordinatorError, OSError, ValueError) as error:
        return _PrepFailed(request, cell, error)
    staging = execution_request.staging_root
    if staging is None or any(not path.is_relative_to(staging) for path in result.written_paths):
        return _PrepFailed(
            request,
            cell,
            CoordinatorError(reason="model preparation escaped private staging"),
        )
    return _PrepReady(request, cell, result, bank)


def _search_job(
    cell_runner: CellRunner,
    request: CellRunRequest,
    cell: CellSchedule,
) -> _SearchReady | _SearchFailed | _SearchExogenous | _SearchInterrupted:
    try:
        return _SearchReady(request, cell, cell_runner.run(request))
    except ExogenousCampaignError as error:
        return _SearchExogenous(request, cell, error)
    except KeyboardInterrupt as error:
        return _SearchInterrupted(request, cell, error)
    except OSError as error:
        return _SearchFailed(request, cell, error)


def _bank_producer_gpu(wave: WaveSchedule, repetition: int) -> int:
    for cell in wave.cells:
        if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS:
            if cell.gpu_index is None:
                raise CoordinatorError(reason="bank producer has no GPU")
            return int(cell.gpu_index)
    raise CoordinatorError(reason="bank producer cell missing")


def _wave_elapsed_seconds(wave: WaveSchedule, wave_now: Callable[[], float]) -> float:
    return min(
        _WAVE_DURATION_SECONDS,
        max(0.0, wave_now() - float(wave.start_monotonic_seconds)),
    )


def _corpus_index_request(request: CellRunRequest) -> CorpusIndexRequest:
    return CorpusIndexRequest(
        artifact_root=request.artifact_root,
        corpus_root=request.corpus_root,
        store_root=request.artifact_root / "corpus-inputs",
        index_path=request.artifact_root / "corpus-index.jsonl",
    )


def _append_due_pre_release_checkpoints(
    request: CellRunRequest, elapsed_seconds: float
) -> tuple[CorpusIndexRecord, ...]:
    index_request = _corpus_index_request(request)
    rows = list(load_corpus_index(index_request))
    while len(rows) < len(SCHEDULE_SECONDS) and SCHEDULE_SECONDS[len(rows)] <= elapsed_seconds:
        scheduled = SCHEDULE_SECONDS[len(rows)]
        rows.append(snapshot_corpus(index_request, scheduled, elapsed_seconds))
    return tuple(rows)


def _expired_cell(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
) -> CellOutcome:
    return _pending_cell(config, manifest, cell, request).model_copy(
        update={
            "terminal_reason": "expired-frozen",
            "coverage_frozen_through_seconds": 10_800,
        }
    )


def _corpus_release_path(request: CellRunRequest) -> Path:
    return request.artifact_root / "corpus-release.json"


def _promote_treatment_corpus(
    request: CellRunRequest,
    bank: CandidateBank,
    model_prep: ModelArmExecutionResult,
) -> tuple[Sha256, ...]:
    stage = request.artifact_root / "corpus-release-staging"
    archive = request.artifact_root / "pre-release-corpus"
    if stage.exists() or archive.exists():
        raise CoordinatorError(reason="corpus release staging already exists")
    stage.mkdir(parents=True)
    promoted: set[Sha256] = set()
    for path in request.corpus_root.iterdir():
        _reject_symlink_path(path)
        content = path.read_bytes()
        digest = sha256_bytes(content)
        _write_once_bytes(stage / path.name, content, label="release C0")
        promoted.add(digest)
    for index, parent in enumerate(bank.selected_parents):
        if sha256_bytes(parent.content) != parent.materialized_sha256:
            raise CoordinatorError(reason="candidate-bank release parent hash mismatch")
        _write_once_bytes(
            stage / f"bank-parent-{index:02d}-{parent.materialized_sha256}.exr",
            parent.content,
            label="release bank parent",
        )
        promoted.add(parent.materialized_sha256)
    for snapshot in model_prep.written_paths:
        _reject_symlink_path(snapshot)
        try:
            content = snapshot.read_bytes()
        except OSError as error:
            raise CoordinatorError(reason="model-prep snapshot unreadable") from error
        digest = sha256_bytes(content)
        _write_once_bytes(stage / f"model-{digest}.exr", content, label="release model snapshot")
        promoted.add(digest)
    try:
        _ = request.corpus_root.replace(archive)
        _ = stage.replace(request.corpus_root)
    except OSError as error:
        if archive.exists() and not request.corpus_root.exists():
            _ = archive.replace(request.corpus_root)
        raise CoordinatorError(reason="atomic corpus directory release failed") from error
    _fsync_dir(request.artifact_root)
    actual = {
        sha256_bytes(path.read_bytes()) for path in request.corpus_root.iterdir() if path.is_file()
    }
    if actual != promoted:
        raise CoordinatorError(reason="released corpus set mismatch")
    return tuple(sorted(promoted))


def _persist_corpus_release(path: Path, release: CorpusReleaseEvidence) -> None:
    payload = canonical_json_bytes(release.model_dump(mode="json", exclude={"evidence_sha256"}))
    bound = release.model_copy(update={"evidence_sha256": sha256_bytes(payload)})
    _write_once_canonical_model(path, bound)


def _load_corpus_release(path: Path, cell_id: str) -> CorpusReleaseEvidence:
    try:
        release = CorpusReleaseEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid corpus release evidence") from error
    _require_canonical_model(path, release)
    unsigned = canonical_json_bytes(release.model_dump(mode="json", exclude={"evidence_sha256"}))
    if release.cell_id != cell_id or release.evidence_sha256 != sha256_bytes(unsigned):
        raise CoordinatorError(reason="corpus release evidence mismatch")
    return release


def _invalidate_runtime_wave(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    wave: WaveSchedule,
    banks: dict[int, CandidateBank],
    records: dict[str, CellExecutionRecord],
    requests: dict[str, CellRunRequest],
    provisional: dict[str, CellRunRequest],
    launched_ids: set[str],
    terminal_ids: set[str],
    source_request: CellRunRequest,
    source_cell: CellSchedule,
    error: ExogenousCampaignError,
) -> tuple[dict[str, CellExecutionRecord], dict[str, CellRunRequest]]:
    for cell in wave.cells:
        cell_id = _cell_id(cell)
        if cell.arm in MODEL_ARMS:
            repetition = int(cell.repetition)
            bank_hash = (
                banks[repetition].selected_bank_sha256 if repetition in banks else config.c0_sha256
            )
            request = _cell_request(
                config,
                manifest,
                cell,
                {repetition: bank_hash},
                config.resource_map,
            )
        else:
            request = provisional[cell_id]
        requests[cell_id] = request
        if cell_id not in launched_ids:
            _append_campaign_event(
                config.artifact_root,
                "launch",
                _cell_payload(config, manifest, cell, request, banks),
            )
            launched_ids.add(cell_id)
    wave_items = tuple((requests[_cell_id(cell)], cell) for cell in wave.cells)
    evidence_path, evidence_sha256 = _persist_exogenous_failure(
        config,
        manifest,
        source_cell=source_cell,
        source_request=source_request,
        wave_items=wave_items,
        error=error,
    )
    for request, cell in wave_items:
        prior = records.get(request.cell_id)
        if request.cell_id in terminal_ids and prior is not None:
            records[request.cell_id] = CellExecutionRecord(
                outcome=prior.outcome.model_copy(
                    update={
                        "exogenous_evidence_path": evidence_path,
                        "exogenous_evidence_sha256": evidence_sha256,
                    }
                ),
                result=prior.result,
                model_prep=prior.model_prep,
            )
            continue
        outcome = _invalidated_cell(
            config,
            manifest,
            cell,
            request,
            evidence_path=evidence_path,
            evidence_sha256=evidence_sha256,
        )
        records[request.cell_id] = CellExecutionRecord(
            outcome,
            None,
            None if prior is None else prior.model_prep,
        )
        if request.cell_id not in terminal_ids:
            _append_campaign_event(
                config.artifact_root,
                "terminal",
                _outcome_payload(outcome),
            )
            terminal_ids.add(request.cell_id)
    return records, requests


def campaign_status(root: Path) -> CampaignStatus:  # noqa: PLR0911
    """Read deterministic status without model or Docker side effects."""
    if not root.exists():
        return CampaignStatus(state="missing", event_count=0)
    if (root / "qualification.json").is_file():
        return CampaignStatus(state="qualified-draft-only", event_count=0)
    if not (root / _MANIFEST_NAME).is_file() or not (root / _EVENTS_NAME).is_file():
        return CampaignStatus(state="corrupt", event_count=0)
    try:
        events = _existing_events(root)
    except CoordinatorError:
        return CampaignStatus(state="corrupt", event_count=0)
    terminal = sum(event.phase == "terminal" for event in events)
    if terminal == _COMPLETE_CELL_COUNT:
        try:
            manifest_payload = _JSON_OBJECT.validate_json((root / _MANIFEST_NAME).read_bytes())
            config_payload = _JSON_OBJECT.validate_python(manifest_payload["config"])
            config = CampaignConfig.model_validate(config_payload)
            schedule = build_campaign_schedule(
                main_start_monotonic_seconds=config.main_start_monotonic_seconds,
                gpu_count=len(config.resource_map.model_cores),
            )
            manifest = build_campaign_manifest(config, schedule)
            _ = load_complete_campaign_outcome(root, manifest)
        except (KeyError, OSError, ValidationError, CoordinatorError):
            return CampaignStatus(state="corrupt", event_count=len(events))
        return CampaignStatus(state="complete", event_count=len(events))
    return CampaignStatus(
        state="resumable",
        event_count=len(events),
    )


def _cell_request(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    bank_by_repetition: dict[int, str],
    resource_map: ResourceMap,
) -> CellRunRequest:
    del manifest
    cell_id = _cell_id(cell)
    root = (
        config.artifact_root
        / "cells"
        / f"wave-{int(cell.wave_index):02d}"
        / f"rep-{int(cell.repetition):02d}"
        / cell.arm
    )
    return CellRunRequest(
        cell_id=cell_id,
        artifact_root=root,
        corpus_root=root / "corpus",
        crash_root=root / "crashes",
        log_root=root / "logs",
        target_commit=TARGET_COMMIT,
        harness_sha256=config.target_harness_sha256,
        config_sha256=config.config_sha256,
        bank_sha256=_bank_hash(config, cell, bank_by_repetition),
        start_monotonic_seconds=_wave_start(config, cell),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        fuzzer_seed=int(cell.fuzzer_seed),
        max_input_length=config.max_input_length,
        target_timeout_seconds=config.target_timeout_seconds,
        rss_limit_mb=config.rss_limit_mb,
        cpu_index=resource_map.search_cores[_search_core_index(cell)].core_id,
        dictionary_path=config.dictionary_path,
        dictionary_sha256=config.dictionary_sha256,
        build=SearchBuildProvenance(
            image_digest=config.search_image_digest,
            binary_sha256=config.search_binary_sha256,
            sanitizers=("address", "undefined"),
        ),
    )


def _run_search_wave(  # pyright: ignore[reportUnusedFunction]  # noqa: PLR0912, PLR0915
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    banks: dict[int, CandidateBank],
    bank_samplers: dict[int, RevisionSampler],
    adapters: dict[int, dict[ModelArm, RevisionSampler]],
    wave_items: tuple[tuple[CellRunRequest, CellSchedule], ...],
    completed: dict[str, _ExistingEvent],
    cell_runner: CellRunner,
    model_executor: ModelExecutor | None,
) -> tuple[CellExecutionRecord, ...]:
    records_by_id: dict[str, CellExecutionRecord] = {}
    pending: list[tuple[CellRunRequest, CellSchedule]] = []
    for request, cell in wave_items:
        if event := completed.get(request.cell_id):
            result = _load_completed_cell_result(request)
            records_by_id[request.cell_id] = CellExecutionRecord(
                outcome=_recovered_cell(event), result=result, model_prep=None
            )
            continue
        bank = banks[int(cell.repetition)] if cell.arm in MODEL_ARMS else None
        pending_cell = _pending_cell(config, manifest, cell, request)
        try:
            _ = seed_cell_corpus(config, pending_cell, bank)
        except C0SourceError as error:
            raise CoordinatorError(reason=error.reason) from error
        if cell.arm in MODEL_ARMS and model_executor is not None:
            prep_path = _model_prep_evidence_path(request)
            if prep_path.exists():
                model_prep = _load_model_prep_result(prep_path, request.cell_id)
            else:
                try:
                    model_prep = model_executor.prepare_corpus(
                        ModelArmExecutionRequest(
                            config=config,
                            cell=pending_cell,
                            bank_sampler=bank_samplers[int(cell.repetition)],
                            frozen_adapter=adapters[int(cell.repetition)][cell.arm],
                            arm=cell.arm,
                            bank_sampler_id=id(bank_samplers[int(cell.repetition)]),
                            revision_delegate_id=id(
                                getattr(
                                    adapters[int(cell.repetition)][cell.arm],
                                    "revise_delegate",
                                    adapters[int(cell.repetition)][cell.arm],
                                )
                            ),
                            bank_ready_elapsed_seconds=_candidate_bank_elapsed_seconds(
                                _bank_evidence_path(config, int(cell.repetition))
                            ),
                        )
                    )
                    _persist_model_prep_result(prep_path, request.cell_id, model_prep)
                except ExogenousCampaignError as error:
                    evidence_path, evidence_sha256 = _persist_exogenous_failure(
                        config,
                        manifest,
                        source_cell=cell,
                        source_request=request,
                        wave_items=wave_items,
                        error=error,
                    )
                    for invalidated_request, invalidated_cell in wave_items:
                        outcome = _invalidated_cell(
                            config,
                            manifest,
                            invalidated_cell,
                            invalidated_request,
                            evidence_path=evidence_path,
                            evidence_sha256=evidence_sha256,
                        )
                        records_by_id[invalidated_request.cell_id] = CellExecutionRecord(
                            outcome=outcome, result=None, model_prep=None
                        )
                    pending.clear()
                    break
                except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
                    outcome = _model_prep_failed_cell(config, manifest, cell, request, error)
                    records_by_id[request.cell_id] = CellExecutionRecord(
                        outcome=outcome, result=None, model_prep=None
                    )
                    continue
            records_by_id[request.cell_id] = CellExecutionRecord(
                outcome=pending_cell, result=None, model_prep=model_prep
            )
        pending.append((request, cell))
    if pending:
        executor = ThreadPoolExecutor(max_workers=len(pending))
        future_to_item = {
            executor.submit(cell_runner.run, request): (request, cell) for request, cell in pending
        }
        exogenous_evidence: tuple[Path, Sha256] | None = None
        interrupted: KeyboardInterrupt | None = None
        try:
            for future in as_completed(future_to_item):
                request, cell = future_to_item[future]
                try:
                    result = future.result()
                except ExogenousCampaignError as error:
                    exogenous_evidence = _persist_exogenous_failure(
                        config,
                        manifest,
                        source_cell=cell,
                        source_request=request,
                        wave_items=wave_items,
                        error=error,
                    )
                    continue
                except KeyboardInterrupt as error:
                    interrupted = error
                    _cancel_active_searches(cell_runner)
                    break
                except OSError as error:
                    outcome = _search_failed_cell(config, manifest, cell, request, error)
                    result = None
                else:
                    outcome = _cell_outcome(config, manifest, cell, request, result)
                prior = records_by_id.get(request.cell_id)
                records_by_id[request.cell_id] = CellExecutionRecord(
                    outcome=outcome,
                    result=result,
                    model_prep=None if prior is None else prior.model_prep,
                )
        except KeyboardInterrupt as error:
            interrupted = error
            _cancel_active_searches(cell_runner)
        finally:
            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                _ = shutdown(wait=interrupted is None, cancel_futures=interrupted is not None)
        if interrupted is not None:
            for record in records_by_id.values():
                if record.outcome.cell_id not in completed and (
                    record.result is not None or record.outcome.terminal_reason != "completed"
                ):
                    _append_campaign_event(
                        config.artifact_root, "terminal", _outcome_payload(record.outcome)
                    )
            raise interrupted
        if exogenous_evidence is not None:
            evidence_path, evidence_sha256 = exogenous_evidence
            for request, cell in wave_items:
                outcome = _invalidated_cell(
                    config,
                    manifest,
                    cell,
                    request,
                    evidence_path=evidence_path,
                    evidence_sha256=evidence_sha256,
                )
                records_by_id[request.cell_id] = CellExecutionRecord(
                    outcome=outcome, result=None, model_prep=None
                )
    ordered = tuple(records_by_id[request.cell_id] for request, _cell in wave_items)
    for record in ordered:
        if record.outcome.cell_id not in completed:
            _append_campaign_event(
                config.artifact_root, "terminal", _outcome_payload(record.outcome)
            )
    return ordered


def _run_deferred_post_search(
    records: list[CellExecutionRecord],
    requests: tuple[CellRunRequest, ...],
    post_search: PostSearchRunner,
    *,
    allow_build: bool = True,
) -> dict[str, PostSearchOutcome]:
    if not hasattr(post_search, "coverage_and_proof"):
        raise CoordinatorError(reason="post-search accounting unavailable")
    requests_by_id = {request.cell_id: request for request in requests}
    outcomes: dict[str, PostSearchOutcome] = {}
    for record in records:
        cell = record.outcome
        if cell.terminal_reason == "treatment-failed":
            request = requests_by_id[cell.cell_id]
            _ensure_treatment_failure_corpus_index(request, cell)
            treatment_coverage = getattr(post_search, "treatment_coverage", None)
            if not callable(treatment_coverage):
                raise CoordinatorError(reason="treatment coverage accounting unavailable")
            try:
                runner = cast("Callable[[CellRunRequest], PostSearchOutcome]", treatment_coverage)
                outcomes[cell.cell_id] = runner(request)
            except (OSError, ValueError) as error:
                raise CoordinatorError(reason="treatment coverage evidence invalid") from error
            continue
        if cell.terminal_reason != "completed":
            continue
        request = requests_by_id[cell.cell_id]
        if record.result is None:
            try:
                result = validate_cell_evidence(request)
                outcomes[cell.cell_id] = load_post_search_outcome(request, result)
            except (OSError, ValueError, CellEvidenceError) as error:
                raise CoordinatorError(reason="completed cell evidence invalid") from error
            continue
        try:
            outcomes[cell.cell_id] = load_post_search_outcome(request, record.result)
        except (OSError, ValueError):
            if not allow_build:
                raise CoordinatorError(reason="completed post-search evidence invalid") from None
            outcomes[cell.cell_id] = post_search.coverage_and_proof(request, record.result)
    return outcomes


def _cancel_active_searches(cell_runner: CellRunner) -> None:
    cancel_active = getattr(cell_runner, "cancel_active", None)
    if callable(cancel_active):
        _ = cancel_active()


def _resident_model_load_accounting(
    config: CampaignConfig, pool: ResidentPoolSurface
) -> tuple[
    ResidentModelLoadEvidence | None,
    ArtifactBackingEvidence | None,
    tuple[int, ...],
]:
    raw_telemetry = getattr(pool, "load_telemetry", None)
    if raw_telemetry is None:
        return None, None, ()
    if not isinstance(raw_telemetry, tuple):
        raise CoordinatorError(reason="resident model-load telemetry is invalid")
    try:
        evidence = build_resident_model_load_evidence(
            cast("tuple[WorkerLoadTelemetry, ...]", raw_telemetry)
        )
        worker_pids = tuple(worker.process_id for worker in evidence.workers)
        backing = persist_resident_model_load_evidence(config.artifact_root, evidence)
        reopened = load_resident_model_load_evidence(
            config.artifact_root,
            backing,
            expected_worker_pids=worker_pids,
        )
    except ResidentEvidenceError as error:
        raise CoordinatorError(reason=f"resident model-load evidence invalid: {error}") from error
    if reopened != evidence:
        raise CoordinatorError(reason="resident model-load aggregate mismatch")
    return evidence, backing, worker_pids


def _complete_from_terminal_events(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    schedule: CampaignSchedule,
    existing_events: tuple[_ExistingEvent, ...],
    post_search: PostSearchRunner,
    persist: bool = True,
) -> CampaignOutcome:
    launches = {event.cell_id: event for event in existing_events if event.phase == "launch"}
    terminals = {event.cell_id: event for event in existing_events if event.phase == "terminal"}
    request_list: list[CellRunRequest] = []
    records: list[CellExecutionRecord] = []
    for cell in schedule.cells:
        cell_id = _cell_id(cell)
        terminal = terminals.get(cell_id)
        if terminal is None:
            raise CoordinatorError(reason="campaign outcome terminal event missing")
        outcome = _recovered_cell(terminal)
        generated = _cell_request(
            config,
            manifest,
            cell,
            {int(cell.repetition): outcome.bank_sha256},
            config.resource_map,
        )
        request = _request_for_terminal_event(
            config=config,
            manifest=manifest,
            cell=cell,
            generated=generated,
            launch=launches.get(cell_id),
            terminal=terminal,
        )
        request_list.append(request)
        records.append(_terminal_record_from_evidence(terminal, request))
    requests = tuple(request_list)
    post_search_by_cell = _run_deferred_post_search(
        records, requests, post_search, allow_build=persist
    )
    banks = {
        repetition: _load_candidate_bank(path, repetition)
        for repetition in range(10)
        if (path := _bank_evidence_path(config, repetition)).exists()
    }
    outcome = _campaign_outcome(
        manifest,
        (),
        tuple(record.outcome for record in records),
        schedule,
        banks,
        post_search_by_cell,
    )
    if persist:
        _persist_raw_arm_reports(
            config.artifact_root,
            outcome,
            post_search_by_cell,
            config.c0_sha256,
            tuple(records),
            requests,
        )
        _persist_campaign_outcome(config.artifact_root, outcome)
    return outcome


def _terminal_record_from_evidence(
    event: _ExistingEvent, request: CellRunRequest
) -> CellExecutionRecord:
    outcome = _recovered_cell(event)
    result = (
        _load_completed_cell_result(request) if outcome.terminal_reason == "completed" else None
    )
    prep_path = _model_prep_evidence_path(request)
    model_prep = (
        _load_model_prep_result(prep_path, request.cell_id)
        if prep_path.exists() and outcome.arm in MODEL_ARMS
        else None
    )
    return CellExecutionRecord(outcome=outcome, result=result, model_prep=model_prep)


def _request_for_terminal_event(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    generated: CellRunRequest,
    launch: _ExistingEvent | None,
    terminal: _ExistingEvent,
) -> CellRunRequest:
    if launch is not None:
        return _request_from_launch(launch, generated)
    outcome = _recovered_cell(terminal)
    expected = _expired_cell(config, manifest, cell, generated)
    if outcome != expected or not _has_never_started_expiration_evidence(config, generated):
        raise CoordinatorError(reason="terminal event requires launch")
    return generated


def _has_never_started_expiration_evidence(config: CampaignConfig, request: CellRunRequest) -> bool:
    search_evidence = (
        request.artifact_root / "run-manifest.json",
        request.artifact_root / "events.jsonl",
        request.artifact_root / "final-result.json",
        request.artifact_root / "attempts",
    )
    if any(path.exists() for path in search_evidence):
        return False
    try:
        rows = load_corpus_index(_corpus_index_request(request))
    except CoverageSeriesError:
        return False
    return (
        tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
        and all(row.cumulative_hashes == (config.c0_sha256,) for row in rows)
        and len({row.cumulative_set_sha256 for row in rows}) == 1
    )


def _adapters_by_repetition(
    config: CampaignConfig,
    schedule: CampaignSchedule,
    pool: ResidentPoolSurface,
    banks: dict[int, CandidateBank],
    worker_pids: set[int],
    *,
    repetitions: tuple[int, ...] | None = None,
) -> dict[int, dict[ModelArm, RevisionSampler]]:
    adapters: dict[int, dict[ModelArm, RevisionSampler]] = {}
    for repetition in range(10) if repetitions is None else repetitions:
        cells = tuple(
            cell
            for cell in schedule.cells
            if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS
        )
        bindings: list[ArmBankBinding] = []
        for cell in cells:
            worker = pool.worker_for(cell)
            if worker.process_id_if_started is not None:
                worker_pids.add(worker.process_id_if_started)
            bindings.append(
                ArmBankBinding(
                    cast("ModelArm", cell.arm),
                    cast("RevisionSampler", worker),
                    config.artifact_root / "mutable" / f"rep-{int(cell.repetition):02d}" / cell.arm,
                )
            )
        built = build_frozen_bank_adapters(banks[repetition], tuple(bindings))
        adapters[repetition] = {adapter.arm: adapter for adapter in built}
    return adapters


def _bank_sampler_for_repetition(
    schedule: CampaignSchedule, pool: ResidentPoolSurface, repetition: int
) -> RevisionSampler:
    for cell in schedule.cells:
        if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS:
            bank_worker_for = getattr(pool, "bank_worker_for", None)
            if callable(bank_worker_for):
                return cast("RevisionSampler", bank_worker_for(cell))
            return cast("RevisionSampler", pool.worker_for(cell))
    raise CoordinatorError(reason="model repetition has no resident worker")


def _bank_evidence_path(config: CampaignConfig, repetition: int) -> Path:
    return _bank_evidence_path_from_root(config.artifact_root, repetition)


def _bank_evidence_path_from_root(root: Path, repetition: int) -> Path:
    return root / "banks" / f"rep-{repetition:02d}" / "candidate-bank.json"


def bank_evidence_path(config: CampaignConfig, repetition: int) -> Path:
    """Return the canonical candidate-bank evidence path for one repetition."""
    return _bank_evidence_path(config, repetition)


def load_persisted_candidate_bank(path: Path, repetition: int) -> CandidateBank:
    """Load and validate canonical candidate-bank evidence."""
    return _load_candidate_bank(path, repetition)


def campaign_mutable_root(config: CampaignConfig, cell: CellSchedule) -> Path:
    """Return the canonical mutable root for one scheduled cell."""
    return _mutable_root(config, cell)


def campaign_trajectory_hash(config: CampaignConfig, cell: CellSchedule, bank_hash: str) -> str:
    """Return the canonical trajectory-set hash for one scheduled cell."""
    return _trajectory_hash(config, cell, bank_hash)


def _persist_candidate_bank(
    path: Path,
    repetition: int,
    bank: CandidateBank,
    elapsed_seconds: float,
    *,
    bank_ready_elapsed_seconds: float | None = None,
    producer_gpu_index: int = 0,
) -> None:
    evidence = _candidate_bank_evidence(
        repetition,
        bank,
        elapsed_seconds,
        bank_ready_elapsed_seconds=(
            elapsed_seconds if bank_ready_elapsed_seconds is None else bank_ready_elapsed_seconds
        ),
        producer_gpu_index=producer_gpu_index,
    )
    payload = canonical_json_bytes(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    evidence = evidence.model_copy(update={"evidence_sha256": sha256_bytes(payload)})
    _write_once_canonical_model(path, evidence)


def _load_candidate_bank(path: Path, repetition: int) -> CandidateBank:
    evidence = _load_candidate_bank_evidence(path, repetition)
    parents = tuple(_frozen_parent(item) for item in evidence.raw_parents)
    selected_by_hash = {parent.raw_parent_sha256: parent for parent in parents}
    try:
        selected = tuple(selected_by_hash[item] for item in evidence.selected_raw_parent_sha256)
    except KeyError as error:
        raise CoordinatorError(reason="candidate-bank selected parent mismatch") from error
    bank = CandidateBank(
        raw_parents=parents,
        selected_parents=selected,
        requested_seed_base=evidence.requested_seed_base,
        duplicates=tuple(
            DuplicateParent(
                item.materialized_sha256,
                item.duplicate_raw_parent_sha256,
                item.kept_raw_parent_sha256,
            )
            for item in evidence.duplicates
        ),
        primary_raw_count=evidence.primary_raw_count,
        supplement_raw_count=evidence.supplement_raw_count,
        unique_parent_count=evidence.unique_parent_count,
        raw_token_states_sha256=_candidate_bank_hash_parts(
            tuple(parent.result.token_state_sha256 for parent in parents)
        ),
        raw_trajectories_sha256=_candidate_bank_hash_parts(
            tuple(parent.result.trajectory_evidence_sha256 for parent in parents)
        ),
        raw_materialized_sha256=_candidate_bank_hash_parts(
            tuple(parent.materialized_sha256 for parent in parents)
        ),
        duplicate_evidence_sha256=_candidate_bank_hash_parts(
            tuple(
                (
                    f"{item.materialized_sha256}:"
                    f"{item.duplicate_raw_parent_sha256}:"
                    f"{item.kept_raw_parent_sha256}"
                )
                for item in evidence.duplicates
            )
        ),
        selected_bank_sha256=_candidate_bank_hash_parts(
            tuple(parent.raw_parent_sha256 for parent in selected)
        ),
        physical_cost=BankGenerationCost(
            evidence.physical_cost.initial_requests,
            evidence.physical_cost.nfe,
            evidence.physical_cost.forward_calls,
            evidence.physical_cost.gpu_ms,
        ),
    )
    rebuilt = _candidate_bank_evidence(
        repetition,
        bank,
        evidence.measured_elapsed_seconds,
        bank_ready_elapsed_seconds=evidence.bank_ready_elapsed_seconds,
        producer_gpu_index=evidence.producer_gpu_index,
    ).model_copy(update={"schema_version": evidence.schema_version})
    if rebuilt.model_dump(mode="json", exclude={"evidence_sha256"}) != evidence.model_dump(
        mode="json", exclude={"evidence_sha256"}
    ):
        raise CoordinatorError(reason="candidate-bank canonical content mismatch")
    return bank


def _load_candidate_bank_evidence(path: Path, repetition: int) -> CandidateBankEvidence:
    try:
        evidence = CandidateBankEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid candidate-bank evidence") from error
    _require_canonical_model(path, evidence)
    if evidence.repetition != repetition:
        raise CoordinatorError(reason="candidate-bank repetition mismatch")
    unsigned = canonical_json_bytes(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    if evidence.evidence_sha256 != sha256_bytes(unsigned):
        raise CoordinatorError(reason="candidate-bank evidence hash mismatch")
    return evidence


def _candidate_bank_elapsed_seconds(path: Path) -> float:
    repetition = int(path.parent.name.removeprefix("rep-"))
    return _load_candidate_bank_evidence(path, repetition).measured_elapsed_seconds


def _candidate_bank_evidence(
    repetition: int,
    bank: CandidateBank,
    elapsed_seconds: float,
    *,
    bank_ready_elapsed_seconds: float | None = None,
    producer_gpu_index: int = 0,
) -> CandidateBankEvidence:
    return CandidateBankEvidence(
        repetition=repetition,
        requested_seed_base=bank.requested_seed_base,
        raw_parents=tuple(_parent_evidence(parent) for parent in bank.raw_parents),
        selected_raw_parent_sha256=tuple(
            parent.raw_parent_sha256 for parent in bank.selected_parents
        ),
        duplicates=tuple(
            DuplicateParentEvidence(
                materialized_sha256=item.materialized_sha256,
                duplicate_raw_parent_sha256=item.duplicate_raw_parent_sha256,
                kept_raw_parent_sha256=item.kept_raw_parent_sha256,
            )
            for item in bank.duplicates
        ),
        primary_raw_count=bank.primary_raw_count,
        supplement_raw_count=bank.supplement_raw_count,
        unique_parent_count=bank.unique_parent_count,
        raw_token_states_sha256=bank.raw_token_states_sha256,
        raw_trajectories_sha256=bank.raw_trajectories_sha256,
        raw_materialized_sha256=bank.raw_materialized_sha256,
        duplicate_evidence_sha256=bank.duplicate_evidence_sha256,
        selected_bank_sha256=bank.selected_bank_sha256,
        physical_cost=BankGenerationCostEvidence(
            initial_requests=bank.physical_cost.initial_requests,
            nfe=bank.physical_cost.nfe,
            forward_calls=bank.physical_cost.forward_calls,
            gpu_ms=bank.physical_cost.gpu_ms,
        ),
        measured_elapsed_seconds=elapsed_seconds,
        physical_duration_seconds=elapsed_seconds,
        bank_ready_elapsed_seconds=(
            elapsed_seconds if bank_ready_elapsed_seconds is None else bank_ready_elapsed_seconds
        ),
        producer_gpu_index=producer_gpu_index,
    )


def _parent_evidence(parent: FrozenParent) -> BankParentEvidence:
    return BankParentEvidence(
        request=parent.request,
        result=parent.result,
        content_hex=parent.content.hex(),
        materialized_sha256=parent.materialized_sha256,
        raw_parent_sha256=parent.raw_parent_sha256,
    )


def _frozen_parent(evidence: BankParentEvidence) -> FrozenParent:
    content = bytes.fromhex(evidence.content_hex)
    if sha256_bytes(content) != evidence.materialized_sha256:
        raise CoordinatorError(reason="candidate-bank materialized byte mismatch")
    if evidence.result.token_state_sha256 != sha256_token_ids(evidence.result.token_ids):
        raise CoordinatorError(reason="candidate-bank token-state hash mismatch")
    if evidence.result.trajectory_evidence_sha256 != revision_trajectory_sha256(evidence.result):
        raise CoordinatorError(reason="candidate-bank trajectory hash mismatch")
    snapshot = materialize_particle(
        decode_particle(evidence.result.selectors, "validity-preserving")
    )
    if content != snapshot.content or evidence.materialized_sha256 != snapshot.sha256:
        raise CoordinatorError(reason="candidate-bank materialized byte mismatch")
    raw_parent_sha256 = _candidate_bank_parent_sha256(evidence)
    if evidence.raw_parent_sha256 != raw_parent_sha256:
        raise CoordinatorError(reason="candidate-bank raw parent hash mismatch")
    return FrozenParent(
        evidence.request,
        evidence.result,
        content,
        evidence.materialized_sha256,
        raw_parent_sha256,
    )


def _candidate_bank_parent_sha256(evidence: BankParentEvidence) -> str:
    return _candidate_bank_hash_parts(
        (
            evidence.request.particle_id,
            evidence.request.seed,
            evidence.result.token_state_sha256,
            evidence.result.trajectory_evidence_sha256,
            evidence.materialized_sha256,
        )
    )


def _candidate_bank_hash_parts(parts: tuple[str | int, ...]) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _model_prep_evidence_path(request: CellRunRequest) -> Path:
    return request.artifact_root / "model-prep-result.json"


def _persist_model_prep_result(
    path: Path,
    cell_id: str,
    result: ModelArmExecutionResult,
    *,
    bank_physical_duration_seconds: float = 0.0,
) -> None:
    if result.written_snapshot_count != len(result.written_paths) or any(
        not snapshot.is_file() or snapshot.is_symlink() for snapshot in result.written_paths
    ):
        return
    evidence = _model_prep_evidence(
        cell_id,
        result,
        bank_physical_duration_seconds=bank_physical_duration_seconds,
    )
    payload = canonical_json_bytes(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    evidence = evidence.model_copy(update={"evidence_sha256": sha256_bytes(payload)})
    _write_once_canonical_model(path, evidence)


def _load_model_prep_result(path: Path, cell_id: str) -> ModelArmExecutionResult:
    evidence = _load_model_prep_evidence(path, cell_id)
    return ModelArmExecutionResult(
        written_snapshot_count=evidence.snapshot_count,
        written_paths=tuple(snapshot.path for snapshot in evidence.snapshots),
        nfe=evidence.nfe,
        forward_calls=evidence.forward_calls,
        gpu_ms=evidence.gpu_ms,
        bank_ready_elapsed_seconds=evidence.bank_ready_elapsed_seconds,
        model_ready_elapsed_seconds=evidence.model_ready_elapsed_seconds,
        pre_release_corpus_set_sha256=evidence.pre_release_corpus_set_sha256,
        producer_gpu_index=evidence.producer_gpu_index,
    )


def _load_model_prep_evidence(path: Path, cell_id: str) -> ModelPrepEvidence:
    try:
        evidence = ModelPrepEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid model-prep evidence") from error
    _require_canonical_model(path, evidence)
    if evidence.cell_id != cell_id or evidence.snapshot_count != len(evidence.snapshots):
        raise CoordinatorError(reason="model-prep evidence cell mismatch")
    unsigned = canonical_json_bytes(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    if evidence.evidence_sha256 != sha256_bytes(unsigned):
        raise CoordinatorError(reason="model-prep evidence hash mismatch")
    for snapshot in evidence.snapshots:
        try:
            content = snapshot.path.read_bytes()
        except OSError as error:
            raise CoordinatorError(reason="model-prep snapshot unreadable") from error
        if len(content) != snapshot.size_bytes or sha256_bytes(content) != snapshot.sha256:
            raise CoordinatorError(reason="model-prep snapshot hash mismatch")
    return evidence


def _model_prep_evidence(
    cell_id: str,
    result: ModelArmExecutionResult,
    *,
    bank_physical_duration_seconds: float = 0.0,
) -> ModelPrepEvidence:
    snapshots = tuple(_snapshot_backing(path) for path in result.written_paths)
    if result.written_snapshot_count != len(snapshots):
        raise CoordinatorError(reason="model-prep snapshot count mismatch")
    return ModelPrepEvidence(
        cell_id=cell_id,
        snapshot_count=result.written_snapshot_count,
        snapshots=snapshots,
        nfe=result.nfe,
        forward_calls=result.forward_calls,
        gpu_ms=result.gpu_ms,
        bank_ready_elapsed_seconds=result.bank_ready_elapsed_seconds,
        bank_physical_duration_seconds=bank_physical_duration_seconds,
        model_ready_elapsed_seconds=result.model_ready_elapsed_seconds,
        pre_release_corpus_set_sha256=result.pre_release_corpus_set_sha256,
        producer_gpu_index=result.producer_gpu_index or 0,
    )


def _snapshot_backing(path: Path) -> SnapshotBackingEvidence:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise CoordinatorError(reason="model-prep snapshot unreadable") from error
    return SnapshotBackingEvidence(path=path, sha256=sha256_bytes(content), size_bytes=len(content))


def _pending_cell(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
) -> CellOutcome:
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=int(cell.repetition),
        wave_index=int(cell.wave_index),
        arm=cell.arm,
        fuzzer_seed=int(cell.fuzzer_seed),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        artifact_root=request.artifact_root,
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=_trajectory_hash(config, cell, request.bank_sha256),
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason="completed",
        coverage_frozen_through_seconds=None,
        search_cpu_seconds=0.0,
    )


def _cell_outcome(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
    result: CellResult,
) -> CellOutcome:
    terminal = _terminal_reason(result)
    treatment_failure_path: Path | None = None
    treatment_failure_sha256: Sha256 | None = None
    trajectory_set_sha256 = _trajectory_hash(config, cell, request.bank_sha256)
    if terminal == "treatment-failed":
        treatment_failure_path, treatment_failure_sha256 = _persist_treatment_failure_message(
            config=config,
            manifest=manifest,
            request=request,
            phase="search",
            exception_class="CellResultFailed",
            message=result.event_stream_sha256,
        )
        trajectory_set_sha256 = treatment_failure_sha256
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=int(cell.repetition),
        wave_index=int(cell.wave_index),
        arm=cell.arm,
        fuzzer_seed=int(cell.fuzzer_seed),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        artifact_root=request.artifact_root,
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=trajectory_set_sha256,
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason=terminal,
        coverage_frozen_through_seconds=10800 if terminal == "treatment-failed" else None,
        search_cpu_seconds=result.search_accounting.cpu_seconds,
        treatment_failure_path=treatment_failure_path,
        treatment_failure_sha256=treatment_failure_sha256,
    )


def _load_completed_cell_result(request: CellRunRequest) -> CellResult:
    try:
        return validate_cell_evidence(request)
    except CellEvidenceError as error:
        raise CoordinatorError(reason="completed cell evidence invalid") from error


def _search_failed_cell(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
    error: OSError,
) -> CellOutcome:
    failure_path, failure_sha256 = _persist_treatment_failure(
        config=config,
        manifest=manifest,
        request=request,
        phase="search",
        error=error,
    )
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=int(cell.repetition),
        wave_index=int(cell.wave_index),
        arm=cell.arm,
        fuzzer_seed=int(cell.fuzzer_seed),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        artifact_root=request.artifact_root,
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=failure_sha256,
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason="treatment-failed",
        coverage_frozen_through_seconds=10_800,
        search_cpu_seconds=0.0,
        treatment_failure_path=failure_path,
        treatment_failure_sha256=failure_sha256,
    )


def _model_prep_failed_cell(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
    error: Exception,
) -> CellOutcome:
    failure_path, failure_sha256 = _persist_treatment_failure(
        config=config,
        manifest=manifest,
        request=request,
        phase="model-prep",
        error=error,
    )
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=int(cell.repetition),
        wave_index=int(cell.wave_index),
        arm=cell.arm,
        fuzzer_seed=int(cell.fuzzer_seed),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        artifact_root=request.artifact_root,
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=failure_sha256,
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason="treatment-failed",
        coverage_frozen_through_seconds=10_800,
        search_cpu_seconds=0.0,
        treatment_failure_path=failure_path,
        treatment_failure_sha256=failure_sha256,
    )


def _persist_treatment_failure(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    request: CellRunRequest,
    phase: Literal["model-prep", "search"],
    error: Exception,
) -> tuple[Path, Sha256]:
    return _persist_treatment_failure_message(
        config=config,
        manifest=manifest,
        request=request,
        phase=phase,
        exception_class=error.__class__.__name__,
        message=str(error),
    )


def _persist_treatment_failure_message(
    *,
    config: CampaignConfig,
    manifest: CampaignManifest,
    request: CellRunRequest,
    phase: Literal["model-prep", "search"],
    exception_class: str,
    message: str,
) -> tuple[Path, Sha256]:
    evidence = TreatmentFailureEvidence(
        cell_id=request.cell_id,
        failure_phase=phase,
        exception_class=exception_class,
        message_sha256=sha256_bytes(message.encode()),
        request_sha256=_model_sha256(request),
        bank_sha256=request.bank_sha256,
        config_sha256=config.config_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
    )
    path = request.artifact_root / f"{phase}-failure.json"
    _write_once_canonical_model(path, evidence)
    return path, _model_sha256(evidence)


def _recovered_cell(event: _ExistingEvent) -> CellOutcome:
    try:
        payload: dict[str, JsonValue | Path] = {
            key: value for key, value in event.payload.items() if key != "phase"
        }
        artifact_root = payload.get("artifact_root")
        if isinstance(artifact_root, str):
            payload["artifact_root"] = Path(artifact_root)
        mutable_root = payload.get("mutable_root")
        if isinstance(mutable_root, str):
            payload["mutable_root"] = Path(mutable_root)
        treatment_failure_path = payload.get("treatment_failure_path")
        if isinstance(treatment_failure_path, str):
            payload["treatment_failure_path"] = Path(treatment_failure_path)
        exogenous_evidence_path = payload.get("exogenous_evidence_path")
        if isinstance(exogenous_evidence_path, str):
            payload["exogenous_evidence_path"] = Path(exogenous_evidence_path)
        return CellOutcome.model_validate(payload)
    except ValidationError as error:
        raise CoordinatorError(reason="invalid recovered terminal outcome") from error


def _campaign_outcome(  # noqa: PLR0917
    manifest: CampaignManifest,
    pids: tuple[int, ...],
    cells: tuple[CellOutcome, ...],
    schedule: CampaignSchedule,
    banks: dict[int, CandidateBank],
    post_search_by_cell: dict[str, PostSearchOutcome],
    *,
    resident_model_load: ResidentModelLoadEvidence | None = None,
    resident_model_load_evidence: ArtifactBackingEvidence | None = None,
) -> CampaignOutcome:
    pairs = tuple(_pair_outcome_for_wave(wave, cells) for wave in schedule.waves)
    logical: dict[ModelArm, int] = {
        arm: sum(
            claim.logical_raw_bank_cost.initial_requests
            for bank in banks.values()
            for claim in bank.accounting.arm_claims
            if claim.arm == arm
        )
        for arm in MODEL_ARMS
    }
    return CampaignOutcome(
        manifest_sha256=manifest.manifest_sha256 or "",
        resident_worker_pids=pids,
        launched_cell_count=len(cells),
        cells=cells,
        pairs=pairs,
        accounting=CampaignAccounting(
            bank_physical_initial_requests=sum(
                bank.accounting.physical_campaign_cost.initial_requests for bank in banks.values()
            ),
            bank_logical_initial_requests_by_arm=logical,
            search_cpu_seconds=sum(cell.search_cpu_seconds for cell in cells),
            coverage_replay_count=sum(
                item.coverage_replay_count for item in post_search_by_cell.values()
            ),
            proof_replay_count=sum(
                item.proof_replay_count for item in post_search_by_cell.values()
            ),
            resident_model_load=resident_model_load,
            resident_model_load_evidence=resident_model_load_evidence,
        ),
    )


def _pair_outcome_for_wave(wave: WaveSchedule, cells: tuple[CellOutcome, ...]) -> PairOutcome:
    wave_index = int(wave.index)
    exogenous = tuple(
        cell
        for cell in cells
        if cell.wave_index == wave_index and cell.exogenous_evidence_path is not None
    )
    if not exogenous:
        return PairOutcome(
            wave_index=wave_index,
            repetitions=(int(wave.repetitions[0]), int(wave.repetitions[1])),
            status="complete",
        )
    first = exogenous[0]
    if first.exogenous_evidence_path is None or first.exogenous_evidence_sha256 is None:
        raise CoordinatorError(reason="exogenous pair evidence missing")
    _ = _load_exogenous_evidence(first.exogenous_evidence_path, first.exogenous_evidence_sha256)
    return PairOutcome(
        wave_index=wave_index,
        repetitions=(int(wave.repetitions[0]), int(wave.repetitions[1])),
        status="exogenous-invalidated",
        exogenous_evidence_path=first.exogenous_evidence_path,
        exogenous_evidence_sha256=first.exogenous_evidence_sha256,
    )


def _apply_wave_exogenous_overlay(
    records_by_id: dict[str, CellExecutionRecord], wave: WaveSchedule
) -> None:
    exogenous = next(
        (
            record.outcome
            for record in records_by_id.values()
            if record.outcome.wave_index == int(wave.index)
            and record.outcome.exogenous_evidence_path is not None
        ),
        None,
    )
    if exogenous is None:
        return
    if exogenous.exogenous_evidence_path is None or exogenous.exogenous_evidence_sha256 is None:
        raise CoordinatorError(reason="exogenous pair evidence missing")
    for cell in wave.cells:
        cell_id = _cell_id(cell)
        record = records_by_id.get(cell_id)
        if record is None:
            continue
        outcome = record.outcome
        records_by_id[cell_id] = CellExecutionRecord(
            outcome=outcome.model_copy(
                update={
                    "exogenous_evidence_path": exogenous.exogenous_evidence_path,
                    "exogenous_evidence_sha256": exogenous.exogenous_evidence_sha256,
                }
            ),
            result=record.result,
            model_prep=record.model_prep,
        )


def _invalidated_cell(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
    *,
    evidence_path: Path,
    evidence_sha256: Sha256,
) -> CellOutcome:
    return CellOutcome(
        cell_id=request.cell_id,
        repetition=int(cell.repetition),
        wave_index=int(cell.wave_index),
        arm=cell.arm,
        fuzzer_seed=int(cell.fuzzer_seed),
        deadline_monotonic_seconds=float(cell.deadline_monotonic_seconds),
        artifact_root=request.artifact_root,
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        trajectory_set_sha256=evidence_sha256,
        config_sha256=config.config_sha256,
        resource_map_sha256=manifest.resource_map_sha256,
        manifest_sha256=manifest.manifest_sha256 or "",
        terminal_reason="exogenous-invalidated",
        coverage_frozen_through_seconds=10800,
        search_cpu_seconds=0.0,
        exogenous_evidence_path=evidence_path,
        exogenous_evidence_sha256=evidence_sha256,
    )


def _persist_exogenous_failure(
    config: CampaignConfig,
    manifest: CampaignManifest,
    *,
    source_cell: CellSchedule,
    source_request: CellRunRequest,
    wave_items: tuple[tuple[CellRunRequest, CellSchedule], ...],
    error: ExogenousCampaignError,
) -> tuple[Path, Sha256]:
    if error.cause not in {
        "gpu-xid",
        "gpu-reset",
        "host-power-loss",
        "storage-io",
        "orchestrator-corruption",
    }:
        raise CoordinatorError(reason="exogenous evidence cause is not preregistered")
    repetitions = tuple(sorted({int(cell.repetition) for _, cell in wave_items}))
    source_wave = int(source_cell.wave_index)
    affected_cell_ids = tuple(
        request.cell_id for request, cell in wave_items if int(cell.wave_index) == source_wave
    )
    raw_evidence = str(error.evidence)
    evidence = ExogenousFailureEvidence(
        cause=error.cause,
        source_cell_id=source_request.cell_id,
        raw_evidence=raw_evidence,
        raw_evidence_sha256=sha256_bytes(raw_evidence.encode()),
        affected_wave_index=source_wave,
        affected_repetitions=cast("tuple[int, int]", repetitions),
        affected_cell_ids=affected_cell_ids,
        prior_event_stream_sha256=_event_stream_sha256(config.artifact_root),
        manifest_sha256=manifest.manifest_sha256 or "",
    )
    path = config.artifact_root / "exogenous" / f"wave-{source_wave:02d}.json"
    _write_once_canonical_model(path, evidence)
    _validate_exogenous_evidence(path, evidence)
    return path, _model_sha256(evidence)


def _ensure_treatment_failure_corpus_index(request: CellRunRequest, cell: CellOutcome) -> None:
    if cell.terminal_reason != "treatment-failed":
        return
    index_request = CorpusIndexRequest(
        artifact_root=request.artifact_root,
        corpus_root=request.corpus_root,
        store_root=request.artifact_root / "corpus-inputs",
        index_path=request.artifact_root / "corpus-index.jsonl",
    )
    try:
        rows = load_corpus_index(index_request)
        if tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS:
            return
        if not rows:
            _ = snapshot_corpus(index_request, 0, 0.0)
        _ = freeze_corpus_index(index_request, freeze_elapsed_seconds=0.0)
    except CoverageSeriesError as error:
        raise CoordinatorError(reason="treatment coverage corpus index invalid") from error


def _validate_exogenous_evidence(path: Path, expected: ExogenousFailureEvidence) -> None:
    try:
        loaded = ExogenousFailureEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="exogenous evidence invalid") from error
    _require_canonical_model(path, loaded)
    if loaded != expected or loaded.raw_evidence_sha256 != sha256_bytes(
        loaded.raw_evidence.encode()
    ):
        raise CoordinatorError(reason="exogenous evidence mismatch")


def _event_stream_sha256(root: Path) -> Sha256:
    path = root / _EVENTS_NAME
    if not path.exists():
        return sha256_bytes(b"")
    return sha256_bytes(path.read_bytes())


def _terminal_reason(result: CellResult) -> Literal["completed", "deadline", "treatment-failed"]:
    match result.status:
        case "completed":
            return "completed"
        case "deadline" | "restart_limit":
            return "deadline"
        case "failed":
            return "treatment-failed"
        case unreachable:
            assert_never(unreachable)


def _persist_manifest(root: Path, manifest: CampaignManifest) -> None:
    path = root / _MANIFEST_NAME
    payload = canonical_json_bytes(manifest.model_dump(mode="json"))
    _write_once_bytes(path, payload, label="manifest")


def load_complete_campaign_outcome(root: Path, manifest: CampaignManifest) -> CampaignOutcome:
    """Load the canonical completed outcome envelope and all backing evidence."""
    path = root / "campaign-outcome.json"
    raw_path = root / "raw-arm-reports.jsonl"
    events_path = root / _EVENTS_NAME
    manifest_path = root / _MANIFEST_NAME
    if not path.exists():
        raise CoordinatorError(reason="complete campaign outcome envelope missing")
    if not raw_path.exists() or not events_path.exists() or not manifest_path.exists():
        raise CoordinatorError(reason="campaign outcome backing evidence missing")
    try:
        envelope = CampaignOutcomeEnvelope.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid campaign outcome envelope") from error
    _require_canonical_model(path, envelope)
    if envelope.outcome.manifest_sha256 != manifest.manifest_sha256:
        raise CoordinatorError(reason="campaign outcome manifest mismatch")
    try:
        manifest_bytes = manifest_path.read_bytes()
        event_bytes = events_path.read_bytes()
        raw_bytes = raw_path.read_bytes()
    except OSError as error:
        raise CoordinatorError(reason="campaign outcome backing evidence missing") from error
    if sha256_bytes(manifest_bytes) != envelope.manifest_artifact_sha256:
        raise CoordinatorError(reason="campaign outcome manifest artifact mismatch")
    if sha256_bytes(event_bytes) != envelope.event_stream_sha256:
        raise CoordinatorError(reason="campaign outcome event stream mismatch")
    if (
        sha256_bytes(raw_bytes) != envelope.raw_report_sha256
        or len(raw_bytes) != envelope.raw_report_size_bytes
    ):
        raise CoordinatorError(reason="campaign outcome raw report mismatch")
    evidence_tree_sha256 = _evidence_tree_sha256(manifest_bytes, event_bytes, raw_bytes)
    if envelope.evidence_tree_sha256 != evidence_tree_sha256:
        raise CoordinatorError(reason="campaign outcome evidence tree mismatch")
    _validate_resident_model_load_accounting(root, envelope.outcome)
    events = _existing_events(root)
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=manifest.config.main_start_monotonic_seconds,
        gpu_count=len(manifest.config.resource_map.model_cores),
    )
    _validate_complete_event_contract(events, schedule, manifest)
    _validate_raw_report_chain(raw_bytes)
    _validate_complete_raw_report_contract(raw_bytes, schedule)
    if not _raw_report_post_search_complete(raw_bytes):
        raise CoordinatorError(reason="campaign outcome post-search evidence incomplete")
    if len(envelope.outcome.cells) != _COMPLETE_CELL_COUNT:
        raise CoordinatorError(reason="campaign outcome cell count mismatch")
    reconstructed = _complete_from_terminal_events(
        config=manifest.config,
        manifest=manifest,
        schedule=schedule,
        existing_events=events,
        post_search=_UnavailablePostSearch(),
        persist=False,
    )
    reconstructed_accounting = reconstructed.accounting.model_copy(
        update={
            "resident_model_load": envelope.outcome.accounting.resident_model_load,
            "resident_model_load_evidence": (
                envelope.outcome.accounting.resident_model_load_evidence
            ),
        }
    )
    reconstructed = reconstructed.model_copy(
        update={
            "resident_worker_pids": envelope.outcome.resident_worker_pids,
            "accounting": reconstructed_accounting,
        }
    )
    if reconstructed != envelope.outcome:
        raise CoordinatorError(reason="campaign outcome semantic mismatch")
    return envelope.outcome


def _validate_resident_model_load_accounting(root: Path, outcome: CampaignOutcome) -> None:
    load = outcome.accounting.resident_model_load
    backing = outcome.accounting.resident_model_load_evidence
    if load is None and backing is None:
        return
    if load is None or backing is None:
        raise CoordinatorError(reason="resident model-load accounting is incomplete")
    try:
        reopened = load_resident_model_load_evidence(
            root,
            backing,
            expected_worker_pids=outcome.resident_worker_pids,
        )
    except ResidentEvidenceError as error:
        raise CoordinatorError(reason=f"resident model-load evidence invalid: {error}") from error
    if reopened != load:
        raise CoordinatorError(reason="resident model-load aggregate mismatch")


def finalize_recovered_campaign_from_events(
    *,
    config: CampaignConfig,
    post_search: PostSearchRunner,
) -> CampaignOutcome:
    """Finalize a recovered 50-terminal campaign through the canonical outcome envelope."""
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds,
        gpu_count=len(config.resource_map.model_cores),
    )
    manifest = build_campaign_manifest(config, schedule)
    existing_events = _existing_events(config.artifact_root)
    terminals = {event.cell_id for event in existing_events if event.phase == "terminal"}
    if len(terminals) != _COMPLETE_CELL_COUNT:
        raise CoordinatorError(reason="recovered campaign is not terminal")
    return _complete_from_terminal_events(
        config=config,
        manifest=manifest,
        schedule=schedule,
        existing_events=existing_events,
        post_search=post_search,
    )


def expire_recovered_campaign(
    *,
    config: CampaignConfig,
    post_search: PostSearchRunner,
) -> CampaignOutcome:
    """Freeze every unfinished cell at its original deadline without new model work."""
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds,
        gpu_count=len(config.resource_map.model_cores),
    )
    manifest = build_campaign_manifest(config, schedule)
    events = _existing_events(config.artifact_root)
    launches = {event.cell_id: event for event in events if event.phase == "launch"}
    terminals = {event.cell_id: event for event in events if event.phase == "terminal"}
    banks = {
        repetition: _load_candidate_bank(path, repetition)
        for repetition in range(10)
        if (path := _bank_evidence_path(config, repetition)).exists()
    }
    requests: list[CellRunRequest] = []
    records: list[CellExecutionRecord] = []
    for cell in schedule.cells:
        cell_id = _cell_id(cell)
        launch = launches.get(cell_id)
        if launch is not None:
            request = _request_from_launch_only(launch)
        else:
            repetition = int(cell.repetition)
            bank_hash = (
                banks[repetition].selected_bank_sha256
                if repetition in banks and cell.arm in MODEL_ARMS
                else config.c0_sha256
            )
            request = _cell_request(
                config,
                manifest,
                cell,
                {repetition: bank_hash},
                config.resource_map,
            )
            pending = _pending_cell(config, manifest, cell, request)
            try:
                _ = seed_cell_corpus(config, pending, None)
            except C0SourceError as error:
                raise CoordinatorError(reason=error.reason) from error
            index_request = _corpus_index_request(request)
            if not load_corpus_index(index_request):
                _ = snapshot_corpus(index_request, 0, 0.0)
        requests.append(request)
        terminal = terminals.get(cell_id)
        if terminal is not None:
            request = _request_for_terminal_event(
                config=config,
                manifest=manifest,
                cell=cell,
                generated=request,
                launch=launch,
                terminal=terminal,
            )
            requests[-1] = request
            records.append(_terminal_record_from_evidence(terminal, request))
            continue
        if launch is None:
            try:
                _ = freeze_corpus_index(_corpus_index_request(request), freeze_elapsed_seconds=0.0)
            except CoverageSeriesError as error:
                raise CoordinatorError(reason="expired recovered corpus index invalid") from error
        else:
            _ = _append_due_pre_release_checkpoints(request, _WAVE_DURATION_SECONDS)
        record = CellExecutionRecord(
            outcome=_expired_cell(config, manifest, cell, request),
            result=None,
            model_prep=(
                _load_model_prep_result(_model_prep_evidence_path(request), request.cell_id)
                if _model_prep_evidence_path(request).exists() and cell.arm in MODEL_ARMS
                else None
            ),
        )
        _append_campaign_event(
            config.artifact_root,
            "terminal",
            _outcome_payload(record.outcome),
        )
        records.append(record)
    post_search_by_cell = _run_deferred_post_search(records, tuple(requests), post_search)
    outcome = _campaign_outcome(
        manifest,
        (),
        tuple(record.outcome for record in records),
        schedule,
        banks,
        post_search_by_cell,
    )
    _persist_raw_arm_reports(
        config.artifact_root,
        outcome,
        post_search_by_cell,
        config.c0_sha256,
        tuple(records),
        tuple(requests),
    )
    _persist_campaign_outcome(config.artifact_root, outcome)
    return outcome


def _persist_campaign_outcome(root: Path, outcome: CampaignOutcome) -> None:
    if len(outcome.cells) != _COMPLETE_CELL_COUNT:
        return
    raw_path = root / "raw-arm-reports.jsonl"
    events_path = root / _EVENTS_NAME
    manifest_path = root / _MANIFEST_NAME
    try:
        raw_bytes = raw_path.read_bytes()
        event_bytes = events_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise CoordinatorError(reason="campaign outcome backing evidence missing") from error
    if not _raw_report_post_search_complete(raw_bytes):
        return
    envelope = CampaignOutcomeEnvelope(
        outcome=outcome,
        manifest_artifact_sha256=sha256_bytes(manifest_bytes),
        event_stream_sha256=sha256_bytes(event_bytes),
        raw_report_sha256=sha256_bytes(raw_bytes),
        raw_report_size_bytes=len(raw_bytes),
        evidence_tree_sha256=_evidence_tree_sha256(manifest_bytes, event_bytes, raw_bytes),
    )
    _write_once_canonical_model(root / "campaign-outcome.json", envelope)


def _evidence_tree_sha256(manifest_bytes: bytes, event_bytes: bytes, raw_bytes: bytes) -> Sha256:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "event_stream_sha256": sha256_bytes(event_bytes),
                "manifest_artifact_sha256": sha256_bytes(manifest_bytes),
                "raw_report_sha256": sha256_bytes(raw_bytes),
            }
        )
    )


def _assert_new_or_resumable_root(root: Path, manifest: CampaignManifest) -> None:
    if not root.exists():
        return
    entries = tuple(root.iterdir())
    if not entries:
        return
    path = root / _MANIFEST_NAME
    if not path.is_file():
        raise CoordinatorError(reason="campaign root is not canonical")
    payload = canonical_json_bytes(manifest.model_dump(mode="json"))
    if path.read_bytes() != payload:
        raise CoordinatorError(reason="manifest mismatch")
    if (root / _EVENTS_NAME).exists():
        _ = _existing_events(root)
        return
    allowed = {path.resolve()}
    if any(entry.resolve() not in allowed for entry in entries):
        raise CoordinatorError(reason="campaign root is not canonical")


def _persist_raw_arm_reports(  # noqa: PLR0915, PLR0917
    root: Path,
    outcome: CampaignOutcome,
    post_search_by_cell: dict[str, PostSearchOutcome],
    c0_sha256: Sha256,
    records: tuple[CellExecutionRecord, ...] = (),
    requests: tuple[CellRunRequest, ...] = (),
) -> None:
    reports = root / "raw-arm-reports.jsonl"
    pair_by_wave = {pair.wave_index: pair.status for pair in outcome.pairs}
    prep_by_cell = {record.outcome.cell_id: record.model_prep for record in records}
    result_by_cell = {record.outcome.cell_id: record.result for record in records}
    request_by_cell = {request.cell_id: request for request in requests}
    lines: list[str] = []
    previous_row_sha256 = "0" * 64
    for cell in outcome.cells:
        post_search = post_search_by_cell.get(cell.cell_id)
        model_prep = prep_by_cell.get(cell.cell_id)
        request = request_by_cell.get(cell.cell_id)
        if request is None:
            raise CoordinatorError(reason="raw arm report missing launch request")
        result = result_by_cell.get(cell.cell_id)
        coverage_path: Path | None = None
        coverage_sha256: Sha256 | None = None
        coverage_envelope_sha256: Sha256 | None = None
        proof_path: Path | None = None
        proof_sha256: Sha256 | None = None
        proof_envelope_sha256: Sha256 | None = None
        proof_replay_count: int | None = None
        proof_wall_seconds: float | None = None
        proof_cpu_seconds: float | None = None
        coverage_replay_count: int | None = None
        deep_parse_path: Path | None = None
        deep_parse_sha256: Sha256 | None = None
        deep_parse_envelope_sha256: Sha256 | None = None
        deep_parse_replay_count: int | None = None
        deep_parse_success_count: int | None = None
        deep_parse_total_count: int | None = None
        if post_search is not None:
            coverage_path = post_search.coverage_series_path
            proof_path = post_search.proof_result_path
            deep_parse_path = post_search.deep_parse_envelope_path
            if coverage_path is not None:
                coverage_sha256 = post_search.coverage_series_sha256
                coverage_envelope_sha256 = post_search.coverage_envelope_sha256
                coverage_replay_count = post_search.coverage_replay_count
            if proof_path is not None:
                proof_sha256 = post_search.proof_result_sha256
                proof_envelope_sha256 = post_search.proof_envelope_sha256
                proof_replay_count = post_search.proof_replay_count
                proof_wall_seconds = post_search.proof_wall_seconds
                proof_cpu_seconds = post_search.proof_cpu_seconds
            if deep_parse_path is not None:
                deep_parse_sha256 = post_search.deep_parse_result_sha256
                deep_parse_envelope_sha256 = post_search.deep_parse_envelope_sha256
                deep_parse_replay_count = post_search.deep_parse_replay_count
                deep_parse_success_count = post_search.deep_parse_success_count
                deep_parse_total_count = post_search.deep_parse_total_count
        coverage_status = "available" if coverage_path is not None else "unavailable"
        proof_status = "available" if proof_path is not None else "unavailable"
        deep_parse_status = "available" if deep_parse_path is not None else "unavailable"
        candidate_bank_path = _bank_evidence_path_from_root(root, int(cell.repetition))
        bank_path = (
            candidate_bank_path if cell.arm in MODEL_ARMS and candidate_bank_path.exists() else None
        )
        bank_evidence = (
            None
            if bank_path is None
            else _load_candidate_bank_evidence(bank_path, int(cell.repetition))
        )
        release_path = _corpus_release_path(request)
        release_evidence = (
            _load_corpus_release(release_path, cell.cell_id)
            if cell.arm in MODEL_ARMS and release_path.exists()
            else None
        )
        snapshot_backing = (
            tuple(_snapshot_backing(path) for path in model_prep.written_paths)
            if model_prep is not None
            else ()
        )
        unsigned = RawArmReportRow(
            campaign_root=root,
            repetition=cell.repetition,
            wave_index=cell.wave_index,
            arm=cell.arm,
            cell_id=cell.cell_id,
            cell_request_sha256=_model_sha256(request),
            cell_result_sha256=None if result is None else _model_sha256(result),
            artifact_root=cell.artifact_root,
            coverage_status=coverage_status,
            coverage_series_path=coverage_path,
            coverage_series_sha256=coverage_sha256,
            coverage_envelope_sha256=coverage_envelope_sha256,
            proof_status=proof_status,
            proof_result_path=proof_path,
            proof_result_sha256=proof_sha256,
            proof_envelope_sha256=proof_envelope_sha256,
            deep_parse_status=deep_parse_status,
            deep_parse_envelope_path=deep_parse_path,
            deep_parse_result_sha256=deep_parse_sha256,
            deep_parse_envelope_sha256=deep_parse_envelope_sha256,
            deep_parse_replay_count=deep_parse_replay_count,
            deep_parse_success_count=deep_parse_success_count,
            deep_parse_total_count=deep_parse_total_count,
            coverage_replay_count=coverage_replay_count,
            c0_baseline_sha256=c0_sha256,
            bank_evidence_path=bank_path,
            bank_evidence_sha256=None if bank_evidence is None else bank_evidence.evidence_sha256,
            bank_measured_elapsed_seconds=None
            if bank_evidence is None
            else bank_evidence.measured_elapsed_seconds,
            bank_physical_duration_seconds=None
            if bank_evidence is None
            else bank_evidence.physical_duration_seconds,
            bank_ready_elapsed_seconds=None
            if model_prep is None
            else model_prep.bank_ready_elapsed_seconds,
            treatment_failure_path=cell.treatment_failure_path,
            treatment_failure_sha256=cell.treatment_failure_sha256,
            exogenous_evidence_path=cell.exogenous_evidence_path,
            exogenous_evidence_sha256=cell.exogenous_evidence_sha256,
            model_prep_result_path=None
            if model_prep is None
            else _model_prep_evidence_path(request),
            model_prep_result_sha256=None
            if model_prep is None
            else sha256_bytes(_model_prep_evidence_path(request).read_bytes()),
            model_snapshot_count=None if model_prep is None else model_prep.written_snapshot_count,
            model_snapshot_paths=tuple(item.path for item in snapshot_backing),
            model_snapshot_sha256=tuple(item.sha256 for item in snapshot_backing),
            model_snapshot_size_bytes=tuple(item.size_bytes for item in snapshot_backing),
            model_nfe=None if model_prep is None else model_prep.nfe,
            model_forward_calls=None if model_prep is None else model_prep.forward_calls,
            model_gpu_ms=None if model_prep is None else model_prep.gpu_ms,
            model_ready_elapsed_seconds=None
            if model_prep is None
            else model_prep.model_ready_elapsed_seconds,
            corpus_release_elapsed_seconds=None
            if release_evidence is None
            else release_evidence.corpus_release_elapsed_seconds,
            pre_release_corpus_set_sha256=None
            if model_prep is None
            else model_prep.pre_release_corpus_set_sha256,
            producer_gpu_index=None if model_prep is None else model_prep.producer_gpu_index,
            corpus_release_path=None if release_evidence is None else release_path,
            corpus_release_sha256=None
            if release_evidence is None
            else sha256_bytes(release_path.read_bytes()),
            proof_replay_count=proof_replay_count,
            proof_wall_seconds=proof_wall_seconds,
            proof_cpu_seconds=proof_cpu_seconds,
            terminal_reason=cell.terminal_reason,
            pair_status=cast(
                "Literal['complete', 'exogenous-invalidated']", pair_by_wave[cell.wave_index]
            ),
            search_cpu_seconds=cell.search_cpu_seconds,
            fuzzer_seed=cell.fuzzer_seed,
            manifest_sha256=cell.manifest_sha256,
            resource_map_sha256=cell.resource_map_sha256,
            previous_row_sha256=previous_row_sha256,
            row_sha256="0" * 64,
        )
        row_sha256 = sha256_bytes(
            canonical_json_bytes(unsigned.model_dump(mode="json", exclude={"row_sha256"}))
        )
        payload = unsigned.model_copy(update={"row_sha256": row_sha256})
        previous_row_sha256 = row_sha256
        lines.append(
            json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        )
    root.mkdir(parents=True, exist_ok=True)
    content = ("\n".join(lines) + "\n").encode()
    if reports.exists():
        existing = reports.read_bytes()
        if existing != content:
            raise CoordinatorError(reason="raw arm report mismatch")
        _validate_raw_report_chain(existing)
        return
    _write_once_bytes(reports, content, label="raw arm report")
    _validate_raw_report_chain(content)


def _validate_raw_report_chain(content: bytes) -> None:
    previous = "0" * 64
    try:
        for line in content.splitlines():
            row = RawArmReportRow.model_validate_json(line)
            _validate_raw_report_row(row)
            if row.previous_row_sha256 != previous:
                raise CoordinatorError(reason="raw arm report chain mismatch")
            unsigned = row.model_dump(mode="json", exclude={"row_sha256"})
            if sha256_bytes(canonical_json_bytes(unsigned)) != row.row_sha256:
                raise CoordinatorError(reason="raw arm report row hash mismatch")
            previous = row.row_sha256
    except ValidationError as error:
        raise CoordinatorError(reason="raw arm report invalid") from error


def _validate_complete_event_contract(
    events: tuple[_ExistingEvent, ...],
    schedule: CampaignSchedule,
    manifest: CampaignManifest,
) -> None:
    scheduled = tuple(_cell_id(cell) for cell in schedule.cells)
    expected = set(scheduled)
    launches = tuple(event.cell_id for event in events if event.phase == "launch")
    terminals = tuple(event.cell_id for event in events if event.phase == "terminal")
    if not set(launches) <= expected or set(terminals) != expected:
        raise CoordinatorError(reason="campaign outcome event cell mismatch")
    if len(launches) != len(set(launches)) or len(terminals) != len(expected):
        raise CoordinatorError(reason="campaign outcome duplicate cell event")
    launches_by_id = {event.cell_id: event for event in events if event.phase == "launch"}
    terminals_by_id = {event.cell_id: event for event in events if event.phase == "terminal"}
    for cell in schedule.cells:
        cell_id = _cell_id(cell)
        if cell_id in launches_by_id:
            continue
        terminal = terminals_by_id[cell_id]
        outcome = _recovered_cell(terminal)
        generated = _cell_request(
            manifest.config,
            manifest,
            cell,
            {int(cell.repetition): outcome.bank_sha256},
            manifest.config.resource_map,
        )
        _ = _request_for_terminal_event(
            config=manifest.config,
            manifest=manifest,
            cell=cell,
            generated=generated,
            launch=None,
            terminal=terminal,
        )


def _validate_complete_raw_report_contract(content: bytes, schedule: CampaignSchedule) -> None:
    scheduled = tuple(_cell_id(cell) for cell in schedule.cells)
    expected = set(scheduled)
    rows = tuple(RawArmReportRow.model_validate_json(line) for line in content.splitlines())
    row_ids = tuple(row.cell_id for row in rows)
    if len(rows) != _COMPLETE_CELL_COUNT or set(row_ids) != expected:
        raise CoordinatorError(reason="campaign outcome raw row cell mismatch")
    if len(set(row_ids)) != len(row_ids):
        raise CoordinatorError(reason="campaign outcome duplicate raw row")
    if row_ids != scheduled:
        raise CoordinatorError(reason="campaign outcome raw row order mismatch")
    for row, cell in zip(rows, schedule.cells, strict=True):
        if (
            row.arm != cell.arm
            or row.repetition != int(cell.repetition)
            or row.wave_index != int(cell.wave_index)
        ):
            raise CoordinatorError(reason="campaign outcome raw row schedule mismatch")


def _validate_raw_report_row(row: RawArmReportRow) -> None:  # noqa: PLR0912, PLR0915
    _require_below(row.campaign_root, row.artifact_root)
    if row.coverage_status == "available":
        _validate_coverage_envelope(row)
    elif (
        row.coverage_series_path is not None
        or row.coverage_series_sha256 is not None
        or row.coverage_envelope_sha256 is not None
        or row.coverage_replay_count is not None
    ):
        raise CoordinatorError(reason="raw arm report unavailable artifact mismatch")
    if row.proof_status == "available":
        _validate_proof_envelope(row)
    elif (
        row.proof_result_path is not None
        or row.proof_result_sha256 is not None
        or row.proof_envelope_sha256 is not None
    ):
        raise CoordinatorError(reason="raw arm report unavailable artifact mismatch")
    if row.deep_parse_status == "available":
        _validate_deep_parse_envelope(row)
    elif (
        row.deep_parse_envelope_path is not None
        or row.deep_parse_result_sha256 is not None
        or row.deep_parse_envelope_sha256 is not None
        or row.deep_parse_replay_count is not None
        or row.deep_parse_success_count is not None
        or row.deep_parse_total_count is not None
    ):
        raise CoordinatorError(reason="raw arm report unavailable artifact mismatch")
    if row.arm in MODEL_ARMS:
        if row.bank_evidence_path is None or row.bank_evidence_sha256 is None:
            if row.terminal_reason not in {"expired-frozen", "exogenous-invalidated"}:
                raise CoordinatorError(reason="raw arm report missing bank evidence")
        else:
            _require_below(
                row.campaign_root / "banks" / f"rep-{row.repetition:02d}",
                row.bank_evidence_path,
            )
            bank_evidence = _load_candidate_bank_evidence(row.bank_evidence_path, row.repetition)
            if (
                bank_evidence.evidence_sha256 != row.bank_evidence_sha256
                or bank_evidence.measured_elapsed_seconds != row.bank_measured_elapsed_seconds
                or bank_evidence.physical_duration_seconds != row.bank_physical_duration_seconds
            ):
                raise CoordinatorError(reason="raw arm report bank evidence mismatch")
    elif (
        row.bank_evidence_path is not None
        or row.bank_evidence_sha256 is not None
        or row.bank_measured_elapsed_seconds is not None
        or row.bank_physical_duration_seconds is not None
        or row.bank_ready_elapsed_seconds is not None
    ):
        raise CoordinatorError(reason="raw arm report control bank mismatch")
    if row.model_prep_result_path is not None or row.model_prep_result_sha256 is not None:
        _validate_model_prep_row(row)
    elif (
        row.bank_ready_elapsed_seconds is not None
        or row.model_snapshot_count is not None
        or row.model_snapshot_paths
        or row.model_snapshot_sha256
        or row.model_snapshot_size_bytes
        or row.model_nfe is not None
        or row.model_forward_calls is not None
        or row.model_gpu_ms is not None
        or row.model_ready_elapsed_seconds is not None
        or row.pre_release_corpus_set_sha256 is not None
        or row.producer_gpu_index is not None
    ):
        raise CoordinatorError(reason="raw arm report missing model-prep evidence")
    if row.corpus_release_path is not None or row.corpus_release_sha256 is not None:
        if row.corpus_release_path is None or row.corpus_release_sha256 is None:
            raise CoordinatorError(reason="raw arm report corpus release mismatch")
        _require_below(row.artifact_root, row.corpus_release_path)
        release = _load_corpus_release(row.corpus_release_path, row.cell_id)
        if (
            sha256_bytes(row.corpus_release_path.read_bytes()) != row.corpus_release_sha256
            or release.corpus_release_elapsed_seconds != row.corpus_release_elapsed_seconds
            or release.pre_release_corpus_set_sha256 != row.pre_release_corpus_set_sha256
            or release.producer_gpu_index != row.producer_gpu_index
        ):
            raise CoordinatorError(reason="raw arm report corpus release mismatch")
    elif row.corpus_release_elapsed_seconds is not None:
        raise CoordinatorError(reason="raw arm report missing corpus release evidence")
    if row.model_snapshot_count is None:
        if row.model_snapshot_paths or row.model_snapshot_sha256 or row.model_snapshot_size_bytes:
            raise CoordinatorError(reason="raw arm report snapshot count mismatch")
    elif row.model_snapshot_count != len(row.model_snapshot_paths):
        raise CoordinatorError(reason="raw arm report snapshot count mismatch")
    if not (
        len(row.model_snapshot_paths)
        == len(row.model_snapshot_sha256)
        == len(row.model_snapshot_size_bytes)
    ):
        raise CoordinatorError(reason="raw arm report snapshot field mismatch")
    for path, digest, size in zip(
        row.model_snapshot_paths,
        row.model_snapshot_sha256,
        row.model_snapshot_size_bytes,
        strict=True,
    ):
        staging_root = row.artifact_root / "model-prep-staging" / "corpus"
        snapshot_root = (
            staging_root
            if path.resolve().is_relative_to(staging_root.resolve())
            else row.artifact_root / "corpus"
        )
        _require_below(snapshot_root, path)
        _validate_regular_artifact(path, digest, size)
    if row.terminal_reason == "treatment-failed":
        if row.treatment_failure_path is None or row.treatment_failure_sha256 is None:
            raise CoordinatorError(reason="raw arm report missing treatment failure evidence")
        _require_below(row.artifact_root, row.treatment_failure_path)
        _ = _load_treatment_failure_evidence(
            row.treatment_failure_path,
            row.treatment_failure_sha256,
            row.cell_id,
        )
    elif row.treatment_failure_path is not None or row.treatment_failure_sha256 is not None:
        raise CoordinatorError(reason="raw arm report treatment failure mismatch")
    if row.exogenous_evidence_path is not None or row.exogenous_evidence_sha256 is not None:
        if row.exogenous_evidence_path is None or row.exogenous_evidence_sha256 is None:
            raise CoordinatorError(reason="raw arm report missing exogenous evidence")
        _require_below(row.campaign_root / "exogenous", row.exogenous_evidence_path)
        evidence = _load_exogenous_evidence(
            row.exogenous_evidence_path, row.exogenous_evidence_sha256
        )
        if (
            row.cell_id not in evidence.affected_cell_ids
            or row.wave_index != evidence.affected_wave_index
            or row.repetition not in evidence.affected_repetitions
            or row.pair_status != "exogenous-invalidated"
        ):
            raise CoordinatorError(reason="raw arm report exogenous evidence mismatch")


def _validate_coverage_envelope(row: RawArmReportRow) -> None:
    if (
        row.coverage_series_path is None
        or row.coverage_series_sha256 is None
        or row.coverage_envelope_sha256 is None
    ):
        raise CoordinatorError(reason="raw arm report missing artifact")
    _require_below(row.artifact_root / "post-search", row.coverage_series_path)
    envelope = _load_typed_raw_artifact(row.coverage_series_path, CoverageSeriesEnvelope)
    if sha256_bytes(row.coverage_series_path.read_bytes()) != row.coverage_envelope_sha256:
        raise CoordinatorError(reason="raw arm report artifact hash mismatch")
    if envelope.coverage_series_sha256 != row.coverage_series_sha256:
        raise CoordinatorError(reason="raw arm report coverage result mismatch")
    if envelope.coverage_series_sha256 != _model_sha256(envelope.coverage_series):
        raise CoordinatorError(reason="raw arm report coverage result mismatch")
    if envelope.coverage_request_sha256 != _model_sha256(envelope.coverage_request):
        raise CoordinatorError(reason="raw arm report coverage request mismatch")
    if row.coverage_replay_count != len(envelope.coverage_series.points):
        raise CoordinatorError(reason="raw arm report coverage replay mismatch")
    if envelope.cell_request_sha256 != row.cell_request_sha256:
        raise CoordinatorError(reason="raw arm report coverage binding mismatch")
    if row.terminal_reason == "completed":
        if envelope.cell_result_sha256 != row.cell_result_sha256:
            raise CoordinatorError(reason="raw arm report coverage binding mismatch")
        _validate_completed_task4_result(row, envelope.cell_request, envelope.cell_result)
    elif row.terminal_reason == "treatment-failed":
        if envelope.cell_result_sha256 != sha256_bytes(b"treatment-failure"):
            raise CoordinatorError(reason="raw arm report coverage binding mismatch")
        _validate_frozen_treatment_coverage(envelope)


def _validate_proof_envelope(row: RawArmReportRow) -> None:
    if (
        row.proof_result_path is None
        or row.proof_result_sha256 is None
        or row.proof_envelope_sha256 is None
    ):
        raise CoordinatorError(reason="raw arm report missing artifact")
    _require_below(row.artifact_root / "post-search", row.proof_result_path)
    envelope = _load_typed_raw_artifact(row.proof_result_path, ProofBatchEnvelope)
    if sha256_bytes(row.proof_result_path.read_bytes()) != row.proof_envelope_sha256:
        raise CoordinatorError(reason="raw arm report artifact hash mismatch")
    if envelope.proof_result_sha256 != row.proof_result_sha256:
        raise CoordinatorError(reason="raw arm report proof result mismatch")
    if envelope.proof_result_sha256 != _model_sha256(envelope.proof_result):
        raise CoordinatorError(reason="raw arm report proof result mismatch")
    if envelope.proof_request_sha256 != _model_sha256(envelope.proof_request):
        raise CoordinatorError(reason="raw arm report proof request mismatch")
    accounting = envelope.proof_result.proof_accounting
    if (
        row.proof_replay_count != accounting.replay_count
        or row.proof_wall_seconds != accounting.wall_seconds
        or row.proof_cpu_seconds != accounting.cpu_seconds
    ):
        raise CoordinatorError(reason="raw arm report proof accounting mismatch")
    if (
        envelope.cell_request_sha256 != row.cell_request_sha256
        or envelope.cell_result_sha256 != row.cell_result_sha256
    ):
        raise CoordinatorError(reason="raw arm report proof binding mismatch")
    _validate_completed_task4_result(row, envelope.cell_request, envelope.cell_result)


def _validate_deep_parse_envelope(row: RawArmReportRow) -> None:  # noqa: PLR0912
    if (
        row.deep_parse_envelope_path is None
        or row.deep_parse_result_sha256 is None
        or row.deep_parse_envelope_sha256 is None
    ):
        raise CoordinatorError(reason="raw arm report missing artifact")
    _require_below(row.artifact_root / "post-search", row.deep_parse_envelope_path)
    envelope = _load_typed_raw_artifact(row.deep_parse_envelope_path, DeepParseEnvelope)
    if sha256_bytes(row.deep_parse_envelope_path.read_bytes()) != row.deep_parse_envelope_sha256:
        raise CoordinatorError(reason="raw arm report artifact hash mismatch")
    if envelope.deep_parse_result_sha256 != row.deep_parse_result_sha256:
        raise CoordinatorError(reason="raw arm report deep-parse result mismatch")
    if envelope.deep_parse_result_sha256 != _model_sha256(envelope.deep_parse_result):
        raise CoordinatorError(reason="raw arm report deep-parse result mismatch")
    if envelope.deep_parse_request_sha256 != _model_sha256(envelope.deep_parse_request):
        raise CoordinatorError(reason="raw arm report deep-parse request mismatch")
    if envelope.cell_request_sha256 != row.cell_request_sha256:
        raise CoordinatorError(reason="raw arm report deep-parse binding mismatch")
    if row.terminal_reason == "completed":
        if envelope.cell_result_sha256 != row.cell_result_sha256:
            raise CoordinatorError(reason="raw arm report deep-parse binding mismatch")
        _validate_completed_task4_result(row, envelope.cell_request, envelope.cell_result)
    elif row.terminal_reason == "treatment-failed":
        if envelope.cell_result_sha256 != envelope.deep_parse_request.cell_result_sha256:
            raise CoordinatorError(reason="raw arm report deep-parse binding mismatch")
    result = envelope.deep_parse_result
    if (
        row.deep_parse_replay_count != result.total_count
        or row.deep_parse_success_count != result.success_count
        or row.deep_parse_total_count != result.total_count
    ):
        raise CoordinatorError(reason="raw arm report deep-parse accounting mismatch")
    for verdict, path, digest in zip(
        result.verdicts,
        envelope.deep_parse_request.input_paths,
        envelope.deep_parse_request.input_sha256,
        strict=True,
    ):
        if verdict.input_path != path or verdict.input_sha256 != digest:
            raise CoordinatorError(reason="raw arm report deep-parse input mismatch")
        _validate_regular_artifact(path, digest, verdict.size_bytes)


def _validate_frozen_treatment_coverage(envelope: CoverageSeriesEnvelope) -> None:
    rows = envelope.coverage_series.corpus_index
    points = envelope.coverage_series.points
    if (
        tuple(row.scheduled_elapsed_seconds for row in rows) != SCHEDULE_SECONDS
        or tuple(point.scheduled_elapsed_seconds for point in points) != SCHEDULE_SECONDS
    ):
        raise CoordinatorError(reason="raw arm report treatment coverage schedule mismatch")
    if rows[0].observation_kind != "observed" or any(
        row.observation_kind != "frozen" for row in rows[1:]
    ):
        raise CoordinatorError(reason="raw arm report treatment coverage freeze mismatch")
    if any(
        point.observation_kind != row.observation_kind
        for point, row in zip(points, rows, strict=True)
    ):
        raise CoordinatorError(reason="raw arm report treatment coverage point mismatch")


def _validate_completed_task4_result(
    row: RawArmReportRow, request: CellRunRequest, result: CellResult
) -> None:
    if row.terminal_reason != "completed":
        return
    try:
        persisted = validate_cell_evidence(request)
    except CellEvidenceError as error:
        raise CoordinatorError(reason="completed cell evidence invalid") from error
    if (
        persisted != result
        or _model_sha256(request) != row.cell_request_sha256
        or _model_sha256(persisted) != row.cell_result_sha256
        or persisted.search_accounting.cpu_seconds != row.search_cpu_seconds
    ):
        raise CoordinatorError(reason="raw arm report completed cell evidence mismatch")


def _validate_model_prep_row(row: RawArmReportRow) -> None:
    if row.model_prep_result_path is None or row.model_prep_result_sha256 is None:
        raise CoordinatorError(reason="raw arm report model-prep evidence mismatch")
    _require_below(row.artifact_root, row.model_prep_result_path)
    evidence = _load_model_prep_evidence(row.model_prep_result_path, row.cell_id)
    if sha256_bytes(row.model_prep_result_path.read_bytes()) != row.model_prep_result_sha256:
        raise CoordinatorError(reason="raw arm report model-prep hash mismatch")
    snapshots = tuple(evidence.snapshots)
    if (
        row.model_snapshot_count != evidence.snapshot_count
        or row.model_snapshot_paths != tuple(snapshot.path for snapshot in snapshots)
        or row.model_snapshot_sha256 != tuple(snapshot.sha256 for snapshot in snapshots)
        or row.model_snapshot_size_bytes != tuple(snapshot.size_bytes for snapshot in snapshots)
        or row.model_nfe != evidence.nfe
        or row.model_forward_calls != evidence.forward_calls
        or row.model_gpu_ms != evidence.gpu_ms
        or row.bank_ready_elapsed_seconds != evidence.bank_ready_elapsed_seconds
        or row.bank_physical_duration_seconds != evidence.bank_physical_duration_seconds
        or row.model_ready_elapsed_seconds != evidence.model_ready_elapsed_seconds
        or row.pre_release_corpus_set_sha256 != evidence.pre_release_corpus_set_sha256
        or row.producer_gpu_index != evidence.producer_gpu_index
    ):
        raise CoordinatorError(reason="raw arm report model-prep mismatch")


def _load_typed_raw_artifact[ModelT: FrozenModel](path: Path, model_type: type[ModelT]) -> ModelT:
    _reject_symlink_path(path)
    if path.is_symlink() or not path.is_file():
        raise CoordinatorError(reason="raw arm report artifact unreadable")
    try:
        raw = path.read_bytes()
        model = model_type.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="raw arm report artifact invalid") from error
    if raw != canonical_json_bytes(model.model_dump(mode="json")):
        raise CoordinatorError(reason="raw arm report artifact noncanonical")
    return model


def _validate_regular_artifact(path: Path, digest: Sha256, size_bytes: int) -> None:
    _reject_symlink_path(path)
    if path.is_symlink() or not path.is_file():
        raise CoordinatorError(reason="raw arm report artifact unreadable")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CoordinatorError(reason="raw arm report artifact unreadable") from error
    if len(payload) != size_bytes or sha256_bytes(payload) != digest:
        raise CoordinatorError(reason="raw arm report artifact hash mismatch")


def _load_treatment_failure_evidence(
    path: Path, digest: Sha256, cell_id: str
) -> TreatmentFailureEvidence:
    evidence = _load_typed_raw_artifact(path, TreatmentFailureEvidence)
    if _model_sha256(evidence) != digest or evidence.cell_id != cell_id:
        raise CoordinatorError(reason="raw arm report treatment failure mismatch")
    return evidence


def _load_exogenous_evidence(path: Path, digest: Sha256) -> ExogenousFailureEvidence:
    evidence = _load_typed_raw_artifact(path, ExogenousFailureEvidence)
    if _model_sha256(evidence) != digest:
        raise CoordinatorError(reason="raw arm report exogenous evidence mismatch")
    if evidence.raw_evidence_sha256 != sha256_bytes(evidence.raw_evidence.encode()):
        raise CoordinatorError(reason="raw arm report exogenous evidence mismatch")
    return evidence


def _require_below(root: Path, path: Path) -> None:
    _reject_symlink_path(root)
    _reject_symlink_path(path)
    try:
        _ = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise CoordinatorError(reason="raw arm report artifact outside root") from error


def _reject_symlink_path(path: Path) -> None:
    candidates = (*tuple(reversed(path.parents)), path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_symlink():
            raise CoordinatorError(reason="raw arm report artifact unsafe symlink")


def _raw_report_post_search_complete(content: bytes) -> bool:
    try:
        for line in content.splitlines():
            row = RawArmReportRow.model_validate_json(line)
            if row.terminal_reason == "completed" and (
                row.coverage_status != "available"
                or row.proof_status != "available"
                or row.deep_parse_status != "available"
            ):
                return False
            if row.terminal_reason == "treatment-failed" and (
                row.coverage_status != "available" or row.proof_status != "unavailable"
            ):
                return False
    except ValidationError as error:
        raise CoordinatorError(reason="raw arm report invalid") from error
    return bool(content.splitlines())


def _write_once_canonical_model(path: Path, model: FrozenModel) -> None:
    payload = canonical_json_bytes(model.model_dump(mode="json"))
    _write_once_bytes(path, payload, label=path.name)


def _write_once_bytes(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CoordinatorError(reason=f"artifact mismatch: {label}")
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise CoordinatorError(reason=f"artifact mismatch: {label}") from None
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_canonical_model(path: Path, model: FrozenModel) -> None:
    if path.read_bytes() != canonical_json_bytes(model.model_dump(mode="json")):
        raise CoordinatorError(reason=f"noncanonical artifact: {path.name}")


def _model_sha256(model: FrozenModel) -> Sha256:
    return sha256_bytes(canonical_json_bytes(model.model_dump(mode="json")))


@dataclass(frozen=True, slots=True)
class _UnavailablePostSearch:
    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        del request, result
        raise CoordinatorError(reason="completed post-search evidence invalid")

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        del request
        raise CoordinatorError(reason="treatment coverage evidence invalid")


@dataclass(frozen=True, slots=True)
class _ExistingEvent:
    phase: Literal["manifest", "launch", "terminal"]
    cell_id: str
    payload: dict[str, JsonValue]


def _append_campaign_event(
    root: Path, phase: Literal["manifest", "launch", "terminal"], payload: dict[str, JsonValue]
) -> None:
    event_type = "campaign_manifest" if phase == "manifest" else "cell_checkpoint"
    _ = append_event(
        root / _EVENTS_NAME,
        timestamp_utc=phase,
        event_type=event_type,
        payload={"phase": phase, **payload},
    )


def _manifest_payload(manifest: CampaignManifest) -> dict[str, JsonValue]:
    return {
        "manifest_sha256": manifest.manifest_sha256 or "",
        "config_sha256": manifest.config.config_sha256,
        "resource_map_sha256": manifest.resource_map_sha256,
    }


def _cell_payload(
    config: CampaignConfig,
    manifest: CampaignManifest,
    cell: CellSchedule,
    request: CellRunRequest,
    banks: dict[int, CandidateBank],
) -> dict[str, JsonValue]:
    evidence = _launch_request_evidence(config, cell, request, banks)
    return {
        "cell_id": request.cell_id,
        "fuzzer_seed": int(cell.fuzzer_seed),
        "deadline_monotonic_seconds": float(cell.deadline_monotonic_seconds),
        "artifact_root": str(request.artifact_root),
        "config_sha256": config.config_sha256,
        "resource_map_sha256": manifest.resource_map_sha256,
        "manifest_sha256": manifest.manifest_sha256 or "",
        "mutable_root": str(_mutable_root(config, cell)),
        "bank_sha256": request.bank_sha256,
        "request": evidence.request.model_dump(mode="json"),
        "request_sha256": evidence.request_sha256,
        "build_sha256": evidence.build_sha256,
        "corpus_seed_manifest_sha256": evidence.corpus_seed_manifest_sha256,
    }


def _outcome_payload(outcome: CellOutcome) -> dict[str, JsonValue]:
    return outcome.model_dump(mode="json")


def _launch_request_evidence(
    config: CampaignConfig,
    cell: CellSchedule,
    request: CellRunRequest,
    banks: dict[int, CandidateBank],
) -> LaunchRequestEvidence:
    del banks
    seed_path = request.artifact_root / "seed-manifest.json"
    try:
        seed_bytes = seed_path.read_bytes()
        _ = CorpusSeedSummary.model_validate_json(seed_bytes)
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="launch corpus seed evidence missing") from error
    return LaunchRequestEvidence(
        request=request,
        request_sha256=sha256_bytes(canonical_json_bytes(request.model_dump(mode="json"))),
        mutable_root=_mutable_root(config, cell),
        bank_sha256=request.bank_sha256,
        build_sha256=sha256_bytes(canonical_json_bytes(request.build.model_dump(mode="json"))),
        corpus_seed_manifest_sha256=sha256_bytes(seed_bytes),
    )


def _request_from_launch(event: _ExistingEvent, generated: CellRunRequest) -> CellRunRequest:
    request = _request_from_launch_only(event)
    if request != generated:
        raise CoordinatorError(reason="launch request mismatch")
    mutable_root = event.payload.get("mutable_root")
    if not isinstance(mutable_root, str) or Path(
        mutable_root
    ) != _expected_mutable_root_from_request(generated):
        raise CoordinatorError(reason="launch mutable root mismatch")
    return request


def _request_from_launch_only(event: _ExistingEvent) -> CellRunRequest:
    raw_request = event.payload.get("request")
    request_sha256 = event.payload.get("request_sha256")
    if not isinstance(raw_request, dict) or not isinstance(request_sha256, str):
        raise CoordinatorError(reason="launch request evidence missing")
    try:
        request = CellRunRequest.model_validate_json(canonical_json_bytes(raw_request))
    except ValidationError as error:
        raise CoordinatorError(reason="invalid launch request evidence") from error
    actual = sha256_bytes(canonical_json_bytes(request.model_dump(mode="json")))
    if actual != request_sha256:
        raise CoordinatorError(reason="launch request mismatch")
    build_sha256 = event.payload.get("build_sha256")
    if not isinstance(build_sha256, str) or build_sha256 != sha256_bytes(
        canonical_json_bytes(request.build.model_dump(mode="json"))
    ):
        raise CoordinatorError(reason="launch build evidence mismatch")
    _validate_launched_seed_evidence(event, request)
    return request


def _expected_mutable_root_from_request(request: CellRunRequest) -> Path:
    rep_root = request.artifact_root.parent
    wave_root = rep_root.parent
    cells_root = wave_root.parent
    campaign_root = cells_root.parent
    return campaign_root / "mutable" / rep_root.name / request.artifact_root.name


def _validate_launched_seed_evidence(event: _ExistingEvent, request: CellRunRequest) -> None:
    seed_sha256 = event.payload.get("corpus_seed_manifest_sha256")
    seed_path = request.artifact_root / "seed-manifest.json"
    try:
        seed_bytes = seed_path.read_bytes()
        summary = CorpusSeedSummary.model_validate_json(seed_bytes)
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="launch corpus seed evidence missing") from error
    if not isinstance(seed_sha256, str) or seed_sha256 != sha256_bytes(seed_bytes):
        raise CoordinatorError(reason="launch corpus seed evidence mismatch")
    expected_hashes = {summary.c0_sha256, *summary.bank_parent_sha256}
    actual_hashes: set[str] = set()
    try:
        paths = tuple(path for path in request.corpus_root.iterdir() if path.is_file())
    except OSError as error:
        raise CoordinatorError(reason="launch corpus evidence missing") from error
    for path in paths:
        _reject_symlink_path(path)
        actual_hashes.add(sha256_bytes(path.read_bytes()))
    if not expected_hashes <= actual_hashes:
        raise CoordinatorError(reason="launch corpus evidence mismatch")


def _existing_events(root: Path) -> tuple[_ExistingEvent, ...]:
    if not (root / _EVENTS_NAME).exists():
        return ()
    try:
        _ = validate_ledger(root / _EVENTS_NAME)
        events: list[_ExistingEvent] = []
        for line in (root / _EVENTS_NAME).read_bytes().splitlines():
            outer = _JSON_OBJECT.validate_json(line)
            payload = _JSON_OBJECT.validate_python(outer["payload"])
            phase = _event_phase(payload.get("phase"))
            cell_id = payload.get("cell_id", "")
            if not isinstance(cell_id, str):
                raise CoordinatorError(reason="invalid campaign event")
            events.append(_ExistingEvent(phase, cell_id, payload))
    except (LedgerError, KeyError, ValidationError) as error:
        raise CoordinatorError(reason=f"invalid campaign event: {error}") from error
    return tuple(events)


def _cell_id(cell: CellSchedule) -> str:
    return f"wave-{int(cell.wave_index):02d}-rep-{int(cell.repetition):02d}-{cell.arm}"


def _event_phase(value: JsonValue | None) -> Literal["manifest", "launch", "terminal"]:
    if value == "manifest":
        return "manifest"
    if value == "launch":
        return "launch"
    if value == "terminal":
        return "terminal"
    raise CoordinatorError(reason="invalid campaign event")


def _bank_hash(config: CampaignConfig, cell: CellSchedule, banks: dict[int, str]) -> str:
    if cell.arm == "libfuzzer-only":
        return config.c0_sha256
    return banks[int(cell.repetition)]


def _trajectory_hash(config: CampaignConfig, cell: CellSchedule, bank_hash: str) -> str:
    if cell.arm == "libfuzzer-only":
        return "0" * 64
    return sha256_bytes(f"{config.config_sha256}:{int(cell.repetition)}:{bank_hash}".encode())


def _mutable_root(config: CampaignConfig, cell: CellSchedule) -> Path:
    return config.artifact_root / "mutable" / f"rep-{int(cell.repetition):02d}" / cell.arm


def _search_core_index(cell: CellSchedule) -> int:
    return (int(cell.repetition) % 2) * 5 + _arm_offset(cell.arm)


def _arm_offset(arm: Arm) -> int:
    match arm:
        case "selective-revision":
            return 0
        case "final-only":
            return 1
        case "full-regenerate":
            return 2
        case "structured-one-shot":
            return 3
        case "libfuzzer-only":
            return 4
        case unreachable:
            assert_never(unreachable)


def _wave_start(config: CampaignConfig, cell: CellSchedule) -> float:
    return float(config.main_start_monotonic_seconds) + (int(cell.wave_index) * 10_800.0)


def _schedule_json(schedule: CampaignSchedule) -> ScheduleJson:
    preregistration: ScheduleJson = {
        "target_commit": schedule.preregistration.target_commit,
        "arms": schedule.preregistration.arms,
        "repetitions": schedule.preregistration.repetitions,
        "wave_count": schedule.preregistration.wave_count,
        "repetitions_per_wave": schedule.preregistration.repetitions_per_wave,
        "wave_duration_seconds": schedule.preregistration.wave_duration_seconds,
        "confirmatory_treatment": schedule.preregistration.confirmatory_treatment,
        "confirmatory_control": schedule.preregistration.confirmatory_control,
        "primary_endpoint": schedule.preregistration.primary_endpoint,
    }
    return {
        "preregistration": preregistration,
        "cells": [
            {
                "repetition": int(cell.repetition),
                "arm": cell.arm,
                "wave_index": int(cell.wave_index),
                "fuzzer_seed": int(cell.fuzzer_seed),
                "gpu_index": None if cell.gpu_index is None else int(cell.gpu_index),
                "deadline": float(cell.deadline_monotonic_seconds),
            }
            for cell in schedule.cells
        ],
    }


def _hash_json(value: ScheduleJson) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_private_data_root(path: Path) -> None:
    root = path.resolve()
    base = Path("/data/dllm-agent")
    try:
        _ = root.relative_to(base)
    except ValueError as error:
        raise CampaignValidationError(
            "artifact root must be an absolute private descendant of /data/dllm-agent"
        ) from error
    if root == base:
        raise CampaignValidationError(
            "artifact root must be a private descendant of /data/dllm-agent"
        )


def _validate_dictionary_file(path: Path, expected_sha256: Sha256) -> None:
    if ".." in path.parts:
        raise CampaignValidationError("dictionary path must not escape its base directory")
    if path.is_symlink() or not path.is_file():
        raise CampaignValidationError("dictionary file is missing or unsafe")
    if sha256_bytes(path.read_bytes()) != expected_sha256:
        raise CampaignValidationError("dictionary hash mismatch")
