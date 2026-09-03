"""Isolated qualification candidate-bank generation."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Protocol, cast

from pydantic import ConfigDict, Field

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import (
    ArtifactPersistenceError,
    write_json_artifact,
)
from dllm_agent.experiments.openexr_one_day.candidate_bank import (
    CandidateBank,
    build_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CoordinatorError,
    ResidentPoolSurface,
    _persist_candidate_bank,
    load_persisted_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS, CellSchedule
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dllm_agent.revision.controller_models import RevisionSampler

QUALIFICATION_BANK_SUMMARY_NAME: Final = "qualification-candidate-banks.json"
_BANK_FILE_NAME: Final = "candidate-bank.json"
_REPETITION_COUNT: Final = 10


class QualificationBankGenerationError(CoordinatorError):
    """Raised when isolated qualification bank generation cannot proceed."""


class _BankWorkerPool(Protocol):
    def bank_worker_for(self, cell: CellSchedule) -> RevisionSampler:
        """Return the worker assigned to produce a repetition bank."""
        ...


class QualificationBankCost(FrozenModel):
    """Model work cost bound into a qualification bank summary."""

    initial_requests: int = Field(ge=0)
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    gpu_ms: int = Field(ge=0)


class QualificationBankRecord(FrozenModel):
    """Stable path, hash, count, and cost binding for one qualification bank."""

    repetition: int = Field(ge=0, le=9)
    path: Path
    artifact_sha256: Sha256
    raw_parent_count: int = Field(ge=32, le=48)
    primary_raw_count: int = Field(ge=32, le=32)
    supplement_raw_count: int = Field(ge=0, le=16)
    selected_parent_count: int = Field(ge=16, le=16)
    selected_bank_sha256: Sha256
    physical_cost: QualificationBankCost
    producer_gpu_index: int = Field(ge=0, le=7)


class QualificationBankManifest(FrozenModel):
    """Canonical persisted summary of the isolated qualification bank set."""

    schema_version: Literal["dllm-openexr-qualification-candidate-banks/v1"] = (
        "dllm-openexr-qualification-candidate-banks/v1"
    )
    config_sha256: Sha256
    bank_root: Path
    banks: tuple[QualificationBankRecord, ...] = Field(min_length=10, max_length=10)
    total_physical_cost: QualificationBankCost


class QualificationBankSummary(FrozenModel):
    """Run result for qualification bank generation or resume."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest: QualificationBankManifest
    summary_artifact_path: Path
    summary_artifact_sha256: Sha256
    generated_bank_count: int = Field(ge=0, le=10)
    reused_bank_count: int = Field(ge=0, le=10)

    @property
    def banks(self) -> tuple[QualificationBankRecord, ...]:
        """Return per-repetition bank records from the persisted manifest."""
        return self.manifest.banks

    @property
    def total_physical_cost(self) -> QualificationBankCost:
        """Return aggregate physical bank cost from the persisted manifest."""
        return self.manifest.total_physical_cost


@dataclass(frozen=True, slots=True)
class _BankBuildInput:
    config: CampaignConfig
    repetition: int
    path: Path
    cell: CellSchedule
    sampler: RevisionSampler


@dataclass(frozen=True, slots=True)
class _BuiltBank:
    bank: CandidateBank
    elapsed_seconds: float


def generate_qualification_candidate_banks(
    config: CampaignConfig,
    resident_pool: ResidentPoolSurface,
) -> QualificationBankSummary:
    """Generate or reload ten isolated qualification candidate banks."""
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    generated: set[int] = set()
    banks: dict[int, CandidateBank] = {}
    bank_root = _qualification_bank_root(config)
    for wave in schedule.waves:
        pending: dict[Future[_BuiltBank], _BankBuildInput] = {}
        wave_inputs = tuple(
            _build_input(config, resident_pool, _model_cell(wave.cells, int(repetition)))
            for repetition in wave.repetitions
        )
        _require_distinct_samplers(wave_inputs)
        with ThreadPoolExecutor(max_workers=len(wave_inputs)) as executor:
            for item in wave_inputs:
                if item.path.exists():
                    banks[item.repetition] = load_persisted_candidate_bank(
                        item.path, item.repetition
                    )
                    continue
                pending[executor.submit(_build_one_bank, item)] = item
            for future in as_completed(pending):
                item = pending[future]
                built = future.result()
                _persist_candidate_bank(
                    item.path,
                    item.repetition,
                    built.bank,
                    built.elapsed_seconds,
                    bank_ready_elapsed_seconds=built.elapsed_seconds,
                    producer_gpu_index=_producer_gpu_index(item.cell),
                )
                banks[item.repetition] = built.bank
                generated.add(item.repetition)
    manifest = _manifest(config, bank_root, _records(config, schedule.cells, banks))
    summary_path, summary_hash = _persist_manifest(config, manifest)
    return QualificationBankSummary(
        manifest=manifest,
        summary_artifact_path=summary_path,
        summary_artifact_sha256=summary_hash,
        generated_bank_count=len(generated),
        reused_bank_count=_REPETITION_COUNT - len(generated),
    )


def _qualification_bank_root(config: CampaignConfig) -> Path:
    return config.artifact_root / "qualification" / "banks"


def _build_input(
    config: CampaignConfig,
    resident_pool: ResidentPoolSurface,
    cell: CellSchedule,
) -> _BankBuildInput:
    repetition = int(cell.repetition)
    return _BankBuildInput(
        config=config,
        repetition=repetition,
        path=_qualification_bank_root(config) / f"rep-{repetition:02d}" / _BANK_FILE_NAME,
        cell=cell,
        sampler=_bank_sampler(resident_pool, cell),
    )


def _bank_sampler(resident_pool: ResidentPoolSurface, cell: CellSchedule) -> RevisionSampler:
    worker_pool = cast("_BankWorkerPool", cast("object", resident_pool))
    bank_worker_for = getattr(worker_pool, "bank_worker_for", None)
    if callable(bank_worker_for):
        return cast("RevisionSampler", bank_worker_for(cell))
    return cast("RevisionSampler", resident_pool.worker_for(cell))


def _require_distinct_samplers(items: Sequence[_BankBuildInput]) -> None:
    sampler_ids = tuple(id(item.sampler) for item in items)
    if len(set(sampler_ids)) != len(sampler_ids):
        raise QualificationBankGenerationError(
            reason="qualification wave assigned one worker to concurrent banks"
        )


def _build_one_bank(item: _BankBuildInput) -> _BuiltBank:
    started = monotonic()
    bank = build_candidate_bank(
        item.sampler,
        item.cell.seed_namespace,
        steps=item.config.candidate_steps,
    )
    return _BuiltBank(bank=bank, elapsed_seconds=max(0.0, monotonic() - started))


def _model_cell(cells: tuple[CellSchedule, ...], repetition: int) -> CellSchedule:
    for cell in cells:
        if int(cell.repetition) == repetition and cell.arm in MODEL_ARMS:
            return cell
    raise QualificationBankGenerationError(reason="qualification bank producer cell missing")


def _producer_gpu_index(cell: CellSchedule) -> int:
    if cell.gpu_index is None:
        raise QualificationBankGenerationError(reason="qualification bank producer has no GPU")
    return int(cell.gpu_index)


def _records(
    config: CampaignConfig,
    cells: tuple[CellSchedule, ...],
    banks: dict[int, CandidateBank],
) -> tuple[QualificationBankRecord, ...]:
    return tuple(
        _record(config, _model_cell(cells, repetition), banks[repetition])
        for repetition in range(10)
    )


def _record(
    config: CampaignConfig,
    cell: CellSchedule,
    bank: CandidateBank,
) -> QualificationBankRecord:
    path = _qualification_bank_root(config) / f"rep-{int(cell.repetition):02d}" / _BANK_FILE_NAME
    return QualificationBankRecord(
        repetition=int(cell.repetition),
        path=path,
        artifact_sha256=sha256_bytes(path.read_bytes()),
        raw_parent_count=len(bank.raw_parents),
        primary_raw_count=bank.primary_raw_count,
        supplement_raw_count=bank.supplement_raw_count,
        selected_parent_count=len(bank.selected_parents),
        selected_bank_sha256=bank.selected_bank_sha256,
        physical_cost=_cost(bank),
        producer_gpu_index=_producer_gpu_index(cell),
    )


def _manifest(
    config: CampaignConfig,
    bank_root: Path,
    records: tuple[QualificationBankRecord, ...],
) -> QualificationBankManifest:
    return QualificationBankManifest(
        config_sha256=config.config_sha256,
        bank_root=bank_root,
        banks=records,
        total_physical_cost=_total_cost(tuple(record.physical_cost for record in records)),
    )


def _cost(bank: CandidateBank) -> QualificationBankCost:
    return QualificationBankCost(
        initial_requests=bank.physical_cost.initial_requests,
        nfe=bank.physical_cost.nfe,
        forward_calls=bank.physical_cost.forward_calls,
        gpu_ms=bank.physical_cost.gpu_ms,
    )


def _total_cost(costs: tuple[QualificationBankCost, ...]) -> QualificationBankCost:
    return QualificationBankCost(
        initial_requests=sum(cost.initial_requests for cost in costs),
        nfe=sum(cost.nfe for cost in costs),
        forward_calls=sum(cost.forward_calls for cost in costs),
        gpu_ms=sum(cost.gpu_ms for cost in costs),
    )


def _persist_manifest(
    config: CampaignConfig,
    manifest: QualificationBankManifest,
) -> tuple[Path, str]:
    try:
        evidence = write_json_artifact(
            experiment_root=config.artifact_root,
            relative_path=Path("qualification") / "banks" / QUALIFICATION_BANK_SUMMARY_NAME,
            value=manifest.model_dump(mode="json"),
        )
    except ArtifactPersistenceError as error:
        raise QualificationBankGenerationError(reason=error.reason) from error
    return config.artifact_root / evidence.path, evidence.sha256
