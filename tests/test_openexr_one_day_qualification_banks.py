"""Qualification candidate-bank generation tests."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Condition
from typing import TYPE_CHECKING, Never, Self

import pytest
from tools.selective_revision_payloads import sha256_bytes, sha256_token_ids

from dllm_agent.experiments.openexr_one_day.coordinator import CampaignConfig, CoordinatorError
from dllm_agent.experiments.openexr_one_day.qualification_banks import (
    QUALIFICATION_BANK_SUMMARY_NAME,
    generate_qualification_candidate_banks,
)
from dllm_agent.revision.controller_models import RevisionResult
from tests.test_openexr_one_day_coordinator import _config_json

if TYPE_CHECKING:
    from types import TracebackType

    from dllm_agent.experiments.openexr_one_day.models import CellSchedule
    from dllm_agent.revision.controller_models import InitialParticleRequest, RevisionRequest


def test_qualification_banks_write_only_isolated_candidate_bank_tree(tmp_path: Path) -> None:
    """Given an entered pool, when banks generate, then only qualification banks are written."""
    # Given
    config = _config(tmp_path)
    pool = _FakeQualificationPool()

    # When
    with pool as entered:
        summary = generate_qualification_candidate_banks(config, entered)

    # Then
    assert len(summary.banks) == 10
    assert summary.generated_bank_count == 10
    assert summary.reused_bank_count == 0
    assert not (config.artifact_root / "banks").exists()
    assert (
        config.artifact_root / "qualification" / "banks" / QUALIFICATION_BANK_SUMMARY_NAME
    ).is_file()
    for record in summary.banks:
        expected = (
            config.artifact_root
            / "qualification"
            / "banks"
            / f"rep-{record.repetition:02d}"
            / "candidate-bank.json"
        )
        assert record.path == expected
        assert record.primary_raw_count == 32
        assert record.supplement_raw_count == 0
        assert record.selected_parent_count == 16
        assert record.raw_parent_count == 32
        assert record.physical_cost.initial_requests == 32
        assert sha256(record.path.read_bytes()).hexdigest() == record.artifact_sha256


def test_qualification_banks_use_wave_pair_workers_concurrently_without_reuse(
    tmp_path: Path,
) -> None:
    """Given blocking workers, when banks generate, then each wave overlaps distinct workers."""
    # Given
    config = _config(tmp_path)
    gate = _WavePairGate()
    pool = _FakeQualificationPool(gate=gate)

    # When
    with pool as entered:
        summary = generate_qualification_candidate_banks(config, entered)

    # Then
    assert summary.generated_bank_count == 10
    assert gate.completed_pairs == 5
    assert gate.concurrent_worker_conflicts == []
    assert len(gate.released_pairs) == 5
    assert all(first != second for first, second in gate.released_pairs)


def test_qualification_banks_resume_reuses_existing_bytes_without_model_calls(
    tmp_path: Path,
) -> None:
    """Given complete banks, when generation resumes, then no worker initials are called."""
    # Given
    config = _config(tmp_path)
    first_pool = _FakeQualificationPool()
    with first_pool as entered:
        first = generate_qualification_candidate_banks(config, entered)
    original_bytes = tuple(record.path.read_bytes() for record in first.banks)
    resume_pool = _FakeQualificationPool()

    # When
    with resume_pool as entered:
        resumed = generate_qualification_candidate_banks(config, entered)

    # Then
    assert resumed.generated_bank_count == 0
    assert resumed.reused_bank_count == 10
    assert tuple(record.path.read_bytes() for record in resumed.banks) == original_bytes
    assert sum(len(worker.initial_requests) for worker in resume_pool.workers.values()) == 0
    assert tuple(record.artifact_sha256 for record in resumed.banks) == tuple(
        record.artifact_sha256 for record in first.banks
    )


def test_qualification_banks_reject_tampered_existing_bank_without_model_calls(
    tmp_path: Path,
) -> None:
    """Given one tampered bank, when generation resumes, then validation fails closed."""
    # Given
    config = _config(tmp_path)
    first_pool = _FakeQualificationPool()
    with first_pool as entered:
        summary = generate_qualification_candidate_banks(config, entered)
    target = summary.banks[3].path
    _ = target.write_bytes(target.read_bytes().replace(b'"repetition":3', b'"repetition":4', 1))
    resume_pool = _FakeQualificationPool()

    # When / Then
    with resume_pool as entered, pytest.raises(CoordinatorError, match="candidate-bank"):
        _ = generate_qualification_candidate_banks(config, entered)
    assert sum(len(worker.initial_requests) for worker in resume_pool.workers.values()) == 0


def test_qualification_banks_run_single_supplement_when_primary_uniqueness_is_low(
    tmp_path: Path,
) -> None:
    """Given duplicated primaries, when banks generate, then each bank takes one supplement."""
    # Given
    config = _config(tmp_path)
    pool = _FakeQualificationPool(primary_unique=8, supplement_new_unique=8)

    # When
    with pool as entered:
        summary = generate_qualification_candidate_banks(config, entered)

    # Then
    assert {record.supplement_raw_count for record in summary.banks} == {16}
    assert {record.raw_parent_count for record in summary.banks} == {48}
    assert {record.selected_parent_count for record in summary.banks} == {16}
    assert summary.total_physical_cost.initial_requests == 480


@dataclass(slots=True)
class _WavePairGate:
    condition: Condition = field(default_factory=Condition)
    waiting_workers: set[int] = field(default_factory=set)
    released_pairs: list[tuple[int, int]] = field(default_factory=list)
    concurrent_worker_conflicts: list[int] = field(default_factory=list)
    completed_pairs: int = 0

    def arrive(self, worker_id: int) -> None:
        with self.condition:
            if worker_id in self.waiting_workers:
                self.concurrent_worker_conflicts.append(worker_id)
            self.waiting_workers.add(worker_id)
            if len(self.waiting_workers) == 2:
                ordered = sorted(self.waiting_workers)
                pair = (ordered[0], ordered[1])
                self.released_pairs.append(pair)
                self.waiting_workers.clear()
                self.completed_pairs += 1
                self.condition.notify_all()
                return
            if not self.condition.wait_for(lambda: not self.waiting_workers, timeout=5.0):
                message = "wave-pair bank generation did not overlap"
                raise AssertionError(message)


@dataclass(slots=True)
class _FakeQualificationWorker:
    worker_id: int
    primary_unique: int
    supplement_new_unique: int
    gate: _WavePairGate | None
    entered: bool = False
    initial_requests: list[InitialParticleRequest] = field(default_factory=list)

    @property
    def process_id_if_started(self) -> int:
        return 10_000 + self.worker_id

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        if not self.entered:
            message = "worker must be entered before bank generation"
            raise AssertionError(message)
        if self.gate is not None and request.particle_id.endswith("-00"):
            self.gate.arrive(self.worker_id)
        call_index = int(request.particle_id.rsplit("-", maxsplit=1)[1])
        self.initial_requests.append(request)
        if call_index < 32:
            variant = call_index % self.primary_unique
        else:
            offset = call_index - 32
            variant = self.primary_unique + (offset % self.supplement_new_unique)
        return _result(variant, call_index)

    def revise(self, request: RevisionRequest) -> Never:
        del request
        message = "qualification bank generation must not call revise"
        raise AssertionError(message)


@dataclass(slots=True)
class _FakeQualificationPool:
    primary_unique: int = 32
    supplement_new_unique: int = 0
    gate: _WavePairGate | None = None
    workers: dict[int, _FakeQualificationWorker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workers = {
            index: _FakeQualificationWorker(
                index,
                self.primary_unique,
                self.supplement_new_unique,
                self.gate,
            )
            for index in range(8)
        }

    def __enter__(self) -> Self:
        for worker in self.workers.values():
            worker.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        for worker in self.workers.values():
            worker.entered = False

    def worker_for(self, cell: CellSchedule) -> _FakeQualificationWorker:
        if cell.gpu_index is None:
            message = "qualification banks require model GPU workers"
            raise AssertionError(message)
        return self.workers[int(cell.gpu_index)]

    def bank_worker_for(self, cell: CellSchedule) -> _FakeQualificationWorker:
        return self.worker_for(cell)


def _config(root: Path) -> CampaignConfig:
    data_root = (
        Path("/data/dllm-agent") / f"qualification-banks-{sha256(str(root).encode()).hexdigest()}"
    )
    shutil.rmtree(data_root, ignore_errors=True)
    return CampaignConfig.model_validate(_config_json(str(data_root)))


def _result(variant: int, call_index: int) -> RevisionResult:
    digits = tuple((variant // (4**index)) % 4 for index in range(5))
    selectors = (digits[0], digits[1], digits[2], digits[3], 0, 0, 0, 0, digits[4], 0)
    token_ids = tuple(selector + 10 for selector in selectors)
    return RevisionResult(
        selectors=selectors,
        token_ids=token_ids,
        slot_positions=tuple(range(10)),
        selected_slot_token_ids=token_ids,
        token_state_sha256=sha256_token_ids(token_ids),
        transfer_evidence_sha256=sha256_bytes(f"transfer:{call_index}".encode()),
        rng_state_sha256=sha256_bytes(f"rng:{call_index}".encode()),
        nfe=1,
        forward_calls=1,
        gpu_ms=10,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="width",
        trajectory_evidence_sha256=sha256_bytes(b"[]"),
    )
