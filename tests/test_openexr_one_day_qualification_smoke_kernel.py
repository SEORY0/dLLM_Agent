"""Qualification smoke kernel regressions."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path  # noqa: TC003 - pytest fixtures use this concrete type at runtime.
from typing import TYPE_CHECKING, Literal, override

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.models import ARMS
from dllm_agent.experiments.openexr_one_day.qualification_smoke import (
    build_qualification_smoke_plan,
    execute_qualification_smoke,
    load_qualification_smoke_result,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeError,
    SmokeCellResult,
    SmokeExecutionDependencies,
    SmokeModelPreparation,
    SmokePostSearchOutcome,
)

if TYPE_CHECKING:
    from dllm_agent.core.models import Sha256


def test_smoke_plan_is_one_two_repetition_wave_with_common_300_second_deadline(
    tmp_path: Path,
) -> None:
    """Given a root, when planned, then smoke is isolated and does not alter main constants."""
    # Given
    start = 12_500.0

    # When
    plan = build_qualification_smoke_plan(
        config_sha256="1" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=start,
    )

    # Then
    assert plan.wave_count == 1
    assert plan.pair_count == 2
    assert plan.cell_count == 10
    assert {cell.repetition for cell in plan.cells} == {0, 1}
    assert tuple(cell.arm for cell in plan.cells[:5]) == ARMS
    assert {cell.deadline_seconds for cell in plan.cells} == {300}
    assert {cell.deadline_monotonic_seconds for cell in plan.cells} == {start + 300.0}
    assert all(cell.artifact_root.is_relative_to(tmp_path / "smoke") for cell in plan.cells)


def test_execute_smoke_persists_hash_bound_result_and_reload_reuses_cell_evidence(
    tmp_path: Path,
) -> None:
    """Given completed cells, when smoke is invoked twice, then the second run reloads artifacts."""
    # Given
    smoke_root = tmp_path / "smoke"
    plan = build_qualification_smoke_plan(
        config_sha256="2" * 64,
        smoke_root=smoke_root,
        start_monotonic_seconds=40_000.0,
    )
    fixture = _fixture()

    # When
    first = execute_qualification_smoke(plan, fixture.dependencies)
    second = execute_qualification_smoke(plan, fixture.dependencies)

    # Then
    assert first == second
    assert (smoke_root / "qualification-smoke-result.json").is_file()
    assert len(first.cells) == 10
    assert first.passed
    assert {outcome.bank_sha256 for outcome in first.cells} == {
        sha256_bytes(b"bank:0"),
        sha256_bytes(b"bank:1"),
    }
    assert all(
        outcome.model_prep_sha256 is not None
        for outcome in first.cells
        if outcome.arm != "libfuzzer-only"
    )
    assert all(
        outcome.model_prep_sha256 is None
        for outcome in first.cells
        if outcome.arm == "libfuzzer-only"
    )
    assert fixture.bank_builder.repetitions == [0, 1]
    assert len(fixture.model_executor.prepared_cell_ids) == 8
    assert len(fixture.cell_runner.run_cell_ids) == 10
    assert len(fixture.evidence_loader.loaded_cell_ids) == 20
    assert len(fixture.post_search.reloaded_cell_ids) == 20
    assert {outcome.attempt_count for outcome in first.cells} == {1}
    assert first.restart_success_pair_ids == (0, 1)
    assert (
        load_qualification_smoke_result(plan, fixture.evidence_loader, fixture.post_search) == first
    )


def test_execute_smoke_runs_all_ten_cells_concurrently_with_one_deadline(tmp_path: Path) -> None:
    """Given a blocking runner, when smoke runs, then all cells launch before any returns."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="6" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=50_000.0,
    )
    barrier_runner = _BarrierCellRunner()
    fixture = _fixture(cell_runner=barrier_runner)

    # When
    result = execute_qualification_smoke(plan, fixture.dependencies)

    # Then
    assert result.passed
    assert barrier_runner.max_active == 10
    assert set(barrier_runner.deadlines) == {50_300.0}


def test_completed_resume_loads_bindings_before_generation(tmp_path: Path) -> None:
    """Given completed smoke artifacts, when resumed, then generation seams are not called."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="7" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=60_000.0,
    )
    first_fixture = _fixture()
    first = execute_qualification_smoke(plan, first_fixture.dependencies)
    resume_fixture = _fixture(
        existing_cell_results=first_fixture.cell_runner.results,
        existing_post_search=first_fixture.post_search.outcomes,
    )

    # When
    resumed = execute_qualification_smoke(plan, resume_fixture.dependencies)

    # Then
    assert resumed == first
    assert resume_fixture.bank_builder.repetitions == []
    assert resume_fixture.model_executor.prepared_cell_ids == []
    assert resume_fixture.cell_runner.run_cell_ids == []
    assert len(resume_fixture.evidence_loader.loaded_cell_ids) == 10


def test_execute_smoke_rejects_preexisting_symlink_result(tmp_path: Path) -> None:
    """Given a result-path symlink, when smoke writes, then it fails closed."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="8" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=70_000.0,
    )
    plan.smoke_root.mkdir(parents=True)
    (plan.smoke_root / "qualification-smoke-result.json").symlink_to(
        tmp_path / "escaped-result.json"
    )

    # When / Then
    with pytest.raises(QualificationSmokeError, match="unsafe smoke artifact path"):
        _ = execute_qualification_smoke(plan, _fixture().dependencies)


def test_resume_rejects_tampered_binding_before_generation(tmp_path: Path) -> None:
    """Given a tampered binding file, when resumed, then no generation hides the mismatch."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="9" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=80_000.0,
    )
    first_fixture = _fixture()
    _ = execute_qualification_smoke(plan, first_fixture.dependencies)
    binding_path = plan.smoke_root / "qualification-smoke-bindings.json"
    payload = canonical_json_bytes(
        {
            "bindings": [
                {
                    "bank_sha256": "0" * 64,
                    "cell_id": plan.cells[0].cell_id,
                    "model_prep_sha256": "0" * 64,
                }
            ],
            "bindings_sha256": "0" * 64,
            "plan_sha256": "0" * 64,
            "schema_version": "dllm-openexr-qualification-smoke-bindings/v1",
        }
    )
    _ = binding_path.write_bytes(payload)
    resume_fixture = _fixture(
        existing_cell_results=first_fixture.cell_runner.results,
        existing_post_search=first_fixture.post_search.outcomes,
    )

    # When / Then
    with pytest.raises(QualificationSmokeError, match="binding"):
        _ = execute_qualification_smoke(plan, resume_fixture.dependencies)
    assert resume_fixture.bank_builder.repetitions == []


def test_load_smoke_rejects_forged_result_with_missing_cell(tmp_path: Path) -> None:
    """Given a persisted result missing one cell, when loaded, then it fails closed."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="3" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=90_000.0,
    )
    fixture = _fixture()
    result = execute_qualification_smoke(plan, fixture.dependencies)
    forged = result.model_copy(update={"cells": result.cells[:-1]})
    _ = (plan.smoke_root / "qualification-smoke-result.json").write_bytes(
        canonical_json_bytes(forged.model_dump(mode="json"))
    )

    # When / Then
    with pytest.raises(QualificationSmokeError, match="exactly 10 smoke cell outcomes"):
        _ = load_qualification_smoke_result(plan, fixture.evidence_loader, fixture.post_search)


def test_execute_smoke_rejects_missing_deferred_deep_parse_evidence(tmp_path: Path) -> None:
    """Given post-search evidence without deep parse, when run, then smoke cannot pass."""
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="4" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=120_000.0,
    )
    fixture = _fixture(deep_parse_status="unavailable")

    # When / Then
    with pytest.raises(QualificationSmokeError, match="deep-parse evidence unavailable"):
        _ = execute_qualification_smoke(plan, fixture.dependencies)


def test_execute_smoke_accepts_cells_stopped_at_the_deadline(tmp_path: Path) -> None:
    """Given cells that ran their whole window, when stopped at the deadline, smoke passes.

    Real Docker start-up eats the ~zero slack in -max_total_time, so healthy real
    cells end as "deadline" (the coordinator's normal terminal reason), not "completed".
    """
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="4" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=120_000.0,
    )
    fixture = _fixture(cell_runner=_CellRunner(status="deadline"))

    # When
    result = execute_qualification_smoke(plan, fixture.dependencies)

    # Then
    assert result.passed


@pytest.mark.parametrize("status", ["failed", "restart_limit"])
def test_execute_smoke_still_rejects_cells_that_did_not_run_their_window(
    tmp_path: Path, status: Literal["failed", "restart_limit"]
) -> None:
    # Given
    plan = build_qualification_smoke_plan(
        config_sha256="4" * 64,
        smoke_root=tmp_path / "smoke",
        start_monotonic_seconds=120_000.0,
    )
    fixture = _fixture(cell_runner=_CellRunner(status=status))

    # When / Then
    with pytest.raises(QualificationSmokeError, match="smoke cell did not complete"):
        _ = execute_qualification_smoke(plan, fixture.dependencies)


@dataclass(frozen=True, slots=True)
class _Bank:
    selected_bank_sha256: Sha256


@dataclass(frozen=True, slots=True)
class _BankBuilder:
    repetitions: list[int] = field(default_factory=list)

    def build(self, repetition: int) -> _Bank:
        self.repetitions.append(repetition)
        return _Bank(selected_bank_sha256=sha256_bytes(f"bank:{repetition}".encode()))


@dataclass(frozen=True, slots=True)
class _ModelExecutor:
    prepared_cell_ids: list[str] = field(default_factory=list)
    preparation_sha256_by_cell: dict[str, str] = field(default_factory=dict)

    def prepare_corpus(self, cell_id: str, bank_sha256: str) -> SmokeModelPreparation:
        self.prepared_cell_ids.append(cell_id)
        preparation_sha256 = sha256_bytes(f"model:{cell_id}:{bank_sha256}".encode())
        self.preparation_sha256_by_cell[cell_id] = preparation_sha256
        return SmokeModelPreparation(
            cell_id=cell_id,
            preparation_sha256=preparation_sha256,
        )


@dataclass(frozen=True, slots=True)
class _CellRunner:
    run_cell_ids: list[str] = field(default_factory=list)
    results: dict[str, SmokeCellResult] = field(default_factory=dict)
    status: Literal["completed", "failed", "deadline", "restart_limit"] = "completed"

    def run(self, cell_id: str, deadline_monotonic_seconds: float) -> SmokeCellResult:
        self.run_cell_ids.append(cell_id)
        result = SmokeCellResult(
            cell_id=cell_id,
            status=self.status,
            attempt_count=1,
            deadline_monotonic_seconds=deadline_monotonic_seconds,
            result_sha256=sha256_bytes(f"cell:{cell_id}".encode()),
        )
        self.results[cell_id] = result
        return result


@dataclass(frozen=True, slots=True)
class _BarrierCellRunner(_CellRunner):
    barrier: threading.Barrier = field(default_factory=lambda: threading.Barrier(10))
    active: int = 0
    max_active: int = 0
    deadlines: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @override
    def run(self, cell_id: str, deadline_monotonic_seconds: float) -> SmokeCellResult:
        with self.lock:
            self.deadlines.append(deadline_monotonic_seconds)
            object.__setattr__(self, "active", self.active + 1)
            object.__setattr__(self, "max_active", max(self.max_active, self.active))
        _ = self.barrier.wait(timeout=5.0)
        try:
            return _CellRunner.run(self, cell_id, deadline_monotonic_seconds)
        finally:
            with self.lock:
                object.__setattr__(self, "active", self.active - 1)


@dataclass(frozen=True, slots=True)
class _EvidenceLoader:
    runner: _CellRunner
    loaded_cell_ids: list[str] = field(default_factory=list)

    def load_cell_result(self, cell_id: str) -> SmokeCellResult:
        self.loaded_cell_ids.append(cell_id)
        return self.runner.results[cell_id]


@dataclass(frozen=True, slots=True)
class _PostSearch:
    deep_parse_status: Literal["available", "unavailable"]
    built_cell_ids: list[str] = field(default_factory=list)
    reloaded_cell_ids: list[str] = field(default_factory=list)
    outcomes: dict[str, SmokePostSearchOutcome] = field(default_factory=dict)

    def coverage_and_proof(self, cell_id: str, result_sha256: str) -> SmokePostSearchOutcome:
        self.built_cell_ids.append(cell_id)
        outcome = _post_search_outcome(cell_id, result_sha256, self.deep_parse_status)
        self.outcomes[cell_id] = outcome
        return outcome

    def load_outcome(self, cell_id: str) -> SmokePostSearchOutcome:
        self.reloaded_cell_ids.append(cell_id)
        return self.outcomes[cell_id]


@dataclass(frozen=True, slots=True)
class _Fixture:
    dependencies: SmokeExecutionDependencies
    bank_builder: _BankBuilder
    model_executor: _ModelExecutor
    cell_runner: _CellRunner
    evidence_loader: _EvidenceLoader
    post_search: _PostSearch


def _fixture_with_options(
    *,
    deep_parse_status: Literal["available", "unavailable"],
    cell_runner: _CellRunner | None = None,
    existing_cell_results: dict[str, SmokeCellResult] | None = None,
    existing_post_search: dict[str, SmokePostSearchOutcome] | None = None,
) -> _Fixture:
    bank_builder = _BankBuilder()
    model_executor = _ModelExecutor()
    runner = cell_runner or _CellRunner()
    if existing_cell_results is not None:
        runner.results.update(existing_cell_results)
    evidence_loader = _EvidenceLoader(runner)
    post_search = _PostSearch(deep_parse_status)
    if existing_post_search is not None:
        post_search.outcomes.update(existing_post_search)
    dependencies = SmokeExecutionDependencies(
        bank_builder=bank_builder,
        model_executor=model_executor,
        cell_runner=runner,
        evidence_loader=evidence_loader,
        post_search=post_search,
    )
    return _Fixture(
        dependencies=dependencies,
        bank_builder=bank_builder,
        model_executor=model_executor,
        cell_runner=runner,
        evidence_loader=evidence_loader,
        post_search=post_search,
    )


def _fixture(
    *,
    deep_parse_status: Literal["available", "unavailable"] = "available",
    cell_runner: _CellRunner | None = None,
    existing_cell_results: dict[str, SmokeCellResult] | None = None,
    existing_post_search: dict[str, SmokePostSearchOutcome] | None = None,
) -> _Fixture:
    return _fixture_with_options(
        deep_parse_status=deep_parse_status,
        cell_runner=cell_runner,
        existing_cell_results=existing_cell_results,
        existing_post_search=existing_post_search,
    )


def _post_search_outcome(
    cell_id: str,
    result_sha256: str,
    deep_parse_status: Literal["available", "unavailable"],
) -> SmokePostSearchOutcome:
    payload = canonical_json_bytes({"cell_id": cell_id, "result_sha256": result_sha256})
    return SmokePostSearchOutcome(
        cell_id=cell_id,
        proof_status="available",
        deep_parse_status=deep_parse_status,
        coverage_envelope_sha256=sha256_bytes(b"coverage:" + payload),
        proof_envelope_sha256=sha256_bytes(b"proof-envelope:" + payload),
        deep_parse_envelope_sha256=sha256_bytes(b"parse-envelope:" + payload),
        proof_result_sha256=sha256(payload).hexdigest(),
        deep_parse_result_sha256=sha256_bytes(b"deep:" + payload),
    )
