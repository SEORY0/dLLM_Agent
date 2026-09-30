"""Production-shaped qualification smoke execution kernel."""

from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
# ruff: noqa: D103
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, assert_never

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.models import ARMS
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    BINDINGS_NAME,
    RESULT_NAME,
    SMOKE_DEADLINE_SECONDS,
    CandidateBankSurface,
    ModelExecutor,
    QualificationSmokeBinding,
    QualificationSmokeBindings,
    QualificationSmokeCell,
    QualificationSmokeCellOutcome,
    QualificationSmokeError,
    QualificationSmokePlan,
    QualificationSmokeResult,
    SmokeCellResult,
    SmokeEvidenceLoader,
    SmokeExecutionDependencies,
    SmokePostSearch,
    SmokePostSearchOutcome,
    ensure_smoke_artifact,
    load_smoke_bindings_file,
    load_smoke_result_file,
    model_sha256,
    reject_unsafe_artifact_path,
    validate_binding_for_cell,
    validate_qualification_smoke_result,
    write_canonical_once,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.core.models import Sha256


def build_qualification_smoke_plan(
    *, config_sha256: Sha256, smoke_root: Path, start_monotonic_seconds: float
) -> QualificationSmokePlan:
    deadline = start_monotonic_seconds + SMOKE_DEADLINE_SECONDS
    cells = tuple(
        QualificationSmokeCell(
            cell_id=f"smoke-rep-{repetition:02d}-{arm}",
            pair_id=repetition,
            repetition=repetition,
            arm=arm,
            fuzzer_seed=1_000_000 + (repetition * 1_000) + arm_offset + 1,
            start_monotonic_seconds=start_monotonic_seconds,
            deadline_monotonic_seconds=deadline,
            deadline_seconds=SMOKE_DEADLINE_SECONDS,
            artifact_root=smoke_root / "cells" / f"rep-{repetition:02d}" / arm,
        )
        for repetition in (0, 1)
        for arm_offset, arm in enumerate(ARMS)
    )
    return QualificationSmokePlan(config_sha256=config_sha256, smoke_root=smoke_root, cells=cells)


def execute_qualification_smoke(
    plan: QualificationSmokePlan, dependencies: SmokeExecutionDependencies
) -> QualificationSmokeResult:
    result_path = plan.smoke_root / RESULT_NAME
    bindings_path = plan.smoke_root / BINDINGS_NAME
    reject_unsafe_artifact_path(result_path)
    reject_unsafe_artifact_path(bindings_path)
    if result_path.exists():
        bindings = load_smoke_bindings_file(plan)
        result = load_qualification_smoke_result(
            plan,
            dependencies.evidence_loader,
            dependencies.post_search,
            bindings=bindings,
        )
        _ = ensure_smoke_artifact(plan, result)
        return result
    plan.smoke_root.mkdir(parents=True, exist_ok=True)
    banks = {repetition: dependencies.bank_builder.build(repetition) for repetition in (0, 1)}
    bindings = _build_bindings(plan, dependencies.model_executor, banks)
    write_canonical_once(bindings_path, bindings)
    result_by_cell = _run_cells_concurrently(plan, dependencies)
    outcomes: list[QualificationSmokeCellOutcome] = []
    binding_by_cell = {binding.cell_id: binding for binding in bindings.bindings}
    for cell in plan.cells:
        result = result_by_cell[cell.cell_id]
        post_search = dependencies.post_search.coverage_and_proof(
            cell.cell_id, result.result_sha256
        )
        outcomes.append(_cell_outcome(cell, binding_by_cell[cell.cell_id], result, post_search))
    reloaded = tuple(
        _cell_outcome(
            cell,
            binding_by_cell[cell.cell_id],
            dependencies.evidence_loader.load_cell_result(cell.cell_id),
            dependencies.post_search.load_outcome(cell.cell_id),
        )
        for cell in plan.cells
    )
    restart_success_pair_ids = _restart_success_pair_ids(plan, tuple(outcomes), reloaded)
    smoke_result = _result_from_cells(plan, tuple(outcomes), restart_success_pair_ids)
    write_canonical_once(result_path, smoke_result)
    _ = ensure_smoke_artifact(plan, smoke_result)
    return smoke_result


def load_qualification_smoke_result(
    plan: QualificationSmokePlan,
    evidence_loader: SmokeEvidenceLoader,
    post_search: SmokePostSearch,
    *,
    bindings: QualificationSmokeBindings | None = None,
) -> QualificationSmokeResult:
    loaded_bindings = load_smoke_bindings_file(plan) if bindings is None else bindings
    result = load_smoke_result_file(plan)
    _validate_result_shape(plan, result)
    binding_by_cell = {binding.cell_id: binding for binding in loaded_bindings.bindings}
    reloaded = tuple(
        _cell_outcome(
            cell,
            binding_by_cell[cell.cell_id],
            evidence_loader.load_cell_result(cell.cell_id),
            post_search.load_outcome(cell.cell_id),
        )
        for cell in plan.cells
    )
    return _require_matching_result(plan, result, reloaded)


def _build_bindings(
    plan: QualificationSmokePlan,
    model_executor: ModelExecutor,
    banks: dict[int, CandidateBankSurface],
) -> QualificationSmokeBindings:
    bindings = tuple(_build_binding(cell, model_executor, banks) for cell in plan.cells)
    unsigned = QualificationSmokeBindings(
        plan_sha256=model_sha256(plan),
        bindings=bindings,
        bindings_sha256="0" * 64,
    )
    payload = canonical_json_bytes(unsigned.model_dump(mode="json", exclude={"bindings_sha256"}))
    return unsigned.model_copy(update={"bindings_sha256": sha256_bytes(payload)})


def _build_binding(
    cell: QualificationSmokeCell,
    model_executor: ModelExecutor,
    banks: dict[int, CandidateBankSurface],
) -> QualificationSmokeBinding:
    bank_sha256 = banks[cell.repetition].selected_bank_sha256
    model_prep_sha256: Sha256 | None
    match cell.arm:
        case "selective-revision" | "final-only" | "full-regenerate" | "structured-one-shot":
            result = model_executor.prepare_corpus(cell.cell_id, bank_sha256)
            if result.cell_id != cell.cell_id:
                raise QualificationSmokeError(reason="model preparation cell mismatch")
            model_prep_sha256 = result.preparation_sha256
        case "libfuzzer-only":
            model_prep_sha256 = None
        case unreachable:
            assert_never(unreachable)
    return QualificationSmokeBinding(
        cell_id=cell.cell_id,
        bank_sha256=bank_sha256,
        model_prep_sha256=model_prep_sha256,
    )


def _run_cells_concurrently(
    plan: QualificationSmokePlan, dependencies: SmokeExecutionDependencies
) -> dict[str, SmokeCellResult]:
    with ThreadPoolExecutor(max_workers=len(plan.cells)) as executor:
        futures = {
            executor.submit(
                dependencies.cell_runner.run,
                cell.cell_id,
                cell.deadline_monotonic_seconds,
            ): cell.cell_id
            for cell in plan.cells
        }
        return {futures[future]: future.result() for future in as_completed(futures)}


def _cell_outcome(
    cell: QualificationSmokeCell,
    binding: QualificationSmokeBinding,
    result: SmokeCellResult,
    post_search: SmokePostSearchOutcome,
) -> QualificationSmokeCellOutcome:
    _validate_cell_evidence(cell, binding, result, post_search)
    unsigned = QualificationSmokeCellOutcome(
        cell_id=cell.cell_id,
        pair_id=cell.pair_id,
        repetition=cell.repetition,
        arm=cell.arm,
        attempt_count=result.attempt_count,
        bank_sha256=binding.bank_sha256,
        model_prep_sha256=binding.model_prep_sha256,
        cell_result_sha256=result.result_sha256,
        coverage_envelope_sha256=post_search.coverage_envelope_sha256,
        proof_envelope_sha256=post_search.proof_envelope_sha256,
        deep_parse_envelope_sha256=post_search.deep_parse_envelope_sha256,
        proof_result_sha256=post_search.proof_result_sha256,
        deep_parse_result_sha256=post_search.deep_parse_result_sha256,
        cell_outcome_sha256="0" * 64,
    )
    payload = canonical_json_bytes(
        unsigned.model_dump(mode="json", exclude={"cell_outcome_sha256"})
    )
    return unsigned.model_copy(update={"cell_outcome_sha256": sha256_bytes(payload)})


def _validate_cell_evidence(
    cell: QualificationSmokeCell,
    binding: QualificationSmokeBinding,
    result: SmokeCellResult,
    post_search: SmokePostSearchOutcome,
) -> None:
    if (
        binding.cell_id != cell.cell_id
        or result.cell_id != cell.cell_id
        or post_search.cell_id != cell.cell_id
    ):
        raise QualificationSmokeError(reason="smoke cell evidence id mismatch")
    validate_binding_for_cell(cell, binding)
    if result.deadline_monotonic_seconds != cell.deadline_monotonic_seconds:
        raise QualificationSmokeError(reason="smoke cell deadline mismatch")
    # A search cell that ran its whole window ends one of two healthy ways: the
    # fuzzer exits on its own ("completed"), or the supervisor stops it at the
    # absolute deadline ("deadline" -- the coordinator's normal terminal reason).
    # Real Docker start-up always eats the ~0 slack in -max_total_time, so real
    # cells finish as "deadline". "failed"/"restart_limit" are still rejected.
    if result.status not in {"completed", "deadline"}:
        raise QualificationSmokeError(reason="smoke cell did not complete")
    if post_search.proof_status != "available":
        raise QualificationSmokeError(reason="deferred proof evidence unavailable")
    if post_search.deep_parse_status != "available":
        raise QualificationSmokeError(reason="deep-parse evidence unavailable")


def _result_from_cells(
    plan: QualificationSmokePlan,
    cells: tuple[QualificationSmokeCellOutcome, ...],
    restart_success_pair_ids: tuple[int, ...],
) -> QualificationSmokeResult:
    _validate_cell_order(plan, cells)
    cell_payloads = [cell.model_dump(mode="json") for cell in cells]
    unsigned = QualificationSmokeResult(
        plan_sha256=model_sha256(plan),
        cells=cells,
        cell_result_set_sha256=sha256_bytes(canonical_json_bytes(cell_payloads)),
        restart_success_pair_ids=restart_success_pair_ids,
        result_sha256="0" * 64,
        passed=True,
    )
    result_sha256 = sha256_bytes(
        canonical_json_bytes(unsigned.model_dump(mode="json", exclude={"result_sha256"}))
    )
    return unsigned.model_copy(update={"result_sha256": result_sha256})


def _require_matching_result(
    plan: QualificationSmokePlan,
    result: QualificationSmokeResult,
    reloaded: tuple[QualificationSmokeCellOutcome, ...],
) -> QualificationSmokeResult:
    expected_restart = _restart_success_pair_ids(plan, result.cells, reloaded)
    if result.restart_success_pair_ids != expected_restart:
        raise QualificationSmokeError(reason="qualification smoke restart audit mismatch")
    expected = _result_from_cells(plan, reloaded, expected_restart)
    if result != expected:
        raise QualificationSmokeError(
            reason="qualification smoke result conflicts with cell evidence"
        )
    return result


def _validate_result_shape(plan: QualificationSmokePlan, result: QualificationSmokeResult) -> None:
    validate_qualification_smoke_result(plan, result)


def _validate_cell_order(
    plan: QualificationSmokePlan,
    cells: tuple[QualificationSmokeCellOutcome, ...],
) -> None:
    if tuple(cell.cell_id for cell in cells) != tuple(cell.cell_id for cell in plan.cells):
        raise QualificationSmokeError(reason="qualification smoke cell set mismatch")


def _restart_success_pair_ids(
    plan: QualificationSmokePlan,
    first_pass: tuple[QualificationSmokeCellOutcome, ...],
    reloaded: tuple[QualificationSmokeCellOutcome, ...],
) -> tuple[int, ...]:
    _validate_cell_order(plan, first_pass)
    _validate_cell_order(plan, reloaded)
    first_by_id = {cell.cell_id: cell for cell in first_pass}
    reloaded_by_id = {cell.cell_id: cell for cell in reloaded}
    pair_ids = tuple(
        pair_id
        for pair_id in sorted({cell.pair_id for cell in plan.cells})
        if all(
            first_by_id[cell.cell_id] == reloaded_by_id[cell.cell_id]
            for cell in plan.cells
            if cell.pair_id == pair_id
        )
    )
    if len(pair_ids) != plan.pair_count:
        raise QualificationSmokeError(reason="qualification smoke restart audit failed")
    return pair_ids
