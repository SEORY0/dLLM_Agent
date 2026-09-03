"""Production adapter contract for qualification smoke execution."""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnnecessaryComparison=false
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, assert_never

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.qualification_smoke import (
    _cell_outcome,
    _require_matching_result,
    _restart_success_pair_ids,
    _result_from_cells,
    _validate_result_shape,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    BINDINGS_NAME,
    RESULT_NAME,
    SMOKE_DEADLINE_SECONDS,
    QualificationSmokeBinding,
    QualificationSmokeBindings,
    QualificationSmokeCell,
    QualificationSmokeCellOutcome,
    QualificationSmokeError,
    QualificationSmokePlan,
    QualificationSmokeResult,
    SmokeCellResult,
    SmokeModelPreparation,
    SmokePostSearchOutcome,
    ensure_smoke_artifact,
    load_smoke_bindings_file,
    load_smoke_result_file,
    model_sha256,
    write_canonical_once,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.candidate_bank import CandidateBank
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.coordinator import CampaignConfig
    from dllm_agent.experiments.openexr_one_day.models import Arm
    from dllm_agent.experiments.openexr_one_day.post_search import PostSearchOutcome


@dataclass(frozen=True, slots=True)
class ProductionSmokeCellBinding:
    """Real production objects bound to one smoke-plan cell."""

    config: CampaignConfig
    plan_cell: QualificationSmokeCell
    request: CellRunRequest
    bank: CandidateBank


class ProductionSmokeAdapter(Protocol):
    """Minimal adapter contract for wiring the private coordinator runtime to smoke."""

    def build_candidate_bank(self, config: CampaignConfig, repetition: int) -> CandidateBank:
        """Build or load the real candidate bank for one smoke repetition."""
        ...

    def bind_cell_request(
        self,
        config: CampaignConfig,
        plan_cell: QualificationSmokeCell,
        bank: CandidateBank,
    ) -> ProductionSmokeCellBinding:
        """Return the exact production cell request for one smoke cell."""
        ...

    def bind_persisted_cell_request(
        self,
        config: CampaignConfig,
        plan_cell: QualificationSmokeCell,
        binding: QualificationSmokeBinding,
    ) -> ProductionSmokeCellBinding:
        """Return the exact production cell request without regenerating banks."""
        ...

    def prepare_model_corpus(self, binding: ProductionSmokeCellBinding) -> SmokeModelPreparation:
        """Prepare a model-arm corpus through the real model/cell interfaces."""
        ...

    def run_cell(self, binding: ProductionSmokeCellBinding) -> CellResult:
        """Run the real Task4 cell process for one smoke cell."""
        ...

    def validate_cell(self, binding: ProductionSmokeCellBinding) -> CellResult:
        """Reload and validate persisted cell artifacts for one smoke cell."""
        ...

    def run_post_search(
        self, binding: ProductionSmokeCellBinding, result: CellResult
    ) -> PostSearchOutcome:
        """Build real coverage, proof, and deep-parse envelopes."""
        ...

    def load_post_search(
        self, binding: ProductionSmokeCellBinding, result: CellResult
    ) -> PostSearchOutcome:
        """Reload persisted coverage, proof, and deep-parse envelopes."""
        ...


def execute_production_qualification_smoke(
    config: CampaignConfig,
    plan: QualificationSmokePlan,
    adapter: ProductionSmokeAdapter,
) -> QualificationSmokeResult:
    """Run smoke using real CampaignConfig, CandidateBank, CellRunRequest, and envelopes."""
    result_path = plan.smoke_root / RESULT_NAME
    if result_path.exists():
        persisted = load_smoke_bindings_file(plan)
        result = _load_production_result(
            plan,
            adapter,
            tuple(
                adapter.bind_persisted_cell_request(config, cell, binding)
                for cell, binding in zip(plan.cells, persisted.bindings, strict=True)
            ),
            persisted,
        )
        _ = ensure_smoke_artifact(plan, result)
        return result
    banks = {repetition: adapter.build_candidate_bank(config, repetition) for repetition in (0, 1)}
    bindings = tuple(
        adapter.bind_cell_request(config, cell, banks[cell.repetition]) for cell in plan.cells
    )
    smoke_bindings = _build_production_bindings(plan, bindings, adapter)
    write_canonical_once(plan.smoke_root / BINDINGS_NAME, smoke_bindings)
    result_by_cell = _run_production_cells_concurrently(bindings, adapter)
    outcomes: list[QualificationSmokeCellOutcome] = []
    binding_by_cell = {binding.cell_id: binding for binding in smoke_bindings.bindings}
    for binding in bindings:
        _validate_binding(binding)
        result = result_by_cell[binding.plan_cell.cell_id]
        post_search = adapter.run_post_search(binding, result)
        outcomes.append(
            _production_cell_outcome(
                binding,
                binding_by_cell[binding.plan_cell.cell_id],
                result,
                post_search,
            )
        )
    reloaded = tuple(
        _reload_production_cell(binding, binding_by_cell[binding.plan_cell.cell_id], adapter)
        for binding in bindings
    )
    restart_success_pair_ids = _restart_success_pair_ids(plan, tuple(outcomes), reloaded)
    smoke_result = _result_from_cells(plan, tuple(outcomes), restart_success_pair_ids)
    write_canonical_once(result_path, smoke_result)
    _ = ensure_smoke_artifact(plan, smoke_result)
    return smoke_result


def _load_production_result(
    plan: QualificationSmokePlan,
    adapter: ProductionSmokeAdapter,
    bindings: tuple[ProductionSmokeCellBinding, ...],
    smoke_bindings: QualificationSmokeBindings,
) -> QualificationSmokeResult:
    result = load_smoke_result_file(plan)
    _validate_result_shape(plan, result)
    binding_by_cell = {binding.cell_id: binding for binding in smoke_bindings.bindings}
    reloaded = tuple(
        _reload_production_cell(binding, binding_by_cell[binding.plan_cell.cell_id], adapter)
        for binding in bindings
    )
    return _require_matching_result(plan, result, reloaded)


def _reload_production_cell(
    binding: ProductionSmokeCellBinding,
    smoke_binding: QualificationSmokeBinding,
    adapter: ProductionSmokeAdapter,
) -> QualificationSmokeCellOutcome:
    result = adapter.validate_cell(binding)
    return _production_cell_outcome(
        binding, smoke_binding, result, adapter.load_post_search(binding, result)
    )


def _build_production_bindings(
    plan: QualificationSmokePlan,
    bindings: tuple[ProductionSmokeCellBinding, ...],
    adapter: ProductionSmokeAdapter,
) -> QualificationSmokeBindings:
    smoke_bindings = tuple(_build_production_binding(binding, adapter) for binding in bindings)
    unsigned = QualificationSmokeBindings(
        plan_sha256=model_sha256(plan),
        bindings=smoke_bindings,
        bindings_sha256="0" * 64,
    )
    payload = canonical_json_bytes(unsigned.model_dump(mode="json", exclude={"bindings_sha256"}))
    return unsigned.model_copy(update={"bindings_sha256": sha256_bytes(payload)})


def _build_production_binding(
    binding: ProductionSmokeCellBinding, adapter: ProductionSmokeAdapter
) -> QualificationSmokeBinding:
    model_prep_sha256 = None
    if _is_model_arm(binding.plan_cell.arm):
        prep = adapter.prepare_model_corpus(binding)
        if prep.cell_id != binding.plan_cell.cell_id:
            raise QualificationSmokeError(reason="model preparation cell mismatch")
        model_prep_sha256 = prep.preparation_sha256
    return QualificationSmokeBinding(
        cell_id=binding.plan_cell.cell_id,
        bank_sha256=binding.bank.selected_bank_sha256,
        model_prep_sha256=model_prep_sha256,
    )


def _run_production_cells_concurrently(
    bindings: tuple[ProductionSmokeCellBinding, ...], adapter: ProductionSmokeAdapter
) -> dict[str, CellResult]:
    with ThreadPoolExecutor(max_workers=len(bindings)) as executor:
        futures = {executor.submit(adapter.run_cell, binding): binding for binding in bindings}
        return {
            futures[future].plan_cell.cell_id: future.result() for future in as_completed(futures)
        }


def _production_cell_outcome(
    binding: ProductionSmokeCellBinding,
    smoke_binding: QualificationSmokeBinding,
    result: CellResult,
    post_search: PostSearchOutcome,
) -> QualificationSmokeCellOutcome:
    smoke_result = SmokeCellResult(
        cell_id=result.cell_id,
        status=result.status,
        attempt_count=result.attempt_count,
        deadline_monotonic_seconds=binding.request.deadline_monotonic_seconds,
        result_sha256=model_sha256(result),
    )
    smoke_post_search = SmokePostSearchOutcome(
        cell_id=result.cell_id,
        proof_status=post_search.proof_status,
        deep_parse_status=post_search.deep_parse_status,
        coverage_envelope_sha256=_required_sha(
            post_search.coverage_envelope_sha256, "coverage envelope"
        ),
        proof_envelope_sha256=_required_sha(post_search.proof_envelope_sha256, "proof envelope"),
        deep_parse_envelope_sha256=_required_sha(
            post_search.deep_parse_envelope_sha256, "deep-parse envelope"
        ),
        proof_result_sha256=post_search.proof_result_sha256,
        deep_parse_result_sha256=_required_sha(
            post_search.deep_parse_result_sha256, "deep-parse result"
        ),
    )
    return _cell_outcome(binding.plan_cell, smoke_binding, smoke_result, smoke_post_search)


def _validate_binding(binding: ProductionSmokeCellBinding) -> None:
    cell = binding.plan_cell
    request = binding.request
    if request.cell_id != cell.cell_id:
        raise QualificationSmokeError(reason="production smoke request cell mismatch")
    if request.artifact_root != cell.artifact_root:
        raise QualificationSmokeError(reason="production smoke request artifact root mismatch")
    if request.deadline_monotonic_seconds != cell.deadline_monotonic_seconds:
        raise QualificationSmokeError(reason="production smoke request deadline mismatch")
    duration = request.deadline_monotonic_seconds - request.start_monotonic_seconds
    if duration != SMOKE_DEADLINE_SECONDS:
        raise QualificationSmokeError(reason="production smoke request is not 300 seconds")
    if request.execution_profile != "qualification-smoke" or request.schedule_seconds != (0, 300):
        raise QualificationSmokeError(reason="production smoke request profile mismatch")


def _is_model_arm(arm: Arm) -> bool:
    match arm:
        case "selective-revision" | "final-only" | "full-regenerate" | "structured-one-shot":
            return True
        case "libfuzzer-only":
            return False
        case unreachable:
            assert_never(unreachable)


def _required_sha(value: str | None, label: str) -> str:
    if value is None:
        raise QualificationSmokeError(reason=f"{label} hash missing")
    return value
