"""Frozen budget and grid contracts for the three-target 24-hour experiment."""

from dllm_agent.experiments.multitarget.budget import (
    ARMS,
    FEEDBACK_ARMS,
    TARGETS,
    ExperimentSpec,
)


def test_default_spec_expands_to_exact_fifteen_cells() -> None:
    # Given
    spec = ExperimentSpec()

    # When
    cells = spec.cells()

    # Then
    assert len(cells) == 15
    assert {(cell.target, cell.arm, cell.repetition) for cell in cells} == {
        (target, arm, 0) for target in TARGETS for arm in ARMS
    }
    assert spec.raw_particle_count == 32
    assert spec.minimum_unique_count == 16


def test_feedback_cell_budget_conserves_exact_twenty_four_worker_hours() -> None:
    # Given
    spec = ExperimentSpec()

    for arm in FEEDBACK_ARMS:
        # When
        allocation = spec.budget.for_arm(arm)

        # Then
        assert allocation.initial_probe_worker_seconds == 480
        assert allocation.revision_probe_worker_seconds == 480
        assert allocation.campaign_worker_seconds == 85_440
        assert allocation.search_worker_seconds == 86_400


def test_non_feedback_cell_budget_keeps_full_campaign_day() -> None:
    # Given
    spec = ExperimentSpec()

    for arm in ("libfuzzer-only", "structured-one-shot"):
        # When
        allocation = spec.budget.for_arm(arm)

        # Then
        assert allocation.initial_probe_worker_seconds == 0
        assert allocation.revision_probe_worker_seconds == 0
        assert allocation.campaign_worker_seconds == 86_400
        assert allocation.search_worker_seconds == 86_400


def test_proof_budget_is_not_silently_charged_to_search_budget() -> None:
    # Given
    spec = ExperimentSpec()

    # When
    allocation = spec.budget.for_arm("selective-revision")

    # Then
    assert allocation.proof_worker_seconds == 0
    assert allocation.search_worker_seconds == 86_400
