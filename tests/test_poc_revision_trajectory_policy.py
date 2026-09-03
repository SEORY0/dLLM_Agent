"""Per-lineage trajectory-aware field policy tests."""

import pytest

from dllm_agent.revision.controller_models import RevisionResult, RevisionTrajectoryStep
from dllm_agent.revision.trajectory_policy import (
    LineageFieldOutcome,
    LineageOutcomeScopeError,
    choose_lineage_field,
    field_trajectories,
)
from dllm_agent.targets.openexr.particle import FIELD_ORDER, FieldName, SelectorSlots


def test_choose_lineage_field_uses_real_commit_timing_for_each_lineage() -> None:
    """Given two trajectories, when different fields commit late, then choices differ."""
    # Given
    width_late = _result(
        particle_offset=1,
        suggested_field="origin_x",
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(0, 1, 3, 8),
                token_ids=(10, 11, 13, 18),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(2,),
                token_ids=(12,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )
    height_late = _result(
        particle_offset=2,
        suggested_field="origin_x",
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(0, 1, 2, 8),
                token_ids=(20, 21, 22, 28),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(3,),
                token_ids=(23,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )

    # When
    width_choice = choose_lineage_field(width_late, outcomes=())
    height_choice = choose_lineage_field(height_late, outcomes=())

    # Then
    assert width_late.suggested_field == "origin_x"
    assert height_late.suggested_field == "origin_x"
    assert width_choice == "width"
    assert height_choice == "height"


def test_field_trajectories_use_slot_positions_for_non_identity_slots() -> None:
    """Given non-identity slots, when width commits late, then width owns that commit."""
    # Given
    result = _result(
        particle_offset=4,
        suggested_field="origin_x",
        slot_positions=(10, 11, 42, 43, 14, 15, 16, 17, 48, 19),
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(10, 11, 43, 48),
                token_ids=(40, 41, 43, 48),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(42,),
                token_ids=(42,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )

    # When
    trajectories = field_trajectories(result)
    chosen = choose_lineage_field(result, outcomes=())

    # Then
    assert {trajectory.field: trajectory.committed_step_index for trajectory in trajectories}[
        "width"
    ] == 3
    assert chosen == "width"


def test_choose_lineage_field_explores_away_from_observed_field_outcomes() -> None:
    """Given rewarded and failed fields, when choosing again, then an untried field is explored."""
    # Given
    result = _result(
        particle_offset=3,
        suggested_field="origin_x",
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(0, 1, 3, 8),
                token_ids=(30, 31, 33, 38),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(2,),
                token_ids=(32,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )
    outcomes = (
        LineageFieldOutcome(lineage_id="lineage-a", field="width", reward_delta=1.0, trials=1),
        LineageFieldOutcome(lineage_id="lineage-a", field="height", reward_delta=-1.0, trials=1),
    )

    # When
    trajectories = field_trajectories(result)
    chosen = choose_lineage_field(result, outcomes=outcomes, lineage_id="lineage-a")

    # Then
    assert {trajectory.field: trajectory.committed_step_index for trajectory in trajectories}[
        "width"
    ] == 3
    assert chosen == "origin_x"


def test_choose_lineage_field_requires_scope_for_non_empty_outcomes() -> None:
    """Given other lineage outcomes, when no scope is provided, then selection rejects them."""
    # Given
    result = _result(
        particle_offset=5,
        suggested_field="origin_x",
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(0, 1, 3, 8),
                token_ids=(50, 51, 53, 58),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(2,),
                token_ids=(52,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )
    outcomes = (
        LineageFieldOutcome(lineage_id="other-lineage", field="width", reward_delta=1.0, trials=1),
    )

    # When / Then
    with pytest.raises(LineageOutcomeScopeError, match="lineage_id is required"):
        _ = choose_lineage_field(result, outcomes=outcomes)

    # Then
    assert choose_lineage_field(result, outcomes=outcomes, lineage_id="lineage-a") == "width"


def _result(
    *,
    particle_offset: int,
    suggested_field: FieldName,
    trajectory_steps: tuple[RevisionTrajectoryStep, ...],
    slot_positions: tuple[int, ...] | None = None,
) -> RevisionResult:
    selectors = _selectors(particle_offset)
    token_ids = tuple(selector + 100 for selector in selectors)
    return RevisionResult(
        selectors=selectors,
        token_ids=token_ids,
        slot_positions=tuple(range(len(FIELD_ORDER))) if slot_positions is None else slot_positions,
        selected_slot_token_ids=token_ids,
        token_state_sha256=f"{particle_offset:064x}",
        transfer_evidence_sha256=f"{particle_offset + 1:064x}",
        rng_state_sha256=f"{particle_offset + 2:064x}",
        nfe=8,
        forward_calls=8,
        gpu_ms=4,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field=suggested_field,
        trajectory_steps=trajectory_steps,
        trajectory_evidence_sha256=f"{particle_offset + 3:064x}",
    )


def _selectors(offset: int) -> SelectorSlots:
    return (
        offset,
        offset + 1,
        offset + 2,
        offset + 3,
        offset + 4,
        offset + 5,
        offset + 6,
        offset + 7,
        offset + 8,
        offset + 9,
    )


def _step(
    *,
    step_index: int,
    committed_positions: tuple[int, ...],
    token_ids: tuple[int, ...],
    confidences: tuple[float, ...],
    remaining_masked: int,
) -> RevisionTrajectoryStep:
    return RevisionTrajectoryStep(
        step_index=step_index,
        block_index=0,
        masked_before=len(committed_positions) + remaining_masked,
        committed_positions=committed_positions,
        token_ids=token_ids,
        confidences=confidences,
        proposal_positions=committed_positions,
        proposal_token_ids=token_ids,
        proposal_confidences=confidences,
        proposal_margins=tuple(0.0 for _ in committed_positions),
        remaining_masked=remaining_masked,
    )
