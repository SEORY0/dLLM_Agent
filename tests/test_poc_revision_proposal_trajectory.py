"""Proposal-level trajectory policy regression tests."""

import pytest

from dllm_agent.revision.controller_models import RevisionResult, RevisionTrajectoryStep
from dllm_agent.revision.trajectory_policy import (
    LineageFieldOutcome,
    choose_lineage_field,
    field_trajectories,
)
from dllm_agent.targets.openexr.particle import FIELD_ORDER


def test_field_trajectories_count_proposal_flip_before_commit() -> None:
    """Given A then B proposals, when B commits, then the trajectory reports one flip."""
    # Given
    width_position = FIELD_ORDER.index("width")
    result = _result(
        (
            _proposal_step(
                step_index=1,
                positions=(width_position,),
                proposal_tokens=(41,),
                confidences=(0.4,),
                margins=(-0.2,),
            ),
            _proposal_step(
                step_index=2,
                positions=(width_position,),
                proposal_tokens=(42,),
                confidences=(0.8,),
                margins=(0.5,),
                commit=True,
            ),
        )
    )

    # When
    trajectories = {trajectory.field: trajectory for trajectory in field_trajectories(result)}

    # Then
    width = trajectories["width"]
    assert width.committed_step_index == 2
    assert width.token_flip_count == 1
    assert width.mask_lifespan == 2
    assert width.mean_confidence == pytest.approx(0.6)
    assert width.confidence_margin == pytest.approx(0.15)
    assert width.remask_pressure == pytest.approx(3.075)


def test_proposal_histories_change_field_choice_with_same_final_tokens() -> None:
    """Given equal commits, when only one field flips proposals, then that field is remasked."""
    # Given
    width = FIELD_ORDER.index("width")
    height = FIELD_ORDER.index("height")
    positions = (width, height)
    width_flips = _history_result(positions, first_tokens=(41, 42))
    height_flips = _history_result(positions, first_tokens=(42, 41))

    # When
    width_choice = choose_lineage_field(width_flips, outcomes=())
    height_choice = choose_lineage_field(height_flips, outcomes=())

    # Then
    assert width_flips.token_ids[width] == width_flips.token_ids[height] == 42
    assert height_flips.token_ids[width] == height_flips.token_ids[height] == 42
    assert width_choice == "width"
    assert height_choice == "height"


def test_true_margin_is_independent_of_peer_proposal_confidence() -> None:
    """Given one shared commit row, when margins differ, then peer confidence is irrelevant."""
    # Given
    width = FIELD_ORDER.index("width")
    height = FIELD_ORDER.index("height")
    result = _result(
        (
            _proposal_step(
                step_index=1,
                positions=(width, height),
                proposal_tokens=(42, 42),
                confidences=(0.6, 0.9),
                margins=(0.05, 0.4),
                commit=True,
            ),
        )
    )

    # When
    trajectories = {trajectory.field: trajectory for trajectory in field_trajectories(result)}

    # Then
    assert trajectories["width"].confidence_margin == pytest.approx(0.05)
    assert trajectories["height"].confidence_margin == pytest.approx(0.4)


def test_repeated_field_failures_reduce_reselection_pressure() -> None:
    """Given a high-pressure field fails, choosing again explores another field."""
    # Given
    positions = (FIELD_ORDER.index("width"), FIELD_ORDER.index("height"))
    result = _history_result(positions, first_tokens=(41, 42))
    failures = (
        LineageFieldOutcome(
            lineage_id="lineage-a",
            field="width",
            reward_delta=-1.0,
            trials=1,
            invalid_count=2,
            duplicate_count=1,
            proof_failure_count=1,
        ),
    )

    # When
    baseline = choose_lineage_field(result, outcomes=())
    after_failures = choose_lineage_field(result, failures, lineage_id="lineage-a")

    # Then
    assert baseline == "width"
    assert after_failures == "height"


def _history_result(positions: tuple[int, int], first_tokens: tuple[int, int]) -> RevisionResult:
    return _result(
        (
            _proposal_step(
                step_index=1,
                positions=positions,
                proposal_tokens=first_tokens,
                confidences=(0.8, 0.8),
                margins=(0.5, 0.5),
            ),
            _proposal_step(
                step_index=2,
                positions=positions,
                proposal_tokens=(42, 42),
                confidences=(0.8, 0.8),
                margins=(0.5, 0.5),
                commit=True,
            ),
        )
    )


def _proposal_step(  # noqa: PLR0913 - compact trajectory fixture.
    *,
    step_index: int,
    positions: tuple[int, ...],
    proposal_tokens: tuple[int, ...],
    confidences: tuple[float, ...],
    margins: tuple[float, ...],
    commit: bool = False,
) -> RevisionTrajectoryStep:
    return RevisionTrajectoryStep(
        step_index=step_index,
        block_index=0,
        masked_before=len(positions),
        committed_positions=positions if commit else (),
        token_ids=proposal_tokens if commit else (),
        confidences=confidences if commit else (),
        proposal_positions=positions,
        proposal_token_ids=proposal_tokens,
        proposal_confidences=confidences,
        proposal_margins=margins,
        remaining_masked=0 if commit else len(positions),
    )


def _result(steps: tuple[RevisionTrajectoryStep, ...]) -> RevisionResult:
    token_ids = tuple(42 for _ in FIELD_ORDER)
    return RevisionResult(
        selectors=(0, 1, 2, 3, 0, 0, 0, 0, 0, 1),
        token_ids=token_ids,
        slot_positions=tuple(range(len(FIELD_ORDER))),
        selected_slot_token_ids=token_ids,
        token_state_sha256="1" * 64,
        transfer_evidence_sha256="2" * 64,
        rng_state_sha256="3" * 64,
        nfe=len(steps),
        forward_calls=len(steps),
        gpu_ms=4,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="origin_x",
        trajectory_steps=steps,
        trajectory_evidence_sha256="4" * 64,
    )
