"""Per-lineage trajectory-aware field selection."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Final, override

from pydantic import Field

from dllm_agent.core.models import FrozenModel, Probability
from dllm_agent.revision.controller_models import RevisionResult, RevisionTrajectoryStep
from dllm_agent.targets.openexr.particle import FIELD_ORDER, MATERIALIZED_MUTABLE_FIELDS, FieldName

type _FieldScore = tuple[float, str]

_REMASKABLE_FIELDS: Final = MATERIALIZED_MUTABLE_FIELDS


@dataclass(frozen=True, slots=True)
class LineageOutcomeScopeError(ValueError):
    """Raised when field outcomes are supplied without a lineage scope."""

    outcome_count: int

    @override
    def __str__(self) -> str:
        """Return a concise scope failure."""
        return f"lineage_id is required when {self.outcome_count} field outcomes are provided"


class FieldTrajectory(FrozenModel):
    """Trajectory features observed for one materialized field."""

    field: FieldName
    committed_step_index: int | None = Field(default=None, ge=1)
    normalized_commit_step: Probability
    mean_confidence: Probability
    confidence_margin: float = Field(ge=-1.0, le=1.0)
    token_flip_count: int = Field(ge=0)
    mask_lifespan: int = Field(ge=0)
    remask_pressure: float = Field(ge=0.0)


class LineageFieldOutcome(FrozenModel):
    """Observed reward history for changing one field in one lineage."""

    lineage_id: str = Field(min_length=1)
    field: FieldName
    reward_delta: float
    trials: int = Field(gt=0)
    invalid_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    proof_failure_count: int = Field(default=0, ge=0)


@dataclass(frozen=True, slots=True)
class _ProposalObservation:
    step_index: int
    confidence: float
    confidence_margin: float
    token_id: int
    committed: bool


def field_trajectories(result: RevisionResult) -> tuple[FieldTrajectory, ...]:
    """Summarize real commit timing and confidence for remaskable fields."""
    max_step = max((step.step_index for step in result.trajectory_steps), default=1)
    return tuple(_field_trajectory(result, field, max_step) for field in _REMASKABLE_FIELDS)


def choose_lineage_field(
    result: RevisionResult,
    outcomes: tuple[LineageFieldOutcome, ...],
    lineage_id: str | None = None,
) -> FieldName:
    """Choose the next remask field for one lineage from trajectory and outcome evidence."""
    lineage_outcomes = _lineage_outcomes(outcomes, lineage_id)
    best = max(
        field_trajectories(result),
        key=lambda trajectory: _field_score(trajectory, lineage_outcomes),
    )
    return best.field


def _field_trajectory(result: RevisionResult, field: FieldName, max_step: int) -> FieldTrajectory:
    field_index = FIELD_ORDER.index(field)
    position = result.slot_positions[field_index]
    proposals = _proposals_for_position(result.trajectory_steps, position)
    if len(proposals) == 0:
        return FieldTrajectory(
            field=field,
            committed_step_index=None,
            normalized_commit_step=0.0,
            mean_confidence=1.0,
            confidence_margin=0.0,
            token_flip_count=0,
            mask_lifespan=0,
            remask_pressure=0.0,
        )
    commit = next((proposal for proposal in proposals if proposal.committed), None)
    mean_confidence = sum(proposal.confidence for proposal in proposals) / len(proposals)
    mean_margin = sum(proposal.confidence_margin for proposal in proposals) / len(proposals)
    normalized_commit_step = 0.0 if commit is None else commit.step_index / max_step
    token_flip_count = sum(
        previous.token_id != current.token_id for previous, current in pairwise(proposals)
    )
    mask_lifespan = len(proposals)
    pressure = (
        normalized_commit_step
        + (1.0 - mean_confidence)
        + ((1.0 - mean_margin) / 2.0)
        + (token_flip_count * 0.25)
        + (mask_lifespan / max_step)
    )
    return FieldTrajectory(
        field=field,
        committed_step_index=None if commit is None else commit.step_index,
        normalized_commit_step=normalized_commit_step,
        mean_confidence=mean_confidence,
        confidence_margin=mean_margin,
        token_flip_count=token_flip_count,
        mask_lifespan=mask_lifespan,
        remask_pressure=pressure,
    )


def _proposals_for_position(
    steps: tuple[RevisionTrajectoryStep, ...], position: int
) -> tuple[_ProposalObservation, ...]:
    proposals: list[_ProposalObservation] = []
    for step in sorted(steps, key=lambda item: (item.step_index, item.block_index)):
        if position in step.proposal_positions:
            position_index = step.proposal_positions.index(position)
            proposals.append(
                _ProposalObservation(
                    step_index=step.step_index,
                    confidence=step.proposal_confidences[position_index],
                    confidence_margin=step.proposal_margins[position_index],
                    token_id=step.proposal_token_ids[position_index],
                    committed=position in step.committed_positions,
                )
            )
    return tuple(proposals)


def _lineage_outcomes(
    outcomes: tuple[LineageFieldOutcome, ...], lineage_id: str | None
) -> tuple[LineageFieldOutcome, ...]:
    if lineage_id is None:
        if len(outcomes) > 0:
            raise LineageOutcomeScopeError(outcome_count=len(outcomes))
        return outcomes
    return tuple(outcome for outcome in outcomes if outcome.lineage_id == lineage_id)


def _field_score(
    trajectory: FieldTrajectory, outcomes: tuple[LineageFieldOutcome, ...]
) -> _FieldScore:
    return (
        trajectory.remask_pressure + _outcome_pressure(trajectory.field, outcomes),
        _tie_name(trajectory.field),
    )


def _outcome_pressure(field: FieldName, outcomes: tuple[LineageFieldOutcome, ...]) -> float:
    pressure = 0.0
    for outcome in outcomes:
        if outcome.field == field:
            pressure += _single_outcome_pressure(outcome)
    return pressure


def _single_outcome_pressure(outcome: LineageFieldOutcome) -> float:
    average_delta = outcome.reward_delta / outcome.trials
    failed_work = outcome.invalid_count + outcome.duplicate_count + outcome.proof_failure_count
    return -((abs(average_delta) * 2.0) + (failed_work * 0.25))


def _tie_name(field: FieldName) -> str:
    return "".join(chr(255 - ord(character)) for character in field)
