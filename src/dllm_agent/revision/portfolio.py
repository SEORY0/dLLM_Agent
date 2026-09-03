"""Immutable per-lineage plateau tracking for online revision."""

from typing import Final

from pydantic import Field

from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.revision.models import ProbeFeedback

type _RewardKey = tuple[bool, int, int, int, int, int]


class LineageReward(FrozenModel):
    """Search-side reward vector used to detect lineage improvement."""

    valid_snapshot: bool
    target_branches_hit: int = Field(ge=0)
    rare_edges: int = Field(ge=0)
    new_edges: int = Field(ge=0)
    value_profile_features: int = Field(ge=0)
    executions_per_second: int = Field(ge=0)


class ProofFamilyAttempt(FrozenModel):
    """Revision/proof budget consumed by one observed proof family."""

    family_id: str = Field(min_length=1)
    attempts: int = Field(ge=0)


class LineageState(FrozenModel):
    """Immutable portfolio state for one parent-child lineage."""

    lineage_id: str = Field(min_length=1)
    latest_feedback_sha256: Sha256 | None = None
    best_feedback_sha256: Sha256 | None = None
    best_reward: LineageReward | None = None
    consecutive_non_improving_epochs: int = Field(default=0, ge=0)
    active_proof_family_id: str | None = None
    proof_family_attempts: tuple[ProofFamilyAttempt, ...] = ()


class PortfolioPolicy(FrozenModel):
    """Plateau threshold and per-family budget for revision work."""

    plateau_epochs: int = Field(gt=0)
    proof_family_attempt_cap: int = Field(gt=0)


def update_lineage(
    state: LineageState,
    feedback: ProbeFeedback,
    *,
    proof_family_id: str | None = None,
    proof_attempted: bool = False,
) -> LineageState:
    """Return updated immutable lineage state after one epoch barrier."""
    reward = _reward_from_feedback(feedback)
    improved = state.best_reward is None or _dominates(reward, state.best_reward)
    attempts = _record_family_attempt(state.proof_family_attempts, proof_family_id, proof_attempted)
    if improved:
        return LineageState(
            lineage_id=state.lineage_id,
            latest_feedback_sha256=feedback.feedback_sha256,
            best_feedback_sha256=feedback.feedback_sha256,
            best_reward=reward,
            consecutive_non_improving_epochs=0,
            active_proof_family_id=proof_family_id or state.active_proof_family_id,
            proof_family_attempts=attempts,
        )
    return LineageState(
        lineage_id=state.lineage_id,
        latest_feedback_sha256=feedback.feedback_sha256,
        best_feedback_sha256=state.best_feedback_sha256,
        best_reward=state.best_reward,
        consecutive_non_improving_epochs=state.consecutive_non_improving_epochs + 1,
        active_proof_family_id=proof_family_id or state.active_proof_family_id,
        proof_family_attempts=attempts,
    )


def revision_due(state: LineageState, policy: PortfolioPolicy) -> bool:
    """Return whether the lineage is plateaued and not capped."""
    if state.consecutive_non_improving_epochs < policy.plateau_epochs:
        return False
    return not _active_family_is_capped(state, policy)


def _reward_from_feedback(feedback: ProbeFeedback) -> LineageReward:
    return LineageReward(
        valid_snapshot=feedback.valid_snapshot,
        target_branches_hit=feedback.target_branches_hit or 0,
        rare_edges=feedback.rare_edges,
        new_edges=feedback.new_edges,
        value_profile_features=feedback.value_profile_features,
        executions_per_second=feedback.executions_per_second or 0,
    )


def _reward_key(reward: LineageReward) -> _RewardKey:
    return (
        reward.valid_snapshot,
        reward.target_branches_hit,
        reward.rare_edges,
        reward.new_edges,
        reward.value_profile_features,
        reward.executions_per_second,
    )


def _dominates(candidate: LineageReward, incumbent: LineageReward) -> bool:
    candidate_key = _reward_key(candidate)
    incumbent_key = _reward_key(incumbent)
    return all(
        candidate_axis >= incumbent_axis
        for candidate_axis, incumbent_axis in zip(candidate_key, incumbent_key, strict=True)
    ) and any(
        candidate_axis > incumbent_axis
        for candidate_axis, incumbent_axis in zip(candidate_key, incumbent_key, strict=True)
    )


def _record_family_attempt(
    attempts: tuple[ProofFamilyAttempt, ...],
    family_id: str | None,
    attempted: bool,
) -> tuple[ProofFamilyAttempt, ...]:
    if family_id is None or not attempted:
        return attempts
    family_attempts = tuple(item for item in attempts if item.family_id == family_id)
    other_attempts = tuple(item for item in attempts if item.family_id != family_id)
    previous = family_attempts[0].attempts if len(family_attempts) == 1 else 0
    return (*other_attempts, ProofFamilyAttempt(family_id=family_id, attempts=previous + 1))


def _active_family_is_capped(state: LineageState, policy: PortfolioPolicy) -> bool:
    family_id = state.active_proof_family_id
    if family_id is None:
        return False
    attempts_by_family: Final = {
        attempt.family_id: attempt.attempts for attempt in state.proof_family_attempts
    }
    return attempts_by_family.get(family_id, 0) >= policy.proof_family_attempt_cap
