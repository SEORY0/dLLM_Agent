"""Deterministic feedback attribution and remask selection policy."""

from dataclasses import dataclass
from math import log, sqrt
from typing import Final, Literal, override

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.revision.models import FieldAttribution, ProbeFeedback, RemaskDecision
from dllm_agent.targets.openexr.particle import MATERIALIZED_MUTABLE_FIELDS, FieldName

type DecisionReason = Literal["ucb", "lexicographic_feedback"]
type RewardKey = tuple[int, int, int, int, bool, int]

REMASKABLE_FIELDS: Final[tuple[FieldName, ...]] = MATERIALIZED_MUTABLE_FIELDS
_REMASKABLE_FIELD_SET: Final = frozenset(REMASKABLE_FIELDS)


@dataclass(frozen=True, slots=True)
class FieldAttributionError(ValueError):
    """Raised when feedback credit is not a controlled one-field delta."""

    field_count: int

    @override
    def __str__(self) -> str:
        """Return a concise attribution failure."""
        return f"field attribution requires exactly one changed field, got {self.field_count}"


@dataclass(frozen=True, slots=True)
class _FeedbackCandidate:
    field: FieldName
    feedback_sha256: Sha256
    reward: RewardKey


class _DecisionHashInput(FrozenModel):
    """Canonical decision payload hashed into RemaskDecision evidence."""

    epoch: int
    selected_field: FieldName
    reason: DecisionReason
    latest_feedback_sha256: tuple[Sha256, ...]
    attributions: tuple[FieldAttribution, ...]
    total_trials: int
    rejected_count: int
    dedup_count: int


def attribute_feedback(
    parent: ProbeFeedback,
    child: ProbeFeedback,
    changed: tuple[FieldName, ...],
) -> tuple[FieldAttribution, ...]:
    """Credit reward delta to the controlled changed field group."""
    if len(changed) != 1:
        raise FieldAttributionError(field_count=len(changed))
    delta = _reward_delta(parent, child)
    field = changed[0]
    return (
        FieldAttribution(
            field=field,
            parent_feedback_sha256=parent.feedback_sha256,
            child_feedback_sha256=child.feedback_sha256,
            reward_delta=delta,
            trials=1,
            rejected_count=child.rejected_count,
            dedup_count=child.dedup_count,
        ),
    )


def decide_remask(
    history: tuple[FieldAttribution, ...],
    epoch: int,
    latest_feedback: tuple[ProbeFeedback, ...] = (),
    prefer_untried: bool = True,
) -> RemaskDecision:
    """Choose the next field by feedback priority, UCB score, and field-name ties."""
    feedback_choice = _feedback_field(history, latest_feedback)
    if feedback_choice is not None:
        reason: DecisionReason = "lexicographic_feedback"
        field = feedback_choice
    else:
        reason = "ucb"
        field = _ucb_field(history, prefer_untried)
    total_trials = sum(attribution.trials for attribution in history)
    rejected_count = sum(attribution.rejected_count for attribution in history)
    dedup_count = sum(attribution.dedup_count for attribution in history)
    decision_hash_payload = _DecisionHashInput(
        epoch=epoch,
        selected_field=field,
        reason=reason,
        latest_feedback_sha256=tuple(feedback.feedback_sha256 for feedback in latest_feedback),
        attributions=history,
        total_trials=total_trials,
        rejected_count=rejected_count,
        dedup_count=dedup_count,
    )
    decision_hash = sha256_bytes(decision_hash_payload.model_dump_json().encode())
    return RemaskDecision(
        sequence=epoch,
        epoch=epoch,
        fields=(field,),
        decision_sha256=decision_hash,
        reason=reason,
        total_trials=total_trials,
        rejected_count=rejected_count,
        dedup_count=dedup_count,
    )


def _feedback_field(
    history: tuple[FieldAttribution, ...],
    feedback: tuple[ProbeFeedback, ...],
) -> FieldName | None:
    if len(feedback) == 0:
        return None
    candidates = _feedback_candidates(history, feedback)
    if len(candidates) == 0:
        return None
    best = max(candidates, key=lambda candidate: (candidate.reward, _tie_name(candidate.field)))
    return best.field


def _feedback_candidates(
    history: tuple[FieldAttribution, ...],
    feedback: tuple[ProbeFeedback, ...],
) -> tuple[_FeedbackCandidate, ...]:
    return tuple(
        _FeedbackCandidate(
            field=attribution.field,
            feedback_sha256=probe_feedback.feedback_sha256,
            reward=_reward_key(probe_feedback),
        )
        for attribution in history
        for probe_feedback in feedback
        if attribution.child_feedback_sha256 == probe_feedback.feedback_sha256
        if _is_remaskable(attribution.field)
        if attribution.reward_delta > 0.0
    )


def _ucb_field(history: tuple[FieldAttribution, ...], prefer_untried: bool) -> FieldName:
    remaskable_history = tuple(item for item in history if _is_remaskable(item.field))
    tried_fields = {item.field for item in remaskable_history}
    if prefer_untried:
        for field in REMASKABLE_FIELDS:
            if field not in tried_fields:
                return field
    if len(remaskable_history) == 0:
        return REMASKABLE_FIELDS[0]
    best = max(
        remaskable_history,
        key=lambda attribution: (
            _ucb_score(remaskable_history, attribution),
            _tie_name(attribution.field),
        ),
    )
    return best.field


def _ucb_score(history: tuple[FieldAttribution, ...], attribution: FieldAttribution) -> float:
    total_trials = max(1, sum(item.trials for item in history))
    return attribution.reward_delta + sqrt(2.0 * log(total_trials) / attribution.trials)


def _tie_name(field: FieldName) -> str:
    return "".join(chr(255 - ord(character)) for character in field)


def _is_remaskable(field: FieldName) -> bool:
    return field in _REMASKABLE_FIELD_SET


def _reward_delta(parent: ProbeFeedback, child: ProbeFeedback) -> float:
    parent_key = _reward_key(parent)
    child_key = _reward_key(child)
    if child_key > parent_key:
        return 1.0
    if child_key < parent_key:
        return -1.0
    return 0.0


def _reward_key(feedback: ProbeFeedback) -> RewardKey:
    return (
        feedback.target_branches_hit or 0,
        feedback.rare_edges,
        feedback.new_edges,
        feedback.value_profile_features,
        feedback.valid_snapshot,
        feedback.executions_per_second or 0,
    )
