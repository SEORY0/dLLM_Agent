"""Pure stock-Stable trajectory feature extraction and stage tournament ranking."""

from itertools import pairwise
from statistics import median

from dllm_agent.hunt.models import HarnessCandidate
from dllm_agent.native.models import ProbeEvent, TrajectoryFeatures


def extract_trajectory_features(candidate: HarnessCandidate) -> TrajectoryFeatures:
    """Extract bounded stock-Stable features from observed steps without true remasking.

    weighted_confidence is sum(confidence * committed) / sum(committed),
    confidence_floor is min(confidence), and volatility is mean adjacent absolute
    confidence difference (zero for one step). commitment_auc is trapezoidal cumulative
    committed fraction over normalized observed-step progress from (0, 0), clamped at
    one using the first masked-token count. nfe_efficiency is 1 - min(nfe / 512, 1).
    trajectory_quality is their fixed 0.35/0.20/0.15/0.20/0.10 weighted combination.
    """
    steps = candidate.steps
    total_committed = sum(step.committed_tokens for step in steps)
    weighted_confidence = (
        sum(step.mean_confidence * step.committed_tokens for step in steps) / total_committed
    )
    confidence_floor = min(step.mean_confidence for step in steps)
    volatility = _volatility(candidate)
    commitment_auc = _commitment_auc(candidate)
    nfe_efficiency = 1.0 - min(candidate.nfe / 512.0, 1.0)
    trajectory_quality = (
        0.35 * weighted_confidence
        + 0.20 * confidence_floor
        + 0.15 * (1.0 - volatility)
        + 0.20 * commitment_auc
        + 0.10 * nfe_efficiency
    )
    return TrajectoryFeatures(
        candidate_id=candidate.candidate_id,
        weighted_confidence=round(weighted_confidence, 6),
        confidence_floor=round(confidence_floor, 6),
        volatility=round(volatility, 6),
        commitment_auc=round(commitment_auc, 6),
        nfe_efficiency=round(nfe_efficiency, 6),
        trajectory_quality=round(trajectory_quality, 6),
    )


def rank_stage(
    events: tuple[ProbeEvent, ...], features: tuple[TrajectoryFeatures, ...]
) -> tuple[str, ...]:
    """Rank event-bearing candidates by dynamic evidence then trajectory and identifier.

    Missing numeric metrics rank below an observed zero; feature-only candidates rank
    after all candidates with observed events, deterministically by identifier.
    """
    events_by_candidate: dict[str, list[ProbeEvent]] = {}
    for event in events:
        events_by_candidate.setdefault(event.candidate_id, []).append(event)
    quality_by_candidate: dict[str, float] = {}
    for feature in features:
        prior = quality_by_candidate.get(feature.candidate_id)
        if prior is None or feature.trajectory_quality > prior:
            quality_by_candidate[feature.candidate_id] = feature.trajectory_quality
    ranked_events = tuple(
        sorted(
            events_by_candidate,
            key=lambda candidate_id: _stage_sort_key(
                candidate_id,
                tuple(events_by_candidate[candidate_id]),
                quality_by_candidate.get(candidate_id, 0.0),
            ),
        )
    )
    feature_only = tuple(sorted(set(quality_by_candidate) - set(events_by_candidate)))
    return ranked_events + feature_only


def _volatility(candidate: HarnessCandidate) -> float:
    """Return mean adjacent confidence movement, with a one-step value of zero."""
    confidences = tuple(step.mean_confidence for step in candidate.steps)
    if len(confidences) == 1:
        return 0.0
    changes = tuple(abs(current - previous) for previous, current in pairwise(confidences))
    return sum(changes) / len(changes)


def _commitment_auc(candidate: HarnessCandidate) -> float:
    """Integrate clamped cumulative commitment over normalized observed-step progress."""
    denominator = candidate.steps[0].masked_tokens_before
    cumulative = 0
    previous_fraction = 0.0
    area = 0.0
    step_width = 1.0 / len(candidate.steps)
    for step in candidate.steps:
        cumulative += step.committed_tokens
        fraction = min(cumulative / denominator, 1.0)
        area += (previous_fraction + fraction) * step_width / 2.0
        previous_fraction = fraction
    return area


def _stage_sort_key(
    candidate_id: str, events: tuple[ProbeEvent, ...], trajectory_quality: float
) -> tuple[int | float | str, ...]:
    """Build the fixed lexicographic tournament key in its declared priority order."""
    branch_coverage = _median(
        tuple(event.coverage.branches for event in events if event.coverage is not None)
    )
    line_coverage = _median(
        tuple(event.coverage.lines for event in events if event.coverage is not None)
    )
    feature_coverage = _median(
        tuple(
            float(event.feature_coverage) for event in events if event.feature_coverage is not None
        )
    )
    edge_coverage = _median(
        tuple(float(event.edge_coverage) for event in events if event.edge_coverage is not None)
    )
    executions_per_second = _median(
        tuple(
            float(event.executions_per_second)
            for event in events
            if event.executions_per_second is not None
        )
    )
    build_success_rate = sum(event.build_status == "succeeded" for event in events) / len(events)
    return (
        0 if any(event.sanitizer_observed for event in events) else 1,
        -build_success_rate,
        *_descending_optional(branch_coverage),
        *_descending_optional(line_coverage),
        *_descending_optional(feature_coverage),
        *_descending_optional(edge_coverage),
        -trajectory_quality,
        *_descending_optional(executions_per_second),
        candidate_id,
    )


def _median(values: tuple[float, ...]) -> float | None:
    """Return a deterministic median while preserving missing-observation semantics."""
    if not values:
        return None
    return float(median(values))


def _descending_optional(value: float | None) -> tuple[int, float]:
    """Order observed numbers before missing values, both in descending order."""
    if value is None:
        return (0, 0.0)
    return (-1, -value)
