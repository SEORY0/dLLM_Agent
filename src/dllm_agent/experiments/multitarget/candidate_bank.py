"""Deterministic candidate-bank freezing, diversity, and remask planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.slot_sampler import SlotSample
    from dllm_agent.targets.structured import SnapshotLike, StructuredTarget

RAW_SAMPLE_COUNT: Final = 32
MINIMUM_UNIQUE_COUNT: Final = 16
MINIMUM_PSEUDO_OBSERVATIONS: Final = 2

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(slots=True)
class CandidateBankError(Exception):
    """Raised when candidate-bank construction or attribution cannot proceed."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the closed-bank failure reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class Candidate:
    """One unique materialized candidate with its denoising sample evidence."""

    sample: SlotSample
    snapshot: SnapshotLike
    source_sample_sha256: str


@dataclass(frozen=True, slots=True)
class DuplicateEvidence:
    """One raw sample that collapsed into a previously observed snapshot."""

    snapshot_sha256: str
    raw_sample_sha256: str
    kept_sample_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateBank:
    """Frozen unique candidate bank with raw and deduplicated hash evidence."""

    candidates: tuple[Candidate, ...]
    duplicates: tuple[DuplicateEvidence, ...]
    raw_count: int
    unique_count: int
    minimum_unique_count: int
    raw_trajectory_sha256: str
    unique_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    """Probe metrics bound to one candidate and, optionally, a deranged source."""

    candidate_sha256: str
    feedback_source_sha256: str
    canary_triggered: bool
    sanitizer_observed: bool
    coverage_edges: int
    value_profile_features: int
    executions_per_second: int
    log_sha256: str


def freeze_candidate_bank(
    target: StructuredTarget, samples: tuple[SlotSample, ...]
) -> CandidateBank:
    """Materialize exactly 32 raw samples into a canonical unique candidate bank."""
    if len(samples) != RAW_SAMPLE_COUNT:
        raise CandidateBankError(reason=f"expected exactly {RAW_SAMPLE_COUNT} raw samples")

    unique_by_snapshot: dict[str, Candidate] = {}
    duplicates: list[DuplicateEvidence] = []
    raw_hashes: list[str] = []
    for sample in samples:
        raw_hash = _sample_hash(sample)
        raw_hashes.append(raw_hash)
        snapshot = target.materialize(sample.selectors)
        kept = unique_by_snapshot.get(snapshot.sha256)
        if kept is None:
            unique_by_snapshot[snapshot.sha256] = Candidate(
                sample=sample,
                snapshot=snapshot,
                source_sample_sha256=raw_hash,
            )
        else:
            duplicates.append(
                DuplicateEvidence(
                    snapshot_sha256=snapshot.sha256,
                    raw_sample_sha256=raw_hash,
                    kept_sample_sha256=kept.source_sample_sha256,
                )
            )

    candidates = tuple(
        sorted(unique_by_snapshot.values(), key=lambda candidate: candidate.snapshot.sha256)
    )
    if len(candidates) < MINIMUM_UNIQUE_COUNT:
        raise CandidateBankError(
            reason=(f"candidate bank requires at least 16 unique snapshots; got {len(candidates)}")
        )

    return CandidateBank(
        candidates=candidates,
        duplicates=tuple(sorted(duplicates, key=_duplicate_key)),
        raw_count=len(samples),
        unique_count=len(candidates),
        minimum_unique_count=MINIMUM_UNIQUE_COUNT,
        raw_trajectory_sha256=_sha256_json(tuple(raw_hashes)),
        unique_snapshot_sha256=_sha256_json(
            tuple(candidate.snapshot.sha256 for candidate in candidates)
        ),
    )


def select_diverse(bank: CandidateBank, count: int) -> tuple[Candidate, ...]:
    """Select unique candidates by farthest-first selector Hamming distance."""
    if count < 1:
        raise CandidateBankError(reason="selection count must be positive")
    if count > bank.unique_count:
        raise CandidateBankError(reason="selection count exceeds unique candidates")

    remaining = list(bank.candidates)
    selected = [remaining.pop(0)]
    while len(selected) < count:
        next_index = min(
            range(len(remaining)),
            key=lambda index: (
                -_nearest_hamming(remaining[index], tuple(selected)),
                remaining[index].snapshot.sha256,
            ),
        )
        selected.append(remaining.pop(next_index))
    return tuple(selected)


def pseudo_feedback(
    observations: tuple[ProbeObservation, ...], *, seed: int
) -> tuple[ProbeObservation, ...]:
    """Rotate feedback metrics through a fixed-point-free seeded cycle."""
    if len(observations) < MINIMUM_PSEUDO_OBSERVATIONS:
        raise CandidateBankError(reason="pseudo-feedback requires at least two observations")
    shift = (seed % (len(observations) - 1)) + 1
    sources = observations[shift:] + observations[:shift]
    return tuple(
        ProbeObservation(
            candidate_sha256=target.candidate_sha256,
            feedback_source_sha256=source.candidate_sha256,
            canary_triggered=source.canary_triggered,
            sanitizer_observed=source.sanitizer_observed,
            coverage_edges=source.coverage_edges,
            value_profile_features=source.value_profile_features,
            executions_per_second=source.executions_per_second,
            log_sha256=source.log_sha256,
        )
        for target, source in zip(observations, sources, strict=True)
    )


def remask_plan(
    candidates: tuple[Candidate, ...],
    observations: tuple[ProbeObservation, ...],
    mutable_indexes: tuple[int, ...],
) -> tuple[int, ...]:
    """Choose one remask field per candidate from feedback association and uncertainty."""
    if not candidates:
        return ()
    if not mutable_indexes:
        raise CandidateBankError(reason="remask planning requires mutable indexes")
    selector_count = len(candidates[0].sample.selectors)
    for index in mutable_indexes:
        if index < 0 or index >= selector_count:
            raise CandidateBankError(reason=f"mutable index is out of bounds: {index}")
    observations_by_hash = _observations_by_hash(observations)
    for candidate in candidates:
        if candidate.snapshot.sha256 not in observations_by_hash:
            raise CandidateBankError(reason="missing observation for candidate hash")
    rewards = tuple(
        _reward(observations_by_hash[candidate.snapshot.sha256]) for candidate in candidates
    )
    associations = {
        index: _association(index, candidates, rewards)
        for index in tuple(dict.fromkeys(mutable_indexes))
    }
    return tuple(
        max(
            associations,
            key=lambda index: (
                associations[index],
                _uncertainty(candidate, index),
                -index,
            ),
        )
        for candidate in candidates
    )


def _sample_hash(sample: SlotSample) -> str:
    trajectory_steps: tuple[JsonValue, ...] = tuple(
        {
            "block_index": step.block_index,
            "committed_positions": step.committed_positions,
            "confidences": step.confidences,
            "masked_before": step.masked_before,
            "remaining_masked": step.remaining_masked,
            "state_sha256": step.state_sha256,
            "step_index": step.step_index,
            "token_ids": step.token_ids,
        }
        for step in sample.trajectory_steps
    )
    payload: dict[str, JsonValue] = {
        "forward_calls": sample.forward_calls,
        "gpu_ms": sample.gpu_ms,
        "nfe": sample.nfe,
        "parent_state_sha256": sample.parent_state_sha256,
        "requested_steps": sample.requested_steps,
        "selected_slot_token_ids": sample.selected_slot_token_ids,
        "selectors": sample.selectors,
        "slot_confidences": sample.slot_confidences,
        "slot_positions": sample.slot_positions,
        "state_sha256": sample.state_sha256,
        "token_ids": sample.token_ids,
        "trajectory_steps": trajectory_steps,
    }
    return _sha256_json(payload)


def _sha256_json(value: JsonValue) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _duplicate_key(duplicate: DuplicateEvidence) -> tuple[str, str, str]:
    return (
        duplicate.snapshot_sha256,
        duplicate.kept_sample_sha256,
        duplicate.raw_sample_sha256,
    )


def _nearest_hamming(candidate: Candidate, selected: tuple[Candidate, ...]) -> int:
    return min(_hamming(candidate.sample.selectors, item.sample.selectors) for item in selected)


def _hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(left_item != right_item for left_item, right_item in zip(left, right, strict=True))


def _observations_by_hash(
    observations: tuple[ProbeObservation, ...],
) -> dict[str, ProbeObservation]:
    bound: dict[str, ProbeObservation] = {}
    for observation in observations:
        if observation.candidate_sha256 in bound:
            raise CandidateBankError(reason="duplicate observation candidate hash")
        bound[observation.candidate_sha256] = observation
    return bound


def _reward(observation: ProbeObservation) -> float:
    return (
        (1_000_000_000.0 if observation.sanitizer_observed else 0.0)
        + (1_000_000.0 * observation.coverage_edges)
        + (1_000.0 * observation.value_profile_features)
        + observation.executions_per_second
    )


def _association(
    index: int, candidates: tuple[Candidate, ...], rewards: tuple[float, ...]
) -> float:
    by_selector: dict[int, list[float]] = {}
    for candidate, reward in zip(candidates, rewards, strict=True):
        by_selector.setdefault(candidate.sample.selectors[index], []).append(reward)
    means = tuple(sum(values) / len(values) for values in by_selector.values())
    return max(means) - min(means)


def _uncertainty(candidate: Candidate, index: int) -> float:
    return 1.0 - candidate.sample.slot_confidences[index]
