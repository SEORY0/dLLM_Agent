# pyright: reportUnnecessaryComparison=false
"""Pure parent-child Pareto archive selection for revision lineages."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never, override

from dllm_agent.revision.controller_models import ProbeRecord

type ArchiveCandidateKey = tuple[str, str]
type ProofNoveltyKey = ArchiveCandidateKey | str


@dataclass(frozen=True, slots=True)
class ArchiveCandidateCost:
    """Cumulative search cost for one archive candidate."""

    nfe: int
    forward_calls: int
    gpu_ms: int


class ProofNovelty(StrEnum):
    """Proof-cause novelty available to archive selection."""

    NONE = "none"
    DUPLICATE_CAUSE = "duplicate_cause"
    NEW_CAUSE = "new_cause"


@dataclass(frozen=True, slots=True)
class ArchiveCandidate:
    """One probed particle plus deterministic archive-ranking annotations."""

    record: ProbeRecord
    proof_novelty: ProofNovelty = ProofNovelty.NONE
    cost: ArchiveCandidateCost | None = None

    @property
    def key(self) -> ArchiveCandidateKey:
        """Return the lineage-local archive identity."""
        return (self.particle_id, self.snapshot_sha256)

    @property
    def particle_id(self) -> str:
        """Return the candidate particle identifier."""
        return self.record.particle.particle_id

    @property
    def snapshot_sha256(self) -> str:
        """Return the materialized snapshot hash used for stable tie-breaking."""
        return self.record.particle.snapshot.sha256


@dataclass(frozen=True, slots=True)
class ArchiveSelection:
    """Deterministic active archive and dropped candidate set."""

    active: tuple[ArchiveCandidate, ...]
    dropped: tuple[ArchiveCandidate, ...]
    protected_keys: tuple[ArchiveCandidateKey, ...]

    @property
    def active_ids(self) -> tuple[str, ...]:
        """Return active particle identifiers in archive order."""
        return tuple(candidate.particle_id for candidate in self.active)

    @property
    def active_keys(self) -> tuple[ArchiveCandidateKey, ...]:
        """Return active candidate keys in archive order."""
        return tuple(candidate.key for candidate in self.active)

    @property
    def dropped_ids(self) -> tuple[str, ...]:
        """Return dropped particle identifiers in deterministic order."""
        return tuple(candidate.particle_id for candidate in self.dropped)

    @property
    def dropped_keys(self) -> tuple[ArchiveCandidateKey, ...]:
        """Return dropped candidate keys in deterministic order."""
        return tuple(candidate.key for candidate in self.dropped)

    @property
    def protected_ids(self) -> tuple[str, ...]:
        """Return protected particle identifiers as a legacy projection."""
        return tuple(key[0] for key in self.protected_keys)


def select_parent_child(  # noqa: PLR0913 - archive selection needs explicit policy inputs.
    parents: tuple[ProbeRecord, ...],
    children: tuple[ProbeRecord, ...],
    *,
    capacity: int,
    protected_ids: frozenset[ArchiveCandidateKey],
    proof_novelty_by_key: Mapping[ProofNoveltyKey, ProofNovelty] | None = None,
    cost_by_key: Mapping[ArchiveCandidateKey, ArchiveCandidateCost] | None = None,
) -> ArchiveSelection:
    """Select a deterministic parent-child non-dominated archive."""
    if capacity < 1:
        raise ArchiveCapacityError(capacity=capacity, protected_count=0)
    novelty_by_key: Mapping[ProofNoveltyKey, ProofNovelty] = (
        {} if proof_novelty_by_key is None else proof_novelty_by_key
    )
    costs_by_key: Mapping[ArchiveCandidateKey, ArchiveCandidateCost] = (
        {} if cost_by_key is None else cost_by_key
    )
    candidates = _candidates((*parents, *children), novelty_by_key, costs_by_key)
    protected_count = sum(candidate.key in protected_ids for candidate in candidates)
    if protected_count > capacity:
        raise ArchiveCapacityError(capacity=capacity, protected_count=protected_count)
    frontier = _frontier(candidates, protected_ids)
    active = _truncate(frontier, capacity, protected_ids)
    active_keys = frozenset(candidate.key for candidate in active)
    dropped = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.key not in active_keys),
            key=_archive_order_key,
        )
    )
    kept_protected_keys = tuple(
        sorted(candidate.key for candidate in active if candidate.key in protected_ids)
    )
    return ArchiveSelection(
        active=active,
        dropped=dropped,
        protected_keys=kept_protected_keys,
    )


@dataclass(frozen=True, slots=True)
class ArchiveCapacityError(ValueError):
    """Raised when archive capacity cannot keep any candidate."""

    capacity: int
    protected_count: int

    @override
    def __str__(self) -> str:
        """Return a user-facing capacity error."""
        if self.protected_count > self.capacity:
            return (
                f"protected candidate count {self.protected_count} "
                f"exceeds archive capacity {self.capacity}"
            )
        return f"archive capacity must be positive, got {self.capacity}"


def _candidates(
    records: tuple[ProbeRecord, ...],
    novelty_by_key: Mapping[ProofNoveltyKey, ProofNovelty],
    cost_by_key: Mapping[ArchiveCandidateKey, ArchiveCandidateCost],
) -> tuple[ArchiveCandidate, ...]:
    return tuple(
        ArchiveCandidate(
            record=record,
            proof_novelty=_proof_novelty(record, novelty_by_key),
            cost=cost_by_key.get(_candidate_key(record)),
        )
        for record in records
    )


def _proof_novelty(
    record: ProbeRecord, novelty_by_key: Mapping[ProofNoveltyKey, ProofNovelty]
) -> ProofNovelty:
    snapshot_sha256 = record.particle.snapshot.sha256
    return novelty_by_key.get(
        (record.particle.particle_id, snapshot_sha256),
        novelty_by_key.get(snapshot_sha256, ProofNovelty.NONE),
    )


def _frontier(
    candidates: tuple[ArchiveCandidate, ...], protected_ids: frozenset[ArchiveCandidateKey]
) -> tuple[ArchiveCandidate, ...]:
    return tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.key in protected_ids or _is_not_dominated(candidate, candidates)
            ),
            key=_archive_order_key,
        )
    )


def _truncate(
    candidates: tuple[ArchiveCandidate, ...],
    capacity: int,
    protected_ids: frozenset[ArchiveCandidateKey],
) -> tuple[ArchiveCandidate, ...]:
    protected = tuple(candidate for candidate in candidates if candidate.key in protected_ids)
    open_slots = capacity - len(protected)
    if open_slots == 0:
        return tuple(sorted(protected, key=_archive_order_key))
    selected_keys = frozenset(candidate.key for candidate in protected)
    non_protected = tuple(
        candidate for candidate in candidates if candidate.key not in selected_keys
    )
    fill = _crowding_truncate(non_protected, open_slots)
    return tuple(sorted((*protected, *fill), key=_archive_order_key))


def _is_not_dominated(
    candidate: ArchiveCandidate, candidates: tuple[ArchiveCandidate, ...]
) -> bool:
    return not any(_dominates(other, candidate) for other in candidates if other != candidate)


def _dominates(left: ArchiveCandidate, right: ArchiveCandidate) -> bool:
    left_vector = _objective_vector(left)
    right_vector = _objective_vector(right)
    return all(
        left_value >= right_value
        for left_value, right_value in zip(left_vector, right_vector, strict=True)
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(left_vector, right_vector, strict=True)
    )


def _objective_vector(candidate: ArchiveCandidate) -> tuple[int, ...]:
    feedback = candidate.record.feedback
    cost = _candidate_cost(candidate)
    return (
        _proof_novelty_score(candidate.proof_novelty),
        int(feedback.valid_snapshot),
        feedback.target_branches_hit or 0,
        feedback.rare_edges,
        feedback.new_edges,
        feedback.value_profile_features,
        feedback.executions_per_second or 0,
        -cost.nfe,
        -cost.forward_calls,
        -cost.gpu_ms,
        -feedback.wall_ms,
        -feedback.rejected_count,
        -feedback.dedup_count,
    )


def _candidate_cost(candidate: ArchiveCandidate) -> ArchiveCandidateCost:
    if candidate.cost is not None:
        return candidate.cost
    result = candidate.record.particle.result
    return ArchiveCandidateCost(
        nfe=result.nfe,
        forward_calls=result.forward_calls,
        gpu_ms=result.gpu_ms,
    )


def _candidate_key(record: ProbeRecord) -> ArchiveCandidateKey:
    return (record.particle.particle_id, record.particle.snapshot.sha256)


def _proof_novelty_score(novelty: ProofNovelty) -> int:
    match novelty:
        case ProofNovelty.NEW_CAUSE:
            return 2
        case ProofNovelty.DUPLICATE_CAUSE:
            return 0
        case ProofNovelty.NONE:
            return 0
        case unreachable:
            assert_never(unreachable)


def _crowding_truncate(
    candidates: tuple[ArchiveCandidate, ...], capacity: int
) -> tuple[ArchiveCandidate, ...]:
    if len(candidates) <= capacity:
        return candidates
    distances = _crowding_distances(candidates)
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -distances[candidate.key],
                candidate.snapshot_sha256,
                candidate.particle_id,
            ),
        )[:capacity]
    )


def _crowding_distances(
    candidates: tuple[ArchiveCandidate, ...],
) -> dict[ArchiveCandidateKey, float]:
    distances = {candidate.key: 0.0 for candidate in candidates}
    objective_count = len(_objective_vector(candidates[0]))
    for objective_index in range(objective_count):
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                _objective_vector(candidate)[objective_index],
                candidate.snapshot_sha256,
                candidate.particle_id,
            ),
        )
        low = _objective_vector(ordered[0])[objective_index]
        high = _objective_vector(ordered[-1])[objective_index]
        if high == low:
            continue
        distances[ordered[0].key] = float("inf")
        distances[ordered[-1].key] = float("inf")
        span = float(high - low)
        for position in range(1, len(ordered) - 1):
            previous_value = _objective_vector(ordered[position - 1])[objective_index]
            next_value = _objective_vector(ordered[position + 1])[objective_index]
            distances[ordered[position].key] += float(next_value - previous_value) / span
    return distances


def _archive_order_key(candidate: ArchiveCandidate) -> tuple[str, str]:
    return candidate.key
