"""Shared candidate-bank contracts for the OpenEXR one-day experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple, override

from tools.selective_revision_payloads import sha256_token_ids
from tools.stable_diffcoder_generation import (
    DuplicateParent,
    FrozenParent,
    revision_trajectory_sha256,
)

from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS, ModelArm, SeedNamespace
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    RevisionRequest,
    RevisionResult,
    RevisionSampler,
)
from dllm_agent.runtime.revision_process import (
    ArmBankCostClaim,
    BankAccounting,
    BankGenerationCost,
    SelectedLineage,
)
from dllm_agent.targets.openexr.particle import decode_particle, materialize_particle

if TYPE_CHECKING:
    from pathlib import Path

RAW_PARENT_COUNT: Final = 32
SUPPLEMENT_PARENT_COUNT: Final = 16
SELECTED_PARENT_COUNT: Final = 16


@dataclass(frozen=True, slots=True)
class CandidateBankError(Exception):
    """Raised when the shared candidate bank cannot be frozen safely."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CandidateBank:
    """Frozen shared raw evidence, selected parents, and cost accounting."""

    raw_parents: tuple[FrozenParent, ...]
    selected_parents: tuple[FrozenParent, ...]
    requested_seed_base: int
    duplicates: tuple[DuplicateParent, ...]
    primary_raw_count: int
    supplement_raw_count: int
    unique_parent_count: int
    raw_token_states_sha256: str
    raw_trajectories_sha256: str
    raw_materialized_sha256: str
    duplicate_evidence_sha256: str
    selected_bank_sha256: str
    physical_cost: BankGenerationCost

    @property
    def selected_lineages(self) -> tuple[SelectedLineage, ...]:  # noqa: D102
        return _selected_lineages(self.selected_parents, self.requested_seed_base)

    @property
    def accounting(self) -> BankAccounting:  # noqa: D102
        selected_cost = _generation_cost(self.selected_parents)
        claims = tuple(
            ArmBankCostClaim(
                _hash_parts((self.selected_bank_sha256, arm, "task5-bank-adjustment")),
                arm,
                selected_cost,
                _subtract_cost(self.physical_cost, selected_cost),
                self.physical_cost,
            )
            for arm in MODEL_ARMS
        )
        return BankAccounting(self.physical_cost, claims)


class ArmBankBinding(NamedTuple):
    """Private revision delegate and mutable corpus root for one model arm."""

    arm: ModelArm
    revise_delegate: RevisionSampler
    corpus_root: Path


@dataclass(frozen=True, slots=True)
class FrozenBankAdapter:
    """Revision sampler exposing shared initials and one private revise delegate."""

    arm: ModelArm
    initial_parents: tuple[FrozenParent, ...]
    requested_seed_base: int
    revise_delegate: RevisionSampler
    corpus_root: Path

    def initial(self, request: InitialParticleRequest) -> RevisionResult:  # noqa: D102
        if request.arm != self.arm:
            raise CandidateBankError(reason="initial request arm does not match frozen adapter")
        lineages = _selected_lineages(self.initial_parents, self.requested_seed_base)
        for lineage, parent in zip(lineages, self.initial_parents, strict=True):
            if request.particle_id == lineage.particle_id:
                if request.seed != lineage.requested_seed or request.steps != lineage.steps:
                    raise CandidateBankError(reason="initial lineage seed or steps mismatch")
                return parent.result
        raise CandidateBankError(reason=f"unknown frozen initial lineage: {request.particle_id}")

    def revise(self, request: RevisionRequest) -> RevisionResult:  # noqa: D102
        return self.revise_delegate.revise(request)


@dataclass(frozen=True, slots=True)
class _BankGenerator:
    sampler: RevisionSampler
    seed_namespace: SeedNamespace
    steps: int

    def parent(self, offset: int) -> FrozenParent:
        seed = int(self.seed_namespace.start_seed) + offset
        if seed >= int(self.seed_namespace.stop_seed_exclusive):
            raise CandidateBankError(reason="candidate-bank seeds exceed repetition namespace")
        request = InitialParticleRequest(
            particle_id=f"bank-r{int(self.seed_namespace.repetition):02d}-{offset:02d}",
            seed=seed,
            steps=self.steps,
            arm="structured-one-shot",
            mode="initial",
        )
        result = self.sampler.initial(request)
        if result.token_state_sha256 != sha256_token_ids(result.token_ids):
            raise CandidateBankError(reason="initial parent token-state hash mismatch")
        if result.trajectory_evidence_sha256 != revision_trajectory_sha256(result):
            raise CandidateBankError(reason="initial parent trajectory evidence hash mismatch")
        snapshot = materialize_particle(decode_particle(result.selectors, "validity-preserving"))
        raw_parent_sha256 = _hash_parts(
            (
                request.particle_id,
                request.seed,
                result.token_state_sha256,
                result.trajectory_evidence_sha256,
                snapshot.sha256,
            )
        )
        return FrozenParent(request, result, snapshot.content, snapshot.sha256, raw_parent_sha256)


def build_candidate_bank(
    sampler: RevisionSampler,
    seed_namespace: SeedNamespace,
    *,
    steps: int,
) -> CandidateBank:
    """Generate and freeze the shared 32-to-16 initial parent bank."""
    if steps < 1:
        raise CandidateBankError(reason="candidate-bank steps must be positive")
    generator = _BankGenerator(sampler=sampler, seed_namespace=seed_namespace, steps=steps)
    primary = tuple(generator.parent(offset) for offset in range(RAW_PARENT_COUNT))
    unique, duplicates = _deduplicate(primary)
    supplement: tuple[FrozenParent, ...] = ()
    if len(unique) < SELECTED_PARENT_COUNT:
        supplement = tuple(
            generator.parent(offset)
            for offset in range(RAW_PARENT_COUNT, RAW_PARENT_COUNT + SUPPLEMENT_PARENT_COUNT)
        )
        unique, duplicates = _deduplicate(primary + supplement)
    if len(unique) < SELECTED_PARENT_COUNT:
        raise CandidateBankError(reason=f"need at least 16 unique: {len(unique)}")

    raw_parents = primary + supplement
    selected = _select_diverse(unique, SELECTED_PARENT_COUNT)
    physical_cost = _generation_cost(raw_parents)
    return CandidateBank(
        raw_parents,
        selected,
        int(seed_namespace.start_seed),
        duplicates,
        len(primary),
        len(supplement),
        len(unique),
        _hash_parts(tuple(parent.result.token_state_sha256 for parent in raw_parents)),
        _hash_parts(tuple(parent.result.trajectory_evidence_sha256 for parent in raw_parents)),
        _hash_parts(tuple(parent.materialized_sha256 for parent in raw_parents)),
        _duplicate_digest(duplicates),
        _hash_parts(tuple(parent.raw_parent_sha256 for parent in selected)),
        physical_cost,
    )


def build_frozen_bank_adapters(
    bank: CandidateBank,
    bindings: tuple[ArmBankBinding, ...],
) -> tuple[FrozenBankAdapter, ...]:
    """Bind one shared frozen bank to four private arm delegates/corpora."""
    if len(bindings) != len(MODEL_ARMS):
        raise CandidateBankError(reason="frozen bank requires exactly four model-arm bindings")
    by_arm = {binding.arm: binding for binding in bindings}
    if set(by_arm) != set(MODEL_ARMS) or len(by_arm) != len(MODEL_ARMS):
        raise CandidateBankError(reason="frozen bank bindings must follow all model arms")
    roots = tuple(binding.corpus_root.resolve() for binding in bindings)
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise CandidateBankError(reason="model-arm mutable corpus roots must be private")
    if len({id(binding.revise_delegate) for binding in bindings}) != len(bindings):
        raise CandidateBankError(reason="model-arm revision delegates must be private")
    return tuple(
        FrozenBankAdapter(
            arm,
            bank.selected_parents,
            bank.requested_seed_base,
            by_arm[arm].revise_delegate,
            by_arm[arm].corpus_root,
        )
        for arm in MODEL_ARMS
    )


def _deduplicate(
    parents: tuple[FrozenParent, ...],
) -> tuple[tuple[FrozenParent, ...], tuple[DuplicateParent, ...]]:
    unique_by_hash: dict[str, FrozenParent] = {}
    duplicates: list[DuplicateParent] = []
    for parent in parents:
        kept = unique_by_hash.get(parent.materialized_sha256)
        if kept is None:
            unique_by_hash[parent.materialized_sha256] = parent
        else:
            duplicates.append(
                DuplicateParent(
                    parent.materialized_sha256, parent.raw_parent_sha256, kept.raw_parent_sha256
                )
            )
    unique = tuple(sorted(unique_by_hash.values(), key=lambda parent: parent.materialized_sha256))
    return unique, tuple(duplicates)


def _select_diverse(parents: tuple[FrozenParent, ...], count: int) -> tuple[FrozenParent, ...]:
    remaining = list(parents)
    selected = [remaining.pop(0)]
    while len(selected) < count:
        index = min(
            range(len(remaining)),
            key=lambda candidate_index: (
                -min(_hamming(remaining[candidate_index], prior) for prior in selected),
                remaining[candidate_index].materialized_sha256,
            ),
        )
        selected.append(remaining.pop(index))
    return tuple(selected)


def _hamming(left: FrozenParent, right: FrozenParent) -> int:
    return sum(a != b for a, b in zip(left.result.selectors, right.result.selectors, strict=True))


def _duplicate_digest(duplicates: tuple[DuplicateParent, ...]) -> str:
    return _hash_parts(tuple(":".join(item) for item in duplicates))


def _selected_lineages(
    parents: tuple[FrozenParent, ...], requested_seed_base: int
) -> tuple[SelectedLineage, ...]:
    return tuple(
        SelectedLineage(
            f"p{index:04d}", parent.request.seed, requested_seed_base + index, parent.request.steps
        )
        for index, parent in enumerate(parents)
    )


def _generation_cost(parents: tuple[FrozenParent, ...]) -> BankGenerationCost:
    return BankGenerationCost(
        len(parents),
        sum(parent.result.nfe for parent in parents),
        sum(parent.result.forward_calls for parent in parents),
        sum(parent.result.gpu_ms for parent in parents),
    )


def _subtract_cost(total: BankGenerationCost, observed: BankGenerationCost) -> BankGenerationCost:
    return BankGenerationCost(*(total[index] - observed[index] for index in range(len(total))))


def _hash_parts(parts: tuple[str | int, ...]) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()
