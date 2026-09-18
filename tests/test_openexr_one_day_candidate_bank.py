"""Behavioral tests for the one-day shared OpenEXR candidate bank."""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError, dataclass, field
from typing import TYPE_CHECKING

import pytest
from tools.selective_revision_payloads import sha256_bytes, sha256_token_ids

from dllm_agent.experiments.openexr_one_day.candidate_bank import (
    RAW_PARENT_COUNT,
    SELECTED_PARENT_COUNT,
    SUPPLEMENT_PARENT_COUNT,
    ArmBankBinding,
    CandidateBank,
    CandidateBankError,
    FrozenBankAdapter,
    build_candidate_bank,
    build_frozen_bank_adapters,
    private_worker_delegate,
)
from dllm_agent.experiments.openexr_one_day.models import (
    MODEL_ARMS,
    ModelArm,
    RepetitionIndex,
    Seed,
    SeedNamespace,
)
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    RevisionRequest,
    RevisionResult,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records model requests.
class DeterministicInitialSampler:
    primary_unique: int
    supplement_new_unique: int = 0
    corrupt_trajectory_digest: bool = False
    initial_requests: list[InitialParticleRequest] = field(default_factory=list)

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        call_index = len(self.initial_requests)
        self.initial_requests.append(request)
        if call_index < RAW_PARENT_COUNT:
            variant = call_index % self.primary_unique
        else:
            offset = call_index - RAW_PARENT_COUNT
            variant = self.primary_unique + (offset % self.supplement_new_unique)
        result = _result(variant, call_index)
        if self.corrupt_trajectory_digest:
            return result.model_copy(update={"trajectory_evidence_sha256": "f" * 64})
        return result

    def revise(self, request: RevisionRequest) -> RevisionResult:
        del request
        raise AssertionError


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records private revision calls.
class RevisionDelegate:
    calls: list[RevisionRequest]

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        del request
        raise AssertionError

    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.calls.append(request)
        return _result(0, len(self.calls))


def test_shared_bank_freezes_identical_initial_evidence_for_all_model_arms(
    tmp_path: Path,
) -> None:
    # Given
    sampler = DeterministicInitialSampler(primary_unique=RAW_PARENT_COUNT)

    # When
    bank = build_candidate_bank(sampler, _seed_namespace(), steps=8)
    bindings = _bindings(tmp_path)
    adapters = build_frozen_bank_adapters(bank, bindings)
    first_lineage = bank.selected_lineages[0]
    initial_results = tuple(
        adapter.initial(
            _initial(
                adapter.arm,
                seed=first_lineage.requested_seed,
                steps=first_lineage.steps,
            )
        )
        for adapter in adapters
    )

    # Then
    assert sampler.initial_requests is not None
    assert len(sampler.initial_requests) == RAW_PARENT_COUNT
    assert bank.primary_raw_count == RAW_PARENT_COUNT
    assert bank.supplement_raw_count == 0
    assert len(bank.selected_parents) == SELECTED_PARENT_COUNT
    assert len({parent.materialized_sha256 for parent in bank.selected_parents}) == 16
    assert all(adapter.initial_parents is bank.selected_parents for adapter in adapters)
    assert len({id(result) for result in initial_results}) == 1
    first_views = tuple(adapter.initial_parents[0] for adapter in adapters)
    assert len({view.content for view in first_views}) == 1
    assert len({view.materialized_sha256 for view in first_views}) == 1
    assert len({view.result.token_state_sha256 for view in first_views}) == 1
    assert len({view.result.trajectory_evidence_sha256 for view in first_views}) == 1
    assert len({binding.corpus_root for binding in bindings}) == len(MODEL_ARMS)
    assert len({id(binding.revise_delegate) for binding in bindings}) == len(MODEL_ARMS)
    assert all(
        claim.logical_raw_bank_cost == bank.physical_cost for claim in bank.accounting.arm_claims
    )


def test_bank_recomputes_every_raw_trajectory_digest_from_canonical_steps() -> None:
    # Given
    sampler = DeterministicInitialSampler(
        primary_unique=RAW_PARENT_COUNT,
        corrupt_trajectory_digest=True,
    )

    # When / Then
    with pytest.raises(CandidateBankError, match="trajectory evidence hash"):
        _ = build_candidate_bank(sampler, _seed_namespace(), steps=8)


def test_all_four_adapters_bind_all_lineages_to_requested_seed_and_source_evidence(
    tmp_path: Path,
) -> None:
    # Given
    bank, _ = _bank()
    adapters = _adapters(bank, tmp_path)

    # When
    results = tuple(
        adapter.initial(
            _initial(adapter.arm, lineage.particle_id, lineage.requested_seed, lineage.steps)
        )
        for adapter in adapters
        for lineage in bank.selected_lineages
    )

    # Then
    assert tuple(lineage.particle_id for lineage in bank.selected_lineages) == tuple(
        f"p{index:04d}" for index in range(SELECTED_PARENT_COUNT)
    )
    assert len(results) == len(MODEL_ARMS) * SELECTED_PARENT_COUNT
    for adapter_index in range(len(MODEL_ARMS)):
        offset = adapter_index * SELECTED_PARENT_COUNT
        assert results[offset : offset + SELECTED_PARENT_COUNT] == tuple(
            parent.result for parent in bank.selected_parents
        )


@pytest.mark.parametrize(("seed_delta", "steps_delta"), [(1, 0), (0, 1)])
def test_frozen_adapter_rejects_wrong_lineage_seed_or_steps(
    tmp_path: Path,
    seed_delta: int,
    steps_delta: int,
) -> None:
    # Given
    bank, _ = _bank()
    adapter = _adapters(bank, tmp_path)[0]
    lineage = bank.selected_lineages[0]
    request = _initial(
        adapter.arm,
        lineage.particle_id,
        lineage.requested_seed + seed_delta,
        lineage.steps + steps_delta,
    )

    # When / Then
    with pytest.raises(CandidateBankError, match="lineage"):
        _ = adapter.initial(request)


def test_bank_runs_one_fixed_supplement_when_primary_uniqueness_is_too_low() -> None:
    # Given
    bank, sampler = _bank(8, 8)

    # Then
    assert sampler.initial_requests is not None
    assert len(sampler.initial_requests) == RAW_PARENT_COUNT + SUPPLEMENT_PARENT_COUNT
    assert bank.supplement_raw_count == SUPPLEMENT_PARENT_COUNT
    assert bank.unique_parent_count == SELECTED_PARENT_COUNT
    assert len(bank.duplicates) == RAW_PARENT_COUNT
    assert bank.physical_cost.initial_requests == RAW_PARENT_COUNT + SUPPLEMENT_PARENT_COUNT
    assert bank.duplicate_evidence_sha256 != "0" * 64


def test_bank_fails_closed_after_the_only_allowed_supplement() -> None:
    # Given
    sampler = DeterministicInitialSampler(primary_unique=1, supplement_new_unique=1)

    # When / Then
    with pytest.raises(CandidateBankError, match="at least 16 unique"):
        _ = build_candidate_bank(sampler, _seed_namespace(), steps=8)
    assert sampler.initial_requests is not None
    assert len(sampler.initial_requests) == RAW_PARENT_COUNT + SUPPLEMENT_PARENT_COUNT


def test_bank_selection_and_aggregate_hashes_are_deterministic_and_immutable() -> None:
    # Given
    first, _ = _bank()
    second, _ = _bank()

    # Then
    assert tuple(parent.materialized_sha256 for parent in first.selected_parents) == tuple(
        parent.materialized_sha256 for parent in second.selected_parents
    )
    assert first.raw_token_states_sha256 == second.raw_token_states_sha256
    assert first.raw_trajectories_sha256 == second.raw_trajectories_sha256
    assert first.raw_materialized_sha256 == second.raw_materialized_sha256
    assert first.selected_bank_sha256 == second.selected_bank_sha256
    with pytest.raises(FrozenInstanceError):
        first.__setattr__("primary_raw_count", 0)


@pytest.mark.parametrize(
    ("primary_unique", "supplement_new_unique", "raw_count"),
    [(RAW_PARENT_COUNT, 0, RAW_PARENT_COUNT), (8, 8, RAW_PARENT_COUNT + SUPPLEMENT_PARENT_COUNT)],
)
def test_bank_accounting_claims_charge_full_raw_cost_once_without_selected_double_count(
    primary_unique: int,
    supplement_new_unique: int,
    raw_count: int,
) -> None:
    # Given
    bank, _ = _bank(primary_unique, supplement_new_unique)

    # When
    accounting = bank.accounting

    # Then
    assert accounting.physical_campaign_cost.initial_requests == raw_count
    assert len(accounting.arm_claims) == len(MODEL_ARMS)
    assert len({claim.claim_id for claim in accounting.arm_claims}) == len(MODEL_ARMS)
    for claim in accounting.arm_claims:
        assert claim.pipeline_visible_selected_cost.initial_requests == SELECTED_PARENT_COUNT
        assert (
            claim.pipeline_visible_selected_cost.initial_requests
            + claim.task5_adjustment.initial_requests
            == raw_count
        )
        assert claim.logical_raw_bank_cost == accounting.physical_campaign_cost


def test_bank_rejects_ancestor_and_descendant_mutable_corpus_roots(tmp_path: Path) -> None:
    # Given
    bank, _ = _bank()
    root = tmp_path / "shared-parent"
    roots = (root, root / "descendant", tmp_path / "third", tmp_path / "fourth")
    bindings = _bindings(tmp_path, roots)

    # When / Then
    with pytest.raises(CandidateBankError, match="corpus roots"):
        _ = build_frozen_bank_adapters(bank, bindings)


def _seed_namespace() -> SeedNamespace:
    return SeedNamespace(
        repetition=RepetitionIndex(0),
        start_seed=Seed(1_000_000),
        stop_seed_exclusive=Seed(1_001_000),
    )


def _bank(
    primary_unique: int = RAW_PARENT_COUNT, supplement_new_unique: int = 0
) -> tuple[CandidateBank, DeterministicInitialSampler]:
    sampler = DeterministicInitialSampler(primary_unique, supplement_new_unique)
    return build_candidate_bank(sampler, _seed_namespace(), steps=8), sampler


def _bindings(tmp_path: Path, roots: tuple[Path, ...] | None = None) -> tuple[ArmBankBinding, ...]:
    corpus_roots = tuple(tmp_path / arm for arm in MODEL_ARMS) if roots is None else roots
    return tuple(
        ArmBankBinding(arm, RevisionDelegate([]), corpus_roots[index])
        for index, arm in enumerate(MODEL_ARMS)
    )


def _adapters(bank: CandidateBank, tmp_path: Path) -> tuple[FrozenBankAdapter, ...]:
    return build_frozen_bank_adapters(bank, _bindings(tmp_path))


def _initial(
    arm: ModelArm,
    particle_id: str = "p0000",
    seed: int = 999,
    steps: int = 8,
) -> InitialParticleRequest:
    return InitialParticleRequest(
        particle_id=particle_id,
        seed=seed,
        steps=steps,
        arm=arm,
        mode="initial",
    )


def _result(variant: int, call_index: int) -> RevisionResult:
    digits = tuple((variant // (4**index)) % 4 for index in range(5))
    selectors = (digits[0], digits[1], digits[2], digits[3], 0, 0, 0, 0, digits[4], 0)
    token_ids = tuple(selector + 10 for selector in selectors)
    return RevisionResult(
        selectors=selectors,
        token_ids=token_ids,
        slot_positions=tuple(range(10)),
        selected_slot_token_ids=token_ids,
        token_state_sha256=sha256_token_ids(token_ids),
        transfer_evidence_sha256=sha256_bytes(f"transfer:{call_index}".encode()),
        rng_state_sha256=sha256_bytes(f"rng:{call_index}".encode()),
        nfe=1,
        forward_calls=1,
        gpu_ms=10,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="width",
        trajectory_evidence_sha256=sha256_bytes(b"[]"),
    )


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records concurrent-entry evidence.
class ConcurrencyTrackingSampler:
    """Fake resident worker recording whether two calls ever overlapped."""

    active: int = 0
    max_concurrent: int = 0
    calls: int = 0
    _guard: threading.Lock = field(default_factory=threading.Lock)

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        del request
        with self._guard:
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
            self.calls += 1
        # ponytail: fixed sleep, not a retry loop -- good enough to make a
        # real race observable without flaking on a slow CI box.
        _ = threading.Event().wait(0.02)
        with self._guard:
            self.active -= 1
        return _result(0, 0)

    def revise(self, request: RevisionRequest) -> RevisionResult:
        del request
        return _result(0, 0)


def test_private_worker_delegate_gives_each_arm_a_distinct_object() -> None:
    # Given one shared underlying worker (e.g. a 1-GPU pool)
    worker = ConcurrencyTrackingSampler()

    # When wrapped once per arm
    selective_revision = private_worker_delegate(worker)
    final_only = private_worker_delegate(worker)

    # Then each wrapper is a distinct object, satisfying build_frozen_bank_adapters'
    # privacy check, even though they share one underlying delegate
    assert selective_revision is not final_only
    assert len({id(selective_revision), id(final_only)}) == 2


def test_private_worker_delegate_serializes_concurrent_calls_to_shared_worker() -> None:
    # Given two arm delegates wrapping the same underlying worker
    worker = ConcurrencyTrackingSampler()
    delegates = (private_worker_delegate(worker), private_worker_delegate(worker))
    request = _initial("selective-revision")

    # When both delegates are called concurrently
    threads = [
        threading.Thread(target=delegate.initial, args=(request,)) for delegate in delegates
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Then the underlying worker never saw two calls active at once
    assert worker.calls == len(delegates)
    assert worker.max_concurrent == 1
