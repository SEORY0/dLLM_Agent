"""Unique-bank, diversity, and feedback attribution contracts."""

import hashlib

import pytest

from dllm_agent.experiments.multitarget.candidate_bank import (
    CandidateBankError,
    ProbeObservation,
    freeze_candidate_bank,
    pseudo_feedback,
    remask_plan,
    select_diverse,
)
from dllm_agent.experiments.multitarget.slot_sampler import SlotSample
from dllm_agent.targets.structured import load_structured_target


def test_exact_thirty_two_raw_trajectories_freeze_at_least_sixteen_unique() -> None:
    # Given
    target = load_structured_target("libpng")
    samples = tuple(
        _sample(tuple((particle // (4**slot)) % 4 for slot in range(10)), particle)
        for particle in range(32)
    )

    # When
    bank = freeze_candidate_bank(target, samples)

    # Then
    assert bank.raw_count == 32
    assert bank.minimum_unique_count == 16
    assert bank.unique_count >= 16
    assert len(bank.raw_trajectory_sha256) == 64
    assert len(bank.unique_snapshot_sha256) == 64


def test_unique_gate_fails_closed_when_materialization_collapses() -> None:
    # Given
    target = load_structured_target("libpng")
    samples = tuple(_sample((0,) * 10, particle) for particle in range(32))

    # When / Then
    with pytest.raises(CandidateBankError, match="at least 16 unique snapshots; got 1"):
        _ = freeze_candidate_bank(target, samples)


def test_diversity_selection_is_hash_stable_under_raw_order_changes() -> None:
    # Given
    target = load_structured_target("libxml2")
    samples = tuple(
        _sample(tuple((particle // (4**slot)) % 4 for slot in range(10)), particle)
        for particle in range(32)
    )
    forward = freeze_candidate_bank(target, samples)
    reverse = freeze_candidate_bank(target, tuple(reversed(samples)))

    # When
    forward_hashes = tuple(candidate.snapshot.sha256 for candidate in select_diverse(forward, 16))
    reverse_hashes = tuple(candidate.snapshot.sha256 for candidate in select_diverse(reverse, 16))

    # Then
    assert forward_hashes == reverse_hashes
    assert len(set(forward_hashes)) == 16


def test_pseudo_feedback_is_fixed_point_free_and_preserves_observation_multiset() -> None:
    # Given
    observations = tuple(
        _observation(candidate_sha256=hashlib.sha256(bytes((index,))).hexdigest(), coverage=index)
        for index in range(16)
    )

    # When
    permuted = pseudo_feedback(observations, seed=7)

    # Then
    assert all(item.feedback_source_sha256 != item.candidate_sha256 for item in permuted)
    assert sorted(item.coverage_edges for item in permuted) == list(range(16))
    assert tuple(item.candidate_sha256 for item in permuted) == tuple(
        item.candidate_sha256 for item in observations
    )


def test_real_and_pseudo_feedback_choose_different_remask_fields() -> None:
    # Given
    target = load_structured_target("libxml2")
    selectors = (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    samples = tuple(
        _sample(values, index, uncertain_field=1) for index, values in enumerate(selectors)
    )
    fillers = tuple(
        _sample(
            tuple((particle // (4**slot)) % 4 for slot in range(10)),
            particle,
        )
        for particle in range(64, 92)
    )
    bank = freeze_candidate_bank(target, (*samples, *fillers))
    candidates = tuple(
        next(candidate for candidate in bank.candidates if candidate.sample.selectors == values)
        for values in selectors
    )
    observations = tuple(
        _observation(candidate.snapshot.sha256, coverage)
        for candidate, coverage in zip(candidates, (0, 0, 100, 100), strict=True)
    )
    mutable_indexes = tuple(range(10))

    # When
    real_plan = remask_plan(candidates, observations, mutable_indexes)
    pseudo_plan = remask_plan(candidates, pseudo_feedback(observations, seed=0), mutable_indexes)

    # Then
    assert set(real_plan) == {0}
    assert set(pseudo_plan) == {1}


def test_remask_plan_ignores_canary_only_selector_association() -> None:
    # Given
    target = load_structured_target("libxml2")
    selectors = (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    samples = tuple(
        _sample(values, index, uncertain_field=1) for index, values in enumerate(selectors)
    )
    fillers = tuple(
        _sample(
            tuple((particle // (4**slot)) % 4 for slot in range(10)),
            particle,
        )
        for particle in range(64, 92)
    )
    bank = freeze_candidate_bank(target, (*samples, *fillers))
    candidates = tuple(
        next(candidate for candidate in bank.candidates if candidate.sample.selectors == values)
        for values in selectors
    )
    canary_observations = tuple(
        _observation(
            candidate.snapshot.sha256,
            coverage=10,
            canary=candidate.sample.selectors[0] == 1,
        )
        for candidate in candidates
    )
    neutral_observations = tuple(
        _observation(candidate.snapshot.sha256, coverage=10, canary=False)
        for candidate in candidates
    )
    mutable_indexes = tuple(range(10))

    # When
    canary_plan = remask_plan(candidates, canary_observations, mutable_indexes)
    neutral_plan = remask_plan(candidates, neutral_observations, mutable_indexes)

    # Then
    assert canary_plan == neutral_plan
    assert set(canary_plan) == {1}


def _sample(
    selectors: tuple[int, ...],
    seed: int,
    *,
    uncertain_field: int | None = None,
) -> SlotSample:
    confidences = tuple(0.1 if index == uncertain_field else 0.9 for index in range(10))
    state_sha256 = hashlib.sha256(f"state:{seed}:{selectors}".encode()).hexdigest()
    return SlotSample(
        token_ids=selectors,
        selectors=selectors,
        slot_positions=tuple(range(10)),
        slot_confidences=confidences,
        selected_slot_token_ids=selectors,
        requested_steps=8,
        nfe=1,
        forward_calls=1,
        gpu_ms=1,
        trajectory_steps=(),
        state_sha256=state_sha256,
    )


def _observation(candidate_sha256: str, coverage: int, *, canary: bool = False) -> ProbeObservation:
    return ProbeObservation(
        candidate_sha256=candidate_sha256,
        feedback_source_sha256=candidate_sha256,
        canary_triggered=canary,
        sanitizer_observed=False,
        coverage_edges=coverage,
        value_profile_features=0,
        executions_per_second=1,
        log_sha256="0" * 64,
    )
