"""RQ4b arm construction, mask-length handling, candidate selection, and budget accounting."""

import pytest

from dllm_agent.harness_synthesis.remask import TokenWindow
from dllm_agent.harness_synthesis.repair_arms import (
    ARMS,
    CONFIRMATORY_SEED_BASE,
    DIAGNOSTIC_ARM,
    DIAGNOSTIC_SEED_SLOT,
    PILOT_REPLICATES,
    PILOT_SEED_BASE,
    SUFFIX_SLACK,
    CandidateOutcome,
    K,
    decode_dropping_span_specials,
    forward_cost,
    infill_continuation,
    infill_mask_lengths,
    outside_span_preserved,
    plan,
    seed_for,
    seed_slot,
    seed_slot_balance,
    select,
    success_within_budget,
    suffix_continuation,
)

_M = 99


def test_infill_mask_lengths_follow_the_frozen_nearest_first_order() -> None:
    assert infill_mask_lengths(14) == (14, 16, 12, 18, 22)


def test_nonpositive_mask_lengths_raise_instead_of_silently_dropping() -> None:
    assert infill_mask_lengths(3) == (3, 5, 1, 7, 11)
    with pytest.raises(ValueError, match="nonpositive"):
        _ = infill_mask_lengths(2)


def test_plan_is_attempt_matched_with_common_seeds() -> None:
    plans = plan(
        TokenWindow(start=4, length=10), 60, sample_index=2, replicate=1, base=PILOT_SEED_BASE
    )
    by_arm: dict[str, list[int]] = {}
    for p in plans:
        by_arm.setdefault(p.arm, []).append(p.seed)
        assert p.steps == p.masked
    assert {arm: len(seeds) for arm, seeds in by_arm.items()} == {
        "infill-sweep": K,
        "suffix": K,
        "resample": K,
    }
    # Same seed slots in every arm (common random numbers); infill's are rotated by
    # replicate (replicate 1 here), so length position k uses slot (k + 1) % K.
    assert sorted(by_arm["infill-sweep"]) == by_arm["suffix"] == by_arm["resample"]
    assert by_arm["infill-sweep"] == [*by_arm["suffix"][1:], by_arm["suffix"][0]]
    suffix = next(p for p in plans if p.arm == "suffix")
    assert suffix.masked == 60 - 4 + SUFFIX_SLACK


def test_oracle_length_adds_one_diagnostic_candidate_outside_the_compared_arms() -> None:
    window = TokenWindow(start=0, length=14)
    plain = plan(window, 60, sample_index=0, replicate=0, base=PILOT_SEED_BASE)
    with_oracle = plan(
        window, 60, sample_index=0, replicate=0, base=PILOT_SEED_BASE, oracle_length=13
    )
    assert with_oracle[: len(plain)] == plain
    extra = with_oracle[-1]
    assert (extra.arm, extra.masked) == (DIAGNOSTIC_ARM, 13)
    assert seed_slot(extra.seed, base=PILOT_SEED_BASE) == DIAGNOSTIC_SEED_SLOT
    assert extra.seed not in {p.seed for p in plain}
    assert DIAGNOSTIC_ARM not in ARMS


def test_every_mask_length_meets_every_seed_slot_once_over_k_replicates() -> None:
    window = TokenWindow(start=0, length=14)
    plans = [
        p
        for r in range(PILOT_REPLICATES)
        for p in plan(window, 60, sample_index=3, replicate=r, base=PILOT_SEED_BASE)
    ]
    balance = seed_slot_balance(plans, base=PILOT_SEED_BASE)
    assert balance == {k: set(range(K)) for k in range(K)}
    # With fewer replicates the design is not balanced; that is why P1 uses R = K.
    short = [p for p in plans if p.seed < seed_for(3, 4, 0, base=PILOT_SEED_BASE)]
    assert any(len(s) < K for s in seed_slot_balance(short, base=PILOT_SEED_BASE).values())


def test_seed_blocks_do_not_collide_between_pilot_and_confirmatory() -> None:
    pilot_max = seed_for(499, 99, 9, base=PILOT_SEED_BASE)
    assert pilot_max < CONFIRMATORY_SEED_BASE
    assert seed_for(0, 0, 0, base=PILOT_SEED_BASE) > 12_999_999  # outside RQ4's reserved block
    with pytest.raises(ValueError, match="range"):
        _ = seed_for(0, 0, 10, base=PILOT_SEED_BASE)


def test_infill_and_suffix_continuations() -> None:
    ids = [1, 2, 3, 4, 5]
    window = TokenWindow(start=1, length=2)
    assert infill_continuation(ids, window, mask_length=4, mask_token_id=_M) == [
        1,
        _M,
        _M,
        _M,
        _M,
        4,
        5,
    ]
    assert suffix_continuation(ids, 1, mask_token_id=_M, slack=2) == [1, _M, _M, _M, _M, _M, _M]


def test_outside_span_preservation_detects_changes_on_either_side() -> None:
    before = [1, 2, 3, 4, 5]
    window = TokenWindow(start=1, length=2)
    assert outside_span_preserved(before, [1, 7, 7, 7, 4, 5], window, mask_length=3)
    assert not outside_span_preserved(before, [0, 7, 7, 7, 4, 5], window, mask_length=3)
    assert not outside_span_preserved(before, [1, 7, 7, 7, 4, 6], window, mask_length=3)
    assert not outside_span_preserved(before, [1, 7, 7, 4, 5], window, mask_length=3)


def test_special_pieces_are_dropped_only_inside_the_span() -> None:
    pieces = ["a", "<|endoftext|>", "b", "<|endoftext|>"]
    assert decode_dropping_span_specials(pieces, 0, 2) == "ab<|endoftext|>"


def test_forward_cost_counts_every_position_per_step() -> None:
    assert forward_cost(10, prompt_length=900, continuation_length=60) == 9600


def _o(index: int, *, compiled: bool, success: bool, cost: int = 10) -> CandidateOutcome:
    return CandidateOutcome(index, compiled, success, cost, 0.0)


def test_select_stops_at_first_functional_success_not_first_compile() -> None:
    outcomes = [
        _o(0, compiled=True, success=False),
        _o(1, compiled=False, success=False),
        _o(2, compiled=True, success=True),
        _o(3, compiled=True, success=True),
    ]
    s = select(outcomes)
    assert s.pass_at_k
    assert (s.selected_index, s.first_success_index, s.compiler_invocations) == (2, 2, 3)
    assert s.cost_to_selection == 30
    assert s.total_cost == 40


def test_select_without_success_charges_all_candidates_and_picks_first_compiled() -> None:
    s = select([_o(0, compiled=False, success=False), _o(1, compiled=True, success=False)])
    assert not s.pass_at_k
    assert (s.selected_index, s.compiler_invocations, s.cost_to_selection) == (1, 2, 20)
    with pytest.raises(ValueError, match="no candidates"):
        _ = select([])


def test_select_respects_frozen_order_not_list_order() -> None:
    s = select([_o(1, compiled=True, success=True), _o(0, compiled=True, success=True)])
    assert s.first_success_index == 0


def test_budget_prefix_can_afford_zero_candidates() -> None:
    cheap = [_o(i, compiled=True, success=i == 2, cost=10) for i in range(5)]
    assert success_within_budget(cheap, 30) == (True, 3)
    assert success_within_budget(cheap, 25) == (False, 2)
    dear = [_o(0, compiled=True, success=True, cost=100)]
    assert success_within_budget(dear, 50) == (False, 0)
