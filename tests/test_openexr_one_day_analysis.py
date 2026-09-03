"""Literal Task 6 contracts for deterministic paired analysis. # noqa: SIZE_OK"""

# ruff: noqa: FBT003

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dllm_agent.experiments.openexr_one_day.analysis import (
    CAMPAIGN_DURATION_SECONDS,
    AnalysisError,
    CoverageCurve,
    DecisionInputs,
    EfficiencyInputs,
    EventTime,
    administrative_event_time,
    arm_auc,
    classify_decision,
    classify_efficiency,
    effective_coverage,
    exact_sign_test,
    holm_adjust,
    paired_bootstrap,
    paired_log_ratio,
    rank_biserial,
    time_to_coverage_threshold,
    time_to_first_proved_cause,
    trapezoid_area,
)


def test_trapezoid_area_returns_literal_four_for_linear_three_point_curve() -> None:
    # Given
    times = (0.0, 1.0, 2.0)
    excess = (0.0, 2.0, 4.0)

    # When
    area = trapezoid_area(times, excess)

    # Then
    assert area == 4.0


def test_arm_auc_normalizes_by_exact_three_hour_campaign_duration() -> None:
    # Given
    curve = CoverageCurve(
        times=(0.0, float(CAMPAIGN_DURATION_SECONDS)),
        branches=(10, 14),
        c0_branches=10,
    )

    # When
    result = arm_auc(curve)

    # Then
    assert result == 2.0


def test_model_readiness_at_checkpoint_gets_no_pre_ready_or_t0_credit() -> None:
    # Given
    times = tuple(float(value) for value in range(0, 10_801, 900))
    curve = CoverageCurve(
        times=times,
        branches=(30,) * 13,
        c0_branches=10,
        readiness_seconds=900.0,
    )

    # When
    effective = effective_coverage(curve)
    result = arm_auc(curve)

    # Then
    assert effective.times[:3] == (0.0, 900.0, 1800.0)
    assert effective.branches[:3] == (10.0, 10.0, 30.0)
    assert result == 17.5


def test_model_readiness_between_checkpoints_inserts_exact_c0_knot() -> None:
    # Given
    curve = CoverageCurve(
        times=(0.0, 900.0, 1800.0, 10_800.0),
        branches=(20, 20, 20, 20),
        c0_branches=10,
        readiness_seconds=450.0,
    )

    # When
    effective = effective_coverage(curve)

    # Then
    assert effective.times == (0.0, 450.0, 900.0, 1800.0, 10_800.0)
    assert effective.branches == (10.0, 10.0, 20.0, 20.0, 20.0)


def test_paired_log_ratio_literal_three_versus_one_is_log_two() -> None:
    # Given / When
    result = paired_log_ratio(3.0, 1.0)

    # Then
    assert result == pytest.approx(0.6931471805599453)
    assert math.expm1(result) == pytest.approx(1.0)


def test_paired_log_ratio_rejects_auc_at_or_below_minus_one() -> None:
    # Given / When / Then
    with pytest.raises(AnalysisError, match="greater than -1"):
        _ = paired_log_ratio(-1.0, 1.0)


@pytest.mark.parametrize(
    ("wins", "losses", "expected"),
    [(10, 0, 0.001953125), (9, 1, 0.021484375), (4, 4, 1.0)],
)
def test_exact_sign_test_uses_literal_two_sided_binomial_tail(
    wins: int, losses: int, expected: float
) -> None:
    # Given / When
    result = exact_sign_test(wins, losses)

    # Then
    assert result == expected


def test_exact_sign_test_excludes_ties_from_denominator() -> None:
    # Given / When
    with_ties = exact_sign_test(9, 1, ties=17)
    without_ties = exact_sign_test(9, 1)

    # Then
    assert with_ties == without_ties == 0.021484375


def test_fixed_seed_ten_thousand_pair_bootstrap_is_byte_deterministic() -> None:
    # Given
    log_ratios = tuple(math.log1p(value / 10.0) for value in range(10))

    # When
    first = paired_bootstrap(log_ratios)
    second = paired_bootstrap(log_ratios)

    # Then
    assert first == second
    assert first.seed == 202_608_310_001
    assert first.resample_count == 10_000
    assert first.lower_uplift == 0.19582607431013982
    assert first.upper_uplift == 0.7


@pytest.mark.parametrize(
    ("wins", "losses", "effect", "all_ties"),
    [(10, 0, 1.0, False), (0, 10, -1.0, False), (0, 0, 0.0, True)],
)
def test_rank_biserial_reports_direction_and_explicit_all_ties(
    wins: int, losses: int, effect: float, all_ties: bool
) -> None:
    # Given / When
    result = rank_biserial(wins, losses)

    # Then
    assert result.effect == effect
    assert result.all_ties is all_ties


def test_holm_adjustment_is_monotone_in_sorted_p_value_order() -> None:
    # Given
    p_values = (0.01, 0.04, 0.03)

    # When
    adjusted = holm_adjust(p_values)

    # Then
    assert adjusted == pytest.approx((0.03, 0.06, 0.06))
    assert tuple(sorted(adjusted)) == pytest.approx((0.03, 0.06, 0.06))


def test_missing_event_is_administratively_censored_not_discovered_at_deadline() -> None:
    # Given / When
    missing = administrative_event_time(None)
    observed_at_deadline = administrative_event_time(10_800.0)

    # Then
    assert missing == EventTime(time_seconds=10_800.0, censored=True)
    assert observed_at_deadline == EventTime(time_seconds=10_800.0, censored=False)


def test_coverage_threshold_is_fixed_at_c0_plus_ten() -> None:
    # Given
    times = (0.0, 900.0, 1800.0, 10_800.0)
    branches = (100, 109, 110, 111)

    # When
    event = time_to_coverage_threshold(times, branches, c0_branches=100)

    # Then
    assert event == EventTime(time_seconds=1800.0, censored=False)


def test_first_proved_cause_uses_matching_search_event_not_proof_time() -> None:
    # Given
    observations = (("a" * 64, 900.0), ("b" * 64, 300.0), ("a" * 64, 1800.0))

    # When
    event = time_to_first_proved_cause(observations, ("a" * 64,))

    # Then
    assert event == EventTime(time_seconds=900.0, censored=False)


def test_no_proved_cause_is_administratively_censored() -> None:
    # Given / When
    event = time_to_first_proved_cause((("a" * 64, 300.0),), ())

    # Then
    assert event == EventTime(time_seconds=10_800.0, censored=True)


def test_proved_input_without_search_event_fails_closed() -> None:
    # Given / When / Then
    with pytest.raises(AnalysisError, match="proved input has no matching search event"):
        _ = time_to_first_proved_cause((("a" * 64, 300.0),), ("b" * 64,))


def test_treatment_failure_curve_must_remain_frozen_through_deadline() -> None:
    # Given
    curve = CoverageCurve(
        times=(0.0, 900.0, 1800.0, 10_800.0),
        branches=(100, 105, 105, 105),
        c0_branches=100,
        failure_freeze_seconds=900.0,
    )

    # When
    result = arm_auc(curve)

    # Then
    assert result == pytest.approx(4.791666666666667)


def test_treatment_failure_curve_rejects_change_after_freeze() -> None:
    # Given
    curve = CoverageCurve(
        times=(0.0, 900.0, 1800.0, 10_800.0),
        branches=(100, 105, 106, 106),
        c0_branches=100,
        failure_freeze_seconds=900.0,
    )

    # When / Then
    with pytest.raises(AnalysisError, match="changed after treatment freeze"):
        _ = arm_auc(curve)


def test_go_classification_accepts_every_literal_boundary() -> None:
    # Given
    inputs = DecisionInputs(
        complete_pair_count=10,
        selective_wins=9,
        median_uplift=0.10,
        final_coverage_retention=0.95,
        deep_parse_drop_percentage_points=5.0,
        pair_integrity=True,
        resource_integrity=True,
        evidence_integrity=True,
    )

    # When
    decision = classify_decision(inputs)

    # Then
    assert decision == "GO-to-scale"


@pytest.mark.parametrize(
    "inputs",
    [
        DecisionInputs(10, 9, 0.0999, 0.95, 5.0, True, True, True),
        DecisionInputs(10, 9, 0.10, 0.9499, 5.0, True, True, True),
        DecisionInputs(10, 9, 0.10, 0.95, 5.0001, True, True, True),
        DecisionInputs(9, 9, -0.5, 1.0, 0.0, True, True, True),
        DecisionInputs(10, 5, -0.5, 1.0, 0.0, True, False, True),
    ],
)
def test_incomplete_or_invalid_gates_take_inconclusive_precedence(
    inputs: DecisionInputs,
) -> None:
    # Given / When
    decision = classify_decision(inputs)

    # Then
    assert decision == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "inputs",
    [
        DecisionInputs(10, 9, 0.0, 1.0, 0.0, True, True, True),
        DecisionInputs(10, 5, 0.5, 1.0, 0.0, True, True, True),
    ],
)
def test_no_go_applies_at_exact_preregistered_boundaries(inputs: DecisionInputs) -> None:
    # Given / When
    decision = classify_decision(inputs)

    # Then
    assert decision == "NO-GO-to-scale"


def test_efficiency_signal_requires_per_repetition_efficiency_and_seven_wins() -> None:
    # Given
    inputs = EfficiencyInputs(
        selective_auc=(120.0,) * 10,
        comparison_auc=(100.0,) * 10,
        selective_nfe=(1000,) * 10,
        comparison_nfe=(1000,) * 10,
        selective_gpu_seconds=(10.0,) * 10,
        comparison_gpu_seconds=(10.0,) * 10,
    )

    # When
    decision = classify_efficiency(inputs)

    # Then
    assert decision == "selective-revision-efficiency-signal"


def test_efficiency_missing_or_nonpositive_denominator_is_inconclusive() -> None:
    # Given
    passing = EfficiencyInputs(
        selective_auc=(120.0,) * 10,
        comparison_auc=(100.0,) * 10,
        selective_nfe=(1000,) * 10,
        comparison_nfe=(1000,) * 10,
        selective_gpu_seconds=(10.0,) * 10,
        comparison_gpu_seconds=(10.0,) * 10,
    )
    invalid = replace(passing, comparison_nfe=(0,) + (1000,) * 9)

    # When
    decision = classify_efficiency(invalid)

    # Then
    assert decision == "INCONCLUSIVE"
