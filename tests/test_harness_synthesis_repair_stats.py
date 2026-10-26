"""Cluster-paired statistics: per-sample differences, exact sign flip, bootstrap, Holm, sizing."""

from statistics import stdev

import pytest

from dllm_agent.harness_synthesis.repair_stats import (
    bootstrap_ci,
    cluster_differences,
    compare,
    holm,
    pilot_sd_upper,
    required_clusters,
    sign_flip_ci,
    sign_flip_p,
)


def test_cluster_differences_average_within_each_sample() -> None:
    pairs = {"b": [(True, False), (False, False)], "a": [(False, True)]}
    assert cluster_differences(pairs) == [-1.0, 0.5]


def test_sign_flip_matches_hand_computed_values() -> None:
    # All-positive n=3: only the identity and its full negation reach |sum|.
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)
    assert sign_flip_p([0.0, 0.0]) == 1.0
    assert sign_flip_p([]) == 1.0
    # n=12, all +1: 2 / 4096.
    assert sign_flip_p([1.0] * 12) == pytest.approx(2 / 4096)


def test_sign_flip_refuses_too_many_clusters_for_exact_enumeration() -> None:
    with pytest.raises(ValueError, match="limited"):
        _ = sign_flip_p([1.0] * 21)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    diffs = [0.0, 0.25, 0.5, 0.75, 1.0]
    low, high = bootstrap_ci(diffs, seed=3)
    assert (low, high) == bootstrap_ci(diffs, seed=3)
    assert low <= 0.5 <= high
    assert bootstrap_ci([], seed=0) == (0.0, 0.0)


def test_compare_counts_replicate_discordance_but_tests_on_clusters() -> None:
    pairs = {
        "s1": [(True, False)] * 4,
        "s2": [(True, False)] * 4,
        "s3": [(False, False)] * 4,
    }
    result = compare(pairs)
    assert result.clusters == 3
    assert result.replicate_pairs == 12
    assert (result.a_only, result.b_only) == (8, 0)
    # 8 discordant replicate pairs would give p=0.008 if treated as independent; the
    # cluster test sees only two informative samples.
    assert result.p_value == pytest.approx(0.5)
    assert result.mean_difference == pytest.approx(2 / 3)


def test_holm_is_monotone_and_capped() -> None:
    adjusted = holm({"H1": 0.01, "H2": 0.04, "H3": 0.5})
    assert adjusted == {"H1": pytest.approx(0.03), "H2": pytest.approx(0.08), "H3": 0.5}
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_required_clusters_grows_as_effect_shrinks() -> None:
    small = required_clusters(0.3, 0.10)
    large = required_clusters(0.3, 0.20)
    assert small > large
    # ((1.96 + 0.8416) * 0.3 / 0.1)^2 = 70.6 -> 71, + 2.
    assert small == 73
    with pytest.raises(ValueError, match="positive"):
        _ = required_clusters(0.0, 0.1)


def test_pilot_sd_upper_exceeds_the_point_estimate() -> None:
    diffs = [0.0, 0.25, -0.25, 0.5, 0.0, 0.25]
    assert pilot_sd_upper(diffs) > stdev(diffs)
    with pytest.raises(ValueError, match="two"):
        _ = pilot_sd_upper([0.1])


def test_sign_flip_ci_inverts_the_test_and_brackets_the_mean() -> None:
    diffs = [0.2, 0.4, 0.0, 0.6, 0.2, 0.0, 0.4, 0.2, 0.0, 0.2, 0.6, 0.4]
    low, high = sign_flip_ci(diffs)
    assert low <= sum(diffs) / len(diffs) <= high
    assert low > 0  # every difference is >= 0 and most are > 0
    assert sign_flip_p([d - low for d in diffs]) > 0.05
    assert sign_flip_p([d - (low - 0.02) for d in diffs]) <= 0.05
    assert sign_flip_ci([]) == (0.0, 0.0)


def test_compare_reports_exact_ci_and_bootstrap_separately() -> None:
    pairs = {f"s{i}": [(i % 2 == 0, False)] * 3 for i in range(8)}
    result = compare(pairs)
    assert result.ci_low <= result.mean_difference <= result.ci_high
    assert result.bootstrap_ci_low <= result.mean_difference <= result.bootstrap_ci_high
