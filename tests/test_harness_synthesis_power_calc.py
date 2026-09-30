"""RQ1 sample size: McNemar paired-proportions formula (Connor 1987)."""

import pytest

from dllm_agent.harness_synthesis.power_calc import required_n, required_n_from_discordant_pairs


def test_required_n_from_discordant_pairs_matches_the_connor_1987_worked_example() -> None:
    # Connor (1987)'s textbook example: pi10=0.5, pi01=0.1, alpha=0.05, power=0.80 -> n=27.
    assert required_n_from_discordant_pairs(pi_10=0.5, pi_01=0.1) == 27


def test_required_n_from_discordant_pairs_raises_for_a_zero_effect() -> None:
    with pytest.raises(ValueError, match="no effect"):
        required_n_from_discordant_pairs(pi_10=0.3, pi_01=0.3)


def test_required_n_from_discordant_pairs_raises_for_a_reversed_effect() -> None:
    with pytest.raises(ValueError, match="no effect"):
        required_n_from_discordant_pairs(pi_10=0.1, pi_01=0.5)


def test_required_n_uses_the_feasibility_pilot_marginals_under_independence() -> None:
    # docs/harness-synthesis-feasibility-20260920.md Result 1: 11/16 confidence-correct,
    # 3/16 random-correct. Independence between the two detectors is a documented,
    # conservative substitute for the pilot's unpreserved exact pairing.
    n = required_n(p_confidence=11 / 16, p_random=3 / 16)

    assert isinstance(n, int)
    assert n > 0


def test_required_n_grows_as_the_effect_size_shrinks() -> None:
    large_effect = required_n(p_confidence=0.6875, p_random=0.1875)
    small_effect = required_n(p_confidence=0.55, p_random=0.45)

    assert small_effect > large_effect


def test_required_n_raises_when_there_is_no_effect_to_detect() -> None:
    with pytest.raises(ValueError, match="no effect"):
        required_n(p_confidence=0.5, p_random=0.5)


def test_required_n_shrinks_with_lower_required_power() -> None:
    high_power = required_n(p_confidence=0.6875, p_random=0.1875, power=0.95)
    low_power = required_n(p_confidence=0.6875, p_random=0.1875, power=0.60)

    assert low_power < high_power
