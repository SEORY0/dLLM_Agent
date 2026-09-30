"""Required sample size for RQ1's paired localization test (Task 3).

RQ1 compares two region-selection detectors (lowest-commit-confidence vs uniformly
random) against the same items — a paired-proportions design, not two independent
samples — so the right sample-size formula is McNemar's, not a two-proportion z-test.
Uses Connor's (1987) closed-form formula from the discordant-pair proportions.

The feasibility pilot (``docs/harness-synthesis-feasibility-20260920.md`` Result 1: 11/16
confidence-correct, 3/16 random-correct) reported only marginal proportions, not which
specific items were discordant and in which direction — that per-item pairing was not
preserved. ``required_n`` treats the two detectors as independent to recover discordant
proportions from those marginals; this is a documented substitute for the real joint
distribution, not a measurement of it, and is conservative in the common case where the
two detectors' outcomes are positively correlated (fewer regions to choose among makes
both detectors more likely to agree, which would only shrink the true required N).
"""

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def required_n_from_discordant_pairs(
    pi_10: float, pi_01: float, *, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Return McNemar's required N given the two discordant-pair proportions.

    ``pi_10`` is P(first detector right, second wrong); ``pi_01`` is the reverse. Connor
    (1987)'s worked example (``pi_10=0.5, pi_01=0.1``) yields ``n=27`` for the defaults.
    """
    if pi_10 <= pi_01:
        reason = "no effect to detect: pi_10 must exceed pi_01"
        raise ValueError(reason)
    diff = pi_10 - pi_01
    total = pi_10 + pi_01
    z_alpha: float = _NORMAL.inv_cdf(1 - alpha / 2)
    z_beta: float = _NORMAL.inv_cdf(power)
    # math.sqrt (not **0.5): a negative base under **0.5 types as complex, which would
    # poison the rest of this expression's type inference even though total/diff are
    # always non-negative here (they're derived from probabilities).
    numerator = z_alpha * math.sqrt(total) + z_beta * math.sqrt(total - diff**2)
    return math.ceil(numerator**2 / diff**2)


def required_n(
    p_confidence: float, p_random: float, *, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Return the required N from the two detectors' marginal success rates.

    Assumes independence between the two detectors to derive discordant-pair
    proportions from marginals alone — see the module docstring for why, and its
    conservative direction.
    """
    if p_confidence <= p_random:
        reason = "no effect to detect: p_confidence must exceed p_random"
        raise ValueError(reason)
    pi_10 = p_confidence * (1 - p_random)
    pi_01 = (1 - p_confidence) * p_random
    return required_n_from_discordant_pairs(pi_10, pi_01, alpha=alpha, power=power)
