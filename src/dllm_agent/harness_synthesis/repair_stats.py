"""Paired, cluster-aware statistics for RQ4b's arm comparisons.

The unit of inference is the benchmark sample (one mutation), not the generation. The R
replicate blocks of one sample share its prompt, its defect and its context, so they are
not independent. Treating them as independent pairs (as a plain sign test on every
(sample, replicate) pair would) overstates the evidence. Each sample contributes one
number, the mean over its replicates of ``success_a - success_b``. Inference uses these
per-sample differences:

- an exact sign-flip permutation test (all ``2**n`` sign patterns, n <= 20);
- a percentile cluster-bootstrap CI with a fixed RNG seed;
- the Holm step-down adjustment across the prespecified primary family;
- a normal-approximation sample size for a confirmatory run, from the pilot's SD of the
  per-sample differences.

Pure; no NumPy, so the tests run anywhere the package does.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from random import Random
from statistics import NormalDist, fmean, stdev

_NORMAL = NormalDist()
_EXACT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Estimate, CI and exploratory/confirmatory p-value for ``a - b``."""

    clusters: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    replicate_pairs: int
    a_only: int
    b_only: int


def cluster_differences(
    pairs: Mapping[str, Sequence[tuple[bool, bool]]],
) -> list[float]:
    """Per-sample mean of ``a - b`` over its replicate pairs, in sorted sample order."""
    return [
        fmean(int(a) - int(b) for a, b in pairs[sample])
        for sample in sorted(pairs)
        if pairs[sample]
    ]


def sign_flip_p(differences: Sequence[float]) -> float:
    """Exact two-sided permutation p-value for mean difference 0 under sign symmetry."""
    n = len(differences)
    if n == 0:
        return 1.0
    if n > _EXACT_LIMIT:
        reason = f"exact sign-flip test limited to {_EXACT_LIMIT} clusters"
        raise ValueError(reason)
    observed = abs(sum(differences))
    hits = sum(
        1
        for signs in product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, differences, strict=True))) >= observed - 1e-12
    )
    return hits / (1 << n)


def bootstrap_ci(
    differences: Sequence[float], *, level: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the mean per-sample difference, resampling samples with replacement."""
    if not differences:
        return (0.0, 0.0)
    rng = Random(seed)  # noqa: S311 - reproducible resampling, not security.
    n = len(differences)
    means = sorted(fmean(rng.choices(differences, k=n)) for _ in range(resamples))
    tail = (1 - level) / 2
    low = means[max(0, math.floor(tail * resamples))]
    high = means[min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)]
    return (low, high)


def compare(pairs: Mapping[str, Sequence[tuple[bool, bool]]], *, seed: int = 0) -> PairedComparison:
    """Full paired comparison of arm ``a`` against arm ``b``."""
    differences = cluster_differences(pairs)
    flat = [pair for sample in pairs.values() for pair in sample]
    low, high = bootstrap_ci(differences, seed=seed)
    return PairedComparison(
        clusters=len(differences),
        mean_difference=fmean(differences) if differences else 0.0,
        ci_low=low,
        ci_high=high,
        p_value=sign_flip_p(differences),
        replicate_pairs=len(flat),
        a_only=sum(1 for a, b in flat if a and not b),
        b_only=sum(1 for a, b in flat if b and not a),
    )


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def required_clusters(sd: float, delta: float, *, alpha: float = 0.05, power: float = 0.80) -> int:
    """Samples needed to detect a mean per-sample difference ``delta`` (two-sided).

    ``n = ((z_{1-alpha/2} + z_{power}) * sd / delta) ** 2``, plus 2 for the t-versus-normal
    gap at small n. ``sd`` is the pilot's SD of the per-sample differences. A pilot of
    about 12 samples estimates it loosely, so ``pilot_sd_upper`` gives a conservative
    input.
    """
    if delta <= 0 or sd <= 0:
        reason = "required_clusters: delta and sd must be positive"
        raise ValueError(reason)
    z = _NORMAL.inv_cdf(1 - alpha / 2) + _NORMAL.inv_cdf(power)
    return math.ceil((z * sd / delta) ** 2) + 2


def pilot_sd_upper(differences: Sequence[float], *, quantile: float = 0.80) -> float:
    """One-sided upper ``quantile`` confidence bound on the SD (chi-square, Wilson-Hilferty).

    Browne (1995) recommends sizing a study from an upper confidence limit of a pilot's
    SD, not its point estimate, because small pilots underestimate it often enough to
    leave the follow-up underpowered.
    """
    n = len(differences)
    if n < 2:  # noqa: PLR2004
        reason = "pilot_sd_upper: need at least two clusters"
        raise ValueError(reason)
    df = n - 1
    z = _NORMAL.inv_cdf(1 - quantile)
    chi2_low = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    return stdev(differences) * math.sqrt(df / chi2_low)
