"""Multi-criterion harness fitness scoring.

``R = w_compile*Compile + w_reach*TargetReach + w_edge*EdgeCoverage +
w_diversity*PathDiversity - w_timeout*Timeout - w_bug*HarnessBug`` (spec §8). Weights are
config-driven, not hardcoded — this repo's convention for every other threshold — and
``diversity``/``bug`` default to 0 because their measurers do not exist yet in the MVP
(spec §5/§6): Task 8's results should decide whether they are needed before adding real
measurers, rather than guessing a nonzero value now.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FitnessWeights:
    """Config-driven coefficients for each fitness term."""

    compile: float
    reach: float
    edge_coverage: float
    timeout: float = 0.0
    diversity: float = 0.0
    bug: float = 0.0


@dataclass(frozen=True, slots=True)
class FitnessInputs:
    """Raw per-harness measurements the fitness function combines.

    Every field is a plain float rather than ``bool`` for the binary-ish signals
    (``compiled``, ``reached_target``, ``timed_out``, ``has_harness_bug``), so ``score``
    stays one weighted-sum expression with no per-term branching.
    """

    compiled: float
    reached_target: float
    edge_coverage: float
    timed_out: float = 0.0
    path_diversity: float = 0.0
    has_harness_bug: float = 0.0


def score(weights: FitnessWeights, inputs: FitnessInputs) -> float:
    """Return the weighted-sum fitness ``R`` for one harness (spec §8's formula)."""
    return (
        weights.compile * inputs.compiled
        + weights.reach * inputs.reached_target
        + weights.edge_coverage * inputs.edge_coverage
        + weights.diversity * inputs.path_diversity
        - weights.timeout * inputs.timed_out
        - weights.bug * inputs.has_harness_bug
    )
