"""Multi-criterion harness fitness scoring (spec §8): config-driven weights, no guesses."""

from dllm_agent.harness_synthesis.fitness import FitnessInputs, FitnessWeights, score


def test_score_combines_every_positive_term_with_its_weight() -> None:
    weights = FitnessWeights(compile=1.0, reach=2.0, edge_coverage=3.0)
    inputs = FitnessInputs(compiled=1.0, reached_target=1.0, edge_coverage=0.5)

    # 1*1 + 2*1 + 3*0.5 = 4.5
    assert score(weights, inputs) == 4.5


def test_score_subtracts_timeout_and_bug_terms() -> None:
    weights = FitnessWeights(compile=1.0, reach=1.0, edge_coverage=1.0, timeout=5.0, bug=10.0)
    inputs = FitnessInputs(
        compiled=1.0, reached_target=1.0, edge_coverage=1.0, timed_out=1.0, has_harness_bug=1.0
    )

    # (1+1+1) - 5 - 10 = -12
    assert score(weights, inputs) == -12.0


def test_diversity_and_bug_weights_default_to_zero_per_mvp_scope() -> None:
    # spec §8: HarnessBug/PathDiversity measurers do not exist yet in the MVP, so their
    # weights start at 0 rather than a guessed nonzero value.
    weights = FitnessWeights(compile=1.0, reach=1.0, edge_coverage=1.0)
    inputs = FitnessInputs(
        compiled=1.0, reached_target=1.0, edge_coverage=1.0, path_diversity=0.9, has_harness_bug=1.0
    )

    # path_diversity and has_harness_bug are measured but contribute nothing at weight 0
    assert score(weights, inputs) == 3.0


def test_a_harness_that_never_compiled_scores_from_whatever_terms_are_measured() -> None:
    # score is a pure formula: it does not itself enforce that a non-compiling harness
    # has zero reach/coverage -- that consistency is the caller's responsibility.
    weights = FitnessWeights(compile=1.0, reach=1.0, edge_coverage=1.0)
    inputs = FitnessInputs(compiled=0.0, reached_target=0.0, edge_coverage=0.0)

    assert score(weights, inputs) == 0.0
