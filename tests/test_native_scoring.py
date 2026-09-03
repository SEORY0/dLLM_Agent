"""Tests for deterministic stock-Stable trajectory scoring and stage ranking."""

import hashlib
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from dllm_agent.hunt.models import CoverageTotalsEvidence, DenoisingStep, HarnessCandidate
from dllm_agent.native.models import (
    BuildStatus,
    NativeHuntConfig,
    ProbeEvent,
    ProbeEvidence,
    ProbeStage,
    TrajectoryFeatures,
)
from dllm_agent.native.scoring import extract_trajectory_features, rank_stage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate(
    candidate_id: str,
    *,
    confidences: tuple[float, float] = (0.8, 0.2),
    nfe: int = 128,
) -> HarnessCandidate:
    return _candidate_with_steps(
        candidate_id,
        (
            DenoisingStep(
                step_index=1,
                masked_tokens_before=10,
                committed_tokens=4,
                mean_confidence=confidences[0],
                state_sha256="1" * 64,
            ),
            DenoisingStep(
                step_index=2,
                masked_tokens_before=6,
                committed_tokens=6,
                mean_confidence=confidences[1],
                state_sha256="2" * 64,
            ),
        ),
        nfe=nfe,
    )


def _candidate_with_steps(
    candidate_id: str,
    steps: tuple[DenoisingStep, ...],
    *,
    nfe: int = 128,
) -> HarnessCandidate:
    """Build one real candidate with an explicitly observed denoising trajectory."""
    source = f'extern "C" int LLVMFuzzerTestOneInput_{candidate_id}() {{ return 0; }}'
    return HarnessCandidate(
        candidate_id=candidate_id,
        source=source,
        source_sha256=_sha256(source),
        seed=7,
        nfe=nfe,
        steps=steps,
    )


def _with_candidate_id(features: TrajectoryFeatures, candidate_id: str) -> TrajectoryFeatures:
    """Assign an unchanged trajectory metric bundle to a different candidate ID."""
    return TrajectoryFeatures(
        candidate_id=candidate_id,
        weighted_confidence=features.weighted_confidence,
        confidence_floor=features.confidence_floor,
        volatility=features.volatility,
        commitment_auc=features.commitment_auc,
        nfe_efficiency=features.nfe_efficiency,
        trajectory_quality=features.trajectory_quality,
    )


@dataclass(frozen=True, slots=True)
class _ProbeMetrics:
    """Observed dynamic metrics used by the real event factory."""

    sanitizer_observed: bool = False
    build_status: BuildStatus = "succeeded"
    branches: float | None = 0.0
    lines: float | None = 0.0
    features: int | None = 0
    edges: int | None = 0
    executions_per_second: int | None = 0


def _event(
    candidate_id: str,
    sequence: int,
    metrics: _ProbeMetrics | None = None,
) -> ProbeEvent:
    metrics = _ProbeMetrics() if metrics is None else metrics
    return ProbeEvent(
        sequence=sequence,
        stage_index=0,
        candidate_id=candidate_id,
        fuzzer_seed=11,
        build_status=metrics.build_status,
        fuzz_status="completed",
        sanitizer_observed=metrics.sanitizer_observed,
        coverage=(
            None
            if metrics.branches is None or metrics.lines is None
            else CoverageTotalsEvidence(
                regions=0.0,
                functions=0.0,
                lines=metrics.lines,
                branches=metrics.branches,
            )
        ),
        feature_coverage=metrics.features,
        edge_coverage=metrics.edges,
        executions_per_second=metrics.executions_per_second,
        evidence=ProbeEvidence(
            build_log_sha256="a" * 64,
            fuzz_log_sha256="b" * 64,
            build_argv_sha256="c" * 64,
            fuzz_argv_sha256="d" * 64,
        ),
    )


def test_extract_trajectory_features_when_steps_are_observed() -> None:
    # Given: a hand-calculated two-step stock-Stable trajectory.
    candidate = _candidate("candidate-01")

    # When: only observed denoising-step fields are projected into controller features.
    features = extract_trajectory_features(candidate)

    # Then: committed weighting, floor, adjacent volatility, AUC, and quality are exact.
    assert features.weighted_confidence == 0.44
    assert features.confidence_floor == 0.2
    assert features.volatility == 0.6
    assert features.commitment_auc == 0.45
    assert features.nfe_efficiency == 0.75
    assert features.trajectory_quality == 0.419


def test_extract_trajectory_features_when_one_step_is_observed_has_zero_volatility() -> None:
    # Given: one observed step has no adjacent confidence pair.
    candidate = _candidate_with_steps(
        "candidate-01",
        (
            DenoisingStep(
                step_index=1,
                masked_tokens_before=4,
                committed_tokens=4,
                mean_confidence=0.6,
                state_sha256="3" * 64,
            ),
        ),
    )

    # When: stock trajectory features are extracted.
    features = extract_trajectory_features(candidate)

    # Then: the defined one-step volatility is zero.
    assert features.volatility == 0.0


def test_extract_trajectory_features_when_commitment_exceeds_initial_mask_clamps_auc() -> None:
    # Given: cumulative commitments exceed the first observed masked-token count.
    candidate = _candidate_with_steps(
        "candidate-01",
        (
            DenoisingStep(
                step_index=1,
                masked_tokens_before=4,
                committed_tokens=3,
                mean_confidence=0.5,
                state_sha256="3" * 64,
            ),
            DenoisingStep(
                step_index=2,
                masked_tokens_before=1,
                committed_tokens=3,
                mean_confidence=0.5,
                state_sha256="4" * 64,
            ),
        ),
    )

    # When: the cumulative commitment fraction reaches more than one.
    features = extract_trajectory_features(candidate)

    # Then: trapezoidal AUC uses fractions 0, 0.75, and clamped 1.0: 0.625.
    assert features.commitment_auc == 0.625


@pytest.mark.parametrize("nfe", [512, 1024])
def test_extract_trajectory_features_when_nfe_reaches_budget_saturates_efficiency(
    nfe: int,
) -> None:
    # Given: realized inference work at or above the 512-NFE budget.
    candidate = _candidate("candidate-01", nfe=nfe)

    # When: stock trajectory features are extracted.
    features = extract_trajectory_features(candidate)

    # Then: efficiency cannot fall below its defined zero floor.
    assert features.nfe_efficiency == 0.0


def test_rank_stage_when_trajectory_bundles_are_exchanged_reverses_dynamic_tie() -> None:
    # Given: fixed dynamic probe events and two different observed trajectory bundles.
    first_bundle = extract_trajectory_features(_candidate("candidate-01"))
    second_bundle = extract_trajectory_features(_candidate("candidate-02", confidences=(0.2, 0.8)))
    events = (
        _event("candidate-01", 1),
        _event("candidate-02", 2),
    )

    # When: the same metric bundles are exchanged between the same candidate IDs.
    baseline = rank_stage(events, (first_bundle, second_bundle))
    pseudo_trace = rank_stage(
        events,
        (
            _with_candidate_id(second_bundle, "candidate-01"),
            _with_candidate_id(first_bundle, "candidate-02"),
        ),
    )

    # Then: only the dynamic tie-break result reverses.
    assert baseline == ("candidate-02", "candidate-01")
    assert pseudo_trace == ("candidate-01", "candidate-02")


def test_rank_stage_when_dynamic_coverage_differs_ignores_trajectory() -> None:
    # Given: lower-quality trajectory has superior observed branch coverage.
    lower_quality = extract_trajectory_features(_candidate("candidate-01"))
    higher_quality = extract_trajectory_features(_candidate("candidate-02", confidences=(0.2, 0.8)))
    events = (
        _event("candidate-01", 1, _ProbeMetrics(branches=80.0)),
        _event("candidate-02", 2, _ProbeMetrics(branches=20.0)),
    )

    # When: a higher-priority dynamic key differs.
    ranked = rank_stage(events, (lower_quality, higher_quality))

    # Then: trajectory quality cannot overturn the observed coverage result.
    assert ranked == ("candidate-01", "candidate-02")


def test_rank_stage_when_metrics_are_missing_sorts_them_below_observed_zero() -> None:
    # Given: one candidate lacks coverage while another observed exactly zero coverage.
    features = (
        TrajectoryFeatures(
            candidate_id="candidate-01",
            weighted_confidence=0.5,
            confidence_floor=0.5,
            volatility=0.0,
            commitment_auc=0.5,
            nfe_efficiency=0.5,
            trajectory_quality=0.5,
        ),
        TrajectoryFeatures(
            candidate_id="candidate-02",
            weighted_confidence=0.5,
            confidence_floor=0.5,
            volatility=0.0,
            commitment_auc=0.5,
            nfe_efficiency=0.5,
            trajectory_quality=0.5,
        ),
        TrajectoryFeatures(
            candidate_id="candidate-03",
            weighted_confidence=1.0,
            confidence_floor=1.0,
            volatility=0.0,
            commitment_auc=1.0,
            nfe_efficiency=1.0,
            trajectory_quality=1.0,
        ),
    )
    events = (
        _event(
            "candidate-01",
            1,
            _ProbeMetrics(branches=None, lines=None, features=None, edges=None),
        ),
        _event("candidate-02", 2),
    )

    # When: stage results are ranked.
    ranked = rank_stage(events, features)

    # Then: observed zero wins over missing and absent-event candidates sort last.
    assert ranked == ("candidate-02", "candidate-01", "candidate-03")


def test_rank_stage_when_sanitizer_and_build_rate_differ_uses_declared_priority() -> None:
    # Given: candidate-01 has a sanitizer observation and candidate-02 builds more often.
    features = (
        TrajectoryFeatures(
            candidate_id="candidate-01",
            weighted_confidence=0.0,
            confidence_floor=0.0,
            volatility=0.0,
            commitment_auc=0.0,
            nfe_efficiency=0.0,
            trajectory_quality=0.0,
        ),
        TrajectoryFeatures(
            candidate_id="candidate-02",
            weighted_confidence=1.0,
            confidence_floor=1.0,
            volatility=0.0,
            commitment_auc=1.0,
            nfe_efficiency=1.0,
            trajectory_quality=1.0,
        ),
    )
    events = (
        _event("candidate-01", 1, _ProbeMetrics(sanitizer_observed=True, build_status="failed")),
        _event("candidate-02", 2),
        _event("candidate-02", 3),
    )

    # When: stage results are ranked.
    ranked = rank_stage(events, features)

    # Then: sanitizer observation precedes build success rate and all later keys.
    assert ranked == ("candidate-01", "candidate-02")


def test_probe_stage_when_seeds_repeat_is_rejected() -> None:
    # Given: a controller configuration with duplicate fuzzer seeds.
    # When / Then: the strict JSON boundary rejects a non-repeated seed schedule.
    with pytest.raises(ValidationError, match="distinct"):
        _ = ProbeStage(duration_seconds=1, fuzzer_seeds=(5, 5), survivors=1)


def test_native_hunt_config_when_stage_is_valid_is_frozen() -> None:
    # Given: the smallest valid deterministic controller configuration.
    config = NativeHuntConfig(
        schema_version="dllm-native-hunt-config/v1",
        stages=(ProbeStage(duration_seconds=1, fuzzer_seeds=(5, 6), survivors=1),),
        proof_duration_seconds=1,
    )

    # When / Then: the external boundary remains immutable.
    with pytest.raises(ValidationError, match="frozen"):
        config.proof_duration_seconds = 2
