"""Strict JSON schemas for the deterministic native-hunt controller."""

from typing import Literal, Self

from pydantic import Field, model_validator

from dllm_agent.core.models import FrozenModel, Probability, Sha256
from dllm_agent.hunt.models import CoverageEvidence, CoverageTotalsEvidence, HuntReport

BuildStatus = Literal["succeeded", "failed", "timed_out"]
FuzzStatus = Literal["completed", "crashed", "timed_out", "skipped"]


class ProbeStage(FrozenModel):
    """One equal-budget fixed-seed probe stage for each active candidate."""

    duration_seconds: int = Field(gt=0)
    fuzzer_seeds: tuple[int, ...] = Field(min_length=2)
    survivors: int = Field(gt=0)

    @model_validator(mode="after")
    def _require_distinct_nonnegative_seeds(self) -> Self:
        """Reject schedules that cannot repeat an independent seed probe."""
        if any(seed < 0 for seed in self.fuzzer_seeds):
            msg = "fuzzer seeds must be nonnegative"
            raise ValueError(msg)
        if len(set(self.fuzzer_seeds)) != len(self.fuzzer_seeds):
            msg = "fuzzer seeds must be distinct"
            raise ValueError(msg)
        return self


class NativeHuntConfig(FrozenModel):
    """Versioned minimal controller configuration for native probe and proof budgets."""

    schema_version: Literal["dllm-native-hunt-config/v1"]
    stages: tuple[ProbeStage, ...] = Field(min_length=1)
    proof_duration_seconds: int = Field(gt=0)


class TrajectoryFeatures(FrozenModel):
    """Bounded stock-Stable trajectory metrics used only as a dynamic tie-breaker.

    weighted_confidence is committed-token weighted; confidence_floor is the lowest
    observed step confidence; volatility is adjacent-step mean absolute difference;
    commitment_auc is cumulative masked-token commitment over observed-step progress.
    """

    candidate_id: str = Field(min_length=1)
    weighted_confidence: Probability
    confidence_floor: Probability
    volatility: Probability
    commitment_auc: Probability
    nfe_efficiency: Probability
    trajectory_quality: Probability


class ProbeEvidence(FrozenModel):
    """Hashes binding a probe's raw build, fuzz, and optional coverage evidence."""

    build_log_sha256: Sha256
    fuzz_log_sha256: Sha256
    build_argv_sha256: Sha256
    fuzz_argv_sha256: Sha256
    coverage: CoverageEvidence | None = None


class ProbeEvent(FrozenModel):
    """Immutable normalized result for one candidate, stage, and fuzzer seed."""

    sequence: int = Field(ge=1)
    stage_index: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)
    fuzzer_seed: int = Field(ge=0)
    build_status: BuildStatus
    fuzz_status: FuzzStatus
    sanitizer_observed: bool
    coverage: CoverageTotalsEvidence | None = None
    feature_coverage: int | None = Field(default=None, ge=0)
    edge_coverage: int | None = Field(default=None, ge=0)
    executions_per_second: int | None = Field(default=None, ge=0)
    evidence: ProbeEvidence


class StageEvent(FrozenModel):
    """Immutable deterministic promotion decision after one probe stage."""

    sequence: int = Field(ge=1)
    stage_index: int = Field(ge=0)
    ranked_candidate_ids: tuple[str, ...] = Field(min_length=1)
    promoted_candidate_ids: tuple[str, ...] = Field(min_length=1)


class SelectionEvent(FrozenModel):
    """Immutable final candidate selection for the separate proof run."""

    sequence: int = Field(ge=1)
    selected_candidate_id: str = Field(min_length=1)


class ControllerEvidence(FrozenModel):
    """Hash-bound controller ledger without timestamps or mutable state."""

    controller_events_sha256: Sha256
    probe_events: tuple[ProbeEvent, ...]
    stage_events: tuple[StageEvent, ...]
    selection: SelectionEvent


class NativeHuntReport(FrozenModel):
    """Versioned native-controller report with a separate existing proof report."""

    schema_version: Literal["dllm-native-hunt-report/v1"]
    config: NativeHuntConfig
    trajectory_features: tuple[TrajectoryFeatures, ...] = Field(min_length=1)
    controller: ControllerEvidence
    proof: HuntReport
