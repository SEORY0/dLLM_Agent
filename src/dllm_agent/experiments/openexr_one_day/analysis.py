"""Deterministic paired statistics and fail-closed Task 6 campaign analysis.

Module-size opt-out ``# noqa: SIZE_OK`` keeps the task's pure statistics, evidence
audit, and canonical report transaction in the single production module assigned by
the preregistered implementation plan.  The literal preregistration thresholds,
typed fail-closed messages, and deterministic non-cryptographic bootstrap are kept
inline so that the analysis contract remains directly auditable.
"""

# ruff: noqa: EM101, EM102, PLR2004, S311, TRY003

from __future__ import annotations

import math
import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, cast, override

from pydantic import Field, TypeAdapter, ValidationError

from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellEvidenceError,
    load_cell_events,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignAccounting,
    CampaignManifest,
    CampaignOutcome,
    CandidateBankEvidence,
    CoordinatorError,
    CorpusReleaseEvidence,
    RawArmReportRow,
    build_campaign_manifest,
    load_complete_campaign_outcome,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CoverageSeriesError,
    validate_coverage_series_backings,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    validate_backing_file,
)
from dllm_agent.experiments.openexr_one_day.models import ARMS, MODEL_ARMS, Arm
from dllm_agent.experiments.openexr_one_day.post_search import (
    CoverageSeriesEnvelope,
    DeepParseEnvelope,
    ProofBatchEnvelope,
)
from dllm_agent.experiments.openexr_one_day.resident_evidence import (
    ResidentEvidenceError,
    ResidentModelLoadEvidence,
    load_resident_model_load_evidence,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule

if TYPE_CHECKING:
    from collections.abc import Sequence

CAMPAIGN_DURATION_SECONDS: Final = 10_800
BOOTSTRAP_SEED: Final = 202_608_310_001
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_LOWER_INDEX: Final = 249
BOOTSTRAP_UPPER_INDEX: Final = 9_749
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

type ScaleDecision = Literal["GO-to-scale", "NO-GO-to-scale", "INCONCLUSIVE"]
type EfficiencyDecision = Literal[
    "selective-revision-efficiency-signal", "NO-EFFICIENCY-SIGNAL", "INCONCLUSIVE"
]
type PairDirection = Literal["win", "loss", "tie"]
type ResourceIntegrityReason = Literal[
    "missing-upstream-schema-field:CampaignAccounting.resident_model_load",
    "missing-upstream-schema-field:CampaignAccounting.resident_model_load_evidence",
    "missing-upstream-value:CampaignAccounting.resident_model_load",
    "missing-upstream-value:CampaignAccounting.resident_model_load_evidence",
]


class AnalysisError(Exception):
    """Typed fail-closed analysis or evidence validation error."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Record the stable fail-closed reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CoverageCurve:
    """One repetition-level target-branch curve plus readiness/failure metadata."""

    times: tuple[float, ...]
    branches: tuple[int, ...]
    c0_branches: int
    readiness_seconds: float | None = None
    failure_freeze_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class EffectiveCoverage:
    """Coverage curve after applying the preregistered readiness rule."""

    times: tuple[float, ...]
    branches: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Fixed-seed percentile interval over paired median uplift."""

    lower_uplift: float
    upper_uplift: float
    seed: Literal[202608310001] = BOOTSTRAP_SEED
    resample_count: Literal[10000] = BOOTSTRAP_RESAMPLES


@dataclass(frozen=True, slots=True)
class RankBiserialResult:
    """Rank-biserial direction with an explicit no-nonties marker."""

    effect: float
    all_ties: bool


@dataclass(frozen=True, slots=True)
class EventTime:
    """Observed or administratively right-censored event time."""

    time_seconds: float
    censored: bool


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """All preregistered primary decision gates in one immutable value."""

    complete_pair_count: int
    selective_wins: int
    median_uplift: float
    final_coverage_retention: float
    deep_parse_drop_percentage_points: float
    pair_integrity: bool
    resource_integrity: bool
    evidence_integrity: bool


@dataclass(frozen=True, slots=True)
class EfficiencyInputs:
    """Per-repetition raw and denominator inputs for the RQ2 efficiency signal."""

    selective_auc: tuple[float, ...]
    comparison_auc: tuple[float, ...]
    selective_nfe: tuple[int | None, ...]
    comparison_nfe: tuple[int | None, ...]
    selective_gpu_seconds: tuple[float | None, ...]
    comparison_gpu_seconds: tuple[float | None, ...]


class EvidenceTreeEntry(FrozenModel):
    """One regular-file leaf in the deterministic campaign evidence tree."""

    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class EvidenceTreeContent(FrozenModel):
    """Exact canonical preimage whose bytes are hashed as the evidence-tree root."""

    schema_version: Literal["dllm-openexr-evidence-tree-content/v1"] = (
        "dllm-openexr-evidence-tree-content/v1"
    )
    entries: tuple[EvidenceTreeEntry, ...]


@dataclass(frozen=True, slots=True)
class AuditedArm:
    """One fully dereferenced raw row and its analysis endpoints."""

    row: RawArmReportRow
    coverage: CoverageSeriesEnvelope
    proof: ProofBatchEnvelope | None
    deep_parse: DeepParseEnvelope
    readiness_seconds: float | None
    auc: float
    final_branches: int
    deep_parse_rate: float
    threshold_time: EventTime
    first_proved_cause_time: EventTime
    proved_cause_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignAudit:
    """Immutable result of the complete Task 5 and Task 6 evidence audit."""

    root: Path
    manifest: CampaignManifest
    outcome: CampaignOutcome
    rows: tuple[RawArmReportRow, ...]
    arms: tuple[AuditedArm, ...]
    source_entries: tuple[EvidenceTreeEntry, ...]
    existing_evidence_tree_sha256: str | None


@dataclass(frozen=True, slots=True)
class PairPoint:
    """One confirmatory repetition-level selective/final-only pair."""

    repetition: int
    selective_auc: float
    final_only_auc: float
    log_ratio: float
    uplift: float
    direction: PairDirection


@dataclass(frozen=True, slots=True)
class ArmEventEndpoints:
    """Censored time endpoints and proved-cause membership for one arm cell."""

    repetition: int
    arm: Arm
    coverage_threshold: EventTime
    first_proved_cause: EventTime
    proved_cause_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecondaryContrast:
    """One preregistered RQ2-RQ4 paired contrast."""

    comparison_arm: Literal["full-regenerate", "structured-one-shot", "libfuzzer-only"]
    wins: int
    losses: int
    ties: int
    exact_sign_p: float
    holm_adjusted_p: float
    rank_biserial: RankBiserialResult


@dataclass(frozen=True, slots=True)
class ArmResourceUsage:
    """Logical per-arm resource totals with full shared-bank charging."""

    arm: Arm
    model_nfe: int | None
    model_gpu_seconds: float | None
    search_cpu_seconds: float
    proof_cpu_seconds: float


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Exact producer accounting and conservative resource-integrity status."""

    resource_integrity: bool
    resource_integrity_reasons: tuple[ResourceIntegrityReason, ...]
    bank_physical_initial_requests: int
    bank_logical_initial_requests_by_arm: tuple[tuple[str, int], ...]
    coverage_replay_count: int
    proof_replay_count: int
    search_cpu_seconds: float
    proof_cpu_seconds: float
    proof_wall_seconds: float
    resident_model_load_duration_seconds: float | None
    resident_model_load_rss_bytes: int | None
    resident_model_load_vram_bytes: int | None
    by_arm: tuple[ArmResourceUsage, ...]


@dataclass(frozen=True, slots=True)
class PairedSummary:
    """Canonical JSON-ready paired analysis summary."""

    schema_version: Literal["dllm-openexr-paired-summary/v1"]
    complete_pair_count: int
    invalid_exogenous_repetitions: tuple[int, ...]
    pairs: tuple[PairPoint, ...]
    wins: int
    losses: int
    ties: int
    paired_median_uplift: float
    bootstrap_95_percent: BootstrapInterval | None
    exact_sign_p: float
    rank_biserial: RankBiserialResult
    final_coverage_retention: float
    deep_parse_drop_percentage_points: float
    event_endpoints: tuple[ArmEventEndpoints, ...]
    secondary_contrasts: tuple[SecondaryContrast, ...]
    efficiency_decision: EfficiencyDecision
    decision: ScaleDecision
    resource_usage: ResourceUsage
    proved_cause_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Completed analysis transaction and deterministic output identity."""

    audit: CampaignAudit
    summary: PairedSummary
    evidence_tree_sha256: str
    output_paths: tuple[Path, ...]


def trapezoid_area(times: Sequence[float], values: Sequence[float]) -> float:
    """Integrate one finite, strictly increasing piecewise-linear curve."""
    if len(times) != len(values) or len(times) < 2:
        raise AnalysisError("trapezoid inputs require equal lengths of at least two")
    if any(not math.isfinite(value) for value in (*times, *values)):
        raise AnalysisError("trapezoid inputs must be finite")
    if any(right <= left for left, right in pairwise(times)):
        raise AnalysisError("trapezoid times must be strictly increasing")
    return math.fsum(
        (right_time - left_time) * (left_value + right_value) / 2.0
        for (left_time, right_time), (left_value, right_value) in zip(
            pairwise(times), pairwise(values), strict=True
        )
    )


def effective_coverage(curve: CoverageCurve) -> EffectiveCoverage:
    """Apply no-pre-ready-credit semantics and insert an exact readiness knot."""
    _validate_curve(curve)
    readiness = curve.readiness_seconds
    if readiness is None:
        return EffectiveCoverage(curve.times, tuple(float(value) for value in curve.branches))
    if not math.isfinite(readiness) or readiness < 0.0:
        raise AnalysisError("readiness time must be finite and nonnegative")
    effective_times = list(curve.times)
    if readiness < curve.times[-1] and readiness not in curve.times:
        effective_times.append(readiness)
        effective_times.sort()
    effective_branches: list[float] = []
    by_time = dict(zip(curve.times, curve.branches, strict=True))
    for elapsed in effective_times:
        if elapsed <= readiness:
            effective_branches.append(float(curve.c0_branches))
        else:
            effective_branches.append(float(by_time[elapsed]))
    return EffectiveCoverage(tuple(effective_times), tuple(effective_branches))


def arm_auc(curve: CoverageCurve) -> float:
    """Return normalized three-hour target-branch excess AUC for one arm."""
    effective = effective_coverage(curve)
    if effective.times[0] != 0.0 or effective.times[-1] != CAMPAIGN_DURATION_SECONDS:
        raise AnalysisError("arm AUC requires exact 0..10800 endpoints")
    excess = tuple(value - curve.c0_branches for value in effective.branches)
    result = trapezoid_area(effective.times, excess) / CAMPAIGN_DURATION_SECONDS
    if result <= -1.0:
        raise AnalysisError("normalized arm AUC must be greater than -1")
    return result


def paired_log_ratio(selective_auc: float, final_only_auc: float) -> float:
    """Return the preregistered log ratio for one matched repetition."""
    if not math.isfinite(selective_auc) or not math.isfinite(final_only_auc):
        raise AnalysisError("paired AUC values must be finite")
    if selective_auc <= -1.0 or final_only_auc <= -1.0:
        raise AnalysisError("paired AUC values must be greater than -1")
    return math.log((1.0 + selective_auc) / (1.0 + final_only_auc))


def exact_sign_test(wins: int, losses: int, ties: int = 0) -> float:
    """Return the exact two-sided binomial sign-test p-value, excluding ties."""
    _validate_counts(wins, losses, ties)
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0
    tail = min(wins, losses)
    numerator = 2.0 * math.fsum(float(math.comb(non_ties, count)) for count in range(tail + 1))
    probability = numerator / float(1 << non_ties)
    return min(1.0, probability)


def rank_biserial(wins: int, losses: int) -> RankBiserialResult:
    """Return sign-based rank-biserial effect and explicit all-ties state."""
    _validate_counts(wins, losses, 0)
    non_ties = wins + losses
    if non_ties == 0:
        return RankBiserialResult(effect=0.0, all_ties=True)
    return RankBiserialResult(effect=(wins - losses) / non_ties, all_ties=False)


def paired_bootstrap(log_ratios: Sequence[float]) -> BootstrapInterval:
    """Return the fixed 10,000-resample paired median-uplift percentile CI."""
    if len(log_ratios) != 10:
        raise AnalysisError("paired bootstrap requires exactly ten repetitions")
    values = tuple(log_ratios)
    if any(not math.isfinite(value) for value in values):
        raise AnalysisError("paired bootstrap values must be finite")
    generator = random.Random(BOOTSTRAP_SEED)
    uplifts = [
        math.expm1(statistics.median(values[generator.randrange(len(values))] for _ in values))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    uplifts.sort()
    return BootstrapInterval(
        lower_uplift=uplifts[BOOTSTRAP_LOWER_INDEX],
        upper_uplift=uplifts[BOOTSTRAP_UPPER_INDEX],
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return deterministic Holm step-down adjusted p-values in input order."""
    if not p_values:
        raise AnalysisError("Holm adjustment requires at least one p-value")
    values = tuple(p_values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise AnalysisError("Holm p-values must be finite values in [0, 1]")
    order = tuple(sorted(range(len(values)), key=lambda index: (values[index], index)))
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return tuple(adjusted)


def administrative_event_time(time_seconds: float | None) -> EventTime:
    """Map a missing event to explicit administrative censoring at 10,800 seconds."""
    if time_seconds is None:
        return EventTime(float(CAMPAIGN_DURATION_SECONDS), censored=True)
    if not math.isfinite(time_seconds) or not 0.0 <= time_seconds <= CAMPAIGN_DURATION_SECONDS:
        raise AnalysisError("event time must lie in the closed campaign interval")
    return EventTime(time_seconds, censored=False)


def time_to_coverage_threshold(
    times: Sequence[float], branches: Sequence[int], *, c0_branches: int
) -> EventTime:
    """Return first C0+10 crossing or administrative censoring."""
    if len(times) != len(branches) or not times:
        raise AnalysisError("coverage threshold inputs require equal nonempty lengths")
    _validate_integer_counts(branches)
    if isinstance(c0_branches, bool) or c0_branches < 0:
        raise AnalysisError("C0 branch count must be a nonnegative integer")
    if any(not math.isfinite(value) for value in times):
        raise AnalysisError("coverage threshold times must be finite")
    if any(right <= left for left, right in pairwise(times)):
        raise AnalysisError("coverage threshold times must be strictly increasing")
    threshold = c0_branches + 10
    observed = next(
        (elapsed for elapsed, count in zip(times, branches, strict=True) if count >= threshold),
        None,
    )
    return administrative_event_time(observed)


def time_to_first_proved_cause(
    observed_input_times: Sequence[tuple[str, float]],
    proved_input_sha256: Sequence[str],
) -> EventTime:
    """Return the earliest search observation later accepted by exact proof."""
    proved = frozenset(proved_input_sha256)
    if not proved:
        return administrative_event_time(None)
    observed: dict[str, list[float]] = {}
    for digest, elapsed in observed_input_times:
        if not math.isfinite(elapsed) or not 0.0 <= elapsed <= CAMPAIGN_DURATION_SECONDS:
            raise AnalysisError("search event time lies outside the campaign interval")
        observed.setdefault(digest, []).append(elapsed)
    if not proved.issubset(observed):
        raise AnalysisError("proved input has no matching search event")
    earliest = min(elapsed for digest in proved for elapsed in observed[digest])
    return administrative_event_time(earliest)


def classify_decision(inputs: DecisionInputs) -> ScaleDecision:
    """Classify with invalid/incomplete precedence, then NO-GO, then GO."""
    _validate_decision_inputs(inputs)
    integrity = inputs.pair_integrity and inputs.resource_integrity and inputs.evidence_integrity
    if inputs.complete_pair_count != 10 or not integrity:
        return "INCONCLUSIVE"
    if inputs.median_uplift <= 0.0 or inputs.selective_wins <= 5:
        return "NO-GO-to-scale"
    if (
        inputs.selective_wins >= 9
        and inputs.median_uplift >= 0.10
        and inputs.final_coverage_retention >= 0.95
        and inputs.deep_parse_drop_percentage_points <= 5.0
    ):
        return "GO-to-scale"
    return "INCONCLUSIVE"


def classify_efficiency(inputs: EfficiencyInputs) -> EfficiencyDecision:
    """Classify the preregistered selective-versus-full efficiency signal."""
    series = (
        inputs.selective_auc,
        inputs.comparison_auc,
        inputs.selective_nfe,
        inputs.comparison_nfe,
        inputs.selective_gpu_seconds,
        inputs.comparison_gpu_seconds,
    )
    if any(len(values) != 10 for values in series):
        return "INCONCLUSIVE"
    if any(not math.isfinite(value) for value in (*inputs.selective_auc, *inputs.comparison_auc)):
        raise AnalysisError("efficiency AUC values must be finite")
    denominators = (
        *inputs.selective_nfe,
        *inputs.comparison_nfe,
        *inputs.selective_gpu_seconds,
        *inputs.comparison_gpu_seconds,
    )
    if any(value is None or isinstance(value, bool) or value <= 0 for value in denominators):
        return "INCONCLUSIVE"
    comparison_raw_median = statistics.median(inputs.comparison_auc)
    if comparison_raw_median <= 0.0:
        return "INCONCLUSIVE"
    selective_nfe = tuple(
        auc * 1000.0 / denominator
        for auc, denominator in zip(inputs.selective_auc, inputs.selective_nfe, strict=True)
        if denominator is not None
    )
    comparison_nfe = tuple(
        auc * 1000.0 / denominator
        for auc, denominator in zip(inputs.comparison_auc, inputs.comparison_nfe, strict=True)
        if denominator is not None
    )
    selective_gpu = tuple(
        auc / denominator
        for auc, denominator in zip(inputs.selective_auc, inputs.selective_gpu_seconds, strict=True)
        if denominator is not None
    )
    comparison_gpu = tuple(
        auc / denominator
        for auc, denominator in zip(
            inputs.comparison_auc, inputs.comparison_gpu_seconds, strict=True
        )
        if denominator is not None
    )
    if statistics.median(comparison_nfe) <= 0.0 or statistics.median(comparison_gpu) <= 0.0:
        return "INCONCLUSIVE"
    raw_retained = statistics.median(inputs.selective_auc) >= 0.95 * comparison_raw_median
    nfe_signal = _efficiency_axis_passes(selective_nfe, comparison_nfe)
    gpu_signal = _efficiency_axis_passes(selective_gpu, comparison_gpu)
    if raw_retained and (nfe_signal or gpu_signal):
        return "selective-revision-efficiency-signal"
    return "NO-EFFICIENCY-SIGNAL"


_RESULT_NAMES: Final = frozenset(
    {
        "raw-arm-reports.jsonl",
        "paired-summary.json",
        "plots-data.json",
        "report.md",
        "evidence-tree.sha256",
    }
)
_CORE_CAMPAIGN_NAMES: Final = frozenset(
    {
        "banks",
        "cells",
        "exogenous",
        "mutable",
        "campaign-events.jsonl",
        "campaign-manifest.json",
        "campaign-outcome.json",
        "raw-arm-reports.jsonl",
        "results",
    }
)


def audit_campaign(campaign_root: Path) -> CampaignAudit:
    """Audit a complete Task 5 campaign and every Task 6 analysis prerequisite."""
    root = _validated_campaign_root(campaign_root)
    _reject_unsafe_tree(root)
    manifest = _load_campaign_manifest(root / "campaign-manifest.json")
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=manifest.config.main_start_monotonic_seconds,
        gpu_count=len(manifest.config.resource_map.model_cores),
    )
    try:
        expected_manifest = build_campaign_manifest(manifest.config, schedule)
        outcome = load_complete_campaign_outcome(root, manifest)
    except CoordinatorError as error:
        raise AnalysisError(f"campaign outcome audit failed: {error}") from error
    if manifest != expected_manifest:
        raise AnalysisError("campaign manifest/config/resource/schedule binding is stale")
    if manifest.config.artifact_root != root:
        raise AnalysisError("campaign manifest artifact root mismatch")
    raw_path = root / "raw-arm-reports.jsonl"
    rows = _load_canonical_raw_rows(raw_path)
    _validate_root_shape(root, rows)
    arms = _audit_arm_rows(root, manifest, outcome, rows)
    source_entries = _collect_file_entries(root, exclude_results=True)
    existing_tree = _validate_existing_results(root)
    return CampaignAudit(
        root=root,
        manifest=manifest,
        outcome=outcome,
        rows=rows,
        arms=arms,
        source_entries=source_entries,
        existing_evidence_tree_sha256=existing_tree,
    )


def analyze_campaign(campaign_root: Path) -> AnalysisResult:
    """Audit, analyze, and atomically install all deterministic Task 6 outputs."""
    audit = audit_campaign(campaign_root)
    summary = _summarize(audit)
    output_payloads = (
        ("raw-arm-reports.jsonl", (audit.root / "raw-arm-reports.jsonl").read_bytes()),
        ("paired-summary.json", canonical_json_bytes(summary)),
        ("plots-data.json", _plots_data_bytes(audit, summary)),
        ("report.md", _render_report(audit, summary).encode()),
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{audit.root.name}-results-", dir=audit.root.parent))
    results = audit.root / "results"
    try:
        for name, payload in output_payloads:
            _ = (stage / name).write_bytes(payload)
        virtual_entries = tuple(
            EvidenceTreeEntry(
                path=f"results/{path.name}",
                sha256=sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(stage.iterdir(), key=lambda item: item.name)
        )
        tree = EvidenceTreeContent(
            entries=tuple(
                sorted(
                    (*audit.source_entries, *virtual_entries),
                    key=lambda entry: entry.path,
                )
            )
        )
        tree_bytes = canonical_json_bytes(tree.model_dump(mode="json"))
        _ = (stage / "evidence-tree.sha256").write_bytes(tree_bytes)
        tree_sha256 = sha256(tree_bytes).hexdigest()
        if _collect_file_entries(audit.root, exclude_results=True) != audit.source_entries:
            raise AnalysisError("campaign evidence changed during analysis")
        if results.exists():
            _require_identical_result_tree(results, stage)
        else:
            _ = stage.replace(results)
        output_paths = tuple(results / name for name in sorted(_RESULT_NAMES))
        return AnalysisResult(audit, summary, tree_sha256, output_paths)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _validated_campaign_root(path: Path) -> Path:
    if path.is_symlink():
        raise AnalysisError("campaign root may not be a symlink")
    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise AnalysisError("campaign root is unavailable") from error
    if not root.is_dir() or root == Path("/"):
        raise AnalysisError("campaign root must be a private directory")
    return root


def _reject_unsafe_tree(root: Path) -> None:
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AnalysisError(f"campaign evidence contains symlink: {path}")
        try:
            relative = path.relative_to(root)
            resolved_relative = path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise AnalysisError("campaign evidence path escapes root") from error
        posix = relative.as_posix()
        if PurePosixPath(posix).as_posix() != posix or resolved_relative.as_posix() != posix:
            raise AnalysisError("campaign evidence path is not canonical")
        if posix in seen:
            raise AnalysisError("campaign evidence contains duplicate relative paths")
        seen.add(posix)
        if not path.is_dir() and not path.is_file():
            raise AnalysisError(f"campaign evidence is not a regular file: {posix}")


def _load_canonical_model[ModelT: FrozenModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        raw = path.read_bytes()
        model = model_type.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise AnalysisError(f"canonical evidence unavailable: {path.name}") from error
    if raw != canonical_json_bytes(model.model_dump(mode="json")):
        raise AnalysisError(f"noncanonical evidence artifact: {path.name}")
    return model


def _load_campaign_manifest(path: Path) -> CampaignManifest:
    try:
        raw = path.read_bytes()
        manifest = CampaignManifest.model_validate(_JSON_OBJECT.validate_json(raw))
    except (OSError, ValidationError) as error:
        raise AnalysisError("canonical evidence unavailable: campaign-manifest.json") from error
    if raw != canonical_json_bytes(manifest.model_dump(mode="json")):
        raise AnalysisError("noncanonical evidence artifact: campaign-manifest.json")
    return manifest


def _load_canonical_raw_rows(path: Path) -> tuple[RawArmReportRow, ...]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AnalysisError("raw arm report is unavailable") from error
    if not raw.endswith(b"\n") or b"\n\n" in raw:
        raise AnalysisError("raw arm report JSONL is not canonical")
    rows: list[RawArmReportRow] = []
    try:
        for line in raw.splitlines():
            row = RawArmReportRow.model_validate_json(line)
            if line != canonical_json_bytes(row.model_dump(mode="json")):
                raise AnalysisError("raw arm report row is not canonical")
            rows.append(row)
    except ValidationError as error:
        raise AnalysisError("raw arm report schema is invalid") from error
    if len(rows) != 50:
        raise AnalysisError("raw arm report must contain exactly 50 rows")
    return tuple(rows)


def _validate_root_shape(root: Path, rows: tuple[RawArmReportRow, ...]) -> None:
    for entry in root.iterdir():
        if entry.name not in _CORE_CAMPAIGN_NAMES:
            raise AnalysisError(f"unknown campaign artifact: {entry.name}")
    bank_root = root / "banks"
    expected_banks = {f"rep-{repetition:02d}" for repetition in range(10)}
    if not bank_root.is_dir() or {path.name for path in bank_root.iterdir()} != expected_banks:
        raise AnalysisError("candidate-bank evidence set is incomplete or unknown")
    if any(
        {path.name for path in (bank_root / name).iterdir()} != {"candidate-bank.json"}
        for name in expected_banks
    ):
        raise AnalysisError("candidate-bank sealed artifact set is unknown")
    expected_cells = {row.artifact_root for row in rows}
    if len(expected_cells) != 50:
        raise AnalysisError("raw arm report contains duplicate cell roots")
    for cell_root in expected_cells:
        _require_below(root / "cells", cell_root, "cell artifact root")
        _validate_cell_top_level(cell_root)
    actual_cells = {path for path in (root / "cells").glob("wave-*/rep-*/*") if path.is_dir()}
    if actual_cells != expected_cells:
        raise AnalysisError("cell evidence grid contains unknown or missing roots")


def _validate_cell_top_level(cell_root: Path) -> None:
    allowed = {
        "attempts",
        "corpus",
        "corpus-index.jsonl",
        "corpus-inputs",
        "corpus-release.json",
        "coverage",
        "crashes",
        "events.jsonl",
        "final-result.json",
        "logs",
        "model-prep-failure.json",
        "model-prep-result.json",
        "model-prep-staging",
        "post-search",
        "pre-release-corpus",
        "proof",
        "run-manifest.json",
        "search-failure.json",
        "seed-manifest.json",
    }
    unknown = {path.name for path in cell_root.iterdir()} - allowed
    if unknown:
        raise AnalysisError(f"unknown sealed cell artifact: {min(unknown)}")
    post_search = cell_root / "post-search"
    if post_search.exists() and {path.name for path in post_search.iterdir()} - {
        "coverage-envelope.json",
        "deep-parse-envelope.json",
        "deep-parse-request.json",
        "deep-parse-result.json",
        "deep-parse-test-evidence",
        "proof-envelope.json",
    }:
        raise AnalysisError("unknown post-search evidence artifact")


def _require_below(root: Path, path: Path, label: str) -> None:
    try:
        _ = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise AnalysisError(f"{label} escapes campaign root") from error


def _validate_existing_results(root: Path) -> str | None:
    results = root / "results"
    if not results.exists():
        return None
    names = frozenset(path.name for path in results.iterdir())
    if not results.is_dir() or names != _RESULT_NAMES:
        raise AnalysisError("results artifact set is unknown or partial")
    tree_path = results / "evidence-tree.sha256"
    tree = _load_canonical_model(tree_path, EvidenceTreeContent)
    expected = EvidenceTreeContent(entries=_collect_file_entries(root, exclude_results=False))
    if tree != expected:
        raise AnalysisError("existing evidence tree does not match exact campaign bytes")
    return sha256(tree_path.read_bytes()).hexdigest()


def _collect_file_entries(root: Path, *, exclude_results: bool) -> tuple[EvidenceTreeEntry, ...]:
    entries: list[EvidenceTreeEntry] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            continue
        if relative == "results/evidence-tree.sha256":
            continue
        if exclude_results and relative.startswith("results/"):
            continue
        content = path.read_bytes()
        entries.append(
            EvidenceTreeEntry(
                path=relative,
                sha256=sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    entries.sort(key=lambda entry: entry.path)
    paths = tuple(entry.path for entry in entries)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise AnalysisError("evidence tree entries are not unique sorted POSIX paths")
    return tuple(entries)


def _require_identical_result_tree(existing: Path, staged: Path) -> None:
    if frozenset(path.name for path in existing.iterdir()) != _RESULT_NAMES:
        raise AnalysisError("results artifact set is unknown or partial")
    for name in sorted(_RESULT_NAMES):
        if (existing / name).read_bytes() != (staged / name).read_bytes():
            raise AnalysisError("deterministic rerun output mismatch")


def _audit_arm_rows(
    root: Path,
    manifest: CampaignManifest,
    outcome: CampaignOutcome,
    rows: tuple[RawArmReportRow, ...],
) -> tuple[AuditedArm, ...]:
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=manifest.config.main_start_monotonic_seconds,
        gpu_count=len(manifest.config.resource_map.model_cores),
    )
    outcomes = {cell.cell_id: cell for cell in outcome.cells}
    audited: list[AuditedArm] = []
    branch_total_by_binary: dict[str, int] = {}
    for row, scheduled in zip(rows, schedule.cells, strict=True):
        cell = outcomes.get(row.cell_id)
        if cell is None:
            raise AnalysisError("raw row has no matching campaign outcome")
        if (
            row.campaign_root != root
            or row.manifest_sha256 != manifest.manifest_sha256
            or row.resource_map_sha256 != manifest.resource_map_sha256
            or row.repetition != int(scheduled.repetition)
            or row.wave_index != int(scheduled.wave_index)
            or row.arm != scheduled.arm
            or row.fuzzer_seed != int(scheduled.fuzzer_seed)
            or cell.deadline_monotonic_seconds != float(scheduled.deadline_monotonic_seconds)
            or cell.fuzzer_seed != row.fuzzer_seed
        ):
            raise AnalysisError("raw row seed/deadline/manifest/resource schedule mismatch")
        if row.pair_status == "exogenous-invalidated":
            continue
        coverage_path = row.artifact_root / "post-search" / "coverage-envelope.json"
        deep_parse_path = row.artifact_root / "post-search" / "deep-parse-envelope.json"
        if row.coverage_series_path != coverage_path:
            raise AnalysisError("raw row has wrong coverage path")
        if row.deep_parse_envelope_path != deep_parse_path:
            raise AnalysisError("raw row has wrong deep-parse path")
        coverage = _load_canonical_model(coverage_path, CoverageSeriesEnvelope)
        deep_parse = _load_canonical_model(deep_parse_path, DeepParseEnvelope)
        _validate_request_bindings(row, manifest, scheduled.deadline_monotonic_seconds, coverage)
        _validate_coverage_grid(coverage, branch_total_by_binary)
        proof = _load_row_proof(row)
        causes = _validate_proved_causes(proof)
        first_cause = _proved_cause_event_time(row, coverage, proof)
        readiness = _analysis_readiness(row, coverage)
        freeze = _treatment_freeze(row, coverage)
        points = coverage.coverage_series.points
        curve = CoverageCurve(
            times=tuple(float(point.scheduled_elapsed_seconds) for point in points),
            branches=tuple(point.branches_covered for point in points),
            c0_branches=points[0].branches_covered,
            readiness_seconds=readiness,
            failure_freeze_seconds=freeze,
        )
        auc = arm_auc(curve)
        effective = effective_coverage(curve)
        threshold = time_to_coverage_threshold(
            effective.times,
            tuple(int(value) for value in effective.branches),
            c0_branches=curve.c0_branches,
        )
        deep_result = deep_parse.deep_parse_result
        audited.append(
            AuditedArm(
                row=row,
                coverage=coverage,
                proof=proof,
                deep_parse=deep_parse,
                readiness_seconds=readiness,
                auc=auc,
                final_branches=points[-1].branches_covered,
                deep_parse_rate=deep_result.success_count / deep_result.total_count,
                threshold_time=threshold,
                first_proved_cause_time=first_cause,
                proved_cause_ids=causes,
            )
        )
    _validate_repetition_c0(tuple(audited), manifest.config.c0_sha256)
    return tuple(audited)


def _validate_request_bindings(
    row: RawArmReportRow,
    manifest: CampaignManifest,
    deadline: float,
    coverage: CoverageSeriesEnvelope,
) -> None:
    config = manifest.config
    request = coverage.cell_request
    coverage_request = coverage.coverage_request
    expected_index = row.artifact_root / "corpus-index.jsonl"
    if (
        request.cell_id != row.cell_id
        or request.artifact_root != row.artifact_root
        or request.fuzzer_seed != row.fuzzer_seed
        or request.deadline_monotonic_seconds != deadline
        or request.config_sha256 != config.config_sha256
        or request.harness_sha256 != config.target_harness_sha256
        or request.build.image_digest != config.search_image_digest
        or request.build.binary_sha256 != config.search_binary_sha256
        or coverage_request.target_commit != config.target_commit
        or coverage_request.image_digest != config.coverage_image_digest
        or coverage_request.binary_sha256 != config.coverage_binary_sha256
        or coverage_request.build_manifest_sha256 != config.coverage_build_manifest_sha256
        or coverage_request.corpus_index.artifact_root != row.artifact_root
        or coverage_request.corpus_index.index_path != expected_index
    ):
        raise AnalysisError("row request/config/coverage provenance mismatch")
    c0_path = row.artifact_root / "pre-release-corpus" / f"c0-{config.c0_sha256}.seed"
    if not c0_path.exists():
        c0_path = row.artifact_root / "corpus" / f"c0-{config.c0_sha256}.seed"
    _require_below(row.artifact_root, c0_path, "C0 evidence path")
    content = c0_path.read_bytes()
    if (
        sha256(content).hexdigest() != config.c0_sha256
        or row.c0_baseline_sha256 != config.c0_sha256
    ):
        raise AnalysisError("C0 source bytes do not match the immutable campaign baseline")


def _validate_coverage_grid(
    envelope: CoverageSeriesEnvelope, branch_total_by_binary: dict[str, int]
) -> None:
    points = envelope.coverage_series.points
    rows = envelope.coverage_series.corpus_index
    times = tuple(point.scheduled_elapsed_seconds for point in points)
    row_times = tuple(row.scheduled_elapsed_seconds for row in rows)
    if times != SCHEDULE_SECONDS or row_times != SCHEDULE_SECONDS:
        raise AnalysisError("coverage requires exact integer 0..10800 schedule")
    if len(points) != len(rows):
        raise AnalysisError("coverage point/corpus-index sequence mismatch")
    totals = {point.branches_total for point in points}
    if len(totals) != 1:
        raise AnalysisError("mixed branch totals for one qualified coverage binary")
    total = next(iter(totals))
    if any(
        isinstance(point.branches_covered, bool)
        or isinstance(point.branches_total, bool)
        or not 0 <= point.branches_covered <= point.branches_total
        for point in points
    ):
        raise AnalysisError("coverage must use integer target-branch counts, not percentages")
    binary = envelope.coverage_request.binary_sha256
    prior_total = branch_total_by_binary.setdefault(binary, total)
    if prior_total != total:
        raise AnalysisError("mixed branch totals for the same qualified coverage binary")
    try:
        validate_coverage_series_backings(
            envelope.coverage_request.corpus_index.artifact_root,
            envelope.coverage_series,
        )
    except CoverageSeriesError as error:
        raise AnalysisError(str(error)) from error
    for point, corpus_row in zip(points, rows, strict=True):
        if (
            point.observation_kind != corpus_row.observation_kind
            or point.corpus_set_sha256 != corpus_row.cumulative_set_sha256
            or point.replayed_input_sha256 != tuple(item.sha256 for item in corpus_row.new_inputs)
        ):
            raise AnalysisError("coverage point is not bound to its corpus-index row")


def _load_row_proof(row: RawArmReportRow) -> ProofBatchEnvelope | None:
    if row.proof_status == "unavailable":
        if row.terminal_reason != "treatment-failed":
            raise AnalysisError("complete row is missing proof evidence")
        return None
    expected = row.artifact_root / "post-search" / "proof-envelope.json"
    if row.proof_result_path != expected:
        raise AnalysisError("raw row has wrong proof path")
    envelope = _load_canonical_model(expected, ProofBatchEnvelope)
    if envelope.proof_result.verdicts:
        _require_below(row.artifact_root, envelope.proof_request.proof_root, "proof evidence root")
    _validate_proof_backings(envelope)
    return envelope


def _validate_proof_backings(proof: ProofBatchEnvelope) -> None:
    root = proof.proof_request.proof_root
    for verdict in proof.proof_result.verdicts:
        if (
            verdict.proof_log_evidence.sha256 != verdict.proof_log_sha256
            or verdict.result_evidence.sha256 != verdict.result_sha256
            or len(verdict.run_argv_evidence) != len(verdict.run_argv_sha256)
        ):
            raise AnalysisError("proof backing evidence hash or count mismatch")
        evidence = (
            (verdict.proof_log_sha256, verdict.proof_log_evidence),
            *zip(verdict.run_argv_sha256, verdict.run_argv_evidence, strict=True),
            (verdict.result_sha256, verdict.result_evidence),
        )
        for expected, descriptor in evidence:
            _require_backing_file(root, expected, descriptor)


def _require_backing_file(
    root: Path,
    expected_sha256: str,
    evidence: ArtifactBackingEvidence | None,
) -> None:
    if evidence is None:
        raise AnalysisError("backing evidence descriptor is missing")
    if evidence.sha256 != expected_sha256:
        raise AnalysisError("backing evidence hash mismatch")
    try:
        _ = validate_backing_file(root, evidence)
    except EvidenceBackingError as error:
        raise AnalysisError(str(error)) from error


def _validate_proved_causes(proof: ProofBatchEnvelope | None) -> tuple[str, ...]:
    if proof is None:
        return ()
    result = proof.proof_result
    proved = tuple(verdict for verdict in result.verdicts if verdict.oracle_status == "proved")
    if any(verdict.replay_count != 2 or len(verdict.run_argv_sha256) != 2 for verdict in proved):
        raise AnalysisError("proved cause lacks exactly two matching fresh replays")
    verdict_causes = {verdict.cause_id for verdict in proved}
    indexed_causes = {cause.cause_id for cause in result.cause_index.causes}
    if None in verdict_causes or verdict_causes != indexed_causes:
        raise AnalysisError("proved cause index does not exactly match proved verdicts")
    return tuple(sorted(indexed_causes))


def _proved_cause_event_time(
    row: RawArmReportRow,
    coverage: CoverageSeriesEnvelope,
    proof: ProofBatchEnvelope | None,
) -> EventTime:
    if proof is None:
        return administrative_event_time(None)
    proved = tuple(
        verdict for verdict in proof.proof_result.verdicts if verdict.oracle_status == "proved"
    )
    if not proved:
        return administrative_event_time(None)
    try:
        events = load_cell_events(coverage.cell_request)
    except CellEvidenceError as error:
        raise AnalysisError("proved-cause search event evidence is invalid") from error
    observations: list[tuple[str, float]] = []
    for event in events:
        elapsed = event.observed_monotonic_seconds - coverage.cell_request.start_monotonic_seconds
        observations.extend(
            (artifact.sha256, elapsed)
            for artifact in event.artifacts
            if artifact.kind == "sanitizer"
        )
    for verdict in proved:
        owner = verdict.first_owner
        owner_matches = any(
            event.attempt_index == owner.attempt_index
            and event.log_sha256 == owner.search_log_sha256
            and any(
                artifact.kind == "sanitizer"
                and artifact.sha256 == owner.input_sha256
                and artifact.stored_path == owner.input_path
                and artifact.size_bytes == owner.size_bytes
                for artifact in event.artifacts
            )
            for event in events
        )
        if owner.cell_id != row.cell_id or not owner_matches:
            raise AnalysisError("proved input has no matching search event")
    return time_to_first_proved_cause(
        observations,
        tuple(verdict.input_sha256 for verdict in proved),
    )


def _analysis_readiness(row: RawArmReportRow, coverage: CoverageSeriesEnvelope) -> float | None:
    if row.arm not in MODEL_ARMS:
        return None
    release_path = row.artifact_root / "corpus-release.json"
    if not release_path.exists():
        if row.terminal_reason in {"treatment-failed", "expired-frozen"}:
            return float(CAMPAIGN_DURATION_SECONDS)
        raise AnalysisError("model readiness evidence is not dereferenceable")
    release = _load_canonical_model(release_path, CorpusReleaseEvidence)
    unsigned = canonical_json_bytes(release.model_dump(mode="json", exclude={"evidence_sha256"}))
    t0 = coverage.coverage_series.points[0]
    if (
        release.evidence_sha256 != sha256(unsigned).hexdigest()
        or release.cell_id != row.cell_id
        or release.bank_ready_elapsed_seconds != row.bank_ready_elapsed_seconds
        or release.pre_release_corpus_set_sha256 != t0.corpus_set_sha256
        or release.corpus_release_elapsed_seconds < release.bank_ready_elapsed_seconds
        or release.corpus_release_elapsed_seconds < release.model_ready_elapsed_seconds
    ):
        raise AnalysisError("model readiness/release evidence is stale or inconsistent")
    return release.corpus_release_elapsed_seconds


def _treatment_freeze(row: RawArmReportRow, coverage: CoverageSeriesEnvelope) -> float | None:
    if row.terminal_reason != "treatment-failed":
        return None
    frozen = tuple(
        item.source_elapsed_seconds
        for item in coverage.coverage_series.corpus_index
        if item.observation_kind == "frozen"
    )
    if not frozen:
        raise AnalysisError("treatment failure has no explicit frozen coverage points")
    return min(frozen)


def _validate_repetition_c0(arms: tuple[AuditedArm, ...], expected_sha256: str) -> None:
    for repetition in range(10):
        repeated = tuple(item for item in arms if item.row.repetition == repetition)
        if not repeated:
            continue
        if len(repeated) != 5:
            raise AnalysisError("complete repetition does not contain all five arms")
        t0_points = tuple(item.coverage.coverage_series.points[0] for item in repeated)
        baselines = {
            (
                point.corpus_set_sha256,
                point.branches_covered,
                point.branches_total,
                item.coverage.coverage_request.binary_sha256,
                item.coverage.coverage_request.build_manifest_sha256,
            )
            for item, point in zip(repeated, t0_points, strict=True)
        }
        if len(baselines) != 1:
            raise AnalysisError("five arms do not share exact scheduled-t0 C0 coverage provenance")
        if any(
            item.coverage.coverage_series.corpus_index[0].cumulative_hashes != (expected_sha256,)
            for item in repeated
        ):
            raise AnalysisError("scheduled-t0 coverage envelope is not C0-only")


def _summarize(audit: CampaignAudit) -> PairedSummary:
    by_key: dict[tuple[int, Arm], AuditedArm] = {
        (item.row.repetition, item.row.arm): item for item in audit.arms
    }
    invalid = tuple(
        sorted({row.repetition for row in audit.rows if row.pair_status == "exogenous-invalidated"})
    )
    complete_repetitions = tuple(
        repetition
        for repetition in range(10)
        if repetition not in invalid and all((repetition, arm) in by_key for arm in ARMS)
    )
    pairs = tuple(
        _pair_point(
            repetition,
            by_key[(repetition, "selective-revision")],
            by_key[(repetition, "final-only")],
        )
        for repetition in complete_repetitions
    )
    wins = sum(pair.direction == "win" for pair in pairs)
    losses = sum(pair.direction == "loss" for pair in pairs)
    ties = sum(pair.direction == "tie" for pair in pairs)
    paired_median_uplift = (
        math.expm1(statistics.median(pair.log_ratio for pair in pairs)) if pairs else 0.0
    )
    bootstrap = (
        paired_bootstrap(tuple(pair.log_ratio for pair in pairs)) if len(pairs) == 10 else None
    )
    selective = tuple(by_key[(rep, "selective-revision")] for rep in complete_repetitions)
    final_only = tuple(by_key[(rep, "final-only")] for rep in complete_repetitions)
    final_retention = _median_ratio(
        tuple(item.final_branches for item in selective),
        tuple(item.final_branches for item in final_only),
    )
    deep_parse_drop = 100.0 * (
        _median_or_zero(tuple(item.deep_parse_rate for item in final_only))
        - _median_or_zero(tuple(item.deep_parse_rate for item in selective))
    )
    contrasts = _secondary_contrasts(by_key, complete_repetitions)
    efficiency = _campaign_efficiency(by_key, complete_repetitions)
    resource = _resource_usage(audit)
    decision = classify_decision(
        DecisionInputs(
            complete_pair_count=len(pairs),
            selective_wins=wins,
            median_uplift=paired_median_uplift,
            final_coverage_retention=final_retention,
            deep_parse_drop_percentage_points=deep_parse_drop,
            pair_integrity=len(pairs) == 10 and not invalid,
            resource_integrity=resource.resource_integrity,
            evidence_integrity=True,
        )
    )
    causes = tuple(sorted({cause for item in audit.arms for cause in item.proved_cause_ids}))
    endpoints = tuple(
        ArmEventEndpoints(
            repetition=item.row.repetition,
            arm=item.row.arm,
            coverage_threshold=item.threshold_time,
            first_proved_cause=item.first_proved_cause_time,
            proved_cause_ids=item.proved_cause_ids,
        )
        for item in audit.arms
    )
    return PairedSummary(
        schema_version="dllm-openexr-paired-summary/v1",
        complete_pair_count=len(pairs),
        invalid_exogenous_repetitions=invalid,
        pairs=pairs,
        wins=wins,
        losses=losses,
        ties=ties,
        paired_median_uplift=paired_median_uplift,
        bootstrap_95_percent=bootstrap,
        exact_sign_p=exact_sign_test(wins, losses, ties),
        rank_biserial=rank_biserial(wins, losses),
        final_coverage_retention=final_retention,
        deep_parse_drop_percentage_points=deep_parse_drop,
        event_endpoints=endpoints,
        secondary_contrasts=contrasts,
        efficiency_decision=efficiency,
        decision=decision,
        resource_usage=resource,
        proved_cause_ids=causes,
    )


def _pair_point(repetition: int, selective: AuditedArm, final_only: AuditedArm) -> PairPoint:
    log_ratio = paired_log_ratio(selective.auc, final_only.auc)
    if log_ratio > 0.0:
        direction: PairDirection = "win"
    elif log_ratio < 0.0:
        direction = "loss"
    else:
        direction = "tie"
    return PairPoint(
        repetition=repetition,
        selective_auc=selective.auc,
        final_only_auc=final_only.auc,
        log_ratio=log_ratio,
        uplift=math.expm1(log_ratio),
        direction=direction,
    )


def _secondary_contrasts(
    by_key: dict[tuple[int, Arm], AuditedArm], repetitions: tuple[int, ...]
) -> tuple[SecondaryContrast, ...]:
    comparison_arms: tuple[
        Literal["full-regenerate", "structured-one-shot", "libfuzzer-only"], ...
    ] = ("full-regenerate", "structured-one-shot", "libfuzzer-only")
    counts: list[tuple[int, int, int, float, RankBiserialResult]] = []
    for arm in comparison_arms:
        differences = tuple(
            by_key[(repetition, "selective-revision")].auc - by_key[(repetition, arm)].auc
            for repetition in repetitions
        )
        wins = sum(value > 0.0 for value in differences)
        losses = sum(value < 0.0 for value in differences)
        ties = sum(value == 0.0 for value in differences)
        counts.append(
            (wins, losses, ties, exact_sign_test(wins, losses, ties), rank_biserial(wins, losses))
        )
    adjusted = holm_adjust(tuple(item[3] for item in counts))
    return tuple(
        SecondaryContrast(
            comparison_arm=arm,
            wins=count[0],
            losses=count[1],
            ties=count[2],
            exact_sign_p=count[3],
            holm_adjusted_p=adjusted[index],
            rank_biserial=count[4],
        )
        for index, (arm, count) in enumerate(zip(comparison_arms, counts, strict=True))
    )


def _campaign_efficiency(
    by_key: dict[tuple[int, Arm], AuditedArm], repetitions: tuple[int, ...]
) -> EfficiencyDecision:
    selective = tuple(by_key[(repetition, "selective-revision")] for repetition in repetitions)
    comparison = tuple(by_key[(repetition, "full-regenerate")] for repetition in repetitions)
    selective_costs = tuple(_model_cost(item.row) for item in selective)
    comparison_costs = tuple(_model_cost(item.row) for item in comparison)
    return classify_efficiency(
        EfficiencyInputs(
            selective_auc=tuple(item.auc for item in selective),
            comparison_auc=tuple(item.auc for item in comparison),
            selective_nfe=tuple(item[0] for item in selective_costs),
            comparison_nfe=tuple(item[0] for item in comparison_costs),
            selective_gpu_seconds=tuple(item[1] for item in selective_costs),
            comparison_gpu_seconds=tuple(item[1] for item in comparison_costs),
        )
    )


def _model_cost(row: RawArmReportRow) -> tuple[int | None, float | None]:
    if row.arm not in MODEL_ARMS or row.bank_evidence_path is None:
        return None, None
    if row.model_nfe is None or row.model_gpu_ms is None:
        return None, None
    bank = _load_canonical_model(row.bank_evidence_path, CandidateBankEvidence)
    return (
        row.model_nfe + bank.physical_cost.nfe,
        (row.model_gpu_ms + bank.physical_cost.gpu_ms) / 1000.0,
    )


def _resource_usage(audit: CampaignAudit) -> ResourceUsage:
    by_arm: list[ArmResourceUsage] = []
    for arm in ARMS:
        rows = tuple(row for row in audit.rows if row.arm == arm)
        audited = tuple(item for item in audit.arms if item.row.arm == arm)
        costs = tuple(_model_cost(row) for row in rows)
        nfe_values = tuple(value[0] for value in costs)
        gpu_values = tuple(value[1] for value in costs)
        model_nfe = (
            None
            if arm not in MODEL_ARMS or any(value is None for value in nfe_values)
            else sum(value for value in nfe_values if value is not None)
        )
        model_gpu = (
            None
            if arm not in MODEL_ARMS or any(value is None for value in gpu_values)
            else math.fsum(value for value in gpu_values if value is not None)
        )
        by_arm.append(
            ArmResourceUsage(
                arm=arm,
                model_nfe=model_nfe,
                model_gpu_seconds=model_gpu,
                search_cpu_seconds=math.fsum(row.search_cpu_seconds for row in rows),
                proof_cpu_seconds=math.fsum(item.row.proof_cpu_seconds or 0.0 for item in audited),
            )
        )
    accounting = audit.outcome.accounting
    logical = tuple(
        (arm, accounting.bank_logical_initial_requests_by_arm[arm]) for arm in MODEL_ARMS
    )
    resident_load, integrity_reasons = _resident_model_load(audit)
    return ResourceUsage(
        resource_integrity=not integrity_reasons,
        resource_integrity_reasons=integrity_reasons,
        bank_physical_initial_requests=accounting.bank_physical_initial_requests,
        bank_logical_initial_requests_by_arm=logical,
        coverage_replay_count=accounting.coverage_replay_count,
        proof_replay_count=accounting.proof_replay_count,
        search_cpu_seconds=accounting.search_cpu_seconds,
        proof_cpu_seconds=math.fsum(row.proof_cpu_seconds or 0.0 for row in audit.rows),
        proof_wall_seconds=math.fsum(row.proof_wall_seconds or 0.0 for row in audit.rows),
        resident_model_load_duration_seconds=(
            None if resident_load is None else resident_load.total_load_seconds
        ),
        resident_model_load_rss_bytes=(
            None if resident_load is None else resident_load.peak_process_rss_bytes
        ),
        resident_model_load_vram_bytes=(
            None if resident_load is None else resident_load.peak_vram_bytes
        ),
        by_arm=tuple(by_arm),
    )


def _resident_model_load(
    audit: CampaignAudit,
) -> tuple[ResidentModelLoadEvidence | None, tuple[ResourceIntegrityReason, ...]]:
    accounting = audit.outcome.accounting
    reasons: list[ResourceIntegrityReason] = []
    if "resident_model_load" not in CampaignAccounting.model_fields:
        reasons.append("missing-upstream-schema-field:CampaignAccounting.resident_model_load")
    if "resident_model_load_evidence" not in CampaignAccounting.model_fields:
        reasons.append(
            "missing-upstream-schema-field:CampaignAccounting.resident_model_load_evidence"
        )
    if reasons:
        return None, tuple(reasons)
    load = cast("object | None", getattr(accounting, "resident_model_load", None))
    backing = cast("object | None", getattr(accounting, "resident_model_load_evidence", None))
    if load is None:
        reasons.append("missing-upstream-value:CampaignAccounting.resident_model_load")
    if backing is None:
        reasons.append("missing-upstream-value:CampaignAccounting.resident_model_load_evidence")
    if reasons:
        return None, tuple(reasons)
    if not isinstance(load, ResidentModelLoadEvidence) or not isinstance(
        backing, ArtifactBackingEvidence
    ):
        raise AnalysisError("resident model-load accounting has an invalid typed value")
    try:
        reopened = load_resident_model_load_evidence(
            audit.root,
            backing,
            expected_worker_pids=audit.outcome.resident_worker_pids,
        )
    except ResidentEvidenceError as error:
        raise AnalysisError(str(error)) from error
    if reopened != load:
        raise AnalysisError("resident model-load accounting does not match exact backing bytes")
    return reopened, ()


def _median_ratio(numerator: tuple[int, ...], denominator: tuple[int, ...]) -> float:
    if not numerator or not denominator:
        return 0.0
    denominator_median = statistics.median(denominator)
    if denominator_median <= 0.0:
        raise AnalysisError("comparison final-coverage median must be positive")
    return statistics.median(numerator) / denominator_median


def _median_or_zero(values: tuple[float, ...]) -> float:
    return statistics.median(values) if values else 0.0


def _plots_data_bytes(audit: CampaignAudit, summary: PairedSummary) -> bytes:
    coverage: list[dict[str, JsonValue]] = []
    for item in audit.arms:
        curve = CoverageCurve(
            times=tuple(
                float(point.scheduled_elapsed_seconds)
                for point in item.coverage.coverage_series.points
            ),
            branches=tuple(
                point.branches_covered for point in item.coverage.coverage_series.points
            ),
            c0_branches=item.coverage.coverage_series.points[0].branches_covered,
            readiness_seconds=item.readiness_seconds,
            failure_freeze_seconds=_treatment_freeze(item.row, item.coverage),
        )
        effective = effective_coverage(curve)
        coverage.append(
            {
                "arm": item.row.arm,
                "repetition": item.row.repetition,
                "c0_branches": curve.c0_branches,
                "readiness_seconds": item.readiness_seconds,
                "scheduled_times_seconds": list(curve.times),
                "scheduled_branches_covered": list(curve.branches),
                "effective_times_seconds": list(effective.times),
                "effective_branches_covered": list(effective.branches),
                "normalized_excess_auc": item.auc,
                "threshold_event": {
                    "time_seconds": item.threshold_time.time_seconds,
                    "censored": item.threshold_time.censored,
                },
                "first_proved_cause_event": {
                    "time_seconds": item.first_proved_cause_time.time_seconds,
                    "censored": item.first_proved_cause_time.censored,
                },
                "proved_cause_ids": list(item.proved_cause_ids),
            }
        )
    payload = {
        "schema_version": "dllm-openexr-plots-data/v1",
        "coverage": coverage,
        "paired_log_ratios": [
            {
                "repetition": pair.repetition,
                "log_ratio": pair.log_ratio,
                "uplift": pair.uplift,
                "direction": pair.direction,
            }
            for pair in summary.pairs
        ],
    }
    return canonical_json_bytes(payload)


def _render_report(audit: CampaignAudit, summary: PairedSummary) -> str:
    invalid = (
        ", ".join(str(value) for value in summary.invalid_exogenous_repetitions)
        if summary.invalid_exogenous_repetitions
        else "none"
    )
    bootstrap = summary.bootstrap_95_percent
    interval = (
        "unavailable (fewer than 10 complete pairs)"
        if bootstrap is None
        else f"[{_number(bootstrap.lower_uplift)}, {_number(bootstrap.upper_uplift)}]"
    )
    lines = [
        "# OpenEXR one-day paired analysis",
        "",
        "## Run completeness",
        "",
        f"- Raw rows: {len(audit.rows)}/50",
        f"- Complete valid confirmatory pairs: {summary.complete_pair_count}/10",
        f"- Exogenous-invalidated repetitions: {invalid}",
        f"- Classification: {summary.decision}",
        "",
        "## Confirmatory selective-revision versus final-only",
        "",
        "| repetition | selective Y | final-only Y | L | uplift | direction |",
        "| ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    lines.extend(
        "".join(
            (
                f"| {pair.repetition} | {_number(pair.selective_auc)} | ",
                f"{_number(pair.final_only_auc)} | {_number(pair.log_ratio)} | ",
                f"{_number(pair.uplift)} | {pair.direction} |",
            )
        )
        for pair in summary.pairs
    )
    lines.extend(
        [
            "",
            f"- Wins/losses/ties: {summary.wins}/{summary.losses}/{summary.ties}",
            f"- Paired median uplift: {_number(summary.paired_median_uplift)}",
            f"- Fixed-seed bootstrap 95% CI: {interval}",
            f"- Exact two-sided sign p-value: {_number(summary.exact_sign_p)}",
            f"- Rank-biserial effect: {_number(summary.rank_biserial.effect)}",
            f"- Rank-biserial all-ties: {str(summary.rank_biserial.all_ties).lower()}",
            f"- Median final-coverage retention: {_number(summary.final_coverage_retention)}",
            (
                "- Median deep-parse rate drop (percentage points): "
                f"{_number(summary.deep_parse_drop_percentage_points)}"
            ),
            "",
            "## Secondary contrasts",
            "",
            "| comparison | wins | losses | ties | exact p | Holm p | rank-biserial |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        "".join(
            (
                f"| {contrast.comparison_arm} | {contrast.wins} | {contrast.losses} | ",
                f"{contrast.ties} | {_number(contrast.exact_sign_p)} | ",
                f"{_number(contrast.holm_adjusted_p)} | ",
                f"{_number(contrast.rank_biserial.effect)} |",
            )
        )
        for contrast in summary.secondary_contrasts
    )
    resource = summary.resource_usage
    lines.extend(
        [
            "",
            f"- Efficiency decision: {summary.efficiency_decision}",
            "",
            "## Exact resource accounting",
            "",
            f"- Resource integrity: {str(resource.resource_integrity).lower()}",
            "- Resource integrity fail-closed reasons: "
            + (
                ", ".join(resource.resource_integrity_reasons)
                if resource.resource_integrity_reasons
                else "none"
            ),
            f"- Physical shared-bank initial requests: {resource.bank_physical_initial_requests}",
            (
                "- Logical shared-bank initial requests by model arm: "
                + ", ".join(
                    f"{arm}={value}" for arm, value in resource.bank_logical_initial_requests_by_arm
                )
            ),
            f"- Search CPU seconds: {_number(resource.search_cpu_seconds)}",
            f"- Coverage replay count: {resource.coverage_replay_count}",
            f"- Proof replay count: {resource.proof_replay_count}",
            f"- Proof CPU seconds: {_number(resource.proof_cpu_seconds)}",
            f"- Proof wall seconds: {_number(resource.proof_wall_seconds)}",
            (
                "- Resident model-load duration/RSS/VRAM: unavailable in Task 5 "
                "outcome schema; resource integrity is false, GO is disabled, and "
                "the missing cost is not duplicated into arm denominators"
            ),
            "",
            (
                "| arm | charged model NFE | charged model GPU seconds | "
                "search CPU seconds | proof CPU seconds |"
            ),
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        "".join(
            (
                f"| {item.arm} | {_optional_number(item.model_nfe)} | ",
                f"{_optional_number(item.model_gpu_seconds)} | ",
                f"{_number(item.search_cpu_seconds)} | ",
                f"{_number(item.proof_cpu_seconds)} |",
            )
        )
        for item in resource.by_arm
    )
    causes = ", ".join(summary.proved_cause_ids) if summary.proved_cause_ids else "none"
    lines.extend(
        [
            "",
            "## Proved causes and censoring",
            "",
            f"- Exact twice-reproduced cause IDs: {causes}",
            "- Missing event-time endpoints are administratively right-censored at 10800 seconds.",
            "- Sparse/zero cause counts receive no significance test.",
            "",
            (
                "| repetition | arm | C0+10 time | C0+10 censored | "
                "first proved-cause time | proved-cause censored |"
            ),
            "| ---: | :--- | ---: | :--- | ---: | :--- |",
        ]
    )
    lines.extend(
        "".join(
            (
                f"| {endpoint.repetition} | {endpoint.arm} | ",
                f"{_number(endpoint.coverage_threshold.time_seconds)} | ",
                f"{str(endpoint.coverage_threshold.censored).lower()} | ",
                f"{_number(endpoint.first_proved_cause.time_seconds)} | ",
                f"{str(endpoint.first_proved_cause.censored).lower()} |",
            )
        )
        for endpoint in summary.event_endpoints
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "This screen cannot establish dLLM-over-AR, general libFuzzer "
                "superiority, zero-day detection rate, or 24-hour/7-day ranking."
            ),
            (
                "Sanitizer strings, fuzzer-native cov, stack hashes alone, and "
                "unreproduced crashes are not counted as proved vulnerabilities."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: float) -> str:
    return format(value, ".17g")


def _optional_number(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return _number(value)


def _validate_curve(curve: CoverageCurve) -> None:
    if len(curve.times) != len(curve.branches) or len(curve.times) < 2:
        raise AnalysisError("coverage curve requires equal lengths of at least two")
    if any(not math.isfinite(value) for value in curve.times):
        raise AnalysisError("coverage times must be finite")
    if any(right <= left for left, right in pairwise(curve.times)):
        raise AnalysisError("coverage times must be strictly increasing")
    _validate_integer_counts(curve.branches)
    if isinstance(curve.c0_branches, bool) or curve.c0_branches < 0:
        raise AnalysisError("C0 branch count must be a nonnegative integer")
    if any(right < left for left, right in pairwise(curve.branches)):
        raise AnalysisError("covered branch counts decreased")
    freeze = curve.failure_freeze_seconds
    if freeze is None:
        return
    if not math.isfinite(freeze) or not curve.times[0] <= freeze <= curve.times[-1]:
        raise AnalysisError("treatment freeze time lies outside the coverage curve")
    prior_indices = tuple(index for index, elapsed in enumerate(curve.times) if elapsed <= freeze)
    frozen_value = curve.branches[prior_indices[-1]]
    if any(
        count != frozen_value
        for elapsed, count in zip(curve.times, curve.branches, strict=True)
        if elapsed > freeze
    ):
        raise AnalysisError("coverage changed after treatment freeze")


def _validate_integer_counts(values: Sequence[int]) -> None:
    if any(isinstance(value, bool) or value < 0 for value in values):
        raise AnalysisError("coverage requires nonnegative integer branch counts")


def _validate_counts(wins: int, losses: int, ties: int) -> None:
    if any(isinstance(value, bool) or value < 0 for value in (wins, losses, ties)):
        raise AnalysisError("sign counts must be nonnegative integers")


def _validate_decision_inputs(inputs: DecisionInputs) -> None:
    if (
        isinstance(inputs.complete_pair_count, bool)
        or isinstance(inputs.selective_wins, bool)
        or inputs.complete_pair_count < 0
        or inputs.selective_wins < 0
        or inputs.selective_wins > inputs.complete_pair_count
    ):
        raise AnalysisError("decision pair and win counts are inconsistent")
    values = (
        inputs.median_uplift,
        inputs.final_coverage_retention,
        inputs.deep_parse_drop_percentage_points,
    )
    if any(not math.isfinite(value) for value in values):
        raise AnalysisError("decision metrics must be finite")


def _efficiency_axis_passes(selective: Sequence[float], comparison: Sequence[float]) -> bool:
    comparison_median = statistics.median(comparison)
    uplift = statistics.median(selective) / comparison_median - 1.0
    wins = sum(left > right for left, right in zip(selective, comparison, strict=True))
    return uplift >= 0.15 and wins >= 7
