"""Task 5 fix5B evidence envelope and raw-report regressions."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import local
from typing import TYPE_CHECKING

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day import coordinator
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CoordinatorError,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    StartGateEvidence,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRecord,
    CoverageSeries,
    CoverageSeriesRequest,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    CoverageSeriesEnvelope,
    HashOnlyDeepParseReplayer,
    PostSearchEvidenceError,
    PostSearchOutcome,
    ProductionPostSearch,
    load_post_search_outcome,
)
from tests.openexr_one_day_deep_parse_fixtures import final_coverage
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix4_recovery_reporting import (
    CountingCoverageBuilder,
    CountingProofBuilder,
    NoopCoverageReplayer,
    StaticProofClock,
    _cell_request,
    _cell_result,
    _coverage_point,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.coverage_series import (
        CoverageReplayer,
    )
    from dllm_agent.experiments.openexr_one_day.models import ModelArm


def test_post_search_envelope_rejects_changed_request_and_result(data_root: Path) -> None:
    # Given
    request = _cell_request(data_root)
    result = _cell_result(request)
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=FullScheduleCoverageBuilder(),
        proof_builder=CountingProofBuilder(),
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )
    first = post_search.coverage_and_proof(request, result)
    changed_request = request.model_copy(update={"fuzzer_seed": request.fuzzer_seed + 1})
    changed_result = result.model_copy(
        update={
            "search_accounting": result.search_accounting.model_copy(update={"cpu_seconds": 12.0})
        }
    )

    # When / Then
    assert load_post_search_outcome(request, result) == first
    with pytest.raises(PostSearchEvidenceError, match="coverage envelope"):
        _ = load_post_search_outcome(changed_request, result)
    with pytest.raises(PostSearchEvidenceError, match="result"):
        _ = load_post_search_outcome(request, changed_result)
    with pytest.raises(PostSearchEvidenceError, match="coverage envelope"):
        _ = post_search.coverage_and_proof(changed_request, result)


def test_completed_coverage_reload_rejects_cross_profile_request_tampering(
    data_root: Path,
) -> None:
    # Given
    request = _cell_request(data_root)
    result = _cell_result(request)
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=FullScheduleCoverageBuilder(),
        proof_builder=CountingProofBuilder(),
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )
    first = post_search.coverage_and_proof(request, result)
    coverage_path = _required_path(first.coverage_series_path)
    envelope = CoverageSeriesEnvelope.model_validate_json(coverage_path.read_bytes())
    smoke_index = envelope.coverage_request.corpus_index.model_copy(
        update={"execution_profile": "qualification-smoke", "schedule_seconds": (0, 300)}
    )
    smoke_request = envelope.coverage_request.model_copy(
        update={
            "execution_profile": "qualification-smoke",
            "schedule_seconds": (0, 300),
            "corpus_index": smoke_index,
        }
    )
    tampered = envelope.model_copy(
        update={
            "coverage_request": smoke_request,
            "coverage_request_sha256": sha256_bytes(
                canonical_json_bytes(smoke_request.model_dump(mode="json"))
            ),
        }
    )
    _ = coverage_path.write_bytes(canonical_json_bytes(tampered.model_dump(mode="json")))

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="current request"):
        _ = post_search.coverage_and_proof(request, result)


def test_completed_coverage_reload_rejects_cross_profile_series_tampering(
    data_root: Path,
) -> None:
    # Given
    request = _cell_request(data_root)
    result = _cell_result(request)
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=FullScheduleCoverageBuilder(),
        proof_builder=CountingProofBuilder(),
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )
    first = post_search.coverage_and_proof(request, result)
    coverage_path = _required_path(first.coverage_series_path)
    envelope = CoverageSeriesEnvelope.model_validate_json(coverage_path.read_bytes())
    smoke_row = envelope.coverage_series.corpus_index[0].model_copy(
        update={"scheduled_elapsed_seconds": 300, "source_elapsed_seconds": 300.0}
    )
    smoke_point = envelope.coverage_series.points[0].model_copy(
        update={"scheduled_elapsed_seconds": 300, "source_elapsed_seconds": 300.0}
    )
    smoke_series = envelope.coverage_series.model_copy(
        update={"corpus_index": (smoke_row,), "points": (smoke_point,)}
    )
    tampered = envelope.model_copy(
        update={
            "coverage_series": smoke_series,
            "coverage_series_sha256": sha256_bytes(
                canonical_json_bytes(smoke_series.model_dump(mode="json"))
            ),
        }
    )
    _ = coverage_path.write_bytes(canonical_json_bytes(tampered.model_dump(mode="json")))

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="schedule"):
        _ = post_search.coverage_and_proof(request, result)


def test_treatment_coverage_reload_rejects_placeholder_profile_tampering(
    data_root: Path,
) -> None:
    # Given
    request = _cell_request(data_root)
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=FullScheduleCoverageBuilder(),
        proof_builder=CountingProofBuilder(),
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )
    first = post_search.treatment_coverage(request)
    coverage_path = _required_path(first.coverage_series_path)
    envelope = CoverageSeriesEnvelope.model_validate_json(coverage_path.read_bytes())
    smoke_placeholder = envelope.cell_result.model_copy(
        update={
            "execution_profile": "qualification-smoke",
            "corpus_index_scheduled_seconds": (0, 300),
        }
    )
    tampered = envelope.model_copy(update={"cell_result": smoke_placeholder})
    _ = coverage_path.write_bytes(canonical_json_bytes(tampered.model_dump(mode="json")))

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="result mismatch"):
        _ = post_search.treatment_coverage(request)


def test_raw_report_rejects_wrong_schema_and_outside_artifact(data_root: Path) -> None:
    # Given
    outside = data_root.parent / f"task5-fix5b-outside-{data_root.name}.json"
    _ = outside.write_text("{}", encoding="utf-8")
    wrong_schema = canonical_json_bytes(
        {
            "schema_version": "wrong",
            "coverage_status": "available",
            "coverage_series_path": outside.as_posix(),
            "coverage_series_sha256": sha256_bytes(outside.read_bytes()),
            "proof_status": "unavailable",
            "proof_result_path": None,
            "proof_result_sha256": None,
            "previous_row_sha256": "0" * 64,
        }
    )

    # When / Then
    with pytest.raises(CoordinatorError, match="raw arm report"):
        coordinator._validate_raw_report_chain(wrong_schema + b"\n")


def test_bank_elapsed_timing_is_recorded_in_all_model_raw_rows(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    config = _qualified_config(data_root)
    clock = StepClock()
    monkeypatch.setattr(coordinator, "monotonic", clock.monotonic)
    model = TimingEchoModelExecutor()

    # When
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=model,
    )

    # Then
    rows = [
        coordinator.RawArmReportRow.model_validate_json(line)
        for line in (config.artifact_root / "raw-arm-reports.jsonl").read_text().splitlines()
    ]
    model_rows = [row for row in rows if row.arm != "libfuzzer-only"]
    assert all(
        row.bank_measured_elapsed_seconds == row.bank_physical_duration_seconds
        for row in model_rows
    )
    assert all(
        row.bank_physical_duration_seconds is not None and row.bank_physical_duration_seconds > 0.0
        for row in model_rows
    )
    assert {row.bank_ready_elapsed_seconds for row in model_rows} == {0.0}
    assert all(
        row.bank_evidence_path is not None
        and row.bank_evidence_path.as_posix().startswith(config.artifact_root.as_posix())
        for row in model_rows
    )


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix5b")
    return Path("/data/dllm-agent") / f"task5-fix5b-{sha256(str(root).encode()).hexdigest()[:12]}"


class _ThreadClockState(local):
    """Independent deterministic clock state for each concurrent bank job."""

    value: float = 0.0


@dataclass(frozen=True, slots=True)
class StepClock:
    state: _ThreadClockState = field(default_factory=_ThreadClockState)

    def monotonic(self) -> float:
        value = self.state.value
        self.state.value += 2.0
        return value


@dataclass(frozen=True, slots=True)
class FullScheduleCoverageBuilder:
    """Return a request-schedule coverage series for envelope tests."""

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        del replayer
        coverage = final_coverage(request.corpus_index.artifact_root, b"fix5-envelope")
        first = coverage.corpus_index[0]
        rows = tuple(
            _scheduled_row(request, first, scheduled) for scheduled in request.schedule_seconds
        )
        return CoverageSeries(
            corpus_index=rows,
            points=tuple(_coverage_point(request, row) for row in rows),
        )


def _scheduled_row(
    request: CoverageSeriesRequest, first: CorpusIndexRecord, scheduled: int
) -> CorpusIndexRecord:
    if scheduled == request.schedule_seconds[0]:
        return first.model_copy(
            update={
                "scheduled_elapsed_seconds": scheduled,
                "source_elapsed_seconds": float(scheduled),
            }
        )
    return first.model_copy(
        update={
            "scheduled_elapsed_seconds": scheduled,
            "observation_kind": "frozen",
            "source_elapsed_seconds": 0.0,
            "new_inputs": (),
        }
    )


def _required_path(path: Path | None) -> Path:
    assert path is not None
    return path


@dataclass(frozen=True, slots=True)
class TimingEchoModelExecutor:
    calls: list[str] = field(default_factory=list)

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        self.calls.append(request.cell.cell_id)
        path = request.cell.artifact_root / "corpus" / "timed-model.exr"
        payload = request.cell.cell_id.encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise AssertionError
        _ = path.write_bytes(payload)
        return ModelArmExecutionResult(
            written_snapshot_count=1,
            written_paths=(path,),
            nfe=3,
            forward_calls=4,
            gpu_ms=5,
            bank_ready_elapsed_seconds=request.bank_ready_elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class PathBackedPostSearch:
    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        return self._post_search().coverage_and_proof(request, result)

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        return self._post_search().treatment_coverage(request)

    @staticmethod
    def _post_search() -> ProductionPostSearch:
        return ProductionPostSearch(
            coverage_replayer=NoopCoverageReplayer(),
            coverage_builder=CountingCoverageBuilder(),
            proof_builder=CountingProofBuilder(),
            clock=StaticProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        )


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified"}
    )


def _passing_gates() -> StartGateEvidence:
    return StartGateEvidence(
        hashes_match=True,
        banks_complete=True,
        read_only_cache=True,
        external_coverage_check=True,
        control_arm_equivalence=True,
        resource_map_valid=True,
        disk_space=True,
        vram=True,
        ram=True,
        swap_stable=True,
        reserved_core_load=True,
        smoke_passed=True,
    )


def _delegates() -> dict[ModelArm, FakeRevisionDelegate]:
    return {
        arm: FakeRevisionDelegate()
        for arm in ("selective-revision", "final-only", "full-regenerate", "structured-one-shot")
    }
