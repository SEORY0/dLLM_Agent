"""Task 5 fix5D canonical recovery-gate regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import tools.run_openexr_one_day as cli
from pydantic import TypeAdapter
from tools.run_openexr_one_day import resume_with_factory

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day import coordinator
from dllm_agent.experiments.openexr_one_day.config_digest import (
    config_payload_sha256,
    load_verified_campaign_config,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CampaignOutcomeEnvelope,
    CoordinatorError,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    StartGateEvidence,
    campaign_status,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import SCHEDULE_SECONDS
from dllm_agent.experiments.openexr_one_day.failure_taxonomy import ExogenousCampaignError
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS, ModelArm
from dllm_agent.experiments.openexr_one_day.post_search import CoverageSeriesEnvelope
from dllm_agent.experiments.openexr_one_day.recovery import RecoveryError, resume_campaign
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix5_state_machine import (
    CountingModelExecutor,
    PathBackedPostSearch,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.runtime import OneDayRuntimeFactory

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def test_cli_resume_enters_runtime_graph_for_complete_fast_path(data_root: Path) -> None:
    config = _qualified_config(data_root)
    expected = _complete_campaign(config)
    graph = ResumeGraph()
    factory = ResumeFactory(graph=graph)

    actual = resume_with_factory(
        config.artifact_root, cast("OneDayRuntimeFactory", cast("object", factory))
    )

    assert actual == expected
    assert factory.build_count == 1
    assert graph.enter_count == 1
    assert graph.exit_count == 1


def test_cli_resume_passes_explicit_monotonic_preflight(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _qualified_config(data_root)
    _persist_minimal_manifest(config)
    graph = ResumeGraph()
    factory = ResumeFactory(graph=graph)
    seen: dict[str, float | None] = {}

    def fake_resume_campaign(*args: object, **kwargs: object) -> CampaignOutcome:
        del args
        seen["now"] = cast("float | None", kwargs.get("now_monotonic_seconds"))
        return cast("CampaignOutcome", object())

    monkeypatch.setattr(time, "monotonic", lambda: 12345.0)
    monkeypatch.setattr(cli, "resume_campaign", fake_resume_campaign)

    _ = resume_with_factory(
        config.artifact_root, cast("OneDayRuntimeFactory", cast("object", factory))
    )

    assert seen["now"] == 12345.0


def test_complete_status_and_resume_require_outcome_envelope(data_root: Path) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    assert campaign_status(config.artifact_root).state == "complete"
    (config.artifact_root / "campaign-outcome.json").unlink()

    assert campaign_status(config.artifact_root).state == "corrupt"
    with pytest.raises(RecoveryError, match="outcome envelope missing"):
        _ = resume_campaign(
            config.artifact_root,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(),
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
        )


def test_raw_validator_cross_checks_rehashed_proof_and_model_prep_rows(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    original_rows = _read_raw_rows(config.artifact_root)

    proof_rows = _mutate_rows(
        original_rows,
        _increment_first_available_proof_replay,
    )
    with pytest.raises(CoordinatorError, match="proof accounting mismatch"):
        coordinator._validate_raw_report_chain(_raw_content(proof_rows))

    prep_rows = _mutate_rows(
        original_rows,
        _increment_first_model_prep_nfe,
    )
    with pytest.raises(CoordinatorError, match="model-prep mismatch"):
        coordinator._validate_raw_report_chain(_raw_content(prep_rows))


def test_raw_validator_rejects_rehashed_coverage_envelope_with_stale_inner_hash(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    original_rows = _read_raw_rows(config.artifact_root)
    row = next(item for item in original_rows if item["coverage_status"] == "available")
    coverage_path = Path(str(row["coverage_series_path"]))
    envelope = CoverageSeriesEnvelope.model_validate_json(coverage_path.read_bytes())
    point = envelope.coverage_series.points[0]
    changed_series = envelope.coverage_series.model_copy(
        update={
            "points": (
                point.model_copy(update={"branches_covered": point.branches_covered + 1}),
                *envelope.coverage_series.points[1:],
            )
        }
    )
    changed_envelope = envelope.model_copy(update={"coverage_series": changed_series})
    _ = coverage_path.write_bytes(canonical_json_bytes(changed_envelope.model_dump(mode="json")))
    row["coverage_envelope_sha256"] = sha256_bytes(coverage_path.read_bytes())

    with pytest.raises(CoordinatorError, match="coverage result mismatch"):
        coordinator._validate_raw_report_chain(_raw_content(_rehash_rows(original_rows)))


def test_completed_rows_require_deep_parse_replay_evidence(data_root: Path) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)

    row = next(item for item in _read_raw_rows(config.artifact_root) if item["cell_result_sha256"])

    assert row["deep_parse_status"] == "available"
    assert isinstance(row["deep_parse_envelope_sha256"], str)
    assert isinstance(row["deep_parse_result_sha256"], str)
    assert isinstance(row["deep_parse_success_count"], int)
    assert isinstance(row["deep_parse_total_count"], int)
    assert row["deep_parse_success_count"] <= row["deep_parse_total_count"]


def test_treatment_failed_rows_carry_frozen_c0_coverage_through_deadline(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    failing_cell = "wave-00-rep-00-libfuzzer-only"

    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(treatment_failures={failing_cell: "fail"}),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=CountingModelExecutor(),
    )

    row = next(
        item for item in _read_raw_rows(config.artifact_root) if item["cell_id"] == failing_cell
    )
    assert row["terminal_reason"] == "treatment-failed"
    assert row["coverage_status"] == "available"
    assert row["proof_status"] == "unavailable"
    assert row["deep_parse_status"] == "available"
    coverage_path = Path(str(row["coverage_series_path"]))
    envelope = CoverageSeriesEnvelope.model_validate_json(coverage_path.read_bytes())
    assert tuple(point.scheduled_elapsed_seconds for point in envelope.coverage_series.points) == (
        *SCHEDULE_SECONDS,
    )
    assert {point.observation_kind for point in envelope.coverage_series.points[1:]} == {"frozen"}


def test_wave_zero_search_starts_before_later_wave_bank_generation(data_root: Path) -> None:
    config = _qualified_config(data_root)
    events: list[str] = []
    pool = FakeResidentPool(events=events)

    _ = coordinate_campaign(
        config,
        resident_pool=pool,
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=TracingCellRunner(events=events),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=CountingModelExecutor(),
    )

    first_wave_search = next(
        index for index, event in enumerate(events) if event == "run:wave-00-rep-00-libfuzzer-only"
    )
    later_wave_bank = next(
        index
        for index, event in enumerate(events)
        if event.startswith("bank:") and ":bank-r02-" in event
    )
    assert first_wave_search < later_wave_bank


def test_wave_zero_baseline_search_starts_before_same_wave_bank_generation(data_root: Path) -> None:
    config = _qualified_config(data_root)
    events: list[str] = []

    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(events=events),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=TracingCellRunner(events=events),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=CountingModelExecutor(),
    )

    first_wave_search = next(
        index for index, event in enumerate(events) if event == "run:wave-00-rep-00-libfuzzer-only"
    )
    same_wave_bank = next(
        index
        for index, event in enumerate(events)
        if event.startswith("bank:") and ":bank-r00-" in event
    )
    assert first_wave_search < same_wave_bank


def test_real_executor_interrupt_cancels_active_searches_before_waiting(data_root: Path) -> None:
    config = _qualified_config(data_root)
    runner = InterruptingRealThreadRunner()
    started = time.monotonic()

    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=runner,
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )

    elapsed = time.monotonic() - started
    assert runner.cancel_called is True
    assert runner.completed_after_interrupt == []
    assert elapsed < 1.0


def test_raw_validator_rejects_completed_cell_result_corruption(data_root: Path) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    raw_content = (config.artifact_root / "raw-arm-reports.jsonl").read_bytes()
    completed = next(
        row for row in _read_raw_rows(config.artifact_root) if row["cell_result_sha256"]
    )
    result_path = Path(str(completed["artifact_root"])) / "final-result.json"
    _ = result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(CoordinatorError, match="completed cell evidence invalid"):
        coordinator._validate_raw_report_chain(raw_content)


def test_partial_resume_rejects_missing_launched_corpus_seed_bytes(data_root: Path) -> None:
    config = _qualified_config(data_root)
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(interrupt_after=2),
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )
    missing_seed = next((config.artifact_root / "cells").glob("**/final-only/corpus/*"))
    missing_seed.unlink()

    with pytest.raises(RecoveryError, match="corpus evidence"):
        _ = resume_campaign(
            config.artifact_root,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(),
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )


def test_partial_resume_raw_report_uses_recovered_model_prep_metrics(data_root: Path) -> None:
    config = _qualified_config(data_root)
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(interrupt_after=2),
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )
    persisted_prep_cell = next(
        f"{Path(path).parent.parent.parent.name}-{Path(path).parent.parent.name}-{Path(path).parent.name}"
        for path in config.artifact_root.glob("cells/**/model-prep-result.json")
    )

    _ = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=CountingModelExecutor(),
    )

    row = next(
        row for row in _read_raw_rows(config.artifact_root) if row["cell_id"] == persisted_prep_cell
    )
    assert row["model_prep_result_path"] is not None
    assert row["model_prep_result_sha256"] is not None
    assert row["model_nfe"] == 7
    assert row["model_forward_calls"] == 8
    assert row["model_gpu_ms"] == 9


def test_model_prep_exogenous_error_invalidates_all_wave_cells(data_root: Path) -> None:
    config = _qualified_config(data_root)
    runner = FakeCellRunner()

    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=runner,
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=ExogenousModelExecutor(),
    )

    first_wave = tuple(cell for cell in outcome.cells if cell.wave_index == 0)
    assert len(first_wave) == 10
    assert {cell.terminal_reason for cell in first_wave} == {"exogenous-invalidated"}
    assert {cell.exogenous_evidence_sha256 for cell in first_wave} != {None}
    assert {pair.status for pair in outcome.pairs if pair.wave_index == 0} == {
        "exogenous-invalidated"
    }
    assert {
        request.cell_id for request in runner.requests if request.cell_id.startswith("wave-00-")
    } == {"wave-00-rep-00-libfuzzer-only", "wave-00-rep-01-libfuzzer-only"}


def test_checked_in_config_serializes_runtime_defaults_in_digest() -> None:
    config_path = Path("configs/openexr-v3.4.14-one-day.json")
    payload = _JSON_OBJECT.validate_json(config_path.read_bytes())

    loaded = load_verified_campaign_config(config_path)
    serialized = loaded.model_dump(mode="json")

    assert {
        "runtime_python",
        "model_cache_dir",
        "prompt_path",
        "qualification_evidence_path",
    } <= set(payload)
    assert payload["config_sha256"] == config_payload_sha256(payload)
    assert loaded.config_sha256 == config_payload_sha256(serialized)


def test_complete_status_rejects_rehashed_envelope_with_missing_raw_row(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    raw_path = config.artifact_root / "raw-arm-reports.jsonl"
    rows = raw_path.read_bytes().splitlines()
    _ = raw_path.write_bytes(b"\n".join(rows[:-1]) + b"\n")
    _rewrite_outcome_envelope_hashes(config.artifact_root)

    assert campaign_status(config.artifact_root).state == "corrupt"


def test_complete_status_rejects_semantically_changed_outcome_envelope(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    _ = _complete_campaign(config)
    path = config.artifact_root / "campaign-outcome.json"
    envelope = CampaignOutcomeEnvelope.model_validate_json(path.read_bytes())
    cells = list(envelope.outcome.cells)
    cells[0] = cells[0].model_copy(update={"terminal_reason": "deadline"})
    changed = envelope.model_copy(
        update={"outcome": envelope.outcome.model_copy(update={"cells": tuple(cells)})}
    )
    _ = path.write_bytes(canonical_json_bytes(changed.model_dump(mode="json")))

    assert campaign_status(config.artifact_root).state == "corrupt"


def test_config_digest_normalizes_omitted_defaults(data_root: Path) -> None:
    payload = _config_json(str(data_root))
    with_defaults = CampaignConfig.model_validate(payload).model_dump(mode="json")

    assert config_payload_sha256(payload) == config_payload_sha256(with_defaults)


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix5d")
    return Path("/data/dllm-agent") / f"task5-fix5d-{sha256(str(root).encode()).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class ResumeFactory:
    graph: ResumeGraph
    build_count: int = 0

    def build(self, config: CampaignConfig) -> ResumeGraph:
        del config
        object.__setattr__(self, "build_count", self.build_count + 1)
        return self.graph

    def run(self, config: CampaignConfig) -> CampaignOutcome:
        raise AssertionError(config)


@dataclass(frozen=True, slots=True)
class ResumeGraph:
    resident_pool: FakeResidentPool = field(default_factory=FakeResidentPool)
    cell_runner: FakeCellRunner = field(default_factory=FakeCellRunner)
    post_search: FakePostSearch = field(default_factory=FakePostSearch)
    gate_runner: FakeGateRunner = field(default_factory=lambda: FakeGateRunner(_passing_gates()))
    model_executor: CountingModelExecutor = field(default_factory=CountingModelExecutor)
    enter_count: int = 0
    exit_count: int = 0

    def __enter__(self) -> Self:
        object.__setattr__(self, "enter_count", self.enter_count + 1)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        object.__setattr__(self, "exit_count", self.exit_count + 1)


@dataclass(frozen=True, slots=True)
class ExogenousModelExecutor:
    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        raise ExogenousCampaignError(cause="gpu-xid", evidence=f"xid:{request.cell.cell_id}")


@dataclass(frozen=True, slots=True)
class TracingCellRunner:
    events: list[str]
    delegate: FakeCellRunner = field(default_factory=FakeCellRunner)

    def run(self, request: CellRunRequest) -> CellResult:
        cell_request = request
        self.events.append(f"run:{cell_request.cell_id}")
        return self.delegate.run(cell_request)


@dataclass(frozen=True, slots=True)
class InterruptingRealThreadRunner:
    requests: list[str] = field(default_factory=list)
    completed_after_interrupt: list[str] = field(default_factory=list)
    release: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_called: bool = False

    def run(self, request: CellRunRequest) -> CellResult:
        with self.lock:
            self.requests.append(request.cell_id)
            should_interrupt = len(self.requests) == 1
        if should_interrupt:
            raise KeyboardInterrupt
        if not self.release.wait(2.0):
            self.completed_after_interrupt.append(request.cell_id)
            return FakeCellRunner().run(request)
        message = "cancelled"
        raise OSError(message)

    def cancel_active(self) -> None:
        object.__setattr__(self, "cancel_called", True)
        self.release.set()


def _complete_campaign(config: CampaignConfig) -> CampaignOutcome:
    return coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=CountingModelExecutor(),
    )


def _read_raw_rows(root: Path) -> list[dict[str, JsonValue]]:
    return [
        _JSON_OBJECT.validate_json(line.encode())
        for line in (root / "raw-arm-reports.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _mutate_rows(
    rows: list[dict[str, JsonValue]],
    mutate: Callable[[dict[str, JsonValue]], None],
) -> list[dict[str, JsonValue]]:
    changed = False
    copied = [dict(row) for row in rows]
    for row in copied:
        if changed:
            break
        before = dict(row)
        mutate(row)
        changed = row != before
    if not changed:
        message = "raw row mutation did not match any row"
        raise AssertionError(message)
    return _rehash_rows(copied)


def _rehash_rows(rows: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    previous = "0" * 64
    rehashed: list[dict[str, JsonValue]] = []
    for row in rows:
        payload = dict(row)
        payload["previous_row_sha256"] = previous
        payload["row_sha256"] = "0" * 64
        digest = sha256_bytes(
            canonical_json_bytes({k: v for k, v in payload.items() if k != "row_sha256"})
        )
        payload["row_sha256"] = digest
        previous = digest
        rehashed.append(payload)
    return rehashed


def _raw_content(rows: list[dict[str, JsonValue]]) -> bytes:
    return (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    ).encode()


def _rewrite_outcome_envelope_hashes(root: Path) -> None:
    path = root / "campaign-outcome.json"
    envelope = CampaignOutcomeEnvelope.model_validate_json(path.read_bytes())
    manifest_bytes = (root / "campaign-manifest.json").read_bytes()
    event_bytes = (root / "campaign-events.jsonl").read_bytes()
    raw_bytes = (root / "raw-arm-reports.jsonl").read_bytes()
    tree = sha256_bytes(
        canonical_json_bytes(
            {
                "event_stream_sha256": sha256_bytes(event_bytes),
                "manifest_artifact_sha256": sha256_bytes(manifest_bytes),
                "raw_report_sha256": sha256_bytes(raw_bytes),
            }
        )
    )
    changed = envelope.model_copy(
        update={
            "manifest_artifact_sha256": sha256_bytes(manifest_bytes),
            "event_stream_sha256": sha256_bytes(event_bytes),
            "raw_report_sha256": sha256_bytes(raw_bytes),
            "raw_report_size_bytes": len(raw_bytes),
            "evidence_tree_sha256": tree,
        }
    )
    _ = path.write_bytes(canonical_json_bytes(changed.model_dump(mode="json")))


def _increment_first_available_proof_replay(row: dict[str, JsonValue]) -> None:
    value = row["proof_replay_count"]
    if row["proof_status"] == "available" and isinstance(value, int):
        row["proof_replay_count"] = value + 1


def _increment_first_model_prep_nfe(row: dict[str, JsonValue]) -> None:
    value = row["model_nfe"]
    if row["model_prep_result_path"] is not None and isinstance(value, int):
        row["model_nfe"] = value + 1


def _qualified_config(data_root: Path) -> CampaignConfig:
    payload = _config_json(str(data_root)) | {"qualification_state": "qualified"}
    config = CampaignConfig.model_validate(payload)
    serialized = config.model_dump(mode="json")
    serialized["config_sha256"] = config_payload_sha256(serialized)
    return CampaignConfig.model_validate(serialized)


def _persist_minimal_manifest(config: CampaignConfig) -> None:
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {"config": config.model_dump(mode="json")}
    _ = (config.artifact_root / "campaign-manifest.json").write_bytes(canonical_json_bytes(payload))


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
    return {arm: FakeRevisionDelegate() for arm in MODEL_ARMS}
