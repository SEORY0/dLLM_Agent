"""Task 5 fix1 reviewer reproductions.

# noqa: SIZE_OK
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Lock
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter, ValidationError
from tools.run_openexr_one_day import run_with_factory

from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, SearchAccounting
from dllm_agent.experiments.openexr_one_day.config_digest import (
    config_payload_sha256,
    load_verified_campaign_config,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CoordinatorError,
    ResourceMap,
    StartGateEvidence,
    campaign_status,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.corpus_seed import seed_cell_corpus
from dllm_agent.experiments.openexr_one_day.failure_taxonomy import ExogenousCampaignError
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS, ModelArm
from dllm_agent.experiments.openexr_one_day.post_search import PostSearchOutcome
from dllm_agent.experiments.openexr_one_day.recovery import resume_campaign
from dllm_agent.experiments.openexr_one_day.runtime import ProductionRuntimeFactory
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)

_REPORT_ROW = TypeAdapter(dict[str, JsonValue])

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellRunRequest


def test_qualified_cli_run_uses_injected_runtime_factory_instead_of_unavailable_sentinel(
    data_root: Path,
) -> None:
    # Given
    factory = FakeRuntimeFactory()
    config = _qualified_config(data_root)

    # When
    outcome = run_with_factory(config, factory)

    # Then
    assert isinstance(outcome, CampaignOutcome)
    assert factory.production_dependencies_constructed is True
    assert outcome.launched_cell_count == 50


def test_production_runtime_factory_constructs_real_dependency_graph_without_placeholders(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)

    # When
    graph = ProductionRuntimeFactory(
        runtime_python=Path("/usr/bin/python3"),
        source_root=Path("/data/dllm-agent/source/openexr"),
        cache_dir=Path("/data/dllm-agent/cache"),
        prompt_path=Path("prompts/openexr-system.md"),
    ).build(config)

    # Then
    names = graph.concrete_dependency_names
    assert "RevisionWorkerClient" in names
    assert "ResidentWorkerPool" in names
    assert "DockerSearchLauncher" in names
    assert "OpenExrProofRunner" in names
    assert "_UnavailableRuntime" not in names


def test_wave_search_cells_launch_concurrently_and_post_search_waits_for_all_search(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    runner = BarrierCellRunner(participants=10)
    post_search = OrderedPostSearch(runner)

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=runner,
        post_search=post_search,
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    assert outcome.launched_cell_count == 50
    assert runner.max_active == 10
    assert post_search.first_post_search_after_search_calls == 50


def test_corpus_seeding_materializes_c0_and_selected_bank_parent_bytes(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    bank = fake_bank_sampler()

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=bank,
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    model_cell = next(cell for cell in outcome.cells if cell.arm == "selective-revision")
    libfuzzer_cell = next(cell for cell in outcome.cells if cell.arm == "libfuzzer-only")
    model_seed = seed_cell_corpus(config, model_cell)
    libfuzzer_seed = seed_cell_corpus(config, libfuzzer_cell)

    # Then
    assert libfuzzer_seed.bank_parent_count == 0
    assert model_seed.bank_parent_count == 16
    assert libfuzzer_seed.c0_sha256 == config.c0_sha256
    assert model_seed.c0_sha256 == config.c0_sha256


def test_resume_preserves_treatment_failed_terminal_outcome(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    failed_cell = "wave-00-rep-00-selective-revision"
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(treatment_failures={failed_cell: "controller-crash"}),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # When
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    resumed = next(cell for cell in outcome.cells if cell.cell_id == failed_cell)
    assert resumed.terminal_reason == "treatment-failed"
    assert resumed.coverage_frozen_through_seconds == 10_800


def test_config_digest_is_detached_and_rejects_placeholder_hashes(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.json"
    payload = _config_json("/data/dllm-agent/digest")
    payload["config_sha256"] = config_payload_sha256(payload)
    _ = config_path.write_text(json.dumps(payload), encoding="utf-8")

    # When
    loaded = load_verified_campaign_config(config_path)

    # Then
    assert loaded.config_sha256 == payload["config_sha256"]
    payload["target_harness_sha256"] = "1" * 64
    payload["config_sha256"] = "1" * 64
    _ = config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CoordinatorError, match="placeholder"):
        _ = load_verified_campaign_config(config_path)


def test_status_marks_arbitrary_noncanonical_root_corrupt(tmp_path: Path) -> None:
    # Given
    root = (
        Path("/data/dllm-agent") / f"task5-corrupt-{sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    )
    root.mkdir(parents=True)
    _ = (root / "random.txt").write_text("not a campaign", encoding="utf-8")

    # When
    status = campaign_status(root)

    # Then
    assert status.state == "corrupt"


def test_typed_exogenous_failure_invalidates_pair_but_oserror_is_treatment_failure(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    exogenous = "wave-01-rep-02-final-only"
    treatment = "wave-02-rep-04-selective-revision"
    runner = TaxonomyCellRunner(
        exogenous_failures={exogenous: ExogenousCampaignError(cause="gpu-xid", evidence="xid 79")},
        treatment_oserrors={treatment: OSError("controller pipe closed")},
    )

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=runner,
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    assert {pair.status for pair in outcome.pairs if pair.wave_index == 1} == {
        "exogenous-invalidated"
    }
    treated = next(cell for cell in outcome.cells if cell.cell_id == treatment)
    assert treated.terminal_reason == "treatment-failed"


def test_resource_map_rejects_search_numa_mismatch_and_role_sibling_overlap(
    data_root: Path,
) -> None:
    # Given
    del data_root
    search_mismatch = _resource_map_json()
    search_core_numa = search_mismatch["core_numa_nodes"]
    if not isinstance(search_core_numa, dict):
        message = "test resource map core_numa_nodes must be mutable"
        raise TypeError(message)
    search_core_numa["8"] = 1
    sibling_overlap = _resource_map_json()
    sibling_overlap["proof_cores"] = [{"core_id": 18, "sibling_core_id": 8}]

    # When / Then
    with pytest.raises(ValidationError):
        _ = ResourceMap.model_validate(search_mismatch)
    with pytest.raises(ValidationError):
        _ = ResourceMap.model_validate(sibling_overlap)


def test_task6_arm_reports_are_persisted_with_coverage_proof_and_bank_ready_fields(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)

    # When
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    report_path = config.artifact_root / "raw-arm-reports.jsonl"
    rows = [
        _REPORT_ROW.validate_json(line)
        for line in report_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 50
    assert {
        "repetition",
        "arm",
        "coverage_series_sha256",
        "c0_baseline_sha256",
        "bank_ready_elapsed_seconds",
        "proof_replay_count",
        "terminal_reason",
        "pair_status",
    } <= set(rows[0])


@dataclass(frozen=True, slots=True)
class FakeRuntimeFactory:
    """Injected CLI factory proving qualified run calls composition, not a sentinel."""

    production_dependencies_constructed: bool = False

    def run(self, config: CampaignConfig) -> CampaignOutcome:
        object.__setattr__(self, "production_dependencies_constructed", True)
        return coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(),
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
        )


@dataclass(frozen=True, slots=True)
class BarrierCellRunner:
    """Cell runner fake that deadlocks unless a whole wave launches concurrently."""

    participants: int
    active: int = 0
    max_active: int = 0
    total_calls: int = 0
    barrier: Barrier = field(init=False)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        object.__setattr__(self, "barrier", Barrier(self.participants))

    def run(self, request: CellRunRequest) -> CellResult:
        with self.lock:
            object.__setattr__(self, "active", self.active + 1)
            object.__setattr__(self, "max_active", max(self.max_active, self.active))
        _ = self.barrier.wait(timeout=1.0)
        with self.lock:
            object.__setattr__(self, "active", self.active - 1)
            object.__setattr__(self, "total_calls", self.total_calls + 1)
        return _completed_cell_result(request)


@dataclass(frozen=True, slots=True)
class OrderedPostSearch:
    """Post-search fake that records whether all search cells completed first."""

    runner: BarrierCellRunner
    first_post_search_after_search_calls: int | None = None

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        del request, result
        if self.first_post_search_after_search_calls is None:
            object.__setattr__(
                self, "first_post_search_after_search_calls", self.runner.total_calls
            )
        return PostSearchOutcome(
            coverage_replay_count=1,
            proof_replay_count=1,
            coverage_series_sha256="7" * 64,
            proof_result_sha256="8" * 64,
            proof_wall_seconds=1.0,
            proof_cpu_seconds=1.0,
        )

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        del request
        return PostSearchOutcome(
            coverage_replay_count=1,
            proof_replay_count=0,
            coverage_series_sha256="7" * 64,
            proof_result_sha256="8" * 64,
            proof_wall_seconds=0.0,
            proof_cpu_seconds=0.0,
        )


@dataclass(frozen=True, slots=True)
class TaxonomyCellRunner:
    """Cell runner fake for typed exogenous versus treatment failures."""

    exogenous_failures: dict[str, ExogenousCampaignError]
    treatment_oserrors: dict[str, OSError]

    def run(self, request: CellRunRequest) -> CellResult:
        if request.cell_id in self.exogenous_failures:
            raise self.exogenous_failures[request.cell_id]
        if request.cell_id in self.treatment_oserrors:
            raise self.treatment_oserrors[request.cell_id]
        return FakeCellRunner().run(request)


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("case")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    path = Path("/data/dllm-agent") / f"task5-fix1-{suffix}"
    if path.exists():
        raise AssertionError(path)
    return path


def _qualified_config(data_root: Path) -> CampaignConfig:
    payload = _config_json(str(data_root)) | {"qualification_state": "qualified"}
    return CampaignConfig.model_validate(payload)


def _completed_cell_result(request: CellRunRequest) -> CellResult:
    return CellResult(
        cell_id=request.cell_id,
        status="completed",
        started_monotonic_seconds=request.start_monotonic_seconds,
        finished_monotonic_seconds=request.deadline_monotonic_seconds,
        coverage_freeze_elapsed_seconds=None,
        attempt_count=1,
        manifest_sha256="a" * 64,
        event_stream_sha256="b" * 64,
        corpus_index_stream_sha256="c" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(range(0, 10_801, 900)),
        search_accounting=SearchAccounting(cpu_seconds=11.0, wall_seconds=10_800.0),
    )


def _config_json(root: str) -> dict[str, JsonValue]:
    return {
        "schema_version": "dllm-openexr-one-day-config/v1",
        "config_id": "task5-fix1-test",
        "qualification_state": "draft",
        "artifact_root": root,
        "model_cache_dir": f"{root}/cache",
        "main_start_monotonic_seconds": 70_000.0,
        "target_commit": "777d231a179de1711d2a942810d9559216ab3f4c",
        "target_harness_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "model_revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        "remote_code_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "search_image_digest": "sha256:" + "5" * 64,
        "search_binary_sha256": "6" * 64,
        "sanitizers": ["address", "undefined"],
        "coverage_image_digest": "sha256:" + "7" * 64,
        "coverage_binary_sha256": "8" * 64,
        "coverage_build_manifest_sha256": "9" * 64,
        "c0_source_path": "fixtures/openexr/c0-validity-preserving-zero.hex",
        "c0_sha256": "ae9d361e47cc503dc91b827f4956fdf8ca16ffb3804ecdfcf83dd627f8a64f96",
        "dictionary_sha256": "9419f2003be8f41d286360ea3dec3f8fa967244111eecec0fd292db0ae6fed93",
        "max_input_length": 1_048_576,
        "target_timeout_seconds": 10,
        "rss_limit_mb": 2_048,
        "candidate_steps": 8,
        "proof_duration_seconds": 30,
        "resource_map": _resource_map_json(),
    }


def _resource_map_json() -> dict[str, JsonValue]:
    return {
        "model_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (0, 1, 2, 3, 10, 11, 12, 13)
        ],
        "search_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (4, 5, 6, 7, 8, 9, 14, 15, 16, 17)
        ],
        "proof_cores": [{"core_id": 18, "sibling_core_id": 38}],
        "monitor_cores": [{"core_id": 19, "sibling_core_id": 39}],
        "gpu_numa_nodes": [0, 0, 0, 0, 1, 1, 1, 1],
        "gpu_records": [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                "numa_node": 0 if index < 4 else 1,
            }
            for index in range(8)
        ],
        "core_numa_nodes": {
            str(index): 0 if index in {*range(10), *range(20, 30)} else 1 for index in range(40)
        },
    }


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
