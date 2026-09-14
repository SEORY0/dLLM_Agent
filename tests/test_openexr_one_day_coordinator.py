"""Task 5 coordinator contract for the OpenEXR one-day experiment. # noqa: SIZE_OK"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import ValidationError

from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CoordinatorError,
    CoreAssignment,
    ResourceMap,
    StartGateEvidence,
    build_campaign_manifest,
    build_cell_requests,
    campaign_status,
    coordinate_campaign,
    load_campaign_config,
)
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dllm_agent.experiments.multitarget.ledger import JsonValue

type ResourceMapPayload = dict[str, JsonValue | tuple[CoreAssignment, ...]]


def test_coordinator_reuses_exactly_eight_resident_worker_pids_across_all_five_waves(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    pool = FakeResidentPool()

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=pool,
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    assert isinstance(outcome, CampaignOutcome)
    assert pool.enter_count == 1
    assert pool.exit_count == 1
    assert tuple(worker.enter_count for worker in pool.workers.values()) == (1,) * 8
    assert tuple(worker.close_count for worker in pool.workers.values()) == (1,) * 8
    assert outcome.resident_worker_pids == tuple(10_000 + gpu for gpu in range(8))
    assert len(pool.worker_requests) == 40
    assert set(pool.worker_requests) == set(range(8))


def test_each_wave_launches_two_repetitions_by_five_arms_with_private_roots_and_bindings(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    cell_runner = FakeCellRunner()
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )

    # When
    requests = build_cell_requests(
        config=config,
        manifest=build_campaign_manifest(config, schedule),
        schedule=schedule,
        bank_by_repetition=_bank_hashes(),
        resource_map=config.resource_map,
    )
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=cell_runner,
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    assert outcome.launched_cell_count == 50
    assert len(cell_runner.requests) == 50
    assert len(requests) == 50
    assert tuple((request.cell_id, request.fuzzer_seed) for request in requests[:10]) == (
        ("wave-00-rep-00-selective-revision", 1_000_001),
        ("wave-00-rep-00-final-only", 1_000_002),
        ("wave-00-rep-00-full-regenerate", 1_000_003),
        ("wave-00-rep-00-structured-one-shot", 1_000_004),
        ("wave-00-rep-00-libfuzzer-only", 1_000_005),
        ("wave-00-rep-01-selective-revision", 1_001_001),
        ("wave-00-rep-01-final-only", 1_001_002),
        ("wave-00-rep-01-full-regenerate", 1_001_003),
        ("wave-00-rep-01-structured-one-shot", 1_001_004),
        ("wave-00-rep-01-libfuzzer-only", 1_001_005),
    )
    assert (
        tuple(request.deadline_monotonic_seconds for request in requests[:10]) == (80_800.0,) * 10
    )
    assert len({request.artifact_root.resolve() for request in requests}) == 50
    assert _all_disjoint(request.artifact_root for request in requests)
    assert all(request.artifact_root.is_relative_to(config.artifact_root) for request in requests)
    assert tuple(request.cpu_index for request in requests[:10]) == (
        4,
        5,
        6,
        7,
        8,
        9,
        14,
        15,
        16,
        17,
    )
    assert all(
        "libfuzzer-only" not in request.bank_sha256
        for request in requests
        if request.cell_id.endswith("libfuzzer-only")
    )


def test_model_arms_share_frozen_bank_identity_while_libfuzzer_receives_only_c0(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    for repetition in range(10):
        model_cells = tuple(
            cell
            for cell in outcome.cells
            if cell.repetition == repetition and cell.arm in MODEL_ARMS
        )
        libfuzzer_cell = next(
            cell
            for cell in outcome.cells
            if cell.repetition == repetition and cell.arm == "libfuzzer-only"
        )
        assert len({cell.bank_sha256 for cell in model_cells}) == 1
        assert len({cell.trajectory_set_sha256 for cell in model_cells}) == 1
        assert len({cell.mutable_root for cell in model_cells}) == 4
        assert libfuzzer_cell.bank_sha256 == config.c0_sha256
        assert libfuzzer_cell.trajectory_set_sha256 == "0" * 64


@pytest.mark.parametrize(
    "gate",
    [
        "hashes_match",
        "banks_complete",
        "read_only_cache",
        "external_coverage_check",
        "control_arm_equivalence",
        "resource_map_valid",
        "disk_space",
        "vram",
        "ram",
        "swap_stable",
        "reserved_core_load",
        "smoke_passed",
    ],
)
def test_every_fail_closed_start_gate_causes_zero_worker_or_target_launches(
    data_root: Path,
    gate: str,
) -> None:
    # Given
    config = _qualified_config(data_root)
    evidence = _passing_gates().model_copy(update={gate: False})
    pool = FakeResidentPool()
    cell_runner = FakeCellRunner()

    # When / Then
    with pytest.raises(CoordinatorError, match=gate.replace("_", "-")):
        _ = coordinate_campaign(
            config,
            resident_pool=pool,
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=cell_runner,
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(evidence),
        )
    assert pool.enter_count == 0
    assert cell_runner.requests == []


@pytest.mark.parametrize(
    "resource_map",
    [
        {
            "model_cores": (CoreAssignment(core_id=0, sibling_core_id=20),) * 8,
            "search_cores": tuple(
                CoreAssignment(core_id=index + 8, sibling_core_id=index + 28) for index in range(10)
            ),
            "proof_cores": (CoreAssignment(core_id=18, sibling_core_id=38),),
            "monitor_cores": (CoreAssignment(core_id=19, sibling_core_id=39),),
            "gpu_numa_nodes": (0, 0, 0, 0, 1, 1, 1, 1),
            "core_numa_nodes": dict.fromkeys(range(40), 0),
        },
        {
            "model_cores": tuple(
                CoreAssignment(core_id=index, sibling_core_id=index + 20) for index in range(8)
            ),
            "search_cores": (CoreAssignment(core_id=1, sibling_core_id=21),),
            "proof_cores": (CoreAssignment(core_id=18, sibling_core_id=38),),
            "monitor_cores": (CoreAssignment(core_id=19, sibling_core_id=39),),
            "gpu_numa_nodes": (0, 0, 0, 0, 1, 1, 1, 1),
            "core_numa_nodes": dict.fromkeys(range(40), 0),
        },
    ],
)
def test_invalid_resource_maps_reject_before_manifest_or_launch(
    resource_map: ResourceMapPayload,
) -> None:
    # Given
    with pytest.raises(ValidationError):
        _ = ResourceMap.model_validate(resource_map)


def test_single_gpu_resource_map_is_accepted() -> None:
    # Given a single-GPU workstation (e.g. one RTX 6000 PRO) resource map
    resource_map = {
        "model_cores": (CoreAssignment(core_id=0, sibling_core_id=20),),
        "search_cores": tuple(
            CoreAssignment(core_id=index + 1, sibling_core_id=index + 21) for index in range(10)
        ),
        "proof_cores": (CoreAssignment(core_id=18, sibling_core_id=38),),
        "monitor_cores": (CoreAssignment(core_id=19, sibling_core_id=39),),
        "gpu_numa_nodes": (0,),
        "core_numa_nodes": dict.fromkeys(range(40), 0),
    }

    # When
    validated = ResourceMap.model_validate(resource_map)

    # Then
    assert len(validated.model_cores) == 1
    assert len(validated.gpu_numa_nodes) == 1


def test_treatment_failure_freezes_curve_and_keeps_pair_but_exogenous_invalidates_pair(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    treatment_cell = "wave-00-rep-00-selective-revision"
    exogenous_cell = "wave-01-rep-02-final-only"
    runner = FakeCellRunner(
        treatment_failures={treatment_cell: "controller-crash"},
        exogenous_failures={exogenous_cell: "gpu-xid"},
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
    treatment = next(cell for cell in outcome.cells if cell.cell_id == treatment_cell)
    invalidated = tuple(pair for pair in outcome.pairs if pair.wave_index == 1)
    untouched = tuple(pair for pair in outcome.pairs if pair.wave_index != 1)
    assert treatment.terminal_reason == "treatment-failed"
    assert treatment.coverage_frozen_through_seconds == 10_800
    assert any(pair.repetitions == (0, 1) and pair.status == "complete" for pair in outcome.pairs)
    assert {pair.status for pair in invalidated} == {"exogenous-invalidated"}
    assert all(pair.status == "complete" for pair in untouched)


def test_search_coverage_proof_and_bank_accounting_are_separate(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    post_search = FakePostSearch()

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=post_search,
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    assert outcome.accounting.bank_physical_initial_requests == 320
    assert outcome.accounting.bank_logical_initial_requests_by_arm == {
        "selective-revision": 320,
        "final-only": 320,
        "full-regenerate": 320,
        "structured-one-shot": 320,
    }
    assert outcome.accounting.search_cpu_seconds == 550.0
    assert outcome.accounting.coverage_replay_count == 650
    assert outcome.accounting.proof_replay_count == 0
    assert len(post_search.coverage_calls) == 50
    assert len(post_search.proof_calls) == 50


def test_run_rejects_checked_in_draft_config_before_side_effects(data_root: Path) -> None:
    # Given
    config = _draft_config(data_root)

    # When / Then
    with pytest.raises(CoordinatorError, match="draft"):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(),
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
        )
    assert not config.artifact_root.exists()


def test_load_checked_in_config_is_explicit_draft_and_status_is_read_only(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.json"
    _ = config_path.write_text(
        json.dumps(_config_json("/data/dllm-agent/status-read")), encoding="utf-8"
    )

    # When
    config = load_campaign_config(config_path)
    status = campaign_status(Path("/data/dllm-agent/status-read"))

    # Then
    assert config.qualification_state == "draft"
    assert status.state == "missing"
    assert status.model_or_docker_calls == 0


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("case")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    path = Path("/data/dllm-agent") / f"task5-{suffix}"
    if path.exists():
        raise AssertionError(path)
    return path


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified"}
    )


def _draft_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(_config_json(str(data_root)))


def _config_json(root: str) -> dict[str, JsonValue]:
    return {
        "schema_version": "dllm-openexr-one-day-config/v1",
        "config_id": "task5-test",
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


def _delegates() -> dict[
    Literal["selective-revision", "final-only", "full-regenerate", "structured-one-shot"],
    FakeRevisionDelegate,
]:
    return {arm: FakeRevisionDelegate() for arm in MODEL_ARMS}


def _bank_hashes() -> dict[int, str]:
    return {repetition: f"{repetition:064x}" for repetition in range(10)}


def _all_disjoint(paths: Iterable[Path]) -> bool:
    resolved = tuple(path.resolve() for path in paths)
    return all(
        left != right and not left.is_relative_to(right) and not right.is_relative_to(left)
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    )
