"""Task 5 append-only recovery and CLI contract tests. # noqa: SIZE_OK"""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter

from dllm_agent.experiments.multitarget.ledger import JsonValue, append_event
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcomeEnvelope,
    StartGateEvidence,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.recovery import (
    CampaignEvent,
    RecoveryError,
    load_campaign_events,
    resume_campaign,
)
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dllm_agent.experiments.openexr_one_day.models import ModelArm


def test_interruption_then_resume_preserves_seed_deadline_paths_manifest_and_predecessor(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    interrupted = FakeCellRunner(interrupt_after=7)

    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=interrupted,
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
        )
    before = load_campaign_events(config.artifact_root)
    last_before = before[-1]

    # When
    resumed_runner = FakeCellRunner()
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=resumed_runner,
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    after = load_campaign_events(config.artifact_root)

    # Then
    assert outcome.launched_cell_count == 50
    assert after[: len(before)] == before
    assert after[len(before)].previous_stream_sha256 == last_before.stream_sha256
    for request in resumed_runner.requests:
        prior = next(
            (event for event in before if event.cell_id == request.cell_id),
            None,
        )
        if prior is not None:
            assert request.fuzzer_seed == prior.fuzzer_seed
            assert request.deadline_monotonic_seconds == prior.deadline_monotonic_seconds
            assert request.artifact_root == prior.artifact_root
            assert request.config_sha256 == prior.config_sha256
            assert event_manifest_hash(before, request.cell_id) == prior.manifest_sha256
        else:
            resumed = next(
                event
                for event in after[len(before) :]
                if event.phase == "launch" and event.cell_id == request.cell_id
            )
            assert request.fuzzer_seed == resumed.fuzzer_seed
            assert request.deadline_monotonic_seconds == resumed.deadline_monotonic_seconds
            assert request.artifact_root == resumed.artifact_root


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("truncate", "truncated"),
        ("gap", "sequence gap"),
        ("hash", "hash mismatch"),
        ("path", "path escape"),
        ("config", "config mismatch"),
        ("resource", "resource-map mismatch"),
        ("duplicate-terminal", "duplicate terminal"),
        ("event-after-terminal", "event after terminal"),
        ("deadline", "deadline mutation"),
        ("seed", "seed mutation"),
    ],
)
def test_recovery_rejects_noncanonical_checkpoint_evidence(
    data_root: Path,
    mutation: str,
    match: str,
) -> None:
    # Given
    config = _qualified_config(data_root)
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    _mutate_campaign_log(config.artifact_root, mutation)

    # When / Then
    with pytest.raises(RecoveryError, match=match):
        _ = load_campaign_events(config.artifact_root)


def test_expired_unfinished_cells_freeze_without_extending_deadline(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(interrupt_after=1),
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
        now_monotonic_seconds=200_000.0,
    )

    # Then
    expired = tuple(cell for cell in outcome.cells if cell.terminal_reason == "expired-frozen")
    assert len(expired) in {49, 50}
    assert all(cell.deadline_monotonic_seconds <= 124_000.0 for cell in expired)
    assert all(cell.coverage_frozen_through_seconds == 10_800 for cell in expired)
    expired_model = next(cell for cell in expired if cell.arm != "libfuzzer-only")
    assert expired_model.bank_sha256 != "0" * 64
    assert expired_model.mutable_root.is_relative_to(config.artifact_root / "mutable")
    assert expired_model.trajectory_set_sha256 != "0" * 64
    events = load_campaign_events(config.artifact_root)
    assert len({event.cell_id for event in events if event.phase == "terminal"}) == 50
    persisted = CampaignOutcomeEnvelope.model_validate_json(
        (config.artifact_root / "campaign-outcome.json").read_bytes()
    )
    assert persisted.outcome == outcome
    assert persisted.outcome.accounting.search_cpu_seconds in {0.0, 11.0}


def test_cli_modes_are_noninteractive_and_fail_closed_for_draft_or_bad_resume(
    tmp_path: Path,
) -> None:
    # Given
    config_path = tmp_path / "draft.json"
    root = Path("/data/dllm-agent") / (
        f"task5-cli-{sha256(str(tmp_path).encode()).hexdigest()[:12]}"
    )
    config = _config_json(str(root))
    config["config_sha256"] = config_payload_sha256(config)
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")
    script = Path("tools/run_openexr_one_day.py").resolve()

    # When
    help_result = _run_cli(script, ["--help"])
    qualify_result = _run_cli(
        script,
        [
            "qualify",
            "--config",
            str(config_path),
            "--root",
            str(root / "qualification"),
        ],
    )
    status_result = _run_cli(script, ["status", "--root", str(root / "qualification")])
    run_result = _run_cli(script, ["run", "--config", str(config_path)])
    resume_result = _run_cli(script, ["resume", "--root", str(tmp_path / "missing")])

    # Then
    assert help_result.returncode == 0
    assert {"qualify", "run", "resume", "status"} <= set(help_result.stdout.split())
    assert qualify_result.returncode == 0
    assert '"state":"qualified-draft-only"' in qualify_result.stdout
    assert status_result.returncode == 0
    assert '"model_or_docker_calls":0' in status_result.stdout
    assert run_result.returncode != 0
    assert "draft" in run_result.stderr
    assert resume_result.returncode != 0
    assert "not resumable" in resume_result.stderr


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("case")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    path = Path("/data/dllm-agent") / f"task5-recovery-{suffix}"
    if path.exists():
        raise AssertionError(path)
    return path


def event_manifest_hash(events: tuple[CampaignEvent, ...], cell_id: str) -> str:
    return next(event.manifest_sha256 for event in events if event.cell_id == cell_id)


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified"}
    )


def _config_json(root: str) -> dict[str, JsonValue]:
    return {
        "schema_version": "dllm-openexr-one-day-config/v1",
        "config_id": "task5-recovery-test",
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
        "resource_map": {
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
    return {
        arm: FakeRevisionDelegate()
        for arm in ("selective-revision", "final-only", "full-regenerate", "structured-one-shot")
    }


def _mutate_campaign_log(root: Path, mutation: str) -> None:  # noqa: C901, PLR0911
    path = root / "campaign-events.jsonl"
    content = path.read_text(encoding="utf-8")
    if mutation == "truncate":
        _ = path.write_text(content.rstrip("\n"), encoding="utf-8")
        return
    if mutation == "gap":
        _ = path.write_text(content.replace('"sequence":0', '"sequence":1', 1), encoding="utf-8")
        return
    if mutation == "hash":
        _ = path.write_text(
            content.replace('"record_sha256":"', '"record_sha256":"f', 1), encoding="utf-8"
        )
        return
    if mutation == "path":
        _append_bad_event(path, {"cell_id": "bad", "artifact_root": "/var/empty/escape"})
        return
    if mutation == "config":
        _append_bad_event(path, {"cell_id": "bad", "config_sha256": "f" * 64})
        return
    if mutation == "resource":
        _append_bad_event(path, {"cell_id": "bad", "resource_map_sha256": "f" * 64})
        return
    if mutation == "duplicate-terminal":
        _append_bad_event(
            path, {"cell_id": "wave-00-rep-00-selective-revision", "phase": "terminal"}
        )
        return
    if mutation == "event-after-terminal":
        _append_bad_event(path, {"cell_id": "wave-00-rep-00-selective-revision", "phase": "launch"})
        return
    if mutation == "deadline":
        _append_bad_event(
            path,
            {
                "cell_id": "wave-00-rep-00-final-only",
                "fuzzer_seed": 1_000_002,
                "artifact_root": str(path.parent / "cells/wave-00/rep-00/final-only"),
                "deadline_monotonic_seconds": 999.0,
            },
        )
        return
    if mutation == "seed":
        _append_bad_event(
            path,
            {
                "cell_id": "wave-00-rep-00-final-only",
                "artifact_root": str(path.parent / "cells/wave-00/rep-00/final-only"),
                "fuzzer_seed": 999,
            },
        )
        return
    raise AssertionError(mutation)


def _append_bad_event(path: Path, payload: dict[str, str | float]) -> None:
    root = path.parent
    manifest = TypeAdapter(dict[str, JsonValue]).validate_json(
        (root / "campaign-manifest.json").read_bytes()
    )
    config = manifest["config"]
    if not isinstance(config, dict):
        raise TypeError
    base: dict[str, JsonValue] = {
        "phase": payload.get("phase", "terminal"),
        "cell_id": "wave-00-rep-00-selective-revision",
        "fuzzer_seed": 1_000_001,
        "deadline_monotonic_seconds": 80_800.0,
        "artifact_root": str(root / "cells/wave-00/rep-00/selective-revision"),
        "config_sha256": config["config_sha256"],
        "resource_map_sha256": manifest["resource_map_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "terminal_reason": "completed",
    }
    base.update(payload)
    _ = append_event(
        path,
        timestamp_utc="bad",
        event_type="cell_checkpoint",
        payload=base,
    )


def _run_cli(script: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
    )
