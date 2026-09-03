"""Manifest recovery contracts for interrupted multitarget campaigns."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeGuard

import pytest
from pydantic import TypeAdapter

from dllm_agent.experiments.multitarget.artifacts import write_json_artifact
from dllm_agent.experiments.multitarget.budget import ExperimentSpec, Target
from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
from dllm_agent.experiments.multitarget.campaign_evidence import (
    finished_cell_manifest,
    launched_cell_manifest,
)
from dllm_agent.experiments.multitarget.campaign_execution import (
    CampaignExecutionPaths,
    FinishedCell,
    launch_main_campaigns,
    wait_main_campaigns,
)
from dllm_agent.experiments.multitarget.campaign_recovery import (
    CampaignRecoveryError,
    load_finished_cells_manifest,
    load_launched_cells_manifest,
)
from dllm_agent.experiments.multitarget.container_monitor import (
    CanarySeries,
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.corpus import CandidateInput
from dllm_agent.experiments.multitarget.docker_runtime import container_argv
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
from dllm_agent.experiments.multitarget.schedule import CampaignSchedule, build_campaign_schedule

type JsonObject = dict[str, object]

_MANIFEST_ADAPTER: TypeAdapter[list[JsonObject]] = TypeAdapter(list[JsonObject])
_RESULT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_STR_LIST_ADAPTER: TypeAdapter[list[str]] = TypeAdapter(list[str])

IMAGE_IDS: dict[Target, str] = {
    "libpng": "sha256:" + ("a" * 64),
    "libtiff": "sha256:" + ("b" * 64),
    "libxml2": "sha256:" + ("c" * 64),
}


def test_rehydrates_launch_manifest_in_schedule_order_without_relaunching(
    tmp_path: Path,
) -> None:
    root, schedule, manifest_path, container_ids = _record_launch_manifest(tmp_path)

    recovered = load_launched_cells_manifest(
        experiment_root=root,
        manifest_path=manifest_path,
        schedule=schedule,
    )

    assert tuple(item.cell for item in recovered) == schedule.cells
    assert tuple(item.launch.container_id for item in recovered) == container_ids
    first = recovered[0]
    assert first.request.profile.program == "libpng_read_fuzzer"
    assert first.request.profile.upstream_commit == "a37d4836519517bdce6cb9d956092321eca3e73b"
    assert first.request.corpus_dir == root / "cells/libpng/libfuzzer-only/rep-0/corpus"
    assert first.request.shared_dir == root / "cells/libpng/libfuzzer-only/rep-0/shared"
    assert first.staged_corpus.base_file_count == 1
    assert first.staged_corpus.candidate_file_count == 0
    assert first.staged_corpus.files[0].source_kind == "base"
    assert container_argv(first.request) == _request_argv(_loads(manifest_path)[0])


def test_rehydrates_finished_cell_manifests_in_schedule_order(
    tmp_path: Path,
) -> None:
    root, schedule, launch_manifest_path, expected = _record_finished_manifests(tmp_path)

    recovered = load_finished_cells_manifest(
        experiment_root=root,
        launch_manifest_path=launch_manifest_path,
        schedule=schedule,
    )

    assert tuple(item.cell for item in recovered) == schedule.cells
    assert len(recovered) == 15
    first = recovered[0]
    expected_first = expected[0]
    assert first.work_dir == expected_first.work_dir
    assert first.log_path == expected_first.log_path
    assert first.result == expected_first.result
    assert first.request == expected_first.request
    assert first.staged_corpus == expected_first.staged_corpus
    assert first.launch == expected_first.launch
    assert first.result.findings[0].path == str(
        root / "cells/libpng/libfuzzer-only/rep-0/shared/findings/crash-container-00"
    )


@pytest.mark.parametrize(
    ("case_name", "message"),
    [
        ("tamper_log_sha", "log evidence"),
        ("tamper_finding_sha", "finding evidence"),
        ("tamper_nested_launch", "nested launch"),
        ("drop_result", "finished result set"),
    ],
)
def test_rejects_tampered_finished_cell_manifests(
    tmp_path: Path,
    case_name: str,
    message: str,
) -> None:
    root, schedule, launch_manifest_path, _expected = _record_finished_manifests(tmp_path)
    _mutate_finished_result(root, schedule, case_name)

    with pytest.raises(CampaignRecoveryError, match=message):
        _ = load_finished_cells_manifest(
            experiment_root=root,
            launch_manifest_path=launch_manifest_path,
            schedule=schedule,
        )


@pytest.mark.parametrize(
    ("case_name", "message"),
    [
        ("drop_launch", "cell count"),
        ("duplicate_cell", "duplicate cell"),
        ("tamper_program", "cell manifest"),
        ("duplicate_container", "duplicate container"),
        ("escape_corpus_path", "outside"),
        ("tamper_duration", "argv"),
        ("tamper_command_evidence", "command evidence"),
        ("tamper_corpus_count", "corpus count"),
        ("missing_corpus_file", "corpus file evidence"),
        ("tamper_corpus_file", "corpus file evidence"),
        ("symlink_corpus_file", "corpus file evidence"),
    ],
)
def test_rejects_tampered_launch_manifests(
    tmp_path: Path,
    case_name: str,
    message: str,
) -> None:
    root, schedule, manifest_path, _container_ids = _record_launch_manifest(tmp_path)
    launches = _loads(manifest_path)
    if case_name in ("missing_corpus_file", "tamper_corpus_file", "symlink_corpus_file"):
        _mutate_corpus_file(root, launches, case_name)
    else:
        _mutate(case_name, launches)
    _ = write_json_artifact(
        experiment_root=root,
        relative_path=Path("manifests/tampered-main-launches.json"),
        value=launches,
    )

    with pytest.raises(CampaignRecoveryError, match=message):
        _ = load_launched_cells_manifest(
            experiment_root=root,
            manifest_path=root / "manifests/tampered-main-launches.json",
            schedule=schedule,
        )


def _record_finished_manifests(
    tmp_path: Path,
) -> tuple[Path, CampaignSchedule, Path, tuple[FinishedCell, ...]]:
    root, schedule, launch_manifest_path, _container_ids = _record_launch_manifest(tmp_path)
    launched = load_launched_cells_manifest(
        experiment_root=root,
        manifest_path=launch_manifest_path,
        schedule=schedule,
    )
    finished = wait_main_campaigns(
        launched,
        wait=_finished_result,
        max_workers=1,
    )
    for item in finished:
        _ = write_json_artifact(
            experiment_root=root,
            relative_path=_finished_result_path(
                item.cell.target,
                item.cell.arm,
                item.cell.repetition,
            ),
            value=finished_cell_manifest(root, item),
        )
    return root, schedule, launch_manifest_path, finished


def _record_launch_manifest(
    tmp_path: Path,
) -> tuple[Path, CampaignSchedule, Path, tuple[str, ...]]:
    root = tmp_path / "experiment"
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    launched = launch_main_campaigns(
        schedule=schedule,
        durations=CampaignDurations.smoke(probe_seconds=1, main_seconds=7),
        paths=_paths(tmp_path, root),
        run_label="recovery",
        initial_inputs={target: _population(f"{target}-initial") for target in IMAGE_IDS},
        revision_inputs={
            target: {
                arm: _population(f"{target}-{arm}")
                for arm in ("feedback-no-remask", "selective-revision", "pseudo-feedback")
            }
            for target in IMAGE_IDS
        },
        launch=lambda request: ContainerLaunch(container_id=f"container-{request.cpu_index:02d}"),
    )
    _ = write_json_artifact(
        experiment_root=root,
        relative_path=Path("manifests/main-launches.json"),
        value=tuple(launched_cell_manifest(root, item) for item in launched),
    )
    return (
        root,
        schedule,
        root / "manifests/main-launches.json",
        tuple(item.launch.container_id for item in launched),
    )


def _finished_result(
    container_id: str,
    *,
    shared_dir: Path,
    timeout_seconds: int,
) -> ContainerRunResult:
    _ = timeout_seconds
    logs_text = f"#7: cov: 11 ft: 12 exec/s: 13\nstat::number_of_executed_units: 14\n{container_id}"
    finding_path = shared_dir / "findings" / f"crash-{container_id}"
    finding_path.parent.mkdir(parents=True, exist_ok=True)
    finding_content = f"finding-{container_id}".encode()
    _ = finding_path.write_bytes(finding_content)
    return ContainerRunResult(
        container_id=container_id,
        exit_code=0,
        docker_status="exited",
        oom_killed=False,
        timed_out=False,
        process_error=None,
        logs_text=logs_text,
        log_sha256=hashlib.sha256(logs_text.encode()).hexdigest(),
        metrics=FuzzerMetrics(
            coverage_edges=11,
            value_profile_features=12,
            executions_per_second=13,
            executed_units=14,
        ),
        canaries={"BUG_A": CanarySeries(reached=((0, 1),), triggered=((30, 1),))},
        findings=(
            ArtifactEvidence(
                path=str(finding_path),
                sha256=hashlib.sha256(finding_content).hexdigest(),
                size_bytes=len(finding_content),
            ),
        ),
    )


def _paths(tmp_path: Path, root: Path) -> CampaignExecutionPaths:
    magma_root = tmp_path / "magma"
    for target, program in {
        "libpng": "libpng_read_fuzzer",
        "libtiff": "tiff_read_rgba_fuzzer",
        "libxml2": "libxml2_xml_read_memory_fuzzer",
    }.items():
        source = magma_root / "targets" / target / "corpus" / program
        source.mkdir(parents=True)
        _ = (source / "seed").write_bytes(target.encode())
    return CampaignExecutionPaths(experiment_root=root, magma_root=magma_root)


def _population(prefix: str) -> tuple[CandidateInput, ...]:
    values: list[CandidateInput] = []
    for index in range(16):
        content = f"{prefix}-{index}".encode()
        values.append(
            CandidateInput(
                snapshot_sha256=hashlib.sha256(content).hexdigest(),
                content=content,
            )
        )
    return tuple(values)


def _mutate_finished_result(root: Path, schedule: CampaignSchedule, case_name: str) -> None:
    first = schedule.cells[0]
    path = root / _finished_result_path(first.target, first.arm, first.repetition)
    if case_name == "drop_result":
        path.unlink()
        return
    result = _load_result(path)
    if case_name == "tamper_log_sha":
        _object_field(result, "log")["sha256"] = "0" * 64
    elif case_name == "tamper_finding_sha":
        findings = _list_field(_object_field(result, "result"), "findings")
        _object_field_in_list(findings, 0)["sha256"] = "0" * 64
    elif case_name == "tamper_nested_launch":
        _object_field(_object_field(result, "launch"), "cell")["target"] = "libtiff"
    else:
        message = f"unknown case: {case_name}"
        raise AssertionError(message)
    path.unlink()
    _ = write_json_artifact(
        experiment_root=root,
        relative_path=_finished_result_path(first.target, first.arm, first.repetition),
        value=result,
    )


def _finished_result_path(target: str, arm: str, repetition: int) -> Path:
    return Path("results/cells") / target / arm / f"rep-{repetition}.json"


def _drop_launch(launches: list[JsonObject]) -> None:
    _ = launches.pop()


def _duplicate_cell(launches: list[JsonObject]) -> None:
    launches[1] = launches[0]


def _tamper_program(launches: list[JsonObject]) -> None:
    _object_field(launches[0], "cell")["program"] = "wrong"


def _duplicate_container(launches: list[JsonObject]) -> None:
    launches[0]["container_id"] = launches[1]["container_id"]


def _escape_corpus_path(launches: list[JsonObject]) -> None:
    _object_field(launches[0], "request")["corpus_dir"] = "../escape"


def _tamper_duration(launches: list[JsonObject]) -> None:
    _object_field(launches[0], "request")["duration_seconds"] = 999


def _tamper_command_evidence(launches: list[JsonObject]) -> None:
    _object_field(launches[0], "request")["command_evidence"] = {"argv_sha256": "wrong"}


def _tamper_corpus_count(launches: list[JsonObject]) -> None:
    _object_field(launches[0], "corpus")["base_file_count"] = 99


def _mutate_corpus_file(root: Path, launches: list[JsonObject], case_name: str) -> None:
    path = _first_corpus_file_path(root, launches)
    if case_name == "missing_corpus_file":
        path.unlink()
        return
    if case_name == "tamper_corpus_file":
        _ = path.write_bytes(b"tampered")
        return
    if case_name == "symlink_corpus_file":
        target = root / "replacement-seed"
        _ = target.write_bytes(b"replacement")
        path.unlink()
        path.symlink_to(target)
        return
    message = f"unknown corpus mutation: {case_name}"
    raise AssertionError(message)


def _first_corpus_file_path(root: Path, launches: list[JsonObject]) -> Path:
    corpus = _object_field(launches[0], "corpus")
    corpus_dir = root / _string_field(corpus, "corpus_dir")
    files = _list_field(corpus, "files")
    first = _object_field_in_list(files, 0)
    return corpus_dir / _string_field(first, "relative_path")


def _mutate(case_name: str, launches: list[JsonObject]) -> None:
    mutators = {
        "drop_launch": _drop_launch,
        "duplicate_cell": _duplicate_cell,
        "tamper_program": _tamper_program,
        "duplicate_container": _duplicate_container,
        "escape_corpus_path": _escape_corpus_path,
        "tamper_duration": _tamper_duration,
        "tamper_command_evidence": _tamper_command_evidence,
        "tamper_corpus_count": _tamper_corpus_count,
    }
    mutators[case_name](launches)


def _request_argv(launch: JsonObject) -> tuple[str, ...]:
    request = _object_field(launch, "request")
    value = request["argv"]
    return tuple(_STR_LIST_ADAPTER.validate_python(value))


def _object_field(value: JsonObject, field: str) -> JsonObject:
    item = value[field]
    assert _is_json_object(item)
    return item


def _object_field_in_list(value: list[object], index: int) -> JsonObject:
    item = value[index]
    assert _is_json_object(item)
    return item


def _list_field(value: JsonObject, field: str) -> list[object]:
    item = value[field]
    assert _is_json_list(item)
    return item


def _string_field(value: JsonObject, field: str) -> str:
    item = value[field]
    assert isinstance(item, str)
    return item


def _loads(path: Path) -> list[JsonObject]:
    return _MANIFEST_ADAPTER.validate_json(path.read_bytes())


def _load_result(path: Path) -> JsonObject:
    return _RESULT_ADAPTER.validate_json(path.read_bytes())


def _is_json_object(value: object) -> TypeGuard[JsonObject]:
    return isinstance(value, dict)


def _is_json_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
